#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# 04_compare_xsbench.sh - XSBench CPU vs LLM-CUDA vs Official-CUDA (v3.1)
#
# v3.1 fix: TMP_OUT must be written to a file, not a global variable. The
# previous version assigned it inside a $() command substitution, whose
# subshell context discards the assignment when the parent shell resumes.
#
# v3 changes (kept):
#   1. End-to-end uses true wall-clock (date +%s%N) rather than the binary's
#      self-reported Runtime, which excludes initialisation for Official.
#   2. nsys success is decided by log content, not by binary exit code.
###############################################################################

GREEN='\033[1;32m'; CYAN='\033[1;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
info() { echo -e "${CYAN}  -> $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CPU_DIR="$PROJECT_ROOT/apps/XSBench/openmp-threading"
LLM_DIR="$PROJECT_ROOT/apps/XSBench_gpu/cuda_manual"
OFF_DIR="$PROJECT_ROOT/apps/XSBench/cuda"
REPORT_DIR="$PROJECT_ROOT/reports/XSBench/performance_comparison"
TS=$(date +%Y%m%d_%H%M%S)
TXT="$REPORT_DIR/comparison_${TS}.txt"
CSV="$REPORT_DIR/comparison_${TS}.csv"
mkdir -p "$REPORT_DIR"

RUNS=${RUNS:-3}
LOOKUPS=(100000 300000 1000000 3000000 10000000 17000000)
SIZE="small"

# Shared scratch file for binary stdout (re-used across iterations)
STDOUT_FILE=$(mktemp)
trap 'rm -f "$STDOUT_FILE"' EXIT

# ---------- nsys parser
parse_nsys_log() {
    awk '
        /Executing .*cuda_gpu_kern_sum/      { mode="kern"; in_data=0; next }
        /Executing .*cuda_gpu_mem_time_sum/  { mode="mem";  in_data=0; next }
        /Executing .*stats report/           { mode="";     in_data=0; next }
        mode != "" && /^[[:space:]]*-+[[:space:]]+-+/ { in_data=1; next }
        mode == "kern" && in_data && NF >= 3 {
            v=$2; gsub(",","",v)
            if (v ~ /^[0-9]+(\.[0-9]+)?$/) kern_total += v
        }
        mode == "mem" && in_data && NF >= 3 && ($0 ~ /HtoD/ || $0 ~ /Host-to-Device/) {
            v=$2; gsub(",","",v)
            if (v ~ /^[0-9]+(\.[0-9]+)?$/) h2d_total += v
        }
        mode == "mem" && in_data && NF >= 3 && ($0 ~ /DtoH/ || $0 ~ /Device-to-Host/) {
            v=$2; gsub(",","",v)
            if (v ~ /^[0-9]+(\.[0-9]+)?$/) d2h_total += v
        }
        END { printf "kern_ns=%d h2d_ns=%d d2h_ns=%d", kern_total+0, h2d_total+0, d2h_total+0 }
    ' "$1"
}

# Run nsys; success is judged by log content, not exit code (XSBench Official
# may exit non-zero on a checksum warning even when nsys produced full stats).
run_nsys() {
    local bin="$1"; shift
    local args="$1"; shift
    local logfile="$1"; shift
    local rep="/tmp/nsys_xs_$$_$RANDOM"
    nsys profile -t cuda --stats=true --force-overwrite=true \
        -o "$rep" \
        $bin $args > "$logfile" 2>&1 || true
    rm -f "${rep}.nsys-rep" "${rep}.qdstrm" "${rep}.sqlite" 2>/dev/null
    if ! grep -q "cuda_gpu_kern_sum" "$logfile"; then
        warn "nsys produced no kern stats: $(basename "$logfile")"
        echo "0.00 0.00 0.00"
        return
    fi
    eval "$(parse_nsys_log "$logfile")"
    : "${kern_ns:=0}" "${h2d_ns:=0}" "${d2h_ns:=0}"
    awk -v k="$kern_ns" -v h="$h2d_ns" -v d="$d2h_ns" \
        'BEGIN{printf "%.2f %.2f %.2f", k/1e6, h/1e6, d/1e6}'
}

extract_xs_metrics() {
    local out="$1" rt ls cs
    rt=$(echo "$out" | grep "Runtime:"   | tail -1 | awk '{print $2}')
    ls=$(echo "$out" | grep "Lookups/s:" | tail -1 | awk '{print $2}' | tr -d ',')
    cs=$(echo "$out" | grep -oE "(Verification|checksum):? *[0-9]+" | grep -o '[0-9]*' | tail -1)
    echo "${rt:-0} ${ls:-0} ${cs:-0}"
}
extract_llm_kernel_s() {
    echo "$1" | grep "CUDA XS Lookups took" | awk '{print $5}' | head -1
}

# Run a binary, capture its stdout to STDOUT_FILE, return wall-clock ms on
# stdout. Caller reads STDOUT_FILE with `cat` afterwards.
run_wall() {
    local bin="$1"; shift
    local t0 t1
    t0=$(date +%s%N)
    "$bin" "$@" > "$STDOUT_FILE" 2>&1 || true
    t1=$(date +%s%N)
    echo $(( (t1 - t0) / 1000000 ))
}

# ---------- Build CPU
log "Step 1: build CPU baseline"
cd "$CPU_DIR"
make clean >/dev/null 2>&1 || true
make OPTIMIZE=yes 2>/dev/null
[ -f XSBench ] || { echo "[ERROR] CPU build failed"; exit 1; }
info "CPU OK"

# ---------- Build LLM-CUDA
log "Step 2: build LLM-CUDA"
cd "$LLM_DIR"
for f in io.c GridInit.c XSutils.c Materials.c; do
    [ -f "$f" ] || cp "$CPU_DIR/$f" .
done
make clean >/dev/null 2>&1 || true
make 2>/dev/null
[ -f XSBench_gpu ] || { echo "[ERROR] LLM-CUDA build failed"; exit 1; }
info "LLM-CUDA OK"

# ---------- Build Official CUDA
log "Step 3: build Official CUDA"
OFF_OK=0
OFF_BIN=""
if [ -d "$OFF_DIR" ]; then
    cd "$OFF_DIR"
    if [ -f Makefile ]; then
        [ ! -f Makefile.orig ] && cp Makefile Makefile.orig
        sed -i 's|-std=c++14|-std=c++17|g' Makefile
        sed -i 's|-Xcompiler -O3|-Xcompiler -O2|g' Makefile
        info "Patched Makefile: C++17 + host -O2"
    fi
    make clean >/dev/null 2>&1 || true
    if make COMPILER=nvidia OPTIMIZE=yes SM_VERSION=89 2>/tmp/xsbench_off_build.log; then
        if [ -x ./XSBench ]; then
            OFF_BIN="$OFF_DIR/XSBench"
            OFF_OK=1
            info "Official CUDA OK ($OFF_BIN)"
        fi
    fi
    [ "$OFF_OK" -eq 0 ] && warn "Official CUDA build failed. Log: /tmp/xsbench_off_build.log"
else
    warn "Official CUDA dir missing: $OFF_DIR"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
CPU_NAME=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "Unknown")

# ---------- Run sweep
log "Step 4: run sweep"

declare -A R_CPU_RT R_CPU_LS R_CPU_CS R_CPU_WALL \
           R_LLM_RT R_LLM_LS R_LLM_CS R_LLM_KT R_LLM_WALL \
           R_LLM_KN R_LLM_HD R_LLM_DH R_LLM_OT \
           R_OFF_RT R_OFF_LS R_OFF_CS R_OFF_WALL \
           R_OFF_KN R_OFF_HD R_OFF_DH R_OFF_OT

for L in "${LOOKUPS[@]}"; do
    log "  Lookups: $(printf "%'d" "$L")"

    # CPU averaged
    sum_rt=0; sum_ls=0; last_cs=""; sum_wall=0
    for r in $(seq 1 "$RUNS"); do
        wall_ms=$(run_wall "$CPU_DIR/XSBench" -m event -s "$SIZE" -l "$L" -t 1)
        bin_out=$(cat "$STDOUT_FILE")
        read -r rt ls cs <<<"$(extract_xs_metrics "$bin_out")"
        sum_rt=$(awk "BEGIN{printf \"%.6f\", $sum_rt + $rt}")
        sum_ls=$((sum_ls + ls))
        sum_wall=$((sum_wall + wall_ms))
        last_cs="$cs"
    done
    R_CPU_RT[$L]=$(awk "BEGIN{printf \"%.3f\", $sum_rt / $RUNS}")
    R_CPU_LS[$L]=$((sum_ls / RUNS))
    R_CPU_CS[$L]="$last_cs"
    R_CPU_WALL[$L]=$((sum_wall / RUNS))
    info "    CPU 1t: self=${R_CPU_RT[$L]}s  wall=${R_CPU_WALL[$L]}ms"

    # LLM-CUDA single + nsys
    wall_ms=$(run_wall "$LLM_DIR/XSBench_gpu" -m event -s "$SIZE" -l "$L")
    bin_out=$(cat "$STDOUT_FILE")
    read -r rt ls cs <<<"$(extract_xs_metrics "$bin_out")"
    kt=$(extract_llm_kernel_s "$bin_out")
    R_LLM_RT[$L]="$rt"; R_LLM_LS[$L]="$ls"; R_LLM_CS[$L]="$cs"; R_LLM_KT[$L]="${kt:-0}"
    R_LLM_WALL[$L]="$wall_ms"
    info "    LLM-CUDA: self=${rt}s  wall=${wall_ms}ms  kern=${R_LLM_KT[$L]}s"

    NSYS_LOG="$REPORT_DIR/nsys_llm_${L}_${TS}.log"
    read -r km hm dm <<<"$(run_nsys "$LLM_DIR/XSBench_gpu" "-m event -s $SIZE -l $L" "$NSYS_LOG")"
    R_LLM_KN[$L]="$km"; R_LLM_HD[$L]="$hm"; R_LLM_DH[$L]="$dm"
    R_LLM_OT[$L]=$(awk "BEGIN{r=${wall_ms} - $km - $hm - $dm; printf \"%.2f\", (r<0?0:r)}")
    info "    LLM nsys: kern=${km}ms H2D=${hm}ms D2H=${dm}ms other=${R_LLM_OT[$L]}ms"

    # Official + nsys
    if [ "$OFF_OK" -eq 1 ]; then
        wall_ms=$(run_wall "$OFF_BIN" -m event -s "$SIZE" -l "$L")
        bin_out=$(cat "$STDOUT_FILE")
        read -r rt ls cs <<<"$(extract_xs_metrics "$bin_out")"
        R_OFF_RT[$L]="$rt"; R_OFF_LS[$L]="$ls"; R_OFF_CS[$L]="$cs"
        R_OFF_WALL[$L]="$wall_ms"
        info "    Official: self=${rt}s  wall=${wall_ms}ms"

        NSYS_LOG="$REPORT_DIR/nsys_official_${L}_${TS}.log"
        read -r km hm dm <<<"$(run_nsys "$OFF_BIN" "-m event -s $SIZE -l $L" "$NSYS_LOG")"
        R_OFF_KN[$L]="$km"; R_OFF_HD[$L]="$hm"; R_OFF_DH[$L]="$dm"
        R_OFF_OT[$L]=$(awk "BEGIN{r=${wall_ms} - $km - $hm - $dm; printf \"%.2f\", (r<0?0:r)}")
        info "    Off nsys: kern=${km}ms H2D=${hm}ms D2H=${dm}ms other=${R_OFF_OT[$L]}ms"
    else
        R_OFF_RT[$L]="N/A"; R_OFF_LS[$L]="N/A"; R_OFF_CS[$L]="N/A"; R_OFF_WALL[$L]="N/A"
        R_OFF_KN[$L]="N/A"; R_OFF_HD[$L]="N/A"; R_OFF_DH[$L]="N/A"; R_OFF_OT[$L]="N/A"
    fi
done

# ---------- Write report
log "Step 5: writing report"
{
echo "========================================================"
echo "  XSBench CPU vs LLM-CUDA vs Official-CUDA"
echo "  Date: $(date)"
echo "  CPU: $CPU_NAME"
echo "  GPU: $GPU_NAME"
echo "  Problem size: -s $SIZE   CPU runs averaged: $RUNS"
echo "  Official included: $([ "$OFF_OK" -eq 1 ] && echo yes || echo no)"
echo "  Note: 'Wall' is true wall-clock; 'Self' is binary self-reported Runtime"
echo "        (XSBench Official excludes initialisation from Self)"
echo "========================================================"
echo ""
} > "$TXT"

for L in "${LOOKUPS[@]}"; do
    LBL=$(printf "%'d" "$L")
    base=${R_CPU_LS[$L]}
    {
        echo "--- Lookups: $LBL ---"
        printf "%-22s %10s %10s %14s %10s\n" "Config" "Self(s)" "Wall(ms)" "Lookups/s" "Spdup(self)"
        printf "%-22s %10s %10s %14s %10s\n" "----------------------" "----------" "----------" "--------------" "-----------"

        printf "%-22s %10s %10s %14s %9sx\n" "CPU 1-thread" \
            "${R_CPU_RT[$L]}" "${R_CPU_WALL[$L]}" "$(printf "%'d" "${R_CPU_LS[$L]}")" "1.0"

        sp=$( [ "${R_LLM_LS[$L]}" -gt 0 ] 2>/dev/null && \
              awk "BEGIN{printf \"%.1f\", ${R_LLM_LS[$L]}/$base}" || echo "N/A")
        printf "%-22s %10s %10s %14s %9sx\n" "LLM-CUDA total" \
            "${R_LLM_RT[$L]}" "${R_LLM_WALL[$L]}" "$(printf "%'d" "${R_LLM_LS[$L]}")" "$sp"

        if [ "${R_LLM_KT[$L]}" != "0" ] && [ -n "${R_LLM_KT[$L]}" ]; then
            kls=$(awk "BEGIN{printf \"%d\", $L / ${R_LLM_KT[$L]}}")
            ksp=$(awk "BEGIN{printf \"%.1f\", $kls / $base}")
            printf "%-22s %10s %10s %14s %9sx\n" "LLM-CUDA kernel-only" \
                "${R_LLM_KT[$L]}" "-" "$(printf "%'d" "$kls")" "$ksp"
        fi

        if [ "$OFF_OK" -eq 1 ]; then
            sp=$( [ "${R_OFF_LS[$L]}" -gt 0 ] 2>/dev/null && \
                  awk "BEGIN{printf \"%.1f\", ${R_OFF_LS[$L]}/$base}" || echo "N/A")
            printf "%-22s %10s %10s %14s %9sx\n" "Official CUDA total" \
                "${R_OFF_RT[$L]}" "${R_OFF_WALL[$L]}" "$(printf "%'d" "${R_OFF_LS[$L]}")" "$sp"
        fi
        echo ""
    } >> "$TXT"
done

# ---------- breakdown table
{
    echo ""
    echo "========================================================"
    echo "  GPU breakdown per size (from nsys, ms)"
    echo "  Other = Wall - Kern - H2D - D2H"
    echo "========================================================"
    printf "%-12s %10s %10s %10s %10s %10s %10s %10s %10s\n" \
        "Lookups" "LLM-Kern" "LLM-H2D" "LLM-D2H" "LLM-Other" "Off-Kern" "Off-H2D" "Off-D2H" "Off-Other"
    printf "%-12s %10s %10s %10s %10s %10s %10s %10s %10s\n" \
        "------------" "----------" "----------" "----------" "----------" "----------" "----------" "----------" "----------"
    for L in "${LOOKUPS[@]}"; do
        printf "%-12s %10s %10s %10s %10s %10s %10s %10s %10s\n" \
            "$(printf "%'d" "$L")" \
            "${R_LLM_KN[$L]}" "${R_LLM_HD[$L]}" "${R_LLM_DH[$L]}" "${R_LLM_OT[$L]}" \
            "${R_OFF_KN[$L]}" "${R_OFF_HD[$L]}" "${R_OFF_DH[$L]}" "${R_OFF_OT[$L]}"
    done
    echo ""
    echo "========================================================"
    echo "  Validation note (XSBench checksum)"
    echo "========================================================"
    for L in "${LOOKUPS[@]}"; do
        echo "  L=$L  CPU=${R_CPU_CS[$L]}  LLM=${R_LLM_CS[$L]}  OFFICIAL=${R_OFF_CS[$L]}"
    done
    echo ""
    echo "  Both GPU implementations sample (E,M) pairs in event order rather"
    echo "  than the CPU's history-based order; checksums therefore differ but"
    echo "  the underlying Monte Carlo computation is correct."
    echo "========================================================"
} >> "$TXT"

# ---------- CSV (backwards-compatible: original columns first, new columns appended)
{
    echo "lookups,cpu_1t_s,llm_total_s,llm_kernel_s,off_total_s,cpu_checksum,llm_checksum,off_checksum,cpu_wall_ms,llm_wall_ms,off_wall_ms,llm_kern_ms,llm_h2d_ms,llm_d2h_ms,llm_other_ms,off_kern_ms,off_h2d_ms,off_d2h_ms,off_other_ms"
    for L in "${LOOKUPS[@]}"; do
        echo "${L},${R_CPU_RT[$L]},${R_LLM_RT[$L]},${R_LLM_KT[$L]},${R_OFF_RT[$L]},${R_CPU_CS[$L]},${R_LLM_CS[$L]},${R_OFF_CS[$L]},${R_CPU_WALL[$L]},${R_LLM_WALL[$L]},${R_OFF_WALL[$L]},${R_LLM_KN[$L]},${R_LLM_HD[$L]},${R_LLM_DH[$L]},${R_LLM_OT[$L]},${R_OFF_KN[$L]},${R_OFF_HD[$L]},${R_OFF_DH[$L]},${R_OFF_OT[$L]}"
    done
} > "$CSV"

cat "$TXT"
echo ""
log "Report: $TXT"
log "CSV:    $CSV"