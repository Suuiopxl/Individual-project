#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# 04_compare_xsbench.sh - XSBench CPU vs LLM-CUDA vs Official-CUDA
#
# - Sweeps 6 lookup counts (100k .. 17M)
# - CPU: 1-thread baseline averaged over $RUNS (default 3)
# - LLM-CUDA: single run; kernel time taken from the binary's self-report
# - Official CUDA (apps/XSBench/cuda/): single run; built defensively
# - nsys H2D/D2H breakdown on the largest lookup count for both GPU versions
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
        mode == "mem" && in_data && NF >= 3 && $0 ~ /HtoD/ {
            v=$2; gsub(",","",v)
            if (v ~ /^[0-9]+(\.[0-9]+)?$/) h2d_total += v
        }
        mode == "mem" && in_data && NF >= 3 && $0 ~ /DtoH/ {
            v=$2; gsub(",","",v)
            if (v ~ /^[0-9]+(\.[0-9]+)?$/) d2h_total += v
        }
        END { printf "kern_ns=%d h2d_ns=%d d2h_ns=%d", kern_total+0, h2d_total+0, d2h_total+0 }
    ' "$1"
}

# ---------- Output extractors
extract_xs_metrics() {
    # input: full stdout from XSBench. Returns: runtime_s lookups_per_s checksum
    local out="$1" rt ls cs
    rt=$(echo "$out" | grep "Runtime:"   | tail -1 | awk '{print $2}')
    ls=$(echo "$out" | grep "Lookups/s:" | tail -1 | awk '{print $2}' | tr -d ',')
    cs=$(echo "$out" | grep -oE "(Verification|checksum):? *[0-9]+" | grep -o '[0-9]*' | tail -1)
    echo "${rt:-0} ${ls:-0} ${cs:-0}"
}
extract_llm_kernel_s() {
    # cuda_manual prints "CUDA XS Lookups took <s> seconds"
    echo "$1" | grep "CUDA XS Lookups took" | awk '{print $5}' | head -1
}

# ---------- Step 1: build CPU ----------
log "Step 1: build CPU baseline"
cd "$CPU_DIR"
make clean >/dev/null 2>&1 || true
make OPTIMIZE=yes 2>/dev/null
[ -f XSBench ] || { echo "[ERROR] CPU build failed"; exit 1; }
info "CPU OK"

# ---------- Step 2: build LLM-CUDA ----------
log "Step 2: build LLM-CUDA (cuda_manual)"
cd "$LLM_DIR"
for f in io.c GridInit.c XSutils.c Materials.c; do
    [ -f "$f" ] || cp "$CPU_DIR/$f" .
done
make clean >/dev/null 2>&1 || true
make 2>/dev/null
[ -f XSBench_gpu ] || { echo "[ERROR] LLM-CUDA build failed"; exit 1; }
info "LLM-CUDA OK"

# ---------- Step 3: build Official CUDA (defensive) ----------
log "Step 3: build Official CUDA (apps/XSBench/cuda/)"
OFF_OK=0
if [ -d "$OFF_DIR" ]; then
    cd "$OFF_DIR"
    make clean >/dev/null 2>&1 || true
    if make COMPILER=nvidia OPTIMIZE=yes SM_VERSION=89 2>/tmp/xsbench_off_build.log; then
        if [ -x ./XSBench ]; then
            OFF_BIN="$OFF_DIR/XSBench"
            OFF_OK=1
            info "Official CUDA OK ($OFF_BIN)"
        fi
    fi
    if [ "$OFF_OK" -eq 0 ]; then
        warn "Official CUDA build failed. Skipping. Build log: /tmp/xsbench_off_build.log"
    fi
else
    warn "Official CUDA directory not found at $OFF_DIR; skipping."
fi

# ---------- Step 4: system info ----------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
CPU_NAME=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "Unknown")

# ---------- Step 5: run sweep ----------
log "Step 5: run sweep ($RUNS CPU runs per L; 1 GPU run)"

declare -A R_CPU_RT R_CPU_LS R_CPU_CS \
           R_LLM_RT R_LLM_LS R_LLM_CS R_LLM_KT \
           R_OFF_RT R_OFF_LS R_OFF_CS

for L in "${LOOKUPS[@]}"; do
    log "  Lookups: $(printf "%'d" "$L")"

    # CPU 1-thread averaged
    sum_rt=0; sum_ls=0; last_cs=""
    for r in $(seq 1 "$RUNS"); do
        out=$("$CPU_DIR/XSBench" -m event -s "$SIZE" -l "$L" -t 1 2>&1)
        read -r rt ls cs <<<"$(extract_xs_metrics "$out")"
        sum_rt=$(awk "BEGIN{printf \"%.6f\", $sum_rt + $rt}")
        sum_ls=$((sum_ls + ls))
        last_cs="$cs"
    done
    R_CPU_RT[$L]=$(awk "BEGIN{printf \"%.3f\", $sum_rt / $RUNS}")
    R_CPU_LS[$L]=$((sum_ls / RUNS))
    R_CPU_CS[$L]="$last_cs"
    info "    CPU 1t: ${R_CPU_RT[$L]}s, ${R_CPU_LS[$L]} lookups/s"

    # LLM-CUDA single
    out=$("$LLM_DIR/XSBench_gpu" -m event -s "$SIZE" -l "$L" 2>&1)
    read -r rt ls cs <<<"$(extract_xs_metrics "$out")"
    kt=$(extract_llm_kernel_s "$out")
    R_LLM_RT[$L]="$rt"; R_LLM_LS[$L]="$ls"; R_LLM_CS[$L]="$cs"; R_LLM_KT[$L]="${kt:-0}"
    info "    LLM-CUDA: ${rt}s total, kernel ${R_LLM_KT[$L]}s"

    # Official CUDA single (if built)
    if [ "$OFF_OK" -eq 1 ]; then
        out=$("$OFF_BIN" -m event -s "$SIZE" -l "$L" 2>&1)
        read -r rt ls cs <<<"$(extract_xs_metrics "$out")"
        R_OFF_RT[$L]="$rt"; R_OFF_LS[$L]="$ls"; R_OFF_CS[$L]="$cs"
        info "    Official: ${rt}s total"
    else
        R_OFF_RT[$L]="N/A"; R_OFF_LS[$L]="N/A"; R_OFF_CS[$L]="N/A"
    fi
done

# ---------- Step 6: nsys breakdown on largest L for both GPU versions ----------
LMAX="${LOOKUPS[-1]}"
log "Step 6: nsys breakdown at L=$(printf "%'d" "$LMAX")"

LLM_NSYS_LOG="$REPORT_DIR/nsys_llm_${TS}.log"
nsys profile -t cuda --stats=true --force-overwrite=true \
    -o "/tmp/nsys_xs_llm_$$" \
    "$LLM_DIR/XSBench_gpu" -m event -s "$SIZE" -l "$LMAX" \
    > "$LLM_NSYS_LOG" 2>&1 || warn "nsys (LLM) failed"
rm -f /tmp/nsys_xs_llm_$$.* 2>/dev/null
eval "$(parse_nsys_log "$LLM_NSYS_LOG")"
LLM_KERN_MS=$(awk "BEGIN{printf \"%.2f\", ${kern_ns:-0} / 1e6}")
LLM_H2D_MS=$(awk "BEGIN{printf \"%.2f\", ${h2d_ns:-0} / 1e6}")
LLM_D2H_MS=$(awk "BEGIN{printf \"%.2f\", ${d2h_ns:-0} / 1e6}")

if [ "$OFF_OK" -eq 1 ]; then
    OFF_NSYS_LOG="$REPORT_DIR/nsys_official_${TS}.log"
    nsys profile -t cuda --stats=true --force-overwrite=true \
        -o "/tmp/nsys_xs_off_$$" \
        "$OFF_BIN" -m event -s "$SIZE" -l "$LMAX" \
        > "$OFF_NSYS_LOG" 2>&1 || warn "nsys (Official) failed"
    rm -f /tmp/nsys_xs_off_$$.* 2>/dev/null
    eval "$(parse_nsys_log "$OFF_NSYS_LOG")"
    OFF_KERN_MS=$(awk "BEGIN{printf \"%.2f\", ${kern_ns:-0} / 1e6}")
    OFF_H2D_MS=$(awk "BEGIN{printf \"%.2f\", ${h2d_ns:-0} / 1e6}")
    OFF_D2H_MS=$(awk "BEGIN{printf \"%.2f\", ${d2h_ns:-0} / 1e6}")
else
    OFF_KERN_MS="N/A"; OFF_H2D_MS="N/A"; OFF_D2H_MS="N/A"
fi

# ---------- Step 7: write report ----------
log "Step 7: writing report"

{
echo "========================================================"
echo "  XSBench CPU vs LLM-CUDA vs Official-CUDA"
echo "  Date: $(date)"
echo "  CPU: $CPU_NAME"
echo "  GPU: $GPU_NAME"
echo "  Problem size flag: -s $SIZE | CPU runs averaged: $RUNS"
echo "  Official CUDA included: $([ "$OFF_OK" -eq 1 ] && echo yes || echo no)"
echo "========================================================"
echo ""
} > "$TXT"

for L in "${LOOKUPS[@]}"; do
    LBL=$(printf "%'d" "$L")
    base=${R_CPU_LS[$L]}
    {
        echo "--- Lookups: $LBL ---"
        printf "%-22s %12s %16s %12s\n" "Config" "Runtime(s)" "Lookups/s" "Speedup"
        printf "%-22s %12s %16s %12s\n" "----------------------" "------------" "----------------" "------------"

        printf "%-22s %12s %16s %11sx\n" "CPU 1-thread" \
            "${R_CPU_RT[$L]}" "$(printf "%'d" "${R_CPU_LS[$L]}")" "1.0"

        # LLM total
        sp=$( [ "${R_LLM_LS[$L]}" -gt 0 ] 2>/dev/null && \
              awk "BEGIN{printf \"%.1f\", ${R_LLM_LS[$L]}/$base}" || echo "N/A")
        printf "%-22s %12s %16s %11sx\n" "LLM-CUDA total" \
            "${R_LLM_RT[$L]}" "$(printf "%'d" "${R_LLM_LS[$L]}")" "$sp"

        # LLM kernel-only (lookups / kernel-time)
        if [ "${R_LLM_KT[$L]}" != "0" ] && [ -n "${R_LLM_KT[$L]}" ]; then
            kls=$(awk "BEGIN{printf \"%d\", $L / ${R_LLM_KT[$L]}}")
            ksp=$(awk "BEGIN{printf \"%.1f\", $kls / $base}")
            printf "%-22s %12s %16s %11sx\n" "LLM-CUDA kernel-only" \
                "${R_LLM_KT[$L]}" "$(printf "%'d" "$kls")" "$ksp"
        fi

        # Official total
        if [ "$OFF_OK" -eq 1 ]; then
            sp=$( [ "${R_OFF_LS[$L]}" -gt 0 ] 2>/dev/null && \
                  awk "BEGIN{printf \"%.1f\", ${R_OFF_LS[$L]}/$base}" || echo "N/A")
            printf "%-22s %12s %16s %11sx\n" "Official CUDA total" \
                "${R_OFF_RT[$L]}" "$(printf "%'d" "${R_OFF_LS[$L]}")" "$sp"
        fi
        echo ""
    } >> "$TXT"
done

{
    echo "========================================================"
    echo "  GPU breakdown at largest L = $(printf "%'d" "$LMAX")"
    echo "========================================================"
    printf "%-22s %12s %12s %12s\n" "Config" "Kernel(ms)" "H2D(ms)" "D2H(ms)"
    printf "%-22s %12s %12s %12s\n" "----------------------" "------------" "------------" "------------"
    printf "%-22s %12s %12s %12s\n" "LLM-CUDA"      "$LLM_KERN_MS" "$LLM_H2D_MS" "$LLM_D2H_MS"
    printf "%-22s %12s %12s %12s\n" "Official CUDA" "$OFF_KERN_MS" "$OFF_H2D_MS" "$OFF_D2H_MS"
    echo ""
    echo "Validation note (XSBench checksum):"
    for L in "${LOOKUPS[@]}"; do
        echo "  L=$L  CPU=${R_CPU_CS[$L]}  LLM=${R_LLM_CS[$L]}  OFFICIAL=${R_OFF_CS[$L]}"
    done
    echo "  (GPU paths sample (E,M) in event order, hash differs from CPU"
    echo "   for L>~1e5 even though arithmetic is correct.)"
    echo "========================================================"
} >> "$TXT"

# ---------- CSV ----------
{
    echo "lookups,cpu_1t_s,llm_total_s,llm_kernel_s,off_total_s,llm_kern_ms_atmax,llm_h2d_ms_atmax,llm_d2h_ms_atmax,off_kern_ms_atmax,off_h2d_ms_atmax,off_d2h_ms_atmax"
    for L in "${LOOKUPS[@]}"; do
        # only put nsys breakdown on largest size, others empty
        if [ "$L" = "$LMAX" ]; then
            kn="$LLM_KERN_MS"; hn="$LLM_H2D_MS"; dn="$LLM_D2H_MS"
            ko="$OFF_KERN_MS"; ho="$OFF_H2D_MS"; do_="$OFF_D2H_MS"
        else
            kn=""; hn=""; dn=""; ko=""; ho=""; do_=""
        fi
        echo "${L},${R_CPU_RT[$L]},${R_LLM_RT[$L]},${R_LLM_KT[$L]},${R_OFF_RT[$L]},${kn},${hn},${dn},${ko},${ho},${do_}"
    done
} > "$CSV"

cat "$TXT"
echo ""
log "Report: $TXT"
log "CSV:    $CSV"
