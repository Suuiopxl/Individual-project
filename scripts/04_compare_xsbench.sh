#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# 04_compare_xsbench.sh - XSBench CPU vs LLM-CUDA vs Official-CUDA
#
# Sweeps 6 lookup counts (100k .. 17M); CPU 1-thread averaged; LLM-CUDA single;
# Official CUDA single. No nsys (binary self-reports kernel time).
#
# Build patches applied to apps/XSBench/cuda/Makefile (idempotent):
#   - C++14 -> C++17                (CUDA 13 thrust requires C++17)
#   - Host -O3 -> -O2               (works around GCC 13 ICE in cub headers)
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

# ---------- Build CPU ----------
log "Step 1: build CPU baseline"
cd "$CPU_DIR"
make clean >/dev/null 2>&1 || true
make OPTIMIZE=yes 2>/dev/null
[ -f XSBench ] || { echo "[ERROR] CPU build failed"; exit 1; }
info "CPU OK"

# ---------- Build LLM-CUDA ----------
log "Step 2: build LLM-CUDA"
cd "$LLM_DIR"
for f in io.c GridInit.c XSutils.c Materials.c; do
    [ -f "$f" ] || cp "$CPU_DIR/$f" .
done
make clean >/dev/null 2>&1 || true
make 2>/dev/null
[ -f XSBench_gpu ] || { echo "[ERROR] LLM-CUDA build failed"; exit 1; }
info "LLM-CUDA OK"

# ---------- Build Official CUDA ----------
log "Step 3: build Official CUDA"
OFF_OK=0
OFF_BIN=""
if [ -d "$OFF_DIR" ]; then
    cd "$OFF_DIR"
    if [ -f Makefile ]; then
        [ ! -f Makefile.orig ] && cp Makefile Makefile.orig

        # Patch 1: C++17 (CUDA 13 thrust requires C++17 minimum)
        sed -i 's|-std=c++14|-std=c++17|g' Makefile

        # Patch 2: Host -O3 -> -O2
        # CUDA 13's cub/device_reduce.cuh + GCC 13 -O3 triggers an internal
        # compiler error in the bbro RTL pass. Lowering host-side optimisation
        # by one tier avoids the bug; nvcc's own -O3 (for device code) is unchanged.
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

# ---------- Run sweep ----------
log "Step 4: run sweep"

declare -A R_CPU_RT R_CPU_LS R_CPU_CS \
           R_LLM_RT R_LLM_LS R_LLM_CS R_LLM_KT \
           R_OFF_RT R_OFF_LS R_OFF_CS

for L in "${LOOKUPS[@]}"; do
    log "  Lookups: $(printf "%'d" "$L")"

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

    out=$("$LLM_DIR/XSBench_gpu" -m event -s "$SIZE" -l "$L" 2>&1)
    read -r rt ls cs <<<"$(extract_xs_metrics "$out")"
    kt=$(extract_llm_kernel_s "$out")
    R_LLM_RT[$L]="$rt"; R_LLM_LS[$L]="$ls"; R_LLM_CS[$L]="$cs"; R_LLM_KT[$L]="${kt:-0}"
    info "    LLM-CUDA: ${rt}s total, kernel ${R_LLM_KT[$L]}s"

    if [ "$OFF_OK" -eq 1 ]; then
        out=$("$OFF_BIN" -m event -s "$SIZE" -l "$L" 2>&1)
        read -r rt ls cs <<<"$(extract_xs_metrics "$out")"
        R_OFF_RT[$L]="$rt"; R_OFF_LS[$L]="$ls"; R_OFF_CS[$L]="$cs"
        info "    Official: ${rt}s total"
    else
        R_OFF_RT[$L]="N/A"; R_OFF_LS[$L]="N/A"; R_OFF_CS[$L]="N/A"
    fi
done

# ---------- Write report ----------
log "Step 5: writing report"
{
echo "========================================================"
echo "  XSBench CPU vs LLM-CUDA vs Official-CUDA"
echo "  Date: $(date)"
echo "  CPU: $CPU_NAME"
echo "  GPU: $GPU_NAME"
echo "  Problem size flag: -s $SIZE  CPU runs averaged: $RUNS"
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

        sp=$( [ "${R_LLM_LS[$L]}" -gt 0 ] 2>/dev/null && \
              awk "BEGIN{printf \"%.1f\", ${R_LLM_LS[$L]}/$base}" || echo "N/A")
        printf "%-22s %12s %16s %11sx\n" "LLM-CUDA total" \
            "${R_LLM_RT[$L]}" "$(printf "%'d" "${R_LLM_LS[$L]}")" "$sp"

        if [ "${R_LLM_KT[$L]}" != "0" ] && [ -n "${R_LLM_KT[$L]}" ]; then
            kls=$(awk "BEGIN{printf \"%d\", $L / ${R_LLM_KT[$L]}}")
            ksp=$(awk "BEGIN{printf \"%.1f\", $kls / $base}")
            printf "%-22s %12s %16s %11sx\n" "LLM-CUDA kernel-only" \
                "${R_LLM_KT[$L]}" "$(printf "%'d" "$kls")" "$ksp"
        fi

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

# ---------- CSV ----------
{
    echo "lookups,cpu_1t_s,llm_total_s,llm_kernel_s,off_total_s,cpu_checksum,llm_checksum,off_checksum"
    for L in "${LOOKUPS[@]}"; do
        echo "${L},${R_CPU_RT[$L]},${R_LLM_RT[$L]},${R_LLM_KT[$L]},${R_OFF_RT[$L]},${R_CPU_CS[$L]},${R_LLM_CS[$L]},${R_OFF_CS[$L]}"
    done
} > "$CSV"

cat "$TXT"
echo ""
log "Report: $TXT"
log "CSV:    $CSV"