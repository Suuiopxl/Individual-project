#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# compare_xsbench.sh — XSBench CPU vs CUDA GPU Performance Comparison
#
# Usage:
#   cd ~/Individual-project
#   bash scripts/compare_xsbench.sh
#
# What it does:
#   1. Builds CPU baseline (openmp-threading) and GPU (cuda_manual) versions
#   2. Runs both across multiple problem sizes and thread counts
#   3. Extracts Lookups/s, runtime, and verification checksums
#   4. Writes a formatted comparison report to reports/XSBench/
###############################################################################

GREEN='\033[1;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
NC='\033[0m'
log()  { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }

# ======================== Paths ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CPU_DIR="$PROJECT_ROOT/apps/XSBench/openmp-threading"
GPU_DIR="$PROJECT_ROOT/apps/XSBench_gpu/cuda_manual"
REPORT_DIR="$PROJECT_ROOT/reports/XSBench/performance_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/comparison_${TIMESTAMP}.txt"

mkdir -p "$REPORT_DIR"

# ======================== Configuration ========================
RUNS=${RUNS:-3}                        # Runs per configuration for averaging
CPU_THREADS=("1" "4")                  # CPU thread counts to test
LOOKUPS=("100000" "1000000" "17000000") # Lookup counts (small, medium, default)
SIZE="small"                           # Problem size (-s flag)

# ======================== Helper: extract metrics from XSBench output ========================
extract_metrics() {
    local output="$1"
    local runtime lookups_s checksum

    # Use tail -1 to get last match (GPU prints RESULTS twice)
    runtime=$(echo "$output" | grep "Runtime:" | tail -1 | awk '{print $2}')
    lookups_s=$(echo "$output" | grep "Lookups/s:" | tail -1 | awk '{print $2}' | tr -d ',')
    # Match both "Verification:" and "Verification checksum:"
    checksum=$(echo "$output" | grep -oE "(Verification|checksum):? *[0-9]+" | grep -o '[0-9]*' | tail -1)

    echo "${runtime:-0} ${lookups_s:-0} ${checksum:-0}"
}

# ======================== Helper: extract kernel time from GPU output ========================
extract_kernel_time() {
    local output="$1"
    local kt
    kt=$(echo "$output" | grep "CUDA XS Lookups took" | awk '{print $5}')
    echo "${kt:-N/A}"
}

# ======================== Step 1: Build ========================
log "Step 1: Building CPU and GPU versions"

info "Building CPU version..."
cd "$CPU_DIR"
make clean >/dev/null 2>&1 || true
make OPTIMIZE=yes 2>/dev/null
if [ ! -f "XSBench" ]; then
    echo "ERROR: CPU build failed"
    exit 1
fi
info "CPU build OK"

info "Building GPU version..."
cd "$GPU_DIR"

# Copy C source files if not present
for f in io.c GridInit.c XSutils.c Materials.c; do
    [ -f "$f" ] || cp "$CPU_DIR/$f" .
done

make clean >/dev/null 2>&1 || true
make 2>/dev/null
if [ ! -f "XSBench_gpu" ]; then
    echo "ERROR: GPU build failed"
    exit 1
fi
info "GPU build OK"

# ======================== Step 2: Detect GPU ========================
log "Step 2: System info"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
CPU_NAME=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "Unknown")
info "CPU: $CPU_NAME"
info "GPU: $GPU_NAME"

# ======================== Step 3: Run benchmarks ========================
log "Step 3: Running benchmarks ($RUNS runs each)"

# Results storage: associative arrays
declare -A RESULTS_RT RESULTS_LS RESULTS_CS RESULTS_KT

for NLOOKUPS in "${LOOKUPS[@]}"; do
    LOOKUP_LABEL=$(printf "%'d" "$NLOOKUPS")
    log "  Lookups: $LOOKUP_LABEL"

    # --- CPU runs ---
    for NT in "${CPU_THREADS[@]}"; do
        KEY="cpu_${NT}t_${NLOOKUPS}"
        info "CPU ${NT}-thread x${RUNS} runs..."

        total_rt=0; total_ls=0; last_cs=""; 
        for r in $(seq 1 "$RUNS"); do
            output=$("$CPU_DIR/XSBench" -m event -s "$SIZE" -l "$NLOOKUPS" -t "$NT" 2>&1)
            read -r rt ls cs <<< "$(extract_metrics "$output")"
            total_rt=$(echo "$total_rt + $rt" | bc)
            total_ls=$((total_ls + ls))
            last_cs="$cs"
        done

        avg_rt=$(echo "scale=3; $total_rt / $RUNS" | bc)
        avg_ls=$((total_ls / RUNS))
        RESULTS_RT[$KEY]="$avg_rt"
        RESULTS_LS[$KEY]="$avg_ls"
        RESULTS_CS[$KEY]="$last_cs"
        RESULTS_KT[$KEY]="N/A"

        info "  avg: ${avg_rt}s, ${avg_ls} lookups/s, checksum: $last_cs"
    done

    # --- GPU runs ---
    KEY="gpu_${NLOOKUPS}"
    info "GPU (CUDA) x${RUNS} runs..."

    total_rt=0; total_ls=0; total_kt=0; last_cs=""
    for r in $(seq 1 "$RUNS"); do
        output=$("$GPU_DIR/XSBench_gpu" -m event -s "$SIZE" -l "$NLOOKUPS" 2>&1)
        read -r rt ls cs <<< "$(extract_metrics "$output")"
        kt=$(extract_kernel_time "$output")
        total_rt=$(echo "$total_rt + $rt" | bc)
        total_ls=$((total_ls + ls))
        total_kt=$(echo "$total_kt + $kt" | bc)
        last_cs="$cs"
    done

    avg_rt=$(echo "scale=3; $total_rt / $RUNS" | bc)
    avg_ls=$((total_ls / RUNS))
    avg_kt=$(echo "scale=3; $total_kt / $RUNS" | bc)
    RESULTS_RT[$KEY]="$avg_rt"
    RESULTS_LS[$KEY]="$avg_ls"
    RESULTS_CS[$KEY]="$last_cs"
    RESULTS_KT[$KEY]="$avg_kt"

    info "  avg: ${avg_rt}s (kernel: ${avg_kt}s), ${avg_ls} lookups/s, checksum: $last_cs"
done

# ======================== Step 4: Generate report ========================
log "Step 4: Writing report to $REPORT_FILE"

cat > "$REPORT_FILE" << HEADER
========================================================
  XSBench CPU vs CUDA GPU Performance Comparison
  Date: $(date)
  CPU: $CPU_NAME
  GPU: $GPU_NAME
  Problem size: $SIZE | Runs per config: $RUNS
========================================================

HEADER

for NLOOKUPS in "${LOOKUPS[@]}"; do
    LOOKUP_LABEL=$(printf "%'d" "$NLOOKUPS")

    echo "--- Lookups: $LOOKUP_LABEL ---" >> "$REPORT_FILE"
    printf "%-20s %12s %16s %12s %12s %10s\n" \
        "Config" "Runtime(s)" "Lookups/s" "Kernel(s)" "Checksum" "Speedup" \
        >> "$REPORT_FILE"
    printf "%-20s %12s %16s %12s %12s %10s\n" \
        "--------------------" "------------" "----------------" "------------" "------------" "----------" \
        >> "$REPORT_FILE"

    # CPU 1-thread as baseline
    BASE_KEY="cpu_1t_${NLOOKUPS}"
    BASE_LS=${RESULTS_LS[$BASE_KEY]}

    for NT in "${CPU_THREADS[@]}"; do
        KEY="cpu_${NT}t_${NLOOKUPS}"
        ls=${RESULTS_LS[$KEY]}
        if [ "$BASE_LS" -gt 0 ] 2>/dev/null; then
            speedup=$(echo "scale=1; $ls / $BASE_LS" | bc)
        else
            speedup="N/A"
        fi
        printf "%-20s %12s %16s %12s %12s %10sx\n" \
            "CPU ${NT}-thread" \
            "${RESULTS_RT[$KEY]}" \
            "$(printf "%'d" "${RESULTS_LS[$KEY]}")" \
            "${RESULTS_KT[$KEY]}" \
            "${RESULTS_CS[$KEY]}" \
            "$speedup" \
            >> "$REPORT_FILE"
    done

    # GPU total
    KEY="gpu_${NLOOKUPS}"
    ls=${RESULTS_LS[$KEY]}
    if [ "$BASE_LS" -gt 0 ] 2>/dev/null; then
        speedup=$(echo "scale=1; $ls / $BASE_LS" | bc)
    else
        speedup="N/A"
    fi

    # GPU kernel-only lookups/s
    kt=${RESULTS_KT[$KEY]}
    if [ "$kt" != "N/A" ] && [ "$(echo "$kt > 0" | bc)" -eq 1 ]; then
        kernel_ls=$(echo "scale=0; $NLOOKUPS / $kt" | bc)
        kernel_speedup=$(echo "scale=1; $kernel_ls / $BASE_LS" | bc)
    else
        kernel_ls="N/A"
        kernel_speedup="N/A"
    fi

    printf "%-20s %12s %16s %12s %12s %10sx\n" \
        "GPU (total)" \
        "${RESULTS_RT[$KEY]}" \
        "$(printf "%'d" "${RESULTS_LS[$KEY]}")" \
        "$kt" \
        "${RESULTS_CS[$KEY]}" \
        "$speedup" \
        >> "$REPORT_FILE"

    if [ "$kernel_ls" != "N/A" ]; then
        printf "%-20s %12s %16s %12s %12s %10sx\n" \
            "GPU (kernel only)" \
            "$kt" \
            "$(printf "%'d" "$kernel_ls")" \
            "$kt" \
            "-" \
            "$kernel_speedup" \
            >> "$REPORT_FILE"
    fi

    echo "" >> "$REPORT_FILE"
done

# Summary section
cat >> "$REPORT_FILE" << 'SUMMARY'
========================================================
  ANALYSIS NOTES
========================================================
- "GPU (total)" includes cudaMalloc + cudaMemcpy + kernel + cudaMemcpy back
- "GPU (kernel only)" is pure computation time on GPU
- Speedup is relative to CPU 1-thread baseline
- Large gap between GPU total and kernel-only indicates data transfer bottleneck
- In production, data stays on GPU across iterations => effective speedup
  approaches the kernel-only number
========================================================
SUMMARY

log "Report saved: $REPORT_FILE"
cat "$REPORT_FILE"

echo ""
log "Done! Results in: $REPORT_DIR/"