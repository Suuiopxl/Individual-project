#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 04_compare_miniGhost.sh — CPU vs GPU performance comparison
#
# Runs single-process comparison at multiple problem sizes to find
# the GPU crossover point. Uses system MPI for CPU and NVIDIA MPI for GPU.
#
# Usage:
#   bash scripts/04_compare_miniGhost.sh
#   RUNS=5 bash scripts/04_compare_miniGhost.sh
###############################################################################

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS:-3}"
TSTEPS=20
NUM_VARS=5
REPORT="$PROJECT_ROOT/reports/miniGhost/performance_comparison"
mkdir -p "$REPORT"

# Binaries
CPU_BIN="$PROJECT_ROOT/apps/miniGhost_build/miniGhost_normal.x"
GPU_SRC="$PROJECT_ROOT/apps/miniGhost_gpu/ref"
GPU_BIN="$GPU_SRC/miniGhost.x"
NVIDIA_MPIRUN="/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin/mpirun"

# Problem sizes to test (NxNxN grid)
SIZES=(50 100 150 200)

log() { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }

# ======================== Verify binaries ========================
if [ ! -x "$CPU_BIN" ]; then
    CPU_BIN="${CPU_BIN%.x}"
    [ -x "$CPU_BIN" ] || { echo "CPU binary not found. Run 01_profile_app.sh first."; exit 1; }
fi

if [ ! -x "$GPU_BIN" ]; then
    log "Building GPU version first..."
    cd "$GPU_SRC"
    make -f makefile.gpu clean 2>/dev/null || true
    make -f makefile.gpu 2>&1 | tail -5
    GPU_BIN=$(find "$GPU_SRC" -name "miniGhost*.x" -type f | head -1)
    [ -x "$GPU_BIN" ] || { echo "GPU build failed."; exit 1; }
fi

# ======================== Run comparisons ========================
TS=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$REPORT/comparison_${TS}.txt"

export ACC_DEVICE_TYPE=nvidia

{
echo "========================================================"
echo "  miniGhost CPU vs GPU Performance Comparison"
echo "  Date: $(date)"
echo "  CPU binary: $CPU_BIN"
echo "  GPU binary: $GPU_BIN"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Runs per size: $RUNS"
echo "  Time steps: $TSTEPS | Variables: $NUM_VARS"
echo "========================================================"
echo ""
printf "%-8s  %-12s  %-12s  %-10s\n" "Size" "CPU (ms)" "GPU (ms)" "Speedup"
printf "%-8s  %-12s  %-12s  %-10s\n" "--------" "------------" "------------" "----------"

for N in "${SIZES[@]}"; do
    MG_ARGS="--nx $N --ny $N --nz $N --num_tsteps $TSTEPS --num_vars $NUM_VARS"

    # CPU runs
    cpu_sum=0
    for i in $(seq 1 $RUNS); do
        t0=$(date +%s%N)
        mpirun --oversubscribe --allow-run-as-root -np 1 "$CPU_BIN" $MG_ARGS >/dev/null 2>&1
        ms=$(( ($(date +%s%N) - t0) / 1000000 ))
        cpu_sum=$((cpu_sum + ms))
    done
    cpu_avg=$((cpu_sum / RUNS))

    # GPU runs
    gpu_sum=0
    for i in $(seq 1 $RUNS); do
        t0=$(date +%s%N)
        "$NVIDIA_MPIRUN" --oversubscribe -np 1 "$GPU_BIN" $MG_ARGS >/dev/null 2>&1
        ms=$(( ($(date +%s%N) - t0) / 1000000 ))
        gpu_sum=$((gpu_sum + ms))
    done
    gpu_avg=$((gpu_sum / RUNS))

    # Speedup
    speedup=$(awk "BEGIN{printf \"%.2f\", $cpu_avg / ($gpu_avg > 0 ? $gpu_avg : 1)}")

    printf "%-8s  %-12s  %-12s  %-10s\n" "${N}^3" "${cpu_avg}" "${gpu_avg}" "${speedup}x"
done

echo ""
echo "========================================================"
} 2>&1 | tee "$RESULT_FILE"

log "Results saved to: $RESULT_FILE"
