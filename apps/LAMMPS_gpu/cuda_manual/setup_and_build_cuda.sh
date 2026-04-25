#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# setup_and_build_cuda.sh — Build, test, and save LAMMPS LJ CUDA benchmark
#
# Results are automatically saved to:
#   reports/LAMMPS/performance_comparison/comparison_<timestamp>.txt
#
# Usage:
#   bash setup_and_build_cuda.sh            # build + test (32000 atoms)
#   bash setup_and_build_cuda.sh 30         # build + test (108000 atoms)
#   bash setup_and_build_cuda.sh 20 30 40   # multi-size benchmark
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Navigate to project root to find reports/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPORT_DIR="$PROJECT_ROOT/reports/LAMMPS/performance_comparison"
mkdir -p "$REPORT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/comparison_${TIMESTAMP}.txt"

log() { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
err() { echo -e "\033[1;31m[ERROR] $*\033[0m" >&2; }

# ---- Check CUDA toolkit ----
log "Checking CUDA environment..."
if ! command -v nvcc &>/dev/null; then
    err "nvcc not found. Please install CUDA toolkit."
    exit 1
fi
nvcc --version | grep "release"

if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. Is GPU driver installed?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
CPU_NAME=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "Unknown")

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# ---- Build ----
log "Building LAMMPS LJ CUDA benchmark..."
make clean
make -j$(nproc)

if [ ! -f lj_cuda_bench ]; then
    err "Build failed — lj_cuda_bench not found"
    exit 1
fi

# ---- Write report header ----
cat > "$REPORT_FILE" <<EOF
========================================================
  LAMMPS LJ Force — CPU vs CUDA GPU Performance Comparison
  Date: $(date)
  CPU: $CPU_NAME
  GPU: $GPU_NAME ($GPU_MEM)
  CUDA: $(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
  Benchmark: LJ pair force (pair_lj_cut), FCC lattice, density=0.8442
  Cutoff: 2.5 sigma, skin=0.3, steps=100
  Strategy: 1 thread/atom, full neighbor list (no Newton 3rd law)
  Data layout: AoS→SoA conversion for coalesced GPU memory access
========================================================

EOF

# ---- Run ----
if [ $# -eq 0 ]; then
    SIZES=(20)
else
    SIZES=("$@")
fi

for ndim in "${SIZES[@]}"; do
    natoms=$((4 * ndim * ndim * ndim))
    log "Running benchmark: ${ndim}^3 * 4 = ${natoms} atoms..."

    # Run and capture output (display to terminal AND save)
    ./lj_cuda_bench "$ndim" | tee -a "$REPORT_FILE"
    echo "" | tee -a "$REPORT_FILE"
done

# ---- Append analysis notes ----
cat >> "$REPORT_FILE" <<'EOF'
========================================================
  ANALYSIS NOTES
========================================================
- "GPU (kernel only)" is pure force computation on GPU
- "GPU (full round-trip)" includes cudaMalloc + H2D + kernel + D2H
- CPU baseline is single-threaded (no OpenMP)
- In production MD, data stays on GPU across timesteps
  → effective speedup approaches kernel-only number
- Full neighbor list (no Newton 3rd law) doubles pair count
  but eliminates atomicAdd race conditions on GPU
- Validation uses max relative error < 1e-4 threshold
========================================================
EOF

log "Report saved to: $REPORT_FILE"
log "Done!"
