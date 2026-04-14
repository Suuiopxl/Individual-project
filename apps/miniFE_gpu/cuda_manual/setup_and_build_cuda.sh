#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# setup_and_build_cuda.sh — Build and run miniFE CUDA benchmark
#
# Run from: apps/miniFE_gpu/cuda_manual/
#
# Usage:
#   bash setup_and_build_cuda.sh          # default 128^3
#   bash setup_and_build_cuda.sh 200       # custom grid
###############################################################################

GREEN='\033[1;32m'
CYAN='\033[1;36m'
NC='\033[0m'
log() { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }

NX="${1:-128}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Paths relative to cuda_manual/
MINIFE_SRC="../../miniFE/openmp/src"

log "miniFE CUDA Benchmark — Build & Run"
info "Grid: ${NX}^3"
info "Directory: $SCRIPT_DIR"

# Step 1: Check prerequisites
log "Checking prerequisites..."
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found. Install CUDA toolkit."; exit 1; }
command -v mpicxx >/dev/null || { echo "ERROR: mpicxx not found. Install OpenMPI."; exit 1; }

NVCC_VER=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
info "nvcc: $NVCC_VER"
info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Step 2: Ensure miniFE common files exist
log "Ensuring miniFE common files..."
if [ -f "$MINIFE_SRC/get_common_files" ]; then
    (cd "$MINIFE_SRC" && bash get_common_files)
fi
if [ -f "$MINIFE_SRC/generate_info_header" ]; then
    (cd "$MINIFE_SRC" && bash generate_info_header "nvcc" "-O2" "miniFE" "MINIFE" 2>/dev/null || true)
fi

# Step 3: Build
log "Building CUDA benchmark..."
make clean 2>/dev/null || true
make NX=$NX 2>&1 | tail -20

if [ ! -f miniFE_cuda_bench ]; then
    echo "ERROR: Build failed."
    exit 1
fi

# Step 4: Run
log "Running benchmark (grid=${NX}^3)..."
echo ""

OMP_NUM_THREADS=1 mpirun --oversubscribe --allow-run-as-root -np 1 \
    ./miniFE_cuda_bench -nx $NX -ny $NX -nz $NX

log "Done!"