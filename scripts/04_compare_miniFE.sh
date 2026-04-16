#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# 04_compare_miniFE.sh — miniFE CPU vs CUDA GPU Performance Comparison
#
# Usage:
#   cd ~/Individual-project
#   bash scripts/04_compare_miniFE.sh            # default 128^3
#   bash scripts/04_compare_miniFE.sh 200         # custom grid size
#
# What it does:
#   1. Builds the CUDA benchmark in apps/miniFE_gpu/cuda_manual/
#   2. Runs CPU OpenMP CG and CUDA CG on the same problem
#   3. Compares performance (MATVEC, DOT, WAXPBY, TOTAL) and correctness
#   4. Saves report to reports/miniFE/performance_comparison/
###############################################################################

GREEN='\033[1;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
NC='\033[0m'
log()  { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }

# ======================== Paths ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_DIR="$PROJECT_ROOT/apps/miniFE_gpu/cuda_manual"
REPORT_DIR="$PROJECT_ROOT/reports/miniFE/performance_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/comparison_${TIMESTAMP}.txt"

NX="${1:-128}"

mkdir -p "$REPORT_DIR"

log "=========================================="
log " miniFE CPU vs CUDA Comparison"
log " Grid: ${NX}^3"
log "=========================================="

# ======================== Prerequisites ========================
log "Checking prerequisites..."

command -v nvcc >/dev/null || { echo "ERROR: nvcc not found."; exit 1; }
command -v mpicxx >/dev/null || { echo "ERROR: mpicxx not found."; exit 1; }

if [ ! -d "$CUDA_DIR" ]; then
    echo "ERROR: CUDA source not found at $CUDA_DIR"
    echo "Please place the CUDA files in apps/miniFE_gpu/cuda_manual/"
    exit 1
fi

info "nvcc: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# ======================== Ensure miniFE common files ========================
MINIFE_SRC="$PROJECT_ROOT/apps/miniFE/openmp/src"
if [ -f "$MINIFE_SRC/get_common_files" ]; then
    (cd "$MINIFE_SRC" && bash get_common_files)
fi
if [ -f "$MINIFE_SRC/generate_info_header" ]; then
    (cd "$MINIFE_SRC" && bash generate_info_header "nvcc" "-O2" "miniFE" "MINIFE" 2>/dev/null || true)
fi

# ======================== Build ========================
log "Building CUDA benchmark..."

cd "$CUDA_DIR"
make clean 2>/dev/null || true
make NX=$NX 2>&1 | tail -5

if [ ! -f "$CUDA_DIR/miniFE_cuda_bench" ]; then
    echo "ERROR: Build failed. Check output above."
    exit 1
fi
info "Build successful"

# ======================== Run ========================
log "Running CPU vs CUDA comparison (grid=${NX}^3)..."

cd "$CUDA_DIR"
OMP_NUM_THREADS=$(nproc) \
mpirun --oversubscribe --allow-run-as-root -np 1 \
    ./miniFE_cuda_bench -nx $NX -ny $NX -nz $NX \
    2>&1 | tee "$REPORT_FILE"

# ======================== Summary ========================
log "=========================================="
log " Comparison Complete!"
log " Report: $REPORT_FILE"
log "=========================================="

# Extract key metrics for quick display
echo ""
echo "--- Quick Summary ---"
grep -E "(MATVEC|DOT|WAXPBY|TOTAL CG)" "$REPORT_FILE" | grep -v "^--"
grep "Max |x_cpu" "$REPORT_FILE"
echo ""
info "Full report: $REPORT_FILE"