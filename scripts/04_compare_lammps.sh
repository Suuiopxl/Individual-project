#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# 04_compare_lammps.sh — LAMMPS CPU vs CUDA GPU Performance Comparison
#
# Usage:
#   cd ~/Individual-project
#   bash scripts/04_compare_lammps.sh
#   bash scripts/04_compare_lammps.sh 20 30 40 50   # custom atom sizes
#
# What it does:
#   1. Builds the CUDA LJ force benchmark (apps/LAMMPS_gpu/cuda_manual/)
#   2. Runs CPU baseline and GPU kernel across multiple problem sizes
#   3. Validates correctness (force & energy comparison)
#   4. Writes a formatted comparison report to reports/LAMMPS/
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

CUDA_DIR="$PROJECT_ROOT/apps/LAMMPS_gpu/cuda_manual"
REPORT_DIR="$PROJECT_ROOT/reports/LAMMPS/performance_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/comparison_${TIMESTAMP}.txt"

mkdir -p "$REPORT_DIR"

# ======================== Configuration ========================
# Default problem sizes: 32K, 108K, 256K atoms (matching bench/in.lj scaling)
if [ $# -gt 0 ]; then
    SIZES=("$@")
else
    SIZES=(20 30 40)
fi

# ======================== Pre-flight checks ========================
log "LAMMPS LJ Force — CPU vs CUDA GPU Comparison"

if ! command -v nvcc &>/dev/null; then
    echo "[ERROR] nvcc not found. Install CUDA toolkit first."
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found. Is GPU driver installed?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
CPU_NAME=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "Unknown")
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')

info "CPU: $CPU_NAME"
info "GPU: $GPU_NAME ($GPU_MEM)"
info "CUDA: $CUDA_VER"
info "Problem sizes: ${SIZES[*]} (ndim → ndim^3 * 4 atoms)"

# ======================== Build ========================
log "Building CUDA benchmark..."

cd "$CUDA_DIR"
make clean 2>/dev/null || true
make -j$(nproc) 2>&1 | tail -5

if [ ! -f lj_cuda_bench ]; then
    echo "[ERROR] Build failed — lj_cuda_bench not found"
    exit 1
fi
info "Build successful"

# ======================== Report header ========================
cat > "$REPORT_FILE" <<EOF
========================================================
  LAMMPS LJ Force — CPU vs CUDA GPU Performance Comparison
  Date: $(date)
  CPU: $CPU_NAME
  GPU: $GPU_NAME ($GPU_MEM)
  CUDA: $CUDA_VER
  Benchmark: LJ pair force (pair_lj_cut), FCC lattice, density=0.8442
  Cutoff: 2.5 sigma, skin=0.3, steps=100
  Strategy: 1 thread/atom, full neighbor list (no Newton 3rd law)
  Data layout: AoS→SoA conversion for coalesced GPU memory access
========================================================

EOF

# ======================== Run benchmarks ========================
cd "$CUDA_DIR"

for ndim in "${SIZES[@]}"; do
    natoms=$((4 * ndim * ndim * ndim))
    log "Benchmarking: ${ndim}^3 × 4 = ${natoms} atoms"

    ./lj_cuda_bench "$ndim" | tee -a "$REPORT_FILE"
    echo "" | tee -a "$REPORT_FILE"
done

# ======================== Analysis notes ========================
cat >> "$REPORT_FILE" <<'EOF'
========================================================
  ANALYSIS NOTES
========================================================
- "GPU (kernel only)" is pure force computation on GPU
- "GPU (full round-trip)" includes H2D memcpy + kernel + D2H memcpy
- CPU baseline is single-threaded, matching original pair_lj_cut.cpp
  (LAMMPS uses MPI domain decomposition, not OpenMP, for parallelism)
- In production MD, data stays on GPU across timesteps
  → effective speedup approaches kernel-only number
- Full neighbor list (no Newton 3rd law) doubles pair count
  but eliminates atomicAdd race conditions on GPU
- Validation uses max relative error < 1e-4 threshold
========================================================
EOF

# ======================== Done ========================
log "Report saved: $REPORT_FILE"
echo ""
cat "$REPORT_FILE" | grep -E "Speedup|CPU|GPU|Atoms:|PASS|FAIL" | head -20
echo ""
log "Done! Full report: $REPORT_FILE"
