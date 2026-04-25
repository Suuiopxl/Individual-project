#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# 04_compare_lammps.sh - LAMMPS LJ pair force CPU vs CUDA
#
# - Sweeps 7 problem sizes (ndim 20..50, i.e. 32k..500k atoms)
# - The cuda_manual binary already prints per-step CPU time, GPU kernel time,
#   GPU full round-trip (H2D/Kern/D2H), and validation (force/PE error).
# - We append a CSV alongside the human-readable text report so pgfplots can
#   read it directly.
###############################################################################

GREEN='\033[1;32m'; CYAN='\033[1;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
info() { echo -e "${CYAN}  -> $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_DIR="$PROJECT_ROOT/apps/LAMMPS_gpu/cuda_manual"
REPORT_DIR="$PROJECT_ROOT/reports/LAMMPS/performance_comparison"
TS=$(date +%Y%m%d_%H%M%S)
TXT="$REPORT_DIR/comparison_${TS}.txt"
CSV="$REPORT_DIR/comparison_${TS}.csv"
mkdir -p "$REPORT_DIR"

# Default sweep: ndim from 20 to 50 in steps of 5 (atoms = 4 * ndim^3)
if [ $# -gt 0 ]; then
    SIZES=("$@")
else
    SIZES=(20 25 30 35 40 45 50)
fi

# ---------- Pre-flight
command -v nvcc >/dev/null      || { echo "[ERROR] nvcc not found"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "[ERROR] nvidia-smi not found"; exit 1; }

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
CPU_NAME=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "Unknown")
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')

log "LAMMPS LJ force - CPU vs CUDA GPU comparison"
info "CPU: $CPU_NAME"
info "GPU: $GPU_NAME ($GPU_MEM)"
info "CUDA: $CUDA_VER"
info "Sizes (ndim): ${SIZES[*]}"

# ---------- Build
log "Building cuda_manual benchmark"
cd "$CUDA_DIR"
make clean 2>/dev/null || true
make -j$(nproc) 2>&1 | tail -5
[ -x lj_cuda_bench ] || { echo "[ERROR] Build failed"; exit 1; }
info "Build OK"

# ---------- Run sweep ----------
{
cat <<EOF
========================================================
  LAMMPS LJ force - CPU vs CUDA GPU
  Date: $(date)
  CPU: $CPU_NAME
  GPU: $GPU_NAME ($GPU_MEM)
  CUDA: $CUDA_VER
  FCC lattice density=0.8442  cutoff=2.5  skin=0.3  steps=100  warmup=5
  Strategy: 1 thread/atom, full neighbor list (no Newton 3rd law)
========================================================

EOF
} > "$TXT"

declare -A R_NA R_CPU R_GKER R_GFULL R_H2D R_DTOH R_FERR R_PEERR R_VAL

cd "$CUDA_DIR"
for ndim in "${SIZES[@]}"; do
    natoms=$((4 * ndim * ndim * ndim))
    log "ndim=${ndim}  atoms=${natoms}"

    out=$(./lj_cuda_bench "$ndim" 2>&1)
    echo "$out" >> "$TXT"
    echo "" >> "$TXT"

    R_NA[$ndim]=$natoms
    # CPU per-step ms
    R_CPU[$ndim]=$(echo "$out" | grep -oE "CPU time:[[:space:]]+[0-9.]+ ms/step" \
                    | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
    # GPU kernel median ms
    R_GKER[$ndim]=$(echo "$out" | grep -oE "GPU kernel:[[:space:]]+[0-9.]+ ms \(median\)" \
                     | grep -oE "[0-9.]+ ms" | head -1 | awk '{print $1}')
    # GPU full round-trip ms (= H2D + Kern + D2H sum)
    R_GFULL[$ndim]=$(echo "$out" | grep -oE "= [0-9.]+ ms" | tail -1 | awk '{print $2}')
    # H2D and D2H components
    R_H2D[$ndim]=$(echo "$out" | grep -oE "H2D=[0-9.]+" | tail -1 | cut -d= -f2)
    R_DTOH[$ndim]=$(echo "$out" | grep -oE "D2H=[0-9.]+" | tail -1 | cut -d= -f2)
    # Validation
    R_FERR[$ndim]=$(echo "$out" | grep "Force max rel error:" | tail -1 | awk '{print $NF}')
    R_PEERR[$ndim]=$(echo "$out" | grep "PE error" | tail -1 | grep -oE "/ [0-9.eE+-]+" | tail -1 | awk '{print $NF}')
    R_VAL[$ndim]=$(echo "$out" | grep "Validation:" | tail -1 | awk '{print $NF}')
done

# ---------- Summary table ----------
{
echo "========================================================"
echo "  Summary across sizes (ms/step unless noted)"
echo "========================================================"
printf "%-6s %-10s %-10s %-12s %-12s %-10s %-10s %-8s\n" \
    "ndim" "atoms" "CPU(ms)" "GPU-kern(ms)" "GPU-full(ms)" "H2D(ms)" "D2H(ms)" "Valid"
printf "%-6s %-10s %-10s %-12s %-12s %-10s %-10s %-8s\n" \
    "------" "----------" "----------" "------------" "------------" "----------" "----------" "--------"
for ndim in "${SIZES[@]}"; do
    printf "%-6s %-10s %-10s %-12s %-12s %-10s %-10s %-8s\n" \
        "$ndim" "${R_NA[$ndim]}" "${R_CPU[$ndim]:-N/A}" "${R_GKER[$ndim]:-N/A}" \
        "${R_GFULL[$ndim]:-N/A}" "${R_H2D[$ndim]:-N/A}" "${R_DTOH[$ndim]:-N/A}" "${R_VAL[$ndim]:-N/A}"
done
echo ""
echo "Speedups vs CPU 1-thread (ms/step ratio):"
for ndim in "${SIZES[@]}"; do
    if [ -n "${R_CPU[$ndim]}" ] && [ -n "${R_GKER[$ndim]}" ]; then
        ksp=$(awk "BEGIN{printf \"%.1f\", ${R_CPU[$ndim]}/${R_GKER[$ndim]}}")
        fsp=$(awk "BEGIN{printf \"%.1f\", ${R_CPU[$ndim]}/${R_GFULL[$ndim]}}")
        printf "  ndim=%-3s   kernel-only=%5sx   full-round-trip=%5sx\n" "$ndim" "$ksp" "$fsp"
    fi
done
echo "========================================================"
} >> "$TXT"

# ---------- CSV ----------
{
    echo "ndim,natoms,cpu_ms_per_step,gpu_kernel_ms,gpu_full_ms,gpu_h2d_ms,gpu_d2h_ms,force_max_rel_err,pe_rel_err,validation"
    for ndim in "${SIZES[@]}"; do
        echo "${ndim},${R_NA[$ndim]},${R_CPU[$ndim]:-},${R_GKER[$ndim]:-},${R_GFULL[$ndim]:-},${R_H2D[$ndim]:-},${R_DTOH[$ndim]:-},${R_FERR[$ndim]:-},${R_PEERR[$ndim]:-},${R_VAL[$ndim]:-}"
    done
} > "$CSV"

cat "$TXT" | tail -30
echo ""
log "Report: $TXT"
log "CSV:    $CSV"
