#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# 04_compare_miniFE.sh - miniFE CPU vs LLM-CUDA vs Official-CUDA
#
# Sweeps 7 grid sizes (50..200 cubes); nsys breakdown for every size.
#
# Patches the upstream Mantevo miniFE/cuda for CUDA 13 / NVHPC 26.1
# (idempotent and safe to re-run):
#   - sm_35 -> sm_89
#   - MPI include path injected into NVCCFLAGS (nvcc itself needs to find mpi.h)
#   - <nvToolsExt.h>      -> <nvtx3/nvToolsExt.h>   (header relocation in CUDA 13)
#   - cudaThreadSetCacheConfig -> cudaDeviceSetCacheConfig (CUDA 13 removal)
#   - drop -l nvToolsExt from LIBS (nvtx3 is header-only)
#   - drop -lnsl, -lutil (legacy libraries no longer in modern glibc)
###############################################################################

GREEN='\033[1;32m'; CYAN='\033[1;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
info() { echo -e "${CYAN}  -> $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLM_DIR="$PROJECT_ROOT/apps/miniFE_gpu/cuda_manual"
OFF_DIR="$PROJECT_ROOT/apps/miniFE/cuda/src"
CPU_REF_SRC="$PROJECT_ROOT/apps/miniFE/openmp/src"
REPORT_DIR="$PROJECT_ROOT/reports/miniFE/performance_comparison"
TS=$(date +%Y%m%d_%H%M%S)
TXT="$REPORT_DIR/comparison_${TS}.txt"
CSV="$REPORT_DIR/comparison_${TS}.csv"
mkdir -p "$REPORT_DIR"

SIZES=(50 75 100 128 150 175 200)

command -v nvcc   >/dev/null || { echo "[ERROR] nvcc not found"; exit 1; }
command -v mpicxx >/dev/null || { echo "[ERROR] mpicxx not found"; exit 1; }

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

# ---------- Build LLM-CUDA bench ----------
log "Building LLM-CUDA benchmark"
[ -d "$LLM_DIR" ] || { echo "[ERROR] LLM dir missing"; exit 1; }

if [ -f "$CPU_REF_SRC/get_common_files" ]; then
    (cd "$CPU_REF_SRC" && bash get_common_files >/dev/null 2>&1) || true
fi
if [ -f "$CPU_REF_SRC/generate_info_header" ]; then
    (cd "$CPU_REF_SRC" && bash generate_info_header "nvcc" "-O2" "miniFE" "MINIFE" >/dev/null 2>&1) || true
fi

cd "$LLM_DIR"
make clean >/dev/null 2>&1 || true
make 2>&1 | tail -3
[ -f "$LLM_DIR/miniFE_cuda_bench" ] || { echo "[ERROR] LLM-CUDA build failed"; exit 1; }
info "LLM-CUDA OK"

# ---------- Build Official CUDA ----------
log "Building Official CUDA"
OFF_OK=0
OFF_BIN=""
if [ -d "$OFF_DIR" ]; then
    cd "$OFF_DIR"
    [ -f get_common_files ]     && bash get_common_files     >/dev/null 2>&1 || true
    [ -f generate_info_header ] && bash generate_info_header "nvcc" "-O3" "miniFE" "MINIFE" >/dev/null 2>&1 || true

    # ---- Patch source files
    [ -f nvtx_stub.h ] && rm -f nvtx_stub.h && info "Removed stale nvtx_stub.h"

    for f in *.hpp *.cuh *.h *.cu *.cpp; do
        [ -f "$f" ] || continue
        sed -i 's|#include <nvToolsExt.h>|#include <nvtx3/nvToolsExt.h>|g' "$f"
        sed -i 's|#include "nvtx_stub.h"|#include <nvtx3/nvToolsExt.h>|g' "$f"
        sed -i 's|cudaThreadSetCacheConfig|cudaDeviceSetCacheConfig|g' "$f"
    done
    info "Source patches applied"

    # ---- Patch Makefile
    if [ -f Makefile ]; then
        [ ! -f Makefile.orig ] && cp Makefile Makefile.orig

        sed -i 's|sm_35|sm_89|g; s|compute_35|compute_89|g' Makefile

        if ! grep -q "INJECTED_MPI_INC" Makefile; then
            MPI_INC_FLAGS=$(mpicxx -show 2>/dev/null | grep -oE '\-I[^ ]+' | tr '\n' ' ')
            [ -z "$MPI_INC_FLAGS" ] && MPI_INC_FLAGS="-I$(dirname "$(dirname "$(command -v mpicxx)")")/include"
            info "MPI include: $MPI_INC_FLAGS"
            sed -i "1i # INJECTED_MPI_INC=1\nNVCCFLAGS_EXTRA=$MPI_INC_FLAGS" Makefile
            sed -i "s|^NVCCFLAGS=|NVCCFLAGS=\$(NVCCFLAGS_EXTRA) |" Makefile
        fi

        # Drop legacy libraries that aren't in modern glibc / nvtx3 is header-only
        sed -i 's|-l[[:space:]]*nvToolsExt||g; s|-lnsl||g; s|-lutil||g' Makefile
    fi

    MPI_HOME_DETECT=$(dirname "$(dirname "$(command -v mpicxx)")")
    export MPI_HOME="${MPI_HOME:-$MPI_HOME_DETECT}"
    info "MPI_HOME=$MPI_HOME"

    if make clean >/dev/null 2>&1 && make MPI_HOME="$MPI_HOME" 2>/tmp/minife_off_build.log; then
        OFF_BIN=$(find "$OFF_DIR" -maxdepth 1 -type f \( -name "miniFE.x" -o -name "miniFE" \) -executable | head -1)
        if [ -n "$OFF_BIN" ] && [ -x "$OFF_BIN" ]; then
            OFF_OK=1
            info "Official CUDA OK ($OFF_BIN)"
        fi
    fi
    [ "$OFF_OK" -eq 0 ] && warn "Official CUDA build failed. Log: /tmp/minife_off_build.log"
else
    warn "Official CUDA dir missing: $OFF_DIR"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
CPU_NAME=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "Unknown")

# ---------- Output extractors ----------
extract_llm_cpu_total()  { echo "$1" | grep -oE "\[CPU CG\] Timings:.*TOTAL=[0-9.]+s" | grep -oE "TOTAL=[0-9.]+" | tail -1 | cut -d= -f2; }
extract_llm_gpu_total()  { echo "$1" | grep -oE "\[CUDA CG\] Timings:.*TOTAL=[0-9.]+s" | grep -oE "TOTAL=[0-9.]+" | tail -1 | cut -d= -f2; }
extract_llm_gpu_matvec() { echo "$1" | grep -oE "\[CUDA CG\] Timings: MATVEC=[0-9.]+" | grep -oE "MATVEC=[0-9.]+" | tail -1 | cut -d= -f2; }
extract_llm_gpu_dot()    { echo "$1" | grep -oE "\[CUDA CG\] Timings:.*DOT=[0-9.]+"   | grep -oE "DOT=[0-9.]+"   | tail -1 | cut -d= -f2; }
extract_llm_gpu_waxpby() { echo "$1" | grep -oE "\[CUDA CG\] Timings:.*WAXPBY=[0-9.]+" | grep -oE "WAXPBY=[0-9.]+" | tail -1 | cut -d= -f2; }
extract_llm_validation() { echo "$1" | grep -oE "Max \|x_cpu - x_gpu\| = [0-9.eE+-]+" | tail -1 | awk '{print $NF}'; }

extract_off_total() {
    local val
    val=$(echo "$1" | grep -E "^[[:space:]]*Total CG Time:" | head -1 | awk -F': *' '{print $2}' | awk '{print $1}')
    [ -z "$val" ] && val=$(echo "$1" | grep -E "^[[:space:]]*Total Program Time:" | head -1 | awk -F': *' '{print $2}' | awk '{print $1}')
    [ -z "$val" ] && val=$(grep -h "Total CG Time:" "$OFF_DIR"/miniFE.*.yaml 2>/dev/null | tail -1 | awk -F': *' '{print $2}' | awk '{print $1}')
    echo "${val:-N/A}"
}

# ---------- nsys helper ----------
run_nsys_with_mpi() {
    local bin="$1"; shift
    local args="$1"; shift
    local logfile="$1"; shift
    local rep="/tmp/nsys_fe_$$_$RANDOM"
    nsys profile -t cuda --stats=true --force-overwrite=true \
        -o "$rep" \
        mpirun --oversubscribe --allow-run-as-root -np 1 \
        $bin $args > "$logfile" 2>&1 || warn "nsys failed: $(basename "$logfile")"
    rm -f "${rep}.nsys-rep" "${rep}.qdstrm" "${rep}.sqlite" 2>/dev/null
    eval "$(parse_nsys_log "$logfile")"
    : "${kern_ns:=0}" "${h2d_ns:=0}" "${d2h_ns:=0}"
    awk -v k="$kern_ns" -v h="$h2d_ns" -v d="$d2h_ns" \
        'BEGIN{printf "%.2f %.2f %.2f", k/1e6, h/1e6, d/1e6}'
}

# ---------- Run sweep ----------
log "Running sweep"
declare -A R_CPU_S R_LLM_S R_LLM_MV R_LLM_DOT R_LLM_WAX R_LLM_VAL R_OFF_S \
           R_LLM_NS_KERN R_LLM_NS_H2D R_LLM_NS_D2H \
           R_OFF_NS_KERN R_OFF_NS_H2D R_OFF_NS_D2H

for N in "${SIZES[@]}"; do
    log "  Size ${N}^3"

    cd "$LLM_DIR"
    OMP_NUM_THREADS=$(nproc) \
        mpirun --oversubscribe --allow-run-as-root -np 1 \
        ./miniFE_cuda_bench -nx $N -ny $N -nz $N \
        > "$REPORT_DIR/llm_run_${N}_${TS}.log" 2>&1
    out=$(cat "$REPORT_DIR/llm_run_${N}_${TS}.log")
    R_CPU_S[$N]=$(extract_llm_cpu_total  "$out")
    R_LLM_S[$N]=$(extract_llm_gpu_total  "$out")
    R_LLM_MV[$N]=$(extract_llm_gpu_matvec "$out")
    R_LLM_DOT[$N]=$(extract_llm_gpu_dot    "$out")
    R_LLM_WAX[$N]=$(extract_llm_gpu_waxpby "$out")
    R_LLM_VAL[$N]=$(extract_llm_validation "$out")
    info "    CPU CG: ${R_CPU_S[$N]}s  LLM-CUDA: ${R_LLM_S[$N]}s  max|err|=${R_LLM_VAL[$N]}"

    NSYS_LOG="$REPORT_DIR/nsys_llm_${N}_${TS}.log"
    cd "$LLM_DIR"
    read -r km hm dm <<<"$(run_nsys_with_mpi ./miniFE_cuda_bench "-nx $N -ny $N -nz $N" "$NSYS_LOG")"
    R_LLM_NS_KERN[$N]="$km"; R_LLM_NS_H2D[$N]="$hm"; R_LLM_NS_D2H[$N]="$dm"
    info "    LLM nsys: kern=${km}ms H2D=${hm}ms D2H=${dm}ms"

    if [ "$OFF_OK" -eq 1 ]; then
        cd "$OFF_DIR"
        out=$(mpirun --oversubscribe --allow-run-as-root -np 1 \
              "$OFF_BIN" -nx $N -ny $N -nz $N 2>&1)
        echo "$out" > "$REPORT_DIR/official_run_${N}_${TS}.log"
        R_OFF_S[$N]=$(extract_off_total "$out")
        info "    Official CG: ${R_OFF_S[$N]}"

        NSYS_LOG="$REPORT_DIR/nsys_official_${N}_${TS}.log"
        read -r km hm dm <<<"$(run_nsys_with_mpi "$OFF_BIN" "-nx $N -ny $N -nz $N" "$NSYS_LOG")"
        R_OFF_NS_KERN[$N]="$km"; R_OFF_NS_H2D[$N]="$hm"; R_OFF_NS_D2H[$N]="$dm"
        info "    Official nsys: kern=${km}ms H2D=${hm}ms D2H=${dm}ms"
    else
        R_OFF_S[$N]="N/A"
        R_OFF_NS_KERN[$N]="N/A"; R_OFF_NS_H2D[$N]="N/A"; R_OFF_NS_D2H[$N]="N/A"
    fi
done

# ---------- Write report ----------
log "Writing report"
{
echo "========================================================"
echo "  miniFE CPU vs LLM-CUDA vs Official-CUDA"
echo "  Date: $(date)"
echo "  CPU: $CPU_NAME"
echo "  GPU: $GPU_NAME"
echo "  Official CUDA included: $([ "$OFF_OK" -eq 1 ] && echo yes || echo no)"
echo "========================================================"
echo ""
printf "%-8s  %-10s  %-12s  %-10s  %-10s  %-10s  %-12s  %-10s\n" \
    "Size" "CPU(s)" "LLM-GPU(s)" "Spdup" "MATVEC(s)" "DOT(s)" "WAXPBY(s)" "Off-GPU(s)"
printf "%-8s  %-10s  %-12s  %-10s  %-10s  %-10s  %-12s  %-10s\n" \
    "--------" "----------" "------------" "----------" "----------" "----------" "------------" "----------"
for N in "${SIZES[@]}"; do
    sp=$(awk "BEGIN{cpu=${R_CPU_S[$N]:-0}; gpu=${R_LLM_S[$N]:-0}; if (gpu>0) printf \"%.2fx\", cpu/gpu; else print \"N/A\"}")
    printf "%-8s  %-10s  %-12s  %-10s  %-10s  %-10s  %-12s  %-10s\n" \
        "${N}^3" "${R_CPU_S[$N]:-N/A}" "${R_LLM_S[$N]:-N/A}" "$sp" \
        "${R_LLM_MV[$N]:-N/A}" "${R_LLM_DOT[$N]:-N/A}" "${R_LLM_WAX[$N]:-N/A}" "${R_OFF_S[$N]:-N/A}"
done
echo ""
echo "========================================================"
echo "  GPU breakdown per size (from nsys, ms)"
echo "========================================================"
printf "%-8s %12s %12s %12s %12s %12s %12s\n" \
    "Size" "LLM-Kern" "LLM-H2D" "LLM-D2H" "Off-Kern" "Off-H2D" "Off-D2H"
printf "%-8s %12s %12s %12s %12s %12s %12s\n" \
    "--------" "------------" "------------" "------------" "------------" "------------" "------------"
for N in "${SIZES[@]}"; do
    printf "%-8s %12s %12s %12s %12s %12s %12s\n" \
        "${N}^3" "${R_LLM_NS_KERN[$N]}" "${R_LLM_NS_H2D[$N]}" "${R_LLM_NS_D2H[$N]}" \
        "${R_OFF_NS_KERN[$N]}" "${R_OFF_NS_H2D[$N]}" "${R_OFF_NS_D2H[$N]}"
done
echo ""
echo "Validation (max|x_cpu - x_gpu|, threshold 1e-6):"
for N in "${SIZES[@]}"; do
    echo "  ${N}^3: ${R_LLM_VAL[$N]:-N/A}"
done
echo "========================================================"
} > "$TXT"

# ---------- CSV ----------
{
    echo "size,cpu_total_s,llm_total_s,llm_matvec_s,llm_dot_s,llm_waxpby_s,off_total_s,llm_kern_ms,llm_h2d_ms,llm_d2h_ms,off_kern_ms,off_h2d_ms,off_d2h_ms"
    for N in "${SIZES[@]}"; do
        echo "${N},${R_CPU_S[$N]:-},${R_LLM_S[$N]:-},${R_LLM_MV[$N]:-},${R_LLM_DOT[$N]:-},${R_LLM_WAX[$N]:-},${R_OFF_S[$N]:-},${R_LLM_NS_KERN[$N]},${R_LLM_NS_H2D[$N]},${R_LLM_NS_D2H[$N]},${R_OFF_NS_KERN[$N]},${R_OFF_NS_H2D[$N]},${R_OFF_NS_D2H[$N]}"
    done
} > "$CSV"

cat "$TXT"
echo ""
log "Report: $TXT"
log "CSV:    $CSV"