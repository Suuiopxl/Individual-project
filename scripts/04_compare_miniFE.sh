#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# 04_compare_miniFE.sh - miniFE CPU vs LLM-CUDA vs Official-CUDA
#
# Sweeps 7 grid sizes (50..200 cubes) using:
#   - apps/miniFE_gpu/cuda_manual/miniFE_cuda_bench  (runs CPU + LLM-CUDA in one go)
#   - apps/miniFE/cuda/src/miniFE.x                  (official CUDA, optional)
# CPU baseline is the OpenMP CG inside the cuda_manual binary.
# nsys breakdown on the largest size for both GPU versions.
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

command -v nvcc >/dev/null  || { echo "[ERROR] nvcc not found"; exit 1; }
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

# ---------- Build LLM-CUDA bench ----------
log "Building LLM-CUDA benchmark"
[ -d "$LLM_DIR" ] || { echo "[ERROR] LLM dir missing: $LLM_DIR"; exit 1; }

# get_common_files / generate_info_header (best-effort)
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

# ---------- Build Official CUDA (defensive) ----------
log "Building Official CUDA (apps/miniFE/cuda/src/)"
OFF_OK=0
OFF_BIN=""
if [ -d "$OFF_DIR" ]; then
    cd "$OFF_DIR"
    # Generate common files
    [ -f get_common_files ]   && bash get_common_files   >/dev/null 2>&1 || true
    [ -f generate_info_header ] && bash generate_info_header "nvcc" "-O3" "miniFE" "MINIFE" >/dev/null 2>&1 || true

    # Patch arch from the historical compute_35 to compute_89; idempotent
    if [ -f Makefile ]; then
        [ ! -f Makefile.orig ] && cp Makefile Makefile.orig
        sed -i 's|arch=compute_35,code=\\"sm_35,compute_35\\"|arch=compute_89,code=\\"sm_89,compute_89\\"|g' Makefile
        # Some Makefile.* files include further nvcc flags
        for f in make_targets Makefile.config Makefile; do
            [ -f "$f" ] && sed -i 's|sm_35|sm_89|g; s|compute_35|compute_89|g' "$f"
        done
    fi

    # NVHPC's mpicxx is the simplest; export MPI_HOME to its parent
    MPI_HOME_DETECT=$(dirname "$(dirname "$(command -v mpicxx)")")
    export MPI_HOME="${MPI_HOME:-$MPI_HOME_DETECT}"

    if make clean >/dev/null 2>&1 && make 2>/tmp/minife_off_build.log; then
        OFF_BIN=$(find "$OFF_DIR" -maxdepth 1 -type f \( -name "miniFE.x" -o -name "miniFE" \) -executable | head -1)
        if [ -n "$OFF_BIN" ] && [ -x "$OFF_BIN" ]; then
            OFF_OK=1
            info "Official CUDA OK ($OFF_BIN)"
        fi
    fi
    [ "$OFF_OK" -eq 0 ] && warn "Official CUDA build failed. See /tmp/minife_off_build.log"
else
    warn "Official CUDA dir missing: $OFF_DIR; skipping."
fi

# ---------- System info ----------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
CPU_NAME=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "Unknown")

# ---------- Output extractors for cuda_manual binary ----------
extract_llm_cpu_total()    { echo "$1" | grep -oE "\[CPU CG\] Timings:.*TOTAL=[0-9.]+s" | grep -oE "TOTAL=[0-9.]+" | tail -1 | cut -d= -f2; }
extract_llm_gpu_total()    { echo "$1" | grep -oE "\[CUDA CG\] Timings:.*TOTAL=[0-9.]+s" | grep -oE "TOTAL=[0-9.]+" | tail -1 | cut -d= -f2; }
extract_llm_gpu_matvec()   { echo "$1" | grep -oE "\[CUDA CG\] Timings: MATVEC=[0-9.]+" | grep -oE "MATVEC=[0-9.]+" | tail -1 | cut -d= -f2; }
extract_llm_gpu_dot()      { echo "$1" | grep -oE "\[CUDA CG\] Timings:.*DOT=[0-9.]+"   | grep -oE "DOT=[0-9.]+"   | tail -1 | cut -d= -f2; }
extract_llm_gpu_waxpby()   { echo "$1" | grep -oE "\[CUDA CG\] Timings:.*WAXPBY=[0-9.]+" | grep -oE "WAXPBY=[0-9.]+" | tail -1 | cut -d= -f2; }
extract_llm_validation()   { echo "$1" | grep -oE "Max \|x_cpu - x_gpu\| = [0-9.eE+-]+" | tail -1 | awk '{print $NF}'; }

# Official miniFE writes a YAML doc with timing fields. Look for "Total CG Time".
extract_off_total() {
    # YAML snippet usually contains: "Total CG Time: X.YYY"
    echo "$1" | grep -E "(Total CG Time|Total Program Time|CG Mflops):" | head -1 | awk -F': *' '{print $2}' | awk '{print $1}'
}

# ---------- Run sweep ----------
log "Running sweep (CPU runs are inside the bench binaries)"
declare -A R_CPU_S R_LLM_S R_LLM_MV R_LLM_DOT R_LLM_WAX R_LLM_VAL R_OFF_S
for N in "${SIZES[@]}"; do
    log "  Size ${N}^3"

    # LLM bench (runs CPU + LLM-CUDA together)
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
    info "    CPU CG total: ${R_CPU_S[$N]}s   LLM-CUDA total: ${R_LLM_S[$N]}s   max|x_cpu-x_gpu|=${R_LLM_VAL[$N]}"

    # Official CUDA bench
    if [ "$OFF_OK" -eq 1 ]; then
        cd "$OFF_DIR"
        out=$(mpirun --oversubscribe --allow-run-as-root -np 1 \
              "$OFF_BIN" -nx $N -ny $N -nz $N 2>&1)
        echo "$out" > "$REPORT_DIR/official_run_${N}_${TS}.log"
        R_OFF_S[$N]=$(extract_off_total "$out")
        [ -z "${R_OFF_S[$N]}" ] && R_OFF_S[$N]="N/A"
        info "    Official CG total: ${R_OFF_S[$N]}"
    else
        R_OFF_S[$N]="N/A"
    fi
done

# ---------- nsys breakdown on largest size ----------
NMAX="${SIZES[-1]}"
log "nsys breakdown at ${NMAX}^3"

LLM_NSYS_LOG="$REPORT_DIR/nsys_llm_${TS}.log"
cd "$LLM_DIR"
nsys profile -t cuda --stats=true --force-overwrite=true \
    -o "/tmp/nsys_fe_llm_$$" \
    mpirun --oversubscribe --allow-run-as-root -np 1 \
    ./miniFE_cuda_bench -nx $NMAX -ny $NMAX -nz $NMAX \
    > "$LLM_NSYS_LOG" 2>&1 || warn "nsys (LLM) failed"
rm -f /tmp/nsys_fe_llm_$$.* 2>/dev/null
eval "$(parse_nsys_log "$LLM_NSYS_LOG")"
LLM_KERN_MS=$(awk "BEGIN{printf \"%.2f\", ${kern_ns:-0} / 1e6}")
LLM_H2D_MS=$(awk "BEGIN{printf \"%.2f\", ${h2d_ns:-0} / 1e6}")
LLM_D2H_MS=$(awk "BEGIN{printf \"%.2f\", ${d2h_ns:-0} / 1e6}")

if [ "$OFF_OK" -eq 1 ]; then
    OFF_NSYS_LOG="$REPORT_DIR/nsys_official_${TS}.log"
    cd "$OFF_DIR"
    nsys profile -t cuda --stats=true --force-overwrite=true \
        -o "/tmp/nsys_fe_off_$$" \
        mpirun --oversubscribe --allow-run-as-root -np 1 \
        "$OFF_BIN" -nx $NMAX -ny $NMAX -nz $NMAX \
        > "$OFF_NSYS_LOG" 2>&1 || warn "nsys (Official) failed"
    rm -f /tmp/nsys_fe_off_$$.* 2>/dev/null
    eval "$(parse_nsys_log "$OFF_NSYS_LOG")"
    OFF_KERN_MS=$(awk "BEGIN{printf \"%.2f\", ${kern_ns:-0} / 1e6}")
    OFF_H2D_MS=$(awk "BEGIN{printf \"%.2f\", ${h2d_ns:-0} / 1e6}")
    OFF_D2H_MS=$(awk "BEGIN{printf \"%.2f\", ${d2h_ns:-0} / 1e6}")
else
    OFF_KERN_MS="N/A"; OFF_H2D_MS="N/A"; OFF_D2H_MS="N/A"
fi

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
echo "  GPU breakdown at ${NMAX}^3 (from nsys)"
echo "========================================================"
printf "%-22s %12s %12s %12s\n" "Config" "Kernel(ms)" "H2D(ms)" "D2H(ms)"
printf "%-22s %12s %12s %12s\n" "----------------------" "------------" "------------" "------------"
printf "%-22s %12s %12s %12s\n" "LLM-CUDA"      "$LLM_KERN_MS" "$LLM_H2D_MS" "$LLM_D2H_MS"
printf "%-22s %12s %12s %12s\n" "Official CUDA" "$OFF_KERN_MS" "$OFF_H2D_MS" "$OFF_D2H_MS"
echo ""
echo "Validation (max|x_cpu - x_gpu|, threshold 1e-6):"
for N in "${SIZES[@]}"; do
    echo "  ${N}^3: ${R_LLM_VAL[$N]:-N/A}"
done
echo "========================================================"
} > "$TXT"

# ---------- CSV ----------
{
    echo "size,cpu_total_s,llm_total_s,llm_matvec_s,llm_dot_s,llm_waxpby_s,off_total_s,llm_kern_ms_atmax,llm_h2d_ms_atmax,llm_d2h_ms_atmax,off_kern_ms_atmax,off_h2d_ms_atmax,off_d2h_ms_atmax"
    for N in "${SIZES[@]}"; do
        if [ "$N" = "$NMAX" ]; then
            kn="$LLM_KERN_MS"; hn="$LLM_H2D_MS"; dn="$LLM_D2H_MS"
            ko="$OFF_KERN_MS"; ho="$OFF_H2D_MS"; do_="$OFF_D2H_MS"
        else
            kn=""; hn=""; dn=""; ko=""; ho=""; do_=""
        fi
        echo "${N},${R_CPU_S[$N]:-},${R_LLM_S[$N]:-},${R_LLM_MV[$N]:-},${R_LLM_DOT[$N]:-},${R_LLM_WAX[$N]:-},${R_OFF_S[$N]:-},${kn},${hn},${dn},${ko},${ho},${do_}"
    done
} > "$CSV"

cat "$TXT"
echo ""
log "Report: $TXT"
log "CSV:    $CSV"
