#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# 04_compare_miniGhost.sh - miniGhost CPU vs OpenACC GPU (v4)
#
# v4: replace the invalid `-t uvm` with the correct page-fault flags
#     --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true
#     (whether these surface in --stats=true depends on nsys version and
#      platform; on WSL2 with a consumer GPU support is partial).
###############################################################################

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS:-3}"
TSTEPS=20
NUM_VARS=5
REPORT="$PROJECT_ROOT/reports/miniGhost/performance_comparison"
mkdir -p "$REPORT"

CPU_BIN="$PROJECT_ROOT/apps/miniGhost_build/miniGhost_normal.x"
GPU_SRC="$PROJECT_ROOT/apps/miniGhost_gpu/ref"
GPU_BIN="$GPU_SRC/miniGhost.x"
NVIDIA_MPIRUN="/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin/mpirun"

SIZES=(50 75 100 125 150 175 200)

GREEN='\033[1;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log()  { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }

# ---------- nsys parser
parse_nsys_log() {
    awk '
        /Executing .*cuda_gpu_kern_sum/        { mode="kern"; col=2; in_data=0; next }
        /Executing .*cuda_gpu_mem_time_sum/    { mode="mem";  col=2; in_data=0; next }
        /Executing .*cuda_um_total_sum/        { mode="uvm";  col=1; in_data=0; next }
        /Executing .*cuda_uvm_total_sum/       { mode="uvm";  col=1; in_data=0; next }
        /Executing .*cuda_gpu_um_total_sum/    { mode="uvm";  col=1; in_data=0; next }
        /Executing .*cuda_um_cpu_page_faults_sum/ { mode="uvm_cpu"; col=1; in_data=0; next }
        /Executing .*cuda_um_gpu_page_faults_sum/ { mode="uvm_gpu"; col=1; in_data=0; next }
        /Executing .*stats report/             { mode="";     in_data=0; next }
        mode != "" && /^[[:space:]]*-+[[:space:]]+-+/ { in_data=1; next }

        mode == "kern" && in_data && NF >= 3 {
            v=$col; gsub(",","",v)
            if (v ~ /^[0-9]+(\.[0-9]+)?$/) kern_total += v
        }
        (mode == "mem" || mode == "uvm") && in_data && NF >= 3 && ($0 ~ /HtoD/ || $0 ~ /Host-to-Device/) {
            v=$col; gsub(",","",v)
            if (v ~ /^[0-9]+(\.[0-9]+)?$/) h2d_total += v
        }
        (mode == "mem" || mode == "uvm") && in_data && NF >= 3 && ($0 ~ /DtoH/ || $0 ~ /Device-to-Host/) {
            v=$col; gsub(",","",v)
            if (v ~ /^[0-9]+(\.[0-9]+)?$/) d2h_total += v
        }
        END { printf "kern_ns=%d h2d_ns=%d d2h_ns=%d", kern_total+0, h2d_total+0, d2h_total+0 }
    ' "$1"
}

# ---------- Verify binaries
log "Verifying binaries"
[ -x "$CPU_BIN" ] || CPU_BIN="${CPU_BIN%.x}"
[ -x "$CPU_BIN" ] || { echo "[ERROR] CPU binary not found."; exit 1; }

if [ ! -x "$GPU_BIN" ]; then
    log "Building GPU version"
    cd "$GPU_SRC"
    make -f makefile.gpu clean 2>/dev/null || true
    make -f makefile.gpu 2>&1 | tail -5
    GPU_BIN=$(find "$GPU_SRC" -name "miniGhost*.x" -type f | head -1)
fi
[ -x "$GPU_BIN" ] || { echo "[ERROR] GPU binary not found."; exit 1; }
echo "  CPU binary: $CPU_BIN"
echo "  GPU binary: $GPU_BIN"

# ---------- Run sweep
TS=$(date +%Y%m%d_%H%M%S)
TXT="$REPORT/comparison_${TS}.txt"
CSV="$REPORT/comparison_${TS}.csv"

export ACC_DEVICE_TYPE=nvidia
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")

{
echo "========================================================"
echo "  miniGhost CPU vs OpenACC GPU - end-to-end + breakdown"
echo "  Date: $(date)"
echo "  CPU binary: $CPU_BIN"
echo "  GPU binary: $GPU_BIN"
echo "  GPU: $GPU_NAME"
echo "  CPU runs per size: $RUNS  GPU runs: 1"
echo "  Time steps: $TSTEPS  Variables: $NUM_VARS"
echo "  nsys: -t cuda,openacc + UM page-fault tracking"
echo "========================================================"
echo ""
printf "%-8s  %-9s  %-9s  %-7s  %-9s  %-9s  %-9s  %-9s\n" \
    "Size" "CPU(ms)" "GPU(ms)" "Spdup" "Kern(ms)" "H2D(ms)" "D2H(ms)" "Other(ms)"
printf "%-8s  %-9s  %-9s  %-7s  %-9s  %-9s  %-9s  %-9s\n" \
    "--------" "---------" "---------" "-------" "---------" "---------" "---------" "---------"
} > "$TXT"

echo "size,cpu_ms,gpu_e2e_ms,speedup,kernel_ms,h2d_ms,d2h_ms,host_overhead_ms" > "$CSV"

for N in "${SIZES[@]}"; do
    log "Size ${N}^3"
    MG_ARGS="--nx $N --ny $N --nz $N --num_tsteps $TSTEPS --num_vars $NUM_VARS"

    # CPU averaged
    cpu_sum=0
    for i in $(seq 1 $RUNS); do
        t0=$(date +%s%N)
        mpirun --oversubscribe --allow-run-as-root -np 1 "$CPU_BIN" $MG_ARGS >/dev/null 2>&1
        ms=$(( ($(date +%s%N) - t0) / 1000000 ))
        cpu_sum=$((cpu_sum + ms))
    done
    cpu_avg=$((cpu_sum / RUNS))

    # GPU end-to-end single
    t0=$(date +%s%N)
    "$NVIDIA_MPIRUN" --oversubscribe -np 1 "$GPU_BIN" $MG_ARGS >/dev/null 2>&1
    gpu_e2e=$(( ($(date +%s%N) - t0) / 1000000 ))

    # nsys with UM page-fault tracking
    NSYS_LOG="$REPORT/nsys_${N}_${TS}.log"
    NSYS_REP="/tmp/nsys_mg_${N}_$$"
    nsys profile -t cuda,openacc \
        --cuda-um-cpu-page-faults=true \
        --cuda-um-gpu-page-faults=true \
        --stats=true --force-overwrite=true \
        -o "$NSYS_REP" \
        "$NVIDIA_MPIRUN" --oversubscribe -np 1 "$GPU_BIN" $MG_ARGS \
        > "$NSYS_LOG" 2>&1 || warn "nsys returned non-zero for size ${N}"
    rm -f "${NSYS_REP}.nsys-rep" "${NSYS_REP}.qdstrm" "${NSYS_REP}.sqlite" 2>/dev/null

    eval "$(parse_nsys_log "$NSYS_LOG")"
    : "${kern_ns:=0}" "${h2d_ns:=0}" "${d2h_ns:=0}"
    kern_ms=$(awk "BEGIN{printf \"%.2f\", $kern_ns / 1e6}")
    h2d_ms=$(awk "BEGIN{printf \"%.2f\", $h2d_ns / 1e6}")
    d2h_ms=$(awk "BEGIN{printf \"%.2f\", $d2h_ns / 1e6}")

    other_ms=$(awk "BEGIN{r = $gpu_e2e - $kern_ms - $h2d_ms - $d2h_ms; printf \"%.2f\", (r<0?0:r)}")
    speedup=$(awk "BEGIN{printf \"%.2f\", $cpu_avg / ($gpu_e2e>0?$gpu_e2e:1)}")

    printf "%-8s  %-9s  %-9s  %-7s  %-9s  %-9s  %-9s  %-9s\n" \
        "${N}^3" "$cpu_avg" "$gpu_e2e" "${speedup}x" "$kern_ms" "$h2d_ms" "$d2h_ms" "$other_ms" >> "$TXT"
    echo "${N},${cpu_avg},${gpu_e2e},${speedup},${kern_ms},${h2d_ms},${d2h_ms},${other_ms}" >> "$CSV"
done

# ---------- Validation
log "Validation pass"
VAL_LOG_DIR="$REPORT/validation_${TS}"
mkdir -p "$VAL_LOG_DIR"

_validate_run() {
    local binary="$1" logfile="$2" size="$3" launcher="$4"
    eval "$launcher --oversubscribe -np 1 '$binary'" \
        --nx "$size" --ny "$size" --nz "$size" \
        --num_tsteps "$TSTEPS" --num_vars "$NUM_VARS" \
        --error_tol 10 --report_diffusion 1 \
        > "$logfile" 2>&1 || true
    grep -q "within error tolerance" "$logfile" && echo "PASS" || echo "FAIL"
}

{
    echo ""
    echo "========================================================"
    echo "  Numerical validation"
    echo "========================================================"
    printf "%-8s  %-8s  %-8s\n" "Size" "CPU" "GPU"
    printf "%-8s  %-8s  %-8s\n" "--------" "--------" "--------"
} >> "$TXT"

for N in "${SIZES[@]}"; do
    CPU_RES=$(_validate_run "$CPU_BIN" "$VAL_LOG_DIR/cpu_${N}.log" "$N" "mpirun --allow-run-as-root")
    GPU_RES=$(_validate_run "$GPU_BIN" "$VAL_LOG_DIR/gpu_${N}.log" "$N" "$NVIDIA_MPIRUN")
    printf "%-8s  %-8s  %-8s\n" "${N}^3" "$CPU_RES" "$GPU_RES" >> "$TXT"
    log "  ${N}^3: CPU=$CPU_RES  GPU=$GPU_RES"
done

{
    echo "========================================================"
    echo "  CSV: $(basename "$CSV")"
    echo "  nsys logs: nsys_<N>_${TS}.log"
    echo "========================================================"
} >> "$TXT"

echo ""
cat "$TXT"
log "Done. Report: $TXT"
log "      CSV:    $CSV"