#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# profile_miniFE_perf.sh — perf/valgrind profiling for miniFE
#
# Fixes: perf record fails when wrapping mpirun on WSL2.
# Solution: run miniFE directly (it works single-process without mpirun),
#           or use valgrind callgrind as fallback.
#
# Usage:
#   cd ~/Individual-project
#   bash scripts/profile_miniFE_perf.sh          # default 128^3
#   bash scripts/profile_miniFE_perf.sh 200       # custom grid
###############################################################################

GREEN='\033[1;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
NC='\033[0m'
log()  { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GRID_SIZE="${1:-128}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="$PROJECT_ROOT/reports/miniFE"
BUILD_DIR="$PROJECT_ROOT/apps/miniFE_build"
SRC_DIR="$PROJECT_ROOT/apps/miniFE/openmp/src"

mkdir -p "$REPORT_DIR" "$BUILD_DIR"

log "=========================================="
log " miniFE Profiling (perf / valgrind)"
log " Grid: ${GRID_SIZE}^3"
log "=========================================="

# ============================================================
# Step 1: Build
# ============================================================
log "Step 1: Building miniFE (-O3, no -pg)"

cd "$SRC_DIR"
make clean >/dev/null 2>&1 || true
make CXX=mpicxx CC=mpicc \
    > "$REPORT_DIR/build_perf_${TIMESTAMP}.log" 2>&1 || {
    echo "[ERROR] Build failed."
    tail -20 "$REPORT_DIR/build_perf_${TIMESTAMP}.log"
    exit 1
}
cp miniFE.x "$BUILD_DIR/miniFE_normal.x"
info "Built: $BUILD_DIR/miniFE_normal.x"

# ============================================================
# Step 2: Try perf (directly on binary, NOT through mpirun)
# ============================================================
log "Step 2: Attempting perf record"

cd "$BUILD_DIR"
RUN_ARGS="-nx $GRID_SIZE -ny $GRID_SIZE -nz $GRID_SIZE"
PERF_OK=false

if command -v perf &>/dev/null; then
    # Lower paranoid level
    sudo sysctl -w kernel.perf_event_paranoid=-1 2>/dev/null || true

    # Key fix: run perf directly on the binary, not through mpirun.
    # miniFE works fine as a single process without mpirun.
    info "Running: perf record ./miniFE_normal.x $RUN_ARGS"

    OMP_NUM_THREADS=1 \
    perf record -g -o "$REPORT_DIR/perf_${TIMESTAMP}.data" -- \
        ./miniFE_normal.x $RUN_ARGS \
        > "$REPORT_DIR/perf_run_${TIMESTAMP}.log" 2>&1 && PERF_OK=true

    if [ "$PERF_OK" = false ]; then
        info "Retrying without -g..."
        OMP_NUM_THREADS=1 \
        perf record -o "$REPORT_DIR/perf_${TIMESTAMP}.data" -- \
            ./miniFE_normal.x $RUN_ARGS \
            > "$REPORT_DIR/perf_run_${TIMESTAMP}.log" 2>&1 && PERF_OK=true
    fi
fi

if [ "$PERF_OK" = true ]; then
    log "Step 3: Generating perf report"

    perf report -i "$REPORT_DIR/perf_${TIMESTAMP}.data" \
        --stdio --no-children -n --percent-limit 0.5 \
        > "$REPORT_DIR/perf_report_${TIMESTAMP}.txt" 2>&1

    # Extract top hotspots
    grep -E '^\s+[0-9]+\.[0-9]+%' "$REPORT_DIR/perf_report_${TIMESTAMP}.txt" \
        | head -20 \
        > "$REPORT_DIR/perf_top20_${TIMESTAMP}.txt" 2>/dev/null || true

    echo ""
    echo "========================================"
    echo "  perf Top Hotspots"
    echo "========================================"
    cat "$REPORT_DIR/perf_top20_${TIMESTAMP}.txt" 2>/dev/null || \
        head -40 "$REPORT_DIR/perf_report_${TIMESTAMP}.txt"
    echo "========================================"

    PROFILER="perf"

else
    # ============================================================
    # Fallback: valgrind callgrind
    # ============================================================
    log "Step 2b: perf failed, using valgrind callgrind"

    if ! command -v valgrind &>/dev/null; then
        info "Installing valgrind..."
        sudo apt-get install -y valgrind 2>/dev/null || {
            echo "[ERROR] Cannot install valgrind. No profiler available."
            exit 1
        }
    fi

    # Smaller grid for callgrind (10-20x slower)
    CG_GRID=$((GRID_SIZE > 80 ? 80 : GRID_SIZE))
    warn "Using grid ${CG_GRID}^3 for callgrind (instrumentation overhead)"

    OMP_NUM_THREADS=1 \
    valgrind --tool=callgrind \
        --callgrind-out-file="$REPORT_DIR/callgrind_${TIMESTAMP}.out" \
        ./miniFE_normal.x \
        -nx $CG_GRID -ny $CG_GRID -nz $CG_GRID \
        > "$REPORT_DIR/callgrind_run_${TIMESTAMP}.log" 2>&1

    info "callgrind complete"

    callgrind_annotate "$REPORT_DIR/callgrind_${TIMESTAMP}.out" \
        > "$REPORT_DIR/callgrind_annotate_${TIMESTAMP}.txt" 2>&1

    echo ""
    echo "========================================"
    echo "  callgrind Top Hotspots"
    echo "========================================"
    head -50 "$REPORT_DIR/callgrind_annotate_${TIMESTAMP}.txt"
    echo "========================================"

    PROFILER="callgrind"
fi

# ============================================================
# Step 4: Generate JSON summary (compatible with 02_analyze_hotspots.py)
# ============================================================
log "Step 4: Generating profiling summary JSON"

python3 << PYEOF
import json, re, os

report_dir = "$REPORT_DIR"
timestamp = "$TIMESTAMP"
grid = "$GRID_SIZE"
profiler = "$PROFILER"

hotspots = []

if profiler == "perf":
    perf_file = os.path.join(report_dir, f"perf_report_{timestamp}.txt")
    if os.path.exists(perf_file):
        with open(perf_file) as f:
            for line in f:
                m = re.match(r'\s+(\d+\.\d+)%\s+(\d+)\s+\S+\s+\[\.\]\s+(.+)', line)
                if m:
                    pct = m.group(1)
                    samples = m.group(2)
                    func = m.group(3).strip()
                    if any(skip in func for skip in ['frame_dummy', '__libc', '_start', '__GI_', '_dl_', '__do_global']):
                        continue
                    hotspots.append({
                        "pct_time": pct,
                        "self_sec": "N/A",
                        "cumulative_sec": "N/A",
                        "calls": samples,
                        "function": func
                    })
                if len(hotspots) >= 15:
                    break

elif profiler == "callgrind":
    cg_file = os.path.join(report_dir, f"callgrind_annotate_{timestamp}.txt")
    if os.path.exists(cg_file):
        in_summary = False
        with open(cg_file) as f:
            for line in f:
                if 'Ir' in line and 'file:function' in line:
                    in_summary = True
                    continue
                if in_summary:
                    m = re.match(r'\s*([\d,]+)\s+\(\s*([\d.]+)%\)\s+(.+)', line)
                    if m:
                        cost = m.group(1).replace(',', '')
                        pct = m.group(2)
                        func_raw = m.group(3).strip()
                        func = func_raw.split(':')[-1] if ':' in func_raw else func_raw
                        if float(pct) >= 1.0:
                            hotspots.append({
                                "pct_time": pct,
                                "self_sec": "N/A",
                                "cumulative_sec": "N/A",
                                "calls": cost,
                                "function": func
                            })
                    if len(hotspots) >= 15:
                        break

summary = {
    "application": "miniFE",
    "language": "cpp",
    "timestamp": timestamp,
    "profiler": profiler,
    "config": {
        "mpi_ranks": 1,
        "omp_threads": 1,
        "problem_args": f"-nx {grid} -ny {grid} -nz {grid}",
        "baseline_runs": 3,
        "build_system": "make"
    },
    "gprof_hotspots": hotspots,
    "source_dir": "$SRC_DIR",
    "next_step": "Run 02_analyze_hotspots.py miniFE to feed hotspot functions to LLM"
}

out_path = os.path.join(report_dir, f"profiling_summary_{timestamp}.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Profiler: {profiler}")
print(f"  JSON: {out_path}")
print(f"  Hotspots found: {len(hotspots)}")
for i, h in enumerate(hotspots[:10]):
    print(f"    {i+1}. {h['pct_time']:>6}% - {h['function']}")
PYEOF

log "=========================================="
log " miniFE Profiling Complete!"
log " Reports: $REPORT_DIR"
log "=========================================="
echo ""
echo "  Next step:"
echo "    source .env"
echo "    python3 scripts/02_analyze_hotspots.py miniFE"