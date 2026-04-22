#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 01_profile_app.sh — Universal profiling pipeline for any configured app
#
# Reads build/run configuration from app_config.json instead of hardcoding.
# Supports: Make-based (miniGhost, XSBench) and CMake-based (udunits, LAMMPS)
#
# Usage:
#   bash scripts/01_profile_app.sh miniGhost
#   bash scripts/01_profile_app.sh udunits
#   NP=4 BASELINE_RUNS=5 bash scripts/01_profile_app.sh XSBench
#
# Directory Convention (from app_config.json):
#   apps/{app}/       ← Cloned source
#   apps/{app}_build/ ← Build artifacts
#   reports/{app}/    ← Output reports
###############################################################################

# ======================== Parameters ========================
APP_NAME="${1:?Usage: bash scripts/01_profile_app.sh <app_name>}"
BASELINE_RUNS="${BASELINE_RUNS:-3}"
TOP_N="${TOP_N:-10}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ======================== Locate project root ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_FILE="$PROJECT_ROOT/app_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] app_config.json not found at $CONFIG_FILE"
    exit 1
fi

# ======================== Helper functions ========================
log()  { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }
err()  { echo -e "\033[1;31m[ERROR] $*\033[0m" >&2; }

# JSON reader using Python (no jq dependency)
cfg() {
    python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    data = json.load(f)
keys = '$1'.split('.')
node = data['$APP_NAME']
for k in keys:
    if isinstance(node, list):
        node = node[int(k)]
    else:
        node = node[k]
if isinstance(node, list):
    print(' '.join(str(x) for x in node))
elif isinstance(node, bool):
    print('true' if node else 'false')
else:
    print(node)
" 2>/dev/null
}

# ======================== Read config ========================
log "Reading configuration for $APP_NAME..."

BUILD_SYSTEM=$(cfg "build.system")
BINARY_NAME=$(cfg "build.binary_name")
SRC_SUBDIR=$(cfg "src_subdir")
REQUIRES_MPI=$(cfg "run.requires_mpi")
NP="${NP:-$(cfg "run.default_np")}"
RUN_ARGS=$(cfg "run.args")
PROFILING_ARGS=$(cfg "run.profiling_args" 2>/dev/null || echo "")
GPROF_FLAGS=$(cfg "profiling.gprof_flags")
LANGUAGE=$(cfg "language")

SRC_DIR="$PROJECT_ROOT/apps/$APP_NAME/$SRC_SUBDIR"
REPORT_DIR="$PROJECT_ROOT/reports/$APP_NAME"
BUILD_DIR="$PROJECT_ROOT/apps/${APP_NAME}_build"

mkdir -p "$REPORT_DIR" "$BUILD_DIR"

log "  Build system: $BUILD_SYSTEM | Language: $LANGUAGE"
log "  Source: $SRC_DIR"
log "  MPI: $REQUIRES_MPI (np=$NP) | Args: $RUN_ARGS"

# ======================== Profiler dispatch ========================
# Different applications may require different profilers. Where gprof is not
# suitable (see the methodology chapter of the report for the miniFE pivot),
# the "profiling.tool" field in app_config.json selects an alternative. When
# absent, the default is gprof and the rest of this script is used unchanged.
PROFILING_TOOL=$(cfg "profiling.tool" 2>/dev/null || echo "gprof")
if [ "$PROFILING_TOOL" = "callgrind" ]; then
    log "Dispatching to Callgrind-based profiler (profiling.tool=callgrind)"
    exec bash "$SCRIPT_DIR/profile_miniFE_perf.sh"
fi

# ======================== Phase 0: Build ========================
phase0_build() {
    log "Phase 0: Build $APP_NAME"

    if [ ! -d "$SRC_DIR" ]; then
        err "Source directory does not exist: $SRC_DIR"
        err "Please run 00_setup_local_env.sh first."
        exit 1
    fi

    if [ "$BUILD_SYSTEM" = "make" ]; then
        build_with_make
    elif [ "$BUILD_SYSTEM" = "cmake" ]; then
        build_with_cmake
    else
        err "Unknown build system: $BUILD_SYSTEM"
        exit 1
    fi
}

build_with_make() {
    local makefile
    makefile=$(cfg "build.makefile")

    cd "$SRC_DIR"

    if [ ! -f "$makefile" ]; then
        err "$makefile not found in $SRC_DIR"
        exit 1
    fi

    # Apply pre-build fixes from config
    local fixes
    fixes=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    data = json.load(f)
fixes = data['$APP_NAME']['build'].get('pre_build_fixes', [])
for fix in fixes:
    print(fix.replace('{makefile}', '$makefile'))
" 2>/dev/null)

    if [ -n "$fixes" ]; then
        log "  Applying pre-build fixes..."
        while IFS= read -r fix; do
            eval "$fix" || warn "Fix command failed: $fix"
        done <<< "$fixes"
    fi

    # Add extra compiler flags (Fortran-specific)
    if [ "$LANGUAGE" = "fortran" ]; then
        local extra_fflags
        extra_fflags=$(cfg "build.extra_fflags" 2>/dev/null || echo "")
        for flag in $extra_fflags; do
            grep -q "FFLAGS += $flag" "$makefile" || echo "FFLAGS += $flag" >> "$makefile"
        done
    fi

    # Determine compiler variables
    local compiler_args=""
    local fc cc ld
    fc=$(cfg "build.compiler.FC" 2>/dev/null || echo "")
    cc=$(cfg "build.compiler.CC" 2>/dev/null || echo "")
    ld=$(cfg "build.compiler.LD" 2>/dev/null || echo "")
    [ -n "$fc" ] && compiler_args="$compiler_args FC=$fc"
    [ -n "$cc" ] && compiler_args="$compiler_args CC=$cc"
    [ -n "$ld" ] && compiler_args="$compiler_args LD=$ld"

    # --- Build normal version ---
    log "  Building normal version (-O2)..."
    make -f "$makefile" clean >/dev/null 2>&1 || true
    eval make -f "$makefile" $compiler_args \
        > "$REPORT_DIR/build_normal_${TIMESTAMP}.log" 2>&1 || {
        err "Build failed, check $REPORT_DIR/build_normal_${TIMESTAMP}.log"
        tail -30 "$REPORT_DIR/build_normal_${TIMESTAMP}.log"
        exit 1
    }
    cp "$BINARY_NAME" "$BUILD_DIR/${APP_NAME}_normal.x" 2>/dev/null || \
    cp "$BINARY_NAME" "$BUILD_DIR/${APP_NAME}_normal" 2>/dev/null || {
        # Try to find the binary
        local found
        found=$(find . -maxdepth 1 -name "$BINARY_NAME" -o -name "*.x" -o -name "*.out" | head -1)
        if [ -n "$found" ]; then
            cp "$found" "$BUILD_DIR/${APP_NAME}_normal"
        else
            err "Cannot find built binary: $BINARY_NAME"
            exit 1
        fi
    }

    # --- Build gprof version ---
    # Check if this app uses its own profile flag (e.g., XSBench: PROFILE=yes)
    local profile_make_args
    profile_make_args=$(cfg "profiling.profile_make_args" 2>/dev/null || echo "")

    if [ -n "$profile_make_args" ]; then
        # App has its own profiling mechanism (e.g., XSBench PROFILE=yes)
        log "  Building gprof version (using: $profile_make_args)..."
        make -f "$makefile" clean >/dev/null 2>&1 || true
        eval make -f "$makefile" $compiler_args $profile_make_args \
            > "$REPORT_DIR/build_gprof_${TIMESTAMP}.log" 2>&1 || {
            err "gprof version build failed"
            tail -30 "$REPORT_DIR/build_gprof_${TIMESTAMP}.log"
            exit 1
        }
    else
        # Fallback: inject -pg via sed (original method, works for miniGhost etc.)
        log "  Building gprof version ($GPROF_FLAGS)..."
        make -f "$makefile" clean >/dev/null 2>&1 || true
        cp "$makefile" "${makefile}.gprof"

        if [ "$LANGUAGE" = "fortran" ]; then
            sed -i "s/^FFLAGS *=/& $GPROF_FLAGS /" "${makefile}.gprof"
        else
            sed -i "s/^CFLAGS *=/& $GPROF_FLAGS /" "${makefile}.gprof"
            sed -i "s/^CXXFLAGS *=/& $GPROF_FLAGS /" "${makefile}.gprof" 2>/dev/null || true
        fi
        echo "LDFLAGS += $GPROF_FLAGS" >> "${makefile}.gprof"

        eval make -f "${makefile}.gprof" $compiler_args \
            > "$REPORT_DIR/build_gprof_${TIMESTAMP}.log" 2>&1 || {
            err "gprof version build failed"
            tail -30 "$REPORT_DIR/build_gprof_${TIMESTAMP}.log"
            exit 1
        }
        rm -f "${makefile}.gprof"
    fi

    cp "$BINARY_NAME" "$BUILD_DIR/${APP_NAME}_gprof.x" 2>/dev/null || \
    cp "$BINARY_NAME" "$BUILD_DIR/${APP_NAME}_gprof" 2>/dev/null || true

    log "  Build complete ✓"
    ls -lh "$BUILD_DIR"/${APP_NAME}_* 2>/dev/null || true
}

build_with_cmake() {
    local cmake_build="$BUILD_DIR/cmake_build"
    mkdir -p "$cmake_build"

    # Install dependencies if configured
    local deps
    deps=$(cfg "build.dependencies" 2>/dev/null || echo "")
    if [ -n "$deps" ]; then
        log "  Installing dependencies: $deps"
        sudo apt-get install -y -qq $deps 2>/dev/null || warn "Some dependencies may need manual installation"
    fi

    local cmake_args
    cmake_args=$(cfg "build.cmake_args" 2>/dev/null || echo "")

    # --- Normal build ---
    log "  Building normal version (CMake)..."
    cd "$cmake_build"
    rm -rf ./*

    local compiler_defs=""
    local cc cxx
    cc=$(cfg "build.compiler.CC" 2>/dev/null || echo "")
    cxx=$(cfg "build.compiler.CXX" 2>/dev/null || echo "")
    [ -n "$cc" ] && compiler_defs="$compiler_defs -DCMAKE_C_COMPILER=$cc"
    [ -n "$cxx" ] && compiler_defs="$compiler_defs -DCMAKE_CXX_COMPILER=$cxx"

    local app_src="$PROJECT_ROOT/apps/$APP_NAME"
    local cmake_subdir
    cmake_subdir=$(cfg "build.cmake_source_subdir" 2>/dev/null || echo "")
    [ -n "$cmake_subdir" ] && app_src="$app_src/$cmake_subdir"
    eval cmake "$app_src" $cmake_args $compiler_defs \
        > "$REPORT_DIR/build_normal_${TIMESTAMP}.log" 2>&1 || {
        err "CMake configure failed"
        tail -30 "$REPORT_DIR/build_normal_${TIMESTAMP}.log"
        exit 1
    }
    make -j$(nproc) >> "$REPORT_DIR/build_normal_${TIMESTAMP}.log" 2>&1 || {
        err "CMake build failed"
        tail -30 "$REPORT_DIR/build_normal_${TIMESTAMP}.log"
        exit 1
    }

    # Find and copy binary
    local binary
    binary=$(find . -name "$BINARY_NAME" -type f 2>/dev/null | head -1)
    if [ -z "$binary" ]; then
        binary=$(find . -type f -executable 2>/dev/null | head -1)
    fi
    if [ -n "$binary" ]; then
        cp "$binary" "$BUILD_DIR/${APP_NAME}_normal"
    else
        warn "Cannot locate built binary, check build log"
    fi

    # --- gprof build ---
    log "  Building gprof version (CMake + $GPROF_FLAGS)..."
    rm -rf ./*
    eval cmake "$app_src" $cmake_args $compiler_defs \
        -DCMAKE_C_FLAGS="$GPROF_FLAGS" \
        -DCMAKE_CXX_FLAGS="$GPROF_FLAGS" \
        -DCMAKE_EXE_LINKER_FLAGS="$GPROF_FLAGS" \
        > "$REPORT_DIR/build_gprof_${TIMESTAMP}.log" 2>&1 || true
    make -j$(nproc) >> "$REPORT_DIR/build_gprof_${TIMESTAMP}.log" 2>&1 || true

    binary=$(find . -name "$BINARY_NAME" -type f 2>/dev/null | head -1)
    [ -n "$binary" ] && cp "$binary" "$BUILD_DIR/${APP_NAME}_gprof"

    log "  Build complete ✓"
    ls -lh "$BUILD_DIR"/${APP_NAME}_* 2>/dev/null || true
}

# ======================== Phase 1: Baseline ========================
phase1_baseline() {
    log "Phase 1: Baseline Benchmarking (${BASELINE_RUNS} runs)"

    cd "$BUILD_DIR"

    # For LAMMPS: copy input file from source to build dir
    local app_root="$PROJECT_ROOT/apps/$APP_NAME"
    if [ -f "$app_root/bench/in.lj" ]; then
        mkdir -p "$BUILD_DIR/bench"
        cp "$app_root/bench/in.lj" "$BUILD_DIR/bench/"
    fi
    local times=()
    local baseline_log="$REPORT_DIR/baseline_${TIMESTAMP}.log"

    # Determine the run command
    local normal_bin
    normal_bin=$(find "$BUILD_DIR" -name "${APP_NAME}_normal*" -type f | head -1)
    if [ -z "$normal_bin" ]; then
        err "Normal binary not found in $BUILD_DIR"
        exit 1
    fi
    chmod +x "$normal_bin"

    # Check for custom benchmark command
    local bench_cmd
    bench_cmd=$(cfg "run.benchmark_command" 2>/dev/null || echo "")

    for i in $(seq 1 "$BASELINE_RUNS"); do
        log "  Baseline run $i / $BASELINE_RUNS"
        local t0=$(date +%s%N)

        if [ -n "$bench_cmd" ]; then
            # Use custom benchmark command
            eval "$bench_cmd" >> "$baseline_log" 2>&1
        elif [ "$REQUIRES_MPI" = "true" ]; then
            mpirun --oversubscribe --allow-run-as-root -np "$NP" \
                "$normal_bin" $RUN_ARGS >> "$baseline_log" 2>&1
        else
            "$normal_bin" $RUN_ARGS >> "$baseline_log" 2>&1
        fi

        local t1=$(date +%s%N)
        local elapsed_ms=$(( (t1 - t0) / 1000000 ))
        times+=("$elapsed_ms")
        echo "  Run $i: ${elapsed_ms} ms"
    done

    # Calculate statistics
    local sum=0
    for t in "${times[@]}"; do sum=$((sum + t)); done
    local avg=$((sum / BASELINE_RUNS))
    local min=${times[0]} max=${times[0]}
    for t in "${times[@]}"; do
        (( t < min )) && min=$t
        (( t > max )) && max=$t
    done

    cat > "$REPORT_DIR/baseline_summary_${TIMESTAMP}.txt" <<EOF
==============================
$APP_NAME Baseline Summary
==============================
Date:       $(date)
Language:   $LANGUAGE
MPI:        $REQUIRES_MPI (ranks: $NP)
Runs:       $BASELINE_RUNS
Args:       $RUN_ARGS
Wall-clock times (ms): ${times[*]}
Average:  ${avg} ms
Min:      ${min} ms
Max:      ${max} ms
==============================
EOF

    cat "$REPORT_DIR/baseline_summary_${TIMESTAMP}.txt"
    log "Baseline complete ✓"
}

# ======================== Phase 2: gprof ========================
phase2_gprof() {
    log "Phase 2: gprof Performance Analysis"

    cd "$BUILD_DIR"

    # For LAMMPS: copy input file from source to build dir
    local app_root="$PROJECT_ROOT/apps/$APP_NAME"
    if [ -f "$app_root/bench/in.lj" ]; then
        mkdir -p "$BUILD_DIR/bench"
        cp "$app_root/bench/in.lj" "$BUILD_DIR/bench/"
    fi
    rm -f gmon.out

    local gprof_bin
    gprof_bin=$(find "$BUILD_DIR" -name "${APP_NAME}_gprof*" -type f | head -1)
    if [ -z "$gprof_bin" ]; then
        warn "gprof binary not found, skipping."
        return
    fi
    chmod +x "$gprof_bin"

    log "  Running gprof version..."
    if [ "$REQUIRES_MPI" = "true" ]; then
        mpirun --oversubscribe --allow-run-as-root -np 1 \
            "$gprof_bin" $RUN_ARGS \
            > "$REPORT_DIR/gprof_run_${TIMESTAMP}.log" 2>&1
    else
        "$gprof_bin" $RUN_ARGS \
            > "$REPORT_DIR/gprof_run_${TIMESTAMP}.log" 2>&1
    fi

    if [ ! -f gmon.out ]; then
        warn "gmon.out not generated, skipping gprof analysis."
        return
    fi

    gprof "$gprof_bin" gmon.out \
        > "$REPORT_DIR/gprof_full_${TIMESTAMP}.txt" 2>&1

    # Extract flat profile Top-N
    awk '
        /^  %   cumulative   self/ { start=1; next }
        /^$/ && start { exit }
        start && NR>0 { print }
    ' "$REPORT_DIR/gprof_full_${TIMESTAMP}.txt" \
        | head -n "$TOP_N" \
        > "$REPORT_DIR/gprof_top${TOP_N}_${TIMESTAMP}.txt"

    log "gprof analysis complete ✓"
    echo "--- gprof Top-$TOP_N ---"
    cat "$REPORT_DIR/gprof_top${TOP_N}_${TIMESTAMP}.txt"
}

# ======================== Phase 3: JSON Summary ========================
phase3_summary() {
    log "Phase 3: Generating JSON Summary"

    local gprof_file="$REPORT_DIR/gprof_top${TOP_N}_${TIMESTAMP}.txt"
    local gprof_hotspots="[]"

    if [ -s "$gprof_file" ]; then
        gprof_hotspots=$(awk '
            BEGIN { printf "[" }
            NF >= 7 {
                gsub(/"/, "\\\"", $7)
                if (NR > 1) printf ","
                printf "\n    {\"pct_time\": \"%s\", \"cumulative_sec\": \"%s\", \"self_sec\": \"%s\", \"calls\": \"%s\", \"function\": \"%s\"}", $1, $2, $3, $4, $7
            }
            END { printf "\n  ]" }
        ' "$gprof_file")
    fi

    cat > "$REPORT_DIR/profiling_summary_${TIMESTAMP}.json" <<ENDJSON
{
  "application": "$APP_NAME",
  "language": "$LANGUAGE",
  "timestamp": "${TIMESTAMP}",
  "config": {
    "mpi_ranks": ${NP},
    "problem_args": "${RUN_ARGS}",
    "baseline_runs": ${BASELINE_RUNS},
    "build_system": "${BUILD_SYSTEM}"
  },
  "gprof_hotspots": ${gprof_hotspots},
  "source_dir": "$SRC_DIR",
  "next_step": "Run 02_analyze_hotspots.py $APP_NAME to feed hotspot functions to LLM"
}
ENDJSON

    log "JSON Summary ✓"
    echo ""
    echo "=============================================="
    echo "  $APP_NAME Profiling Complete!"
    echo "  Report directory: $REPORT_DIR"
    echo "=============================================="
    echo ""
    echo "  Next step: source .env && python3 scripts/02_analyze_hotspots.py $APP_NAME"
}

# ======================== Main ========================
main() {
    log "=========================================="
    log " $APP_NAME Profiling Pipeline"
    log " Build: $BUILD_SYSTEM | Lang: $LANGUAGE"
    log " MPI: $REQUIRES_MPI (np=$NP) | Runs: $BASELINE_RUNS"
    log "=========================================="

    phase0_build
    phase1_baseline
    phase2_gprof
    phase3_summary
}

main "$@"