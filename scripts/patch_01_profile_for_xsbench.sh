#!/usr/bin/env bash
###############################################################################
# patch_01_profile_for_xsbench.sh
#
# Apply this patch to your existing 01_profile_app.sh to support XSBench's
# Makefile which uses "PROFILE=yes" instead of sed-injecting -pg into CFLAGS.
#
# Usage:
#   cd ~/Individual-project
#   bash patch_01_profile_for_xsbench.sh
#
# What this does:
#   1. Backs up the original 01_profile_app.sh
#   2. Patches the build_with_make() function to check for "profile_make_args"
#   3. Patches the run section to use "profiling_args" when available
###############################################################################

set -euo pipefail

SCRIPT="scripts/01_profile_app.sh"

if [ ! -f "$SCRIPT" ]; then
    echo "[ERROR] $SCRIPT not found. Run from project root."
    exit 1
fi

# Backup
cp "$SCRIPT" "${SCRIPT}.bak"
echo "[OK] Backed up to ${SCRIPT}.bak"

# ============================================================
# Patch 1: Replace the gprof build section in build_with_make()
# ============================================================
# We need to find the "Build gprof version" block and replace it with
# a version that checks for profile_make_args in the config.

cat > /tmp/xsbench_profile_patch.py << 'PYTHON_PATCH'
import re
import sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

# ---------- Patch 1: gprof build method ----------
# Find the "Build gprof version" section and replace with dual-mode logic
old_gprof_block = '''    # --- Build gprof version ---
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

    eval make -f "${makefile}.gprof" $compiler_args \\
        > "$REPORT_DIR/build_gprof_${TIMESTAMP}.log" 2>&1 || {
        err "gprof version build failed"
        tail -30 "$REPORT_DIR/build_gprof_${TIMESTAMP}.log"
        exit 1
    }

    cp "$BINARY_NAME" "$BUILD_DIR/${APP_NAME}_gprof.x" 2>/dev/null || \\
    cp "$BINARY_NAME" "$BUILD_DIR/${APP_NAME}_gprof" 2>/dev/null || true
    rm -f "${makefile}.gprof"'''

new_gprof_block = '''    # --- Build gprof version ---
    # Check if this app uses its own profile flag (e.g., XSBench: PROFILE=yes)
    local profile_make_args
    profile_make_args=$(cfg "profiling.profile_make_args" 2>/dev/null || echo "")

    if [ -n "$profile_make_args" ]; then
        # App has its own profiling mechanism (e.g., XSBench PROFILE=yes)
        log "  Building gprof version (using: $profile_make_args)..."
        make -f "$makefile" clean >/dev/null 2>&1 || true
        eval make -f "$makefile" $compiler_args $profile_make_args \\
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

        eval make -f "${makefile}.gprof" $compiler_args \\
            > "$REPORT_DIR/build_gprof_${TIMESTAMP}.log" 2>&1 || {
            err "gprof version build failed"
            tail -30 "$REPORT_DIR/build_gprof_${TIMESTAMP}.log"
            exit 1
        }
        rm -f "${makefile}.gprof"
    fi

    cp "$BINARY_NAME" "$BUILD_DIR/${APP_NAME}_gprof.x" 2>/dev/null || \\
    cp "$BINARY_NAME" "$BUILD_DIR/${APP_NAME}_gprof" 2>/dev/null || true'''

if old_gprof_block in content:
    content = content.replace(old_gprof_block, new_gprof_block)
    print("[PATCH 1] Replaced gprof build section with dual-mode logic")
else:
    print("[PATCH 1] WARNING: Could not find exact gprof block to replace.")
    print("          You may need to manually edit the build_with_make() function.")
    print("          The key change: check cfg 'profiling.profile_make_args' and if set,")
    print("          use 'make PROFILE=yes' instead of sed-injecting -pg.")

# ---------- Patch 2: Support profiling_args for gprof run ----------
# Find where RUN_ARGS is used for the gprof run and add profiling_args override
# Look for the gprof execution section

# Add PROFILING_ARGS read near the top where RUN_ARGS is read
old_run_args = 'RUN_ARGS=$(cfg "run.args")'
new_run_args = '''RUN_ARGS=$(cfg "run.args")
PROFILING_ARGS=$(cfg "run.profiling_args" 2>/dev/null || echo "")'''

if old_run_args in content and 'PROFILING_ARGS' not in content:
    content = content.replace(old_run_args, new_run_args, 1)
    print("[PATCH 2] Added PROFILING_ARGS variable read from config")
else:
    if 'PROFILING_ARGS' in content:
        print("[PATCH 2] PROFILING_ARGS already present, skipping")
    else:
        print("[PATCH 2] WARNING: Could not find RUN_ARGS line to patch")

# ---------- Patch 3: Use PROFILING_ARGS during gprof run ----------
# In the gprof execution phase, use PROFILING_ARGS if available
# We need to find where gprof binary is run and swap args
# The typical pattern is: $GPROF_BIN $RUN_ARGS or mpirun ... $GPROF_BIN $RUN_ARGS

# Add a helper right before the gprof run section
old_gprof_run_marker = 'log "Phase 2: gprof profiling"'
new_gprof_run_marker = '''log "Phase 2: gprof profiling"

    # Use profiling-specific args if configured (e.g., single-thread + more lookups)
    local GPROF_RUN_ARGS="${PROFILING_ARGS:-$RUN_ARGS}"
    if [ -n "$PROFILING_ARGS" ]; then
        log "  Using profiling-specific args: $GPROF_RUN_ARGS"
    fi'''

if old_gprof_run_marker in content and 'GPROF_RUN_ARGS' not in content:
    content = content.replace(old_gprof_run_marker, new_gprof_run_marker, 1)
    # Also replace $RUN_ARGS with $GPROF_RUN_ARGS in the gprof run command
    # This is trickier — we look for the execution line after Phase 2
    # Try to find and replace in the gprof execution section only
    content = content.replace(
        '$RUN_ARGS 2>&1 | tee "$REPORT_DIR/gprof_run',
        '$GPROF_RUN_ARGS 2>&1 | tee "$REPORT_DIR/gprof_run'
    )
    print("[PATCH 3] Added GPROF_RUN_ARGS support for gprof execution phase")
else:
    if 'GPROF_RUN_ARGS' in content:
        print("[PATCH 3] GPROF_RUN_ARGS already present, skipping")
    else:
        print("[PATCH 3] WARNING: Could not find gprof run marker to patch")

with open(filepath, 'w') as f:
    f.write(content)

print("\n[DONE] Patching complete.")
PYTHON_PATCH

python3 /tmp/xsbench_profile_patch.py "$SCRIPT"

echo ""
echo "[OK] Patched $SCRIPT"
echo ""
echo "Summary of changes:"
echo "  1. build_with_make() now checks 'profiling.profile_make_args' in config"
echo "     → XSBench uses 'make PROFILE=yes' instead of sed -pg injection"
echo "  2. Added PROFILING_ARGS from 'run.profiling_args' config"
echo "     → XSBench gprof runs with '-t 1 -s large -l 1000000' for better data"
echo "  3. gprof execution uses GPROF_RUN_ARGS when profiling_args is set"
echo ""
echo "You can now run:"
echo "  bash scripts/01_profile_app.sh XSBench"