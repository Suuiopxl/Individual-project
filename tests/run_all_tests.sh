#!/usr/bin/env bash
#
# run_all_tests.sh --- End-to-end integration test across all four apps.
#
# Lives under tests/ (separate from scripts/, which holds pipeline stages).
# For every configured application this script:
#   1. Runs the profiling stage (scripts/01_profile_app.sh)
#   2. Runs the CPU-vs-GPU comparison stage (scripts/04_compare_<app>.sh),
#      which also performs numerical validation where one is defined.
#
# Stages 02 and 02b (the LLM agent) are intentionally skipped: they
# require API credentials, non-determinism makes them unsuitable for a
# repeatable integration test, and their outputs are already committed
# as artefacts under reports/<app>/.
#
# A short PASS/FAIL summary is printed at the end, and the full log of
# each per-app run is saved under reports/integration_<timestamp>/.
#
# Usage:
#   bash tests/run_all_tests.sh
#   APPS="miniGhost XSBench" bash tests/run_all_tests.sh   # subset
#   STOP_ON_FAIL=1 bash tests/run_all_tests.sh             # fail fast
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/reports/integration_${TS}"
mkdir -p "$LOG_DIR"

APPS="${APPS:-miniGhost XSBench miniFE LAMMPS}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

# ANSI colours
G='\033[1;32m' R='\033[1;31m' Y='\033[1;33m' N='\033[0m'
log()  { echo -e "\n${G}[$(date +%H:%M:%S)] $*${N}"; }
warn() { echo -e "${Y}[WARN] $*${N}"; }
err()  { echo -e "${R}[ERROR] $*${N}" >&2; }

# Maps each app to the correct compare script (naming isn't uniform).
compare_script_for() {
    case "$1" in
        miniGhost) echo "$SCRIPTS_DIR/04_compare_miniGhost.sh" ;;
        XSBench)   echo "$SCRIPTS_DIR/04_compare_xsbench.sh"   ;;
        miniFE)    echo "$SCRIPTS_DIR/04_compare_miniFE.sh"    ;;
        LAMMPS)    echo "$SCRIPTS_DIR/04_compare_lammps.sh"    ;;
        *)         echo "" ;;
    esac
}

# Run one stage. Returns 0 on success, non-zero otherwise; captures stdout
# and stderr together into a per-stage log file.
run_stage() {
    local app="$1" stage="$2" cmd="$3"
    local logfile="$LOG_DIR/${app}_${stage}.log"
    echo "    [${stage}] $cmd" >> "$LOG_DIR/summary.txt"
    if eval "$cmd" > "$logfile" 2>&1; then
        echo "    [${stage}] PASS (log: ${app}_${stage}.log)" | tee -a "$LOG_DIR/summary.txt"
        return 0
    else
        echo "    [${stage}] FAIL (see ${app}_${stage}.log)"  | tee -a "$LOG_DIR/summary.txt"
        return 1
    fi
}

declare -A APP_RESULT

log "Integration test starting"
log "Logs: $LOG_DIR"
log "Apps: $APPS"

for app in $APPS; do
    log "===================== $app ====================="
    echo "=== $app ===" >> "$LOG_DIR/summary.txt"

    # Stage: profile (01)
    if ! run_stage "$app" "profile" \
         "bash $SCRIPTS_DIR/01_profile_app.sh $app"; then
        APP_RESULT[$app]="FAIL (profile)"
        [ "$STOP_ON_FAIL" = "1" ] && break
        continue
    fi

    # Stage: compare + validate (04)
    CMP=$(compare_script_for "$app")
    if [ -z "$CMP" ] || [ ! -x "$CMP" ]; then
        warn "$app has no compare script; skipping"
        APP_RESULT[$app]="SKIP (no compare script)"
        continue
    fi
    if ! run_stage "$app" "compare" "bash $CMP"; then
        APP_RESULT[$app]="FAIL (compare)"
        [ "$STOP_ON_FAIL" = "1" ] && break
        continue
    fi

    APP_RESULT[$app]="PASS"
done

# ----------------------- Summary ----------------------------------------
log "===================== Summary ====================="
{
    echo ""
    echo "========================================================"
    echo "  Integration Test Summary"
    echo "  Date: $(date)"
    echo "========================================================"
    printf "  %-12s  %s\n" "App" "Result"
    printf "  %-12s  %s\n" "------------" "--------------------"
    for app in $APPS; do
        printf "  %-12s  %s\n" "$app" "${APP_RESULT[$app]:-NOT RUN}"
    done
    echo "========================================================"
} | tee -a "$LOG_DIR/summary.txt"

# Exit non-zero if anything failed, so the script fits in CI-style usage.
overall=0
for app in $APPS; do
    case "${APP_RESULT[$app]:-NOT RUN}" in
        PASS|"SKIP"*) ;;
        *) overall=1 ;;
    esac
done
exit $overall