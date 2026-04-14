#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# fix_git_and_commit.sh — Diagnose & fix git tracking for apps/, then commit
#
# Run from the project root:
#   cd ~/Individual-project
#   bash fix_git_and_commit.sh
###############################################################################

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
NC='\033[0m'

log()  { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }
err()  { echo -e "${RED}[ERROR] $*${NC}" >&2; }
info() { echo -e "${CYAN}  → $*${NC}"; }

PROJECT_ROOT="$(pwd)"

# Verify we're in the project root
if [ ! -f "$PROJECT_ROOT/app_config.json" ]; then
    err "Please run this script from the project root (where app_config.json is)."
    err "Usage: cd ~/Individual-project && bash fix_git_and_commit.sh"
    exit 1
fi

log "=========================================="
log " Git Diagnostic & Fix Tool"
log " Project: $PROJECT_ROOT"
log "=========================================="

# ============================================================
# Step 1: Diagnose — why apps/ is not tracked
# ============================================================
log "Step 1: Diagnosing git tracking issues..."

echo ""
echo "--- Current .gitignore contents ---"
cat .gitignore 2>/dev/null || echo "(no .gitignore found)"
echo "---"

echo ""
echo "--- Checking for nested .git directories in apps/ ---"
NESTED_GITS=()
if [ -d "apps" ]; then
    while IFS= read -r gdir; do
        NESTED_GITS+=("$gdir")
        warn "Found nested .git: $gdir"
    done < <(find apps/ -name ".git" -type d 2>/dev/null)

    if [ ${#NESTED_GITS[@]} -eq 0 ]; then
        info "No nested .git directories found — good."
    else
        warn "Found ${#NESTED_GITS[@]} nested .git dir(s). These cause git to treat"
        warn "cloned repos as submodules and SKIP their contents!"
    fi
else
    err "No apps/ directory found!"
    exit 1
fi

echo ""
echo "--- Checking what git currently ignores in apps/ ---"
git status apps/ --short 2>/dev/null | head -20
echo ""
echo "--- Files git would track in apps/ (sample) ---"
git ls-files --others --exclude-standard apps/ 2>/dev/null | head -20
echo "(showing first 20 untracked files)"

# ============================================================
# Step 2: Fix — remove nested .git dirs so content is tracked
# ============================================================
log "Step 2: Fixing nested .git directories..."

if [ ${#NESTED_GITS[@]} -gt 0 ]; then
    for gdir in "${NESTED_GITS[@]}"; do
        app_dir="$(dirname "$gdir")"
        info "Removing $gdir (so git tracks $app_dir as regular files)"
        rm -rf "$gdir"
    done
    info "All nested .git directories removed."
else
    info "Nothing to fix — no nested .git dirs."
fi

# Also check for .gitmodules (in case any were accidentally registered)
if [ -f ".gitmodules" ]; then
    warn "Found .gitmodules file — checking for app entries..."
    cat .gitmodules
    info "If apps are listed here, you may want to remove those entries."
fi

# ============================================================
# Step 3: Update .gitignore — keep binaries/build out, allow source
# ============================================================
log "Step 3: Updating .gitignore..."

# Back up old gitignore
cp .gitignore .gitignore.bak 2>/dev/null || true

cat > .gitignore << 'GITIGNORE'
# ==================== API keys ====================
.env
API.txt

# ==================== Build artifacts ====================
*.o
*.mod
*.x
*_build/

# ==================== Profiling data ====================
gmon.out
perf.data*

# ==================== Large / binary files ====================
*.log
*.so
*.a
*.dylib

# ==================== OS files ====================
.DS_Store
Thumbs.db

# ==================== Python ====================
__pycache__/
*.pyc
*.pyo

# ==================== App-specific ignores ====================
# Ignore large build dirs inside cloned app sources
apps/*/build/
apps/*/.git/
apps/*/CMakeFiles/
apps/*/CMakeCache.txt
apps/*_gpu/original_backup/

# LAMMPS specific (very large repo)
apps/LAMMPS/doc/
apps/LAMMPS/examples/
apps/LAMMPS/bench/
apps/LAMMPS/potentials/
apps/LAMMPS/tools/
apps/LAMMPS/cmake/
apps/LAMMPS/unittest/
apps/LAMMPS_gpu/doc/
apps/LAMMPS_gpu/examples/
apps/LAMMPS_gpu/bench/
apps/LAMMPS_gpu/potentials/
apps/LAMMPS_gpu/tools/
apps/LAMMPS_gpu/cmake/
apps/LAMMPS_gpu/unittest/
GITIGNORE

info "Updated .gitignore (backed up old version to .gitignore.bak)"

# ============================================================
# Step 4: Check what will be committed
# ============================================================
log "Step 4: Preview what will be committed..."

echo ""
echo "--- git status summary ---"
git add -A --dry-run 2>/dev/null | head -40
echo "..."

# Count files
TOTAL_FILES=$(git add -A --dry-run 2>/dev/null | wc -l)
info "Total files to stage: $TOTAL_FILES"

# Warn if too many files (LAMMPS repo is huge)
if [ "$TOTAL_FILES" -gt 500 ]; then
    warn "That's a lot of files! If LAMMPS is included, consider adding it"
    warn "to .gitignore or using git submodules instead."
    echo ""
    echo "Large directories:"
    du -sh apps/*/ 2>/dev/null | sort -rh | head -10
fi

# ============================================================
# Step 5: Stage and Commit
# ============================================================
log "Step 5: Stage and Commit"

echo ""
read -p "Proceed with staging and committing? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    info "Aborted. You can review and commit manually:"
    info "  git add -A && git status"
    info "  git commit -m 'your message'"
    exit 0
fi

git add -A
echo ""
echo "--- Staged changes summary ---"
git diff --cached --stat | tail -5

# Commit
COMMIT_MSG="XSBench: setup profiling

- Remove nested .git dirs in cloned app sources (was preventing tracking)
- Update .gitignore: allow app source code, exclude build artifacts
- Current status:
  * miniGhost: complete (profiling + OpenACC patch + comparison done)
  * XSBench: config ready, deployment pending verification
  * udunits: in progress
  * LAMMPS: planned"

git commit -m "$COMMIT_MSG"

log "Commit created successfully!"
echo ""
git log --oneline -3
echo ""

# ============================================================
# Step 6: Push
# ============================================================
read -p "Push to origin? [y/N] " push_confirm
if [[ "$push_confirm" == "y" || "$push_confirm" == "Y" ]]; then
    git push origin "$(git branch --show-current)"
    log "Pushed to GitHub!"
else
    info "Skipped push. Run 'git push' when ready."
fi

# ============================================================
# Step 7: Verify XSBench deployment status
# ============================================================
log "Step 7: Verifying application deployment status..."

echo ""
echo "=== Application Deployment Check ==="
echo ""

for app in miniGhost XSBench udunits LAMMPS; do
    echo -n "  $app: "
    app_dir="apps/$app"
    gpu_dir="apps/${app}_gpu"

    if [ ! -d "$app_dir" ]; then
        echo -e "${RED}NOT CLONED${NC}"
        continue
    fi

    # Check if source files exist
    src_subdir=$(python3 -c "
import json
with open('app_config.json') as f:
    data = json.load(f)
print(data.get('$app', {}).get('src_subdir', 'src'))
" 2>/dev/null || echo "src")

    src_count=$(find "$app_dir/$src_subdir" -type f \( -name "*.c" -o -name "*.h" -o -name "*.F" -o -name "*.cpp" -o -name "*.f90" \) 2>/dev/null | wc -l)

    if [ "$src_count" -gt 0 ]; then
        echo -n -e "${GREEN}CLONED${NC} ($src_count source files)"
    else
        echo -n -e "${YELLOW}CLONED but no source files in $src_subdir/${NC}"
    fi

    # Check GPU copy
    if [ -d "$gpu_dir" ]; then
        echo -n " | GPU copy: YES"
    else
        echo -n " | GPU copy: NO"
    fi

    # Check build artifacts
    build_dir="apps/${app}_build"
    if [ -d "$build_dir" ] && [ "$(ls -A "$build_dir" 2>/dev/null)" ]; then
        echo -n " | Built: YES"
    else
        echo -n " | Built: NO"
    fi

    # Check reports
    report_dir="reports/$app"
    if [ -d "$report_dir" ] && [ "$(ls "$report_dir"/*.json 2>/dev/null)" ]; then
        echo " | Profiled: YES"
    else
        echo " | Profiled: NO"
    fi
done

echo ""
log "Done! Next steps:"
echo "  1. If XSBench shows CLONED, run:  bash scripts/01_profile_app.sh XSBench"
echo "  2. If NOT CLONED, run:            bash scripts/00_setup_local_env.sh XSBench"
echo ""