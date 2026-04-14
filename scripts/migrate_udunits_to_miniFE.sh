#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# migrate_udunits_to_miniFE.sh
#
# Migrates the project:
#   - Replace udunits with miniFE in app_config.json
#   - Change LAMMPS GPU strategy from Kokkos to CUDA
#   - Update README.md and script headers
#   - Clone miniFE, clean up udunits directories
#   - Commit to git
#
# NOTE: No auto-patch modules are created for miniFE or LAMMPS.
#       CUDA porting for these apps follows the manual LLM workflow
#       (same as XSBench/cuda_manual).
#
# Usage:
#   cd ~/Individual-project
#   bash scripts/migrate_udunits_to_miniFE.sh
###############################################################################

log()  { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }
info() { echo -e "\033[1;34m[INFO] $*\033[0m"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

log "=========================================="
log " Migration: udunits -> miniFE"
log " LAMMPS GPU: Kokkos -> CUDA"
log " Repo root: $PROJECT_ROOT"
log "=========================================="

# ============================================================
# Step 1: Backup current config
# ============================================================
log "Step 1: Backing up current configuration"
BACKUP_SUFFIX=$(date +%Y%m%d_%H%M%S)
cp app_config.json "app_config.json.bak.${BACKUP_SUFFIX}"
info "Backup: app_config.json.bak.${BACKUP_SUFFIX}"

# ============================================================
# Step 2: Update app_config.json
# ============================================================
log "Step 2: Updating app_config.json"

python3 << 'PYEOF'
import json

with open("app_config.json", "r") as f:
    config = json.load(f)

# --- Remove udunits ---
if "udunits" in config:
    del config["udunits"]
    print("  Removed: udunits")

# --- Add miniFE ---
if "miniFE" not in config:
    config["miniFE"] = {
        "language": "cpp",
        "domain": "Implicit finite element (CG solver / SpMV)",
        "repo_url": "https://github.com/Mantevo/miniFE.git",
        "src_subdir": "openmp/src",
        "build": {
            "system": "make",
            "makefile": "Makefile",
            "compiler": { "CXX": "mpicxx", "CC": "mpicc" },
            "binary_name": "miniFE.x",
            "dependencies": [],
            "pre_build_fixes": [],
            "_note_profile": "miniFE openmp/src/Makefile uses standard CXX and CXXFLAGS."
        },
        "run": {
            "requires_mpi": True,
            "default_np": 1,
            "args": "-nx 128 -ny 128 -nz 128",
            "baseline_runs": 3,
            "_note": "Single MPI rank for profiling; increase nx/ny/nz for longer runtime"
        },
        "profiling": {
            "gprof_flags": "-pg",
            "top_n": 10,
            "hotspot_keywords": [
                "matvec",
                "waxpby",
                "dot",
                "cg_solve",
                "impose_dirichlet"
            ]
        },
        "gpu_patch": {
            "strategy": "cuda",
            "gpu_compiler": { "CXX": "nvcc", "CC": "nvcc" },
            "gpu_flags": "-O2 -arch=sm_89",
            "gpu_binary_name": "miniFE_gpu.x",
            "source_extensions": [".cpp", ".hpp", ".h"],
            "official_cuda_dir": "cuda",
            "_note": "CUDA port done manually via LLM. miniFE ships with official cuda/ dir for comparison."
        }
    }
    print("  Added: miniFE")
else:
    print("  miniFE already exists, skipping")

# --- Update LAMMPS GPU: Kokkos -> CUDA ---
if "LAMMPS" in config:
    gpu = config["LAMMPS"].get("gpu_patch", {})
    old_strategy = gpu.get("strategy", "unknown")
    gpu["strategy"] = "cuda"
    gpu.pop("patch_module", None)
    gpu["gpu_compiler"] = { "CXX": "nvcc", "CC": "nvcc" }
    gpu["gpu_flags"] = "-O2 -arch=sm_89"
    gpu["_note"] = "CUDA port done manually via LLM. Target: pair_lj_cut.cpp force loop."
    config["LAMMPS"]["gpu_patch"] = gpu
    print(f"  Updated: LAMMPS GPU strategy {old_strategy} -> cuda")

# --- Also clean up XSBench: remove patch_module if present ---
if "XSBench" in config:
    xsgpu = config["XSBench"].get("gpu_patch", {})
    xsgpu.pop("patch_module", None)
    config["XSBench"]["gpu_patch"] = xsgpu

with open("app_config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("  app_config.json written successfully")
PYEOF

# ============================================================
# Step 3: Update README.md
# ============================================================
log "Step 3: Updating README.md"

# If user has the new README.md alongside this script, use it
NEW_README="$SCRIPT_DIR/../README.md.new"
if [ -f "$NEW_README" ]; then
    cp "$NEW_README" "$PROJECT_ROOT/README.md"
    info "README.md replaced from README.md.new"
else
    # Fallback: sed-based in-place edits
    # Architecture tree
    sed -i 's|udunits_patch\.py.*$|miniGhost_patch.py       ← OpenMP → OpenACC (Fortran, automated)|' README.md
    sed -i '/xsbench_patch\.py/d' README.md
    sed -i '/lammps_patch\.py/d' README.md

    # Test Applications table
    sed -i '/| udunits/d' README.md
    if ! grep -q "| miniFE" README.md; then
        sed -i '/| XSBench.*Done/a | miniFE | C++/MPI | Implicit FE (CG/SpMV) | Make | CUDA (manual LLM) | 🔄 In progress |' README.md
    fi
    sed -i 's/| LAMMPS | C++ | Molecular dynamics | CMake | Kokkos |/| LAMMPS | C++\/MPI | Molecular dynamics | CMake | CUDA (manual LLM) |/' README.md

    # Example commands
    sed -i 's/00_setup_local_env.sh udunits/00_setup_local_env.sh miniFE/' README.md
    sed -i 's/01_profile_app.sh udunits/01_profile_app.sh miniFE/' README.md

    info "README.md updated via sed (consider replacing with the full new version)"
fi

# ============================================================
# Step 4: Update 03_apply_gpu_patch.py header
# ============================================================
log "Step 4: Updating 03_apply_gpu_patch.py header"

PATCH_SCRIPT="$PROJECT_ROOT/scripts/03_apply_gpu_patch.py"
if [ -f "$PATCH_SCRIPT" ]; then
    # Update the docstring patch listing
    sed -i '/udunits_patch\.py/d' "$PATCH_SCRIPT"
    sed -i '/lammps_patch\.py/d' "$PATCH_SCRIPT"
    sed -i 's|python3 scripts/03_apply_gpu_patch.py udunits|python3 scripts/03_apply_gpu_patch.py miniGhost|' "$PATCH_SCRIPT"
    info "03_apply_gpu_patch.py updated"
fi

# ============================================================
# Step 5: Clean up old patch files (udunits)
# ============================================================
log "Step 5: Cleaning up old patch files"

for PATCH_DIR in "$PROJECT_ROOT/patches" "$PROJECT_ROOT/scripts/patches"; do
    if [ -f "$PATCH_DIR/udunits_patch.py" ]; then
        rm "$PATCH_DIR/udunits_patch.py"
        info "Removed: $PATCH_DIR/udunits_patch.py"
    fi
done

# ============================================================
# Step 6: Clean up udunits app directories
# ============================================================
log "Step 6: Cleaning up old udunits directories"

for dir in apps/udunits apps/udunits_gpu reports/udunits; do
    if [ -d "$PROJECT_ROOT/$dir" ]; then
        echo -n "  Remove $dir? [y/N] "
        read -r confirm
        if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
            rm -rf "$PROJECT_ROOT/$dir"
            info "Removed: $dir"
        else
            info "Skipped: $dir"
        fi
    fi
done

rm -f "$PROJECT_ROOT/scripts/04_compare_udunits.sh"

# ============================================================
# Step 7: Clone miniFE
# ============================================================
log "Step 7: Setting up miniFE"

MINIFE_DIR="$PROJECT_ROOT/apps/miniFE"
if [ ! -d "$MINIFE_DIR" ]; then
    info "Cloning miniFE from GitHub..."
    git clone https://github.com/Mantevo/miniFE.git "$MINIFE_DIR" 2>&1 | tail -3
    rm -rf "$MINIFE_DIR/.git"
    info "miniFE cloned (.git removed)"
else
    info "apps/miniFE already exists, skipping clone"
fi

# Create GPU working copy
MINIFE_GPU="$PROJECT_ROOT/apps/miniFE_gpu"
if [ ! -d "$MINIFE_GPU" ]; then
    cp -r "$MINIFE_DIR" "$MINIFE_GPU"
    # Create cuda_manual directory (same pattern as XSBench)
    mkdir -p "$MINIFE_GPU/cuda_manual"
    info "GPU copy created: apps/miniFE_gpu (with cuda_manual/)"
else
    info "apps/miniFE_gpu already exists"
    mkdir -p "$MINIFE_GPU/cuda_manual"
fi

mkdir -p "$PROJECT_ROOT/reports/miniFE"
info "Report directory ready"

# ============================================================
# Step 8: Update .gitignore
# ============================================================
log "Step 8: Updating .gitignore"

GITIGNORE="$PROJECT_ROOT/.gitignore"
if [ -f "$GITIGNORE" ]; then
    if ! grep -q "miniFE_build" "$GITIGNORE"; then
        echo "" >> "$GITIGNORE"
        echo "# miniFE build artifacts" >> "$GITIGNORE"
        echo "apps/miniFE_build/" >> "$GITIGNORE"
        info ".gitignore: added miniFE_build"
    fi
fi

# ============================================================
# Step 9: Summary
# ============================================================
log "=========================================="
log " Migration Complete!"
log "=========================================="
echo ""
echo "  Changes:"
echo "    app_config.json  — udunits removed, miniFE added, LAMMPS -> CUDA"
echo "    README.md        — updated (replace with new version for best results)"
echo "    03_apply_gpu_patch.py — removed udunits/lammps patch references"
echo "    apps/miniFE      — cloned from Mantevo/miniFE"
echo "    apps/miniFE_gpu  — working copy with cuda_manual/ directory"
echo ""
echo "  Workflow for miniFE/LAMMPS CUDA port:"
echo "    1. Profile:   bash scripts/01_profile_app.sh miniFE"
echo "    2. Analyze:   python3 scripts/02_analyze_hotspots.py miniFE"
echo "    3. Feed hotspot code to LLM, get CUDA kernels"
echo "    4. Place CUDA code in apps/miniFE_gpu/cuda_manual/"
echo "    5. Build & benchmark against CPU baseline"
echo ""

# ============================================================
# Step 10: Git commit
# ============================================================
read -p "Commit these changes to git? [y/N] " git_confirm
if [[ "$git_confirm" == "y" || "$git_confirm" == "Y" ]]; then
    git add -A

    echo ""
    echo "--- Staged changes ---"
    git diff --cached --stat | tail -10
    echo ""

    git commit -m "refactor: replace udunits with miniFE, LAMMPS GPU -> CUDA

- Remove udunits (too lightweight for meaningful GPU acceleration)
- Add miniFE: implicit FE mini-app (CG solver, SpMV hotspot)
  * Baseline: openmp/src, GPU target: cuda_manual/ (LLM-driven)
  * Key hotspots: matvec (SpMV), waxpby, dot
- Change LAMMPS GPU strategy from Kokkos to CUDA
  (aligns with project goal: LLM-driven C/C++ -> CUDA transformation)
- CUDA porting uses manual LLM workflow (like XSBench), not auto-patch
- Update README, config, script headers; clean up udunits"

    log "Committed!"
    git log --oneline -5
else
    info "Skipped. Commit manually: git add -A && git commit"
fi