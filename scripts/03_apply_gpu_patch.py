#!/usr/bin/env python3
"""
03_apply_gpu_patch.py — Universal GPU optimization patcher.

Architecture:
    This script handles the common workflow (backup, restore, copy, makefile generation).
    App-specific patch logic lives in separate modules under patches/ directory.

    patches/
    ├── __init__.py
    ├── miniGhost_patch.py    ← OpenMP → OpenACC for Fortran stencil code
    ├── udunits_patch.py      ← C library → CUDA kernel extraction
    ├── xsbench_patch.py      ← C OpenMP → CUDA
    └── lammps_patch.py       ← Kokkos GPU backend

Usage:
    python3 scripts/03_apply_gpu_patch.py miniGhost
    python3 scripts/03_apply_gpu_patch.py udunits
"""

import os
import re
import sys
import shutil
import importlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config_loader import load_app_config, get_app_paths
except ImportError:
    import json
    def load_app_config(app_name):
        with open(PROJECT_ROOT / "app_config.json") as f:
            data = json.load(f)
        return data[app_name]
    def get_app_paths(app_name):
        config = load_app_config(app_name)
        return {
            "project_root": PROJECT_ROOT,
            "gpu_src_dir": PROJECT_ROOT / "apps" / f"{app_name}_gpu" / config.get("src_subdir", "src"),
            "backup_dir": PROJECT_ROOT / "apps" / f"{app_name}_gpu" / "original_backup",
            "report_dir": PROJECT_ROOT / "reports" / app_name,
        }


# ============================================================
# Common patch utilities (available to all patch modules)
# ============================================================
def replace_omp_with_acc(content: str, mappings: dict) -> tuple:
    """
    Generic OMP → ACC replacement.
    mappings: dict of {omp_pattern: acc_replacement}
    Returns (new_content, change_count)
    """
    original = content
    for omp_pat, acc_rep in mappings.items():
        content = re.sub(omp_pat, acc_rep, content, flags=re.MULTILINE | re.IGNORECASE)
    changes = sum(1 for a, b in zip(original.splitlines(), content.splitlines()) if a != b)
    return content, changes


def patch_file_with_regex(filepath: str, replacements: list) -> int:
    """
    Apply a list of regex replacements to a file.
    replacements: list of (pattern, replacement, flags) tuples
    Returns total number of lines changed.
    """
    with open(filepath, "r") as f:
        content = f.read()

    original = content
    for pattern, replacement, flags in replacements:
        content = re.sub(pattern, replacement, content, flags=flags)

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        changes = sum(1 for a, b in zip(original.splitlines(), content.splitlines()) if a != b)
        return changes
    return 0


# ============================================================
# Load app-specific patch module
# ============================================================
def load_patch_module(app_name: str, config: dict):
    """
    Try to import the app-specific patch module.
    Looks in: scripts/patches/{app_name}_patch.py
    """
    patch_dir = SCRIPT_DIR / "patches"

    # Ensure patches directory exists
    patch_dir.mkdir(exist_ok=True)
    init_file = patch_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Auto-generated: patch modules for each application\n")

    module_name = f"{app_name}_patch"
    module_path = patch_dir / f"{module_name}.py"

    if not module_path.exists():
        print(f"  [WARN] No patch module found at {module_path}")
        print(f"  Creating a template patch module...")
        create_patch_template(module_path, app_name, config)
        print(f"  [INFO] Edit {module_path} to add app-specific GPU patches.")
        return None

    sys.path.insert(0, str(patch_dir))
    try:
        module = importlib.import_module(module_name)
        return module
    except Exception as e:
        print(f"  [ERROR] Failed to load patch module: {e}")
        return None


def create_patch_template(module_path: Path, app_name: str, config: dict):
    """Generate a template patch module for a new application."""
    language = config.get("language", "c")
    strategy = config.get("gpu_patch", {}).get("strategy", "cuda")

    template = f'''#!/usr/bin/env python3
"""
{app_name}_patch.py — GPU optimization patches for {app_name}.

Language: {language}
Strategy: {strategy}

Each patch function takes gpu_src_dir (str) and returns bool (success).
The apply_all_patches function is called by 03_apply_gpu_patch.py.
"""

import os
import re


def apply_all_patches(gpu_src_dir: str) -> dict:
    """
    Apply all GPU patches for {app_name}.

    Returns:
        dict with patch results, e.g.:
        {{"file1.c": {{"changes": 5, "status": "ok"}},
         "file2.c": {{"changes": 0, "status": "skipped"}}}}
    """
    results = {{}}

    # TODO: Add patch functions for each hotspot file
    # Example:
    # results["hotspot_file.c"] = patch_hotspot(gpu_src_dir)

    print(f"  [TODO] No patches implemented yet for {app_name}.")
    print(f"  Edit this file to add GPU optimization logic.")

    return results


# Example patch function template:
# def patch_hotspot(gpu_src_dir: str) -> dict:
#     fpath = os.path.join(gpu_src_dir, "hotspot_file.c")
#     if not os.path.exists(fpath):
#         return {{"changes": 0, "status": "not found"}}
#
#     with open(fpath, "r") as f:
#         content = f.read()
#
#     # Apply transformations...
#     # new_content = content.replace(...)
#
#     with open(fpath, "w") as f:
#         f.write(new_content)
#
#     return {{"changes": N, "status": "ok"}}
'''
    with open(module_path, "w") as f:
        f.write(template)


# ============================================================
# Generate GPU Makefile / CMakeLists
# ============================================================
def generate_gpu_build_file(gpu_src_dir: str, app_name: str, config: dict):
    """Generate the appropriate GPU build file based on build system."""
    build_system = config.get("build", {}).get("system", "make")
    gpu_config = config.get("gpu_patch", {})

    if build_system == "make":
        generate_gpu_makefile(gpu_src_dir, app_name, config)
    elif build_system == "cmake":
        generate_gpu_cmake(gpu_src_dir, app_name, config)

    print(f"  [OK] Generated GPU build file for {build_system}")


def generate_gpu_makefile(gpu_src_dir: str, app_name: str, config: dict):
    """Generate a Makefile for GPU compilation."""
    gpu_config = config.get("gpu_patch", {})
    language = config.get("language", "c")
    gpu_binary = gpu_config.get("gpu_binary_name", f"{app_name}_gpu")

    # Support both unified gpu_flags and separate fflags/cflags/ldflags
    gpu_flags = gpu_config.get("gpu_flags", "")
    gpu_fflags = gpu_config.get("gpu_fflags", f"-O2 {gpu_flags}" if gpu_flags else "-O2")
    gpu_cflags = gpu_config.get("gpu_cflags", f"-O2 {gpu_flags}" if gpu_flags else "-O2")
    gpu_ldflags = gpu_config.get("gpu_ldflags", gpu_flags)

    compiler = gpu_config.get("gpu_compiler", {})
    fc = compiler.get("FC", "nvfortran")
    cc = compiler.get("CC", "nvc")
    ld = compiler.get("LD", fc if language == "fortran" else cc)

    # Auto-detect MPI paths if the app requires MPI
    requires_mpi = config.get("run", {}).get("requires_mpi", False)
    mpi_compile_flags = ""
    mpi_link_flags = ""

    if requires_mpi:
        import subprocess
        # Use the configured FC compiler for MPI detection (it may be NVIDIA's mpifort)
        mpifort_cmd = fc  # fc is already set from gpu_compiler.FC
        try:
            mpi_compile_flags = subprocess.check_output(
                [mpifort_cmd, "--showme:compile"], stderr=subprocess.DEVNULL
            ).decode().strip()
            mpi_link_flags = subprocess.check_output(
                [mpifort_cmd, "--showme:link"], stderr=subprocess.DEVNULL
            ).decode().strip()
            print(f"  [INFO] MPI wrapper: {mpifort_cmd}")
            print(f"  [INFO] MPI compile flags: {mpi_compile_flags}")
            print(f"  [INFO] MPI link flags:    {mpi_link_flags}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If the configured compiler IS an MPI wrapper, it handles MPI internally
            # No extra flags needed in that case
            print(f"  [INFO] {mpifort_cmd} does not support --showme (likely handles MPI internally)")

    # Priority 1: Use explicit build_order from config (critical for Fortran module dependencies)
    build_order = gpu_config.get("build_order", [])
    objs_section = ""

    if build_order:
        objs_section = "OBJS = \\\n"
        for i, obj in enumerate(build_order):
            if i < len(build_order) - 1:
                objs_section += f"    {obj} \\\n"
            else:
                objs_section += f"    {obj}\n"
        print(f"  [INFO] Using explicit build_order from config ({len(build_order)} files)")

    # Priority 2: Try to read OBJS from original makefile
    if not objs_section.strip():
        orig_makefile = config.get("build", {}).get("makefile", "Makefile")
        orig_path = os.path.join(gpu_src_dir, orig_makefile)

        if os.path.exists(orig_path):
            with open(orig_path) as f:
                in_objs = False
                for line in f:
                    if line.strip().startswith("OBJS"):
                        in_objs = True
                    if in_objs:
                        objs_section += line
                        if not line.rstrip().endswith("\\"):
                            break
            if objs_section.strip():
                print(f"  [INFO] Read OBJS from {orig_makefile}")

    # Priority 3: Auto-discover source files (works for C/C++, risky for Fortran)
    if not objs_section.strip():
        src_extensions = config.get("gpu_patch", {}).get("source_extensions", [".F"])
        discovered = []
        for ext in src_extensions:
            ext_clean = ext.lstrip(".")
            import glob as _glob
            for fpath_found in sorted(_glob.glob(os.path.join(gpu_src_dir, f"*.{ext_clean}"))):
                obj_name = os.path.basename(fpath_found).rsplit(".", 1)[0] + ".o"
                if obj_name not in discovered:
                    discovered.append(obj_name)

        if discovered:
            objs_section = "OBJS = \\\n"
            for i, obj in enumerate(discovered):
                if i < len(discovered) - 1:
                    objs_section += f"    {obj} \\\n"
                else:
                    objs_section += f"    {obj}\n"
            print(f"  [INFO] Auto-discovered {len(discovered)} source files for OBJS")
            if language == "fortran":
                print(f"  [WARN] Auto-discovery may produce wrong order for Fortran modules!")
                print(f"         Consider adding 'build_order' to app_config.json")
        else:
            print(f"  [WARN] No source files found in {gpu_src_dir}")

    # Determine flags
    full_fflags = gpu_fflags
    full_cflags = gpu_cflags if gpu_cflags != "-O2" else ""
    full_ldflags = gpu_ldflags
    if mpi_compile_flags:
        full_fflags += f" {mpi_compile_flags}"
    if mpi_link_flags:
        full_ldflags += f" {mpi_link_flags}"

    # For mixed Fortran/C projects, CFLAGS needs the same preprocessor defines
    # Extract -D flags from fflags for CFLAGS
    cpp_defines = " ".join(f for f in gpu_fflags.split() if f.startswith("-D"))
    if not full_cflags:
        full_cflags = f"-O2 {cpp_defines}"
    elif cpp_defines:
        full_cflags += f" {cpp_defines}"

    if language == "fortran":
        suffix_rule = """.SUFFIXES: .F .o
.F.o:
\t$(FC) $(FFLAGS) -c $< -o $@"""
        flags_section = f"FFLAGS   = {full_fflags}"
    else:
        suffix_rule = """.SUFFIXES: .c .o
.c.o:
\t$(CC) $(CFLAGS) -c $< -o $@"""
        flags_section = f"CFLAGS   = {full_cflags}"

    makefile = f"""# makefile.gpu — {app_name} GPU build (auto-generated)
FC       = {fc}
CC       = {cc}
LD       = {ld}

{flags_section}
CFLAGS   = {full_cflags}
LDFLAGS  = {full_ldflags}

"""
    # Check if there's a make_targets include file (common in Fortran projects like miniGhost)
    make_targets_path = os.path.join(gpu_src_dir, "make_targets")
    if os.path.exists(make_targets_path) and language == "fortran":
        # make_targets already defines .F.o and .c.o rules, OBJS, dependencies, and targets
        # EXEC name stays as make_targets defines it (e.g. miniGhost.x)
        # The 04_compare script will find the actual binary produced
        makefile += f"""include make_targets
"""
        print(f"  [INFO] Using existing make_targets for OBJS and build rules")
    else:
        if objs_section:
            makefile += objs_section + "\n"

        makefile += f"""TARGET = {gpu_binary}

{suffix_rule}

$(TARGET): $(OBJS)
\t$(LD) $(LDFLAGS) -o $@ $(OBJS)

clean:
\trm -f *.o *.mod $(TARGET)
"""
    fpath = os.path.join(gpu_src_dir, "makefile.gpu")
    with open(fpath, "w") as f:
        f.write(makefile)


def generate_gpu_cmake(gpu_src_dir: str, app_name: str, config: dict):
    """Generate a CMakeLists.txt overlay for GPU compilation."""
    gpu_config = config.get("gpu_patch", {})
    gpu_flags = gpu_config.get("gpu_flags", "")

    cmake_content = f"""# CMake GPU overlay for {app_name} (auto-generated)
# Usage: cmake -B build_gpu -DGPU_ENABLED=ON && cmake --build build_gpu

option(GPU_ENABLED "Enable GPU acceleration" ON)

if(GPU_ENABLED)
    # Enable CUDA or OpenACC based on strategy
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} {gpu_flags}")
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} {gpu_flags}")
    message(STATUS "GPU acceleration enabled with flags: {gpu_flags}")
endif()
"""
    fpath = os.path.join(gpu_src_dir, "CMakeLists_gpu.cmake")
    with open(fpath, "w") as f:
        f.write(cmake_content)


# ============================================================
# Generate performance comparison script
# ============================================================
def create_compare_script(project_root: str, app_name: str, config: dict):
    """Generate the CPU vs GPU performance comparison script."""
    gpu_config = config.get("gpu_patch", {})
    gpu_binary = gpu_config.get("gpu_binary_name", f"{app_name}_gpu")
    run_config = config.get("run", {})
    requires_mpi = run_config.get("requires_mpi", False)
    default_np = run_config.get("default_np", 1)
    run_args = run_config.get("args", "")

    mpi_prefix_cpu = f'mpirun --oversubscribe -np $NP' if requires_mpi else ''
    mpi_prefix_gpu = f'mpirun --oversubscribe -np $NP' if requires_mpi else ''

    script = f'''#!/usr/bin/env bash
set -euo pipefail

# 04_compare_performance.sh — CPU vs GPU comparison for {app_name}
# Auto-generated by 03_apply_gpu_patch.py

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="{app_name}"
NP="${{NP:-{default_np}}}"; RUNS="${{RUNS:-3}}"
RUN_ARGS="{run_args}"
CPU_BIN="$PROJECT_ROOT/apps/${{APP}}_build/${{APP}}_normal"
GPU_SRC="$PROJECT_ROOT/apps/${{APP}}_gpu/{config.get('src_subdir', 'src')}"
REPORT="$PROJECT_ROOT/reports/$APP/performance_comparison"
mkdir -p "$REPORT"
log() {{ echo -e "\\n\\033[1;32m[$(date +%H:%M:%S)] $*\\033[0m"; }}

# Find CPU binary (with or without .x extension)
if [ ! -x "$CPU_BIN" ] && [ -x "${{CPU_BIN}}.x" ]; then
    CPU_BIN="${{CPU_BIN}}.x"
fi
[ -x "$CPU_BIN" ] || {{ echo "CPU baseline missing. Run 01_profile_app.sh first."; exit 1; }}

log "Building GPU version..."
cd "$GPU_SRC"
if [ -f makefile.gpu ]; then
    make -f makefile.gpu clean 2>/dev/null || true
    make -f makefile.gpu 2>&1 | tee "$REPORT/build_gpu.log"
    # Find GPU binary: try configured name, then original project binary name
    GPU_BIN=""
    for candidate in "$GPU_SRC/{gpu_binary}" "$GPU_SRC/miniGhost.x" "$GPU_SRC/XSBench"; do
        if [ -x "$candidate" ]; then
            GPU_BIN="$candidate"
            break
        fi
    done
    if [ -z "$GPU_BIN" ]; then
        # Last resort: find most recently modified executable
        GPU_BIN=$(find "$GPU_SRC" -maxdepth 1 -name "*.x" -o -type f -executable | head -1)
    fi
else
    echo "No GPU makefile found. Build manually."
    exit 1
fi
[ -n "$GPU_BIN" ] && [ -x "$GPU_BIN" ] || {{ echo "GPU build failed - no binary found"; exit 1; }}
echo "GPU binary: $GPU_BIN"

log "CPU Baseline ($RUNS runs)"
cpu=()
for i in $(seq 1 $RUNS); do
  t0=$(date +%s%N)
  {mpi_prefix_cpu} "$CPU_BIN" $RUN_ARGS >/dev/null 2>&1
  ms=$(( ($(date +%s%N) - t0) / 1000000 )); cpu+=($ms); echo "  Run $i: ${{ms}}ms"
done

log "GPU Version ($RUNS runs)"
gpu=()
for i in $(seq 1 $RUNS); do
  t0=$(date +%s%N)
  {mpi_prefix_gpu} "$GPU_BIN" $RUN_ARGS >/dev/null 2>&1
  ms=$(( ($(date +%s%N) - t0) / 1000000 )); gpu+=($ms); echo "  Run $i: ${{ms}}ms"
done

cs=0; for t in "${{cpu[@]}}"; do cs=$((cs+t)); done; ca=$((cs/RUNS))
gs=0; for t in "${{gpu[@]}}"; do gs=$((gs+t)); done; ga=$((gs/RUNS))
sp=$(awk "BEGIN{{printf \\"%.2f\\",$ca/($ga>0?$ga:1)}}")
TS=$(date +%Y%m%d_%H%M%S)

cat | tee "$REPORT/comparison_$TS.txt" <<EOF
========================================
$APP CPU vs GPU  $(date)
========================================
Args:  $RUN_ARGS | Ranks: $NP | Runs: $RUNS
CPU avg: ${{ca}}ms  GPU avg: ${{ga}}ms
Speedup: ${{sp}}x
========================================
EOF
'''
    fpath = os.path.join(project_root, "scripts", f"04_compare_{app_name}.sh")
    with open(fpath, "w") as f:
        f.write(script)
    os.chmod(fpath, 0o755)
    print(f"  [OK] Generated scripts/04_compare_{app_name}.sh")


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/03_apply_gpu_patch.py <app_name>")
        sys.exit(1)

    app_name = sys.argv[1]
    config = load_app_config(app_name)
    paths = get_app_paths(app_name)

    gpu_src_dir = str(paths["gpu_src_dir"])
    backup_dir = str(paths["backup_dir"])
    src_extensions = config.get("gpu_patch", {}).get("source_extensions", [".c", ".h"])

    print("=" * 60)
    print(f"  {app_name} — GPU Optimization ({config.get('gpu_patch', {}).get('strategy', 'N/A')})")
    print("=" * 60)

    if not os.path.isdir(gpu_src_dir):
        print(f"[ERROR] GPU source copy does not exist: {gpu_src_dir}")
        print(f"  Run 00_setup_local_env.sh to create it.")
        sys.exit(1)

    # Restore from backup (supports repeated runs)
    if os.path.isdir(backup_dir) and os.listdir(backup_dir):
        for f in os.listdir(backup_dir):
            if any(f.endswith(ext) for ext in src_extensions):
                shutil.copy2(os.path.join(backup_dir, f), os.path.join(gpu_src_dir, f))
        print(f"  Restored from backup (safe for re-run)\n")
    else:
        os.makedirs(backup_dir, exist_ok=True)
        for f in os.listdir(gpu_src_dir):
            if any(f.endswith(ext) for ext in src_extensions):
                shutil.copy2(os.path.join(gpu_src_dir, f), os.path.join(backup_dir, f))
        print(f"  First run: backed up to {backup_dir}\n")

    # Load and run app-specific patches
    print("[1/3] Applying app-specific patches...")
    patch_module = load_patch_module(app_name, config)
    if patch_module and hasattr(patch_module, "apply_all_patches"):
        results = patch_module.apply_all_patches(gpu_src_dir)
        for fname, result in results.items():
            status = result.get("status", "unknown")
            changes = result.get("changes", 0)
            print(f"  {fname}: {changes} changes ({status})")
    else:
        print(f"  [SKIP] No patches applied. Edit patches/{app_name}_patch.py to add them.")

    # Generate GPU build file
    print("\n[2/3] Generating GPU build file...")
    generate_gpu_build_file(gpu_src_dir, app_name, config)

    # Generate comparison script
    print("\n[3/3] Generating performance comparison script...")
    create_compare_script(str(paths["project_root"]), app_name, config)

    print("\n" + "=" * 60)
    print(f"  {app_name} GPU patching complete!")
    print("=" * 60)
    print(f"\n  Next steps:")
    print(f"    1. Review patches: diff {backup_dir}/ {gpu_src_dir}/")
    print(f"    2. Build GPU version: bash scripts/04_compare_{app_name}.sh")


if __name__ == "__main__":
    main()