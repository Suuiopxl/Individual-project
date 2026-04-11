#!/usr/bin/env python3
"""
miniGhost_patch.py — OpenMP → OpenACC patches for miniGhost (Fortran).

Strategy: Conservative approach using !$acc kernels directive.
Instead of trying to precisely map each OMP directive to an ACC equivalent,
we use the simpler !$acc kernels region which lets the compiler auto-parallelize
the loops inside. This avoids mismatched directive pairs.
"""

import os
import re


def apply_all_patches(gpu_src_dir: str) -> dict:
    """Apply all OpenACC patches for miniGhost."""
    results = {}
    results["MG_STENCIL_COMPS.F"] = patch_omp_to_acc(gpu_src_dir, "MG_STENCIL_COMPS.F")
    results["MG_SUM_GRID.F"] = patch_omp_to_acc(gpu_src_dir, "MG_SUM_GRID.F")
    return results


def patch_omp_to_acc(gpu_src_dir: str, filename: str) -> dict:
    """
    Generic OMP → ACC kernels replacement for any Fortran file.

    Replaces:
        !$OMP PARALLEL ...  →  !$acc kernels
        !$OMP DO ...        →  (commented out, kernels handles it)
        !$OMP ENDDO         →  (commented out)
        !$OMP END PARALLEL  →  !$acc end kernels
        Any other !$OMP     →  (commented out)
    """
    fpath = os.path.join(gpu_src_dir, filename)
    if not os.path.exists(fpath):
        return {"changes": 0, "status": "not found"}

    with open(fpath, "r") as f:
        lines = f.readlines()

    new_lines = []
    changes = 0

    for line in lines:
        stripped = line.strip().upper()

        # Skip non-OMP lines
        if not stripped.startswith('!$OMP'):
            new_lines.append(line)
            continue

        indent = line[:len(line) - len(line.lstrip())]

        # !$OMP PARALLEL ... → !$acc kernels
        if re.match(r'!\$OMP\s+PARALLEL\b', stripped):
            new_lines.append(f'{indent}!$acc kernels\n')
            changes += 1
            continue

        # !$OMP END PARALLEL → !$acc end kernels (check this BEFORE the DO patterns)
        if re.match(r'!\$OMP\s+END\s+PARALLEL', stripped):
            new_lines.append(f'{indent}!$acc end kernels\n')
            changes += 1
            continue

        # !$OMP DO ... → comment out (kernels directive handles loops)
        if re.match(r'!\$OMP\s+DO\b', stripped):
            new_lines.append(f'{indent}!  {line.strip()}\n')
            changes += 1
            continue

        # !$OMP ENDDO / !$OMP END DO → comment out
        if re.match(r'!\$OMP\s+(ENDDO|END\s*DO)', stripped):
            new_lines.append(f'{indent}!  {line.strip()}\n')
            changes += 1
            continue

        # Any other OMP directive → comment out
        new_lines.append(f'{indent}!  {line.strip()}\n')
        changes += 1

    with open(fpath, "w") as f:
        f.writelines(new_lines)

    return {"changes": changes, "status": "ok" if changes > 0 else "no OMP directives"}