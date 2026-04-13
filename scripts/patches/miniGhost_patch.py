#!/usr/bin/env python3
"""
miniGhost_patch.py — Precise OpenMP → OpenACC patches for miniGhost.

Based on analysis of the actual OMP directives in the source:

MG_STENCIL_COMPS.F pattern (4 stencil functions, same structure):
    !$OMP PARALLEL              → (remove)
    !$OMP DO [PRIVATE(...)]     → !$acc parallel loop collapse(3) [private(...)]
    ...triple nested DO...
    !$OMP ENDDO                 → !$acc end parallel loop
    !$OMP DO                    → !$acc parallel loop collapse(3)
    ...triple nested DO...
    !$OMP ENDDO                 → !$acc end parallel loop
    !$OMP END PARALLEL          → (remove)

MG_SUM_GRID.F:
    !$OMP PARALLEL DO REDUCTION(+: LSUM) → !$acc parallel loop collapse(3) reduction(+:LSUM)
    ...triple nested DO...
    !$OMP END PARALLEL DO                → !$acc end parallel loop
"""

import os
import re


def apply_all_patches(gpu_src_dir: str) -> dict:
    """Apply all OpenACC patches for miniGhost."""
    results = {}
    results["MG_STENCIL_COMPS.F"] = patch_stencil(gpu_src_dir)
    results["MG_SUM_GRID.F"] = patch_sum_grid(gpu_src_dir)
    return results


def patch_stencil(gpu_src_dir: str) -> dict:
    """
    MG_STENCIL_COMPS.F: Replace OMP directives with ACC parallel loop.
    Each stencil has two loop nests inside one PARALLEL region.
    We remove PARALLEL/END PARALLEL and convert each DO to parallel loop.
    """
    fpath = os.path.join(gpu_src_dir, "MG_STENCIL_COMPS.F")
    if not os.path.exists(fpath):
        return {"changes": 0, "status": "not found"}

    with open(fpath, "r") as f:
        lines = f.readlines()

    new_lines = []
    changes = 0

    for line in lines:
        stripped = line.strip().upper()

        if not stripped.startswith('!$OMP'):
            new_lines.append(line)
            continue

        indent = line[:len(line) - len(line.lstrip())]

        # !$OMP PARALLEL → remove (each DO becomes its own parallel loop)
        if stripped == '!$OMP PARALLEL' or re.match(r'!\$OMP\s+PARALLEL\s*$', stripped):
            new_lines.append(f'{indent}! (OMP PARALLEL removed — each loop is its own ACC region)\n')
            changes += 1
            continue

        # !$OMP END PARALLEL → remove
        if re.match(r'!\$OMP\s+END\s+PARALLEL\s*$', stripped):
            new_lines.append(f'{indent}! (OMP END PARALLEL removed)\n')
            changes += 1
            continue

        # !$OMP DO PRIVATE(...) → !$acc parallel loop collapse(3) private(...)
        if re.match(r'!\$OMP\s+DO\s+PRIVATE', stripped):
            # Extract the PRIVATE clause
            priv_match = re.search(r'PRIVATE\s*\(([^)]+)\)', line, re.IGNORECASE)
            if priv_match:
                priv_vars = priv_match.group(1)
                new_lines.append(f'{indent}!$acc parallel loop collapse(3) private({priv_vars})\n')
            else:
                new_lines.append(f'{indent}!$acc parallel loop collapse(3)\n')
            changes += 1
            continue

        # !$OMP DO → !$acc parallel loop collapse(3)
        if re.match(r'!\$OMP\s+DO\s*$', stripped):
            new_lines.append(f'{indent}!$acc parallel loop collapse(3)\n')
            changes += 1
            continue

        # !$OMP ENDDO → !$acc end parallel loop
        if re.match(r'!\$OMP\s+ENDDO\s*$', stripped):
            new_lines.append(f'{indent}!$acc end parallel loop\n')
            changes += 1
            continue

        # Anything else unexpected → comment out
        new_lines.append(f'{indent}! {line.strip()}\n')
        changes += 1

    with open(fpath, "w") as f:
        f.writelines(new_lines)

    return {"changes": changes, "status": "ok" if changes > 0 else "no OMP directives"}


def patch_sum_grid(gpu_src_dir: str) -> dict:
    """
    MG_SUM_GRID.F: Replace OMP PARALLEL DO REDUCTION with ACC parallel loop reduction.
    """
    fpath = os.path.join(gpu_src_dir, "MG_SUM_GRID.F")
    if not os.path.exists(fpath):
        return {"changes": 0, "status": "not found"}

    with open(fpath, "r") as f:
        lines = f.readlines()

    new_lines = []
    changes = 0

    for line in lines:
        stripped = line.strip().upper()

        if not stripped.startswith('!$OMP'):
            new_lines.append(line)
            continue

        indent = line[:len(line) - len(line.lstrip())]

        # !$OMP PARALLEL DO REDUCTION(+: LSUM) → !$acc parallel loop collapse(3) reduction(+:LSUM)
        if re.match(r'!\$OMP\s+PARALLEL\s+DO\s+REDUCTION', stripped):
            red_match = re.search(r'REDUCTION\s*\(\s*([^)]+)\)', line, re.IGNORECASE)
            if red_match:
                red_clause = red_match.group(1).replace(' ', '')
                new_lines.append(f'{indent}!$acc parallel loop collapse(3) reduction({red_clause})\n')
            else:
                new_lines.append(f'{indent}!$acc parallel loop collapse(3)\n')
            changes += 1
            continue

        # !$OMP END PARALLEL DO → !$acc end parallel loop
        if re.match(r'!\$OMP\s+END\s+PARALLEL\s+DO', stripped):
            new_lines.append(f'{indent}!$acc end parallel loop\n')
            changes += 1
            continue

        # Anything else → comment out
        new_lines.append(f'{indent}! {line.strip()}\n')
        changes += 1

    with open(fpath, "w") as f:
        f.writelines(new_lines)

    return {"changes": changes, "status": "ok" if changes > 0 else "no OMP directives"}