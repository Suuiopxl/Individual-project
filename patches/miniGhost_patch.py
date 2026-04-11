#!/usr/bin/env python3
"""
miniGhost_patch.py — OpenMP → OpenACC patches for miniGhost (Fortran).

Core idea:
    miniGhost source already has !$OMP parallel directives.
    OpenACC !$acc directives map closely to them:

        !$OMP PARALLEL        →  !$acc data present(GRID) create(WORK)
        !$OMP DO              →  !$acc parallel loop collapse(3)
        !$OMP ENDDO           →  !$acc end parallel loop
        !$OMP END PARALLEL    →  !$acc end data
"""

import os
import re


def apply_all_patches(gpu_src_dir: str) -> dict:
    """Apply all OpenMP → OpenACC patches for miniGhost."""
    results = {}

    results["MG_STENCIL_COMPS.F"] = patch_stencil(gpu_src_dir)
    results["MG_SUM_GRID.F"] = patch_sum_grid(gpu_src_dir)
    results["MG_FLUX_ACCUMULATE.F"] = patch_flux_accumulate(gpu_src_dir)

    return results


def patch_stencil(gpu_src_dir: str) -> dict:
    """MG_STENCIL_COMPS.F: Replace OpenMP with OpenACC."""
    fpath = os.path.join(gpu_src_dir, "MG_STENCIL_COMPS.F")
    if not os.path.exists(fpath):
        return {"changes": 0, "status": "not found"}

    with open(fpath, "r") as f:
        content = f.read()

    original = content

    content = re.sub(
        r'^(\s*)!\$OMP\s+PARALLEL\s*$',
        r'\1!$acc data present(GRID) create(WORK)',
        content, flags=re.MULTILINE | re.IGNORECASE
    )
    content = re.sub(
        r'^(\s*)!\$OMP\s+DO\s*$',
        r'\1!$acc parallel loop collapse(3)',
        content, flags=re.MULTILINE | re.IGNORECASE
    )
    content = re.sub(
        r'^(\s*)!\$OMP\s+ENDDO\s*$',
        r'\1!$acc end parallel loop',
        content, flags=re.MULTILINE | re.IGNORECASE
    )
    content = re.sub(
        r'^(\s*)!\$OMP\s+END\s+PARALLEL\s*$',
        r'\1!$acc end data',
        content, flags=re.MULTILINE | re.IGNORECASE
    )

    if content == original:
        return {"changes": 0, "status": "no OMP directives found"}

    with open(fpath, "w") as f:
        f.write(content)

    changes = sum(1 for a, b in zip(original.splitlines(), content.splitlines()) if a != b)
    return {"changes": changes, "status": "ok"}


def patch_sum_grid(gpu_src_dir: str) -> dict:
    """MG_SUM_GRID.F: Replace OpenMP with OpenACC, add reduction(+:LSUM) for sum loops."""
    fpath = os.path.join(gpu_src_dir, "MG_SUM_GRID.F")
    if not os.path.exists(fpath):
        return {"changes": 0, "status": "not found"}

    with open(fpath, "r") as f:
        lines = f.readlines()

    new_lines = []
    changes = 0

    for i, line in enumerate(lines):
        stripped = line.strip().upper()

        if stripped.startswith('!$OMP'):
            indent = line[:len(line) - len(line.lstrip())]

            if re.match(r'!\$OMP\s+PARALLEL\s*$', stripped, re.IGNORECASE):
                new_lines.append(f'{indent}! (OMP PARALLEL removed, using OpenACC)\n')
                changes += 1
                continue

            if re.match(r'!\$OMP\s+DO\b', stripped) and 'END' not in stripped:
                next_chunk = ''.join(lines[i+1:i+10]).upper()
                if 'LSUM' in next_chunk:
                    new_lines.append(f'{indent}!$acc parallel loop collapse(3) reduction(+:LSUM)\n')
                else:
                    new_lines.append(f'{indent}!$acc parallel loop collapse(3)\n')
                changes += 1
                continue

            if re.match(r'!\$OMP\s+ENDDO', stripped):
                new_lines.append(f'{indent}!$acc end parallel loop\n')
                changes += 1
                continue

            if re.match(r'!\$OMP\s+END\s+PARALLEL', stripped):
                new_lines.append(f'{indent}! (OMP END PARALLEL removed)\n')
                changes += 1
                continue

        new_lines.append(line)

    # Fallback: if no OMP directives found, try inserting ACC directly
    if changes == 0:
        new_lines = []
        looking = False
        do_depth = 0
        for line in lines:
            upper = line.strip().upper()
            if re.search(r'LSUM\s*=\s*0', upper):
                looking = True
            if looking and re.match(r'\s+DO\s+\w+\s*=', upper):
                do_depth += 1
                if do_depth == 1:
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(f'{indent}!$acc parallel loop collapse(3) reduction(+:LSUM) present(GRID)\n')
                    changes += 1
            new_lines.append(line)
            if looking and 'END' in upper and 'DO' in upper:
                do_depth -= 1
                if do_depth == 0:
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(f'{indent}!$acc end parallel loop\n')
                    looking = False
                    changes += 1

    with open(fpath, "w") as f:
        f.writelines(new_lines)

    return {"changes": changes, "status": "ok" if changes > 0 else "no changes"}


def patch_flux_accumulate(gpu_src_dir: str) -> dict:
    """MG_FLUX_ACCUMULATE.F: Same OMP → ACC replacement."""
    fpath = os.path.join(gpu_src_dir, "MG_FLUX_ACCUMULATE.F")
    if not os.path.exists(fpath):
        return {"changes": 0, "status": "not found"}

    with open(fpath, "r") as f:
        content = f.read()

    original = content

    content = re.sub(
        r'^(\s*)!\$OMP\s+PARALLEL\s*$',
        r'\1!$acc parallel',
        content, flags=re.MULTILINE | re.IGNORECASE
    )
    content = re.sub(
        r'^(\s*)!\$OMP\s+DO\b(.*?)$',
        r'\1!$acc parallel loop\2',
        content, flags=re.MULTILINE | re.IGNORECASE
    )
    content = re.sub(
        r'^(\s*)!\$OMP\s+ENDDO\s*$',
        r'\1!$acc end parallel loop',
        content, flags=re.MULTILINE | re.IGNORECASE
    )
    content = re.sub(
        r'^(\s*)!\$OMP\s+END\s+PARALLEL\s*$',
        r'\1!$acc end parallel',
        content, flags=re.MULTILINE | re.IGNORECASE
    )

    if content != original:
        with open(fpath, "w") as f:
            f.write(content)
        changes = sum(1 for a, b in zip(original.splitlines(), content.splitlines()) if a != b)
        return {"changes": changes, "status": "ok"}
    else:
        return {"changes": 0, "status": "no OMP directives found"}