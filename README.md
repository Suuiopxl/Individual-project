# AI4Science — LLM-Driven GPU Optimization Pipeline

Automatically discover and optimize performance bottlenecks in HPC scientific applications using Large Language Models.

## Architecture

```
Individual-project/
├── app_config.json              ← Central config for ALL applications
├── config_loader.py             ← Shared Python config reader
│
├── scripts/                     ← Universal pipeline scripts
│   │ patches/                     ← App-specific GPU patch modules
│   │   ├── miniGhost_patch.py       ← OpenMP → OpenACC (Fortran)
│   │   ├── udunits_patch.py         ← C library → CUDA kernels
│   │   ├── xsbench_patch.py         ← C OpenMP → CUDA
│   │   ├── lammps_patch.py          ← Kokkos GPU backend
│   ├── 00_setup_local_env.sh    ← Environment setup (reads config)
│   ├── 01_profile_app.sh        ← Build + Baseline + gprof (reads config)
│   ├── 02_analyze_hotspots.py   ← Hotspot extraction → LLM analysis
│   ├── 03_apply_gpu_patch.py    ← GPU patch framework (plugin-based)
│   └── 04_compare_*.sh          ← Auto-generated per-app comparison scripts
│
│
├── apps/                        ← Application source code
│   ├── miniGhost/               ← Original source (read-only)
│   ├── miniGhost_gpu/           ← GPU-patched copy
│   ├── miniGhost_build/         ← Build artifacts
│   ├── udunits/                 ← ...
│   └── ...
│
├── reports/                     ← Analysis results per application
│   ├── miniGhost/
│   │   ├── baseline_summary_*.txt
│   │   ├── profiling_summary_*.json
│   │   ├── llm_analysis_*.md
│   │   └── performance_comparison/
│   └── udunits/ ...
│
└── .env                         ← API keys (not in Git)
```

## Quick Start

### 1. Setup (all apps or specific)

```bash
bash scripts/00_setup_local_env.sh            # Set up all apps
bash scripts/00_setup_local_env.sh udunits     # Set up only udunits
```

### 2. Profile

```bash
bash scripts/01_profile_app.sh miniGhost
bash scripts/01_profile_app.sh udunits
```

### 3. LLM Analysis

```bash
source .env
python3 scripts/02_analyze_hotspots.py miniGhost
```

### 4. Apply GPU Patches

```bash
python3 scripts/03_apply_gpu_patch.py miniGhost
```

### 5. Performance Comparison

```bash
bash scripts/04_compare_miniGhost.sh
```

## Adding a New Application

1. Add an entry to `app_config.json` with build, run, profiling, and gpu_patch sections
2. Run `bash scripts/00_setup_local_env.sh <app_name>` to clone and set up
3. Run the profiling pipeline to discover hotspots
4. Create `patches/<app_name>_patch.py` with GPU optimization logic (a template is auto-generated)
5. Run comparison to evaluate speedup

## Test Applications

| App | Language | Domain | Build | GPU Strategy | Status |
|-----|----------|--------|-------|-------------|--------|
| miniGhost | Fortran/MPI | 3D stencil heat diffusion | Make | OpenACC | ✅ Done |
| udunits | C | Physical unit conversion | CMake | CUDA | 🔄 In progress |
| XSBench | C | Nuclear cross-section lookup | Make | CUDA | 📋 Planned |
| LAMMPS | C++ | Molecular dynamics | CMake | Kokkos | 📋 Planned |

## Environment Requirements

- **OS**: Ubuntu 22.04+ (native Linux or WSL2)
- **GPU**: NVIDIA GPU + driver
- **Compilers**: GCC, NVIDIA HPC SDK (nvfortran, nvc)
- **MPI**: OpenMPI
- **Python**: 3.10+, openai / anthropic libraries
- **API**: OpenAI or Anthropic API key