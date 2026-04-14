# AI4Science — LLM-Driven GPU Optimization Pipeline

Automatically discover and optimize performance bottlenecks in HPC scientific applications using Large Language Models.

## Architecture

```
Individual-project/
├── app_config.json              ← Central config for ALL applications
├── config_loader.py             ← Shared Python config reader
│
├── scripts/                     ← Universal pipeline scripts
│   ├── patches/                   ← Auto-patch modules (miniGhost only)
│   │   └── miniGhost_patch.py       ← OpenMP → OpenACC (Fortran, automated)
│   ├── 00_setup_local_env.sh    ← Environment setup (reads config)
│   ├── 01_profile_app.sh        ← Build + Baseline + gprof (reads config)
│   ├── 02_analyze_hotspots.py   ← Hotspot extraction → LLM analysis
│   ├── 03_apply_gpu_patch.py    ← GPU patch framework (miniGhost)
│   └── 04_compare_*.sh          ← Per-app comparison scripts
│
├── apps/                        ← Application source code
│   ├── miniGhost/               ← Original source
│   ├── miniGhost_gpu/           ← OpenACC-patched copy
│   ├── XSBench/                 ← OpenMP baseline + official cuda/
│   ├── XSBench_gpu/             ← LLM-generated CUDA (cuda_manual/)
│   ├── miniFE/                  ← OpenMP baseline + official cuda/
│   ├── miniFE_gpu/              ← LLM-generated CUDA (cuda_manual/)
│   ├── LAMMPS/                  ← Source (local only, too large for GitHub)
│   └── LAMMPS_gpu/              ← LLM-generated CUDA
│
├── reports/                     ← Analysis results per application
│   ├── miniGhost/
│   ├── XSBench/
│   ├── miniFE/
│   └── LAMMPS/
│
└── .env                         ← API keys (not in Git)
```

## Workflow

Two GPU porting strategies are used in this project:

1. **Automated patch** (miniGhost): Script-driven OMP→OpenACC replacement via `03_apply_gpu_patch.py`. Demonstrates the feasibility and limitations of automated directive-based porting.

2. **Manual LLM-driven CUDA port** (XSBench, miniFE, LAMMPS): Feed hotspot source code to LLM, receive CUDA kernel implementations, manually integrate and benchmark. This approach handles the complexity of full CUDA rewrites (memory management, kernel launch, data transfer) that automated patching cannot.

## Quick Start

### 1. Setup

```bash
bash scripts/00_setup_local_env.sh            # Set up all apps
bash scripts/00_setup_local_env.sh miniFE      # Set up only miniFE
```

### 2. Profile

```bash
bash scripts/01_profile_app.sh miniFE
```

### 3. LLM Analysis

```bash
source .env
python3 scripts/02_analyze_hotspots.py miniFE
```

### 4a. Apply GPU Patch (miniGhost — automated)

```bash
python3 scripts/03_apply_gpu_patch.py miniGhost
```

### 4b. CUDA Port (XSBench/miniFE/LAMMPS — manual LLM)

Feed hotspot code to LLM → receive CUDA implementation → place in `apps/<app>_gpu/cuda_manual/` → build and benchmark.

### 5. Performance Comparison

```bash
bash scripts/04_compare_miniGhost.sh
bash scripts/04_compare_xsbench.sh
```

## Test Applications

| App | Language | Domain | Build | GPU Strategy | Status |
|-----|----------|--------|-------|-------------|--------|
| miniGhost | Fortran/MPI | 3D stencil heat diffusion | Make | OpenACC (auto-patch) | ✅ Done |
| XSBench | C | Nuclear cross-section lookup | Make | CUDA (manual LLM) | ✅ Done |
| miniFE | C++/MPI | Implicit FE (CG/SpMV) | Make | CUDA (manual LLM) | 🔄 In progress |
| LAMMPS | C++/MPI | Molecular dynamics | CMake | CUDA (manual LLM) | 📋 Planned |

## Environment Requirements

- **OS**: Ubuntu 22.04+ (native Linux or WSL2)
- **GPU**: NVIDIA GPU + driver
- **Compilers**: GCC, NVIDIA HPC SDK (nvfortran, nvc, nvcc)
- **MPI**: OpenMPI
- **Python**: 3.10+, openai / anthropic libraries
- **API**: OpenAI or Anthropic API key