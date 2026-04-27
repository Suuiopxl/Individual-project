# AI4Science — LLM-Driven GPU Optimisation Pipeline

A reproducible pipeline that uses a large language model to find performance bottlenecks in HPC scientific applications, decide whether and how to move them onto a GPU, and carry out the porting itself.

## What this is

Modern scientific codes can often be made much faster by running their hot loops on a GPU, but writing the GPU version is specialist work that not every scientific developer has time for. This project tests whether a code-aware LLM, sitting inside a profile → recommend → translate loop, can do most of that work end to end on real applications. Four mini-apps from different scientific domains were used as test subjects.

## Architecture

```
Individual-project/
├── app_config.json              ← Central config for ALL applications
├── config_loader.py             ← Shared Python config reader (legacy; see below)
│
├── scripts/                     ← Authoritative pipeline scripts
│   ├── patches/                   ← Auto-patch modules (miniGhost only)
│   │   └── miniGhost_patch.py       ← OpenMP → OpenACC (Fortran, automated)
│   ├── 00_setup_local_env.sh    ← Environment setup (reads config)
│   ├── 01_profile_app.sh        ← Build + baseline + profiling (reads config)
│   ├── 02_analyze_hotspots.py   ← Hotspot extraction + LLM Phase A / Phase B
│   ├── 03_apply_gpu_patch.py    ← GPU patch driver (miniGhost)
│   └── 04_compare_*.sh          ← Per-app comparison scripts (4 total)
│
├── apps/                        ← Application source code
│   ├── miniGhost/               ← Original source
│   ├── miniGhost_gpu/           ← OpenACC-patched copy
│   ├── XSBench/                 ← OpenMP baseline + official cuda/
│   ├── XSBench_gpu/             ← LLM-generated CUDA (cuda_manual/)
│   ├── miniFE/                  ← OpenMP baseline + official cuda/
│   ├── miniFE_gpu/              ← LLM-generated CUDA (cuda_manual/)
│   ├── LAMMPS/                  ← Upstream not in repo (too large); clone separately
│   └── LAMMPS_gpu/              ← LLM-generated standalone CUDA benchmark
│
├── reports/                     ← Per-application analysis output
│   ├── miniGhost/
│   ├── XSBench/
│   ├── miniFE/
│   ├── LAMMPS/
│   └── conversation/            ← LLM conversation transcripts per apps
│
└── .env                         ← API keys (not in Git)
```

## The agent: two phases

The agent in `02_analyze_hotspots.py` runs in two distinct phases with a human checkpoint between them.

1. **Phase A — Recommendation.** Given the profiling output for an application, the agent decides whether the hot region is a sensible GPU porting target, picks a framework (OpenACC or CUDA), and gives an expected speed-up envelope with reasoning.
2. **Phase B — Translation.** Conditioned on Phase A's recommendation and the hotspot source, the agent produces a compilable port (an OpenACC-directive patch for miniGhost, or a hand-written CUDA kernel and host wrapper for the others).
3. **Human checkpoint.** Between the two phases, the developer reviews the Phase A recommendation. Phase B is invoked separately so that a poor recommendation does not waste a translation round.

## Workflow

Two GPU porting strategies are used in this project:

1. **Automated directive patch** (miniGhost): script-driven OpenMP → OpenACC replacement via `03_apply_gpu_patch.py`, demonstrating where automated directive porting works and where it does not.
2. **LLM-driven CUDA port** (XSBench, miniFE, LAMMPS): hotspot source is fed to the agent in Phase B; the returned CUDA kernels are integrated into `apps/<app>_gpu/cuda_manual/` and benchmarked.

## Quick Start

### 1. Setup

```bash
bash scripts/00_setup_local_env.sh            # set up all apps
bash scripts/00_setup_local_env.sh miniFE     # set up only miniFE
```

### 2. Profile

```bash
bash scripts/01_profile_app.sh miniFE
```

`gprof` is used by default. The pipeline falls back to Callgrind when gprof attribution is unreliable (this happened on miniFE), and Nsight Systems is used for GPU-side timeline analysis.

### 3. LLM Analysis (Phase A + Phase B)
```bash
# Set your API key (or create a .env file with OPENAI_API_KEY=sk-xxx)
export OPENAI_API_KEY="sk-your-key-here"
# Or: export ANTHROPIC_API_KEY="sk-ant-your-key-here"
python3 scripts/02_analyze_hotspots.py miniFE
```

Outputs land under `reports/<app>/`.

### 4a. Apply GPU Patch (miniGhost only now — automated path)

```bash
python3 scripts/03_apply_gpu_patch.py miniGhost
```

### 4b. CUDA Port (XSBench / miniFE / LAMMPS — manual integration of LLM output)

Place the agent's CUDA implementation under `apps/<app>_gpu/cuda_manual/`, build it with the supplied Makefile, and run.

### 5. Performance Comparison

```bash
bash scripts/04_compare_miniGhost.sh
bash scripts/04_compare_xsbench.sh
bash scripts/04_compare_minife.sh
bash scripts/04_compare_lammps.sh
```

## Test Applications

| App | Language | Domain | Build | GPU Strategy | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| miniGhost | Fortran/MPI | 3D stencil heat diffusion | Make | OpenACC (auto-patch) | ✅ Done | |
| XSBench | C | Nuclear cross-section lookup | Make | CUDA (LLM-ported) | ✅ Done | Compared against official CUDA port |
| miniFE | C++/MPI | Implicit FE (CG / SpMV) | Make | CUDA (LLM-ported) | ✅ Done | Compared against official CUDA port |
| LAMMPS | C++/MPI | Molecular dynamics (LJ pair-force) | CMake | CUDA (LLM-ported) | ✅ Done | Upstream too large for the repo; LLM-ported standalone benchmark in `apps/LAMMPS_gpu/` |

## Conversation logs

Transcripts of the LLM interactions are stored under `reports/conversation/`. They are kept for reproducibility and to support the disclosure made in the final report on how the LLM was used.

## Environment

* **OS**: Ubuntu 22.04+ (native Linux or WSL2)
* **GPU**: NVIDIA GPU + driver
* **Compilers**: GCC, NVIDIA HPC SDK (`nvfortran`, `nvc`, `nvcc`), CUDA Toolkit
* **MPI**: OpenMPI
* **Python**: 3.10+, with `openai` and / or `anthropic` libraries installed
* **API key**: OpenAI or Anthropic (Claude Opus 4.6 was the primary backend used in this project)

## Legacy files

A few files at the repository root are early-stage prototypes that were superseded by the canonical scripts under `scripts/`. They are kept for history but should not be used. New work should ignore them.

* `analyze_hotspots.py` (root) — superseded by `scripts/02_analyze_hotspots.py`
* `config_loader.py` (root) — superseded by the version under `scripts/`
* `profile_miniGhost.sh` — superseded by `scripts/01_profile_app.sh`
* `fix_git_and_commit.sh` — one-off maintenance script, not part of the pipeline
* `results.yaml` — early aggregation experiment; current results live under `reports/<app>/`
* `app_config.json.bak.*` — timestamped backups of `app_config.json`
* `.gitignore.bak` — backup of `.gitignore`
