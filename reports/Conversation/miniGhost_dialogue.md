# miniGhost Phase B: OpenMP-to-OpenACC GPU Porting — Technical Dialogue

---

**User:** We have completed the Phase A analysis for miniGhost and the LLM agent has recommended OpenACC as the GPU porting strategy. The existing OpenMP directives in the source code already mark the parallel regions. How should we implement the GPU patch framework so that it is reusable across multiple applications?

**Assistant:** The patch framework should be application-agnostic. The driver script `03_apply_gpu_patch.py` handles the common workflow: backing up original source files, restoring from backup on re-runs, and generating the GPU build file. App-specific patch logic lives in separate plugin modules under a `patches/` directory. For miniGhost, that would be `patches/miniGhost_patch.py`, which implements an `apply_all_patches(gpu_src_dir)` function. The driver dynamically imports this module based on the `gpu_patch.patch_module` field in `app_config.json`. Adding a new directive port for another app would mean writing one more module under `patches/` and naming it in the configuration — nothing in the driver changes.

---

**User:** The GPU build is failing. nvfortran reports syntax errors like "Label field of continuation line is not blank" on `DRIVER.F`. What is going wrong?

**Assistant:** The problem is that nvfortran defaults to fixed-form Fortran for `.F` files, but miniGhost's source is written in free-form. The compiler flag `-Mfreeform` needs to be added to `FFLAGS` in the GPU build configuration. In `app_config.json`, the `gpu_fflags` field should be:

```
"-O2 -acc -gpu=mem:managed -Minfo=accel -Mfreeform"
```

This tells nvfortran to parse `.F` files as free-form source, matching the behaviour of gfortran's `-ffree-form` flag that the CPU baseline build uses.

---

**User:** After adding `-Mfreeform`, the build now fails with `gfortran: error: unrecognized command-line option '-acc'`. The system `mpifort` wrapper is calling gfortran, not nvfortran.

**Assistant:** The system-installed `mpifort` is a wrapper around gfortran, so passing NVIDIA-specific flags like `-acc` and `-Mfreeform` to it will be rejected. There are two approaches: use nvfortran directly and manually link MPI, or use the NVIDIA HPC SDK's own MPI wrapper which internally calls nvfortran.

Checking for NVIDIA's bundled MPI:

```bash
ls /opt/nvidia/hpc_sdk/Linux_x86_64/*/comm_libs/*/bin/mpifort 2>/dev/null
```

The NVIDIA HPC SDK at `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin/mpifort` provides an MPI wrapper that calls nvfortran natively. Set this as the Fortran compiler in `app_config.json`:

```json
"gpu_compiler": {
    "FC": "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin/mpifort",
    "CC": "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin/mpicc",
    "LD": "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin/mpicc"
}
```

Note that `LD` is set to `mpicc` (not `mpifort`), because miniGhost's `main()` is written in C. If the linker were `mpifort`, it would inject Fortran's `f90main.o` containing its own `main()` symbol, creating a multiple-definition conflict with the C `main()` in `main.c`.

---

**User:** The build still fails on `MG_CONSTANTS.F` with errors like "Non-constant expression where constant expression required" and "Illegal selector - KIND parameter has unknown value for data type". This happens even when using NVIDIA's own mpifort. What is going on?

**Assistant:** The source uses C preprocessor macros (`#if defined _MG_MPI`, `#if defined _MG_INT4`, etc.) to control data types and MPI features. When these macros are not defined at compile time, the preprocessor falls through to an `#else` branch that contains an intentional error message in plain English rather than valid Fortran — that is what nvfortran is trying to parse.

Examining the original `makefile.mpi.gnu` reveals the required defines:

```makefile
PROTOCOL = -D_MG_MPI
REAL_PRECISION = -D_MG_REAL8
INT_PRECISION = -D_MG_INT4
```

These must be passed to the compiler. The `gpu_fflags` field should include them:

```
"-O2 -acc -gpu=mem:managed -Minfo=accel -Mfreeform -D_MG_MPI -D_MG_INT4 -D_MG_REAL8"
```

Additionally, `CFLAGS` for the C file (`main.c`) also needs `-D_MG_MPI -D_MG_INT4 -D_MG_REAL8`, because `main.c` has its own `#ifdef` guards checking for the communication protocol.

---

**User:** Compilation order is wrong. nvfortran reports `Unable to open MODULE file mg_constants_mod.mod` when compiling `DRIVER.F`. The auto-discovered OBJS list is alphabetically sorted, so `DRIVER.F` is compiled before `MG_CONSTANTS.F`.

**Assistant:** Fortran modules require that the file defining a module be compiled before any file that `USE`s it, so that the `.mod` file exists when the dependent file is compiled. An alphabetical sort puts `DRIVER` before `MG_CONSTANTS`, which breaks the dependency chain.

miniGhost's repository includes a `make_targets` file that already defines the correct OBJS list and all inter-file dependencies. The most reliable approach is to have the generated `makefile.gpu` include this file directly:

```makefile
# makefile.gpu — miniGhost GPU build (auto-generated)
FC       = /opt/nvidia/hpc_sdk/.../mpifort
CC       = /opt/nvidia/hpc_sdk/.../mpicc
LD       = /opt/nvidia/hpc_sdk/.../mpicc

FFLAGS   = -O2 -acc -gpu=mem:managed -Minfo=accel -Mfreeform -D_MG_MPI -D_MG_INT4 -D_MG_REAL8 ...
CFLAGS   = -O2 -D_MG_MPI -D_MG_INT4 -D_MG_REAL8
LDFLAGS  = -acc -gpu=mem:managed -fortranlibs ...

include make_targets
```

The `make_targets` file provides the correct OBJS ordering, the `.F.o` and `.c.o` suffix rules, the dependency graph, and the link target. The generated `makefile.gpu` only overrides the compiler paths and flags.

---

**User:** Linking fails with `multiple definition of main`: one from `main.c` and one from nvfortran's `f90main.o`. The link line uses NVIDIA's mpifort as the linker.

**Assistant:** When `mpifort` is used as the linker, it automatically inserts the Fortran runtime entry point (`f90main.o`), which contains its own `main()` that calls `MAIN_`. This conflicts with the C `main()` in `main.c`.

The solution is to use `mpicc` as the linker instead, which does not insert the Fortran entry point. To ensure the Fortran runtime and OpenACC runtime are still linked in, add `-fortranlibs` to `LDFLAGS`:

```json
"LD": "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin/mpicc",
"gpu_ldflags": "-acc -gpu=mem:managed -fortranlibs"
```

This mirrors the original makefile's approach where `LD=$(CC)` (i.e. mpicc) is used for linking.

---

**User:** The build now succeeds. All 27 Fortran files and `main.c` compile and link successfully. However, when running the GPU binary under the 04 comparison script, the GPU version produces no output. The CPU baseline completes normally.

**Assistant:** The 04 comparison script was using the system `mpirun` (which wraps gfortran's MPI runtime) to launch the GPU binary (which was compiled against NVIDIA's MPI runtime). This runtime mismatch can cause silent hangs or segfaults.

The GPU binary must be launched with NVIDIA's own mpirun:

```bash
/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin/mpirun --oversubscribe -np 1 ./miniGhost.x ...
```

Testing with a single MPI rank confirms the GPU binary runs correctly and produces valid results:

```
Computation within error tolerance.
```

The 04 comparison script should use system `mpirun` for the CPU baseline and NVIDIA's `mpirun` for the GPU version.

---

**User:** The patched source with `!$acc kernels` compiles, but the compiler diagnostic `-Minfo=accel` confirms that loops are being parallelised on the GPU. Here is an excerpt:

```
mg_stencil_2d5pt:
     76, Loop is parallelizable
     77, Loop is parallelizable
     78, Loop is parallelizable
         Generating NVIDIA GPU code
         76, !$acc loop gang, vector(128) collapse(3)
```

And `NVCOMPILER_ACC_NOTIFY=3` shows actual kernel launches:

```
launch CUDA kernel  file=MG_STENCIL_COMPS.F function=mg_stencil_2d5pt line=78
  device=0 num_gangs=977 num_workers=1 vector_length=128 grid=977 block=128
```

But performance results at multiple problem sizes show the GPU is consistently slower than the CPU:

```
Size      CPU (ms)      GPU (ms)      Speedup
50^3      439           2268          0.19x
100^3     1586          4836          0.33x
300^3     39885         54185         0.74x
500^3     184098        243268        0.76x
```

The trend shows the gap narrowing at larger sizes but the GPU never overtakes. Before the project refactoring, I had achieved GPU speedups on this application at 256^3. What changed?

**Assistant:** Two factors are at play:

First, the original patch used `!$acc parallel loop collapse(3)` with explicit data directives (`!$acc data present(GRID) create(WORK)`), while the current version uses `!$acc kernels`. The `kernels` directive is more conservative — it asks the compiler to analyse the loop and decide how to parallelise it, whereas `parallel loop collapse(3)` directly instructs the compiler to map all three loop dimensions to GPU threads. `parallel loop` typically generates more efficient GPU code because the programmer has already verified the loop is safe to parallelise.

Second, the original patch wrapped the entire parallel region in a data directive that kept data resident on the GPU across multiple loop invocations within the same time step. With `!$acc kernels` alone, each kernel region triggers its own implicit data transfer, so data may be copied to the GPU at the start and back to the CPU at the end of every loop nest.

The fix is to go back to the precise `!$acc parallel loop collapse(3)` approach, but this time with correct directive pairing based on analysis of the actual OMP directives in the source.

---

**User:** What are the actual OMP directives in the original source files?

**Assistant:** Examining the backup of the unmodified source:

```bash
$ grep -n '!\$OMP' original_backup/MG_STENCIL_COMPS.F
73:!$OMP PARALLEL
74:!$OMP DO
87:!$OMP ENDDO
88:!$OMP DO
97:!$OMP ENDDO
98:!$OMP END PARALLEL
136:!$OMP PARALLEL
137:!$OMP DO
...
266:!$OMP PARALLEL 
267:!$OMP DO PRIVATE(SLICE_BACK, SLICE_MINE, SLICE_FRONT)
...

$ grep -n '!\$OMP' original_backup/MG_SUM_GRID.F
81:!$OMP PARALLEL DO REDUCTION(+: LSUM)
89:!$OMP END PARALLEL DO
```

Each stencil function has the same pattern: one `!$OMP PARALLEL` region containing two consecutive `!$OMP DO` loops (the stencil sweep and the copy-back). The fourth stencil variant (`3d27pt`) also carries a `PRIVATE(...)` clause. `MG_SUM_GRID.F` uses a combined `!$OMP PARALLEL DO REDUCTION` directive.

The precise substitution rules for the `parallel loop` patch are therefore:

| Original OMP directive | Replacement ACC directive | Rationale |
|---|---|---|
| `!$OMP PARALLEL` | (remove) | Each DO becomes its own `parallel loop` region |
| `!$OMP DO` | `!$acc parallel loop collapse(3)` | Direct parallelisation of the 3D loop nest |
| `!$OMP DO PRIVATE(...)` | `!$acc parallel loop collapse(3) private(...)` | Preserves thread-private variables |
| `!$OMP ENDDO` | `!$acc end parallel loop` | Closes the parallel loop region |
| `!$OMP END PARALLEL` | (remove) | Matches the removed `PARALLEL` |
| `!$OMP PARALLEL DO REDUCTION(+:LSUM)` | `!$acc parallel loop collapse(3) reduction(+:LSUM)` | Preserves the reduction semantics |
| `!$OMP END PARALLEL DO` | `!$acc end parallel loop` | Closes the reduction loop |

The earlier `!$acc kernels` approach was introduced as a safe fallback when the first `parallel loop` attempt produced an `Unmatched directive END - PARALLEL` error. That error occurred because the regex patch replaced `!$OMP PARALLEL` with `!$acc data present(GRID) create(WORK)` and `!$OMP END PARALLEL` with `!$acc end data`, but the fourth stencil function had a trailing space after `!$OMP PARALLEL ` (line 266) that the regex did not match, leaving one region without a closing directive.

The corrected patch file `patches/miniGhost_patch.py` handles each directive variant explicitly by checking line-by-line and matching the full set of patterns observed in the source.

---

**User:** Looking at the task manager during a GPU benchmark run, the GPU utilisation graph shows three distinct peaks separated by idle periods during what should be continuous GPU computation. Where do these idle periods come from?

**Assistant:** The three peaks correspond to the three benchmark runs (RUNS=3) in the 04 comparison script. Within each peak, the idle periods between GPU kernel launches are caused by the application's structure: miniGhost performs stencil computation on the GPU, then returns to the CPU for MPI halo exchange (boundary communication between ranks), then goes back to the GPU for the next time step. Even with `np 1` (no actual MPI communication), the MPI calls still execute on the CPU, and managed memory (`-gpu=mem:managed`) triggers page migration between host and device memory at each transition. On WSL2, this page migration happens at 4KB granularity through page faults rather than through GPU-native 2MB large pages, adding extra latency.

This is a fundamental characteristic of miniGhost's communication-heavy design: it is a halo-exchange benchmark, and the ratio of communication to computation is intentionally high. For a stencil code to benefit significantly from GPU acceleration, the data would need to remain resident on the GPU across multiple time steps, with explicit `!$acc data` regions spanning the outer time-step loop rather than being re-established at every kernel launch.
