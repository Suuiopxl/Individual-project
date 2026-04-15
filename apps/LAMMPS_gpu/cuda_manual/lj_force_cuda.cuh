/*
 * lj_force_cuda.cuh — CUDA kernel declarations for LAMMPS LJ pair force
 *
 * LLM-generated CUDA port of LAMMPS PairLJCut::compute() hotspot.
 * Profiling data:
 *   - PairLJCut::compute()   78.65% of total runtime
 *   - NPairBin::build()      16.85% (neighbor list — kept on CPU)
 *   - ev_tally()              1.12% (folded into kernel)
 *
 * Strategy:
 *   - One CUDA thread per atom i (processes all neighbors of i)
 *   - Newton's 3rd law DISABLED on GPU (no f[j] -= fpair race)
 *     → neighbor list is full (symmetric), each pair computed twice
 *     → eliminates atomicAdd overhead, better GPU scaling
 *   - AoS → SoA data layout for coalesced memory access
 *   - Neighbor list flattened into contiguous array with offset index
 *
 * Hardware target: NVIDIA RTX 4080 Laptop (sm_89, Ada Lovelace)
 */

#ifndef LJ_FORCE_CUDA_CUH
#define LJ_FORCE_CUDA_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ============================================================
// Error checking macro
// ============================================================
#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
              __FILE__, __LINE__, cudaGetErrorString(err));               \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

// ============================================================
// GPU data container for LJ force computation
// Holds all device-side arrays and manages H2D / D2H transfers.
// ============================================================
struct LJForceGPU {
    // Atom positions (SoA layout on GPU)
    double *d_x, *d_y, *d_z;

    // Atom forces (SoA layout on GPU)
    double *d_fx, *d_fy, *d_fz;

    // Atom types (1-indexed, matching LAMMPS convention)
    int *d_type;

    // Flattened neighbor list
    // neighbors[neigh_offset[ii] .. neigh_offset[ii] + numneigh[ii] - 1]
    // stores the neighbor indices for atom ilist[ii]
    int *d_ilist;       // [inum] — list of local atom indices
    int *d_numneigh;    // [inum] — number of neighbors per atom
    int *d_neighbors;   // [total_neighs] — flattened neighbor indices
    int *d_neigh_offset;// [inum] — start offset in d_neighbors for each atom

    // LJ parameters (ntypes+1 × ntypes+1, 1-indexed)
    // Flattened as 1D array: param[itype*(ntypes+1) + jtype]
    double *d_cutsq;    // cutoff distance squared
    double *d_lj1;      // 48 * epsilon * sigma^12
    double *d_lj2;      // 24 * epsilon * sigma^6
    double *d_lj3;      // 4 * epsilon * sigma^12  (for energy)
    double *d_lj4;      // 4 * epsilon * sigma^6   (for energy)
    double *d_offset;   // energy offset at cutoff

    // special_lj weighting factors (length 4)
    double *d_special_lj;

    // Energy reduction scratch
    double *d_pe_atom;    // [inum] — per-atom potential energy
    double *d_pe_result;  // [1] — total PE (reduced on GPU)

    // Dimensions
    int natoms;          // total atoms (nlocal + nghost)
    int nlocal;          // number of local (owned) atoms
    int inum;            // number of atoms in neighbor list
    int ntypes;          // number of atom types
    int total_neighs;    // sum of all numneigh[i]

    // Timing
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_h2d_start, ev_h2d_stop;
    cudaEvent_t ev_d2h_start, ev_d2h_stop;
    cudaEvent_t ev_kern_start, ev_kern_stop;
};

// ============================================================
// Lifecycle functions
// ============================================================

// Allocate GPU memory and create timing events
void lj_gpu_allocate(LJForceGPU &gpu,
                     int natoms, int nlocal, int inum,
                     int ntypes, int total_neighs);

// Upload atom data, neighbor list, and LJ parameters to GPU
void lj_gpu_upload(LJForceGPU &gpu,
                   // Atom data (AoS: x[i][3])
                   const double *x_aos, const int *type,
                   // Neighbor list
                   const int *ilist, const int *numneigh,
                   const int *neighbors_flat, const int *neigh_offset,
                   // LJ params (ntypes+1)^2 flattened
                   const double *cutsq, const double *lj1,
                   const double *lj2, const double *lj3,
                   const double *lj4, const double *offset,
                   const double *special_lj);

// Download forces from GPU (SoA → AoS conversion on host)
void lj_gpu_download_forces(LJForceGPU &gpu,
                            double *f_aos, double *pe_total);

// Free all GPU memory and destroy events
void lj_gpu_free(LJForceGPU &gpu);

// ============================================================
// Kernel launcher
// ============================================================

// Launch LJ force kernel
// eflag: compute energy if nonzero
void lj_gpu_compute(LJForceGPU &gpu, int eflag);

// ============================================================
// Timing query (milliseconds)
// ============================================================
void lj_gpu_get_timings(LJForceGPU &gpu,
                        float *h2d_ms, float *kernel_ms, float *d2h_ms);

#endif // LJ_FORCE_CUDA_CUH
