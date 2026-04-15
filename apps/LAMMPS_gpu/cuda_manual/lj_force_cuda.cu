/*
 * lj_force_cuda.cu — CUDA kernels for LAMMPS LJ pair force computation
 *
 * LLM-generated CUDA port of PairLJCut::compute() from pair_lj_cut.cpp.
 *
 * Parallelization: one thread per atom i, serial loop over neighbors of i.
 * Newton's 3rd law is DISABLED — each pair is computed by both atoms.
 * This avoids write conflicts (atomicAdd) and is standard for GPU MD.
 *
 * Memory layout: SoA on GPU for coalesced access.
 * LJ parameters: flattened 2D arrays indexed as [itype * stride + jtype].
 */

#include "lj_force_cuda.cuh"
#include <cstring>

// ============================================================
// CUDA Kernel: LJ pair force (one thread per atom)
// ============================================================
__global__ void lj_force_kernel(
    // Atom positions (SoA)
    const double * __restrict__ x,
    const double * __restrict__ y,
    const double * __restrict__ z,
    // Atom forces output (SoA)
    double * __restrict__ fx,
    double * __restrict__ fy,
    double * __restrict__ fz,
    // Atom types
    const int * __restrict__ type,
    // Neighbor list (flattened)
    const int * __restrict__ ilist,
    const int * __restrict__ numneigh,
    const int * __restrict__ neighbors,
    const int * __restrict__ neigh_offset,
    // LJ parameters (stride = ntypes+1, 1-indexed)
    const double * __restrict__ cutsq,
    const double * __restrict__ lj1,
    const double * __restrict__ lj2,
    const double * __restrict__ lj3,
    const double * __restrict__ lj4,
    const double * __restrict__ offset_arr,
    // Special LJ weights
    const double * __restrict__ special_lj,
    // Per-atom energy output (NULL if eflag==0)
    double * __restrict__ pe_atom,
    // Dimensions
    int inum,
    int nlocal,
    int ntypes_p1,   // ntypes + 1 (array stride)
    int eflag)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    if (ii >= inum) return;

    int i = ilist[ii];
    double xtmp = x[i];
    double ytmp = y[i];
    double ztmp = z[i];
    int itype = type[i];

    double fix = 0.0;
    double fiy = 0.0;
    double fiz = 0.0;
    double pe_i = 0.0;

    int jnum   = numneigh[ii];
    int offset = neigh_offset[ii];

    for (int jj = 0; jj < jnum; jj++) {
        int j_raw = neighbors[offset + jj];

        // Extract special bond mask (top 2 bits in LAMMPS convention)
        // sbmask(j) = j >> SBBITS & 3, NEIGHMASK = 0x3FFFFFFF
        int sbindex = (j_raw >> 30) & 0x3;
        double factor_lj = special_lj[sbindex];
        int j = j_raw & 0x3FFFFFFF;  // NEIGHMASK

        double delx = xtmp - x[j];
        double dely = ytmp - y[j];
        double delz = ztmp - z[j];
        double rsq  = delx * delx + dely * dely + delz * delz;

        int jtype = type[j];
        int param_idx = itype * ntypes_p1 + jtype;

        if (rsq < cutsq[param_idx]) {
            double r2inv = 1.0 / rsq;
            double r6inv = r2inv * r2inv * r2inv;
            double forcelj = r6inv * (lj1[param_idx] * r6inv - lj2[param_idx]);
            double fpair = factor_lj * forcelj * r2inv;

            fix += delx * fpair;
            fiy += dely * fpair;
            fiz += delz * fpair;

            if (eflag) {
                double evdwl = r6inv * (lj3[param_idx] * r6inv - lj4[param_idx])
                             - offset_arr[param_idx];
                evdwl *= factor_lj;
                // Half energy to avoid double counting (full neighbor list)
                pe_i += 0.5 * evdwl;
            }
        }
    }

    fx[i] = fix;
    fy[i] = fiy;
    fz[i] = fiz;

    if (eflag && pe_atom != nullptr) {
        pe_atom[ii] = pe_i;
    }
}

// ============================================================
// Reduction kernel: sum per-atom PE to get total PE
// ============================================================
__global__ void reduce_pe_kernel(const double *pe_atom, double *pe_total, int n)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? pe_atom[i] : 0.0;
    __syncthreads();

    // Standard tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(pe_total, sdata[0]);
    }
}

// ============================================================
// Allocate GPU memory
// ============================================================
void lj_gpu_allocate(LJForceGPU &gpu,
                     int natoms, int nlocal, int inum,
                     int ntypes, int total_neighs)
{
    gpu.natoms       = natoms;
    gpu.nlocal       = nlocal;
    gpu.inum         = inum;
    gpu.ntypes       = ntypes;
    gpu.total_neighs = total_neighs;

    int np1sq = (ntypes + 1) * (ntypes + 1);

    // Atom data (SoA)
    CUDA_CHECK(cudaMalloc(&gpu.d_x,  natoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_y,  natoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_z,  natoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_fx, natoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_fy, natoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_fz, natoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_type, natoms * sizeof(int)));

    // Neighbor list
    CUDA_CHECK(cudaMalloc(&gpu.d_ilist,        inum * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_numneigh,     inum * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_neighbors,    total_neighs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_neigh_offset, inum * sizeof(int)));

    // LJ parameters
    CUDA_CHECK(cudaMalloc(&gpu.d_cutsq,  np1sq * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_lj1,    np1sq * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_lj2,    np1sq * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_lj3,    np1sq * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_lj4,    np1sq * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_offset, np1sq * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_special_lj, 4 * sizeof(double)));

    // Energy
    CUDA_CHECK(cudaMalloc(&gpu.d_pe_atom,   inum * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_pe_result, sizeof(double)));

    // Zero forces
    CUDA_CHECK(cudaMemset(gpu.d_fx, 0, natoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_fy, 0, natoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_fz, 0, natoms * sizeof(double)));

    // Create timing events
    CUDA_CHECK(cudaEventCreate(&gpu.ev_start));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_stop));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_h2d_stop));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_d2h_stop));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_kern_start));
    CUDA_CHECK(cudaEventCreate(&gpu.ev_kern_stop));
}

// ============================================================
// Upload data to GPU
// ============================================================
void lj_gpu_upload(LJForceGPU &gpu,
                   const double *x_aos, const int *type,
                   const int *ilist, const int *numneigh,
                   const int *neighbors_flat, const int *neigh_offset,
                   const double *cutsq, const double *lj1,
                   const double *lj2, const double *lj3,
                   const double *lj4, const double *offset,
                   const double *special_lj)
{
    int natoms = gpu.natoms;
    int inum   = gpu.inum;
    int np1sq  = (gpu.ntypes + 1) * (gpu.ntypes + 1);

    CUDA_CHECK(cudaEventRecord(gpu.ev_h2d_start));

    // Convert AoS positions → SoA on host, then upload
    double *h_x = (double *)malloc(natoms * sizeof(double));
    double *h_y = (double *)malloc(natoms * sizeof(double));
    double *h_z = (double *)malloc(natoms * sizeof(double));

    for (int i = 0; i < natoms; i++) {
        h_x[i] = x_aos[i * 3 + 0];
        h_y[i] = x_aos[i * 3 + 1];
        h_z[i] = x_aos[i * 3 + 2];
    }

    CUDA_CHECK(cudaMemcpy(gpu.d_x, h_x, natoms * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_y, h_y, natoms * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_z, h_z, natoms * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_type, type, natoms * sizeof(int), cudaMemcpyHostToDevice));

    // Neighbor list
    CUDA_CHECK(cudaMemcpy(gpu.d_ilist,        ilist,          inum * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_numneigh,     numneigh,       inum * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_neighbors,    neighbors_flat, gpu.total_neighs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_neigh_offset, neigh_offset,   inum * sizeof(int), cudaMemcpyHostToDevice));

    // LJ parameters
    CUDA_CHECK(cudaMemcpy(gpu.d_cutsq,  cutsq,  np1sq * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_lj1,    lj1,    np1sq * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_lj2,    lj2,    np1sq * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_lj3,    lj3,    np1sq * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_lj4,    lj4,    np1sq * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_offset, offset, np1sq * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_special_lj, special_lj, 4 * sizeof(double), cudaMemcpyHostToDevice));

    // Zero forces and energy
    CUDA_CHECK(cudaMemset(gpu.d_fx, 0, natoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_fy, 0, natoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_fz, 0, natoms * sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_pe_result, 0, sizeof(double)));

    CUDA_CHECK(cudaEventRecord(gpu.ev_h2d_stop));

    free(h_x);
    free(h_y);
    free(h_z);
}

// ============================================================
// Launch kernel
// ============================================================
void lj_gpu_compute(LJForceGPU &gpu, int eflag)
{
    int blockSize = 256;
    int gridSize  = (gpu.inum + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaEventRecord(gpu.ev_kern_start));

    lj_force_kernel<<<gridSize, blockSize>>>(
        gpu.d_x, gpu.d_y, gpu.d_z,
        gpu.d_fx, gpu.d_fy, gpu.d_fz,
        gpu.d_type,
        gpu.d_ilist, gpu.d_numneigh,
        gpu.d_neighbors, gpu.d_neigh_offset,
        gpu.d_cutsq, gpu.d_lj1, gpu.d_lj2,
        gpu.d_lj3, gpu.d_lj4, gpu.d_offset,
        gpu.d_special_lj,
        eflag ? gpu.d_pe_atom : nullptr,
        gpu.inum, gpu.nlocal,
        gpu.ntypes + 1,
        eflag);

    CUDA_CHECK(cudaGetLastError());

    // Reduce per-atom PE if needed
    if (eflag) {
        int rblock = 256;
        int rgrid  = (gpu.inum + rblock - 1) / rblock;
        reduce_pe_kernel<<<rgrid, rblock, rblock * sizeof(double)>>>(
            gpu.d_pe_atom, gpu.d_pe_result, gpu.inum);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(gpu.ev_kern_stop));
}

// ============================================================
// Download forces from GPU
// ============================================================
void lj_gpu_download_forces(LJForceGPU &gpu,
                            double *f_aos, double *pe_total)
{
    int natoms = gpu.natoms;

    CUDA_CHECK(cudaEventRecord(gpu.ev_d2h_start));

    // Download SoA forces
    double *h_fx = (double *)malloc(natoms * sizeof(double));
    double *h_fy = (double *)malloc(natoms * sizeof(double));
    double *h_fz = (double *)malloc(natoms * sizeof(double));

    CUDA_CHECK(cudaMemcpy(h_fx, gpu.d_fx, natoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fy, gpu.d_fy, natoms * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fz, gpu.d_fz, natoms * sizeof(double), cudaMemcpyDeviceToHost));

    // Convert SoA → AoS
    for (int i = 0; i < natoms; i++) {
        f_aos[i * 3 + 0] = h_fx[i];
        f_aos[i * 3 + 1] = h_fy[i];
        f_aos[i * 3 + 2] = h_fz[i];
    }

    // Download PE
    if (pe_total) {
        CUDA_CHECK(cudaMemcpy(pe_total, gpu.d_pe_result, sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventRecord(gpu.ev_d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu.ev_d2h_stop));

    free(h_fx);
    free(h_fy);
    free(h_fz);
}

// ============================================================
// Get timing breakdown
// ============================================================
void lj_gpu_get_timings(LJForceGPU &gpu,
                        float *h2d_ms, float *kernel_ms, float *d2h_ms)
{
    CUDA_CHECK(cudaEventSynchronize(gpu.ev_d2h_stop));
    CUDA_CHECK(cudaEventElapsedTime(h2d_ms,    gpu.ev_h2d_start,  gpu.ev_h2d_stop));
    CUDA_CHECK(cudaEventElapsedTime(kernel_ms, gpu.ev_kern_start, gpu.ev_kern_stop));
    CUDA_CHECK(cudaEventElapsedTime(d2h_ms,    gpu.ev_d2h_start,  gpu.ev_d2h_stop));
}

// ============================================================
// Free GPU memory
// ============================================================
void lj_gpu_free(LJForceGPU &gpu)
{
    cudaFree(gpu.d_x);  cudaFree(gpu.d_y);  cudaFree(gpu.d_z);
    cudaFree(gpu.d_fx); cudaFree(gpu.d_fy); cudaFree(gpu.d_fz);
    cudaFree(gpu.d_type);
    cudaFree(gpu.d_ilist);
    cudaFree(gpu.d_numneigh);
    cudaFree(gpu.d_neighbors);
    cudaFree(gpu.d_neigh_offset);
    cudaFree(gpu.d_cutsq);
    cudaFree(gpu.d_lj1);  cudaFree(gpu.d_lj2);
    cudaFree(gpu.d_lj3);  cudaFree(gpu.d_lj4);
    cudaFree(gpu.d_offset);
    cudaFree(gpu.d_special_lj);
    cudaFree(gpu.d_pe_atom);
    cudaFree(gpu.d_pe_result);

    cudaEventDestroy(gpu.ev_start);
    cudaEventDestroy(gpu.ev_stop);
    cudaEventDestroy(gpu.ev_h2d_start);
    cudaEventDestroy(gpu.ev_h2d_stop);
    cudaEventDestroy(gpu.ev_d2h_start);
    cudaEventDestroy(gpu.ev_d2h_stop);
    cudaEventDestroy(gpu.ev_kern_start);
    cudaEventDestroy(gpu.ev_kern_stop);
}
