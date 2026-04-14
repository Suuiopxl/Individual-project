/*
 * miniFE_cuda_kernels.cu — CUDA implementation of miniFE CG solver kernels
 *
 * LLM-generated CUDA port targeting the three CG hotspots:
 *   1. matvec_std  (CSR SpMV)       — one thread per row
 *   2. waxpby / daxpby              — one thread per element
 *   3. dot (parallel reduction)     — block-level reduce + final reduce
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -c miniFE_cuda_kernels.cu -o miniFE_cuda_kernels.o
 */

#include "miniFE_cuda_kernels.cuh"
#include <cfloat>
#include <cmath>
#include <cstring>
#include <sys/time.h>

// ============================================================
// Timing helper (same as miniFE's mytimer)
// ============================================================
static double gpu_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

// ============================================================
// Tuning constants
// ============================================================
static const int BLOCK_SIZE     = 256;   // threads per block for SpMV and element-wise ops
static const int REDUCE_BLOCK   = 256;   // threads per block for reduction
static const int MAX_REDUCE_BLK = 1024;  // max blocks for first-level reduction

// ============================================================
//  Kernel 1: CSR SpMV — y = A * x
//  One thread per row, scalar accumulation.
//  This is the simplest correct CSR SpMV; for miniFE's ~27 nonzeros/row
//  it gives good performance without warp-level tricks.
// ============================================================
__global__
void kernel_csr_spmv(const int    nrows,
                     const int*   __restrict__ row_offsets,
                     const int*   __restrict__ cols,
                     const double* __restrict__ vals,
                     const double* __restrict__ x,
                     double*       __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int row_start = row_offsets[row];
    int row_end   = row_offsets[row + 1];

    double sum = 0.0;
    for (int j = row_start; j < row_end; ++j) {
        sum += vals[j] * x[cols[j]];
    }
    y[row] = sum;
}

// ============================================================
//  Kernel 2a: waxpby — w = alpha * x + beta * y
//  Handles all special cases (alpha=1, beta=0, etc.) on the GPU
//  for simplicity; branch divergence is negligible since all
//  threads take the same path.
// ============================================================
__global__
void kernel_waxpby(const int     n,
                   const double  alpha,
                   const double* __restrict__ x,
                   const double  beta,
                   const double* __restrict__ y,
                   double*       __restrict__ w)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (beta == 0.0) {
        w[i] = (alpha == 1.0) ? x[i] : alpha * x[i];
    } else {
        w[i] = (alpha == 1.0) ? x[i] + beta * y[i]
                               : alpha * x[i] + beta * y[i];
    }
}

// ============================================================
//  Kernel 2b: daxpby — y = alpha * x + beta * y  (in-place)
//  Maps to CG operations: x += alpha*p and r -= alpha*Ap
// ============================================================
__global__
void kernel_daxpby(const int     n,
                   const double  alpha,
                   const double* __restrict__ x,
                   const double  beta,
                   double*       __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (alpha == 1.0 && beta == 1.0) {
        y[i] += x[i];
    } else if (beta == 1.0) {
        y[i] += alpha * x[i];
    } else if (alpha == 1.0) {
        y[i] = x[i] + beta * y[i];
    } else if (beta == 0.0) {
        y[i] = alpha * x[i];
    } else {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

// ============================================================
//  Kernel 3: Dot product — two-phase reduction
//  Phase 1: each block reduces a chunk to one partial sum
//  Phase 2: a single block reduces all partial sums
// ============================================================
__global__
void kernel_dot_partial(const int     n,
                        const double* __restrict__ x,
                        const double* __restrict__ y,
                        double*       __restrict__ partial)
{
    __shared__ double sdata[REDUCE_BLOCK];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread accumulates over its grid-stride range
    double sum = 0.0;
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        sum += x[idx] * y[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block-level tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// Phase 2: reduce partial sums from Phase 1
__global__
void kernel_dot_final(const int     nblocks,
                      const double* __restrict__ partial,
                      double*       __restrict__ result)
{
    __shared__ double sdata[REDUCE_BLOCK];
    int tid = threadIdx.x;

    double sum = 0.0;
    for (int i = tid; i < nblocks; i += blockDim.x) {
        sum += partial[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = sdata[0];
    }
}

// Self-dot variant: result = sum(x[i] * x[i])
__global__
void kernel_dot_r2_partial(const int     n,
                           const double* __restrict__ x,
                           double*       __restrict__ partial)
{
    __shared__ double sdata[REDUCE_BLOCK];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        double xi = x[idx];
        sum += xi * xi;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// Host-side wrapper functions
// ============================================================

// --- Allocate device memory and upload CSR matrix ---
void cuda_cg_allocate(CudaCGData& cg,
                      int nrows, int ncols, int nnz,
                      const int* row_offsets,
                      const int* packed_cols,
                      const double* packed_coefs)
{
    cg.nrows = nrows;
    cg.ncols = ncols;
    cg.nnz   = nnz;

    // Number of blocks for reduction
    cg.reduce_blocks = (nrows + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
    if (cg.reduce_blocks > MAX_REDUCE_BLK) cg.reduce_blocks = MAX_REDUCE_BLK;

    // Allocate CSR matrix
    CUDA_CHECK(cudaMalloc(&cg.d_row_offsets, (nrows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cg.d_packed_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cg.d_packed_coefs, nnz * sizeof(double)));

    // Upload matrix (immutable during CG)
    CUDA_CHECK(cudaMemcpy(cg.d_row_offsets, row_offsets,
                          (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cg.d_packed_cols, packed_cols,
                          nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cg.d_packed_coefs, packed_coefs,
                          nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate vectors
    CUDA_CHECK(cudaMalloc(&cg.d_x,  ncols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cg.d_b,  nrows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cg.d_r,  nrows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cg.d_p,  ncols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cg.d_Ap, nrows * sizeof(double)));

    // Zero out vectors
    CUDA_CHECK(cudaMemset(cg.d_r,  0, nrows * sizeof(double)));
    CUDA_CHECK(cudaMemset(cg.d_p,  0, ncols * sizeof(double)));
    CUDA_CHECK(cudaMemset(cg.d_Ap, 0, nrows * sizeof(double)));

    // Reduction workspace
    CUDA_CHECK(cudaMalloc(&cg.d_dot_result, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cg.d_temp_reduce, cg.reduce_blocks * sizeof(double)));

    printf("  [CUDA] Allocated: nrows=%d, ncols=%d, nnz=%d, reduce_blocks=%d\n",
           nrows, ncols, nnz, cg.reduce_blocks);
}

// --- Upload x and b vectors to device ---
void cuda_cg_upload_vectors(CudaCGData& cg,
                            const double* x, int x_len,
                            const double* b, int b_len)
{
    CUDA_CHECK(cudaMemcpy(cg.d_x, x, x_len * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cg.d_b, b, b_len * sizeof(double), cudaMemcpyHostToDevice));
}

// --- Download solution vector x from device ---
void cuda_cg_download_x(CudaCGData& cg, double* x_host, int len)
{
    CUDA_CHECK(cudaMemcpy(x_host, cg.d_x, len * sizeof(double), cudaMemcpyDeviceToHost));
}

// --- Free all device memory ---
void cuda_cg_free(CudaCGData& cg)
{
    cudaFree(cg.d_row_offsets);
    cudaFree(cg.d_packed_cols);
    cudaFree(cg.d_packed_coefs);
    cudaFree(cg.d_x);
    cudaFree(cg.d_b);
    cudaFree(cg.d_r);
    cudaFree(cg.d_p);
    cudaFree(cg.d_Ap);
    cudaFree(cg.d_dot_result);
    cudaFree(cg.d_temp_reduce);
}

// ============================================================
// Kernel launch wrappers
// ============================================================

void cuda_matvec(CudaCGData& cg, const double* d_x_in, double* d_y_out)
{
    int grid = (cg.nrows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_csr_spmv<<<grid, BLOCK_SIZE>>>(
        cg.nrows, cg.d_row_offsets, cg.d_packed_cols, cg.d_packed_coefs,
        d_x_in, d_y_out);
}

void cuda_waxpby(int n,
                 double alpha, const double* d_x,
                 double beta,  const double* d_y,
                 double* d_w)
{
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_waxpby<<<grid, BLOCK_SIZE>>>(n, alpha, d_x, beta, d_y, d_w);
}

void cuda_daxpby(int n,
                 double alpha, const double* d_x,
                 double beta,  double* d_y)
{
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_daxpby<<<grid, BLOCK_SIZE>>>(n, alpha, d_x, beta, d_y);
}

double cuda_dot(CudaCGData& cg, int n,
                const double* d_x, const double* d_y)
{
    // Phase 1: block-level partial sums
    kernel_dot_partial<<<cg.reduce_blocks, REDUCE_BLOCK>>>(
        n, d_x, d_y, cg.d_temp_reduce);

    // Phase 2: reduce partials to single value
    kernel_dot_final<<<1, REDUCE_BLOCK>>>(
        cg.reduce_blocks, cg.d_temp_reduce, cg.d_dot_result);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(&cg.h_dot_result, cg.d_dot_result,
                          sizeof(double), cudaMemcpyDeviceToHost));
    return cg.h_dot_result;
}

double cuda_dot_r2(CudaCGData& cg, int n, const double* d_x)
{
    kernel_dot_r2_partial<<<cg.reduce_blocks, REDUCE_BLOCK>>>(
        n, d_x, cg.d_temp_reduce);

    kernel_dot_final<<<1, REDUCE_BLOCK>>>(
        cg.reduce_blocks, cg.d_temp_reduce, cg.d_dot_result);

    CUDA_CHECK(cudaMemcpy(&cg.h_dot_result, cg.d_dot_result,
                          sizeof(double), cudaMemcpyDeviceToHost));
    return cg.h_dot_result;
}

// ============================================================
//  Full CG solve on GPU
//  Mirrors cg_solve() from cg_solve.hpp but all computation
//  stays on the GPU. Only scalar dot results come back to host.
// ============================================================
int cuda_cg_solve(CudaCGData& cg,
                  int max_iter,
                  double tolerance,
                  double& normr,
                  double* cg_times)
{
    int nrows = cg.nrows;
    int ncols = cg.ncols;

    double tWAXPY = 0, tDOT = 0, tMATVEC = 0;
    double total_time = gpu_timer();
    double t0;

    double one  = 1.0;
    double zero = 0.0;
    double rtrans = 0, oldrtrans = 0;
    int num_iters = 0;

    int print_freq = max_iter / 10;
    if (print_freq > 50) print_freq = 50;
    if (print_freq < 1)  print_freq = 1;

    // p = x (copy x into p)
    t0 = gpu_timer();
    cuda_waxpby(ncols, one, cg.d_x, zero, cg.d_x, cg.d_p);
    cudaDeviceSynchronize();
    tWAXPY += gpu_timer() - t0;

    // Ap = A * p
    t0 = gpu_timer();
    cuda_matvec(cg, cg.d_p, cg.d_Ap);
    cudaDeviceSynchronize();
    tMATVEC += gpu_timer() - t0;

    // r = b - Ap
    t0 = gpu_timer();
    cuda_waxpby(nrows, one, cg.d_b, -one, cg.d_Ap, cg.d_r);
    cudaDeviceSynchronize();
    tWAXPY += gpu_timer() - t0;

    // rtrans = dot(r, r)
    t0 = gpu_timer();
    rtrans = cuda_dot_r2(cg, nrows, cg.d_r);
    tDOT += gpu_timer() - t0;

    normr = std::sqrt(rtrans);
    printf("  [CUDA CG] Initial Residual = %.6e\n", normr);

    // --- CG iteration loop ---
    for (int k = 1; k <= max_iter && normr > tolerance; ++k) {

        if (k == 1) {
            // p = r
            t0 = gpu_timer();
            cuda_waxpby(nrows, one, cg.d_r, zero, cg.d_r, cg.d_p);
            cudaDeviceSynchronize();
            tWAXPY += gpu_timer() - t0;
        } else {
            oldrtrans = rtrans;

            t0 = gpu_timer();
            rtrans = cuda_dot_r2(cg, nrows, cg.d_r);
            tDOT += gpu_timer() - t0;

            double beta = rtrans / oldrtrans;

            // p = r + beta * p
            t0 = gpu_timer();
            cuda_daxpby(nrows, one, cg.d_r, beta, cg.d_p);
            cudaDeviceSynchronize();
            tWAXPY += gpu_timer() - t0;
        }

        normr = std::sqrt(rtrans);
        if (k % print_freq == 0 || k == max_iter) {
            printf("  [CUDA CG] Iteration = %d   Residual = %.6e\n", k, normr);
        }

        // Ap = A * p
        t0 = gpu_timer();
        cuda_matvec(cg, cg.d_p, cg.d_Ap);
        cudaDeviceSynchronize();
        tMATVEC += gpu_timer() - t0;

        // p_ap_dot = dot(Ap, p)
        t0 = gpu_timer();
        double p_ap_dot = cuda_dot(cg, nrows, cg.d_Ap, cg.d_p);
        tDOT += gpu_timer() - t0;

        double alpha = rtrans / p_ap_dot;

        // x = x + alpha * p
        // r = r - alpha * Ap
        t0 = gpu_timer();
        cuda_daxpby(ncols, alpha, cg.d_p, one, cg.d_x);
        cuda_daxpby(nrows, -alpha, cg.d_Ap, one, cg.d_r);
        cudaDeviceSynchronize();
        tWAXPY += gpu_timer() - t0;

        num_iters = k;
    }

    double total = gpu_timer() - total_time;

    // Report timings
    cg_times[0] = tWAXPY;
    cg_times[1] = tDOT;
    cg_times[2] = tMATVEC;
    cg_times[3] = total;

    printf("\n  [CUDA CG] Converged in %d iterations\n", num_iters);
    printf("  [CUDA CG] Final Residual = %.6e\n", normr);
    printf("  [CUDA CG] Timings: MATVEC=%.4fs  DOT=%.4fs  WAXPBY=%.4fs  TOTAL=%.4fs\n",
           tMATVEC, tDOT, tWAXPY, total);

    return num_iters;
}
