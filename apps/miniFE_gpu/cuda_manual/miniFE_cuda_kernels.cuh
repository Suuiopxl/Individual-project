/*
 * miniFE_cuda_kernels.cuh — CUDA kernel declarations for miniFE CG solver
 *
 * LLM-generated CUDA port of miniFE OpenMP hotspots:
 *   1. CSR SpMV  (matvec_std)  — 55% of runtime
 *   2. waxpby / daxpby         — 2.2% of runtime
 *   3. dot product             — 1.3% of runtime
 *
 * Types (from Makefile):
 *   MINIFE_SCALAR         = double
 *   MINIFE_LOCAL_ORDINAL  = int
 *   MINIFE_GLOBAL_ORDINAL = int
 */

#ifndef MINIFE_CUDA_KERNELS_CUH
#define MINIFE_CUDA_KERNELS_CUH

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
// Device memory manager for CSR matrix + vectors
// Holds all GPU-side pointers; handles H2D / D2H transfers.
// ============================================================
struct CudaCGData {
    // CSR matrix on device
    int*    d_row_offsets;   // [nrows+1]
    int*    d_packed_cols;   // [nnz]
    double* d_packed_coefs;  // [nnz]

    // Vectors on device
    double* d_x;    // solution vector       [ncols] (ncols >= nrows for external entries)
    double* d_b;    // RHS vector            [nrows]
    double* d_r;    // residual              [nrows]
    double* d_p;    // search direction      [ncols]
    double* d_Ap;   // matrix-vector product [nrows]

    // Reduction scratch space
    double* d_dot_result;    // [1] — partial result for dot products
    double* d_temp_reduce;   // [nblocks] — block-level partial sums

    // Dimensions
    int nrows;
    int ncols;
    int nnz;
    int reduce_blocks;  // number of blocks used for reduction

    // Host-side staging for scalar results
    double h_dot_result;
};

// ============================================================
// Lifecycle: allocate, upload, download, free
// ============================================================
void cuda_cg_allocate(CudaCGData& cg,
                      int nrows, int ncols, int nnz,
                      const int* row_offsets,
                      const int* packed_cols,
                      const double* packed_coefs);

void cuda_cg_upload_vectors(CudaCGData& cg,
                            const double* x, int x_len,
                            const double* b, int b_len);

void cuda_cg_download_x(CudaCGData& cg, double* x_host, int len);

void cuda_cg_free(CudaCGData& cg);

// ============================================================
// Kernel wrappers (called from host)
// ============================================================

// y = A * x  (CSR SpMV)
void cuda_matvec(CudaCGData& cg,
                 const double* d_x_in,
                 double* d_y_out);

// w = alpha * x + beta * y
void cuda_waxpby(int n,
                 double alpha, const double* d_x,
                 double beta,  const double* d_y,
                 double* d_w);

// y = alpha * x + beta * y  (in-place variant used by daxpby)
void cuda_daxpby(int n,
                 double alpha, const double* d_x,
                 double beta,  double* d_y);

// result = dot(x, y)
double cuda_dot(CudaCGData& cg, int n,
                const double* d_x, const double* d_y);

// result = dot(x, x)  — optimized self-dot
double cuda_dot_r2(CudaCGData& cg, int n,
                   const double* d_x);

// ============================================================
// Full CG solve on GPU
// Returns: number of iterations performed
// ============================================================
int cuda_cg_solve(CudaCGData& cg,
                  int max_iter,
                  double tolerance,
                  double& normr,
                  double* cg_times);  // [4]: WAXPY, DOT, MATVEC, TOTAL

#endif // MINIFE_CUDA_KERNELS_CUH
