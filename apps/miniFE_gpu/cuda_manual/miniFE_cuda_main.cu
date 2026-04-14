/*
 * miniFE_cuda_main.cu — Standalone CUDA CG benchmark for miniFE
 *
 * This follows the exact same setup sequence as driver.hpp:
 *   1. Create mesh → generate_matrix_structure(mesh, A) → assemble_FE_data
 *   2. impose_dirichlet → make_local_matrix
 *   3. Run CPU CG baseline
 *   4. Run CUDA CG
 *   5. Compare
 *
 * Build: see Makefile
 * Run:   mpirun -np 1 ./miniFE_cuda_bench -nx 128 -ny 128 -nz 128
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>
#include <limits>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

// miniFE headers — exact same includes as main.cpp / driver.hpp
#include <Box.hpp>
#include <BoxPartition.hpp>
#include <box_utils.hpp>
#include <Parameters.hpp>
#include <utils.hpp>
#include <YAML_Doc.hpp>

#if MINIFE_INFO != 0
#include <miniFE_info.hpp>
#else
#include <miniFE_no_info.hpp>
#endif

#ifndef MINIFE_VERSION
#define MINIFE_VERSION "2.0.0-cuda"
#endif

// Matrix & vector types
#ifdef MINIFE_CSR_MATRIX
#include <CSRMatrix.hpp>
#elif defined(MINIFE_ELL_MATRIX)
#include <ELLMatrix.hpp>
#else
#include <CSRMatrix.hpp>
#endif

#include <Vector.hpp>
#include <Vector_functions.hpp>
#include <SparseMatrix_functions.hpp>
#include <simple_mesh_description.hpp>
#include <generate_matrix_structure.hpp>
#include <assemble_FE_data.hpp>
#include <make_local_matrix.hpp>
#include <exchange_externals.hpp>
#include <compute_matrix_stats.hpp>
#include <mytimer.hpp>
#include <outstream.hpp>

// CUDA solver
#include "miniFE_cuda_kernels.cuh"

// Default types (same as Makefile)
#ifndef MINIFE_SCALAR
#define MINIFE_SCALAR double
#endif
#ifndef MINIFE_LOCAL_ORDINAL
#define MINIFE_LOCAL_ORDINAL int
#endif
#ifndef MINIFE_GLOBAL_ORDINAL
#define MINIFE_GLOBAL_ORDINAL int
#endif

// Timing enums (match cg_solve.hpp)
enum { CG_WAXPY=0, CG_DOT=1, CG_MATVEC=2, CG_MATVECDOT=3, CG_TOTAL=4, NUM_CG_TIMERS=5 };

// ============================================================
static double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

// ============================================================
// CPU CG baseline — same algorithm as cg_solve.hpp
// ============================================================
template<typename MatrixType, typename VectorType>
int run_cpu_cg(MatrixType& A, const VectorType& b, VectorType& x,
               int max_iter, double tolerance,
               double& normr, double* cg_times)
{
    typedef typename MatrixType::ScalarType Scalar;
    typedef typename miniFE::TypeTraits<Scalar>::magnitude_type magnitude;
    typedef typename MatrixType::LocalOrdinalType LocalOrdinal;

    size_t nrows = A.rows.size();
    LocalOrdinal ncols = A.num_cols;

    VectorType r(b.startIndex, nrows);
    VectorType p(0, ncols);
    VectorType Ap(b.startIndex, nrows);

    Scalar one = 1.0, zero = 0.0;
    magnitude rtrans = 0, oldrtrans = 0;

    int print_freq = max_iter / 10;
    if (print_freq > 50) print_freq = 50;
    if (print_freq < 1)  print_freq = 1;

    double tWAXPY = 0, tDOT = 0, tMATVEC = 0;
    double total = wall_time();
    double t0;
    int num_iters = 0;

    miniFE::matvec_std<MatrixType, VectorType> mv;

    // p = x
    t0 = wall_time();
    miniFE::waxpby(one, x, zero, x, p);
    tWAXPY += wall_time() - t0;

    // Ap = A * p
    t0 = wall_time();
    mv(A, p, Ap);
    tMATVEC += wall_time() - t0;

    // r = b - Ap
    t0 = wall_time();
    miniFE::waxpby(one, b, -one, Ap, r);
    tWAXPY += wall_time() - t0;

    // rtrans = dot(r, r)
    t0 = wall_time();
    rtrans = miniFE::dot_r2(r);
    tDOT += wall_time() - t0;

    normr = std::sqrt(rtrans);
    printf("  [CPU CG] Initial Residual = %.6e\n", normr);

    magnitude brkdown_tol = std::numeric_limits<magnitude>::epsilon();

    for (int k = 1; k <= max_iter && normr > tolerance; ++k) {
        if (k == 1) {
            t0 = wall_time();
            miniFE::waxpby(one, r, zero, r, p);
            tWAXPY += wall_time() - t0;
        } else {
            oldrtrans = rtrans;
            t0 = wall_time();
            rtrans = miniFE::dot_r2(r);
            tDOT += wall_time() - t0;
            magnitude beta = rtrans / oldrtrans;
            t0 = wall_time();
            miniFE::daxpby(one, r, beta, p);
            tWAXPY += wall_time() - t0;
        }

        normr = std::sqrt(rtrans);
        if (k % print_freq == 0 || k == max_iter) {
            printf("  [CPU CG] Iteration = %d   Residual = %.6e\n", k, normr);
        }

        t0 = wall_time();
        mv(A, p, Ap);
        tMATVEC += wall_time() - t0;

        t0 = wall_time();
        magnitude p_ap_dot = miniFE::dot(Ap, p);
        tDOT += wall_time() - t0;

        if (p_ap_dot < brkdown_tol) {
            if (p_ap_dot < 0) {
                std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!" << std::endl;
                break;
            }
            else brkdown_tol = 0.1 * p_ap_dot;
        }

        magnitude alpha = rtrans / p_ap_dot;

        t0 = wall_time();
        miniFE::daxpby(alpha, p, one, x);
        miniFE::daxpby(-alpha, Ap, one, r);
        tWAXPY += wall_time() - t0;

        num_iters = k;
    }

    total = wall_time() - total;
    cg_times[CG_WAXPY]  = tWAXPY;
    cg_times[CG_DOT]    = tDOT;
    cg_times[CG_MATVEC] = tMATVEC;
    cg_times[CG_TOTAL]  = total;

    printf("\n  [CPU CG] Converged in %d iterations\n", num_iters);
    printf("  [CPU CG] Final Residual = %.6e\n", normr);
    printf("  [CPU CG] Timings: MATVEC=%.4fs  DOT=%.4fs  WAXPBY=%.4fs  TOTAL=%.4fs\n",
           tMATVEC, tDOT, tWAXPY, total);

    return num_iters;
}

// ============================================================
// Main — follows driver.hpp setup sequence exactly
// ============================================================
int main(int argc, char** argv)
{
    miniFE::Parameters params;
    miniFE::get_parameters(argc, argv, params);

    int numprocs = 1, myproc = 0;
    miniFE::initialize_mpi(argc, argv, numprocs, myproc);

    typedef MINIFE_SCALAR Scalar;
    typedef MINIFE_LOCAL_ORDINAL LocalOrdinal;
    typedef MINIFE_GLOBAL_ORDINAL GlobalOrdinal;

#ifdef MINIFE_CSR_MATRIX
    typedef miniFE::CSRMatrix<Scalar, LocalOrdinal, GlobalOrdinal> MatrixType;
#else
    typedef miniFE::CSRMatrix<Scalar, LocalOrdinal, GlobalOrdinal> MatrixType;
#endif
    typedef miniFE::Vector<Scalar, LocalOrdinal, GlobalOrdinal> VectorType;

    int global_nx = params.nx;
    int global_ny = params.ny;
    int global_nz = params.nz;

    printf("============================================================\n");
    printf("  miniFE CUDA CG Benchmark\n");
    printf("  Grid: %d x %d x %d (elements)\n", global_nx, global_ny, global_nz);
    printf("  Procs: %d\n", numprocs);
    printf("============================================================\n\n");

    // --- Exactly follows driver.hpp lines 130-237 ---

    Box global_box = { 0, global_nx, 0, global_ny, 0, global_nz };
    std::vector<Box> local_boxes(numprocs);
    box_partition(0, numprocs, 2, global_box, &local_boxes[0]);

    Box& my_box = local_boxes[myproc];

    // Create mesh
    printf("[Setup] Creating mesh...\n");
    double t_setup = wall_time();

    miniFE::simple_mesh_description<GlobalOrdinal> mesh(global_box, my_box);

    // Generate matrix structure (2-arg version: mesh, A)
    printf("[Setup] Generating matrix structure...\n");
    MatrixType A;
    miniFE::generate_matrix_structure(mesh, A);

    GlobalOrdinal local_nrows = A.rows.size();
    GlobalOrdinal my_first_row = local_nrows > 0 ? A.rows[0] : -1;

    // Create vectors b and x
    VectorType b(my_first_row, local_nrows);
    VectorType x(my_first_row, local_nrows);

    // Assemble FE data
    printf("[Setup] Assembling FE data...\n");
    miniFE::assemble_FE_data(mesh, A, b, params);

    // Apply Dirichlet boundary conditions
    printf("[Setup] Imposing Dirichlet BC...\n");
    miniFE::impose_dirichlet(0.0, A, b, global_nx+1, global_ny+1, global_nz+1, mesh.bc_rows_0);
    miniFE::impose_dirichlet(1.0, A, b, global_nx+1, global_ny+1, global_nz+1, mesh.bc_rows_1);

    // Make local matrix (global → local index transform + MPI setup)
    printf("[Setup] Making matrix indices local...\n");
    miniFE::make_local_matrix(A);

    t_setup = wall_time() - t_setup;

    int nrows = A.rows.size();
    int ncols = A.num_cols;
    int nnz   = A.num_nonzeros();

    // Compute stats
    std::ostringstream oss;
    YAML_Doc ydoc("miniFE", MINIFE_VERSION, ".", oss.str());
    size_t global_nnz = miniFE::compute_matrix_stats(A, myproc, numprocs, ydoc);

    printf("[Setup] Done in %.3f s\n", t_setup);
    printf("[Setup] Local: nrows=%d  ncols=%d  nnz=%d\n", nrows, ncols, nnz);
    printf("[Setup] Global nnz=%zu\n", global_nnz);
    printf("\n");

    int max_iter = 200;
    double tolerance = std::numeric_limits<typename miniFE::TypeTraits<Scalar>::magnitude_type>::epsilon();

    // ============================================================
    // CPU baseline CG
    // ============================================================
    printf("=== CPU OpenMP CG Solve ===\n");

    // Save initial x for GPU run (x starts as zero, but just in case)
    std::vector<double> x_init(ncols);
    for (int i = 0; i < (int)x.coefs.size(); ++i) x_init[i] = x.coefs[i];

    double cpu_cg_times[NUM_CG_TIMERS] = {0};
    double cpu_normr = 0;
    int cpu_iters = run_cpu_cg(A, b, x, max_iter, tolerance, cpu_normr, cpu_cg_times);

    // Save CPU solution for comparison
    std::vector<double> x_cpu_solution(nrows);
    for (int i = 0; i < nrows; ++i) x_cpu_solution[i] = x.coefs[i];

    printf("\n");

    // ============================================================
    // CUDA CG
    // ============================================================
    printf("=== CUDA CG Solve ===\n");

    // Reset x to initial values
    for (int i = 0; i < (int)x.coefs.size(); ++i) x.coefs[i] = x_init[i];

    CudaCGData cg;
    cuda_cg_allocate(cg, nrows, ncols, nnz,
                     &A.row_offsets[0],
                     &A.packed_cols[0],
                     &A.packed_coefs[0]);

    cuda_cg_upload_vectors(cg, &x.coefs[0], ncols, &b.coefs[0], nrows);

    // GPU warmup
    cuda_matvec(cg, cg.d_x, cg.d_Ap);
    cudaDeviceSynchronize();

    double gpu_cg_times[4] = {0};
    double gpu_normr = 0;
    int gpu_iters = cuda_cg_solve(cg, max_iter, tolerance, gpu_normr, gpu_cg_times);

    // Download GPU solution
    std::vector<double> x_gpu_solution(ncols);
    cuda_cg_download_x(cg, &x_gpu_solution[0], ncols);

    cuda_cg_free(cg);
    printf("\n");

    // ============================================================
    // Compare
    // ============================================================

    printf("============================================================\n");
    printf("  Performance Comparison\n");
    printf("============================================================\n");
    printf("  %-20s %12s %12s %10s\n", "Metric", "CPU (OMP)", "GPU (CUDA)", "Speedup");
    printf("  %-20s %12s %12s %10s\n", "--------------------", "----------", "----------", "--------");
    printf("  %-20s %12d %12d\n", "Iterations", cpu_iters, gpu_iters);
    printf("  %-20s %12.6e %12.6e\n", "Final Residual", cpu_normr, gpu_normr);

    if (gpu_cg_times[2] > 1e-12)
        printf("  %-20s %12.4f %12.4f %9.2fx\n", "MATVEC (s)",
               cpu_cg_times[CG_MATVEC], gpu_cg_times[2],
               cpu_cg_times[CG_MATVEC] / gpu_cg_times[2]);
    if (gpu_cg_times[1] > 1e-12)
        printf("  %-20s %12.4f %12.4f %9.2fx\n", "DOT (s)",
               cpu_cg_times[CG_DOT], gpu_cg_times[1],
               cpu_cg_times[CG_DOT] / gpu_cg_times[1]);
    if (gpu_cg_times[0] > 1e-12)
        printf("  %-20s %12.4f %12.4f %9.2fx\n", "WAXPBY (s)",
               cpu_cg_times[CG_WAXPY], gpu_cg_times[0],
               cpu_cg_times[CG_WAXPY] / gpu_cg_times[0]);
    if (gpu_cg_times[3] > 1e-12)
        printf("  %-20s %12.4f %12.4f %9.2fx\n", "TOTAL CG (s)",
               cpu_cg_times[CG_TOTAL], gpu_cg_times[3],
               cpu_cg_times[CG_TOTAL] / gpu_cg_times[3]);
    printf("============================================================\n");

    // Verify
    double max_diff = 0;
    for (int i = 0; i < nrows; ++i) {
        double diff = std::fabs(x_cpu_solution[i] - x_gpu_solution[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max |x_cpu - x_gpu| = %.6e  %s\n",
           max_diff, max_diff < 1e-6 ? "[PASS]" : "[CHECK]");
    printf("============================================================\n");

    miniFE::finalize_mpi();
    return 0;
}
