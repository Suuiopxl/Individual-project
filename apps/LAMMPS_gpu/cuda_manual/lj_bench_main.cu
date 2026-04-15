/*
 * lj_bench_main.cu — Standalone benchmark for LAMMPS LJ CUDA force kernel
 *
 * This program:
 *   1. Generates an FCC lattice of LJ atoms (same as LAMMPS bench/in.lj)
 *   2. Builds a full neighbor list on CPU (no Newton's 3rd law)
 *   3. Computes forces on CPU (reference implementation)
 *   4. Computes forces on GPU (CUDA kernel)
 *   5. Validates correctness (max force error)
 *   6. Benchmarks both and reports speedup
 *
 * Usage:
 *   ./lj_cuda_bench [natoms_per_dim]
 *   Default: 20 (= 20^3 * 4 = 32000 atoms, matching bench/in.lj)
 *
 * Build:
 *   make          (uses Makefile in this directory)
 */

#include "lj_force_cuda.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

// ============================================================
// LJ Parameters (matching LAMMPS bench/in.lj)
// ============================================================
static const double LJ_EPSILON = 1.0;
static const double LJ_SIGMA   = 1.0;
static const double LJ_CUTOFF  = 2.5;
static const double FCC_DENSITY = 0.8442;  // reduced density
static const int    NSTEPS      = 100;      // benchmark iterations
static const int    WARMUP      = 5;        // GPU warmup iterations

// ============================================================
// Timer helper
// ============================================================
struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ============================================================
// Generate FCC lattice
// ============================================================
void generate_fcc_lattice(int nx, int ny, int nz,
                          double density,
                          std::vector<double> &x_aos,
                          double box[3])
{
    // Lattice constant for FCC at given density
    // 4 atoms per unit cell, density = 4 / a^3
    double a = pow(4.0 / density, 1.0 / 3.0);

    box[0] = nx * a;
    box[1] = ny * a;
    box[2] = nz * a;

    // FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    double basis[4][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5},
        {0.0, 0.5, 0.5}
    };

    int natoms = 4 * nx * ny * nz;
    x_aos.resize(natoms * 3);

    int idx = 0;
    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                for (int b = 0; b < 4; b++) {
                    x_aos[idx * 3 + 0] = (ix + basis[b][0]) * a;
                    x_aos[idx * 3 + 1] = (iy + basis[b][1]) * a;
                    x_aos[idx * 3 + 2] = (iz + basis[b][2]) * a;
                    idx++;
                }
            }
        }
    }
}

// ============================================================
// Add thermal velocity perturbation to positions
// (simulates a few MD steps so atoms are off-lattice)
// ============================================================
void perturb_positions(std::vector<double> &x_aos, int natoms,
                       double box[3], double temperature)
{
    // Simple random displacement scaled by temperature
    srand(87287);  // same seed as LAMMPS bench/in.lj
    double disp_scale = 0.05 * pow(4.0 / FCC_DENSITY, 1.0 / 3.0);

    for (int i = 0; i < natoms; i++) {
        for (int d = 0; d < 3; d++) {
            double r = (double)rand() / RAND_MAX - 0.5;
            x_aos[i * 3 + d] += r * disp_scale;
            // Periodic wrap
            if (x_aos[i * 3 + d] < 0)      x_aos[i * 3 + d] += box[d];
            if (x_aos[i * 3 + d] >= box[d]) x_aos[i * 3 + d] -= box[d];
        }
    }
}

// ============================================================
// Build full neighbor list (no Newton's 3rd law)
// Uses cell list for O(N) construction
// ============================================================
struct NeighList {
    std::vector<int> ilist;
    std::vector<int> numneigh;
    std::vector<int> neighbors;    // flattened
    std::vector<int> neigh_offset; // offset into neighbors for each atom
    int inum;
    int total_neighs;
};

void build_neighbor_list(const std::vector<double> &x_aos, int natoms,
                         double box[3], double cutoff, double skin,
                         NeighList &nlist)
{
    double cutsq = (cutoff + skin) * (cutoff + skin);

    // Cell list construction
    double cellsize = cutoff + skin;
    int ncx = std::max(1, (int)(box[0] / cellsize));
    int ncy = std::max(1, (int)(box[1] / cellsize));
    int ncz = std::max(1, (int)(box[2] / cellsize));
    int ncells = ncx * ncy * ncz;

    double cx = box[0] / ncx;
    double cy = box[1] / ncy;
    double cz = box[2] / ncz;

    // Assign atoms to cells
    std::vector<std::vector<int>> cells(ncells);
    for (int i = 0; i < natoms; i++) {
        int ix = (int)(x_aos[i * 3 + 0] / cx);
        int iy = (int)(x_aos[i * 3 + 1] / cy);
        int iz = (int)(x_aos[i * 3 + 2] / cz);
        ix = std::min(ix, ncx - 1);
        iy = std::min(iy, ncy - 1);
        iz = std::min(iz, ncz - 1);
        cells[iz * ncy * ncx + iy * ncx + ix].push_back(i);
    }

    // Build neighbor list
    nlist.ilist.resize(natoms);
    nlist.numneigh.resize(natoms, 0);
    nlist.neigh_offset.resize(natoms);

    std::vector<std::vector<int>> per_atom_neighs(natoms);

    for (int i = 0; i < natoms; i++) {
        double xi = x_aos[i * 3 + 0];
        double yi = x_aos[i * 3 + 1];
        double zi = x_aos[i * 3 + 2];

        int cix = (int)(xi / cx); cix = std::min(cix, ncx - 1);
        int ciy = (int)(yi / cy); ciy = std::min(ciy, ncy - 1);
        int ciz = (int)(zi / cz); ciz = std::min(ciz, ncz - 1);

        // Search 27 neighboring cells
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int jx = (cix + dx + ncx) % ncx;
                    int jy = (ciy + dy + ncy) % ncy;
                    int jz = (ciz + dz + ncz) % ncz;
                    int jcell = jz * ncy * ncx + jy * ncx + jx;

                    for (int j : cells[jcell]) {
                        if (j == i) continue;

                        double delx = xi - x_aos[j * 3 + 0];
                        double dely = yi - x_aos[j * 3 + 1];
                        double delz = zi - x_aos[j * 3 + 2];

                        // Periodic boundary (minimum image)
                        if (delx >  0.5 * box[0]) delx -= box[0];
                        if (delx < -0.5 * box[0]) delx += box[0];
                        if (dely >  0.5 * box[1]) dely -= box[1];
                        if (dely < -0.5 * box[1]) dely += box[1];
                        if (delz >  0.5 * box[2]) delz -= box[2];
                        if (delz < -0.5 * box[2]) delz += box[2];

                        double rsq = delx * delx + dely * dely + delz * delz;
                        if (rsq < cutsq) {
                            per_atom_neighs[i].push_back(j);
                        }
                    }
                }
            }
        }
    }

    // Flatten
    nlist.inum = natoms;
    nlist.total_neighs = 0;
    for (int i = 0; i < natoms; i++) {
        nlist.ilist[i] = i;
        nlist.numneigh[i] = (int)per_atom_neighs[i].size();
        nlist.neigh_offset[i] = nlist.total_neighs;
        nlist.total_neighs += nlist.numneigh[i];
    }

    nlist.neighbors.resize(nlist.total_neighs);
    for (int i = 0; i < natoms; i++) {
        int off = nlist.neigh_offset[i];
        for (int jj = 0; jj < nlist.numneigh[i]; jj++) {
            nlist.neighbors[off + jj] = per_atom_neighs[i][jj];
        }
    }
}

// ============================================================
// CPU reference LJ force computation (no Newton 3rd law)
// Matches the GPU kernel logic exactly
// ============================================================
void cpu_lj_force(const std::vector<double> &x_aos, int natoms,
                  const int *ilist, const int *numneigh,
                  const int *neighbors, const int *neigh_offset,
                  int inum,
                  double cutsq_val, double lj1_val, double lj2_val,
                  double lj3_val, double lj4_val, double offset_val,
                  std::vector<double> &f_aos, double &pe_total)
{
    f_aos.assign(natoms * 3, 0.0);
    pe_total = 0.0;

    for (int ii = 0; ii < inum; ii++) {
        int i = ilist[ii];
        double xtmp = x_aos[i * 3 + 0];
        double ytmp = x_aos[i * 3 + 1];
        double ztmp = x_aos[i * 3 + 2];

        double fix = 0.0, fiy = 0.0, fiz = 0.0;
        double pe_i = 0.0;

        int jnum   = numneigh[ii];
        int offset = neigh_offset[ii];

        for (int jj = 0; jj < jnum; jj++) {
            int j = neighbors[offset + jj];

            double delx = xtmp - x_aos[j * 3 + 0];
            double dely = ytmp - x_aos[j * 3 + 1];
            double delz = ztmp - x_aos[j * 3 + 2];
            double rsq  = delx * delx + dely * dely + delz * delz;

            if (rsq < cutsq_val) {
                double r2inv = 1.0 / rsq;
                double r6inv = r2inv * r2inv * r2inv;
                double forcelj = r6inv * (lj1_val * r6inv - lj2_val);
                double fpair = forcelj * r2inv;

                fix += delx * fpair;
                fiy += dely * fpair;
                fiz += delz * fpair;

                double evdwl = r6inv * (lj3_val * r6inv - lj4_val) - offset_val;
                pe_i += 0.5 * evdwl;
            }
        }

        f_aos[i * 3 + 0] = fix;
        f_aos[i * 3 + 1] = fiy;
        f_aos[i * 3 + 2] = fiz;
        pe_total += pe_i;
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv)
{
    // Parse args
    int ndim = 20;  // default: 20^3 * 4 = 32000 atoms (matches bench/in.lj)
    if (argc > 1) ndim = atoi(argv[1]);

    int natoms = 4 * ndim * ndim * ndim;

    printf("========================================================\n");
    printf("  LAMMPS LJ Force — CPU vs CUDA GPU Benchmark\n");
    printf("========================================================\n");
    printf("  Grid:     %d x %d x %d FCC\n", ndim, ndim, ndim);
    printf("  Atoms:    %d\n", natoms);
    printf("  Cutoff:   %.1f (skin=0.3)\n", LJ_CUTOFF);
    printf("  Steps:    %d (warmup: %d)\n", NSTEPS, WARMUP);

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU:      %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("========================================================\n\n");

    // ---- Generate system ----
    printf("[1/6] Generating FCC lattice...\n");
    std::vector<double> x_aos;
    double box[3];
    generate_fcc_lattice(ndim, ndim, ndim, FCC_DENSITY, x_aos, box);
    perturb_positions(x_aos, natoms, box, 1.44);
    printf("  Box: %.4f x %.4f x %.4f\n", box[0], box[1], box[2]);

    // Atom types: all type 1 (single component LJ)
    std::vector<int> type(natoms, 1);

    // ---- LJ parameters ----
    // LAMMPS convention: 1-indexed, ntypes=1, arrays are (ntypes+1)^2
    int ntypes = 1;
    int np1 = ntypes + 1;
    int np1sq = np1 * np1;

    std::vector<double> cutsq(np1sq, 0.0);
    std::vector<double> lj1(np1sq, 0.0), lj2(np1sq, 0.0);
    std::vector<double> lj3(np1sq, 0.0), lj4(np1sq, 0.0);
    std::vector<double> offset_arr(np1sq, 0.0);
    double special_lj[4] = {1.0, 0.0, 0.0, 0.0};  // default: no special bonds

    // Set params for type pair (1,1)
    double sigma6  = pow(LJ_SIGMA, 6.0);
    double sigma12 = sigma6 * sigma6;
    cutsq[1 * np1 + 1]  = LJ_CUTOFF * LJ_CUTOFF;
    lj1[1 * np1 + 1]    = 48.0 * LJ_EPSILON * sigma12;
    lj2[1 * np1 + 1]    = 24.0 * LJ_EPSILON * sigma6;
    lj3[1 * np1 + 1]    = 4.0  * LJ_EPSILON * sigma12;
    lj4[1 * np1 + 1]    = 4.0  * LJ_EPSILON * sigma6;

    // Energy offset at cutoff
    double r2inv_c = 1.0 / (LJ_CUTOFF * LJ_CUTOFF);
    double r6inv_c = r2inv_c * r2inv_c * r2inv_c;
    offset_arr[1 * np1 + 1] = r6inv_c * (lj3[1 * np1 + 1] * r6inv_c - lj4[1 * np1 + 1]);

    // ---- Build neighbor list ----
    printf("[2/6] Building full neighbor list (no Newton 3rd law)...\n");
    NeighList nlist;
    Timer timer;
    timer.start();
    build_neighbor_list(x_aos, natoms, box, LJ_CUTOFF, 0.3, nlist);
    double nlist_ms = timer.stop_ms();

    double avg_neighs = (double)nlist.total_neighs / natoms;
    printf("  Neighbors: %d total, %.1f avg/atom\n", nlist.total_neighs, avg_neighs);
    printf("  Build time: %.1f ms\n", nlist_ms);

    // ---- CPU baseline ----
    printf("[3/6] CPU baseline (%d iterations)...\n", NSTEPS);
    std::vector<double> f_cpu;
    double pe_cpu = 0.0;

    // Warmup
    cpu_lj_force(x_aos, natoms,
                 nlist.ilist.data(), nlist.numneigh.data(),
                 nlist.neighbors.data(), nlist.neigh_offset.data(),
                 nlist.inum,
                 cutsq[1 * np1 + 1], lj1[1 * np1 + 1], lj2[1 * np1 + 1],
                 lj3[1 * np1 + 1], lj4[1 * np1 + 1], offset_arr[1 * np1 + 1],
                 f_cpu, pe_cpu);

    timer.start();
    for (int step = 0; step < NSTEPS; step++) {
        cpu_lj_force(x_aos, natoms,
                     nlist.ilist.data(), nlist.numneigh.data(),
                     nlist.neighbors.data(), nlist.neigh_offset.data(),
                     nlist.inum,
                     cutsq[1 * np1 + 1], lj1[1 * np1 + 1], lj2[1 * np1 + 1],
                     lj3[1 * np1 + 1], lj4[1 * np1 + 1], offset_arr[1 * np1 + 1],
                     f_cpu, pe_cpu);
    }
    double cpu_total_ms = timer.stop_ms();
    double cpu_per_step = cpu_total_ms / NSTEPS;

    printf("  CPU time:   %.2f ms/step (total %.1f ms)\n", cpu_per_step, cpu_total_ms);
    printf("  CPU PE:     %.6f\n", pe_cpu);

    // ---- GPU ----
    printf("[4/6] GPU CUDA kernel...\n");

    // Allocate
    LJForceGPU gpu;
    lj_gpu_allocate(gpu, natoms, natoms, nlist.inum, ntypes, nlist.total_neighs);

    // Upload (once for benchmark — in real MD, upload each step)
    lj_gpu_upload(gpu, x_aos.data(), type.data(),
                  nlist.ilist.data(), nlist.numneigh.data(),
                  nlist.neighbors.data(), nlist.neigh_offset.data(),
                  cutsq.data(), lj1.data(), lj2.data(),
                  lj3.data(), lj4.data(), offset_arr.data(),
                  special_lj);

    // Warmup
    for (int w = 0; w < WARMUP; w++) {
        CUDA_CHECK(cudaMemset(gpu.d_fx, 0, natoms * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu.d_fy, 0, natoms * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu.d_fz, 0, natoms * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu.d_pe_result, 0, sizeof(double)));
        lj_gpu_compute(gpu, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark: kernel only (data already on GPU)
    float kern_times[NSTEPS];
    timer.start();
    for (int step = 0; step < NSTEPS; step++) {
        CUDA_CHECK(cudaMemset(gpu.d_fx, 0, natoms * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu.d_fy, 0, natoms * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu.d_fz, 0, natoms * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu.d_pe_result, 0, sizeof(double)));

        cudaEvent_t ks, ke;
        CUDA_CHECK(cudaEventCreate(&ks));
        CUDA_CHECK(cudaEventCreate(&ke));
        CUDA_CHECK(cudaEventRecord(ks));
        lj_gpu_compute(gpu, 1);
        CUDA_CHECK(cudaEventRecord(ke));
        CUDA_CHECK(cudaEventSynchronize(ke));
        CUDA_CHECK(cudaEventElapsedTime(&kern_times[step], ks, ke));
        CUDA_CHECK(cudaEventDestroy(ks));
        CUDA_CHECK(cudaEventDestroy(ke));
    }
    double gpu_total_ms = timer.stop_ms();

    // Sort kernel times and take median
    std::sort(kern_times, kern_times + NSTEPS);
    float kern_median = kern_times[NSTEPS / 2];
    float kern_min    = kern_times[0];
    float kern_max    = kern_times[NSTEPS - 1];
    double gpu_per_step = gpu_total_ms / NSTEPS;

    printf("  GPU total:  %.2f ms/step (incl. memset overhead)\n", gpu_per_step);
    printf("  GPU kernel: %.3f ms (median), %.3f ms (min), %.3f ms (max)\n",
           kern_median, kern_min, kern_max);

    // Benchmark: full round-trip (upload + kernel + download)
    float h2d_ms, kernel_ms, d2h_ms;
    CUDA_CHECK(cudaMemset(gpu.d_pe_result, 0, sizeof(double)));
    lj_gpu_upload(gpu, x_aos.data(), type.data(),
                  nlist.ilist.data(), nlist.numneigh.data(),
                  nlist.neighbors.data(), nlist.neigh_offset.data(),
                  cutsq.data(), lj1.data(), lj2.data(),
                  lj3.data(), lj4.data(), offset_arr.data(),
                  special_lj);
    lj_gpu_compute(gpu, 1);
    std::vector<double> f_gpu(natoms * 3, 0.0);
    double pe_gpu = 0.0;
    lj_gpu_download_forces(gpu, f_gpu.data(), &pe_gpu);
    lj_gpu_get_timings(gpu, &h2d_ms, &kernel_ms, &d2h_ms);

    float total_roundtrip = h2d_ms + kernel_ms + d2h_ms;
    printf("  Round-trip: H2D=%.3f + Kern=%.3f + D2H=%.3f = %.3f ms\n",
           h2d_ms, kernel_ms, d2h_ms, total_roundtrip);
    printf("  GPU PE:     %.6f\n", pe_gpu);

    // ---- Validation ----
    printf("\n[5/6] Validating correctness...\n");
    double max_err = 0.0;
    double max_rel_err = 0.0;
    int err_count = 0;

    for (int i = 0; i < natoms; i++) {
        for (int d = 0; d < 3; d++) {
            double cpu_val = f_cpu[i * 3 + d];
            double gpu_val = f_gpu[i * 3 + d];
            double err = fabs(cpu_val - gpu_val);
            double rel = (fabs(cpu_val) > 1e-10) ? err / fabs(cpu_val) : err;
            if (err > max_err) max_err = err;
            if (rel > max_rel_err) max_rel_err = rel;
            if (rel > 1e-6) err_count++;
        }
    }

    double pe_err = fabs(pe_cpu - pe_gpu);
    double pe_rel = (fabs(pe_cpu) > 1e-10) ? pe_err / fabs(pe_cpu) : pe_err;

    printf("  Force max abs error: %.6e\n", max_err);
    printf("  Force max rel error: %.6e\n", max_rel_err);
    printf("  PE error (abs/rel):  %.6e / %.6e\n", pe_err, pe_rel);
    printf("  Atoms with rel err > 1e-6: %d / %d\n", err_count, natoms * 3);

    bool pass = (max_rel_err < 1e-4) && (pe_rel < 1e-4);
    printf("  Validation: %s\n", pass ? "PASS" : "FAIL");

    // ---- Summary ----
    printf("\n[6/6] Performance Summary\n");
    printf("========================================================\n");
    printf("  %-25s %10s %10s %10s\n", "Config", "ms/step", "Pairs/s", "Speedup");
    printf("  %-25s %10s %10s %10s\n", "-------------------------", "----------", "----------", "----------");

    double pairs_per_step = (double)nlist.total_neighs;
    double cpu_pairs_s = pairs_per_step / (cpu_per_step / 1000.0);
    double gpu_kern_pairs_s = pairs_per_step / (kern_median / 1000.0);
    double gpu_rt_pairs_s = pairs_per_step / (total_roundtrip / 1000.0);

    printf("  %-25s %10.3f %10.2e %10s\n",
           "CPU (1 thread)", cpu_per_step, cpu_pairs_s, "1.0x");
    printf("  %-25s %10.3f %10.2e %10.1fx\n",
           "GPU (kernel only)", (double)kern_median, gpu_kern_pairs_s,
           cpu_per_step / kern_median);
    printf("  %-25s %10.3f %10.2e %10.1fx\n",
           "GPU (full round-trip)", (double)total_roundtrip, gpu_rt_pairs_s,
           cpu_per_step / total_roundtrip);
    printf("========================================================\n");

    // Cleanup
    lj_gpu_free(gpu);

    return pass ? 0 : 1;
}
