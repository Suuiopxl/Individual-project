#ifndef __XSBENCH_HEADER_CUH__
#define __XSBENCH_HEADER_CUH__

// =============================================================================
// XSBench CUDA Header
// Manually converted from openmp-threading/XSbench_header.h
// Changes: added __device__ qualifiers, CUDA includes, GPU error checking,
//          verification array in SimulationData for per-thread reduction
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

// GPU error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Grid types
#define UNIONIZED 0
#define NUCLIDE 1
#define HASH 2

// Simulation types
#define HISTORY_BASED 1
#define EVENT_BASED 2

// Binary Mode Type
#define NONE 0
#define READ 1
#define WRITE 2

// Starting Seed
#define STARTING_SEED 1070

// =============================================================================
// Structures (same as CPU version)
// =============================================================================

typedef struct{
    double energy;
    double total_xs;
    double elastic_xs;
    double absorbtion_xs;
    double fission_xs;
    double nu_fission_xs;
} NuclideGridPoint;

typedef struct{
    int nthreads;
    long n_isotopes;
    long n_gridpoints;
    int lookups;
    char * HM;
    int grid_type;
    int hash_bins;
    int particles;
    int simulation_method;
    int binary_mode;
    int kernel_id;
} Inputs;

typedef struct{
    int * num_nucs;
    double * concs;
    int * mats;
    double * unionized_energy_array;
    int * index_grid;
    NuclideGridPoint * nuclide_grid;
    int length_num_nucs;
    int length_concs;
    int length_mats;
    int length_unionized_energy_array;
    long length_index_grid;
    int length_nuclide_grid;
    int max_num_nucs;
    // GPU-specific: per-lookup verification stored in array, reduced on host
    unsigned long long * verification;
    int length_verification;
} SimulationData;

// =============================================================================
// Host-side function prototypes (io.c, XSutils.c, Materials.c, GridInit.c)
// Wrapped in extern "C" because these are compiled by gcc as C code
// =============================================================================

extern "C" {

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int(long a);
Inputs read_CLI( int argc, char * argv[] );
void print_CLI_error(void);
void print_inputs(Inputs in, int nprocs, int version);
int print_results( Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash );
void binary_write( Inputs in, SimulationData SD );
SimulationData binary_read( Inputs in );

// GridInit.cu
SimulationData grid_init_do_not_profile( Inputs in, int mype );
SimulationData move_simulation_data_to_device( Inputs in, int mype, SimulationData SD );
void release_device_memory(SimulationData GSD);

// XSutils.cu
int NGP_compare( const void * a, const void * b );
int double_compare(const void * a, const void * b);
size_t estimate_mem_usage( Inputs in );
double get_time(void);

// Materials.cu
int * load_num_nucs(long n_isotopes);
int * load_mats( int * num_nucs, long n_isotopes, int * max_num_nucs );
double * load_concs( int * num_nucs, int max_num_nucs );

} // end extern "C"

// =============================================================================
// Device-side function prototypes (Simulation.cu)
// =============================================================================

// Device helper functions
__device__ long grid_search_device( long n, double quarry, double * A);
__device__ long grid_search_nuclide_device( long n, double quarry, NuclideGridPoint * A, long low, long high);
__device__ void calculate_micro_xs_device(
    double p_energy, int nuc, long n_isotopes, long n_gridpoints,
    double * egrid, int * index_data, NuclideGridPoint * nuclide_grids,
    long idx, double * xs_vector, int grid_type, int hash_bins );
__device__ void calculate_macro_xs_device(
    double p_energy, int mat, long n_isotopes, long n_gridpoints,
    int * num_nucs, double * concs, double * egrid, int * index_data,
    NuclideGridPoint * nuclide_grids, int * mats,
    double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs );
__device__ int pick_mat_device( uint64_t * seed );
__device__ double LCG_random_double_device(uint64_t * seed);
__device__ uint64_t fast_forward_LCG_device(uint64_t seed, uint64_t n);

// CUDA kernel
__global__ void xs_lookup_kernel(Inputs in, SimulationData GSD);

// Host-side simulation entry point
unsigned long long run_event_based_simulation_cuda(Inputs in, SimulationData SD, int mype);

#endif
