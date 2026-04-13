// =============================================================================
// XSbench_header.h — C-compatible shim for the CUDA build
//
// The original .c files (io.c, GridInit.c, XSutils.c, Materials.c) include
// this header via #include "XSbench_header.h". It must compile cleanly with
// gcc (no CUDA keywords). The .cu files include XSbench_header.cuh instead.
//
// This header defines the SAME structs and host-only prototypes, but without
// __device__ / __global__ declarations or CUDA includes.
// =============================================================================

#ifndef __XSBENCH_HEADER_H__
#define __XSBENCH_HEADER_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

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
// Structures — must be IDENTICAL to those in XSbench_header.cuh
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
    // Must match .cuh: verification array (unused by C files, but struct layout must match)
    unsigned long long * verification;
    int length_verification;
} SimulationData;

// =============================================================================
// Host-only function prototypes
// =============================================================================

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

// HostHelpers.c
double LCG_random_double(uint64_t * seed);
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high);

// GridInit.c
SimulationData grid_init_do_not_profile( Inputs in, int mype );

// XSutils.c
int NGP_compare( const void * a, const void * b );
int double_compare(const void * a, const void * b);
size_t estimate_mem_usage( Inputs in );
double get_time(void);

// Materials.c
int * load_num_nucs(long n_isotopes);
int * load_mats( int * num_nucs, long n_isotopes, int * max_num_nucs );
double * load_concs( int * num_nucs, int max_num_nucs );


#ifdef __cplusplus
}
#endif

#endif
