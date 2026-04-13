// =============================================================================
// Simulation.cu — XSBench CUDA kernel (event-based simulation)
//
// Manually converted from openmp-threading/Simulation.c
// Key changes:
//   - All inner functions become __device__ functions
//   - The OpenMP parallel for is replaced by a CUDA kernel
//   - Each thread handles one independent XS lookup
//   - Verification values stored in per-thread array, reduced on host
// =============================================================================

#include "XSbench_header.cuh"

// =============================================================================
// Device function: Linear Congruential Generator (LCG) random double
// =============================================================================
__device__ double LCG_random_double_device(uint64_t * seed)
{
    const uint64_t m = 9223372036854775808ULL; // 2^63
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return (double) (*seed) / (double) m;
}

// =============================================================================
// Device function: Fast-forward LCG state by n steps
// =============================================================================
__device__ uint64_t fast_forward_LCG_device(uint64_t seed, uint64_t n)
{
    const uint64_t m = 9223372036854775808ULL; // 2^63
    uint64_t a = 2806196910506780709ULL;
    uint64_t c = 1ULL;

    n = n % m;

    uint64_t a_new = 1;
    uint64_t c_new = 0;

    while(n > 0)
    {
        if(n & 1)
        {
            a_new *= a;
            c_new = c_new * a + c;
        }
        c *= (a + 1);
        a *= a;
        n >>= 1;
    }

    return (a_new * seed + c_new) % m;
}

// =============================================================================
// Device function: Pick material based on probabilistic distribution
// =============================================================================
__device__ int pick_mat_device( uint64_t * seed )
{
    // Volume fractions of materials in the reactor core
    // Must match the CPU version in Simulation.c exactly
    double dist[12];
    dist[0]  = 0.140;  // fuel
    dist[1]  = 0.052;  // cladding
    dist[2]  = 0.275;  // cold, borated water
    dist[3]  = 0.134;  // hot, borated water
    dist[4]  = 0.154;  // RPV
    dist[5]  = 0.064;  // Lower, radial reflector
    dist[6]  = 0.066;  // Upper reflector / top plate
    dist[7]  = 0.055;  // bottom plate
    dist[8]  = 0.008;  // bottom nozzle
    dist[9]  = 0.015;  // top nozzle
    dist[10] = 0.025;  // top of fuel assemblies
    dist[11] = 0.013;  // bottom of fuel assemblies

    double roll = LCG_random_double_device(seed);

    // Same cumulative loop as CPU version
    for( int i = 0; i < 12; i++ )
    {
        double running = 0;
        for( int j = i; j > 0; j-- )
            running += dist[j];
        if( roll < running )
            return i;
    }

    return 0;
}

// =============================================================================
// Device function: Binary search on unionized energy grid
// =============================================================================
__device__ long grid_search_device( long n, double quarry, double * A)
{
    long lowerLimit = 0;
    long upperLimit = n-1;
    long examinationPoint;
    long length = upperLimit - lowerLimit;

    while( length > 1 )
    {
        examinationPoint = lowerLimit + ( length / 2 );

        if( A[examinationPoint] > quarry )
            upperLimit = examinationPoint;
        else
            lowerLimit = examinationPoint;

        length = upperLimit - lowerLimit;
    }

    return lowerLimit;
}

// =============================================================================
// Device function: Binary search on nuclide energy grid
// =============================================================================
__device__ long grid_search_nuclide_device( long n, double quarry,
    NuclideGridPoint * A, long low, long high)
{
    long lowerLimit = low;
    long upperLimit = high;
    long examinationPoint;
    long length = upperLimit - lowerLimit;

    while( length > 1 )
    {
        examinationPoint = lowerLimit + ( length / 2 );

        if( A[examinationPoint].energy > quarry )
            upperLimit = examinationPoint;
        else
            lowerLimit = examinationPoint;

        length = upperLimit - lowerLimit;
    }

    return lowerLimit;
}

// =============================================================================
// Device function: Calculate microscopic cross section for a single nuclide
// =============================================================================
__device__ void calculate_micro_xs_device(
    double p_energy, int nuc, long n_isotopes, long n_gridpoints,
    double * egrid, int * index_data, NuclideGridPoint * nuclide_grids,
    long idx, double * xs_vector, int grid_type, int hash_bins )
{
    double f;
    NuclideGridPoint * low, * high;

    if( grid_type == NUCLIDE )
    {
        idx = grid_search_nuclide_device( n_gridpoints, p_energy,
            &nuclide_grids[nuc * n_gridpoints], 0, n_gridpoints - 1);

        if( idx == n_gridpoints - 1 )
            low = &nuclide_grids[nuc * n_gridpoints + idx - 1];
        else
            low = &nuclide_grids[nuc * n_gridpoints + idx];
    }
    else if( grid_type == UNIONIZED )
    {
        if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
            low = &nuclide_grids[nuc * n_gridpoints + index_data[idx * n_isotopes + nuc] - 1];
        else
            low = &nuclide_grids[nuc * n_gridpoints + index_data[idx * n_isotopes + nuc]];
    }
    else // HASH grid
    {
        int u_low = index_data[idx * n_isotopes + nuc];

        int u_high;
        if( idx == hash_bins - 1 )
            u_high = n_gridpoints - 1;
        else
            u_high = index_data[(idx + 1) * n_isotopes + nuc] + 1;

        double e_low  = nuclide_grids[nuc * n_gridpoints + u_low].energy;
        double e_high = nuclide_grids[nuc * n_gridpoints + u_high].energy;
        int lower;
        if( p_energy <= e_low )
            lower = 0;
        else if( p_energy >= e_high )
            lower = n_gridpoints - 1;
        else
            lower = grid_search_nuclide_device( n_gridpoints, p_energy,
                &nuclide_grids[nuc * n_gridpoints], u_low, u_high);

        if( lower == n_gridpoints - 1 )
            low = &nuclide_grids[nuc * n_gridpoints + lower - 1];
        else
            low = &nuclide_grids[nuc * n_gridpoints + lower];
    }

    high = low + 1;

    // Interpolation factor
    f = (high->energy - p_energy) / (high->energy - low->energy);

    // Interpolate all 5 cross section channels
    xs_vector[0] = high->total_xs      - f * (high->total_xs      - low->total_xs);
    xs_vector[1] = high->elastic_xs    - f * (high->elastic_xs    - low->elastic_xs);
    xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
    xs_vector[3] = high->fission_xs    - f * (high->fission_xs    - low->fission_xs);
    xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

// =============================================================================
// Device function: Calculate macroscopic cross section for a material
// Aggregates micro XS from all constituent nuclides weighted by concentration
// =============================================================================
__device__ void calculate_macro_xs_device(
    double p_energy, int mat, long n_isotopes, long n_gridpoints,
    int * num_nucs, double * concs, double * egrid, int * index_data,
    NuclideGridPoint * nuclide_grids, int * mats,
    double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs )
{
    int p_nuc;
    long idx = -1;
    double conc;

    // Zero out the result vector
    for( int k = 0; k < 5; k++ )
        macro_xs_vector[k] = 0;

    // Perform one binary search per macro lookup (unionized grid)
    if( grid_type == UNIONIZED )
        idx = grid_search_device( n_isotopes * n_gridpoints, p_energy, egrid);
    else if( grid_type == HASH )
    {
        double du = 1.0 / hash_bins;
        idx = (long)(p_energy / du);
    }

    // Loop over all nuclides in this material, accumulate weighted micro XS
    for( int j = 0; j < num_nucs[mat]; j++ )
    {
        double xs_vector[5];
        p_nuc = mats[mat * max_num_nucs + j];
        conc  = concs[mat * max_num_nucs + j];

        calculate_micro_xs_device(
            p_energy, p_nuc, n_isotopes, n_gridpoints,
            egrid, index_data, nuclide_grids,
            idx, xs_vector, grid_type, hash_bins );

        for( int k = 0; k < 5; k++ )
            macro_xs_vector[k] += xs_vector[k] * conc;
    }
}

// =============================================================================
// CUDA Kernel: Event-based XS lookup — one thread per lookup
// This replaces the OpenMP parallel for loop in run_event_based_simulation()
// =============================================================================
__global__ void xs_lookup_kernel(Inputs in, SimulationData GSD)
{
    // Global thread ID = one independent XS lookup
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid >= in.lookups )
        return;

    // Each thread gets its own deterministic random seed
    // Forward from the starting seed by 2*tid (need 2 random samples per lookup)
    uint64_t seed = STARTING_SEED;
    seed = fast_forward_LCG_device(seed, 2 * (uint64_t)tid);

    // Sample a random energy and material
    double p_energy = LCG_random_double_device(&seed);
    int mat         = pick_mat_device(&seed);

    // Thread-local result buffer for 5 XS reaction channels
    double macro_xs_vector[5] = {0};

    // Perform the macroscopic cross section lookup
    calculate_macro_xs_device(
        p_energy, mat,
        in.n_isotopes, in.n_gridpoints,
        GSD.num_nucs, GSD.concs,
        GSD.unionized_energy_array, GSD.index_grid,
        GSD.nuclide_grid, GSD.mats,
        macro_xs_vector,
        in.grid_type, in.hash_bins, GSD.max_num_nucs );

    // Compute verification hash: find the max XS channel index
    double max_val = -1.0;
    int max_idx = 0;
    for( int j = 0; j < 5; j++ )
    {
        if( macro_xs_vector[j] > max_val )
        {
            max_val = macro_xs_vector[j];
            max_idx = j;
        }
    }

    // Store per-thread verification value (reduced on host later)
    GSD.verification[tid] = (unsigned long long)(max_idx + 1);
}

// =============================================================================
// Host function: Launch CUDA event-based simulation
// =============================================================================
unsigned long long run_event_based_simulation_cuda(Inputs in, SimulationData SD, int mype)
{
    if( mype == 0 )
        printf("Beginning CUDA event-based simulation...\n");

    // =========================================================================
    // Step 1: Move all simulation data to GPU device memory
    // =========================================================================
    SimulationData GSD = move_simulation_data_to_device(in, mype, SD);

    // =========================================================================
    // Step 2: Configure and launch kernel
    // =========================================================================
    int threads_per_block = 256;
    int num_blocks = (in.lookups + threads_per_block - 1) / threads_per_block;

    if( mype == 0 )
        printf("Launching kernel: %d blocks x %d threads (%d lookups)\n",
            num_blocks, threads_per_block, in.lookups);

    double start = get_time();

    xs_lookup_kernel<<<num_blocks, threads_per_block>>>(in, GSD);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    double stop = get_time();

    if( mype == 0 )
        printf("CUDA XS Lookups took %.3lf seconds\n", stop - start);

    // =========================================================================
    // Step 3: Copy verification array back to host and reduce
    // =========================================================================
    unsigned long long * verification_host =
        (unsigned long long *) malloc(in.lookups * sizeof(unsigned long long));

    gpuErrchk( cudaMemcpy(
        verification_host, GSD.verification,
        in.lookups * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost) );

    unsigned long long verification = 0;
    for( int i = 0; i < in.lookups; i++ )
        verification += verification_host[i];

    free(verification_host);

    // =========================================================================
    // Step 4: Free device memory
    // =========================================================================
    release_device_memory(GSD);

    return verification;
}
