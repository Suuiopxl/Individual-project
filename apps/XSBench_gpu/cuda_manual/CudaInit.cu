// =============================================================================
// GridInit.cu — GPU memory management for XSBench CUDA version
//
// Contains:
//   - grid_init_do_not_profile(): unchanged from CPU version (host-side init)
//   - move_simulation_data_to_device(): cudaMalloc + cudaMemcpy all arrays
//   - release_device_memory(): cudaFree all device arrays
//
// The original grid_init_do_not_profile() code is compiled from the CPU
// source (GridInit.c) via extern "C" linkage or included here.
// For simplicity, we only define the GPU transfer functions here.
// GridInit.c is compiled separately as a C file.
// =============================================================================

#include "XSbench_header.cuh"

// =============================================================================
// Move all simulation data arrays from host to device
// Returns a new SimulationData struct with device pointers
// =============================================================================
SimulationData move_simulation_data_to_device( Inputs in, int mype, SimulationData SD )
{
    if( mype == 0 ) printf("Allocating and copying data to GPU...\n");

    SimulationData GSD;  // Device-side copy with GPU pointers

    // Copy scalar fields directly
    GSD.length_num_nucs              = SD.length_num_nucs;
    GSD.length_concs                 = SD.length_concs;
    GSD.length_mats                  = SD.length_mats;
    GSD.length_unionized_energy_array = SD.length_unionized_energy_array;
    GSD.length_index_grid            = SD.length_index_grid;
    GSD.length_nuclide_grid          = SD.length_nuclide_grid;
    GSD.max_num_nucs                 = SD.max_num_nucs;

    // ---- num_nucs (int array, length = length_num_nucs) ----
    size_t sz_num_nucs = SD.length_num_nucs * sizeof(int);
    gpuErrchk( cudaMalloc((void**)&GSD.num_nucs, sz_num_nucs) );
    gpuErrchk( cudaMemcpy(GSD.num_nucs, SD.num_nucs, sz_num_nucs, cudaMemcpyHostToDevice) );

    // ---- concs (double array, length = length_concs) ----
    size_t sz_concs = SD.length_concs * sizeof(double);
    gpuErrchk( cudaMalloc((void**)&GSD.concs, sz_concs) );
    gpuErrchk( cudaMemcpy(GSD.concs, SD.concs, sz_concs, cudaMemcpyHostToDevice) );

    // ---- mats (int array, length = length_mats) ----
    size_t sz_mats = SD.length_mats * sizeof(int);
    gpuErrchk( cudaMalloc((void**)&GSD.mats, sz_mats) );
    gpuErrchk( cudaMemcpy(GSD.mats, SD.mats, sz_mats, cudaMemcpyHostToDevice) );

    // ---- unionized_energy_array (double, may be zero length) ----
    if( SD.length_unionized_energy_array > 0 )
    {
        size_t sz_uea = SD.length_unionized_energy_array * sizeof(double);
        gpuErrchk( cudaMalloc((void**)&GSD.unionized_energy_array, sz_uea) );
        gpuErrchk( cudaMemcpy(GSD.unionized_energy_array, SD.unionized_energy_array, sz_uea, cudaMemcpyHostToDevice) );
    }
    else
    {
        GSD.unionized_energy_array = NULL;
    }

    // ---- index_grid (int array, may be zero length) ----
    if( SD.length_index_grid > 0 )
    {
        size_t sz_ig = SD.length_index_grid * sizeof(int);
        gpuErrchk( cudaMalloc((void**)&GSD.index_grid, sz_ig) );
        gpuErrchk( cudaMemcpy(GSD.index_grid, SD.index_grid, sz_ig, cudaMemcpyHostToDevice) );
    }
    else
    {
        GSD.index_grid = NULL;
    }

    // ---- nuclide_grid (NuclideGridPoint array) ----
    size_t sz_ng = SD.length_nuclide_grid * sizeof(NuclideGridPoint);
    gpuErrchk( cudaMalloc((void**)&GSD.nuclide_grid, sz_ng) );
    gpuErrchk( cudaMemcpy(GSD.nuclide_grid, SD.nuclide_grid, sz_ng, cudaMemcpyHostToDevice) );

    // ---- verification array (per-thread output, device-only initially) ----
    GSD.length_verification = in.lookups;
    size_t sz_ver = in.lookups * sizeof(unsigned long long);
    gpuErrchk( cudaMalloc((void**)&GSD.verification, sz_ver) );
    gpuErrchk( cudaMemset(GSD.verification, 0, sz_ver) );

    // Report memory usage
    if( mype == 0 )
    {
        size_t total = sz_num_nucs + sz_concs + sz_mats + sz_ng + sz_ver;
        if( SD.length_unionized_energy_array > 0 )
            total += SD.length_unionized_energy_array * sizeof(double);
        if( SD.length_index_grid > 0 )
            total += SD.length_index_grid * sizeof(int);
        printf("GPU memory allocated: %.2f MB\n", (double)total / (1024.0 * 1024.0));
    }

    return GSD;
}

// =============================================================================
// Free all device memory
// =============================================================================
void release_device_memory(SimulationData GSD)
{
    gpuErrchk( cudaFree(GSD.num_nucs) );
    gpuErrchk( cudaFree(GSD.concs) );
    gpuErrchk( cudaFree(GSD.mats) );
    if( GSD.unionized_energy_array != NULL )
        gpuErrchk( cudaFree(GSD.unionized_energy_array) );
    if( GSD.index_grid != NULL )
        gpuErrchk( cudaFree(GSD.index_grid) );
    gpuErrchk( cudaFree(GSD.nuclide_grid) );
    gpuErrchk( cudaFree(GSD.verification) );
}
