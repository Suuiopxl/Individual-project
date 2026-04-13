// =============================================================================
// Main.cu — XSBench CUDA version entry point
//
// Based on openmp-threading/Main.c
// Changes:
//   - Forces event-based simulation (GPU requires independent lookups)
//   - Calls run_event_based_simulation_cuda() instead of CPU version
//   - Removed OpenMP and MPI code paths for clarity
// =============================================================================

#include "XSbench_header.cuh"

int main( int argc, char* argv[] )
{
    // =========================================================================
    // Initialization & Command Line Read-In
    // =========================================================================
    int version = 20;
    int mype = 0;
    int nprocs = 1;

    // Process CLI arguments
    Inputs in = read_CLI( argc, argv );

    // Force event-based simulation for GPU
    // (History-based has loop-carried dependencies, not suitable for GPU)
    if( in.simulation_method == HISTORY_BASED )
    {
        if( mype == 0 )
            printf("NOTE: GPU mode requires event-based simulation. Switching from history to event mode.\n");
        in.simulation_method = EVENT_BASED;
        // Event-based default lookups if user used history-based default
        if( in.lookups == 34 )
            in.lookups = 17000000;
    }

    // Print input summary
    if( mype == 0 )
        print_inputs( in, nprocs, version );

    // =========================================================================
    // Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
    // (This runs on the host — same as CPU version)
    // =========================================================================
    SimulationData SD = grid_init_do_not_profile( in, mype );

    // =========================================================================
    // Cross Section (XS) Parallel Lookup Simulation — CUDA
    // =========================================================================
    if( mype == 0 )
    {
        border_print();
        center_print("SIMULATION", 79);
        border_print();
    }

    double start = get_time();

    unsigned long long vhash = run_event_based_simulation_cuda(in, SD, mype);

    double stop = get_time();

    // =========================================================================
    // Print / Save Results and Validate
    // =========================================================================
    if( mype == 0 )
    {
        printf("\n");
        border_print();
        center_print("RESULTS", 79);
        border_print();

        double runtime = stop - start;
        printf("Runtime:     %.3lf seconds\n", runtime);
        printf("Lookups:     "); fancy_int(in.lookups);
        printf("Lookups/s:   "); fancy_int((int)((double)in.lookups / runtime));
        printf("Verification: %llu\n", vhash);

        // Known verification checksums for event-based simulation
        // -s small: lookups=17000000 -> vhash should match
        print_results( in, mype, runtime, nprocs, vhash );
    }

    // =========================================================================
    // Cleanup host memory
    // =========================================================================
    free(SD.num_nucs);
    free(SD.concs);
    free(SD.mats);
    if( SD.unionized_energy_array != NULL )
        free(SD.unionized_energy_array);
    if( SD.index_grid != NULL )
        free(SD.index_grid);
    free(SD.nuclide_grid);

    return 0;
}
