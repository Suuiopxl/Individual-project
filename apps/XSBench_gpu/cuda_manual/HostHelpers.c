// =============================================================================
// HostHelpers.c — Host-side implementations of functions needed by GridInit.c
//                 and Materials.c during initialization (runs on CPU only)
// =============================================================================

#include "XSbench_header.h"

// LCG random double — host version (same algorithm as device version)
double LCG_random_double(uint64_t * seed)
{
    const uint64_t m = 9223372036854775808ULL; // 2^63
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return (double) (*seed) / (double) m;
}

// Fast-forward LCG — host version
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
    const uint64_t m = 9223372036854775808ULL;
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

// Binary search on nuclide grid — host version
long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high)
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
