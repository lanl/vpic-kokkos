// Compare the Kokkos and legacy rho_p against a double-precision sum of the
// same per-particle float charge weights. The double sum has its own rounding
// error, but over a few hundred float terms it is many orders of magnitude
// below the float tolerance used here, so it serves as the reference.
//
// Float summation error depends on the order of the additions, which differs
// between the legacy loop, OpenMP (duplicated scatter) and GPUs (atomics), so
// neither implementation can serve as the reference for the other. For a sum
// of n positive terms in any order, the error is bounded by (n-1)*u*sum with
// u = FLT_EPSILON/2. Each grid node is checked against that bound for its own
// contribution count, plus a few ulps for the weights themselves.

#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"

#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

#include "src/species_advance/species_advance.h"
#include "src/vpic/vpic.h"

void vpic_simulation::user_diagnostics() {}

void
vpic_simulation::user_initialization( int num_cmdline_arguments,
                                      char ** cmdline_argument )
{
    double L  = 1e2;
    int ncell = 8;
    int npart = 32*ncell*ncell*ncell;  // 32 per cell: ~256 contributions per interior grid node

    define_units( 1, 1 );
    define_timestep( 1 );
    define_periodic_grid( 0, 0, 0,   // Grid low corner
            L, L, L,   // Grid high corner
            ncell, ncell, ncell,   // Grid resolution
            1, 1, 1 ); // Processor configuration
    define_material( "vacuum", 1.0, 1.0, 0.0 );
    define_field_array();

    field(1,1,1).ex  = 1;
    field(1,2,1).ex  = 1;
    field(1,1,2).ex  = 1;
    field(1,2,2).ex  = 1;

    field(1,1,1).ey  = 2;
    field(1,1,2).ey  = 2;
    field(2,1,1).ey  = 2;
    field(2,1,2).ey  = 2;

    field(1,1,1).ez  = 3;
    field(2,1,1).ez  = 3;
    field(1,2,1).ez  = 3;
    field(2,2,1).ez  = 3;

    species_t * sp_temp;
    species_t * sp = define_species( "test_species", 1., 1., npart, npart, 0, 0 );

    for (int i = 0; i < npart; i++)
    {
        float x = uniform( rng(0), 0, L);
        float y = uniform( rng(0), 0, L);
        float z = uniform( rng(0), 0, L);

        inject_particle( sp , x, y, z, 0., 0., 0., 1., 0., 0);
    }

    // Make sure kokkos views have correct data
    field_array->copy_to_device();
    LIST_FOR_EACH( sp_temp, species_list ) {
      sp_temp->copy_to_device();
    }

    advance_p( sp, interpolator_array, field_array );

    // advance_p moved the particles on the device; give the legacy array the
    // same particles
    sp->copy_to_host();

    // Call both functions
    k_accumulate_rho_p( field_array, sp );
    accumulate_rho_p( field_array, sp );

    // Copy the data back to host view
    Kokkos::deep_copy(field_array->k_f_h, field_array->k_f_d);

    // Reference: the same float weights as both implementations,
    // accumulated in double, with the number of contributions to each grid node
    const float q_8V = sp->q*grid->r8V;
    const int sy = grid->sy;
    const int sz = grid->sz;
    std::vector<double> reference(grid->nv, 0.0);
    std::vector<int> count(grid->nv, 0);
    for (size_t n = 0; n < (size_t)sp->np; n++)
    {
        float w0 = sp->k_p_h(n, particle_var::dx);
        float w1 = sp->k_p_h(n, particle_var::dy);
        float dz = sp->k_p_h(n, particle_var::dz);
        float w7 = sp->k_p_h(n, particle_var::w)*q_8V;
        int v = sp->k_p_i_h(n);
        float w2, w3, w4, w5, w6;

        w6 = w7 - w0*w7;  w7 = w7 + w0*w7;
        w4 = w6 - w1*w6;  w5 = w7 - w1*w7;
        w6 = w6 + w1*w6;  w7 = w7 + w1*w7;
        w0 = w4 - dz*w4;  w1 = w5 - dz*w5;  w2 = w6 - dz*w6;  w3 = w7 - dz*w7;
        w4 = w4 + dz*w4;  w5 = w5 + dz*w5;  w6 = w6 + dz*w6;  w7 = w7 + dz*w7;

        const int node[8] = { v, v+1, v+sy, v+sy+1, v+sz, v+sz+1, v+sz+sy, v+sz+sy+1 };
        const float w[8]  = { w0, w1, w2, w3, w4, w5, w6, w7 };
        for (int k = 0; k < 8; k++)
        {
            reference[node[k]] += w[k];
            count[node[k]]++;
        }
    }

    int failed = 0;
    double worst_kokkos = 0, worst_legacy = 0;

    for (int i = 0; i < grid->nv; i++)
    {
        double kokkos = field_array->k_f_h(i, field_var::rhof);
        double legacy = field_array->f[i].rhof;

        if (count[i] == 0)
        {
            if (kokkos != 0 || legacy != 0)
            {
                std::cout << " Failed at " << i << ": no particles contribute, but kokkos "
                          << kokkos << " legacy " << legacy << std::endl;
                failed++;
            }
            continue;
        }

        // (n-1)*u bounds the summation, a few more ulps cover the weights
        const double tol = (count[i] + 8) * 0.5*FLT_EPSILON;
        const double err_kokkos = std::abs(kokkos - reference[i]) / reference[i];
        const double err_legacy = std::abs(legacy - reference[i]) / reference[i];
        worst_kokkos = std::max(worst_kokkos, err_kokkos / tol);
        worst_legacy = std::max(worst_legacy, err_legacy / tol);

        if (err_kokkos > tol || err_legacy > tol)
        {
            std::cout << " Failed at " << i << " (" << count[i] << " contributions, tolerance "
                      << tol << "): kokkos error " << err_kokkos << ", legacy error "
                      << err_legacy << std::endl;
            failed++;
        }
    }

    std::cout << "Worst error as a fraction of the tolerance: kokkos " << worst_kokkos
              << ", legacy " << worst_legacy << std::endl;
    if( failed )
    {  std::cout << "FAIL" << std::endl;
    }
    REQUIRE_FALSE(failed);

    std::cout << "pass" << std::endl;
}

TEST_CASE( "Kokkos and legacy rho_p match a double-precision sum", "[rho_p]" )
{
    vpic_simulation* simulation = new vpic_simulation;
    simulation->initialize( 0, NULL );
    simulation->finalize();
    delete simulation;
}

// Boot before Catch parses the command line, so Kokkos options such as
// --kokkos-num-threads are consumed here
int main( int argc, char* argv[] )
{
    boot_services( &argc, &argv );

    int result = Catch::Session().run( argc, argv );

    halt_services();

    return result;
}
