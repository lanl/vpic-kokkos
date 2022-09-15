// Modification of the rho_p test to test the kokkos port of hydro_p.
// Written by Scott V. Luedtke, XCP-6, September 14, 2022

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"

#include <iostream>

#include "src/species_advance/species_advance.h"
#include "src/vpic/vpic.h"

void vpic_simulation::user_diagnostics() {}

void
vpic_simulation::user_initialization( int num_cmdline_arguments,
                                      char ** cmdline_argument )
{
    double L  = 1e2;
    int npart = 1000000;

    std::cout << "Initializing grid and fields" << std::endl;

    define_units( 1, 1 );
    define_timestep( 1 );
    define_periodic_grid( 0, 0, 0,   // Grid low corner
            L, L, L,   // Grid high corner
            1, 1, 1,   // Grid resolution
            1, 1, 1 ); // Processor configuration
    define_material( "vacuum", 1.0, 1.0, 0.0 );
    define_field_array();

    hydro_array_t *hydro_array_legacy = new_hydro_array(grid);
    hydro_array_t *hydro_array_kokkos = new_hydro_array(grid);

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

    std::cout << "Initializing particles" << std::endl;
    species_t * sp_temp;
    species_t * sp = define_species( "test_species", 1., 1., npart, npart, 0, 0 );

    int failed = 0;

    for (int i = 0; i < npart; i++)
    {
        float x = uniform( rng(0), 0, L);
        float y = uniform( rng(0), 0, L);
        float z = uniform( rng(0), 0, L);

        // Some random relativistic momemtum
        float px = normal( rng(0), 0, 0.5);
        float py = normal( rng(0), 0, 1);
        float pz = normal( rng(0), 0, 2);
        
        inject_particle( sp , x, y, z, px, py, pz, 1., 0., 0);
    }

    // Make sure kokkos views have correct data
    field_array->copy_to_device();
    LIST_FOR_EACH( sp_temp, species_list ) {
      sp_temp->copy_to_device();
    }

    // Call both functions
    std::cout << "Accumulating legacy" << std::endl;
    accumulate_hydro_p( hydro_array_legacy, sp, interpolator_array );

    std::cout << "Accumulating kokkos" << std::endl;
    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;

    k_hydro_d_t hydro_view("hydro_d", sp->g->nv);

    accumulate_hydro_p_kokkos(
        particles,
        particles_i,
        hydro_view,
        interpolators_k,
        sp
    );

    k_hydro_d_t::HostMirror hydro_view_h("hydro_d_h", sp->g->nv);
    Kokkos::deep_copy( hydro_view_h , hydro_view);

    for(int i=0; i<hydro_view_h.extent(0); i++) {
      hydro_array_kokkos->h[i].jx = hydro_view_h(i, hydro_var::jx);
      hydro_array_kokkos->h[i].jy = hydro_view_h(i, hydro_var::jy);
      hydro_array_kokkos->h[i].jz = hydro_view_h(i, hydro_var::jz);
      hydro_array_kokkos->h[i].rho = hydro_view_h(i, hydro_var::rho);
      hydro_array_kokkos->h[i].px = hydro_view_h(i, hydro_var::px);
      hydro_array_kokkos->h[i].py = hydro_view_h(i, hydro_var::py);
      hydro_array_kokkos->h[i].pz = hydro_view_h(i, hydro_var::pz);
      hydro_array_kokkos->h[i].ke = hydro_view_h(i, hydro_var::ke);
      hydro_array_kokkos->h[i].txx = hydro_view_h(i, hydro_var::txx);
      hydro_array_kokkos->h[i].tyy = hydro_view_h(i, hydro_var::tyy);
      hydro_array_kokkos->h[i].tzz = hydro_view_h(i, hydro_var::tzz);
      hydro_array_kokkos->h[i].tyz = hydro_view_h(i, hydro_var::tyz);
      hydro_array_kokkos->h[i].tzx = hydro_view_h(i, hydro_var::tzx);
      hydro_array_kokkos->h[i].txy = hydro_view_h(i, hydro_var::txy);
    }


    synchronize_hydro_array( hydro_array_legacy );
    synchronize_hydro_array( hydro_array_kokkos );

    // Copy the data back to host view
    Kokkos::deep_copy(field_array->k_f_h, field_array->k_f_d);

    // If two numbers are very close to zero, the relative error can get big
    // just from numerical rounding
    // The relative tolerance might seem high, but summation can accumulate
    // some pretty significant errors.  With much more parallelism on GPUs in
    // the Kokkos implementation, I expect higher accuracy.
    // On my desktop CPU this passed with reltol = 1e-18.
    std::cout << std::scientific << std::setprecision(12) << std::endl;
    float abstol = 1e3*std::numeric_limits<float>::min();
    float reltol = 1e-4;

    std::cout << "Absolute tolerance is " << abstol << std::endl;
    std::cout << "Relative tolerance is " << reltol << std::endl;

    // This is how many pipelines there are inside the array
    for (int i = 0; i < grid->nv; i++)
    {
        double a = hydro_array_legacy->h[i].jx;
        double b = hydro_array_kokkos->h[i].jx;
        double rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].jy;
        b = hydro_array_kokkos->h[i].jy;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].jz;
        b = hydro_array_kokkos->h[i].jz;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].rho;
        b = hydro_array_kokkos->h[i].rho;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].px;
        b = hydro_array_kokkos->h[i].px;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].py;
        b = hydro_array_kokkos->h[i].py;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].ke;
        b = hydro_array_kokkos->h[i].ke;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].txx;
        b = hydro_array_kokkos->h[i].txx;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].tyy;
        b = hydro_array_kokkos->h[i].tyy;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].txx;
        b = hydro_array_kokkos->h[i].txx;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].tyz;
        b = hydro_array_kokkos->h[i].tyz;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].tzx;
        b = hydro_array_kokkos->h[i].tzx;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }

        a = hydro_array_legacy->h[i].txy;
        b = hydro_array_kokkos->h[i].txy;
        rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from hydro legacy and hydro kokkos " << a << " " << b << std::endl;
            failed++;
        }
    }
    if( failed )
    {  std::cout << "FAIL" << std::endl;
    }
    REQUIRE_FALSE(failed);

    std::cout << "pass" << std::endl;
}
