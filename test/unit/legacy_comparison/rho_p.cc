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
    int nstep = 100;

    define_units( 1, 1 );
    define_timestep( 1 );
    define_periodic_grid( 0, 0, 0,   // Grid low corner
            L, L, L,   // Grid high corner
            1, 1, 1,   // Grid resolution
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
    species_t * sp;
   #if defined(FIELD_IONIZATION)
    grid->lambda  = 1; // these need to de defined as nonzero when 
    grid->t_to_SI = 1; // field ionization is enabled 
    grid->l_to_SI = 1; 
    grid->q_to_SI = 1;
    grid->m_to_SI = 1;
    Kokkos::View<double*> ionization_energy("my_kokkos_view", 1);
    double ionization_energy_values[] = {0}; // in eV
    ionization_energy(0) = ionization_energy_values[0];
    sp = define_species( "test_species", 1.,ionization_energy, 1,0,0, 1., npart, npart, 0, 0);
    // electron needs to be defined when  FI is enabled
    species_t * electron = define_species("electron",-1.,ionization_energy, 0,0,0, 1., npart, npart, 0, 0); 
   #else
    sp = define_species( "test_species", 1., 1., npart, npart, 0, 0 );
   #endif

    int failed = 0;

    for (int i = 0; i < npart; i++)
    {
        float x = uniform( rng(0), 0, L);
        float y = uniform( rng(0), 0, L);
        float z = uniform( rng(0), 0, L);

        // Put two sets of particle in the exact same space
        inject_particle( sp , x, y, z, 0., 0., 0., 1., 0., 0);
    }

    // Make sure kokkos views have correct data
    field_array->copy_to_device();
    LIST_FOR_EACH( sp_temp, species_list ) {
      sp_temp->copy_to_device();
    }

    #if defined(FIELD_IONIZATION)
    advance_p( sp, interpolator_array, field_array, species_list );
    #else
    advance_p( sp, interpolator_array, field_array );
    #endif

    // Call both functions
    k_accumulate_rho_p( field_array, sp );
    accumulate_rho_p( field_array, sp );

    // Copy the data back to host view
    Kokkos::deep_copy(field_array->k_f_h, field_array->k_f_d);

    // Update to reflect the hydro_p test
    // The Kokkos version uses a scatter access that should produce a more
    // accurate sum, so we have a large tolerance here
    std::cout << std::scientific << std::setprecision(12) << std::endl;
    float abstol = 1e3*std::numeric_limits<float>::min();
    float reltol = 1e-4;

    std::cout << "Absolute tolerance is " << abstol << std::endl;
    std::cout << "Relative tolerance is " << reltol << std::endl;

    // This is how many pipelines there are inside the array
    for (int i = 0; i < grid->nv; i++)
    {
        double a = field_array->k_f_h(i, field_var::rhof);
        double b = field_array->f[i].rhof;
        double rel_err = (a-b)/a;
        if (abs(rel_err) > reltol && abs(a)>abstol && abs(b)>abstol)
        {
            std::cout << " Failed at " << i << " with relative error " << rel_err << " from k_field and field " << a << " " << b << std::endl;
            failed++;
        }
    }
    if( failed )
    {  std::cout << "FAIL" << std::endl;
    }
    REQUIRE_FALSE(failed);

    std::cout << "pass" << std::endl;
}

TEST_CASE( "vectors can be sized and resized", "[vector]" ) {

    std::vector<int> v( 5 );

    REQUIRE( v.size() == 5 );
    REQUIRE( v.capacity() >= 5 );

    //boot_checkpt(NULL, NULL);
    int pargc = 0;
    char str[] = "bin/vpic";
    char **pargv = (char **) malloc(sizeof(char **));
    pargv[0] = str;
    //serial.boot(&pargc, &pargv);
    //thread.boot(&pargc, &pargv);
    boot_services( &pargc, &pargv );

    SECTION( "resizing bigger changes size and capacity" )
    {
        int num_particles = 64;
        v.resize( 10 );

        REQUIRE( v.size() == 10 );
        REQUIRE( v.capacity() >= 10 );

        // initialize all the variables
        //particle_t *p_arr = particle_arr(num_particles, 1);
        //species_t *sp = make_species(p_arr, 1, 1);

        vpic_simulation* simulation = new vpic_simulation;
        simulation->initialize( pargc, pargv );

        simulation->finalize();
        delete simulation;
        if( world_rank==0 ) log_printf( "normal exit\n" );

        halt_mp();
    }
}
