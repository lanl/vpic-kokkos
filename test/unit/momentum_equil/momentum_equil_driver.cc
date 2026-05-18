#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"

/*
Test is based off of examples/stopping simulation that models
the momentum and temperature equilibration of a beam and 
initially stationary background ion species. The beam has 1/10
the density of the background population. Both populations
have an initial temperature of 5 keV and the beam has an
initial velocity of 655 km/s (10.33*v_thermal).

The test case is originally from Rambo and Procassini 1995.

Only the first 5ps are simulated at high resolution. In the 
example/stopping/stopping.cxx script, the simulation shows
equilibration occurs over 100ps.

The test compares data to the energies file output from the
example simulation that was confirmed to match expected results.

todo: run test for var_wt = true and false
*/

#include "deck/wrapper.h"

#include "src/species_advance/species_advance.h"
#include "src/vpic/vpic.h"

#include "compare_energies.h"

std::string energy_file_name = "./energies";
std::string beam_energy_file_name = BEAM_ENERGY_FILE;

begin_globals {
  int energies_interval;
  int hydro_interval;

  // Output variables
  DumpParameters hIdParams;
  DumpParameters hBdParams;
  std::vector<DumpParameters *> outputParams;
};

void vpic_simulation::user_initialization( int num_cmdline_arguments,
                                           char ** cmdline_argument )
{

  // Use natural hybrid-PIC units:
  double ec  = 1.0;  // Charge normalization
  double mi  = 1.0;  // Mass normalization
  double mu0 = 1.0;  // Magnetic constanst
  double b0  = 1.0;  // Magnetic field. // Note for this problem B=0 (but we can still pick a reference field/units).
  double n0  = 1.0;  // Density

  // Derived normalization parameters:
  double v_A = b0/sqrt(mu0*n0*mi); // Alfven velocity
  double wci = ec*b0/mi;          // Cyclotron freq.
  double di = v_A/wci;            // Ion skin-depth

  // Initial conditions for model:
  double qi = 1.0;          // Particle charge
  double vbeam = 10.33;     // Beam velocity (655 km/s)
  double nbeam = 0.1;       // Beam density.
  double Ti = 1.0;          // Ion temperature
  double c_s = 1.0;         // Electron sound speed.
  double pert = 0.02;       // Size of density perturbation.
  double Lx = 1;            // Size of domain.
  
  double eta = 0.0;         // Plasma resistivity.
  double hypereta = 0.0;    // Plasma hyper-resistivity.

  // Derived quantities for model:
  double vthi = sqrt(Ti/mi);// Ion thermal velocity

  // Numerical parameters
  double taui    = 50;      // Simulation run time in wci^-1.
  double quota   = 2.0;     // run quota in hours
  double quota_sec = quota*3600;  // Run quota in seconds
  
  double Ly    = 1.0*di;    // size of box in y dimension
  double Lz    = 1.0*di;    // size of box in z dimension

  double nx = 1;
  double ny = 1;
  double nz = 1;
  
  double topology_x = 1; // Number of domains in x, y, and z
  double topology_y = 1;
  double topology_z = 1;

  // Derived numerical parameters
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;

  // Calculate particle weights
  bool var_wt = true;
  double nppc = 10000;
  double nppc_backgrnd, nppc_beam;
  double qi_backgrnd, qi_beam;
  double Ni_backgrnd, Ni_beam;
  double Np_backgrnd, Np_beam;

  if (var_wt) { // equal ppc, unequal weights, weight = nV/N

    nppc_backgrnd = nppc;
    Np_backgrnd = n0*Lx*Ly*Lz;
    Ni_backgrnd = nppc*nx*ny*nz;
    Ni_backgrnd = trunc_granular(Ni_backgrnd,nproc());
    qi_backgrnd = ec*Np_backgrnd/Ni_backgrnd;

    nppc_beam = nppc;
    Np_beam = nbeam*Lx*Ly*Lz;
    Ni_beam = nppc_beam*nx*ny*nz;
    Ni_beam = trunc_granular(Ni_beam,nproc());
    qi_beam = ec*Np_beam/Ni_beam;

  } else { // var_wt=false

    nppc_backgrnd = nppc;                              // Average number of macro particle per cell per species 
    Np_backgrnd = n0*Lx*Ly*Lz;                         // Total number of physical background ions
    Ni_backgrnd = nppc_backgrnd*nx*ny*nz;              // Total macroparticle ions in box
    Ni_backgrnd = trunc_granular(Ni_backgrnd,nproc()); // Make it divisible by number of processors
    qi_backgrnd = ec*Np_backgrnd/Ni_backgrnd;          // Charge per macro ion

    nppc_beam = nppc * nbeam / n0;
    Np_beam = nbeam*Lx*Ly*Lz;
    Ni_beam = nppc_beam*nx*ny*nz;
    Ni_beam = trunc_granular(Ni_beam,nproc());
    qi_beam = ec*Np_beam/Ni_beam;

  } // endif(var_wt)

  // Intervals for output
  int interval = 100;
  global->energies_interval = interval;
  global->hydro_interval    = interval;

  //////////////////////////////////////////////////////////////////////////////
  // Setup the grid

  // Setup basic grid parameters
  define_units(1.0, 1.0);
  // define_timestep( dt );

  // Define the grid
  define_periodic_grid(  -0.5*Lx, -0.5*Ly, -0.5*Lz,    // Low corner
                          0.5*Lx,  0.5*Ly, 0.5*Lz,     // High corner
                         nx, ny, nz,             // Resolution
                         topology_x, topology_y, topology_z); // Topology

  //  grid->te = Te;
  //  grid->den = 1.0;
  grid->eta = eta;
  //  grid->hypereta = hypereta;

  // ***** Set Field Boundary Conditions *****
  sim_log("Periodic boundaries");
  // Do nothing - periodic is default.

  // ***** Set Particle Boundary Conditions *****
  // Do nothing - periodic is default.
 
  //////////////////////////////////////////////////////////////////////////////
  // Setup materials
  sim_log("Setting up materials. ");
  define_material( "vacuum", 1 );

  
  //////////////////////////////////////////////////////////////////////////////                                                                                                                                                                                                       // Finalize Field Advance
  define_field_array(NULL); // second argument is damp, default to 0
  sim_log("Finalized Field Advance");
 

  //////////////////////////////////////////////////////////////////////////////
  // Setup the species
  sim_log("Setting up species. ");
  double nmax_backgrnd = 1.5*Ni_backgrnd/nproc();
  double nmovers_backgrnd = 0.1*nmax_backgrnd;
  double nmax_beam = 1.5*Ni_beam/nproc();
  double nmovers_beam = 0.1*nmax_beam;

  double sort_interval = 10000;  // How often to sort particles (1 cell so dont sort)
  double sort_method = 1;        // 0=in place and 1=out of place
  species_t *ion = define_species("ion", ec, mi, nmax_backgrnd, nmovers_backgrnd, sort_interval, sort_method);
  species_t *beam = define_species("beam", ec, mi, nmax_beam, nmovers_beam, sort_interval, sort_method);

  //////////////////////////////////////////////////////////////////////////////
  // Define Coulomb collisions
  sim_log("Setting up collisions. ");
  int ncoll_coulomb = 1;
  double ln_Lambda = 10.0;
  double lnL_8pi = ln_Lambda / (8.0 * M_PI);

  // cvar0 = q1^2 * q2^2 * lnL / (8 * pi)
  double cvar0_ib = (ion->q * ion->q) * (beam->q *  beam->q) * lnL_8pi;
  double cvar0_ii = (ion->q * ion->q) * (ion->q *  ion->q) * lnL_8pi;
  double cvar0_bb = (beam->q * beam->q) * (beam->q * beam->q) * lnL_8pi;

  define_collision_op(takizuka_abe("ta_bi", ion,  beam, cvar0_ib, ncoll_coulomb, var_wt));
  // define_collision_op(takizuka_abe("ta_ii", ion,  ion, cvar0_ii, ncoll_coulomb, var_wt));
  // define_collision_op(takizuka_abe("ta_bb", beam, beam, cvar0_bb, ncoll_coulomb, var_wt));

  ion->last_indexed = -1;
  beam->last_indexed = -1;

  ///////////////////////////////////////////////
  // Setup time steps

  // Determine the time step
  double tau_ib = 1.0 / (std::sqrt(2) * cvar0_ib);
  double t_final = 1000.0 * tau_ib * 0.05; // ony simulate first 5%
  double dt = tau_ib / 50.0;
  num_step = (int)(t_final / dt);

  define_timestep( dt );
  
  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation

  // sim_log( "***********************************************" );
  // sim_log("* Topology:                       " << topology_x
  //   << " " << topology_y << " " << topology_z);
  // sim_log ( "taui = " << taui );
  // sim_log ( "num_step = " << num_step );
  // sim_log ( "Lx = " << Lx/di );
  // sim_log ( "Ly = " << Ly/di );
  // sim_log ( "Lz = " << Lz/di );
  // sim_log ( "Ti = " << Ti );
  // sim_log ( "nx = " << nx );
  // sim_log ( "ny = " << ny );
  // sim_log ( "nz = " << nz );
  // sim_log ( "nproc = " << nproc ()  );
  // sim_log ( "nppc_backgrnd = " << nppc_backgrnd );
  // sim_log ( "nppc_beam = " << nppc_beam );
  // sim_log ( "b0 = " << b0 );
  // sim_log ( "v_A = " << v_A );
  // sim_log ( "di = " << di );
  // sim_log ( "Ni_backgrnd = " << Ni_backgrnd );
  // sim_log ( "Ni_beam = " << Ni_beam );
  // sim_log ( "total # of particles = " << Ni_backgrnd + Ni_beam );
  // sim_log ( "dt*wci = " << wci*dt );
  // sim_log ( "dx = " << Lx/(di*nx) );
  // sim_log ( "dy = " << Ly/(di*ny) );
  // sim_log ( "dz = " << Lz/(di*nz) );
  // sim_log ( "n0 = " << n0 );
  // sim_log ( "vbeam = " << vbeam );
  // sim_log ( "nbeam = " << nbeam );
  // sim_log ( "cvar0_ib = " << cvar0_ib );

  ////////////////////////////
  // Load fields
  sim_log( "Loading fields" );

  // Note: everywhere is a region that encompasses the entire simulation                                                                                                                   
  // In general, regions are specied as logical equations (i.e. x>0 && x+y<2) 
  set_region_field( everywhere, 0, 0, 0, 0.0, 0, 0);
  set_region_eta_multipliers( everywhere, 1, 1, 0 );
  set_region_te( everywhere, 1.0);

  // LOAD PARTICLES
  sim_log( "Loading particles" );

  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);
  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);
  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);

  repeat( Ni_backgrnd ) {
    double x, y, z, r, ux, uy, uz, d0;
    x = uniform( rng(0), xmin, xmax );
    y = uniform( rng(0), ymin, ymax );
    z = uniform( rng(0), zmin, zmax );
    
    ux = normal( rng(0), 0, vthi );                                                                                                                                             
    uy = normal( rng(0), 0, vthi );
    uz = normal( rng(0), 0, vthi );

#ifdef VARIABLE_CHARGE
    inject_particle( ion, x, y, z, ux, uy, uz, qi_backgrnd, 0, 0, qi );
#else
    inject_particle( ion, x, y, z, ux, uy, uz, qi_backgrnd, 0, 0 );
#endif
  }

  // Low-density ion beam
  repeat( Ni_beam ) {
    double x, y, z, r, ux, uy, uz, d0;
    x = uniform( rng(0), xmin, xmax );
    y = uniform( rng(0), ymin, ymax );
    z = uniform( rng(0), zmin, zmax );
    
    ux = normal( rng(0), 0.0, vthi );                                                                                                                                             
    uy = normal( rng(0), 0.0, vthi );
    uz = normal( rng(0), vbeam, vthi );

#ifdef VARIABLE_CHARGE
    inject_particle( beam, x, y, z, ux, uy, uz, qi_beam, 0, 0, qi );
#else
    inject_particle( beam, x, y, z, ux, uy, uz, qi_beam, 0, 0 );
#endif
  }

  sim_log( "Finished loading particles" );

  /*--------------------------------------------------------------------------
   * New dump definition
   *------------------------------------------------------------------------*/

  global->hIdParams.format = band;
  sim_log ( "Ion species output format = band" );

  global->hBdParams.format = band;
  sim_log ( "Beam ion species output format = band" );

  dump_mkdir("hydro");

  // relative path to ion species data from global header
  sprintf(global->hIdParams.baseDir, "hydro");
  dump_mkdir(global->hIdParams.baseDir);
  sprintf(global->hIdParams.baseFileName, "ionhydro");
  global->hIdParams.stride_x = 1;
  global->hIdParams.stride_y = 1;
  global->hIdParams.stride_z = 1;
  global->outputParams.push_back(&global->hIdParams);

  sim_log ( "Ion species x-stride " << global->hIdParams.stride_x );
  sim_log ( "Ion species y-stride " << global->hIdParams.stride_y );
  sim_log ( "Ion species z-stride " << global->hIdParams.stride_z );

  // relative path to ion species data from global header
  sprintf(global->hBdParams.baseDir, "hydro");
  dump_mkdir(global->hBdParams.baseDir);
  sprintf(global->hBdParams.baseFileName, "beamhydro");
  global->hBdParams.stride_x = 1;
  global->hBdParams.stride_y = 1;
  global->hBdParams.stride_z = 1;
  global->outputParams.push_back(&global->hBdParams);

  sim_log ( "Beam species x-stride " << global->hBdParams.stride_x );
  sim_log ( "Beam species y-stride " << global->hBdParams.stride_y );
  sim_log ( "Beam species z-stride " << global->hBdParams.stride_z );

  const uint32_t allfields      (0xffffffff);
  global->hIdParams.output_variables( allfields );
  global->hBdParams.output_variables( allfields );

#ifdef DUMP_WITH_HDF5
  enable_hdf5_dump();
#endif

  /*--------------------------------------------------------------------------
   * Convenience functions for simlog output
   *------------------------------------------------------------------------*/

  char varlist[512];
  create_hydro_list(varlist, global->hIdParams);
  sim_log ( "Ion species variable list: " << varlist );

  create_hydro_list(varlist, global->hBdParams);
  sim_log ( "Beam species variable list: " << varlist );

  sim_log("*** Finished with user-specified initialization ***");

  sim_log(" Test compares " << energy_file_name << " to " << beam_energy_file_name);

}

#define should_dump(x)                                                  \
  (global->x##_interval>0 && remainder(step(), global->x##_interval) == 0)

void vpic_simulation::user_diagnostics() 
{
  if(step()==0) dump_mkdir("hydro");
  if(should_dump(energies)) dump_energies(energy_file_name.c_str(), 1);
  if(should_dump(hydro)) {
    hydro_dump("ion", global->hIdParams);
    hydro_dump("beam", global->hBdParams);
  }
}

TEST_CASE( "Check if momentum equil. gives correct velocity and temperature (within tol)", "[energy]" )
{
    // Before we run this, we must make sure we remove the energy file
    std::ofstream ofs;
    ofs.open(energy_file_name, std::ofstream::out | std::ofstream::trunc);
    ofs.close();

    // Init and run sim
    vpic_simulation simulation = vpic_simulation();
    simulation.initialize( 0, NULL );
    while( simulation.advance() );
    simulation.finalize();

    if( world_rank==0 ) log_printf( "normal exit\n" );

    // Compare energies to make sure everything worked out OK (within 1%)
    const unsigned short beam_mask = 0b011000000;
    // const unsigned short ion_mask = 0b011000000;

    float tolerance = 0.1;

    REQUIRE(
            test_utils::compare_energies(energy_file_name, beam_energy_file_name,
                tolerance, beam_mask, test_utils::FIELD_ENUM::Individual, 1, "err_beam.out", 0, -1)
           );

    // REQUIRE(
    //         test_utils::compare_energies(energy_file_name, beam_energy_file_name,
    //             tolerance, ion_mask, test_utils::FIELD_ENUM::Sum, 1, "err_ion.out", 3, -1)
    //        );
}

begin_particle_injection {
  // No particle injection for this simulation
}

begin_current_injection {
  // No current injection for this simulation
}

begin_field_injection {
  // No field injection for this simulation
}

begin_particle_collisions{
  // No collisions for this simulation
}

// Manually implement catch main
int main( int argc, char* argv[] )
{

    // Setup
    boot_services( &argc, &argv );

    int result = Catch::Session().run( argc, argv );

    // clean-up...
    halt_services();

    return result;
}
