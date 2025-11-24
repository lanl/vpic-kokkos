//////////////////////////////////////////////////////
//
//   Ion-ion momentum equilibration
//
//////////////////////////////////////////////////////

#define DUMP_WITH_HDF5

#ifdef DUMP_WITH_HDF5
#ifndef VPIC_ENABLE_HDF5
#error "VPIC_ENABLE_HDF5" is required
#endif
#endif

//////////////////////////////////////////////////////

begin_globals {

  int restart_interval;
  int energies_interval;
  int fields_interval;
  int Ihydro_interval;
  int Iparticle_interval;
  int Bhydro_interval;
  int Bparticle_interval;
  int quota_check_interval;  // How frequently to check if quota exceeded

  int rtoggle;               // enables save of last 2 restart dumps for safety
  int write_restart;     // global flag for all to write restart files
  int write_end_restart; // global flag for all to write restart files
  
  double quota_sec;          // Run quota in seconds
  double b0;                 // B0
  double v_A;
  double topology_x;       // domain topology
  double topology_y;
  double topology_z;

  // Output variables
  DumpParameters hIdParams;
  DumpParameters hBdParams;
  std::vector<DumpParameters *> outputParams;

};

begin_initialization {
  
  // Use natural hybrid-PIC units:
  double ec   = 1.0;  // Charge normalization
  double mi   = 1.0;  // Mass normalization
  double mu0  = 1.0;  // Magnetic constanst
  double b0 = 1.0;    // Magnetic field. // Note for this problem B=0 (but we can still pick a reference field/units).
  double n0 = 1.0;    // Density

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

  } else { 

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
  int interval = 200; // int(num_step/100);
  int Ihydro_interval = interval;
  int Bhydro_interval = interval;
  int energies_interval = interval;
  int restart_interval = 0*interval;
  int fields_interval = 0*interval;
  int Iparticle_interval = 0*interval;
  int Bparticle_interval = 0*interval;
  int quota_check_interval = 100;


  ///////////////////////////////////////////////
  // Setup high level simulation parameters
  status_interval      = 10000; //num_step/10;
  sync_shared_interval = status_interval;
  clean_div_e_interval = status_interval;
  clean_div_b_interval = status_interval;

  global->restart_interval     = restart_interval;
  global->energies_interval    = energies_interval;
  global->fields_interval      = fields_interval;
  global->Ihydro_interval      = Ihydro_interval;
  global->Bhydro_interval      = Bhydro_interval;
  global->Iparticle_interval   = Iparticle_interval;
  global->Bparticle_interval   = Bparticle_interval;
  global->quota_check_interval = quota_check_interval;
  global->quota_sec            = quota_sec;
  global->rtoggle              = 0;

  global->b0  = b0;
  global->v_A  = v_A;

  global->topology_x  = topology_x;
  global->topology_y  = topology_y;
  global->topology_z  = topology_z;

 
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
  double t_final = 1000.0 * tau_ib;
  double dt = tau_ib / 50.0;
  num_step = (int)(t_final / dt);

  define_timestep( dt );
  
  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation

  sim_log( "***********************************************" );
  sim_log("* Topology:                       " << topology_x
    << " " << topology_y << " " << topology_z);
  sim_log ( "taui = " << taui );
  sim_log ( "num_step = " << num_step );
  sim_log ( "Lx = " << Lx/di );
  sim_log ( "Ly = " << Ly/di );
  sim_log ( "Lz = " << Lz/di );
  sim_log ( "Ti = " << Ti );
  sim_log ( "nx = " << nx );
  sim_log ( "ny = " << ny );
  sim_log ( "nz = " << nz );
  sim_log ( "nproc = " << nproc ()  );
  sim_log ( "nppc_backgrnd = " << nppc_backgrnd );
  sim_log ( "nppc_beam = " << nppc_beam );
  sim_log ( "b0 = " << b0 );
  sim_log ( "v_A = " << v_A );
  sim_log ( "di = " << di );
  sim_log ( "Ni_backgrnd = " << Ni_backgrnd );
  sim_log ( "Ni_beam = " << Ni_beam );
  sim_log ( "total # of particles = " << Ni_backgrnd + Ni_beam );
  sim_log ( "dt*wci = " << wci*dt );
  sim_log ( "energies_interval: " << energies_interval );
  sim_log ( "dx = " << Lx/(di*nx) );
  sim_log ( "dy = " << Ly/(di*ny) );
  sim_log ( "dz = " << Lz/(di*nz) );
  sim_log ( "n0 = " << n0 );
  sim_log ( "vbeam = " << vbeam );
  sim_log ( "nbeam = " << nbeam );
  sim_log ( "cvar0_ib = " << cvar0_ib );

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

  /*--------------------------------------------------------------------------
   * Set data output format
   *
   * This option allows the user to specify the data format for an output
   * dump.  Legal settings are 'band' and 'band_interleave'.  Band-interleave
   * format is the native storage format for data in VPIC.  For field data,
   * this looks something like:
   *
   *   ex0 ey0 ez0 div_e_err0 cbx0 ... ex1 ey1 ez1 div_e_err1 cbx1 ...
   *
   * Banded data format stores all data of a particular state variable as a
   * contiguous array, and is easier for ParaView to process efficiently.
   * Banded data looks like:
   *
   *   ex0 ex1 ex2 ... exN ey0 ey1 ey2 ...
   *
   *------------------------------------------------------------------------*/

  // global->hedParams.format = band;
  // sim_log ( "Electron species output format = band" );

  global->hIdParams.format = band;
  sim_log ( "Ion species output format = band" );

  global->hBdParams.format = band;
  sim_log ( "Beam ion species output format = band" );

  /*--------------------------------------------------------------------------
   * Set stride
   *
   * This option allows data down-sampling at output.  Data are down-sampled
   * in each dimension by the stride specified for that dimension.  For
   * example, to down-sample the x-dimension of the field data by a factor
   * of 2, i.e., half as many data will be output, select:
   *
   *   global->fdParams.stride_x = 2;
   *
   * The following 2-D example shows down-sampling of a 7x7 grid (nx = 7,
   * ny = 7.  With ghost-cell padding the actual extents of the grid are 9x9.
   * Setting the strides in x and y to equal 2 results in an output grid of
   * nx = 4, ny = 4, with actual extents 6x6.
   *
   * G G G G G G G G G
   * G X X X X X X X G
   * G X X X X X X X G         G G G G G G
   * G X X X X X X X G         G X X X X G
   * G X X X X X X X G   ==>   G X X X X G
   * G X X X X X X X G         G X X X X G
   * G X X X X X X X G         G X X X X G
   * G X X X X X X X G         G G G G G G
   * G G G G G G G G G
   *
   * Note that grid extents in each dimension must be evenly divisible by
   * the stride for that dimension:
   *
   *   nx = 150;
   *   global->fdParams.stride_x = 10; // legal -> 150/10 = 15
   *
   *   global->fdParams.stride_x = 8; // illegal!!! -> 150/8 = 18.75
   *------------------------------------------------------------------------*/

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

  /*--------------------------------------------------------------------------
   * Set output fields
   *
   * It is now possible to select which state-variables are output on a
   * per-dump basis.  Variables are selected by passing an or-list of
   * state-variables by name.  For example, to only output the x-component
   * of the electric field and the y-component of the magnetic field, the
   * user would call output_variables like:
   *
   *   global->fdParams.output_variables( ex | cby );
   *
   * NOTE: OUTPUT VARIABLES ARE ONLY USED FOR THE BANDED FORMAT.  IF THE
   * FORMAT IS BAND-INTERLEAVE, ALL VARIABLES ARE OUTPUT AND CALLS TO
   * 'output_variables' WILL HAVE NO EFFECT.
   *
   * ALSO: DEFAULT OUTPUT IS NONE!  THIS IS DUE TO THE WAY THAT VPIC
   * HANDLES GLOBAL VARIABLES IN THE INPUT DECK AND IS UNAVOIDABLE.
   *
   * For convenience, the output variable 'all' is defined:
   *
   *   global->fdParams.output_variables( all );
   *------------------------------------------------------------------------*/
  /* CUT AND PASTE AS A STARTING POINT
   * REMEMBER TO ADD APPROPRIATE GLOBAL DUMPPARAMETERS VARIABLE

   output_variables( all );

   output_variables( electric | div_e_err | magnetic | div_b_err |
                     tca      | rhob      | current  | rhof |
                     emat     | nmat      | fmat     | cmat );

   output_variables( current_density  | charge_density |
                     momentum_density | mass_density     | stress_tensor );
   */

  const uint32_t allfields      (0xffffffff);

  global->hIdParams.output_variables( allfields ); // current_density | charge_density | stress_tensor );
  global->hBdParams.output_variables( allfields ); // current_density | charge_density | stress_tensor );

#ifdef DUMP_WITH_HDF5
  // For writing XDMF file when using HDF5 dump
  // field_interval = global->fields_interval;
  // hydro_interval = global->Ihydro_interval;
    
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

  // Upon completion of the initialization, the following occurs:
  // - The synchronization error (tang E, norm B) is computed between domains
  //   and tang E / norm B are synchronized by averaging where discrepancies
  //   are encountered.
  // - The initial divergence error of the magnetic field is computed and
  //   one pass of cleaning is done (for good measure)
  // - The bound charge density necessary to give the simulation an initially
  //   clean divergence e is computed.
  // - The particle momentum is uncentered from u_0 to u_{-1/2}
  // - The user diagnostics are called on the initial state
  // - The physics loop is started
  //
  // The physics loop consists of:
  // - Advance particles from x_0,u_{-1/2} to x_1,u_{1/2}
  // - User particle injection at x_{1-age}, u_{1/2} (use inject_particles)
  // - User current injection (adjust field(x,y,z).jfx, jfy, jfz)
  // - Advance B from B_0 to B_{1/2}
  // - Advance E from E_0 to E_1
  // - User field injection to E_1 (adjust field(x,y,z).ex,ey,ez,cbx,cby,cbz)
  // - Advance B from B_{1/2} to B_1
  // - (periodically) Divergence clean electric field
  // - (periodically) Divergence clean magnetic field
  // - (periodically) Synchronize shared tang e and norm b
  // - Increment the time step
  // - Call user diagnostics
  // - (periodically) Print a status message

} //begin_initialization

#define should_dump(x)                                                  \
  (global->x##_interval>0 && remainder(step(), global->x##_interval) == 0)

begin_diagnostics {

  /*--------------------------------------------------------------------------
   * NOTE: YOU CANNOT DIRECTLY USE C FILE DESCRIPTORS OR SYSTEM CALLS ANYMORE
   *
   * To create a new directory, use:
   *
   *   dump_mkdir("full-path-to-directory/directoryname")
   *
   * To open a file, use: FileIO class
   *
   * Example for file creation and use:
   *
   *   // declare file and open for writing
   *   // possible modes are: io_write, io_read, io_append,
   *   // io_read_write, io_write_read, io_append_read
   *   FileIO fileIO;
   *   FileIOStatus status;
   *   status= fileIO.open("full-path-to-file/filename", io_write);
   *
   *   // formatted ASCII  output
   *   fileIO.print("format string", varg1, varg2, ...);
   *
   *   // binary output
   *   // Write n elements from array data to file.
   *   // T is the type, e.g., if T=double
   *   // fileIO.write(double * data, size_t n);
   *   // All basic types are supported.
   *   fileIO.write(T * data, size_t n);
   *
   *   // close file
   *   fileIO.close();
   *------------------------------------------------------------------------*/

  /*--------------------------------------------------------------------------
   * Data output directories
   * WARNING: The directory list passed to "global_header" must be
   * consistent with the actual directories where fields and species are
   * output using "field_dump" and "hydro_dump".
   *
   * DIRECTORY PATHES SHOULD BE RELATIVE TO
   * THE LOCATION OF THE GLOBAL HEADER!!!
   *------------------------------------------------------------------------*/

  global->restart_interval = 1000000;
  global->quota_sec = 23.5*3600.0;

  //  const int nsp=global->nsp;
  const int nx=grid->nx;
  const int ny=grid->ny;
  const int nz=grid->nz;

  /*--------------------------------------------------------------------------
   * Normal rundata dump
   *------------------------------------------------------------------------*/
  if(step()==0) {
    dump_mkdir("fields");
    dump_mkdir("hydro");
    dump_mkdir("rundata");
    // dump_mkdir("restore0");
    // dump_mkdir("restore1");  // 1st backup
    dump_mkdir("particle");
    dump_mkdir("rundata");

    dump_grid("rundata/grid");

    dump_materials("rundata/materials");
    dump_species("rundata/species");
    global_header("global", global->outputParams);
  } // if

  /*--------------------------------------------------------------------------
   * Normal rundata energies dump
   *------------------------------------------------------------------------*/
  if(should_dump(energies)) {
    dump_energies("rundata/energies", step() == 0 ? 0 : 1);
  } // if

  /*--------------------------------------------------------------------------
   * Ion species output
   *------------------------------------------------------------------------*/

  if(should_dump(Ihydro)) {
    hydro_dump("ion", global->hIdParams);
  }
  if(should_dump(Bhydro)) {
    hydro_dump("beam", global->hBdParams);
  }


  /*--------------------------------------------------------------------------
  * Time averaging
  *------------------------------------------------------------------------*/

  //#include "time_average.cxx"
  //#include "time_average_cori.cxx"

  // Shut down simulation when wall clock time exceeds global->quota_sec.
  // Note that the mp_elapsed() is guaranteed to return the same value for all
  // processors (i.e., elapsed time on proc #0), and therefore the abort will
  // be synchronized across processors. Note that this is only checked every
  // few timesteps to eliminate the expensive mp_elapsed call from every
  // timestep. mp_elapsed has an ALL_REDUCE in it!


  // if ( (step()>0 && global->quota_check_interval>0
  //       && (step() & global->quota_check_interval)==0 ) || (global->write_end_restart) ) {

  //   if ( (global->write_end_restart) ) {
  //     global->write_end_restart = 0; // reset restart flag

  //     //   if( uptime() > global->quota_sec ) {
  //     sim_log( "Allowed runtime exceeded for this job.  Terminating....\n");
  //     double dumpstart = uptime();

  //     if(!global->rtoggle) {
  //       global->rtoggle = 1;
  //       //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
  //       checkpt("restore1", 0);
  //       //    } END_TURNSTILE;
  //     } else {
  //       global->rtoggle = 0;
  //       //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
  //       checkpt("restore0", 0);
  //       //    } END_TURNSTILE;
  //     } // if

  //     mp_barrier(  ); // Just to be safe
  //     sim_log( "Restart dump restart completed." );
  //     double dumpelapsed = uptime() - dumpstart;
  //     sim_log("Restart duration "<< dumpelapsed);
  //     exit(0); // Exit or abort?                                                                                
  //   }
  //   //    } 
  //   if( uptime() > global->quota_sec ) global->write_end_restart = 1;
  // }


} // end diagnostics

// ***********  PARTICLE INJECTION  - OPEN BOUNDARY *********
begin_particle_injection {
} // end particle injection

//*******************  CURRENT INJECTION ********************
begin_current_injection {
} // end current injection

//*******************  FIELD INJECTION **********************
begin_field_injection {
}  // end field injection


//*******************  COLLISIONS ***************************
begin_particle_collisions {
} // end collisions

