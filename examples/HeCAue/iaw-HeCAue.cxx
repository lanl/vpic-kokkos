//////////////////////////////////////////////////////
//
//   Landau-damped Ion Acoustic Wave
//
//////////////////////////////////////////////////////

//#define NUM_TURNSTILES 16384

//////////////////////////////////////////////////////
struct IonParams {
  const char* name;  // e.g., "ion", "ion2", "C+", ...
  double n;          // physical number density (like ni, n2)
  double nppc;       // computational particles per cell used to build np
  double q;          // charge
  double m;          // mass
  double vt;         // thermal speed (stddev)
  double v0;         // drift in x (mean)
};

    std::vector<IonParams> ion_cfg = {
       {"He", 1.0, 10000, 2.0, 4.0, 1.5811388300841898 , 0.0}
      ,{"C",  0.1, 1000, 6.0, 12., 1.5275252316519468 , 0.6462}
      ,{"Au", 1.0, 10000, 30.,197., 0.07124704998790964, 0.9693}
    };

begin_globals {

  int restart_interval;
  int energies_interval;
  int fields_interval;
  int ehydro_interval;
  int Hhydro_interval;
  int eparticle_interval;
  int Hparticle_interval;
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
  DumpParameters fdParams;
  DumpParameters hedParams;
  DumpParameters hHdParams;
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
  double Ti = 1.0;      // Ion temperature
  double gamma = 1.0;//5.0/3.0;   // Ratio of specific heats.
  double c_s = 1.0;         // Electron sound speed.
  double pert = 0.02;       // Size of density perturbation.
  double Lx = 1;           // Size of domain.
  double kx = 2.0*M_PI/Lx;  // Wavenumber of perturbation.
  
  double eta = 0.0;         // Plasma resistivity.
  double hypereta = 0.0;    // Plasma hyper-resistivity.

  
  // Derived quantities for model:
  //double Te = c_s/(gamma);  // Electron temperature.
  double vthi = sqrt(Ti/mi);// Ion thermal velocity

  // Numerical parameters
  double taui    = 50;      // Simulation run time in wci^-1.
  double quota   = 2.0;     // run quota in hours
  double quota_sec = quota*3600;  // Run quota in seconds
  
  double Ly    = 1.0; //*di;    // size of box in y dimension
  double Lz    = 1.0; //*di;    // size of box in z dimension

  double nx = 1;
  double ny = 1;
  double nz = 1;

  double nppc  = 1000;    // Average number of macro particle per cell per species 
  
  double topology_x = 1; // Number of domains in x, y, and z
  double topology_y = 1;
  double topology_z = 1;


  // Derived numerical parameters
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;


  double Ni  = nppc*nx*ny*nz;       // Total macroparticle ions in box
  double Np  = n0*Lx*Ly*Lz;         // Total number of physical background ions
  Ni = trunc_granular(Ni,nproc());  // Make it divisible by number of processors
  double qi = ec*Np/Ni;             // Charge per macro ion

  
  // Determine the time step
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);  // courant length
  double dt = 4e-4;                               // time step

  double sort_interval = 1;  // How often to sort particles
  
  // Intervals for output
  num_step = int(taui/(wci*dt));
  int restart_interval = 20000;
  int energies_interval = 200;
  int interval = int(num_step/100);//0.2/(wci*dt));
  int fields_interval = interval;
  int ehydro_interval = interval;
  int Hhydro_interval = interval;
  int eparticle_interval = 0*interval;
  int Hparticle_interval = 0*interval;
  int quota_check_interval     = 100;


  ///////////////////////////////////////////////
  // Setup high level simulation parameters
  status_interval      = num_step/100;
  sync_shared_interval = status_interval;
  clean_div_e_interval = status_interval;
  clean_div_b_interval = status_interval;

  global->restart_interval     = restart_interval;
  global->energies_interval    = energies_interval;
  global->fields_interval      = fields_interval;
  global->ehydro_interval      = ehydro_interval;
  global->Hhydro_interval      = Hhydro_interval;
  global->eparticle_interval   = eparticle_interval;
  global->Hparticle_interval   = Hparticle_interval;
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
  define_units(1.0, 1.0);//c, eps0 );
  define_timestep( dt );

  // Define the grid
  define_periodic_grid(  -0.5*Lx, -0.5*Ly, -0.5*Lz,    // Low corner
                          0.5*Lx,  0.5*Ly, 0.5*Lz,     // High corner
                         nx, ny, nz,             // Resolution
                         topology_x, topology_y, topology_z); // Topology


  //define_material( "vacuum", 1.0, 1.0, 0.0 );
  //define_field_array();
  double me = 1.0/1837.0; 
  double ne = 32.6;
  double vte = 42.8485709398;  
  double Te = 1.0;
  fluid_species_t * sp_fl = define_fluid_species( "test_fluid_species", -ec, me/mi );
  set_region_fluid( everywhere, "test_fluid_species", ne, Te, ne*Te ); //density, tmperature, pressure
  
  //  grid->te = Te;
  //  grid->den = 1.0;
  grid->eta = eta;
  //  grid->hypereta = hypereta;
  //  grid->gamma = gamma;

  //  grid->nsub = 1; // Number of substeps for field solve.
  //  grid->nsm = 2;  // Number of binomial smoothing passes (to fields & moments).
  //  grid->nsmb = 0; // Timesteps between additional smooths of magnetic field (0 is off).

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
  double nmax = 1.5*Ni/nproc();
  double nmovers = 0.1*nmax;
  double sort_method = 1;   // 0=in place and 1=out of place
  //species_t *ion = define_species("ion", ec, mi, nmax, nmovers, sort_interval, sort_method);

  std::vector<species_t*>      ions;
  std::vector<hydro_array_t*>  hydro_ions;  
  std::vector<double> n_i, m_i;
  
  int num_is = ion_cfg.size();

  ions.reserve(num_is);
  hydro_ions.reserve(num_is);
  n_i.reserve(num_is);
  m_i.reserve(num_is);

  const double N3 = nx*ny*nz;
  for (const auto& s : ion_cfg) {
      const double np_i  = s.nppc * N3;             // computational particle count
      const double w_i   = s.n / np_i;              // weight per particle (fixed-wt path)
      //printf("#w_i=%e\n",w_i);
      n_i.push_back(s.n);
      m_i.push_back(s.m);

      species_t* ion_s   = define_species(s.name, s.q, s.m, (int)np_i, 1, 0, 0);
      ions.push_back(ion_s);
      hydro_array_t* ha  = new_hydro_array(grid);
      hydro_ions.push_back(ha);
#if defined(varWT)
#else
  // ---- fixed-weight injection ----
      repeat( (int)np_i ) inject_particle( ion_s,
					   uniform(rng(0), grid->x0, grid->x1),
					   uniform(rng(0), grid->y0, grid->y1),
					   uniform(rng(0), grid->z0, grid->z1),
					   normal(rng(0), s.v0, s.vt),
					   normal(rng(0), 0.0 , s.vt),
					   normal(rng(0), 0.0 , s.vt),
					   w_i, 0, 0 );
#endif

      ion_s->copy_to_device();
  }      
  
  auto M_ln_Lamda = 10;
  double cvar0 = M_ln_Lamda/(8.0*M_PI);       //in SI: (e^4*n0*Lambda)/(8*pi*eps0^2*m_e^2*c^3)
  // --> internal operator multiplies by dt!
  //int sort_interval = 1;  
  int ncoll = (int) sort_interval;  // How frequently to do collisions
  
  
  bool var_wt = false;
  // all unique ion ↔ ion pairs
  for (size_t i = 0; i < ions.size(); ++i) {
      for (size_t j = i; j < ions.size(); ++j) {
	  std::string cname = std::string("ta_coll_") + ions[i]->name + "_" + ions[j]->name;
	  define_collision_op( takizuka_abe(cname.c_str(), ions[i], ions[j], cvar0, ncoll, var_wt) );
      }
  }
  for (auto* ion_s : ions) {
      std::string cname = std::string("Lemons_coll_e_") + ion_s->name;
      define_collision_op(lemons(cname.c_str(), ion_s, sp_fl, cvar0, ncoll, field_array));
  }

  for (auto* ion_s : ions) {
      ion_s->last_indexed = -1;
  }

  /*
  auto& interpolators_k = interpolator_array->k_i_d;
  int istep = 1;
  // Do the benchmark
  double elapsed = wallclock();
  repeat( num_step ) {
      apply_collision_op_list( collision_op_list, *kokkos_rng );
  double pix_sum = 0.0, piy_sum = 0.0, piz_sum = 0.0;
double kit_sum = 0.0, nmi_sum = 0.0, ni_sum = 0.0;
  float tot_momentum1 = 0;
  float tot_momentum2 = 0;
  float tot_momentum3 = 0;
  float tot_en = 0;

for (size_t k = 0; k < ions.size(); ++k) {
  auto* ion_s = ions[k];
  auto* ha    = hydro_ions[k];

  auto& k_p   = ion_s->k_p_d;
  auto& k_pi  = ion_s->k_p_i_d;

  Kokkos::deep_copy(ha->k_h_d, 0.0f);
  accumulate_hydro_p_kokkos_nomove_ngp(
      k_p, k_pi, ha->k_h_d, interpolators_k, ion_s );
  ha->copy_to_host();

  const double nk  = n_i[k];
  const double mk  = m_i[k];

  double v1 = (ha->h[13]).px/(nk*mk);
  double v2 = (ha->h[13]).py/(nk*mk);
  double v3 = (ha->h[13]).pz/(nk*mk);
  double k2 = ((ha->h[13]).txx + (ha->h[13]).tyy + (ha->h[13]).tzz)/nk;
  double Tk = (k2 - mk*(v1*v1 + v2*v2 + v3*v3))/3.0;

  // per-species printout (kept same format/widthing)
  printf("%.15e\t%.15e\t%.15e\t%.15e\t", v1, v2, v3, Tk);

  // accumulate combined-ion stats
  pix_sum += (ha->h[13]).px;
  piy_sum += (ha->h[13]).py;
  piz_sum += (ha->h[13]).pz;
  kit_sum += ((ha->h[13]).txx + (ha->h[13]).tyy + (ha->h[13]).tzz);
  nmi_sum += nk*mk;
  ni_sum  += nk;

  // global conservation tallies
  tot_momentum1 += nk*mk*v1;
  tot_momentum2 += nk*mk*v2;
  tot_momentum3 += nk*mk*v3;
  tot_en        += k2*nk;
}

 float dmom1 = 0, dmom2 = 0, dmom3 = 0, den = 0;
 dmom1 = tot_momentum1; // - tot_momentum1_0;
 dmom2 = tot_momentum2; // - tot_momentum2_0;
 dmom3 = tot_momentum3; // - tot_momentum3_0;
 den   = tot_en; // - tot_en_0;
 
 printf("%.15e\t%.15e\t%.15e\t%.15e\n", dmom1, dmom2, dmom3, den);
  ++istep;   
  }
  exit(1);
  */
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
  sim_log ( "pert = " << pert );
  sim_log ( "Ti = " << Ti );
  sim_log ( "gamma = " << gamma );
  sim_log ( "Te = " << Te );
  sim_log ( "nx = " << nx );
  sim_log ( "ny = " << ny );
  sim_log ( "nz = " << nz );
  sim_log ( "nproc = " << nproc ()  );
  sim_log ( "nppc = " << nppc );
  sim_log ( "b0 = " << b0 );
  sim_log ( "v_A = " << v_A );
  sim_log ( "di = " << di );
  sim_log ( "Ni = " << Ni );
    sim_log ( "total # of particles = " << Ni );
  sim_log ( "dt*wci = " << wci*dt );
  sim_log ( "energies_interval: " << energies_interval );
  sim_log ( "dx = " << Lx/(di*nx) );
  sim_log ( "dy = " << Ly/(di*ny) );
  sim_log ( "dz = " << Lz/(di*nz) );
  sim_log ( "n0 = " << n0 );

 // Dump simulation information to file "info.bin" for translate script
  if (rank() == 0 ) {
    FileIO fp_coll;
    if ( ! (fp_coll.open("HeCAue.txt", io_write)==ok) ) ERROR(("Cannot open file."));
    fp_coll.print("#step vx vy vz T vx vy vz T .... .... dmomx dmomy dmomz den \n");
    fp_coll.close();
    
    FileIO fp_info;

    // write binary info file

    if ( ! (fp_info.open("info.bin", io_write)==ok) ) ERROR(("Cannot open file."));
    
    fp_info.write(&topology_x, 1 );
    fp_info.write(&topology_y, 1 );
    fp_info.write(&topology_z, 1 );

    fp_info.write(&Lx, 1 );
    fp_info.write(&Ly, 1 );
    fp_info.write(&Lz, 1 );

    fp_info.write(&nx, 1 );
    fp_info.write(&ny, 1 );
    fp_info.write(&nz, 1 );

    fp_info.write(&dt, 1 );

    fp_info.close();

}


  ////////////////////////////
  // Load fields
sim_log( "Loading fields" );

// Note: everywhere is a region that encompasses the entire simulation                                                                                                                   
// In general, regions are specied as logical equations (i.e. x>0 && x+y<2) 
 set_region_field( everywhere, 0, 0, 0, 0.0, 0, 0);
 set_region_te(everywhere, Te);
 double uex = 0.9039;
 set_region_ue(everywhere, uex);

 // LOAD PARTICLES
  sim_log( "Loading particles" );

  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);
  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);
  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);



 
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

  global->fdParams.format = band;
  sim_log ( "Fields output format = band" );

  global->hedParams.format = band;
  sim_log ( "Electron species output format = band" );

  global->hHdParams.format = band;
  sim_log ( "Ion species output format = band" );

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

  //  // relative path to fields data from global header
  //  sprintf(global->fdParams.baseDir, "fields");
  //  // base file name for fields output
  //  sprintf(global->fdParams.baseFileName, "fields");
   sprintf(global->fdParams.baseDir, "fields/");
   dump_mkdir("fields");
   dump_mkdir(global->fdParams.baseDir);
   // base file name for fields output
   sprintf(global->fdParams.baseFileName, "fields");

  global->fdParams.stride_x = 1;
  global->fdParams.stride_y = 1;
  global->fdParams.stride_z = 1;

  // add field parameters to list
  global->outputParams.push_back(&global->fdParams);

  sim_log ( "Fields x-stride " << global->fdParams.stride_x );
  sim_log ( "Fields y-stride " << global->fdParams.stride_y );
  sim_log ( "Fields z-stride " << global->fdParams.stride_z );

  //  // relative path to electron species data from global header
  //sprintf(global->hedParams.baseDir, "hydro");
  //

  //  // relative path to electron species data from global header
  sprintf(global->hHdParams.baseDir, "hydro");
  //sprintf(global->hHdParams.baseDir, "hydro/%d",NUMFOLD);
  dump_mkdir("hydro");
  dump_mkdir(global->hHdParams.baseDir);

  //// base file name for fields output
  //sprintf(global->hHdParams.baseFileName, "Hhydro");

  // base file name for fields output
  sprintf(global->hHdParams.baseFileName, "Hhydro");

  global->hHdParams.stride_x = 1;
  global->hHdParams.stride_y = 1;
  global->hHdParams.stride_z = 1;

  sim_log ( "Ion species x-stride " << global->hHdParams.stride_x );
  sim_log ( "Ion species y-stride " << global->hHdParams.stride_y );
  sim_log ( "Ion species z-stride " << global->hHdParams.stride_z );

  // add electron species parameters to list
  global->outputParams.push_back(&global->hHdParams);

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

  //global->fdParams.output_variables( electric | magnetic );
  global->hedParams.output_variables( current_density | charge_density | stress_tensor );
  global->hHdParams.output_variables( current_density | charge_density | stress_tensor );


  const uint32_t allfields      (0xffffffff);
  
  global->fdParams.output_variables( allfields );
// global->hedParams.output_variables( all );
// global->hHdParams.output_variables( all );

  /*--------------------------------------------------------------------------
   * Convenience functions for simlog output
   *------------------------------------------------------------------------*/

  char varlist[512];
  create_field_list(varlist, global->fdParams);

  sim_log ( "Fields variable list: " << varlist );

  //create_hydro_list(varlist, global->hedParams);

  //sim_log ( "Electron species variable list: " << varlist );

  create_hydro_list(varlist, global->hHdParams);

  sim_log ( "Ion species variable list: " << varlist );

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

  // Adam: Can override some global params here
  // num_step = 171000;
  //global->fields_interval = 1358;
  //global->ehydro_interval = 1358;
  //global->Hhydro_interval = 1358;

    FileIO fp_coll;
    if ( ! (fp_coll.open("HeCAue.txt", io_append)==ok) ) ERROR(("Cannot open file."));    
    field_array->copy_to_host();
    interpolator_array->copy_to_device();
    auto& interpolators_k = interpolator_array->k_i_d;
    //printf("%e ne=%e, pe=%.15e uxyz=%e,%e,%e, se=%e, Pointer _field: %p, %p\n",step()*grid->dt, field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::rhof), field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::pe), field_array->k_f_h(13, field_var::ux), field_array->k_f_h(13, field_var::uy), field_array->k_f_h(13, field_var::uz), field_array->k_f_h(13, field_var::se), field_array, (void *)&field_array->k_f_d);
    float tot_momentum1,tot_momentum2, tot_momentum3, tot_en;
    tot_en = 0; //1.5*field_array->k_f_h(VOXEL(1,1,1,1,1,1), field_var::pe); //assuming dV=1
    tot_momentum1 = 0;
    tot_momentum2 = 0;
    tot_momentum3 = 0;
    fp_coll.print("%e\t%.15e\t%.15e\t%.15e\t%.15e\t ",step()*grid->dt,field_array->k_f_h(13, field_var::ux), field_array->k_f_h(13, field_var::uy), field_array->k_f_h(13, field_var::uz), field_array->k_f_h(VOXEL(1,1,1,1,1,1), field_var::pe)/32.6);
    float ue = 0.0;
    species_t *sp;
    int si = ion_cfg.size()-1;
    LIST_FOR_EACH( sp, species_list )
    {
	auto& particles = sp->k_p_d;
	auto& particles_i = sp->k_p_i_d;
	Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
	accumulate_hydro_p_kokkos_nomove_ngp(
					     particles,
					     particles_i,
					     hydro_array->k_h_d,
					     interpolators_k,
					     sp
					     );
	hydro_array->copy_to_host();
	auto ni = ion_cfg[si].n;
	auto mi = ion_cfg[si].m;
	auto qi = ion_cfg[si].q;
	//printf("mi=%e,%e,ni=%e\n",mi,sp->m,ni);
	float vi1 = (hydro_array->h[13]).px/(ni*mi);
	ue += qi*ni*vi1;
	float vi2 = (hydro_array->h[13]).py/(ni*mi);
	float vi3 = (hydro_array->h[13]).pz/(ni*mi);
	float ki = (hydro_array->h[13]).txx + (hydro_array->h[13]).tyy + (hydro_array->h[13]).tzz;
	float ke2 = ki/ni;
	float Ti = (ke2 - mi*(vi1*vi1 + vi2*vi2 + vi3*vi3))/3.0;
	tot_momentum1 += ni*mi*vi1;
	tot_momentum2 += ni*mi*vi2;
	tot_momentum3 += ni*mi*vi3;
	tot_en += ki*0.5;
	fp_coll.print("%.15e\t%.15e\t%.15e\t%.15e\t ",vi1,vi2,vi3,Ti);	
	--si;
    }
    tot_momentum1 += field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::sx);
    fp_coll.print("%.15e\t%.15e\t%.15e\t%.15e\t ",tot_momentum1,tot_momentum2, tot_momentum3, tot_en+field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::pe)*1.5);
    
    fp_coll.print("\n");

    fp_coll.close();

    ue /=32.6; //ne
    field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::ux) = ue;
    field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::uy) = 0;
    field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::uz) = 0;
    
    field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::sx) = 0;
    field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::se) = tot_en;
    //field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::sen) = tot_en;
    //printf("%e, sx=%e, ne=%e, pe=%e uxyz=%e,%e,%e Pointer _field: %p, %p\n",step()*grid->dt, field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::sx), field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::rhof), field_array->k_f_d(VOXEL(1,1,1,1,1,1), field_var::pe), field_array->k_f_h(13, field_var::ux), field_array->k_f_h(13, field_var::uy), field_array->k_f_h(13, field_var::uz), field_array, (void *)&field_array->k_f_d);
  
  global->restart_interval = 3000;
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
    dump_mkdir("restore0");
    dump_mkdir("restore1");  // 1st backup
    dump_mkdir("particle");
    dump_mkdir("rundata");

    
    // Make subfolders for restart
    //    char restorefold[128];
    //sprintf(restorefold, "restore0/%i", NUMFOLD);
    //    sprintf(restorefold, "restore0");
    //    dump_mkdir(restorefold);
    //    sprintf(restorefold, "restore1/%i", NUMFOLD);
    //    sprintf(restorefold, "restore1");
    //    dump_mkdir(restorefold);
    //    sprintf(restorefold, "restore2/%i", NUMFOLD);
    //    dump_mkdir(restorefold);

    // And rundata 
    //    char rundatafold[128];
    //    char rundatafile[128];
    //    sprintf(rundatafold, "rundata/%i", NUMFOLD);
    ///    sprintf(rundatafold, "rundata");
    //    dump_mkdir(rundatafold);

    dump_grid("rundata/grid");
    //    sprintf(rundatafile, "rundata/%i/grid", NUMFOLD);
    //    dump_grid(rundatafile);

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
   * Field data output
   *------------------------------------------------------------------------*/

  if(step() == 1 || should_dump(fields)) field_dump(global->fdParams);

  /*--------------------------------------------------------------------------
   * Electron species output
   *------------------------------------------------------------------------*/

  //if(should_dump(ehydro)) hydro_dump("electron", global->hedParams);

  /*--------------------------------------------------------------------------
   * Ion species output
   *------------------------------------------------------------------------*/

  //if(should_dump(Hhydro)) hydro_dump("ion", global->hHdParams);


  /*--------------------------------------------------------------------------
  * Time averaging
  *------------------------------------------------------------------------*/

  //  #include "time_average.cxx"
   //#include "time_average_cori.cxx"

  /*--------------------------------------------------------------------------
   * Restart dump
   *------------------------------------------------------------------------*/

  if(step() && !(step()%global->restart_interval)) {
    global->write_restart = 1; // set restart flag. the actual restart files are written during the next step
  } else {
    if (global->write_restart) {

      global->write_restart = 0; // reset restart flag
      double dumpstart = uptime();
      if(!global->rtoggle) {
        global->rtoggle = 1;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
	checkpt("restore1/restore", 0);
	//	DUMP_INJECTORS(1);
	//    } END_TURNSTILE;
      } else {
        global->rtoggle = 0;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
	checkpt("restore0/restore", 0);
	//	DUMP_INJECTORS(0);
	//    } END_TURNSTILE;
      } // if

      //    mp_barrier();
      sim_log( "Restart dump completed");
      double dumpelapsed = uptime() - dumpstart;
      sim_log("Restart duration "<<dumpelapsed);
    } // if global->write_restart
  }


  /*  // Dump particle data

  char subdir[36];

  if ( should_dump(Hparticle) && step() !=0
       && step() > 56*(global->fields_interval)  ) {
    sprintf(subdir,"particle/T.%d/Hparticle",step());
    dump_particles("ion", subdir);
    }*/

  // Shut down simulation when wall clock time exceeds global->quota_sec.
  // Note that the mp_elapsed() is guaranteed to return the same value for all
  // processors (i.e., elapsed time on proc #0), and therefore the abort will
  // be synchronized across processors. Note that this is only checked every
  // few timesteps to eliminate the expensive mp_elapsed call from every
  // timestep. mp_elapsed has an ALL_REDUCE in it!


  if ( (step()>0 && global->quota_check_interval>0
        && (step() & global->quota_check_interval)==0 ) || (global->write_end_restart) ) {

    if ( (global->write_end_restart) ) {
      global->write_end_restart = 0; // reset restart flag

      //   if( uptime() > global->quota_sec ) {
      sim_log( "Allowed runtime exceeded for this job.  Terminating....\n");
      double dumpstart = uptime();

      if(!global->rtoggle) {
        global->rtoggle = 1;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
        checkpt("restore1", 0);
        //    } END_TURNSTILE;
      } else {
        global->rtoggle = 0;
        //      BEGIN_TURNSTILE(NUM_TURNSTILES) {
        checkpt("restore0", 0);
        //    } END_TURNSTILE;
      } // if

      mp_barrier(  ); // Just to be safe
      sim_log( "Restart dump restart completed." );
      double dumpelapsed = uptime() - dumpstart;
      sim_log("Restart duration "<< dumpelapsed);
      exit(0); // Exit or abort?                                                                                
    }
    //    } 
    if( uptime() > global->quota_sec ) global->write_end_restart = 1;
  }


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

