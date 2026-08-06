//////////////////////////////////////////////////////////////////////////////
//
//   Plasma sphere - spherical-grid verification deck
//
//   A cold-ion plasma sphere on a spherical (r, theta, phi) grid with n
//   external E or B fields. The ions are loaded with a small radial density
//   perturbation on top of a uniform background. With no confining fields the
//   only restoring force is the electron-pressure gradient in Ohm's law, so
//   the perturbation should oscillate (a spherical ion-acoustic breathing
//   mode) rather than grow or run away.
//
//////////////////////////////////////////////////////////////////////////////

begin_globals {

  int restart_interval;
  int energies_interval;
  int fields_interval;
  int Hhydro_interval;
  int quota_check_interval;

  int rtoggle;               // enables save of last 2 restart dumps for safety
  int write_restart;
  int write_end_restart;

  double quota_sec;          // Run quota in seconds
  double topology_x;
  double topology_y;
  double topology_z;

  // Output variables
  DumpParameters fdParams;
  DumpParameters hHdParams;
  std::vector<DumpParameters *> outputParams;

};

begin_initialization {
  double ec   = 1.0;  // Charge normalization
  double mi   = 1.0;  // Mass normalization
  double mu0  = 1.0;  // Magnetic constant
  double b0   = 1.0;  // Reference B for units (fields are zero in this problem)
  double n0   = 1.0;  // Background density

  double v_A = b0/sqrt(mu0*n0*mi); // Alfven velocity
  double wci = ec*b0/mi;           // Cyclotron freq.
  double di  = v_A/wci;            // Ion skin-depth

  double Ti    = 1.0/3.0;    // Ion temperature
  double gamma = 5.0/3.0;    // Electron adiabatic index
  double c_s   = 1.0;        // Electron sound speed
  double Te    = c_s/gamma;  // Electron temperature
  double vthi  = sqrt(Ti/mi);// Ion thermal velocity

  double pert   = 0.05;      // Fractional size of the density perturbation
  double eta      = 0.0;     // Resistivity
  double hypereta = 1e-5;    // Hyper-resistivity: raised 1e-2->1e-1 to test grid-mode damping

  double Rmax  = 8.0*di;     // Outer radius
  double Rmin  = 0.25*Rmax;  // Inner radius (off-center: keeps near-center cells
                             // well-populated; a smaller Rmin makes the tiny
                             // r^2 sin(theta) cells under-resolved -> grad(pe)
                             // noise -> field blow-up).
  double rpk   = 0.55*Rmax;  // Radius of the perturbation shell (inside [Rmin,Rmax])
  double rwid  = 0.15*Rmax;  // Width of the perturbation shell

  double nx = 32;            // radial cells
  double ny = 16;            // polar (theta) cells,  theta in [0, pi]
  double nz = 16;            // azimuthal (phi) cells, phi in [0, 2pi]

  double topology_x = 1;
  double topology_y = 1;
  double topology_z = 1;

  double Lx = Rmax - Rmin;   // radial extent (for info/logging)
  double Ly = M_PI;          // theta extent
  double Lz = 2.0*M_PI;      // phi extent

  // --------------------------------------------------------------------------
  // Particles
  // --------------------------------------------------------------------------
  double nppc = 800;                 // macro-particles per cell (high, so the
                                     // small near-center cells stay well-populated)
  double Ni   = nppc*nx*ny*nz;       // total macro ions
  Ni = trunc_granular(Ni, nproc());  // divisible by ranks
  // Physical background ions occupy the spherical SHELL Rmin<r<Rmax.
  double Vol  = (4.0/3.0)*M_PI*(Rmax*Rmax*Rmax - Rmin*Rmin*Rmin);
  double Np   = n0*Vol;
  double qi   = ec*Np/Ni;            // charge per macro ion (uniform-density weight)

  // --------------------------------------------------------------------------
  // Timestep and run length
  // --------------------------------------------------------------------------
  double dt   = 0.02;
  double taui = 40.0;                // run time in wci^-1
  num_step    = int(taui/(wci*dt));

  double sort_interval = 10;
  double quota     = 2.0;
  double quota_sec = quota*3600;

  int restart_interval    = 200000;
  int energies_interval   = 1;
  int interval            = int(num_step/100);
  if (interval < 1) interval = 1;
  int fields_interval     = interval;
  int Hhydro_interval     = interval;
  int quota_check_interval = 100;

  // High-level parameters
  status_interval      = interval;
  sync_shared_interval = 0;  // hybrid solver manages its own sync
  clean_div_e_interval = 0;
  clean_div_b_interval = 0;

  global->restart_interval     = restart_interval;
  global->energies_interval    = energies_interval;
  global->fields_interval      = fields_interval;
  global->Hhydro_interval      = Hhydro_interval;
  global->quota_check_interval = quota_check_interval;
  global->quota_sec            = quota_sec;
  global->rtoggle              = 0;
  global->write_restart        = 0;
  global->write_end_restart    = 0;
  global->topology_x           = topology_x;
  global->topology_y           = topology_y;
  global->topology_z           = topology_z;

  // --------------------------------------------------------------------------
  // Grid
  // --------------------------------------------------------------------------
  define_units(1.0, 1.0);
  define_timestep(dt);

  // Spherical grid: low corner (r0, theta0, phi0), high corner (r1, theta1, phi1).
  define_periodic_grid( Rmin, 0.0,    0.0,        // Low corner  (r, theta, phi)
                        Rmax, M_PI,   2.0*M_PI,   // High corner
                        nx, ny, nz,               // Resolution
                        topology_x, topology_y, topology_z );

  grid->eos_den     = 1.0;
  grid->eta         = eta;
  grid->hypereta    = hypereta;
  grid->eos_gamma   = gamma;
  grid->eos_gamma_0 = gamma;
  grid->init_spherical_grid();

  grid->nsub = 10;   // field subcycles
  grid->nsm  = 2;    // moment smoothing passes
  grid->nsmb = 0;
  grid->den_floor_ohm = 0.2;
  grid->den_floor_pe  = 0.2;

  // Identify boundary domains
  int ix, iy, iz;
  ix = int(rank()) % int(topology_x);
  iy = (int(rank()) / int(topology_x)) % int(topology_y);
  iz = int(rank()) / (int(topology_x)*int(topology_y));

  // Inner-r face = the center r=0 (point singularity).
  if ( ix==0 )              set_domain_field_bc( BOUNDARY(-1,0,0), spherical_center_fields );
  // Outer-r face = physical wall.
  if ( ix==topology_x-1 )   set_domain_field_bc( BOUNDARY( 1,0,0), pec_fields );
  // Theta faces = the polar axis (theta=0 and theta=pi).
  if ( iy==0 )              set_domain_field_bc( BOUNDARY(0,-1,0), spherical_axis_fields );
  if ( iy==topology_y-1 )   set_domain_field_bc( BOUNDARY(0, 1,0), spherical_axis_fields );
  // Phi faces are periodic (default) - do nothing.

  if ( ix==0 )              set_domain_particle_bc( BOUNDARY(-1,0,0), spherical_center_particles );
  if ( ix==topology_x-1 )   set_domain_particle_bc( BOUNDARY( 1,0,0), reflect_particles );
  if ( iy==0 )              set_domain_particle_bc( BOUNDARY(0,-1,0), spherical_axis_particles );
  if ( iy==topology_y-1 )   set_domain_particle_bc( BOUNDARY(0, 1,0), spherical_axis_particles );

  sim_log("Setting up materials.");
  define_material( "vacuum", 1 );
  define_field_array(NULL);

  sim_log("Setting up species.");
  double nmax    = 1.5*Ni/nproc();
  double nmovers = 0.1*nmax;
  double sort_method = 1;
  species_t *ion = define_species("ion", ec, mi, nmax, nmovers, sort_interval, sort_method);

  sim_log( "***********************************************" );
  sim_log( "Plasma sphere (spherical-grid verification)" );
  sim_log( "Topology: " << topology_x << " " << topology_y << " " << topology_z );
  sim_log( "num_step = " << num_step );
  sim_log( "Rmax = " << Rmax << "  nr,ntheta,nphi = " << nx << " " << ny << " " << nz );
  sim_log( "nppc = " << nppc << "  Ni = " << Ni );
  sim_log( "Ti = " << Ti << "  Te = " << Te << "  gamma = " << gamma );
  sim_log( "pert = " << pert << "  rpk = " << rpk << "  rwid = " << rwid );
  sim_log( "dt*wci = " << wci*dt );
  sim_log( "***********************************************" );

  // Dump info.bin for the translate script (matches examples/mirror layout:
  // tx,ty,tz, Lx,Ly,Lz, nx,ny,nz, dt as doubles).
  if (rank() == 0) {
    FileIO fp_info;
    if ( ! (fp_info.open("info.bin", io_write)==ok) ) ERROR(("Cannot open file."));
    fp_info.write(&topology_x, 1);
    fp_info.write(&topology_y, 1);
    fp_info.write(&topology_z, 1);
    fp_info.write(&Lx, 1);
    fp_info.write(&Ly, 1);
    fp_info.write(&Lz, 1);
    fp_info.write(&nx, 1);
    fp_info.write(&ny, 1);
    fp_info.write(&nz, 1);
    fp_info.write(&dt, 1);
    fp_info.write(&mi, 1);
    fp_info.close();
  }

  sim_log( "Loading fields" );
  set_region_field( everywhere, 0, 0, 0, 0, 0, 0 );  // no E, no B
  set_region_te( everywhere, Te );

  // Load particles
  //
  // Uniform density in the sphere with a radial shell perturbation:
  //   n(r) = n0 * ( 1 + pert * exp(-((r-rpk)/rwid)^2) )
  // Positions are sampled in physical (r,theta,phi); VPIC interprets the
  // injected coordinates in the grid's coordinate system. To get uniform
  // sampling in physical volume we sample r with the r^2 sin(theta) weight via
  // rejection against the perturbed profile.
  sim_log( "Loading particles" );

  double nmax_prof = 1.0 + pert;      // peak of the profile envelope for rejection
  repeat( Ni ) {
    double r, theta, phi, ux, uy, uz;

    // Rejection sample (r,theta) so that the accepted density in PHYSICAL space
    // is proportional to n(r): pdf(r,theta) ~ r^2 sin(theta) * n(r).
    double accept, trial;
    do {
      // Volume-uniform radius within the shell [Rmin,Rmax]: r^3 uniform.
      r     = cbrt( uniform(rng(0), Rmin*Rmin*Rmin, Rmax*Rmax*Rmax) );
      theta = acos( uniform(rng(0), -1.0, 1.0) );        // ~ sin(theta)
      double prof = 1.0 + pert*exp( -((r-rpk)/rwid)*((r-rpk)/rwid) );
      accept = prof / nmax_prof;
      trial  = uniform(rng(0), 0.0, 1.0);
    } while ( trial > accept );
    phi = uniform(rng(0), 0.0, 2.0*M_PI);

    // Cold-ish Maxwellian (Cartesian velocity components, as VPIC expects)
    ux = normal( rng(0), 0, vthi );
    uy = normal( rng(0), 0, vthi );
    uz = normal( rng(0), 0, vthi );

    // Inject in spherical coordinates (r, theta, phi). Uniform weight qi since
    // the r^2 sin(theta) volume weighting is handled by the sampling above.
    inject_particle( ion, r, theta, phi, ux, uy, uz, qi, 0, 0 );
  }
  sim_log( "Finished loading particles" );

  // --------------------------------------------------------------------------
  // Output setup
  // --------------------------------------------------------------------------
  global->fdParams.format = band;
  global->hHdParams.format = band;

  sprintf(global->fdParams.baseDir, "fields/");
  dump_mkdir("fields");
  dump_mkdir(global->fdParams.baseDir);
  sprintf(global->fdParams.baseFileName, "fields");
  global->fdParams.stride_x = 1;
  global->fdParams.stride_y = 1;
  global->fdParams.stride_z = 1;
  global->outputParams.push_back(&global->fdParams);

  sprintf(global->hHdParams.baseDir, "hydro");
  dump_mkdir("hydro");
  dump_mkdir(global->hHdParams.baseDir);
  sprintf(global->hHdParams.baseFileName, "Hhydro");
  global->hHdParams.stride_x = 1;
  global->hHdParams.stride_y = 1;
  global->hHdParams.stride_z = 1;
  global->outputParams.push_back(&global->hHdParams);

  global->hHdParams.output_variables( allvars );  // dump ALL hydro vars: translate_faster expects the full hydro layout
  const uint32_t allfields(0xffffffff);
  global->fdParams.output_variables( allfields );

  char varlist[1024];
  create_field_list(varlist, global->fdParams);
  sim_log( "Fields variable list: " << varlist );
  create_hydro_list(varlist, global->hHdParams);
  sim_log( "Ion species variable list: " << varlist );

  sim_log("*** Finished with user-specified initialization ***");

} //begin_initialization

#define should_dump(x)                                                  \
  (global->x##_interval>0 && remainder(step(), global->x##_interval) == 0)

begin_diagnostics {

  const int nx=grid->nx;
  const int ny=grid->ny;
  const int nz=grid->nz;

  if(step()==0) {
    dump_mkdir("fields");
    dump_mkdir("hydro");
    dump_mkdir("rundata");
    dump_mkdir("restore0");
    dump_mkdir("restore1");
    dump_grid("rundata/grid");
    dump_materials("rundata/materials");
    dump_species("rundata/species");
    global_header("global", global->outputParams);
  }

  if(should_dump(energies)) {
    dump_energies("rundata/energies", step() == 0 ? 0 : 1);
  }

  if(step() == 1 || should_dump(fields)) field_dump(global->fdParams);

  if(should_dump(Hhydro)) hydro_dump("ion", global->hHdParams);

  // Restart dump
  if(step() && !(step()%global->restart_interval)) {
    global->write_restart = 1;
  } else {
    if (global->write_restart) {
      global->write_restart = 0;
      if(!global->rtoggle) { global->rtoggle = 1; checkpt("restore1/restore", 0); }
      else                 { global->rtoggle = 0; checkpt("restore0/restore", 0); }
      sim_log( "Restart dump completed");
    }
  }

} // end diagnostics

begin_particle_injection {
} // end particle injection

begin_current_injection {
} // end current injection

begin_field_injection {
} // end field injection

begin_particle_collisions {
} // end collisions
