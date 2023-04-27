// Hybrid astrophysical explosion
//Ari Le, LANL, 2023


// If you want to use global variables (for example, to store the dump
// intervals for your diagnostics section), it must be done in the globals
// section. Variables declared the globals section will be preserved across
// restart dumps. For example, if the globals section is:
//   begin_globals {
//     double variable;
//   } end_globals
// the double "variable" will be visible to other input deck sections as
// "global->variable". Note: Variables declared in the globals section are set
// to zero before the user's initialization block is executed. Up to 16K
// of global variables can be defined.

begin_globals {
  double energies_interval;
  double fields_interval;
  double ehydro_interval;
  double ihydro_interval;
  double eparticle_interval;
  double iparticle_interval;
  double restart_interval;


  //  Variables for new output format

  DumpParameters fdParams;
  DumpParameters iParams;
  DumpParameters diParams;

  std::vector<DumpParameters *> outputParams;

};

begin_initialization {
  // At this point, there is an empty grid and the random number generator is
  // seeded with the rank. The grid, materials, species need to be defined.
  // Then the initial non-zero fields need to be loaded at time level 0 and the
  // particles (position and momentum both) need to be loaded at time level 0.


  // Diagnostic messages can be passed written (usually to stderr)
  sim_log( "Computing simulation parameters");

  // Define the system of units for this problem (natural units)
  double ec   = 1; // Charge normalization
  double me   = 1; // Mass normalization
  double c    = 1; // Speed of light
  double eps0 = 1; // Permittivity of space
  double di   = 1;

  // Physics parameters
  double mi_me   = 1; // Ion mass 
  double md_mi   = 3; // Debris Ion mass / background ion mass
  double Zd      = 1; // Debris charge state
  double Vd_Va    = 15;       // Debris radial velocity Alfven Mach number
  double Rm_di    = 150.0;     // Equal mass radius/ion inertial length
  double L_di     = 0.1*Rm_di; // Inital debris length scale  / ion inertial [n=n0*exp(-(r/L)^2)]
  double Tid_Tib   = 1.0;      // Debris Ion temperature / background ion temperature 
  double rhoi_L  = 1;    // Ion thermal gyroradius / Sheet thickness
  double Ti_Te   = 1;    // Ion temperature / electron temperature
  double wpe_wce = 1;    // Electron plasma freq / electron cycltron freq
  double theta   = 0.5*M_PI;    // Orientation of the B. 0 gives out of plane By, M_PI/2 gives Bx
  double taui    = 100;  // Simulation wci's to run
  double beta_b  = .1;        //background ion beta
  double nd_n0    = (Rm_di*Rm_di/L_di/L_di)/md_mi;    //peak debris density/background density
  

  double beta_d = 1.0*nd_n0;  // peak debris ion beta

  // Numerical parameters
  double Lx        = 1000*di;  // How big should the box be in the x direction
  double Ly        = 1*di;  // How big should the box be in the y direction
  double Lz        = 1000*di;  // How big should the box be in the z direction
  double nx        = 512;    // Global resolution in the x direction
  double ny        = 1;    // Global resolution in the y direction
  double nz        = 512;     // Global resolution in the z direction
  double nppc      = 200;    // Average number of macro particles per cell per species
  double tx        = 16;
  double ty        = 1;
  double tz        = 8;

  double pi = M_PI;

  // Derived quantities

  double  L    = L_di*di;                      //debris radius
  double mi   = me*mi_me;                      // Ion mass
  double kTe  = beta_b/2;                      // Electron temperature
  double kTi  = kTe*Ti_Te;                     // Ion temperature
  double vthe = sqrt(2*kTe/me);                // Electron thermal velocity (B.D. convention)
  double v_A  = (1);                            // based on n0
  double Vd   = Vd_Va*v_A;                      // debris radial drift speed 
  double mib = mi;                             //Ion mass
  double mid = mi*md_mi;                       //Ion mass
  double Teb = 0.5*beta_b;                     //background Electron temperature
  double Tib = Teb;                            //Backround Ion temperature 
  double Tid = 0.5*beta_d/(nd_n0);             // 
  double vthib = sqrt(Tib/mib);                // Ion thermal velocity in sheath
  double vthid = 0.1*Vd;                       //Ion thermal velocity;
  double b0   = 1.;                            // Asymptotic magnetic field strength
  double n0   = 1.;                            // Peak electron density (also peak ion density)
  double Npe  = n0*Ly*Lz*Lx;                   // Number of physical electrons in box
  double Npi  = Npe;                           // Number of physical ions in box
  double Ne   = nppc*nx*ny*nz;                 // Total macro electrons in box
  Ne = trunc_granular(Ne,nproc());             // Make it divisible by number of processors
  double Ni   = Ne;                            // Total macro ions in box
  double we   = Npe/Ne;                        // Weight of a macro electron
  double wi   = Npi/Ni;                        // Weight of a macro ion
  double Ndi  = pi*n0*nd_n0*Ly*L*L;            //number physical debris ion
  double Ni2  = trunc_granular(Ni/50,nproc());   // Number macro debris ions
  double wid  = Ndi/Ni2;                        // Weight of a macro ion



  // Determine the timestep
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);      // Courant length
  double dt = 0.01;                           // time step
  //if( wpe*dt>wpedt_max ) dt=wpedt_max/wpe;            // Override time step if plasma frequency limited

  ////////////////////////////////////////
  // Setup high level simulation parmeters

  num_step             = int(5000);
  status_interval      = 100;//int(1./(wci*dt));
  sync_shared_interval = status_interval;
  clean_div_e_interval = status_interval;
  clean_div_b_interval = status_interval;

  global->energies_interval  = status_interval;
  global->fields_interval    = status_interval;
  global->ehydro_interval    = status_interval;
  global->ihydro_interval    = status_interval;
  global->eparticle_interval = status_interval; // Do not dump
  global->iparticle_interval = status_interval; // Do not dump
  global->restart_interval   = status_interval*5.0; // Do not dump

  ///////////////////////////
  // Setup the space and time

  // Setup basic grid parameters
  define_units( c, eps0 );
  define_timestep( dt );

  // Parition a periodic box among the processors sliced uniformly along y
  define_periodic_grid( -0.5*Lx, -0.5*Ly, -0.5*Lz,    // Low corner
                         0.5*Lx, 0.5*Ly, 0.5*Lz,  // High corner
                         nx, ny, nz,      // Resolution
                         tx, ty, tz ); // Topology

  grid->te = kTe;
  grid->den = 1.0;
  grid->eta = 0;
  grid->hypereta=0.005; // was 0.0001
  grid->gamma = 1.0;

  grid->nsm=0;
  grid->nsmb=0;
  grid->nsub=1;
  
   define_material( "vacuum", 1 );
  // Note: define_material defaults to isotropic materials with mu=1,sigma=0
  // Tensor electronic, magnetic and conductive materials are supported
  // though. See "shapes" for how to define them and assign them to regions.
  // Also, space is initially filled with the first material defined.

  // If you pass NULL to define field array, the standard field array will
  // be used (if damp is not provided, no radiation damping will be used).
  define_field_array( NULL );

  ////////////////////
  // Setup the species
  double nmax = 1.5*Ni2;

  // Allow 50% more local_particles in case of non-uniformity
  // VPIC will pick the number of movers to use for each species
  // Both species use out-of-place sorting
  species_t * ion      = define_species( "ion",  ec, mi,  14*Ni/nproc(), -1, 40, 1 );
  species_t * dion     = define_species( "dion", ec, mid, nmax,          -1, 40, 1 );

  //species_t * electron = define_species( "electron", -ec, me, 1.5*Ne/nproc(), -1, 20, 1 );

  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation

  sim_log( "" );
  sim_log( "System of units" );
  sim_log( "L = " << L );
  sim_log( "ec = " << ec );
  sim_log( "me = " << me );
  sim_log( "c = " << c );
  sim_log( "eps0 = " << eps0 );
  sim_log( "" );
  sim_log( "Physics parameters" );
  sim_log( "rhoi/L = " << rhoi_L );
  sim_log( "Ti/Te = " << Ti_Te );
  sim_log( "wpe/wce = " << wpe_wce );
  sim_log( "mi/me = " << mi_me );
  sim_log( "theta = " << theta );
  sim_log( "taui = " << taui );
  sim_log( "" );
  sim_log( "Numerical parameters" );
  sim_log( "num_step = " << num_step );
  sim_log( "dt = " << dt );
  sim_log( "Lx = " << Lx << ", Lx/L = " << Lx/L );
  sim_log( "Ly = " << Ly << ", Ly/L = " << Ly/L );
  sim_log( "Lz = " << Lz << ", Lz/L = " << Lz/L );
  sim_log( "nx = " << nx << ", dx = " << Lx/nx << ", L/dx = " << L*nx/Lx );
  sim_log( "ny = " << ny << ", dy = " << Ly/ny << ", L/dy = " << L*ny/Ly );
  sim_log( "nz = " << nz << ", dz = " << Lz/nz << ", L/dz = " << L*nz/Lz );
  sim_log( "nppc = " << nppc );
  sim_log( "courant = " << c*dt/dg );
  sim_log( "" );
  sim_log( "Ion parameters" );
  sim_log( "qpi = "  << ec << ", mi = " << mi << ", qpi/mi = " << ec/mi );
  sim_log( "vthib = " << vthib << ", vthib/c = " << vthib/c << ", kTi = " << kTi  );
  sim_log( "Npi = " << Npi << ", Ni = " << Ni << ", Npi/Ni = " << Npi/Ni << ", wi = " << wi );
  sim_log( "" );
  sim_log( "Miscellaneous" );
  sim_log( "nptotal = " << Ni + Ne );
  sim_log( "nproc = " << nproc() );
  sim_log( "" );


  // Dump simulation information to file "info"
  if (rank() == 0 ) {

    FileIO fp_info;

    // write binary info file

    if ( ! (fp_info.open("info.bin", io_write)==ok) ) ERROR(("Cannot open file."));
    
    fp_info.write(&tx, 1 );
    fp_info.write(&ty, 1 );
    fp_info.write(&tz, 1 );

    fp_info.write(&Lx, 1 );
    fp_info.write(&Ly, 1 );
    fp_info.write(&Lz, 1 );

    fp_info.write(&nx, 1 );
    fp_info.write(&ny, 1 );
    fp_info.write(&nz, 1 );

    fp_info.write(&dt, 1 );

    fp_info.write(&mi_me, 1 );
    fp_info.write(&wpe_wce, 1 );
    fp_info.write(&vthe, 1 );
    fp_info.write(&vthib, 1 );
    fp_info.write(&status_interval, 1 );
    fp_info.close();

}

  ////////////////////////////
  // Load fields and particles

  sim_log( "Loading fields" );


  set_region_field( everywhere, 0, 0, 0,      // Eletric field (Doesn't matter)
                    sin(theta),cos(theta), 0 ); // Magnetic field
  // Note: everywhere is a region that encompasses the entire simulation
  // In general, regions are specied as logical equations (i.e. x>0 && x+y<2)
  
  sim_log( "Loading particles" );

  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);
  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);
  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);

  repeat( Ni/nproc() ) {
    double x, y, z, ux, uy, uz, d0; 

    // Pick an appropriately distributed random location for the pair
    x = uniform( rng(0), xmin, xmax );
    y = uniform( rng(0), ymin, ymax );
    z = uniform( rng(0), zmin, zmax );

    // For the ion, pick an isothermal normalized momentum in the drift frame
    // (this is a proper thermal equilibrium in the non-relativistic limit),
    // boost it from the drift frame to the frame with the magnetic field
    // along z and then rotate it into the lab frame. Then load the particle.
    // Repeat the process for the electron.

    ux = normal( rng(0), 0, vthib );
    uy = normal( rng(0), 0, vthib );
    uz = normal( rng(0), 0, vthib );
    inject_particle( ion, x, y, z, ux, uy, uz, wi, 0, 0 );

  }

  double L_sqrttwo = L/sqrt(2);
  double twopi = 2*pi;


//loops over all debris particles, not just ones in this domain... 
  repeat ( Ni2 ) {
    double x, y, z, ux, uy, uz ;
    double r, phi;
        x = normal(rng(0),0,L_sqrttwo);
	y = uniform( rng(0), -0.5*Ly, 0.5*Ly);
	z = normal(rng(0),0,L_sqrttwo);
	
	phi = atan2(z,x);


       if ((x>=xmin) &&  (x<=xmax) && (z>=zmin) && (z<=zmax) && (y>=ymin) && (y<=ymax)) {


        ux = normal(rng(0),0,vthid) + Vd*cos(phi);
    	uy = normal(rng(0),0,vthid);
    	uz = normal(rng(0),0,vthid) + Vd*sin(phi);

        inject_particle(dion, x, y, z, ux, uy, uz, wid, 0, 0 );


 		// inject_particles() will return an error for particles no on this
   		// node and will not inject particle locally


	} // if
    
  } //



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

	global->fdParams.format = band;
	// relative path to fields data from global header
	sprintf(global->fdParams.baseDir, "fields");

	// base file name for fields output
	sprintf(global->fdParams.baseFileName, "fields");

	global->fdParams.stride_x = 1;
	global->fdParams.stride_y = 1;
	global->fdParams.stride_z = 1;

	global->fdParams.output_variables( all );
	
	char varlist[512];
	create_field_list(varlist, global->fdParams);


	// add field parameters to list
	global->outputParams.push_back(&global->fdParams);

	// relative path to ion species data from global header
	sprintf(global->iParams.baseDir, "hydro");
	sprintf(global->diParams.baseDir, "hydro");
	
	// base file name for hydro  output
	sprintf(global->iParams.baseFileName, "ihydro");
	sprintf(global->diParams.baseFileName, "dihydro");

	global->iParams.stride_x = 1;
	global->iParams.stride_y = 1;
	global->iParams.stride_z = 1;

	global->diParams.stride_x = 1;
	global->diParams.stride_y = 1;
	global->diParams.stride_z = 1;

	sim_log ( "Ion species x-stride " << global->iParams.stride_x );
	sim_log ( "Ion species y-stride " << global->iParams.stride_y );
	sim_log ( "Ion species z-stride " << global->iParams.stride_z );

	global->iParams.output_variables( current_density | charge_density | stress_tensor );
	global->diParams.output_variables( current_density | charge_density | stress_tensor );

	create_hydro_list(varlist, global->iParams);
	sim_log ( "Ion  species variable list: " << varlist );
	create_hydro_list(varlist, global->diParams);
	sim_log ( "dIon  species variable list: " << varlist );

	// add ion  species parameters to list
	global->outputParams.push_back(&global->iParams);
	global->outputParams.push_back(&global->diParams);
}

begin_diagnostics {

  if (step()==0 ) {
	  global_header("global", global->outputParams);
   if (step( )==0 ) {
	  dump_mkdir("fields");
	  dump_mkdir("hydro");
	  dump_mkdir("rundata");
	  dump_mkdir("restore");
	  dump_mkdir("restore1");  // 1st backup
	  dump_mkdir("restore2");  // 2nd backup
	  dump_mkdir("particle");
	  dump_grid("rundata/grid");
	  dump_materials("rundata/materials");
	  dump_species("rundata/species");
	  global_header("global", global->outputParams);
	} // if

  }

# define should_dump(x) (global->x##_interval>0 && remainder(step(),global->x##_interval)==0)

  if( step()==-10 ) {
    // A grid dump contains all grid parameters, field boundary conditions,
    // particle boundary conditions and domain connectivity information. This
    // is stored in a binary format. Each rank makes a grid dump
    dump_grid("grid");

    // A materials dump contains all the materials parameters. This is in a
    // text format. Only rank 0 makes the materials dump
    dump_materials("materials");

    // A species dump contains the physics parameters of a species. This is in
    // a text format. Only rank 0 makes the species dump
    dump_species("species");
  }

  // Energy dumps store all the energies in various directions of E and B
  // and the total kinetic (not including rest mass) energies of each species
  // species in a simple text format. By default, the energies are appended to
  // the file. However, if a "0" is added to the dump_energies call, a new
  // energies dump file will be created. The energies are in the units of the
  // problem and are all time centered appropriately. Note: When restarting a
  // simulation from a restart dump made at a prior time step to the last
  // energies dump, the energies file will have a "hiccup" of intervening
  // time levels. This "hiccup" will not occur if the simulation is aborted
  // immediately following a restart dump. Energies dumps are in a text
  // format and the layout is documented at the top of the file. Only rank 0
  // makes makes an energies dump.
  //if( should_dump(energies) ) dump_energies( "energies", step()==0 ? 0 : 1 );

  // Field dumps store the raw electromagnetic fields, sources and material
  // placement and a number of auxilliary fields. E, B and RHOB are
  // timecentered, JF and TCA are half a step old. Material fields are static
  // and the remaining fields (DIV E ERR, DIV B ERR and RHOF) are for
  // debugging purposes. By default, field dump filenames are tagged with
  // step(). However, if a "0" is added to the call, the filename will not be
  // tagged. The JF that gets stored is accumulated with a charge-conserving
  // algorithm. As a result, JF is not valid until at least one timestep has
  // been completed. Field dumps are in a binary format. Each rank makes a
  // field dump.
  if( step()==-10 )         dump_fields("fields/fields"); // Get first valid total J
  if( should_dump(fields) ) field_dump(global->fdParams);

  // Hydro dumps store particle charge density, current density and
  // stress-energy tensor. All these quantities are known at the time
  // t = time().  All these quantities are accumulated trilinear
  // node-centered. By default, species dump filenames are tagged with
  // step(). However, if a "0" is added to the call, the filename will not
  // be tagged. Note that the current density accumulated by this routine is
  // purely diagnostic. It is not used by the simulation and it is not
  // accumulated using a self-consistent charge-conserving method. Hydro dumps
  // are in a binary format. Each rank makes a hydro dump.
  // if( should_dump(ehydro) ) dump_hydro("electron","ehydro");
	if(should_dump(ihydro)) hydro_dump("ion", global->iParams);
	if(should_dump(ihydro)) hydro_dump("dion", global->diParams);


  // Particle dumps store the particle data for a given species. The data
  // written is known at the time t = time().  By default, particle dumps
  // are tagged with step(). However, if a "0" is added to the call, the
  // filename will not be tagged. Particle dumps are in a binary format.
  // Each rank makes a particle dump.
  //if( should_dump(eparticle) ) dump_particles("electron","eparticle");
  //if( should_dump(iparticle) ) dump_particles("ion",     "iparticle");

  // A checkpt is made by calling checkpt( fbase, tag ) where fname is a string
  // and tag is an integer.  A typical usage is:
  //   checkpt( "checkpt", step() ).
  // This will cause each process to write their simulation state to a file
  // whose name is based on fbase, tag and the node's rank.  For the above
  // usage, if called on step 314 on a 4 process run, the four files:
  //   checkpt.314.0, checkpt.314.1, checkpt.314.2, checkpt.314.3
  // to be written.  The simulation can then be restarted from this point by
  // invoking the application with "--restore checkpt.314".  checkpt must be
  // the _VERY_ LAST_ diagnostic called.  If not, diagnostics performed after
  // the checkpt but before the next timestep will be missed on restore.
  // Restart dumps are in a binary format unique to the each simulation.

  if( should_dump(restart) ) checkpt( "restore/checkpt", 0*step() );

  // If you want to write a checkpt after a certain amount of simulation time,
  // use uptime() in conjunction with checkpt.  For example, this will cause
  // the simulation state to be written after 7.5 hours of running to the
  // same file every time (useful for dealing with quotas on big machines).
  //if( uptime()>=27000 ) {
  //  checkpt( "timeout", 0 );
  //  abort(0);
  //}

# undef should_dump

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
