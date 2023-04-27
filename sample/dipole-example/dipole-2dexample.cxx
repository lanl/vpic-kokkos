// Hybrid Dipole + solar wind.  
// Ari Le, LANL, 2023


// If you want to use global variables (for example, to store the dump
// intervals for your diagnostics section), it must be done in the globals
// section. Variables declared the globals section will be preserved acrosst
// restart dumps. For example, if the globals section is:
//   begin_globals {
//     double variable;
//   } end_globals
// the double "variable" will be visible to other input deck sections as
// "global->variable". Note: Variables declared in the globals section are set
// to zero before the user's initialization block is executed. Up to 16K
// of global variables can be defined.

#include "injection.cxx"   //  Subroutine to compute re-injection velocity

#define NUM_TURNSTILES 1024

begin_globals {
  double energies_interval;
  double fields_interval;
  double ehydro_interval;
  double ihydro_interval;
  double eparticle_interval;
  double iparticle_interval;
  double restart_interval;

  double topology_x;       // domain topology 
  double topology_y;
  double topology_z;

  int rtoggle;  //for restarts
  int check_quota_interval;
  int quota_sec;

  //  Variables for new output format

  DumpParameters fdParams;
  DumpParameters iParams;
  DumpParameters diParams;

  std::vector<DumpParameters *> outputParams;


// Variables for Open BC Model
  double nb;      // Background density
  int nsp;        // Number of Species
  double vth[1];  // Thermal velocity of background components
  double q[1];    // Species charge
  double uf[3];   // Initial Fluid Drift
  double bimf[3]; // upstream IMF magnetic fiel
  double nfac; // Normalization factor to convert particles per cell to density
  int left,right,top,bottom;  // Keep track of boundary domains
  // Moments for bottom injectors
  double *nbot, *ubot, *pbot, *bbot, *fbot;
  // Moments for top injectors
  double *ntop, *utop, *ptop, *btop, *ftop;
  // Moments for left injectors
  double *nleft, *uleft, *pleft, *bleft, *fleft;
  // Moments for right injectors
  double *nright, *uright, *pright, *bright, *fright;

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
  double di   = 1; // use Alfvenic units

  // Physics parameters
  double mi_me   = 1; // NOT used. Ion mass / electron mass
  double Ti_Te   = 1;    // Ion temperature / electrSon temperature
  double beta_b  = 0.5;  //ion background beta based on B=1
  double R_P     =  80.0;  //planet radius in di
  double RM_RP   = 1.3;  //2D magnetopuase standoff/planet radius
  double bximf   =  cos(20.*M_PI/180);
  double byimf   =  0*sin(17.0*M_PI/180);
  double bzimf   = sin(20.*M_PI/180);
  double vdx     = 10.0;
  double vdy     = 0;
  double vdz     = 0;
  double taui    = 180; //ion cyclotron times to run

  // Numerical parameters
  double Lx        = 1600*di;   // How big should the box be in the x direction
  double Ly        = 1*di;     // How big should the box be in the y direction
  double Lz        = 3200*di;  // How big should the box be in the z direction
  double nx        = 800;      // Global resolution in the x direction
  double ny        = 1;        // Global resolution in the y direction
  double nz        = 1600;     // Global resolution in the z direction
  double nppc      = 100;       // Average number of macro particles per cell per species
  double tx        = 16;
  double ty        = 1;
  double tz        = 32;

  double quota     = 12.0; //hours. 

  // Set planet and dipole location

  double x_P = 0 + 0.1*Lx;
  double y_P = 0;
  double z_P = 0;

  double x_d = x_P;
  double y_d = y_P;
  double z_d = z_P;

  // Derived quantities

  double MD   = sqrt(2.0*vdx*vdx)*RM_RP*RM_RP*R_P*R_P;  //rho*v^2 = B^2/mu0
  double mi   = me*mi_me;                      // Ion mass
  double kTe  = beta_b/2;                      // Electron temperature
  double kTi  = kTe*Ti_Te;                     // Ion temperature
  double v_A  = (1);                            // based on n0
  double mib = mi;                             //Ion mass
  double Teb = 0.5*beta_b;                     //background Electron temperature
  double Tib = Teb;                            //Backround Ion temperature 
  double vthib = sqrt(Tib/mib);                // Ion thermal velocity in sheath
  double b0   = 1.;                            // Asymptotic magnetic field strength
  double n0   = 1.;                            // Peak electron density (also peak ion density)
  double Npe  = n0*Ly*Lz*Lx;                   // Number of physical electrons in box
  double Npi  = Npe;                           // Number of physical ions in box
  double Ne   = nppc*nx*ny*nz;                 // Total macro electrons in box
  Ne = trunc_granular(Ne,nproc());             // Make it divisible by number of processors
  double Ni   = Ne;                            // Total macro ions in box
  double we   = Npe/Ne;                        // Weight of a macro electron
  double wi   = Npi/Ni;                        // Weight of a macro ion
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;
  double nfac = wi/(hx*hy*hz);            // Convert density to particles per cell
 

  // need to assign global topology before calls to RANK_TO_INDEX

  global->topology_x  = tx;  
  global->topology_y  = ty;  
  global->topology_z  = tz;  

//  Determine which domains area along the boundaries - Use macro from grid/partition.c

# define RANK_TO_INDEX(rank,ix,iy,iz) BEGIN_PRIMITIVE {                   \
    int _ix, _iy, _iz;                                                    \
    _ix  = (rank);                        /* ix = ix+gpx*( iy+gpy*iz ) */ \
    _iy  = _ix/int(global->topology_x);   /* iy = iy+gpy*iz */                    \
    _ix -= _iy*int(global->topology_x);   /* ix = ix */                           \
    _iz  = _iy/int(global->topology_y);   /* iz = iz */                           \
    _iy -= _iz*int(global->topology_y);   /* iy = iy */ 	        	  \
    (ix) = _ix;                                                           \
    (iy) = _iy;                                                           \
    (iz) = _iz;                                                           \
  } END_PRIMITIVE 


  int ix, iy, iz, left=0,right=0,top=0,bottom=0;
  RANK_TO_INDEX( int(rank()), ix, iy, iz );
  if ( ix ==0 ) left=1;
  if ( ix ==tx-1 ) right=1;
  if ( iz ==0 ) bottom=1;
  if ( iz ==tz-1 ) top=1;

  // Determine the timestep
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);      // Courant length
  double dt = 0.002;                           // time step
  //if( wpe*dt>wpedt_max ) dt=wpedt_max/wpe;            // Override time step if plasma frequency limited

  ////////////////////////////////////////
  // Setup high level simulation parmeters

  num_step             = int(taui/dt);
  status_interval      = int(2./dt);
  sync_shared_interval = status_interval;
  clean_div_e_interval = status_interval;
  clean_div_b_interval = status_interval;

  global->rtoggle=1;
  global->check_quota_interval=100;
  global->quota_sec=int(quota*3600.0);

  global->energies_interval  = status_interval;
  global->fields_interval    = status_interval;
  global->ehydro_interval    = status_interval;
  global->ihydro_interval    = status_interval;
  global->eparticle_interval = status_interval; // Do not dump
  global->iparticle_interval = 50*status_interval; // Do not dump
  global->restart_interval   = status_interval*5.0; // Do not dump


// Parameters for the open boundary model
  global->nsp = 1;
  global->nb  = 1;
  global->vth[0]  = vthib;
  global->q[0]  = wi;
  global->nfac  = nfac;
  global->uf[0] = vdx;
  global->uf[1] = vdy;
  global->uf[2] = vdz;

  global->bimf[0] = bximf;
  global->bimf[1] = byimf;
  global->bimf[2] = bzimf;


  global->left = left;
  global->right = right;
  global->top = top;
  global->bottom = bottom;


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
  grid->eta = 1.0e-4;
  grid->hypereta=5.0e-3;


  grid->gamma = 1.0; // electron adiabatic index
  grid->nsub = 5.0; //number of field subcycles
  grid->nsmb=2000; //smooth B every nsmooth steps
  grid->nsm=0;     // 0  = don't use field/moment smoothing
  
  define_material( "vacuum", 1 );

  //first number epsx multiplies E in hyb_advance_e, 
  //second numberepsy multiples eta_hyper in hyb_hypereta
  
  material_t * planet = define_material("planet",0.,0.,1.,
					         1.,1.,1.,
					         0.,0.,0.);

  material_t * layer = define_material("layer", 1.,50,1.,
					         1.,1.,1.,
					         0.,0.,0.);

  material_t * layer2 = define_material("layer2", 1.,100.,1.,
					         1.,1.,1.,
					         0.,0.,0.);

  // Note: define_material defaults to isotropic materials with mu=1,sigma=0
  // Tensor electronic, magnetic and conductive materials are supported
  // though. See "shapes" for how to define them and assign them to regions.
  // Also, space is initially filled with the first material defined.

  // If you pass NULL to define field array, the standard field array will
  // be used (if damp is not provided, no radiation damping will be used).
  define_field_array( NULL);



 // ***** Set Boundary Conditions *****

 sim_log("Absorbing fields on X & Z-boundaries");

  if ( ix==0 )
    set_domain_field_bc( BOUNDARY(-1,0,0),pec_fields );
  if ( ix==tx-1 )
   set_domain_field_bc( BOUNDARY( 1,0,0), pec_fields );
  if ( iz==0 )
    set_domain_field_bc( BOUNDARY(0,0,-1), pec_fields );
  if ( iz==tz-1 )
   set_domain_field_bc( BOUNDARY( 0,0,1), pec_fields );
  

 // ***** Set Particle Boundary Conditions *****


  sim_log("Absorb particles on X & Z-boundaries");
  if ( iz==0 )
    set_domain_particle_bc( BOUNDARY(0,0,-1), absorb_particles );
  if ( iz==tz-1 )
    set_domain_particle_bc( BOUNDARY(0,0,1), absorb_particles );
 
  if ( ix==0 )
    set_domain_particle_bc( BOUNDARY(-1,0,0), absorb_particles );
  if ( ix==tx-1 )
    set_domain_particle_bc( BOUNDARY(1,0,0), absorb_particles );


  // Absorbing inner BC + dissipative layers

#define R2P ( (x-x_P)*(x-x_P) + (y-y_P)*(y-y_P)  + (z-z_P)*(z-z_P) )

# define INSIDE_PLANET (R2P < R_P*R_P)

   set_region_material(INSIDE_PLANET, planet, planet);

# define INSIDE_LAYER ( (R2P > (R_P)*(R_P)) && (R2P < (R_P+20*hx)*(R_P+20*hz) ) )

  set_region_material(INSIDE_LAYER, layer2, layer2);
    set_region_bc(INSIDE_PLANET, absorb_particles, absorb_particles,absorb_particles);


# define BOUNDARY_LAYER ( (x<-0.48*Lx || z<-0.48*Lz || z>0.48*Lz) )

  set_region_material(BOUNDARY_LAYER, layer, layer);

# define BOUNDARY_LAYER2 ( ( x>0.48*Lx ) )

  set_region_material(BOUNDARY_LAYER2, layer2, layer2);

  
  ////////////////////
  // Setup the species
  double nmax = 10*Ni/nproc();
  double nmovers = 0.1*nmax;
  double sort_interval = 40;

  // Allow 50% more local_particles in case of non-uniformity
  // VPIC will pick the number of movers to use for each species
  // Both species use out-of-place sorting
  species_t * ion      = define_species( "ion",  ec, mi, nmax, nmovers, sort_interval, 1 );
  //species_t * dion     = define_species( "dion", ec, mid, nmax,          -1, 40, 1 );

  //species_t * electron = define_species( "electron", -ec, me, 1.5*Ne/nproc(), -1, 20, 1 );

  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation

  sim_log( "" );
  sim_log( "System of units" );
  sim_log( "ec = " << ec );
  sim_log( "me = " << me );
  sim_log( "c = " << c );
  sim_log( "eps0 = " << eps0 );
  sim_log( "" );
  sim_log( "Physics parameters" );
  sim_log( "Ti/Te = " << Ti_Te );
  sim_log( "mi/me = " << mi_me );
  sim_log( "" );
  sim_log( "Numerical parameters" );
  sim_log( "num_step = " << num_step );
  sim_log( "dt = " << dt );
  sim_log( "Lx = " << Lx );
  sim_log( "Ly = " << Ly  );
  sim_log( "Lz = " << Lz  );
  sim_log( "nx = " << nx << ", dx = " << Lx/nx  );
  sim_log( "ny = " << ny << ", dy = " << Ly/ny  );
  sim_log( "nz = " << nz << ", dz = " << Lz/nz  );
  sim_log( "nppc = " << nppc );
  sim_log( "courant = " << c*dt/dg );
  sim_log( "" );
  sim_log( "Ion parameters" );
  sim_log( "qpi = "  << ec << ", mi = " << mi << ", qpi/mi = " << ec/mi );
  sim_log( "vthib = " << vthib << ", vthib/c = " << vthib/c << ", kTi = " << kTi  );
  sim_log( "Npi = " << Npi << ", Ni = " << Ni << ", Npi/Ni = " << Npi/Ni << ", wi = " << wi );
  sim_log( "" );
  sim_log( "Electron parameters" );
  sim_log( "qpe = "  << -ec << ", me = " << me << ", qpe/me = " << -ec/me );
  sim_log( "Npe = " << Npe << ", Ne = " << Ne << ", Npe/Ne = " << Npe/Ne << ", we = " << we );
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
    fp_info.write(&mib, 1 );
    fp_info.write(&vthib, 1 );
    fp_info.write(&vthib, 1 );
    fp_info.write(&status_interval, 1 );
    fp_info.close();

}

  ////////////////////////////
  // Load fields and particles

  sim_log( "Loading fields" );

  double c_d=0.25*hz;
 
  double thet = 0*M_PI;
     
#define xgd (x-x_d)
#define zgd (z-z_d)
#define xg (xgd*cos(thet) - zgd*sin(thet))
#define zg (xgd*sin(thet) + zgd*cos(thet))

#define DM (-MD/c_d)
#define XD (xg)
#define ZDT (zg-c_d/2.0)
#define ZDB (zg+c_d/2.0)
#define DBX (DM*(XD/(XD*XD+ZDT*ZDT))-DM*(XD/(XD*XD+ZDB*ZDB)))
#define DBZ (DM*(ZDT/(XD*XD+ZDT*ZDT))-DM*(ZDB/(XD*XD+ZDB*ZDB)))



  //MIRROR DIPOLE

#define MXD  (XD + 2*x_d + Lx)
#define MDM  (-DM)
#define MZDT (ZDT)
#define MZDB (ZDB)
#define MDBX (MDM*(MXD/(MXD*MXD+MZDT*MZDT))-MDM*(MXD/(MXD*MXD+MZDB*MZDB)))
#define MDBZ (MDM*(MZDT/(MXD*MXD+MZDT*MZDT))-MDM*(MZDB/(MXD*MXD+MZDB*MZDB)))


    set_region_field( everywhere, 0, 0, 0,      // Eletric field (Doesn't matter)
    0,0,0); // Magnetic field


    set_region_bext(everywhere, DBX+MDBX+bximf, byimf , DBZ+MDBZ+bzimf); // Magnetic field
    

    // Note: everywhere is a region that encompasses the entire simulation
  // In general, regions are specied as logical equations (i.e. x>0 && x+y<2)
  
  sim_log( "Loading particles" );

  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);
  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);
  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);

  repeat( Ni/nproc() ) {
    double x, y, z, ux, uy, uz, d0, vfactor;

    // Pick an appropriately distributed random location for the pair
    //do {
    //  x = L*atanh( uniform( rng(0), -1, 1 ) );
    //} while( x<=-0.5*Lx || x>=0.5*Lx );
    
    x = uniform( rng(0), xmin, xmax );
    y = uniform( rng(0), ymin, ymax );
    z = uniform( rng(0), zmin, zmax );
    
    if (!INSIDE_PLANET) {
    // For the ion, pick an isothermal normalized momentum in the drift frame
    // (this is a proper thermal equilibrium in the non-relativistic limit),
    // boost it from the drift frame to the frame with the magnetic field
    // along z and then rotate it into the lab frame. Then load the particle.
    // Repeat the process for the electron.

      vfactor = 1.0 - 0.5*exp(-((x-x_P)*(x-x_P)+(z-z_P)*(z-z_P))/R_P/R_P);
    ux = normal( rng(0), 0, vthib ) + vfactor*vdx;
    uy = normal( rng(0), 0, vthib ) + vfactor*vdy;
    uz = normal( rng(0), 0, vthib ) + vfactor*vdz;
    inject_particle( ion, x, y, z, ux, uy, uz, wi, 0, 0 );
    }

    //ux = normal( rng(0), 0, uthe );
    //uy = normal( rng(0), 0, uthe );
    //uz = normal( rng(0), 0, uthe );
    //d0 = gdre*uy + sqrt(ux*ux+uy*uy+uz*uz+1)*udre;
    //uy = d0*cs - uz*sn;
    //uz = d0*sn + uz*cs;
    //inject_particle( electron, x, y, z, ux, uy, uz, we, 0, 0 );
  }


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
	//sprintf(global->diParams.baseDir, "hydro");
	
	// base file name for hydro  output
	sprintf(global->iParams.baseFileName, "ihydro");
	//sprintf(global->diParams.baseFileName, "dihydro");

	global->iParams.stride_x = 1;
	global->iParams.stride_y = 1;
	global->iParams.stride_z = 1;

	//global->diParams.stride_x = 1;
	//global->diParams.stride_y = 1;
	//global->diParams.stride_z = 1;

	sim_log ( "Ion species x-stride " << global->iParams.stride_x );
	sim_log ( "Ion species y-stride " << global->iParams.stride_y );
	sim_log ( "Ion species z-stride " << global->iParams.stride_z );

	global->iParams.output_variables( current_density | charge_density | stress_tensor );
	//global->diParams.output_variables( current_density | charge_density | stress_tensor );

	create_hydro_list(varlist, global->iParams);
	sim_log ( "Ion  species variable list: " << varlist );
	//create_hydro_list(varlist, global->diParams);
	//sim_log ( "dIon  species variable list: " << varlist );

	// add ion  species parameters to list
	global->outputParams.push_back(&global->iParams);
	//global->outputParams.push_back(&global->diParams);
}

begin_diagnostics {

  if (step()==0 ) {
	  global_header("global", global->outputParams);
   if (step( )==0 ) {
	  dump_mkdir("fields");
	  dump_mkdir("hydro");
	  dump_mkdir("rundata");
	  dump_mkdir("restore0");
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
  const int nsp = global->nsp;
  const int nx = grid->nx;
  const int ny = grid->ny;
  const int nz = grid->nz;

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
	//if(should_dump(ihydro)) hydro_dump("dion", global->diParams);


  // Particle dumps store the particle data for a given species. The data
  // written is known at the time t = time().  By default, particle dumps
  // are tagged with step(). However, if a "0" is added to the call, the
  // filename will not be tagged. Particle dumps are in a binary format.
  // Each rank makes a particle dump.
  //if( should_dump(eparticle) ) dump_particles("electron","eparticle");
	//if( step()>0 && should_dump(iparticle) ) dump_particles("ion",     "particle/iparticle");

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

	
	if( should_dump(restart) && step()>0) {
               if (global->rtoggle==1) {
                  global->rtoggle=2;
	          checkpt( "restore1/checkpt", 0);
	          DUMP_INJECTORS(1);
                }
               else {
                  global->rtoggle=1;
	          checkpt( "restore2/checkpt", 0);
	          DUMP_INJECTORS(2);
                }        
	}

  // If you want to write a checkpt after a certain amount of simulation time,
  // use uptime() in conjunction with checkpt.  For example, this will cause
  // the simulation state to be written after 7.5 hours of running to the
  // same file every time (useful for dealing with quotas on big machines).
	//global->quota_sec=600;
    if( step()>0 && global->check_quota_interval>0 && (step()&global->check_quota_interval)==0 ) {
      if( uptime() > global->quota_sec ) {
      sim_log( "Allowed runtime exceeded for this job.  Terminating....\n");
               if (global->rtoggle==1) {
                  BEGIN_TURNSTILE(NUM_TURNSTILES){
                  global->rtoggle=2;
	          checkpt( "restore1/checkpt", 0);
	          DUMP_INJECTORS(1);
                  } END_TURNSTILE;
                }
               else {
                  BEGIN_TURNSTILE(NUM_TURNSTILES){
                  global->rtoggle=1;
	          checkpt( "restore2/checkpt", 0);
	          DUMP_INJECTORS(2);
                  } END_TURNSTILE;
                }
      sim_log( "Restart dump restart completed." );
      mp_barrier();
      halt_mp();
      exit(0); // Exit or abort?
    }
  }

#undef should_dump

} //end diagnostics

// *******************  PARTICLE INJECTION  - OPEN BOUNDARY ********************
begin_particle_injection {
  int inject;
  double x, y, z, ux,uy,uz,age, flux, vtherm, vd;
  double uv[3];
    double zcell;
  const int nsp=global->nsp;
  const int nx=grid->nx;
  const int ny=grid->ny;
  const int nz=grid->nz;
  const double sqpi =1.772453850905516;
  const double dt=grid->dt;
  const double hx=grid->dx;
  const double hy=grid->dy;
  const double hz=grid->dz;
  const double nb=global->nb;
  const double nfac=global->nfac;
  const double vinjx = global->uf[0];
  const double vinjy = global->uf[1];
  const double vinjz = global->uf[2];


  // Initialize the injectors on the first call
  static int initted=0;
  if ( !initted ) {

    initted=1;

    if (rank() == 0)
      MESSAGE(("------------Initializing the Particle Injectors-------------"));

    // Intialize injectors

    if (global->right) {

      DEFINE_INJECTOR(right,ny,nz);

      if (step()==0) {
        for ( int n=1; n<=nsp; n++ ) {
          double cn = (uf(2)/vth(2))/(vth(n)/vth(2));
          for ( int k=1;k<=nz; k++ ) {
            for ( int j=1;j<=ny; j++ ) {
		bright(n,k,j) = 0;
		nright(n,k,j) = nb/nfac;
		//fright(n,k,j) = (nb*sqrt(2)*vth(n))/(2*hx*sqpi*nfac);
		uright(1,n,k,j) = vinjx;
		uright(2,n,k,j) = vinjy;
		uright(3,n,k,j) = vinjz;
		pright(1,2,n,k,j)=pright(2,1,n,k,j)=pright(1,3,n,k,j)
		  =pright(3,1,n,k,j)=pright(2,3,n,k,j)=pright(3,2,n,k,j)=0;
		pright(1,1,n,k,j) = (nb*2*vth(n)*vth(n))/(2*nfac);
		pright(2,2,n,k,j) = (nb*2*vth(n)*vth(n))/(2*nfac);
		pright(3,3,n,k,j) = pright(1,1,n,k,j);
            }
          }
        }  // end for
      } // endif

        else {
      switch (global->rtoggle) {
      case 1: READ_INJECTOR(right,ny,nz,2); break;
      case 2: READ_INJECTOR(right,ny,nz,1); break;
      default: ERROR(("Bad rtoggle for injector read"));
        }
     }

    } //end right boundary

    if (global->left) {

      DEFINE_INJECTOR(left,ny,nz);

      if (step()==0) {
        for ( int n=1; n<=nsp; n++ ) {
           for ( int k=1;k<nz+1; k++ ) {
            for ( int j=1;j<=ny; j++ ) {
		bleft(n,k,j) = 0;
		nleft(n,k,j) = nb/nfac;
		//fleft(n,k,j) = (nb*sqrt(2)*vth(n))/(2*hx*sqpi*nfac);
		uleft(1,n,k,j) = vinjx;
		uleft(2,n,k,j) = vinjy;
		uleft(3,n,k,j) = vinjz;
		pleft(1,2,n,k,j)=pleft(2,1,n,k,j)=pleft(1,3,n,k,j)=pleft(3,1,n,k,j)
		  =pleft(2,3,n,k,j)=pleft(3,2,n,k,j)=0;
		pleft(1,1,n,k,j) = (nb*2*vth(n)*vth(n))/(2*nfac);
		pleft(2,2,n,k,j) = (nb*2*vth(n)*vth(n))/(2*nfac);
		pleft(3,3,n,k,j) = pleft(1,1,n,k,j);
            }
          }
        } // end for
      } //endif

    else {
      switch (global->rtoggle) {
      case 1: READ_INJECTOR(left,ny,nz,2); break;
      case 2: READ_INJECTOR(left,ny,nz,1); break;
        }
     }

    } // end left boundary

    if (global->top) {

      DEFINE_INJECTOR(top,ny,nx);

      if (step()==0) {
        for ( int n=1; n<=nsp; n++ ) {
          for ( int i=1;i<=nx; i++ ) {
            for ( int j=1;j<=ny; j++ ) {
              btop(n,i,j) = 0;
              ntop(n,i,j) = nb/nfac;
              //ftop(n,i,j) = ntop(n,i,j)*sqrt(2)*vth(n)/(2*hz*sqpi);
              utop(1,n,i,j) = vinjx;
              utop(2,n,i,j) = vinjy;
              utop(3,n,i,j) = vinjz;
              ptop(1,2,n,i,j)=ptop(2,1,n,i,j)=ptop(1,3,n,i,j)=ptop(3,1,n,i,j)
                =ptop(2,3,n,i,j)=ptop(3,2,n,i,j)=0;
              ptop(1,1,n,i,j) = ntop(n,i,j)*2*vth(n)*vth(n)/2;;
              ptop(2,2,n,i,j) = ntop(n,i,j)*2*vth(n)*vth(n)/2;;
              ptop(3,3,n,i,j) = ntop(n,i,j)*2*vth(n)*vth(n)/2;;
            }
          }
        } // end for
      } //endif

    else {
      switch (global->rtoggle) {
      case 1: READ_INJECTOR(top,ny,nx,2); break;
      case 2: READ_INJECTOR(top,ny,nx,1); break;
        }
     }

    } // end top boundary

    if (global->bottom) {

      DEFINE_INJECTOR(bot,ny,nx);

      if (step()==0) {
        for ( int n=1; n<=nsp; n++ ) {
          for ( int i=1;i<=nx; i++ ) {
            for ( int j=1;j<=ny; j++ ) {
              bbot(n,i,j) = 0;
              nbot(n,i,j) = nb/nfac;
              //fbot(n,i,j) = nbot(n,i,j)*sqrt(2)*vth(n)/(2*hz*sqpi);
              ubot(1,n,i,j) = vinjx;
              ubot(2,n,i,j) = vinjy;
              ubot(3,n,i,j) = vinjz;
              pbot(1,2,n,i,j)=pbot(2,1,n,i,j)=pbot(1,3,n,i,j)
                =pbot(3,1,n,i,j)=pbot(2,3,n,i,j)=pbot(3,2,n,i,j)=0;
              pbot(1,1,n,i,j) = nbot(n,i,j)*2*vth(n)*vth(n)/2;
              pbot(2,2,n,i,j) = nbot(n,i,j)*2*vth(n)*vth(n)/2;
              pbot(3,3,n,i,j) = nbot(n,i,j)*2*vth(n)*vth(n)/2;
            }
          }
        } // end for
      } //endif

    else {
      switch (global->rtoggle) {
      case 1: READ_INJECTOR(bot,ny,nx,2); break;
      case 2: READ_INJECTOR(bot,ny,nx,1); break;
        }
     }

    }  // end bottom boundary

    if (rank() == 0)
      MESSAGE(("------------------------------------------------------------"));

  } // End of Intialization

  // Inject particles on Left Boundary
  if (global->left) {
    for ( int n=1; n<=nsp; n++ ) {
      species_t * species = find_species_id(n-1,species_list );
      for ( int k=1;k<=nz; k++ ) {
        for ( int j=1;j<=ny; j++ ) {
	  vtherm = sqrt(2.0)*vth(n);
	  vd = vinjx/vtherm;
	  bleft(n,k,j) = bleft(n,k,j) + dt*nleft(n,k,j)*vtherm*(exp(-vd*vd)/sqpi + vd*(erf(vd)+1))/(2*hx);
          inject = (int) bleft(n,k,j);
          bleft(n,k,j) = bleft(n,k,j) - (double) inject;
	  double uflow[3] = {uleft(1,n,k,j),uleft(2,n,k,j),uleft(3,n,k,j)};
	  double press[9] = {pleft(1,1,n,k,j),pleft(1,2,n,k,j),pleft(1,3,n,k,j),
			       pleft(2,1,n,k,j),pleft(2,2,n,k,j),pleft(2,3,n,k,j),
			       pleft(3,1,n,k,j),pleft(3,2,n,k,j),pleft(3,3,n,k,j)};
   	  repeat(inject) {
            x = grid->x0;
            y = grid->y0 + hy*(j-1) + hy*uniform( rng(0), 0.0, 1.0 );
            z = grid->z0 + hz*(k-1) + hz*uniform( rng(0), 0.0, 1.0 );
            age = 0;
	    //ux = vth(n)*sqrt(-log(uniform(rng(0),0,0.99999)));
	    //uy =normal(rng(0),0,vth(n));
	    //uz =normal(rng(0),0,vth(n));
	    compute_injection(uv,nleft(n,k,j),uflow,press,1,2,3,rng(0));
	    inject_particle(species, x, y, z, uv[0], uv[1], uv[2], q(n), age, 0 );
	    //if (rank()==0) MESSAGE((" Injecting left  --> ux=%e",ux));     
          }
        }
      }
    }
  } // end left injector

  // Inject particles on Right Boundary

  if (global->right) {
    for ( int n=1; n<=nsp; n++ ) {
      species_t * species = find_species_id(n-1,species_list );
      for ( int k=1;k<=nz; k++ ) {
        for ( int j=1;j<=ny; j++ ) {
	  vtherm = sqrt(2.0)*vth(n);
	  vd = -vinjx/vtherm;
	  bright(n,k,j) = bright(n,k,j) + dt*nright(n,k,j)*vtherm*(exp(-vd*vd)/sqpi + vd*(erf(vd)+1))/(2*hx);	    
	  inject = (int) bright(n,k,j);
          bright(n,k,j) = bright(n,k,j) - (double) inject;
	    double uflow[3] = {uright(1,n,k,j),uright(2,n,k,j),uright(3,n,k,j)};
	    double press[9] = {pright(1,1,n,k,j),pright(1,2,n,k,j),pright(1,3,n,k,j),
			       pright(2,1,n,k,j),pright(2,2,n,k,j),pright(2,3,n,k,j),
			       pright(3,1,n,k,j),pright(3,2,n,k,j),pright(3,3,n,k,j)};	  
	  repeat(inject) {
            x = grid->x1;
            y = grid->y0 + hy*(j-1) + hy*uniform( rng(0), 0.0, 1.0 );
            z = grid->z0 + hz*(k-1) + hz*uniform( rng(0), 0.0, 1.0 );
            age = 0;
	    //ux = -vth(n)*sqrt(-log(uniform(rng(0),0,0.99999)));
	    //uy =normal(rng(0),0,vth(n));
	    //uz =normal(rng(0),0,vth(n));
	    compute_injection(uv,nright(n,k,j),uflow,press,-1,2,3,rng(0));
	    inject_particle(species, x, y, z, uv[0], uv[1], uv[2], q(n), age, 0 );

          }
        }
      }
    }
  } // end right injector

  // Inject particles on Top Boundary

  if (global->top) {
    for ( int n=1; n<=nsp; n++ ) {
      species_t * species = find_species_id(n-1,species_list );
      for ( int i=1;i<=nx; i++ ) {
        for ( int j=1;j<=ny; j++ ) {
	  vtherm = sqrt(2.0)*vth(n);
	  vd = -vinjz/vtherm;
	  btop(n,i,j) = btop(n,i,j) + dt*ntop(n,i,j)*vtherm*(exp(-vd*vd)/sqpi + vd*(erf(vd)+1))/(2*hz);	    
          inject = (int) btop(n,i,j);
          btop(n,i,j) = btop(n,i,j) - (double) inject;
	  double uflow[3] = {utop(1,n,i,j),utop(2,n,i,j),utop(3,n,i,j)};
	  double press[9] = {ptop(1,1,n,i,j),ptop(1,2,n,i,j),ptop(1,3,n,i,j),
			       ptop(2,1,n,i,j),ptop(2,2,n,i,j),ptop(2,3,n,i,j),
			       ptop(3,1,n,i,j),ptop(3,2,n,i,j),ptop(3,3,n,i,j)};	    
	  repeat(inject) {
            x = grid->x0 + hx*(i-1) + hx*uniform( rng(0), 0.0, 1.0 );
            y = grid->y0 + hy*(j-1) + hy*uniform( rng(0), 0.0, 1.0 );
            z = grid->z1;
            age = 0;
	    //ux = normal(rng(0),0,vth(n));
	    //uy = normal(rng(0),0,vth(n));
	    //uz = -vth(n)*sqrt(-log(uniform(rng(0),0,0.99999)));
	    compute_injection(uv,ntop(n,i,j),uflow,press,-3,1,2,rng(0));
    	    inject_particle(species, x, y, z, uv[0], uv[1], uv[2], q(n), age, 0 );



          }
        }
      }
    }
  }  // end top injector

  // Inject particles on Bottom Boundary
  if (global->bottom) {
    for ( int n=1; n<=nsp; n++ ) {
      species_t * species = find_species_id(n-1,species_list );
      for ( int i=1;i<=nx; i++ ) {
        for ( int j=1;j<=ny; j++ ) {
	  vtherm = sqrt(2.0)*vth(n);
	  vd = vinjz/vtherm;
	  bbot(n,i,j) = bbot(n,i,j) + dt*nbot(n,i,j)*vtherm*(exp(-vd*vd)/sqpi + vd*(erf(vd)+1))/(2*hz);
          inject = (int) bbot(n,i,j);
          bbot(n,i,j) = bbot(n,i,j) - (double) inject;
	  double uflow[3] = {ubot(1,n,i,j),ubot(2,n,i,j),ubot(3,n,i,j)};
	  double press[9] = {pbot(1,1,n,i,j),pbot(1,2,n,i,j),pbot(1,3,n,i,j),
			     pbot(2,1,n,i,j),pbot(2,2,n,i,j),pbot(2,3,n,i,j),
			     pbot(3,1,n,i,j),pbot(3,2,n,i,j),pbot(3,3,n,i,j)};
	  repeat(inject) {
            x = grid->x0 + hx*(i-1) + hx*uniform( rng(0), 0.0, 1.0 );
            y = grid->y0 + hy*(j-1) + hy*uniform( rng(0), 0.0, 1.0 );
            z = grid->z0;
            age = 0;
	    //ux = normal(rng(0),0,vth(n));
	    //uy = normal(rng(0),0,vth(n));
	    //uz = vth(n)*sqrt(-log(uniform(rng(0),0,0.99999)));
	    compute_injection(uv,nbot(n,i,j),uflow,press,3,1,2,rng(0));
	    inject_particle(species, x, y, z, uv[0], uv[1], uv[2], q(n), age, 0 );

          }
        }
      }
    }
  } // end bottom injector

    
} // end particle injection


begin_current_injection {

  // No current injection for this simulation

}

begin_field_injection {

  // No field injection for this simulation

 const int nx=grid->nx;
  const int ny=grid->ny;
  const int nz=grid->nz;
  int x,y,z;
  double bimfx = global->bimf[0];
  double bimfy = global->bimf[1];
  double bimfz = global->bimf[2];


  double vimfx = global->uf[0];
  double vimfy = global->uf[1];
  double vimfz = global->uf[2];

  double r = .0025;


  // There macros are from local.c to apply boundary conditions
#define XYZ_LOOP(xl,xh,yl,yh,zl,zh)             \
  for( z=zl; z<=zh; z++ )                       \
    for( y=yl; y<=yh; y++ )                     \
      for( x=xl; x<=xh; x++ )

#define EDGE_LOOP(x) XYZ_LOOP(x,x,1,ny,1,nz)
#define LOOP() XYZ_LOOP(1,nx,1,ny,1,nz)
  
  // LEFT Boundary
  if (global->left) {

    r=0.0025;
        
    //EDGE_LOOP(0) field(x,y,z).cbx = (1.0-r)*field(x,y,z).cbx + r*bimfx;
    EDGE_LOOP(0) field(x,y,z).cby = (1.0-r)*field(x,y,z).cby ;
    EDGE_LOOP(0) field(x,y,z).cbz = (1.0-r)*field(x,y,z).cbz ;
    
    //EDGE_LOOP(1) field(x,y,z).cbx = (1.0-r)*field(x,y,z).cbx + r*(field(x+1,y,z+1).cbx+field(x+1,y,z-1).cbx)/2.0;
    EDGE_LOOP(1) field(x,y,z).cby = (1.0-r)*field(x,y,z).cby ;
    EDGE_LOOP(1) field(x,y,z).cbz = (1.0-r)*field(x,y,z).cbz ;
    
    //}
    
  }
  
  // Right Boundary
  if (global->right) {

    r=0.01;
  EDGE_LOOP(nx) field(x,y,z).cby = (1.0-r)*field(x,y,z).cby + r*(field(x-1,y,z+1).cby+field(x-1,y,z-1).cby)/2.0;
  EDGE_LOOP(nx) field(x,y,z).cbz = (1.0-r)*field(x,y,z).cbz + r*(field(x-1,y,z+1).cbz+field(x-1,y,z-1).cbz)/2.0;    
   }
  
}

begin_particle_collisions{

  // No collisions for this simulation

}
