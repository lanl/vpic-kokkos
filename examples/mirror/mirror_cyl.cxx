/////////////////////////////////////////////////////
//
//   Gas Dynamic Trap - CYLINDRICAL VERSION
//
//////////////////////////////////////////////////////

#include "injection.cxx"   //  Subroutine to compute re-injection velocity

#define NUM_TURNSTILES 256

//////////////////////////////////////////////////////

begin_globals {

  int restart_interval;
  int energies_interval;
  int fields_interval;
  int ehydro_interval;
  int Hhydro_interval;
  int eparticle_interval;
  int Hparticle_interval;
  int quota_check_interval;  //  How frequently to check if quote exceeded
  int sort_interval;  //  How frequently to check if quote exceeded


  int rtoggle;             // enables save of last two restart dumps for safety
  double quota_sec;        // Run quota in seconds
  double b0;               // B0

  species_t *ion;
  species_t *electron;
  //species_t *beam;

  //int rng_seed;
  double qe;
  double qi;
  double vthe;
  double vthi;
  double ane;
  double ani;
  double Lx;  // Now Lr (radial extent)
  
  double topology_x;       // domain topology (now r, theta, z)
  double topology_y;
  double topology_z;

  // parameters for the collision model

  int    ee_collisions;         // Flag to signal we want to do e-e collisions. 
  int    ei_collisions;         // Flag to signal we want to do e-i collisions. 
  int    ii_collisions;         // Flag to signal we want to do i-i collisions. 

  double cvar;                  // Base variance (dimensionless) used in particle collision 
  double nppc_max;              // Max number of particles/cell (used to define key array size). 
  int tstep_coll;            // Collision interval (=multiple of sort interval). 
  double Z;
  double mi_me;
  double nfac;      //  Normalization factor to convert particles per cell into density

  //  Variables for Open BC Model
  int nsp;          //  Number of Species
  double npleft[2];  // Left Densities
  double npright[2]; // Right Densities
  double vth[2];    // Thermal velocity 
  double q[2];      // Species charge
  double ur;       //  Fluid velociy on right
  double ul;       //  Fluid velociy on left
  double ubx;       //  beam parallel velociy on left
  double ubz;       //  beam perp velociy on left
  double Iin;  
  double Iout;
  double xp;
  double zp;
  double r[2];     //density + temperature relaxation paramters
  double Lxp, Lzp;  //where to inject particles


  int left,right,top,bottom,center;  // Keep track of boundary domains

  double *nleft, *uleft, *pleft, *bleft, *fleft;     // Moments for left injectors
  double *nright, *uright, *pright, *bright, *fright; // Moments for right injectors
//  Variables for new output format

  DumpParameters fdParams;
  //DumpParameters hedParams;
  DumpParameters hHdParams;
  DumpParameters hBdParams;
  std::vector<DumpParameters *> outputParams;
};

begin_initialization {

 // Use natural PIC units

  double ec   = 1;         // Charge normalization
  double me   = 1;         // Mass normalization
  double c = 1;            // velocity normalization (speed of light) 
  double de = 1;           // Length normalization (electron inertial length) 
  double eps0 = 1;         // Permittivity of space


  // Plasma parameters

  double mime   = 1.0;    // Ion mass / electron mass. Not used in hybrid
  double TiTe   = 1.0;      // Ion temperature / electron temperature
  double Z   = 1.0;      // Ion charge
  double vthe = 0.05;
  double beta_e = 0.03;  // By setting vthe and beta_e, this determines wpe/wce below
  double ane = 1.0;
  double ani= 1.0;
  double n0 = 1.0;  // Scale to normalize density
  double nb = 0.03; //peak beam density


  // Simulation time and quota

  double taui    = 500;    // simulation wci's to run
  double quota   = 3.85;   // run quota in hours
  double quota_sec = quota*3600;  // Run quota in seconds
  
  //derived qunatities

  double wpe  = c/de;                    // electron plasma frequency
  double wpewce = sqrt(beta_e)/(vthe*sqrt(2.0));  // Electron plasma freq/Electron Cyclotron
  double wce  = wpe/wpewce;             // Electron cyclotron freqeuncy
  double b0   = me*c*wce/ec;            // Magnetic field strength
  double mi   = me*mime;                // Ion mass
  double vthi = vthe*sqrt(TiTe/mime);   //  Ion thermal velocity - based on parallel temperature 
  double ubx = sqrt(5.0)*vthi;          //beam parallel velocity;  
  double ubz = sqrt(5.0)*vthi;         //beam perp velocity;    
  double wci  = wce/mime;               // Ion cyclotron frequency
  double wpi  = wpe/sqrt(mime);         // ion plasma frequency
  double di   = c/wpi;                  // ion inertial length
  double beta_i = beta_e*sqrt(TiTe);

  // Simulation  parameters - NOW IN CYLINDRICAL (r, theta, z)

  double nppc = 400;        // Average number of macro particle per cell per species

  double Lr  = 30.*di;      // RADIAL extent (was Lx)
  double Ltheta  = 2*M_PI;  // Full azimuthal extent (was Ly)
  double Lz  = 300.*di;     // Axial extent (was Lz, now much longer for trap axis)

  double Lxp = Lr/4.5;      // Now radial injection size
  double Lzp = Lz/4.5;      // Axial injection size
  
  double topology_x = 1;    // r direction
  double topology_y = 1;    // theta direction 
  double topology_z = 4;    // z direction (more decomposition along axis)

  double nx = 104;          // Cells in r (was 1024)
  double ny = 1;            // Cells in theta (keep 1 for 2D axisymmetric)
  double nz = 1024;         // Cells in z (was 104, now along trap axis)

  double hx = Lr/nx;        // cell size in r
  double hy = Ltheta/ny;    // cell size in theta
  double hz = Lz/nz;        // cell size in z

  double Npart   = nppc*nx*ny*nz;  // total macro electrons in box
  Npart = trunc_granular(Npart,nproc()); // Make it divisible by number of processors

 double qe = -ec*Lr*Ltheta*Lz/Npart;  // Charge per macro electron (volume now cylindrical)
 double qi =  ec*Lr*Ltheta*Lz/Npart;  // Charge per macro electron       
 double nfac = qi/(hx*hy*hz);         // Convert density to particles per cell
      
  // Determine the time step

  double dt = 0.01/wci;            // time step
   
  //  Intervals for various output and checks

  int energies_interval = 100;
  int interval = int(20.0/(wci*dt)); 
  int fields_interval = interval;
  int ehydro_interval = interval;
  int Hhydro_interval = interval;
  int restart_interval = 5*interval;
  int eparticle_interval = 1000*interval;
  int Hparticle_interval = 1000*interval;
  int quota_check_interval = 100;
  
  
// User injection parameters (Kokkos)
  kokkos_field_injection = true; // Directly modify fields on device.
  field_injection_interval = 1;  
  kokkos_particle_injection = false; // To avoid copying whole particle array to host
  particle_injection_interval = -1;
  current_injection_interval = -1;


  // Collision parameters
  // In CGS, variance of tan theta = 2 pi e^4 n_e dt_coll loglambda / (m_ab^2 u^3)

  int ii_collisions = 0;    // Turn off collisions 
  int ee_collisions = 0;    // 
  int ei_collisions = 0;    // 

  double sort_interval = 20;        // Sort interval for particles
  int tstep_coll = (int) 5*sort_interval;     // How frequently to apply collision operator - Must be even multiple of sort frequency
  double dt_coll = dt*(tstep_coll);        // in (1/wpe)  - Want to keep tstep_col < 1
  double nuei_wce = 0.01;                  //  Set collision frequncy relative to cycltron frequency
  double cvar = dt_coll*3.0*sqrt(2.0*M_PI)/4.0*pow(vthe,3)*nuei_wce/wpewce;  
  double nppc_max   = 20*nppc;             // Max possible number of particles/cell of ea. species 
                                           // (to size the key array in collision handler)

  //  Determine which domains area along the boundaries - Use macro from grid/partition.c

# define RANK_TO_INDEX(rank,ix,iy,iz) BEGIN_PRIMITIVE {                   \
    int _ix, _iy, _iz;                                                    \
    _ix  = (rank);                        /* ix = ix+gpx*( iy+gpy*iz ) */ \
    _iy  = _ix/int(topology_x);   /* iy = iy+gpy*iz */                    \
    _ix -= _iy*int(topology_x);   /* ix = ix */                           \
    _iz  = _iy/int(topology_y);   /* iz = iz */                           \
    _iy -= _iz*int(topology_y);   /* iy = iy */ 	        	  \
    (ix) = _ix;                                                           \
    (iy) = _iy;                                                           \
    (iz) = _iz;                                                           \
  } END_PRIMITIVE 

  int ix, iy, iz, left=0, right=0, top=0, bottom=0,center=0;
  RANK_TO_INDEX( int(rank()), ix, iy, iz ); 
  if ( ix ==0 ) left=1;            // At r=0 (axis)
  if ( ix==topology_x-1) right=1;  // At r=Rmax (outer boundary)
  if ( iz ==0 ) bottom=1;          // At z=0 (bottom endcap)
  if ( iz ==topology_z-1 ) top=1;  // At z=Lz (top endcap)
  if ( abs( float(iz) - (topology_z/2.-0.5) ) < 4. ) center=1; // Center in z now


  ///////////////////////////////////////////////
  // Setup high level simulation parameters
  num_step             = int(taui/(wci*dt));
  status_interval      = 200;
  sync_shared_interval = status_interval/2;
  clean_div_e_interval = 2*status_interval;
  clean_div_b_interval = 2*status_interval;

  global->restart_interval   = restart_interval;
  global->energies_interval  = energies_interval; 
  global->fields_interval    = fields_interval;  
  global->ehydro_interval    = ehydro_interval;  
  global->Hhydro_interval    = Hhydro_interval;  
  global->eparticle_interval = eparticle_interval;  
  global->Hparticle_interval = Hparticle_interval;  
  global->quota_check_interval     = quota_check_interval;
  global->quota_sec          = quota_sec;  
  global->sort_interval      = sort_interval;  


  global->rtoggle            = 0;  
  global->b0  = b0;              

  global->topology_x  = topology_x;  
  global->topology_y  = topology_y;  
  global->topology_z  = topology_z;  

  global-> vthe = vthe;      // Electron thermal velocity - based on parallel temperature  
  global-> ane = ane;       //  Electron temp anisotropy  T_perp/T_par
  global-> ani = ani;       //  Ion temp anisotropy  T_perp/T_par
  global-> vthi = vthi; 
  global-> qe = qe;
  global-> qi = qi; 
  global-> Lx = Lr;         // Store as Lx but it's really Lr
  global-> Lxp = Lxp; 
  global-> Lzp = Lzp; 

  // Collision model parameters

  global->ee_collisions            = ee_collisions; 
  global->ii_collisions            = ii_collisions; 
  global->ei_collisions            = ei_collisions; 

  global->cvar                     = cvar; 
  global->nppc_max                 = nppc_max;
  global->tstep_coll               = tstep_coll; 
  global->mi_me                    = mime; 
  global->Z                        = Z;
  global->nfac                     = nfac; 

  //  Parameters for the open boundary model 

  global->nsp = 2;

  global->ur  = 0.0;  
  global->ul  = 0.0;
  global->ubx  = ubx;
  global->ubz  = ubz;  

  global->q[0]  = qi;  
  global->q[1]  = qi;  
 
  global->npleft[0]  = n0;
  global->npleft[1]  = nb;

  global->npright[0]  = n0;
  global->npright[1]  = 0;

  global->vth[0]  = sqrt(2)*vthi;

  global->left = left;
  global->right = right;
  global->top = top;
  global->bottom = bottom;
  global->center = center;

  global->r[0] = 0.01;
  global->r[1] = 0.01;


  ////////////////////////////////////////////////////////////////////////////////////////////
  // Setup the grid - CYLINDRICAL

  // Setup basic grid parameters
  grid->dx = hx;
  grid->dy = hy;
  grid->dz = hz;
  grid->dt = dt;
  grid->cvac = c;
  grid->eps0 = eps0;


  grid->eta = 1.0e-3;
  grid->hypereta=1.0e-4;

  grid->eos_den = 1.0;
  grid->eos_gamma = 1.0; //isothermal electrons
  grid->nsub = 10.0; //number of field subcycles
  grid->nsmb=0;
  grid->den_floor_ohm = 0.05;
  grid->den_floor_pe = 0.05;

 // Define periodic CYLINDRICAL grid (periodic in theta only)
  define_periodic_grid(  0.1, 0., -0.5*Lz,     // Low corner (r=0, theta=0, z_min)
			  Lr, Ltheta, 0.5*Lz,     // High corner (r_max, 2pi, z_max)
			  nx, ny, nz,              // Resolution
			  topology_x, topology_y, topology_z); // Topology
  
  // CRITICAL: Initialize as cylindrical, not Cartesian!
  grid->init_cylindrical_grid();

  ////////////////////////////////////////////////////////////////////////////////////////////
  // Setup materials

  sim_log("Setting up materials. "); 

  define_material( "vacuum", 1 );

  define_field_array(NULL);// second argument is damp, default to 0

 // Boundary conditions - NOW IN CYLINDRICAL COORDS
 // Note: x->r, y->theta, z->z
 // Axis (r=0) and outer radial boundary need special treatment
 // Top/bottom are z boundaries

  sim_log("Conducting fields on all boundaries"); 
  if ( ix==0 )            set_domain_field_bc( BOUNDARY(-1,0,0), pec_fields ); // Axis
  if ( ix==topology_x-1 ) set_domain_field_bc( BOUNDARY( 1,0,0), pec_fields ); // Outer radius
  if ( iz==0 )            set_domain_field_bc( BOUNDARY(0,0,-1), pec_fields ); // z bottom
  if ( iz==topology_z-1 ) set_domain_field_bc( BOUNDARY( 0,0,1), pec_fields ); // z top


  sim_log("Absorb particles on all boundaries"); 
  if ( ix==0 )            set_domain_particle_bc( BOUNDARY(-1,0,0), absorb_particles ); // Axis
  if ( ix==topology_x-1 ) set_domain_particle_bc( BOUNDARY(1,0,0), absorb_particles );  // Outer radius
  if ( iz==0 )            set_domain_particle_bc( BOUNDARY(0,0,-1), absorb_particles ); // z bottom
  if ( iz==topology_z-1 ) set_domain_particle_bc( BOUNDARY(0,0,1), absorb_particles );  // z top


// Reflecting inner BC - NOW IN CYLINDRICAL
// Coils are now defined in (r, theta, z) space
// For axisymmetric case, coils are rings at specific (r, z) positions

  // Mirror coil positions in cylindrical coordinates
  double r_coil = 0.25*Lr;   // Radial position of magnetic coils
  double z_mirror1 = 0.35*Lz;  // First mirror position
  double z_mirror2 = -0.35*Lz; // Second mirror position
  double R_coil = 0.25*Lr;   // Coil size

// Define coil regions in cylindrical coords (r, theta, z)
// Remember: x->r, y->theta, z->z
#define R2COIL1 ( (x-r_coil)*(x-r_coil) + (z-z_mirror1)*(z-z_mirror1) )
#define R2COIL2 ( (x-r_coil)*(x-r_coil) + (z-z_mirror2)*(z-z_mirror2) )

# define INSIDE_COIL1 (R2COIL1 < R_coil*R_coil )
# define INSIDE_COIL2 (R2COIL2 < R_coil*R_coil )


  set_region_bc(INSIDE_COIL1, reflect_particles, reflect_particles,reflect_particles);
  set_region_bc(INSIDE_COIL2, reflect_particles, reflect_particles,reflect_particles);


// Resistive layer around coils
#define R21 ( (x-r_coil)*(x-r_coil) + (z-z_mirror1)*(z-z_mirror1) )
#define R22 ( (x-r_coil)*(x-r_coil) + (z-z_mirror2)*(z-z_mirror2) )

# define INSIDE_LAYER1 ( (R21 < 1.5*R_coil*R_coil) && (R2COIL1 > R_coil*R_coil)  )
# define INSIDE_LAYER2 ( (R22 < 1.5*R_coil*R_coil) && (R2COIL2 > R_coil*R_coil)  )

//Set resistive layer multipliers. set_region_eta_multipliers(REGION, hyper_eta mult., eta mult., E field mult.)

  set_region_eta_multipliers(INSIDE_LAYER1, 10.0, 1.,1.);
  set_region_eta_multipliers(INSIDE_LAYER2, 10.0, 1.,1.);
  
  // Damping near z boundaries
  set_region_eta_multipliers(z>0.45*Lz, 1., 1., 0.);
  set_region_eta_multipliers(z<-0.45*Lz, 1., 1., 0.);
  
  // Damping near axis and outer radius
  set_region_eta_multipliers(x<0.015*Lr, 1., 1., 0.);  // Near axis
  set_region_eta_multipliers(x>0.985*Lr, 1., 1., 0.);  // Near outer radius
  
  set_region_eta_multipliers(INSIDE_COIL1, 1., 1., 0.);
  set_region_eta_multipliers(INSIDE_COIL2, 1., 1., 0.);

  ////////////////////////////////////////////////////////////////////////////////////////////
  // Setup the species

  sim_log("Setting up species. ");

  double nmax = 2.*Npart/nproc();
  double nmovers = 0.1*nmax;
  double sort_method = 1; //0=in place, 1=out of place 

  species_t *ion      = define_species("ion"     ,ec,mi,nmax,nmovers,sort_interval,sort_method);
  species_t *beam      = define_species("beam"   ,ec,mi,nmax,nmovers,sort_interval,sort_method);



////////////////////////////////////////////////////////////////////////////////////////////
 

  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation

  sim_log( "***********************************************" );
  sim_log("* Topology:                       "<<topology_x<<" "<<topology_y<<" "<<topology_z); 
  sim_log("* CYLINDRICAL COORDINATES: (r, theta, z)");
  sim_log ( "beta_e = " << beta_e );
  sim_log ( "beta_i = " << beta_i );
  sim_log ( "Ti/Te = " << TiTe ) ;
  sim_log ( "wpe/wce = " << wpewce );
  sim_log ( "mi/me = " << mime );
  sim_log ( "taui = " << taui );
  sim_log ( "num_step = " << num_step );
  sim_log ( "Lr/di = " << Lr/di );
  sim_log ( "Lr/de = " << Lr/de );
  sim_log ( "Lz/di =" << Lz/di );
  sim_log ( "Lz/de = " << Lz/de );
  sim_log ( "nx (r-cells) = " << nx );
  sim_log ( "ny (theta-cells) = " << ny );
  sim_log ( "nz (z-cells) = " << nz ); 
  sim_log ( "nproc = " << nproc ()  );
  sim_log ( "nppc = " << nppc );
  sim_log ( " b0 = " << b0 );
  sim_log ( " di = " << di );
  sim_log ( " Npart = " << Npart );
  sim_log ( "total # of particles = " << 2*Npart );
  sim_log ( "dt*wpe = " << wpe*dt ); 
  sim_log ( "dt*wce = " << wce*dt );
  sim_log ( "dt*wci = " << wci*dt );
  sim_log ( " energies_interval: " << energies_interval );
  sim_log ( "dr/de = " << Lr/(de*nx) );
  sim_log ( "dtheta = " << Ltheta/ny );
  sim_log ( "dz/de = " << Lz/(de*nz) );
  sim_log ( "vthi/c = " << global->vthi/c );
  sim_log ( "vthe/c = " << global->vthe/c );
  sim_log ( "nu/wce = "<<nuei_wce);
  sim_log ( "nu*dt_coll = "<<nuei_wce/wpewce*dt_coll);
  
  // Dump simulation information to file "info"
  if (rank() == 0 ) {
    FILE *fp_info;
    if ( ! (fp_info=fopen("info", "w")) ) ERROR(("Cannot open file."));
    fprintf(fp_info, "           ***** Simulation parameters ***** \n");
    fprintf(fp_info, "           ***** CYLINDRICAL GEOMETRY ***** \n");
    fprintf(fp_info, "		 beta_e	=		%e\n", beta_e);
    fprintf(fp_info, "		 beta_i	=		%e\n", beta_i);
    fprintf(fp_info, "		 Ti/Te	=		%e\n", TiTe );
    fprintf(fp_info, "		 wpe/wce = 		%e\n", wpewce );
    fprintf(fp_info, "		 mi/me =		%e\n", mime );
    fprintf(fp_info, "		 taui =			%e\n", taui );
    fprintf(fp_info, "		 num_step = 		%i\n", num_step );
    fprintf(fp_info, "		 Lr/de = 		%e\n", Lr/de );
    fprintf(fp_info, "		 Lz/de =		%e\n", Lz/de );
    fprintf(fp_info, "		 Lr/di = 		%e\n", Lr/di );
    fprintf(fp_info, "		 Lz/di =		%e\n", Lz/di );
    fprintf(fp_info, "		 nx (r) = 		%e\n", nx );
    fprintf(fp_info, "		 ny (theta) = 		%e\n", ny );
    fprintf(fp_info, "		 nz (z) =		%e\n", nz );
    fprintf(fp_info, "		 nproc = 		%e\n", nproc() );
    fprintf(fp_info, "		 nppc = 		%e\n", nppc );
    fprintf(fp_info, "		 b0 =			%e\n", b0 );
    fprintf(fp_info, "		 di = 			%e\n", di );
    fprintf(fp_info, "		 Npart =        	%e\n", Npart );
    fprintf(fp_info, "		 total # of particles = %e\n", 2*Npart );
    fprintf(fp_info, "		 dt*wpe = 		%e\n", wpe*dt );
    fprintf(fp_info, "		 dt*wce = 		%e\n", wce*dt );
    fprintf(fp_info, "		 dt*wci = 		%e\n", wci*dt );
    fprintf(fp_info, "		 energies_interval: 	%i\n", energies_interval);
    fprintf(fp_info, "		 dr/de =		%e\n", Lr/(de*nx) );
    fprintf(fp_info, "		 dtheta =		%e\n", Ltheta/ny );
    fprintf(fp_info, "		 dz/de =		%e\n", Lz/(de*nz) );
    fprintf(fp_info, "		 vthi/c =		%e\n", global->vthi/c );
    fprintf(fp_info, "		 vthe/c =		%e\n", global->vthe/c );
    fprintf(fp_info, "           nu/wce =               %e\n", nuei_wce);
    fprintf(fp_info, "           nu*dt_coll:            %e\n", nuei_wce/wpewce*dt_coll);
    fprintf(fp_info, "		 ***************************\n");
    fclose(fp_info);
}


  // Dump simulation information to file "info.bin" for translate script
  if (rank() == 0 ) {

    FileIO fp_info;

    // write binary info file

    if ( ! (fp_info.open("info.bin", io_write)==ok) ) ERROR(("Cannot open file."));
    
    fp_info.write(&topology_x, 1 );
    fp_info.write(&topology_y, 1 );
    fp_info.write(&topology_z, 1 );

    fp_info.write(&Lr, 1 );
    fp_info.write(&Ltheta, 1 );
    fp_info.write(&Lz, 1 );

    fp_info.write(&nx, 1 );
    fp_info.write(&ny, 1 );
    fp_info.write(&nz, 1 );

    fp_info.write(&dt, 1 );

    fp_info.write(&mime, 1 );
    fp_info.write(&mi, 1 );
    fp_info.write(&vthe, 1 );
    fp_info.write(&vthi, 1 );
    fp_info.write(&status_interval, 1 );
    fp_info.close();

}
  ////////////////////////////
  // Load fields - CYLINDRICAL MAGNETIC MIRROR

  // For a cylindrical magnetic mirror, we need Br, Btheta, Bz components
  // The trap has strong Bz along axis with mirror coils creating field compression
  
  // Simple axisymmetric mirror field model
  // Bz varies along z with mirrors at +/- z_mirror
  // Br provides radial confinement
  
  double z_mirror = 0.35*Lz;  // Mirror throat positions
  double B_mirror = 2.0;       // Field at mirror throat
  double B_center = 0.5;       // Field at center
  
  // Simple model: Bz = B_center + B_mirror_contrib
  // For axisymmetric: Btheta = 0, and Br comes from div B = 0
  
#define BZ_MIRROR ( B_center + (B_mirror-B_center)*exp(-((z-z_mirror)*(z-z_mirror))/(0.1*Lz*Lz)) \
                             + (B_mirror-B_center)*exp(-((z+z_mirror)*(z+z_mirror))/(0.1*Lz*Lz)) )

  // Br from continuity (simplified): Br ~ -0.5 * r * dBz/dz
#define BR_MIRROR ( -0.5*x*( (B_mirror-B_center)*2.0*(z-z_mirror)/(0.1*Lz*Lz)*exp(-((z-z_mirror)*(z-z_mirror))/(0.1*Lz*Lz)) \
                            +(B_mirror-B_center)*2.0*(z+z_mirror)/(0.1*Lz*Lz)*exp(-((z+z_mirror)*(z+z_mirror))/(0.1*Lz*Lz)) ) )

  sim_log( "Loading fields - Cylindrical Magnetic Mirror" );
  
  // E field starts at zero, B field is the mirror configuration
  // In cylindrical: x->r, y->theta, z->z
  // Field components: ex->Er, ey->Etheta, ez->Ez, cbx->Br, cby->Btheta, cbz->Bz
  
  set_region_field( everywhere, 0, 0, 0,        // Electric field (Er, Etheta, Ez)
  		                BR_MIRROR, 0, BZ_MIRROR );  // Magnetic field (Br, Btheta, Bz)

  // External field (if using split B)
  set_region_bext( everywhere, BR_MIRROR, 0, BZ_MIRROR );
  set_region_te(everywhere, vthe*vthe);

  // LOAD PARTICLES - NOW IN CYLINDRICAL

  sim_log( "Loading particles in cylindrical coordinates" );

  // Fast load of particles
  // Note: x,y,z in grid space are now r, theta, z
  // But velocities remain in PHYSICAL Cartesian (vx, vy, vz)
  
  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);  // r range
  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);  // theta range
  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);  // z range
  
  repeat ( Npart/nproc() ) {

  double r = uniform(rng(0),xmin,xmax);         // radial position
  double theta = uniform(rng(0),ymin,ymax);     // azimuthal position
  double z = uniform(rng(0),zmin,zmax);         // axial position

  // Inject in central region only
  // Now checking in cylindrical: small r, central z
  if (r < Lxp && (abs(z) < Lzp)){
    
    // CRITICAL: Velocities are still in PHYSICAL Cartesian (vx, vy, vz)!
    // Not in (vr, vtheta, vz)! VPIC handles the transformation internally.
    inject_particle( ion, r, theta, z,
		     normal(rng(0),0,vthi),  // vx (Cartesian)
		     normal(rng(0),0,vthi),  // vy (Cartesian)
		     normal(rng(0),0,vthi),  // vz (Cartesian)
		     qi, 0 ,0 );
  }
  }
  sim_log( "Finished loading particles" );

   /*--------------------------------------------------------------------------
     * New dump definition
     *------------------------------------------------------------------------*/

    /*--------------------------------------------------------------------------
	 * Set data output format
     *------------------------------------------------------------------------*/

	global->fdParams.format = band;
	sim_log ( "Fields output format = band" );

	global->hHdParams.format = band;
	sim_log ( "Ion species output format = band" );

	global->hBdParams.format = band;
	sim_log ( "Beam species output format = band" );

    /*--------------------------------------------------------------------------
	 * Set stride
     *------------------------------------------------------------------------*/
	
	sprintf(global->fdParams.baseDir, "fields");
	sprintf(global->fdParams.baseFileName, "fields");

	global->fdParams.stride_x = 1;
	global->fdParams.stride_y = 1;
	global->fdParams.stride_z = 1;

	global->outputParams.push_back(&global->fdParams);

	sim_log ( "Fields r-stride " << global->fdParams.stride_x );
	sim_log ( "Fields theta-stride " << global->fdParams.stride_y );
	sim_log ( "Fields z-stride " << global->fdParams.stride_z );

	sprintf(global->hHdParams.baseDir, "hydro");
	sprintf(global->hHdParams.baseFileName, "Hhydro");

	global->hHdParams.stride_x = 1;
	global->hHdParams.stride_y = 1;
	global->hHdParams.stride_z = 1;

	sim_log ( "Ion species r-stride " << global->hHdParams.stride_x );
	sim_log ( "Ion species theta-stride " << global->hHdParams.stride_y );
	sim_log ( "Ion species z-stride " << global->hHdParams.stride_z );

	global->outputParams.push_back(&global->hHdParams);

	sprintf(global->hBdParams.baseDir, "hydro");
	sprintf(global->hBdParams.baseFileName, "Bhydro");

	global->hBdParams.stride_x = 1;
	global->hBdParams.stride_y = 1;
	global->hBdParams.stride_z = 1;

	global->outputParams.push_back(&global->hBdParams);

    /*--------------------------------------------------------------------------
	 * Set output fields
     *------------------------------------------------------------------------*/

	global->fdParams.output_variables( allvars );
	global->hHdParams.output_variables( allvars );
	global->hBdParams.output_variables( allvars );

	char varlist[512];
	create_field_list(varlist, global->fdParams);
	sim_log ( "Fields variable list: " << varlist );

	create_hydro_list(varlist, global->hHdParams);
	sim_log ( "Ion species variable list: " << varlist );

	create_hydro_list(varlist, global->hBdParams);
	sim_log ( "Beam species variable list: " << varlist );

	sim_log("*** Finished with user-specified initialization ***");
	sim_log("*** CYLINDRICAL GEOMETRY (r, theta, z) ***");

} //begin_initialization

#define should_dump(x) \
	(global->x##_interval>0 && remainder(step(), global->x##_interval) == 0)

begin_diagnostics {

	if(step()==0) {
		dump_mkdir("fields");
		dump_mkdir("hydro");
		dump_mkdir("rundata");
		dump_mkdir("data");
		dump_mkdir("restart0");
		dump_mkdir("restart1");
		dump_mkdir("restart2");
		dump_mkdir("particles");

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
	if(should_dump(Hhydro)) hydro_dump("beam", global->hBdParams);

	// Restart dump
  const int nx=grid->nx,ny=grid->ny,nz=grid->nz, nsp=global->nsp;

	if(step() && !(step()%global->restart_interval)) {
		if(!global->rtoggle) {
			global->rtoggle = 1;
			checkpt("restart1/restart", 0);
		}
		else {
			global->rtoggle = 0;
			checkpt("restart2/restart", 0);
		}
	}

  // Dump particle data
	char subdir[36];
	if ( should_dump(Hparticle) && step()>0 ) {
	  sprintf(subdir,"particles/T.%d",step()); 
	  dump_mkdir(subdir);
	  sprintf(subdir,"particles/T.%d/ion",step()); 
	  dump_particles("ion", subdir);    
	
	  sprintf(subdir,"particles/T.%d",step()); 
	  sprintf(subdir,"particles/T.%d/beam",step()); 
	  dump_particles("beam", subdir);    
	 }

  // Check quota
  if( step()>0 && global->quota_check_interval>0 && (step()&global->quota_check_interval)==0 ) {
    if( uptime() > global->quota_sec ) {
      sim_log( "Allowed runtime exceeded for this job.  Terminating....\n");

      BEGIN_TURNSTILE(NUM_TURNSTILES){
      checkpt("restart0/restart",0);
      DUMP_INJECTORS(0);
      } END_TURNSTILE;

      sim_log( "Restart dump restart completed." );
      exit(0);
    }
  }

} // end diagnostics


begin_current_injection {
  // No current injection for this simulation
}

begin_field_injection {

 const int nx=grid->nx;
 const int ny=grid->ny;
 const int nz=grid->nz;
 int x,y,z;
 const int numcell = 5;
 const double r = 0.99;

// Damping boundaries in cylindrical coords
// x->r, y->theta, z->z

Kokkos::MDRangePolicy<Kokkos::Rank<3>> axis_face({1, 1, 1}, {nz+1, ny+1, numcell+1});  // Near axis
Kokkos::MDRangePolicy<Kokkos::Rank<3>> outer_face({1, 1, nx+1-numcell}, {nz+1, ny+1, nx+1}); // Outer radius

k_field_t& k_field = field_array->k_f_d;
  
  // Near AXIS (r=0)
  if (0*global->left) {
		Kokkos::parallel_for("inject_fields: axis_damp", axis_face, KOKKOS_LAMBDA(const int z, const int y, const int x) {
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) *= r;  // Btheta
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) *= r;  // Bz
	    	});
	}
	
  // Outer RADIUS
  if (0*global->right) {
		Kokkos::parallel_for("inject_fields: outer_damp", outer_face, KOKKOS_LAMBDA(const int z, const int y, const int x) {
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) *= r;  // Btheta
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) *= r;  // Bz
	    	});
	}

}


begin_particle_collisions {
  //none for this deck
}
 
begin_particle_injection{
  //none for this deck
}