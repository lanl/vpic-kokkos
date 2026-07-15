/////////////////////////////////////////////////////
//
//   Gas Dynamic Trap - Cylindrical
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
  double Lx;
  
  double topology_x;       // domain topology 
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

  // Simulation  parameters - CYLINDRICAL (R-Z)

  double nppc = 400;        // Average number of macro particle per cell per species

  double Lr  = 300.*di;     // size of box in r dimension (radial)
  double Ltheta  = 2.0*M_PI; // full azimuthal range (single cell)
  double Lz  = 30.*di;      // size of box in z dimension

  double Lxp = Lr/4.5;      // x now means r
  double Lzp = Lz/4.5;
  
  double topology_x = 32;  // Number of domains in r, theta, and z
  double topology_y = 1;   // Single cell in theta
  double topology_z = 4;  

  double nr = 1024;   // Number of cells in r-direction
  double ntheta = 1;  // Single cell in theta
  double nz = 104;    // Number of cells in z-direction

  double r_min = 0.01*Lr;  // Small buffer at r=0 to avoid singularity
  double hr = (Lr - r_min)/nr;   // cell size in r
  double htheta = Ltheta/ntheta; // cell size in theta
  double hz = Lz/nz;             // cell size in z

  // Adjust for cylindrical geometry - use average radius for particle count
  double r_avg = (r_min + Lr)/2.0;
  double Npart   = nppc*nr*ntheta*nz;  // total macro particles in box
  Npart = trunc_granular(Npart,nproc()); // Make it divisible by number of processors

  double qe = -ec*M_PI*(Lr*Lr - r_min*r_min)*Lz/Npart;  // Charge per macro electron (cylindrical volume)
  double qi =  ec*M_PI*(Lr*Lr - r_min*r_min)*Lz/Npart;  // Charge per macro ion
  double nfac = qi/(2.0*M_PI*r_avg*hr*htheta*hz);       // Convert density to particles per cell (approximate)
      
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
  if ( ix ==0 ) left=1;  // Inner radial boundary
  if ( ix==topology_x-1) right=1;  // Outer radial boundary
  if ( iz ==0 ) bottom=1;
  if ( iz ==topology_z-1 ) top=1;
  if ( abs( float(ix) - (topology_x/2.-0.5) ) < 4. ) center=1;


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
  global-> Lx = Lr;  // Now means Lr
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

  //global->q[0]  = -qe;  
  global->q[0]  = qi;  
  global->q[1]  = qi;  
 
  //global->npleft[0]  = n0;
  global->npleft[0]  = n0;
  global->npleft[1]  = nb;

  //global->npright[0]  = 0.0;
  global->npright[0]  = n0;
  global->npright[1]  = 0;

  //global->vth[0]  = sqrt(2)*vthe;
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
  grid->dx = hr;
  grid->dy = htheta;
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

  // Define cylindrical grid (r, theta, z)
  // r from r_min to Lr, theta full 2*pi (1 cell), z from -Lz/2 to Lz/2
  define_periodic_grid(  r_min, 0.0, -0.5*Lz,    // Low corner (r_min, 0, -Lz/2)
                         Lr, Ltheta, 0.5*Lz,      // High corner (Lr, 2*pi, Lz/2)
                         nr, ntheta, nz,          // Resolution
                         topology_x, topology_y, topology_z); // Topology
  grid->init_cylindrical_grid();  // Initialize cylindrical grid

  ////////////////////////////////////////////////////////////////////////////////////////////
  // Setup materials

  sim_log("Setting up materials. "); 

  define_material( "vacuum", 1 );

  define_field_array(NULL);// second argument is damp, default to 0

  // Boundary conditions for cylindrical geometry
  // Boundary conditions for cylindrical geometry
  sim_log("Setting boundary conditions for cylindrical geometry"); 
  
  // Axis boundary (r=0) - handled internally by cylindrical grid
  // Outer radial boundary (r=Lr)
  if ( ix==topology_x-1 ) set_domain_field_bc( BOUNDARY( 1,0,0), pec_fields ); 
  // Z boundaries
  if ( iz==0 )            set_domain_field_bc( BOUNDARY(0,0,-1), pec_fields ); 
  if ( iz==topology_z-1 ) set_domain_field_bc( BOUNDARY( 0,0,1), pec_fields ); 


  sim_log("Absorb particles on radial and z boundaries"); 
  if ( ix==topology_x-1 ) set_domain_particle_bc( BOUNDARY(1,0,0), absorb_particles );
  if ( iz==0 )            set_domain_particle_bc( BOUNDARY(0,0,-1), absorb_particles );
  if ( iz==topology_z-1 ) set_domain_particle_bc( BOUNDARY(0,0,1), absorb_particles );


// Reflecting inner BC - Coils in cylindrical geometry
// Note: x, y, z in particle coordinates are r, theta, z
// Coil positions need to be adapted for cylindrical geometry

  double r_P1 = 0.15*Lr, r_P2 = 0.85*Lr, z_P = Lz/2, R_P = 0.25*Lz;

// In cylindrical coordinates, coils are rings at specific (r,z) locations
// Using x (which is r) and z for coil positions
#define R2P1 ( 0.02*(x-r_P1)*(x-r_P1) + (z-z_P)*(z-z_P) )
#define R2P2 ( 0.02*(x-r_P1)*(x-r_P1) + (z+z_P)*(z+z_P) )
#define R2P3 ( 0.02*(x-r_P2)*(x-r_P2) + (z-z_P)*(z-z_P) )
#define R2P4 ( 0.02*(x-r_P2)*(x-r_P2) + (z+z_P)*(z+z_P) )

# define INSIDE_COIL1 (R2P1 < R_P*R_P )
# define INSIDE_COIL2 (R2P2 < R_P*R_P )
# define INSIDE_COIL3 (R2P3 < R_P*R_P )
# define INSIDE_COIL4 (R2P4 < R_P*R_P )


  set_region_bc(INSIDE_COIL1, reflect_particles, reflect_particles,reflect_particles);
  set_region_bc(INSIDE_COIL2, reflect_particles, reflect_particles,reflect_particles);
  set_region_bc(INSIDE_COIL3, reflect_particles, reflect_particles,reflect_particles);
  set_region_bc(INSIDE_COIL4, reflect_particles, reflect_particles,reflect_particles);


#define R21 ( 0.02*(x-r_P1)*(x-r_P1) + (z-z_P)*(z-z_P) )
#define R22 ( 0.02*(x-r_P1)*(x-r_P1) + (z+z_P)*(z+z_P) )
#define R23 ( 0.02*(x-r_P2)*(x-r_P2) + (z-z_P)*(z-z_P) )
#define R24 ( 0.02*(x-r_P2)*(x-r_P2) + (z+z_P)*(z+z_P) )

# define INSIDE_LAYER1 ( (R21 < 1.5*R_P*R_P) && (R2P1 > R_P*R_P)  )
# define INSIDE_LAYER2 ( (R22 < 1.5*R_P*R_P) && (R2P2 > R_P*R_P)  )
# define INSIDE_LAYER3 ( (R23 < 1.5*R_P*R_P) && (R2P3 > R_P*R_P)  )
# define INSIDE_LAYER4 ( (R24 < 1.5*R_P*R_P) && (R2P4 > R_P*R_P)  )

//Set resistive layer multipliers. set_region_eta_multipliers(REGION, hyper_eta mult., eta mult., E field mult.)

  set_region_eta_multipliers(INSIDE_LAYER1, 10.0, 1.,1.);
  set_region_eta_multipliers(INSIDE_LAYER2, 10.0, 1.,1.);
  set_region_eta_multipliers(INSIDE_LAYER3, 10.0, 1.,1.);
  set_region_eta_multipliers(INSIDE_LAYER4, 10.0, 1.,1.);
  
  set_region_eta_multipliers(z>0.45*Lz, 1., 1., 0.);
  set_region_eta_multipliers(z<-0.45*Lz, 1., 1., 0.);
  
  set_region_eta_multipliers(x<0.015*Lr, 1., 1., 0.);  // Near axis
  set_region_eta_multipliers(x>0.985*Lr, 1., 1., 0.);  // Outer boundary
  
  set_region_eta_multipliers(INSIDE_COIL1, 1., 1., 0.);
  set_region_eta_multipliers(INSIDE_COIL2, 1., 1., 0.);
  set_region_eta_multipliers(INSIDE_COIL3, 1., 1., 0.);
  set_region_eta_multipliers(INSIDE_COIL4, 1., 1., 0.);

  ////////////////////////////////////////////////////////////////////////////////////////////
  // Setup the species

  sim_log("Setting up species. ");

  double nmax = 2.*Npart/nproc();
  double nmovers = 0.1*nmax;
  double sort_method = 1; //0=in place, 1=out of place 

  //species_t *electron = define_species("electron",-ec,me,nmax,nmovers,sort_interval,sort_method);
  species_t *ion      = define_species("ion"     ,ec,mi,nmax,nmovers,sort_interval,sort_method);
  species_t *beam      = define_species("beam"   ,ec,mi,nmax,nmovers,sort_interval,sort_method);



////////////////////////////////////////////////////////////////////////////////////////////
 

  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation

  sim_log( "***********************************************" );
  sim_log("* Topology:                       "<<topology_x<<" "<<topology_y<<" "<<topology_z); 
  sim_log ( "beta_e = " << beta_e );
  sim_log ( "beta_i = " << beta_i );
  sim_log ( "Ti/Te = " << TiTe ) ;
  sim_log ( "wpe/wce = " << wpewce );
  sim_log ( "mi/me = " << mime );
  sim_log ( "taui = " << taui );
  sim_log ( "num_step = " << num_step );
  sim_log ( "Lr/di = " << Lr/di );  // Changed from Lx
  sim_log ( "Lr/de = " << Lr/de );
  sim_log ( "Ltheta (radians) = " << Ltheta );
  sim_log ( "Lz/di = " << Lz/di );
  sim_log ( "Lz/de = " << Lz/de );
  sim_log ( "nr = " << nr );  // Changed from nx
  sim_log ( "ntheta = " << ntheta );  // Changed from ny
  sim_log ( "nz = " << nz ); 
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
  sim_log ( "dr/de = " << Lr/(de*nr) );
  sim_log ( "dtheta (radians) = " << Ltheta/ntheta );
  sim_log ( "dz/de = " << Lz/(de*nz) );
  //sim_log ( "dx/debye = " << (Lx/nx)/Ldeb  );
  sim_log ( "vthi/c = " << global->vthi/c );
  sim_log ( "vthe/c = " << global->vthe/c );
  sim_log ( "nu/wce = "<<nuei_wce);
  sim_log ( "nu*dt_coll = "<<nuei_wce/wpewce*dt_coll);
  sim_log ( "r_min = " << r_min << " (buffer at axis)" );
  
  // Dump simulation information to file "info"
  if (rank() == 0 ) {
    FILE *fp_info;
    if ( ! (fp_info=fopen("info", "w")) ) ERROR(("Cannot open file."));
    fprintf(fp_info, "           ***** Simulation parameters (CYLINDRICAL) ***** \n");
    fprintf(fp_info, "		 beta_e	=		%e\n", beta_e);
    fprintf(fp_info, "		 beta_i	=		%e\n", beta_i);
    fprintf(fp_info, "		 Ti/Te	=		%e\n", TiTe );
    fprintf(fp_info, "		 wpe/wce = 		%e\n", wpewce );
    fprintf(fp_info, "		 mi/me =		%e\n", mime );
    fprintf(fp_info, "		 taui =			%e\n", taui );
    fprintf(fp_info, "		 num_step = 		%i\n", num_step );
    fprintf(fp_info, "		 Lr/de = 		%e\n", Lr/de );
    fprintf(fp_info, "		 Ltheta (rad) = 	%e\n", Ltheta );
    fprintf(fp_info, "		 Lz/de =		%e\n", Lz/de );
    fprintf(fp_info, "		 Lr/di = 		%e\n", Lr/di );
    fprintf(fp_info, "		 Lz/di =		%e\n", Lz/di );
    fprintf(fp_info, "		 nr = 			%e\n", nr );
    fprintf(fp_info, "		 ntheta = 		%e\n", ntheta );
    fprintf(fp_info, "		 nz =			%e\n", nz );
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
    fprintf(fp_info, "		 dr/de =		%e\n", Lr/(de*nr) );
    fprintf(fp_info, "		 dtheta (rad) =		%e\n", Ltheta/ntheta );
    fprintf(fp_info, "		 dz/de =		%e\n", Lz/(de*nz) );
    fprintf(fp_info, "		 vthi/c =		%e\n", global->vthi/c );
    fprintf(fp_info, "		 vthe/c =		%e\n", global->vthe/c );
    fprintf(fp_info, "           nu/wce =               %e\n", nuei_wce);
    fprintf(fp_info, "           nu*dt_coll:            %e\n", nuei_wce/wpewce*dt_coll);
    fprintf(fp_info, "           r_min =                %e\n", r_min);
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

    fp_info.write(&nr, 1 );
    fp_info.write(&ntheta, 1 );
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
  // Load fields - CYLINDRICAL MAGNETIC FIELD

// For cylindrical geometry with azimuthal symmetry, we need B_r and B_z
// The field formulas need to be adapted for cylindrical coils (current loops)

#define BXC(I,rc,zc,L) (-2.0*I/8.0/atan(L/2.0/zc)*(atan((x-rc)/(z-zc))+atan((L-x+rc)/(z-zc))-atan((x-rc)/(z+zc))-atan((L-x+rc)/(z+zc))) )
#define BZC(I,rc,zc,L) ( I/8.0/atan(L/2.0/zc)*log(((x-rc)*(x-rc)+(z-zc)*(z-zc))*((L-x+rc)*(L-x+rc)+(z+zc)*(z+zc))/((((L-x+rc)*(L-x+rc)+(z-zc)*(z-zc)))*(((x-rc)*(x-rc)+(z+zc)*(z+zc))))) )

double Lcoil1 = 0.6*Lr;
double Lcoil2 = 0.1*Lr;

double z1 = 0.55*Lz;
double r1 = 0.2*Lr, r2 = 0.8*Lr, r3 = 0.1*Lr, r4 = 0.9*Lr; 


double B0=0.5;
double B1 = 0.1;
double I1 = 1.45*B1, I2 = 2.0*B0-0.5*B1, I4 = 0.53*B1;

// Note: x here means r in cylindrical coordinates
// BX will be B_r, BZ will be B_z
// B_theta is zero due to azimuthal symmetry
#define BX ( BXC(I1,r1,z1,Lcoil1) + BXC(I2,r2,z1,Lcoil2) + BXC(I2,r3,z1,Lcoil2) + BXC(I4,r4,z1,Lcoil2) +  BXC(I4,r_min,z1,Lcoil2) )
#define BZ ( BZC(I1,r1,z1,Lcoil1) + BZC(I2,r2,z1,Lcoil2) + BZC(I2,r3,z1,Lcoil2) + BZC(I4,r4,z1,Lcoil2) +  BZC(I4,r_min,z1,Lcoil2) )

  sim_log( "Loading fields (cylindrical geometry)" );
  set_region_field( everywhere, 0, 0, 0,       // Electric field
  		                0, 0 ,0 );    // Magnetic field

  set_region_bext( everywhere,  BX, 0 ,BZ );    // External Magnetic field (B_r, B_theta=0, B_z)
  set_region_te(everywhere, vthe*vthe);

  // LOAD PARTICLES - CYLINDRICAL

  sim_log( "Loading particles (cylindrical geometry)" );

  // Do a fast load of the particles
  // Particle positions: x=r, y=theta, z=z

  //seed_rand( rng_seed*nproc() + rank() );  //Generators desynchronized
  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);  // r range
  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);  // theta range
  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);  // z range
  
  repeat ( Npart/nproc() ) {

  double x = uniform(rng(0),xmin,xmax);  // x is r
  double y = uniform(rng(0),ymin,ymax);  // y is theta
  double z = uniform(rng(0),zmin,zmax);  // z is z

 /* inject_particle( electron, x, y, z,
                      normal(rng(0),0,vthe),
		      normal(rng(0),0,vthe),
         	      normal(rng(0),0,vthe),-qe, 0, 0);
*/

  // Injection region in (r, z) - note x is r here
  if (abs(x-Lr/2.)< Lxp && (abs(z) < Lzp)){
    
    // Velocities are in physical Cartesian (vx, vy, vz)
    inject_particle( ion, x, y, z,
		     normal(rng(0),0,vthi),
		     normal(rng(0),0,vthi),
		     normal(rng(0),0,vthi),qi, 0 ,0 );
  }
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

	global->fdParams.format = band;

	sim_log ( "Fields output format = band" );

	//global->hedParams.format = band;

	//sim_log ( "Electron species output format = band" );

	global->hHdParams.format = band;

	sim_log ( "Ion species output format = band" );

	global->hBdParams.format = band;

	sim_log ( "Beam species output format = band" );

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
	 *   nr = 150;
	 *   global->fdParams.stride_x = 10; // legal -> 150/10 = 15
	 *
	 *   global->fdParams.stride_x = 8; // illegal!!! -> 150/8 = 18.75
     *------------------------------------------------------------------------*/
	
	// relative path to fields data from global header
	sprintf(global->fdParams.baseDir, "fields");

	// base file name for fields output
	sprintf(global->fdParams.baseFileName, "fields");

	global->fdParams.stride_x = 1;  // r-direction
	global->fdParams.stride_y = 1;  // theta-direction (only 1 cell anyway)
	global->fdParams.stride_z = 1;  // z-direction

	// add field parameters to list
	global->outputParams.push_back(&global->fdParams);

	sim_log ( "Fields r-stride " << global->fdParams.stride_x );
	sim_log ( "Fields theta-stride " << global->fdParams.stride_y );
	sim_log ( "Fields z-stride " << global->fdParams.stride_z );

	// relative path to ion species data from global header
	sprintf(global->hHdParams.baseDir, "hydro");

	// base file name for fields output
	sprintf(global->hHdParams.baseFileName, "Hhydro");

	global->hHdParams.stride_x = 1;
	global->hHdParams.stride_y = 1;
	global->hHdParams.stride_z = 1;

	sim_log ( "Ion species r-stride " << global->hHdParams.stride_x );
	sim_log ( "Ion species theta-stride " << global->hHdParams.stride_y );
	sim_log ( "Ion species z-stride " << global->hHdParams.stride_z );

	// add ion species parameters to list
	global->outputParams.push_back(&global->hHdParams);


	// relative path to beam species data from global header
	sprintf(global->hBdParams.baseDir, "hydro");

	// base file name for beam hydro output
	sprintf(global->hBdParams.baseFileName, "Bhydro");

	global->hBdParams.stride_x = 1;
	global->hBdParams.stride_y = 1;
	global->hBdParams.stride_z = 1;

	// add beam species parameters to list
	global->outputParams.push_back(&global->hBdParams);

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
                      momentum_density | ke_density     | stress_tensor );
	*/

	//  These have just the most useful things turned on
	//   global->fdParams.output_variables( electric | magnetic | current | div_e_err );
	//    global->hHdParams.output_variables( current_density | charge_density | ke_density | stress_tensor);
	
	// Here we are dumping everything

	global->fdParams.output_variables( allvars );
	global->hHdParams.output_variables( allvars );
	global->hBdParams.output_variables( allvars );

	/*--------------------------------------------------------------------------
	 * Convenience functions for simlog output
	 *------------------------------------------------------------------------*/

	char varlist[512];
	create_field_list(varlist, global->fdParams);

	sim_log ( "Fields variable list: " << varlist );

	create_hydro_list(varlist, global->hHdParams);

	sim_log ( "Ion species variable list: " << varlist );

	create_hydro_list(varlist, global->hBdParams);

	sim_log ( "Beam species variable list: " << varlist );


	sim_log("*** Finished with user-specified initialization (CYLINDRICAL) ***");


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

#define should_dump(x) \
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

	if(step()==0) {
		dump_mkdir("fields");
		dump_mkdir("hydro");
		dump_mkdir("rundata");
		dump_mkdir("data");
		dump_mkdir("restart0");
		dump_mkdir("restart1");  // 1st backup
		dump_mkdir("restart2");  // 2nd backup
		dump_mkdir("particles");

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
	 * Field data output
	 *------------------------------------------------------------------------*/

	if(step() == 1 || should_dump(fields)) field_dump(global->fdParams);

	/*--------------------------------------------------------------------------
	 * Ion species output
	 *------------------------------------------------------------------------*/

	if(should_dump(Hhydro)) hydro_dump("ion", global->hHdParams);
	if(should_dump(Hhydro)) hydro_dump("beam", global->hBdParams);


	/*--------------------------------------------------------------------------
	 * Restart dump
	 *------------------------------------------------------------------------*/

  const int nx=grid->nx,ny=grid->ny,nz=grid->nz, nsp=global->nsp;

	if(step() && !(step()%global->restart_interval)) {
		if(!global->rtoggle) {
			global->rtoggle = 1;
			checkpt("restart1/restart", 0);
		}
		else {
			global->rtoggle = 0;
			checkpt("restart2/restart", 0);
		} // if
	} // if

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

  // Shut down simulation when wall clock time exceeds global->quota_sec. 
  // Note that the mp_elapsed() is guaranteed to return the same value for all
  // processors (i.e., elapsed time on proc #0), and therefore the abort will 
  // be synchronized across processors. Note that this is only checked every
  // few timesteps to eliminate the expensive mp_elapsed call from every
  // timestep. mp_elapsed has an ALL_REDUCE in it!
  
  if( step()>0 && global->quota_check_interval>0 && (step()%global->quota_check_interval)==0 ) {
    if( uptime() > global->quota_sec ) {
      sim_log( "Allowed runtime exceeded for this job.  Terminating....\n");

      BEGIN_TURNSTILE(NUM_TURNSTILES){
      checkpt("restart0/restart",0);
      } END_TURNSTILE;

      sim_log( "Restart dump completed." );
      exit(0);

    }
  }

} // end diagnostics


begin_current_injection {

  // No current injection for this simulation

}

begin_field_injection {

 const int nx=grid->nx;  // nr in cylindrical
 const int ny=grid->ny;  // ntheta (=1)
 const int nz=grid->nz;
 int x,y,z;  // Note: x is r, y is theta, z is z
 const int numcell = 5; //damp B field over this many cells at boundaries

 const double r = 0.99;  //1-damp rate

// For cylindrical geometry:
// - Inner radial boundary (r=r_min, near axis) - handled by grid
// - Outer radial boundary (r=Lr)
// - Z boundaries (top and bottom)

Kokkos::MDRangePolicy<Kokkos::Rank<3>> right_face({1, 1, nx+1-numcell}, {nz+1, ny+1, nx+1});  // Outer r boundary
Kokkos::MDRangePolicy<Kokkos::Rank<3>> bottom_face({1, 1, 1}, {numcell+1, ny+1, nx+1});       // Bottom z boundary
Kokkos::MDRangePolicy<Kokkos::Rank<3>> top_face({nz+1-numcell, 1, 1}, {nz+1, ny+1, nx+1});    // Top z boundary

k_field_t& k_field = field_array->k_f_d;
  
  // Outer Radial Boundary (r=Lr)
  if (0*global->right) {
		Kokkos::parallel_for("inject_fields: outer_r_boundary", right_face, KOKKOS_LAMBDA(const int z, const int y, const int x) {
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) *= r;  // B_theta
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) *= r;  // B_z
	    	});
	}
	
  // Bottom Z Boundary
  if (0*global->bottom) {
		Kokkos::parallel_for("inject_fields: bottom_z_boundary", bottom_face, KOKKOS_LAMBDA(const int z, const int y, const int x) {
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) *= r;  // B_r
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) *= r;  // B_theta
	    	});
	}

  // Top Z Boundary
  if (0*global->top) {
		Kokkos::parallel_for("inject_fields: top_z_boundary", top_face, KOKKOS_LAMBDA(const int z, const int y, const int x) {
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) *= r;  // B_r
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) *= r;  // B_theta
	    	});
	}

}


begin_particle_collisions {
//none for this deck
}
 
begin_particle_injection{
//none for this deck
} // end particle injection