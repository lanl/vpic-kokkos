#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
//#include "mpi.h"
//#include "src/species_advance/species_advance.h"
//#include "src/vpic/vpic.h"
#include "deck/wrapper.h"

int tx, ty, tz;
double psum_integrated_poynting_flux_tally;
double psum_integrated_poynting_flux_tally_new;
int psum_integration_offset = 0;
float emax;
double legacy_gpsum[6], new_gpsum[6];

# define RANK_TO_INDEX(rank,ix,iy,iz) BEGIN_PRIMITIVE { \
	int _ix, _iy, _iz;                                    \
	_ix  = (rank);        /* ix = ix+gpx*( iy+gpy*iz ) */ \
	_iy  = _ix/int(global->topology_x);   /* iy = iy+gpy*iz */            \
	_ix -= _iy*int(global->topology_x);   /* ix = ix */                   \
	_iz  = _iy/int(global->topology_y);   /* iz = iz */                   \
	_iy -= _iz*int(global->topology_y);   /* iy = iy */                   \
	(ix) = _ix;                                           \
  (iy) = _iy;                                           \
  (iz) = _iz;                                           \
} END_PRIMITIVE

begin_globals {
  float emax;                   // E0 of the laser pump
  float emax_seed;              // E0 of the seed
  float omega_0;                // w0/wpe - frequency of pump
  float omega_seed;             // seed frequency
  float domega;                 // dw0/wpe for chirp
  float chirp_length;           // time for frequency of chrip to go from omega0 -> omega0+domega
  float pulse_length;           // laser pulse rise time
  float vthe; 			// vthe/c   <- these are needed to make movie files
  float vthi_He;                // vthi_He/c
  float vthi_H;                 // vthi_H/c
  int field_interval;           // how frequently to dump field built-in diagnostic
  int energies_interval;
  int restart_interval; 	// how frequently to write restart file. 
  int quota_check_interval;     // how often to check if runtime quota exceeded
  int poynting_interval;        // how frequently to dump poynting flux at boundaries. 
  int velocity_interval;        // how frequently to dump velocity space
  int fft_ex_interval;          // how frequently to save ex fft data
  int fft_ez_interval;          // how frequently to save ez fft data
  int fft_ey_interval;          // how frequently to save ey fft data
  int eparticle_interval;       // how frequently to dump particle data
  int Hparticle_interval;       //  
  int Heparticle_interval;      //
  int mobile_ions;		// flag: 0 if ions are not to be pushed
  int H_present;                // flag nonzero when H ions are present. 
  int He_present;               // flag nonzero when He ions are present.  
  int rtoggle;                  // Enables save of last two restart dumps for safety
  double quota_sec;             // Run quota in seconds

  int do_collisions;            // Flag for whether to do collisions
  int self_collisions_only;     // Whether to turn on cross-species collisions
  int tstep_coll;               // How many time steps to sub-cycle the collision operator
  int load_particles;           // Whether to load particles 
  int nppc_max;                 // Maximum number particles/cell possible
  double cvar;                  // Collision variable
  double mime_H;                // proton to electron mass ratio
  double mime_He;               // alpha to electron mass ratio

  // Parameters for 2d and 3d Gaussian wave launch
  float waist;			// how wide the focused beam is
  float width;			
  float zcenter;		// center of beam at boundary in z
  float ycenter;		// center of beam at boundary in y
  float xfocus;			// how far from boundary to focus
  float mask;			// # gaussian widths from beam center where I nonzero

  double topology_x;            // domain topology needed to normalize Poynting diagnostic 
  double topology_y;
  double topology_z;

  double Lz;                    // Size of box in z

  // Used for fft diagnostic
  double Lx;                    // Size of box in x
  double xmin_domain;           // Location in x of left boundary 
  double ey_xloc;               // x location of ey(z) taken for FFT diagnostic 


  double lambda;                // Wavelength in terms of skin depths
  double wpe1ps;

  // Ponyting diagnostic flags - which output to turn on

  // From old poynting data block
  int write_poynting_data;

  // write_backscatter_only flag:  when this flag is nonzero, it means to only compute 
  // poynting data for the lower-x surface.  This flag affects both the summed poynting 
  // data as well as the surface data. 

  int write_backscatter_only;  // nonzero means we only write backscatter diagnostic for fields

  // write_poynting_sum is nonzero if you wish to write a file containing the integrated
  // poynting data on one or more surfaces.  If write_backscatter_only is nonzero, then 
  // the output file will be a time series of double precision numbers containing the 
  // integrated poynting flux at that point in time.  If write_backscatter_only is zero,
  // then for each time step, six double precision numbers are written, containing the 
  // poynting flux through the lower x, upper x, lower y, upper y, lower z, and upper z
  // surfaces. 

  int write_poynting_sum;      // nonzero if we wish to write integrated Poynting data

  // write_poynting_faces is probably useless, but I put it here anyway in case it's not. 
  // When this flag is nonzero, it will print out the poynting flux at each of a 2D array 
  // of points on the surface.  When this flag is turned on and write_backscatter_only is 
  // nonzero, then only the 2D array of points on the lower-x boundary surface are written
  // for each time step.  When this flag is turned on and write_bacscatter_only is 
  // zero, then it will write 2D array data for the lower x, upper x, lower y, upper y, 
  // lower z, upper z surfaces for each time step. 

  int write_poynting_faces;    // nonzero if we wish to write Poynting data on sim boundaries 

  // write_eb_faces is nonzero when we wish to get the raw e and b data at the boundary
  // (e.g., to be used with a filter to distinguish between SRS and SBS backscatter).  
  // When this flag is on and write_backscatter_only is nonzero, then only the 2D array
  // of points on the lower-x boundary surface are written for each time step.  When 
  // this flag is turned on and write_backscatteR_only is zero, then it will write 2D
  // array data for the lower x, upper x, lower y, upper y, lower z, upper z surfaces for
  // each time step.  When turned on, four files are produced: e1, e2, cb1, cb2.  The 
  // values of the quantities printed depend on the face one is considering:  for the 
  // x faces, e1 = ey, e2 = ez, cb1 = cby, cb2 = cbz.  Similarly for y and z surfaces, 
  // but where 1 and 2 stand for (z,x) and (x,y) coordinates, respectively.  

  int write_eb_faces;          // nonzero if we wish to write E and B data on sim boundaries

  // write_side_scatter is nonzero when we want to write side scatter data.  For now
  // it doesn't turn off side scatter writes of poynting and field surface data, just 
  // the poynting sum data.  This flag can be zero while write_backscatter_only and 
  // write_poynting_sum flags are nonzero and we will write poynting sum data for left
  // and right most processors (two sum data files).  If this flag is on while the 
  // other two  are on, then we only write a single sum data and we use mpi collectives
  // to aggregate the data. 
  
  int write_side_scatter;      // nonzero if we wish to write side scatter sum data

  float theta;                 // launch angle of pump relative to surface normal (positive = upward)
  float theta_seed;            // launch angle of seed relative to surface normal (positive = upward)

  int launch_laser;            // whether to launch pump laser
  int launch_seed;             // whether to launch seed pulse

  // -------------------------------------------------------------------------------------------------
  // For the time-averaging field diagnostic:

  // The adjustable parameters of these, dis_interval and dis_nav, are now set by the variables
  // AVG_SPACING and AVG_TOTAL_STEPS in the beginning of the include file so they can be more
  // easily found and edited
  //
  // The data are saved in the interval
  //
  // j*dis_interval <= step < j*dis_interval + dis_nav
  //
  // where j is an integer. the output is assigned time index corresponding to the start of the interval
  //
  // The code is restart-aware. The associated global variables are
  //
  // global->restart_interval is assumed to be defined

  int dis_nav;                             // number of steps to average over
  int dis_interval;                        // number of steps between outputs
  int dis_iter;                            // iteration count. 0 means we are not averaging at the moment
  int dis_begin_int;                       // first time step of the interval

  // -------------------------------------------------------------------------------------------------

  // For integrade poynting flux tally diagnostic 
  double psum_integrated_poynting_flux_tally; 
  double psum_integrated_poynting_flux_tally_new;
  int    psum_integration_offset; 

};

void
vpic_simulation::user_initialization( int num_cmdline_arguments,
                                      char ** cmdline_argument )
{

  sim_log( "*** Begin initialization. ***" ); 
  mp_barrier(); // Barrier to ensure we started okay. 
  sim_log( "*** Begin initialization2. ***" ); 

  float elementary_charge  = 4.8032e-10;       // stat coulomb
  float elementary_charge2 = elementary_charge * elementary_charge; 
  float speed_of_light     = 2.99792458e10;    // cm/sec
  float m_e                = 9.1094e-28;       // g
  float k_boltz            = 1.6022e-12;       // ergs/eV
  float mec2 = m_e*speed_of_light*speed_of_light/k_boltz;
  float mpc2 = mec2*1836.0;
  float eps0 = 1;                       

  double cfl_req   = 0.98;      // How close to Courant should we try to run
  double damp      = 0;         // Level of radiation damping
  double iv_thick  = 2;         // Thickness (in cells) of imperm. vacuum region
  int    psum_integration_offset = 0; 

  float t_e = 3000;              // Electron temp, eV
  float t_i = t_e/2.0;              // Ion temp, eV
  float n_e_over_n_crit   = 0.04;      // n_e/n_crit 

  float laser_intensity  = 6.0e14/0.274915; // Units of W/cm^2, ave. intensity NIC  2.9e14W/cm^2, pump1, 20.1 degree, seed30
  float seed_intensity   = 6.0e14/0.274915;// Units of W/cm^2, ave. intensity NIC seed 2.9e12W/cm^2
  float vacuum_wavelength = 351 * 1e-7; // 0.351 micron blue light

  float theta             = -23.55335044860839; // Launch angle in degrees relative to injection surface; positive angle is upward
  float theta_seed        =  23.55335044860839; // Launch angle in degrees relative to injection surface; positive angle is upward

  float box_size_x = 500 * 1e-4;   // Microns
  float box_size_z = 357.9599927772741 * 1e-4;   // Microns (ignored if 1d or 2d in plane)

  int mobile_ions         = 1;        // Whether or not to push ions 
  int He_present=1, H_present=0;      // Parameters used later. Set to unity to initialize. 
  double f_He             = 1.0;        // Ratio of number density of He to total ion density
  double f_H              = 1-f_He;   // Ratio of number density of H  to total ion density 
  if ( f_He==1 ) H_present=0; 
  if ( f_He==0 ) He_present=0; 

  int load_particles = 1;         // Flag to turn off particle load for testing wave launch. 
  double nppc        = 256;

  // FIXME:  Put in the real values here rather than approximate values: 
  float A_H      = 1;                 
  float A_He     = 4;                 
  float Z_H      = 1;
  float Z_He     = 2; 
  float mic2_H   = mpc2*A_H;
  float mic2_He  = mpc2*A_He;
  float mime_H   = mic2_H/mec2;
  float mime_He  = mic2_He/mec2;

  double uthe    = sqrt(t_e/mec2);    // vthe/c
  double uthi_H  = sqrt(t_i/mic2_H);  // vthi/c 
  double uthi_He = sqrt(t_i/mic2_He); // vthi/c

  double delta = (vacuum_wavelength/(2.0*M_PI))/sqrt(n_e_over_n_crit);
  double n_e = speed_of_light*speed_of_light*m_e/(4.0*M_PI*elementary_charge2*delta*delta);
  double debye = uthe*delta;
  double wpe1ps=1e-12* speed_of_light/delta;

  double nx = 560; //11250;
  double ny = 1;         // 2D problem in x-z plane
  double nz = 396; //7950;

  double hx = box_size_x/(delta*nx);   // in c/wpe
  double hz = box_size_z/(delta*nz);
  double hy = hz;

  double cell_size_x  = delta*hx/debye;         // Cell size in Debye lengths
  double cell_size_z  = delta*hz/debye;         // Cell size in Debye lengths
//double cell_size_y  = delta*hy/debye;         // Cell size in Debye lengths

  // Set up mesh parameters
  double Lx = nx*hx;          // in c/wpe
  double Ly = ny*hy;   
  double Lz = nz*hz;   

  double dt = cfl_req*courant_length(Lx, Ly, Lz, nx, ny, nz); 

  double topology_x = 2; //16; //75; // 150 cells per MPI
  double topology_y = 1;
  double topology_z = 2; //8; //53; // 150 cells per MPI

// laser focusing parameters
  int  launch_laser = 0; //1;        // Whether to launch pump laser
  int  launch_seed  = 0; //1;        // Whether to launch seed pulse

// These are not used for Raman amp. exp. sim.
  double f_number   = 4.5;                      // f number of beam
  double lambda     = vacuum_wavelength/delta;  // Wavelength in terms of skin depths
  double waist      = f_number*lambda;          // in c/wpe, width of beam at focus
  double xfocus     = Lx/2;                     // in c/wpe, the x0 in NPIC
  double ycenter    = 0;                        // spot centered in y on lhs boundary
  double zcenter    = 0;                        // spot centered in z on lhs boundary
  double mask       = 1.5;                      // Set drive I>0 if r>mask*width at boundary.
  double width = waist*sqrt(1+(lambda*xfocus/(M_PI*waist*waist))*(lambda*xfocus/(M_PI*waist*waist)));

  double omega_0            = sqrt(1.0/n_e_over_n_crit); // w0/wpe
  double omega_seed         = omega_0 - 0.00686335;        // omega_seed/wpe
//double omega_seed         = omega_0 ;        // omega_seed/wpe

  double domega    = omega_0*0.05;               // 5% chirp w0/wpe
  double chirp_length = 2.0*1175.0;                    // in wpe^-1

  double intensity_cgs = 1e7*laser_intensity;        // [ergs/(s*cm^2)]
  double intensity_cgs_seed = 1e7*seed_intensity;    // [ergs/(s*cm^2)]

  // FIXME:  Set emax, emax_seed according to experimental parameters.

  double emax = 
    sqrt(2.0*intensity_cgs/
	 (m_e*speed_of_light*speed_of_light*speed_of_light*n_e)); // at the waist, of NPIC
  double emax_seed = 
    sqrt(2.0*intensity_cgs_seed/
	 (m_e*speed_of_light*speed_of_light*speed_of_light*n_e)); // at the waist, of NPIC

  double nsteps_cycle=trunc_granular(2*M_PI/(dt*omega_0),1)+1;
//dt = 2*M_PI/omega_0/nsteps_cycle; // nsteps_cycle time steps in one laser cycle

  double ey_xloc = Lx * 0.5;              // x location for Ey(z) write in fft diagnostic 

// intervals 
  float t_stop = 50*wpe1ps;     // Runtime in 1/omega_pe
  double pulse_length  = 0.25*wpe1ps;  // units of 1/wpe
  int poynting_interval   = 10; //int(M_PI/(dt*6.0));       // Num. steps between dumping poynting flux to resolve w/wpe=6
  int fft_ex_interval     = poynting_interval ;       // Num steps between writing Ex in fft_slice
  int fft_ey_interval     = poynting_interval ;       // Num steps between writing Ey in fft_slice
  int fft_ez_interval     = poynting_interval ;       // Num steps between writing Ez in fft_slice
  int field_interval       = int(1.0*wpe1ps/dt);         // Num. steps between saving field, hydro data
  int energies_interval = 0.1*field_interval;
// restart_interval has to be multiples of field_interval
  int restart_interval       = 0; //field_interval;
//int restart_interval     = int(51.0/dt);       // Num. steps between restart dumps
//restart_interval     = 0;  //DEBUG      // Num. steps between restart dumps

  int quota_check_interval = 20;
//int velocity_interval   = int(0.2*wpe1ps/dt);     //  Num steps between writing poynting flux; not used in NIC
  int velocity_interval   = 50; //0.1*field_interval;     //  Num steps between writing poynting flux; not used in NIC

  int eparticle_interval  = 0;
  int Hparticle_interval  = 0;
  int Heparticle_interval = 0;

  int ele_sort_freq       = 20*2; 
  int ion_sort_freq       = 5*ele_sort_freq; 

  double quota = 5.7;              // Run quota in hours.
  double quota_sec = quota*3600;  // Run quota in seconds. 
				  
  double Ne    = nppc*nx*ny*nz;             // Number of macro electrons in box
  Ne = trunc_granular(Ne, nproc());         // Make Ne divisible by number of processors       
  double Ni    = Ne;                        // Number of macro ions of each species in box
  double Npe   = Lx*Ly*Lz;                  // Number of physical electrons in box, wpe = 1
  double Npi   = Npe/(Z_H*f_H+Z_He*f_He);   // Number of physical ions in box
  double qe    = -Npe/Ne;                   // Charge per macro electron
  double qi_H  = Z_H *f_H *Npi/Ni;          // Charge per H macro ion
//double qi_He = Z_He*f_He*Npi/Ni;          // Charge per He macro ion
  double qi_He = Npi/Ni;          // Charge per He macro ion

  // Turn on integrated backscatter poynting diagnostic - right now there is a bug in this, so we 
  // only write the integrated backscatter time history on the left face. 
 
  int write_poynting_data = 1;                    // Whether to write poynting data to file (or just stdout)
  int write_backscatter_only = 0;                 // Nonzero means only write lower x face
  int write_poynting_sum   = 1;                   // Whether to write integrated Poynting data 
  int write_side_scatter   = 1;                   // Turns on side scatter if nonzero
  int write_poynting_faces = 1; //0;                   // Whether to write poynting data on sim boundary faces
  int write_eb_faces       = 1; //0;                   // Whether to write e and b field data on sim boundary faces

  // Collision parameters
  // In CGS, variance of tan theta = 2 pi e^4 n_e dt_coll loglambda / (m_ab^2 c^3)
  sim_log("Setting up particle collision parameters. ");
  double nu_e;                           // Electron collision frequency nu_0^{e\e} in units of wpe 
                                         // (See NRL Formulary, p. 32)
  {
    // Compute nu_e from plasma parameters given.  Spitzer collisions assumed.
    // According to NRL plasma formuary, nu_e = nu_0^{e\e} (2 u_the^3)
    // where nu_0^{e\e} = 4 pi e^4 n_e lambda / (m_e^2 v_e^3)
    // We need to compute n_0^{e\e} for v_e = u_the*c

    // First, compute n_e_cgs
    double n_e_crit = pow( 2*M_PI*3e10/(5.64e4*vacuum_wavelength), 2 );    
    double n_e_cgs = n_e_over_n_crit * n_e_crit; 

    // Next, define log-lambda (we could compute this, but let's set it to fixed value for now)
    double loglambda = 6;

    // Then, compute nu_e, assuming that v_e = uthe*c
    double nu_e_cgs = 4*M_PI*pow( 4.8032e-10,4 )*n_e_cgs*loglambda*pow( 9.1094e-28,-2 )*pow( uthe*3e10,-3 ); 

    // Finally, normalize to wpe
    double wpe_cgs = 5.64e4*sqrt(n_e_cgs); 
    nu_e = nu_e_cgs / wpe_cgs; 
    sim_log("* nu_e/wpe= "<<nu_e);
  }

  /////////////////////////////////////////////#####################################
  // FIXME: If collisions are needed, then we need to use the updated collision model 
  //        - this one probably is still buggy
  // 
  int do_collisions = 0;                 // Flag to turn on/off Takizuka and Abe collisions
  int self_collisions_only = 0;          // Flag for particles to scatter off same species only 
  int tstep_coll = ion_sort_freq;        // How frequently to apply collision operator: 
                                         // N.B. Needs to be a multiple of ion_sort_freq
  double dt_coll = tstep_coll*dt;        // Units of wpe

  // cvar = variance of tan theta in collision operator for electron-electron collisions
  //      = (8 pi e^4 n_e log-lambda) / (m_e^2 c^3)
  double cvar=2.0*nu_e*pow(uthe,3)*dt_coll;  // Coefficient for collision model
  double nppc_max = 40*nppc;             // Used internally in collision algorithm.  Should be sufficient? 

  // Throw warning if collision interval is too large
  if ( nu_e*2*dt_coll>0.1 ) {
    sim_log( "*** Warning: scattering interval may be too large to sample electron scattering accurately." ); 
    sim_log( "*** typical tan theta step: "<<sqrt(nu_e*2*dt_coll) ); 
  }

  // PRINT SIMULATION PARAMETERS 
  sim_log("***** Simulation parameters *****");
  sim_log("* Processors:                    "<<nproc());
  sim_log("* nu_e*2*dt_coll=                "<<nu_e*2*dt_coll);
  sim_log("* nsteps_cycle =                 "<<nsteps_cycle); 
  sim_log("* Time step, max time, nsteps =  "<<dt<<" "<<t_stop<<" "<<int(t_stop/(dt))); 
  sim_log("* wpe1ps =                       "<<wpe1ps); 
  sim_log("* Debye length, delta =          "<<debye<<" "<<delta);
  sim_log("* cell size in x, z =            "<<cell_size_x<<" "<<cell_size_z);
  sim_log("* Lx, Ly, Lz =                   "<<Lx<<" "<<Ly<<" "<<Lz);
  sim_log("* nx, ny, nz =                   "<<nx<<" "<<ny<<" "<<nz);
  sim_log("* Charge/macro electron =        "<<qe);
  sim_log("* Charge/macro He =              "<<qi_He);
  sim_log("* Charge/macro H =               "<<qi_H);
  sim_log("* Average particles/processor:   "<<Ne/nproc());
  sim_log("* Average particles/cell:        "<<nppc);
  sim_log("* Do we have mobile ions?        "<<(mobile_ions ? "Yes" : "No"));
  sim_log("* Is there He present?           "<<(He_present ? "Yes" : "No")); 
  sim_log("* Is there H present ?           "<<(H_present ? "Yes" : "No")); 
  sim_log("* Omega_0, Omega_seed:           "<<(omega_0)<<" "<<omega_seed);
  sim_log("* Omega_pe:                      "<<1);
  sim_log("* pulse_length:                  "<<pulse_length);
  sim_log("* Domega, chirp_length:          "<<domega<<" "<<chirp_length);
  sim_log("* Plasma density, ne/nc:         "<<n_e<<" "<<n_e_over_n_crit);
  sim_log("* Vac wavelength,I_laser:        "<<vacuum_wavelength<<" "<<laser_intensity);
  sim_log("* T_e, T_i, m_e, m_i_H, m_i_He:  "<<t_e<<" "<<t_i<<" "<<1<<" "<<mime_H<<" "<<mime_He);
  sim_log("* Radiation damping:             "<<damp);
  sim_log("* Fraction of courant limit:     "<<cfl_req);
  sim_log("* vthe/c:                        "<<uthe);
  sim_log("* vthi_H/c, vth_He/c:            "<<uthi_H<<" "<<uthi_He);
  sim_log("* emax:                          "<<emax);
  sim_log("* emax_seed:                     "<<emax_seed);
  sim_log("* restart interval:              "<<restart_interval); 
  sim_log("* energies_interval:             "<< energies_interval );
  sim_log("* quota_check_interval:           "<<quota_check_interval);
  sim_log("* velocity interval:             "<<velocity_interval); 
  sim_log("* poynting interval:             "<<poynting_interval); 
  sim_log("* fft ex save interval:          "<<fft_ex_interval); 
  sim_log("* fft ey save interval:          "<<fft_ey_interval); 
  sim_log("* fft ez save interval:          "<<fft_ez_interval); 
  sim_log("* f#, waist:                     "<<f_number<<" "<<waist);
  sim_log("* width, xfocus:                 "<<width<<" "<<xfocus);
  sim_log("* ycenter, zcenter, mask:        "<<ycenter<<" "<<zcenter<<mask);
  sim_log("* quota (hours):                 "<<quota);
  sim_log("* load_particles:                "<<(load_particles ? "Yes" : "No")); 
  sim_log("* do_collisions:                 "<<(do_collisions ? "Yes" : "No")); 
  sim_log("* self_collisions_only:          "<<(self_collisions_only ? "Yes" : "No")); 
  sim_log("* tstep_coll:                    "<<tstep_coll); 
  sim_log("* nppc_max:                      "<<nppc_max); 
  sim_log("* nu_e:                          "<<nu_e); 
  sim_log("* cvar:                          "<<cvar); 
  sim_log("* mime_H:                        "<<mime_H); 
  sim_log("* mime_He:                       "<<mime_He); 
  sim_log("* ele_sort_freq:                 "<<ele_sort_freq); 
  sim_log("* ion_sort_freq:                 "<<ion_sort_freq); 
  sim_log("* theta, theta_seed:             "<<theta<<" "<<theta_seed); 
  sim_log("* launch_laser:                  "<<launch_laser);  
  sim_log("* launch_seed:                   "<<launch_seed); 
  sim_log("* ey_xloc (for Ey(z) FFT write): "<<ey_xloc); 
  sim_log("* psum_integration_offset:       "<<psum_integration_offset); 
  sim_log("*********************************");


  // SETUP HIGH-LEVEL SIMULATION PARMETERS
  sim_log("Setting up high-level simulation parameters. "); 
  num_step             = 50; //int(t_stop/(dt)); 
  status_interval      = 25; 
  sync_shared_interval = status_interval/1;
  clean_div_e_interval = status_interval/1;
  clean_div_b_interval = status_interval/10;

  // For maxwellian reinjection, we need more than the default number of
  // passes (3) through the boundary handler
  // Note:  We have to adjust sort intervals for maximum performance on Cell.
  num_comm_round = 6;

  global->field_interval           = field_interval; 
  global->restart_interval         = restart_interval;
  global->energies_interval        = energies_interval;
  global->quota_check_interval     = quota_check_interval;
  global->poynting_interval        = poynting_interval; 
  global->velocity_interval        = velocity_interval; 
  global->fft_ex_interval          = fft_ex_interval; 
  global->fft_ey_interval          = fft_ey_interval; 
  global->fft_ez_interval          = fft_ez_interval; 
  global->vthe                     = uthe;     // c=1
  global->vthi_He                  = uthi_He;  // c=1
  global->vthi_H                   = uthi_H;   // c=1
  global->emax                     = emax; 
  global->emax_seed                = emax_seed; 
  global->omega_0                  = omega_0;
  global->omega_seed               = omega_seed; 
  global->domega                   = domega;
  global->chirp_length             = chirp_length;
  global->pulse_length             = pulse_length;
  global->mobile_ions              = mobile_ions; 
  global->H_present                = H_present; 
  global->He_present               = He_present; 
  global->lambda                   = lambda; 
  global->wpe1ps                   = wpe1ps; 
  global->waist                    = waist; 
  global->width                    = width; 
  global->xfocus                   = xfocus; 
  global->ycenter                  = ycenter; 
  global->zcenter                  = zcenter; 
  global->mask                     = mask; 
  global->quota_sec                = quota_sec;
  global->rtoggle                  = 0; 
  global->eparticle_interval       = eparticle_interval;
  global->Hparticle_interval       = Hparticle_interval;
  global->Heparticle_interval      = Heparticle_interval;
  global->load_particles           = load_particles; 
  global->do_collisions            = do_collisions; 
  global->self_collisions_only     = self_collisions_only; 
  global->tstep_coll               = tstep_coll;
  global->nppc_max                 = (int)nppc_max; 
  global->cvar                     = cvar;
  global->mime_H                   = mime_H; 
  global->mime_He                  = mime_He; 

  global->topology_x               = topology_x; 
  global->topology_y               = topology_y; 
  global->topology_z               = topology_z; 

  global->Lz                       = Lz;

  global->Lx                       = Lx; 
  global->xmin_domain              = 0;  
  global->ey_xloc                  = ey_xloc; 

// use old poynting diag
  global->write_poynting_data      = write_poynting_data;

  global->write_poynting_sum       = write_poynting_sum;
  global->write_poynting_faces     = write_poynting_faces;
  global->write_eb_faces           = write_eb_faces;
  global->write_backscatter_only   = write_backscatter_only;
  global->write_side_scatter       = write_side_scatter;    

  global->theta                    = theta; 
  global->theta_seed               = theta_seed; 

  global->launch_laser             = launch_laser; 
  global->launch_seed              = launch_seed; 

  global->psum_integrated_poynting_flux_tally = 0; // initialization 
  global->psum_integrated_poynting_flux_tally_new = 0; // initialization 
  global->psum_integration_offset             = psum_integration_offset; 

  // SETUP THE GRID ===============================================================
  sim_log("Setting up computational grid."); 
  grid->dx = hx;
  grid->dy = hy;
  grid->dz = hz;
  grid->dt = dt;
  grid->cvac = 1;
  grid->eps0 = eps0;

  // FIXME:  Set up the mesh for load balancing with vacuum boundaries. 

  // Partition a periodic box among the processors sliced uniformly in x: 
  define_periodic_grid( 0,         -0.5*Ly,    -0.5*Lz,       // Low corner
                        Lx,         0.5*Ly,     0.5*Lz,       // High corner
                        nx,         ny,         nz,           // Resolution
                        topology_x, topology_y, topology_z ); // Topology

  sim_log("Defined periodic grid");

  int ix, iy, iz; 
  RANK_TO_INDEX( int(rank()), ix, iy, iz );  // Get position of domain in global topology

  // Override field boundary conditions 
  if ( ix == 0) {                                 // Leftmost proc.
    set_domain_field_bc( BOUNDARY(-1,0,0), absorb_fields );
  }
  if ( ix == topology_x - 1 ) {                   // Rightmost proc.
    set_domain_field_bc( BOUNDARY( 1,0,0), absorb_fields );
  }
  if ( iz == 0) {                                 // Topmost proc.
    set_domain_field_bc( BOUNDARY(0,0,-1), absorb_fields );
  }
  if ( iz == topology_z - 1 ) {                   // Bottommost proc.
    set_domain_field_bc( BOUNDARY(0,0, 1), absorb_fields );
  }

  // SETUP THE SPECIES ==============================================================================
  sim_log("Setting up species. ");
  

  double max_local_np              = 1.3*Ne/nproc();
  double max_local_nm              = max_local_np / 10.0;
  sim_log( "num electron, ion macroparticles: "<<max_local_np );
  sim_log("- Creating electron species.");
  species_t *electron = NULL; 
  species_t *ion_H    = NULL;
  species_t *ion_He   = NULL;
  electron = define_species("electron", -1, 1, max_local_np, max_local_nm, ele_sort_freq, 1);
  if ( mobile_ions ) {
    if ( H_present ) {
      sim_log("- Creating H species.");
      ion_H  = define_species("H",  Z_H,  mime_H,  max_local_np, max_local_nm, ion_sort_freq, 1);
    }
    if ( He_present ) {
      sim_log("- Creating He species.");
      ion_He = define_species("He", Z_He, mime_He, max_local_np, max_local_nm, ion_sort_freq, 1);
    }
  }
  // Light error checking on define_species 
  if ( electron==NULL )                              sim_log_local(" ERROR: electron species not defined.");  
  if ( mobile_ions && H_present  && ion_H  == NULL ) sim_log_local(" ERROR: ion_H    species not defined.");  
  if ( mobile_ions && He_present && ion_He == NULL ) sim_log_local(" ERROR: ion_He   species not defined.");  

  sim_log("Done setting up species.");

  // SETUP THE MATERIALS ============================================================================
  sim_log("Setting up materials. "); 
  define_material( "vacuum", 1 );
  define_field_array( NULL, damp ); 

  // Paint the simulation volume with materials and boundary conditions
# define iv_region (   x<      hx*iv_thick || x>Lx  -hx*iv_thick  \
                    || z<-Lz/2+hz*iv_thick || z>Lz/2-hz*iv_thick ) /* all boundaries are i.v. */ 

  //set_region_bc( iv_region, maxwellian_reinjection_tally, maxwellian_reinjection_tally, maxwellian_reinjection_tally );
  set_region_bc( iv_region, reflect_particles, reflect_particles,reflect_particles);

  // LOAD THE PARTICLES =============================================================================
  // 
  // load a linear ramp from NE_NCR_MIN to NE_NCR_MAX

  // Mean ne/ncr for the simulation - found at center of box in x
# define NE_NCR_MEAN   (0.04)

  // ne/ncr variation as one goes from middle of box in x to either edge in x
# define NE_NCR_CHANGE (0.01)

  // Automatically defined from the above
# define NE_NCR_MAX (NE_NCR_MEAN + NE_NCR_CHANGE)
# define NE_NCR_MIN (NE_NCR_MEAN - NE_NCR_CHANGE)

  // Density macro - given x and z values, return ne/ncr
  //                 assumes that 0 <= X <= Lx and that -Lz/2 <= Z <= Lz/2
# define DENSITY( Z, X ) \
    ( NE_NCR_MIN * (1.0 - (X)/Lx) + NE_NCR_MAX * (X)/Lx )

  // Load particles
  if ( load_particles ) {
    sim_log("Loading particles.");
    // Fast load of particles--don't bother fixing artificial domain correlations
    double xmin=grid->x0, xmax=grid->x1;
    double ymin=grid->y0, ymax=grid->y1;
    double zmin=grid->z0, zmax=grid->z1;

    // The (NE_NCR_MAX / NE_NCR_MEAN) is needed to get max density right, since the
    // macroparticle charge is set in such a way as to correspond to mean density

    repeat( Ne * (NE_NCR_MAX / NE_NCR_MEAN) / (topology_x*topology_y*topology_z) ) {
      double x = uniform( rng(0), xmin, xmax );
      double y = uniform( rng(0), ymin, ymax );
      double z = uniform( rng(0), zmin, zmax );
      if ( iv_region ) continue;           // Particle fell in iv_region.  Don't load.

      // Rejection method - 2D density profile in (x,z) plane
      if ( uniform( rng(0), 0, NE_NCR_MAX ) > DENSITY( z, x ) ) continue;

      inject_particle( electron, x, y, z,
                       normal( rng(0), 0, uthe ), 
                       normal( rng(0), 0, uthe ), 
                       normal( rng(0), 0, uthe ), 
                       fabs(qe), 0, 0 );

      if ( mobile_ions ) {
        if ( H_present )  // Inject an H macroion on top of macroelectron
          inject_particle( ion_H, x, y, z,
                           normal( rng(0), 0, uthi_H ), 
                           normal( rng(0), 0, uthi_H ), 
                           normal( rng(0), 0, uthi_H ), 
                           fabs(qi_H), 0, 0 );
        if ( He_present ) // Inject an He macroion on top of macroelectron
          inject_particle( ion_He, x, y, z,
                           normal( rng(0), 0, uthi_He ), 
                           normal( rng(0), 0, uthi_He ), 
                           normal( rng(0), 0, uthi_He ), 
                           fabs(qi_He), 0, 0 );
      }
    } // repeat 
  } // if 

 /*------------------------------------------------------------------------*/

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
}

void vpic_simulation::user_diagnostics() {

# define should_dump(x) \
  (global->x##_interval>0 && remainder(step(),global->x##_interval)==0)

  // Field and hydro data writes contained in time_average_v3_He.cxx

  //------------------------------------
  //
  // Sum Poynting flux on boundaries; write summed Poynting flux to file every global->velocity_interval steps 
  //
  // Poynting flux in a given direction is defined as the projection
  // of E x B along the unit vector pointing into the simulation domain normal to box face

  field_array->copy_to_host();

  BEGIN_PRIMITIVE { 
    int ix, iy, iz; 

    // MKS 
    // n.b. 1/mu0 = c^2 * eps0 and poynting should be dA * dt * ExB * (1/mu0)
    float norm_xface = grid->dt * grid->dy * grid->dz * grid->eps0 * grid->cvac * grid->cvac;
    float norm_zface = grid->dt * grid->dx * grid->dy * grid->eps0 * grid->cvac * grid->cvac;
  
    RANK_TO_INDEX( rank(), ix, iy, iz );
  
    // 3D decomposition
    // Test whether processor is on an MPI domain edge
    // if (    ( ix==0 || ix==global->topology_x-1 )  
    //      || ( iy==0 || iy==global->topology_y-1 ) 
    //      || ( iz==0 || iz==global->topology_z-1 ) ) {

    // 2D decomposition in x and z
    if (    ( ix==0 || ix==global->topology_x-1 )  
         || ( iz==0 || iz==global->topology_z-1 ) ) {
      int i, j, k; 
      int offset = global->psum_integration_offset; 
 
      //--------------------------------------------------------------- 
      // if on lower x boundary, sum poynting flux through lower x face 
      if ( ix==0 ) { 

        // Ey * cBz on lower x, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ey  @ i,    j+0.5,k (all 1:nx+1,1:ny,1:nz+1; int 2:nx,1:ny,2:nz)
        // f(i,j,k).cbz @ i+0.5,j+0.5,k (all 1:nx,  1:ny,1:nz+1; int 1:nx,1:ny,2:nz)
        //
        for ( j=1; j<=grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            double ey, cbz;
            ey    = 0.25*(  field(1+offset,j,k).ey  + field(1+offset,j,k+1).ey
                          + field(2+offset,j,k).ey  + field(2+offset,j,k+1).ey );
            cbz   = 0.50*(  field(1+offset,j,k).cbz + field(1+offset,j,k+1).cbz ); 
            // unit normal to box pointing inward is +ix so += below 
            global->psum_integrated_poynting_flux_tally += ( ey*cbz )*norm_xface;
          } // for 
        } // for 

        // Ez * cBy on lower x, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ez  @ i,j,k+0.5     (all 1:nx+1,1:ny+1,1:nz; int 2:nx,2:ny,1:nz)
        // f(i,j,k).cby @ i+0.5,j,k+0.5 (all 1:nx,  1:ny+1,1:nz; int 1:nx,2:ny,1:nz)
        // 
        for ( j=1; j<=grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            double ez, cby;
            ez    = 0.25*(  field(1+offset,j,k).ez  + field(1+offset,j+1,k).ez    
                          + field(2+offset,j,k).ez  + field(2+offset,j+1,k).ez );
            cby   = 0.50*(  field(1+offset,j,k).cby + field(1+offset,j+1,k).cby );    
            // unit normal to box pointing inward is +ix so += below 
            global->psum_integrated_poynting_flux_tally += ( -ez*cby )*norm_xface;
          } // for 
        } // for 

      } // if 
  
      //--------------------------------------------------------------- 
      // if on upper x boundary, sum poynting flux through upper x face 
      if ( ix==global->topology_x-1 ) {

        // Ey * cBz on upper x, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ey  @ i,    j+0.5,k (all 1:nx+1,1:ny,1:nz+1; int 2:nx,1:ny,2:nz)
        // f(i,j,k).cbz @ i+0.5,j+0.5,k (all 1:nx,  1:ny,1:nz+1; int 1:nx,1:ny,2:nz)
        //
        for ( j=1; j<=grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            double ey, cbz;
            ey    = 0.25*(  field(grid->nx+0-offset,j,k).ey  + field(grid->nx+0-offset,j,k+1).ey
                          + field(grid->nx+1-offset,j,k).ey  + field(grid->nx+1-offset,j,k+1).ey );
            cbz   = 0.50*(  field(grid->nx+0-offset,j,k).cbz + field(grid->nx+0-offset,j,k+1).cbz ); 
            // unit normal to box pointing inward is -ix so -= below 
            global->psum_integrated_poynting_flux_tally -= ( ey*cbz )*norm_xface;
          } // for 
        } // for 

        // Ez * cBy on upper x, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ez  @ i,j,k+0.5     (all 1:nx+1,1:ny+1,1:nz; int 2:nx,2:ny,1:nz)
        // f(i,j,k).cby @ i+0.5,j,k+0.5 (all 1:nx,  1:ny+1,1:nz; int 1:nx,2:ny,1:nz)
        //
        for ( j=1; j<=grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            double ez, cby;
            ez    = 0.25*(  field(grid->nx+0-offset,j,k).ez  + field(grid->nx+0-offset,j+1,k).ez    
                          + field(grid->nx+1-offset,j,k).ez  + field(grid->nx+1-offset,j+1,k).ez );
            cby   = 0.50*(  field(grid->nx+0-offset,j,k).cby + field(grid->nx+0-offset,j+1,k).cby );    
            // unit normal to box pointing inward is -ix so -= below 
            global->psum_integrated_poynting_flux_tally -= ( -ez*cby )*norm_xface;
          } // for 
        } // for 

      } // if 
 
      //--------------------------------------------------------------- 
      // if on lower z boundary, sum poynting flux through lower z face 
      if ( iz==0 ) {

        // Ex * cBy on lower z, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ex  @ i+0.5,j,k     (all 1:nx,1:ny+1,1:nz+1; int 1:nx,2:ny,2:nz)
        // f(i,j,k).cby @ i+0.5,j,k+0.5 (all 1:nx,1:ny+1,1:nz;   int 1:nx,2:ny,1:nz)
        //
        for ( i=1; i<=grid->nx; ++i ) {
          for ( j=1; j<=grid->ny; ++j ) {
            double ex, cby;
            ex    = 0.25*(  field(i,j  ,1+offset).ex + field(i,j  ,2+offset).ex 
                          + field(i,j+1,1+offset).ex + field(i,j+1,2+offset).ex );
            cby   = 0.50*(  field(i,j  ,1+offset).cby+ field(i,j+1,1+offset).cby ); 
            // unit normal to box pointing inward is +ix so += below 
            global->psum_integrated_poynting_flux_tally += ( ex*cby )*norm_zface;
          } // for 
        } // for 

        // Ey * cBx on lower z, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ey  @ i,j+0.5,k     (all 1:nx+1,1:ny,1:nz+1; int 2:nx,1:ny,2:nz)
        // f(i,j,k).cbx @ i,j+0.5,k+0.5 (all 1:nx+1,1:ny,1:nz;   int 2:nx,1:ny,1:nz)
        //
        for ( i=1; i<=grid->nx; ++i ) {
          for ( j=1; j<=grid->ny; ++j ) {
            double ey, cbx;
            ey    = 0.25*(  field(i  ,j,1+offset).ey  + field(i  ,j,2+offset).ey       
                          + field(i+1,j,1+offset).ey  + field(i+1,j,2+offset).ey );
            cbx   = 0.50*(  field(i  ,j,1+offset).cbx + field(i+1,j,1+offset).cbx );    
            // unit normal to box pointing inward is +ix so += below 
            global->psum_integrated_poynting_flux_tally += ( -ey*cbx )*norm_zface;
          } // for 
        } // for 

      } // if

      //--------------------------------------------------------------- 
      // if on upper z boundary, sum poynting flux through upper z face 
      if ( iz==global->topology_z-1 ) {

        // Ex * cBy on upper z, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ex  @ i+0.5,j,k     (all 1:nx,1:ny+1,1:nz+1; int 1:nx,2:ny,2:nz)
        // f(i,j,k).cby @ i+0.5,j,k+0.5 (all 1:nx,1:ny+1,1:nz;   int 1:nx,2:ny,1:nz)
        //
        for ( i=1; i<=grid->nx; ++i ) {
          for ( j=1; j<=grid->ny; ++j ) {
            double ex, cby;
            ex    = 0.25*(  field(i,j  ,grid->nz+0-offset).ex  + field(i,j  ,grid->nz+1-offset).ex 
                          + field(i,j+1,grid->nz+0-offset).ex  + field(i,j+1,grid->nz+1-offset).ex );
            cby   = 0.50*(  field(i,j  ,grid->nz+0-offset).cby + field(i,j+1,grid->nz+0-offset).cby ); 
            // unit normal to box pointing inward is +iz so -= below 
            global->psum_integrated_poynting_flux_tally -= ( ex*cby )*norm_zface;
          } // for 
        } // for 

        // Ey * cBx on upper z, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ey  @ i,j+0.5,k     (all 1:nx+1,1:ny,1:nz+1; int 2:nx,1:ny,2:nz)
        // f(i,j,k).cbx @ i,j+0.5,k+0.5 (all 1:nx+1,1:ny,1:nz;   int 2:nx,1:ny,1:nz)
        //
        for ( i=1; i<=grid->nx; ++i ) {
          for ( j=1; j<=grid->ny; ++j ) {
            double ey, cbx;
            ey    = 0.25*(  field(i  ,j,grid->nz+0-offset).ey +  field(i  ,j,grid->nz+1-offset).ey       
                          + field(i+1,j,grid->nz+0-offset).ey +  field(i+1,j,grid->nz+1-offset).ey );
            cbx   = 0.50*(  field(i  ,j,grid->nz+0-offset).cbx + field(i+1,j,grid->nz+0-offset).cbx );    
            // unit normal to box pointing inward is +iz so -= below 
            global->psum_integrated_poynting_flux_tally -= ( -ey*cbx )*norm_zface;
          } // for 
        } // for 
      } // if                        
    } // if  

  } END_PRIMITIVE;
  //------------------------------------


#if DEBUG_SYNCHRONIZE_IO
  // DEBUG 
  mp_barrier(); 
  sim_log( "FFT done; going to Poynting." ); 
#endif 

  // POYNTING DIAGNOSTIC ========================================================================
# if 1
  //----------------------------------------------------------------------------
  // Poynting diagnostic.  Lin needs the raw E and B fields at the boundary
  // in order to perform digital filtering to extract the SRS component from the 
  // SBS.  We use random access binary writes with a stride of length:
  // 
  // stride =   2* grid->nz * global->topology_z  // lower, upper x faces
  //          + 2* grid->nx * global->topology_x; // lower, upper z faces
  // 
  // On the x faces, e1 = ey, e2 = ez, cb1 = cby, cb2 = cbz
  // On the z faces, e1 = ex, e2 = ey, cb1 = cbx, cb2 = cby
  //
  // We also write 4-element arrays of integrated poynting flux on each face:
  // 
  // vals = {lower x, upper x, lower z, upper z}
  // 
  // Note:  This diagnostic assumes uniform domains.
  // 
  // Also note:  Poynting flux in a given direction is defined as the projection
  // of E x B along the unit vector in that direction.  
  //---------------------------------------------------------------------------- 

# define ALLOCATE(A,LEN,TYPE)                                             \
    if ( !((A)=(TYPE *)malloc((size_t)(LEN)*sizeof(TYPE))) ) ERROR(("Cannot allocate.")); 

  // From grid/partition.c: used to determine which domains are on edge

  BEGIN_PRIMITIVE {
    static double *pvec =NULL, *e1vec =NULL, *e2vec =NULL, *cb1vec =NULL, *cb2vec =NULL;
    static double *gpvec=NULL, *ge1vec=NULL, *ge2vec=NULL, *gcb1vec=NULL, *gcb2vec=NULL;
    static double *psum, *gpsum, norm;
    static uint64_t stride;  // Force tmp variable in seek() to be uint64_t and not int!
    static int sum_stride, initted=0;
    int ncells_x = int(grid->nx*global->topology_x);
    int ncells_z = int(grid->nz*global->topology_z);

    if ( !initted ) {
      if ( global->write_backscatter_only ) {
        stride     = uint64_t(ncells_z); // x faces
        sum_stride = 1;
      } else {
        stride     = uint64_t( 2*( ncells_x + ncells_z ) );
        sum_stride = 4;
      }
      ALLOCATE( psum,   sum_stride, double ); ALLOCATE( gpsum,   sum_stride, double );
      ALLOCATE( pvec,   stride,     double ); ALLOCATE( gpvec,   stride,     double );
      ALLOCATE( e1vec,  stride,     double ); ALLOCATE( ge1vec,  stride,     double );
      ALLOCATE( e2vec,  stride,     double ); ALLOCATE( ge2vec,  stride,     double );
      ALLOCATE( cb1vec, stride,     double ); ALLOCATE( gcb1vec, stride,     double );
      ALLOCATE( cb2vec, stride,     double ); ALLOCATE( gcb2vec, stride,     double );
      norm = 1.0 / (grid->cvac*grid->cvac*global->emax*global->emax);
      initted=1;
    }

    // Setup bitfield for poynting diagnostic
    int poynting_flags = 0;
    if ( global->write_backscatter_only ) {
      poynting_flags = NegXFace;
    } else {
      poynting_flags = NegXFace | PosXFace;
      if ( global->write_side_scatter ) {
        poynting_flags = poynting_flags | NegZFace | PosZFace;
      }
      if ( global->write_poynting_sum ) {
        poynting_flags = poynting_flags | PoyntingSum;
      }
    }
    auto poynting_tally = poynting_flux_tally(poynting_flags, grid->eps0);
    auto poynting_tally_h = Kokkos::create_mirror_view(poynting_tally);
    Kokkos::deep_copy(poynting_tally_h, poynting_tally);
    const double flux_tally = poynting_tally_h(0) + poynting_tally_h(1)
                            + poynting_tally_h(2) + poynting_tally_h(3)
                            + poynting_tally_h(4) + poynting_tally_h(5);
    global->psum_integrated_poynting_flux_tally_new += flux_tally;

    REQUIRE_THAT(global->psum_integrated_poynting_flux_tally_new,
                 Catch::Matchers::WithinRel(global->psum_integrated_poynting_flux_tally_new, 0.0000001) );

    // FIXME:  Rewrite using permutation-symmetric macros
    // FIXME:  Don't we have to do something special for mp on Roadrunner? 

    // Note:  Will dump core if we dump poynting by mistake on time t=0  

    if ( step()>0 && should_dump(poynting) ) {
      uint64_t ii;  // To shut the compiler up.
      int i, j, k, k1, k2, ix, iy, iz, skip, index;

      // Initialize arrays to zero
      for ( ii=0; ii<stride; ++ii ) {
        pvec[ii]    = 0;
        e1vec[ii]   = 0;
        e2vec[ii]   = 0;
        cb1vec[ii]  = 0;
        cb2vec[ii]  = 0;
        gpvec[ii]   = 0;
        ge1vec[ii]  = 0;
        ge2vec[ii]  = 0;
        gcb1vec[ii] = 0;
        gcb2vec[ii] = 0;
      }
      RANK_TO_INDEX( int(rank()), ix, iy, iz );  // Get position of domain in global topology

      skip=0;

      // Lower x face
      if ( ix==0 ) {
        for ( j=0; j< grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            float e1, e2, cb1, cb2;
            // In output, the 2D surface arrays A[j,k] are FORTRAN indexed: 
            // The j quantity varyies fastest, k, slowest. 
//          index = int(  ((iy*grid->ny) + j-1)  // FIXED 
            index = int(  ((iy*grid->ny) + j-0)
                        + ((iz*grid->nz) + k-1) * (grid->ny*global->topology_y)
                        + skip);
            k1  = INDEX_FORTRAN_3(1,j+1,k+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
            k2  = INDEX_FORTRAN_3(2,j+1,k+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
            e1  = field(k2).ey;
            e2  = field(k2).ez;
            cb1 = 0.5*(field(k1).cby+field(k2).cby);
            cb2 = 0.5*(field(k1).cbz+field(k2).cbz);
            pvec[index]   = ( e1*cb2-e2*cb1 )*norm;
            e1vec[index]  = e1;
            e2vec[index]  = e2;
            cb1vec[index] = cb1;
            cb2vec[index] = cb2;
          }
        }
      }

      if ( global->write_backscatter_only==0 ) {

        skip+=ncells_z;

        // Upper x face
        if ( ix==global->topology_x-1 ) {
          for ( j=0; j< grid->ny; ++j ) {
            for ( k=1; k<=grid->nz; ++k ) {
              float e1, e2, cb1, cb2;
              index = int(  ((iy*grid->ny) + j-0)
                          + ((iz*grid->nz) + k-1) * (grid->ny*global->topology_y)
                          + skip);
              k1  = INDEX_FORTRAN_3(grid->nx-1,j+1,k+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
              k2  = INDEX_FORTRAN_3(grid->nx  ,j+1,k+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
              e1  = field(k2).ey;
              e2  = field(k2).ez;
              cb1 = 0.5*(field(k1).cby+field(k2).cby);
              cb2 = 0.5*(field(k1).cbz+field(k2).cbz);
              pvec[index]   = ( e1*cb2-e2*cb1 )*norm;
              e1vec[index]  = e1;
              e2vec[index]  = e2;
              cb1vec[index] = cb1;
              cb2vec[index] = cb2;
            }
          }
        }
        skip+=ncells_z;


        if ( global->write_side_scatter ) {
          // Lower z face
          if ( iz==0 ) {
            for ( j=0; j< grid->nx; ++j ) {
              for ( k=1; k<=grid->ny; ++k ) {
                float e1, e2, cb1, cb2;
                index = int(  ((ix*grid->nx) + j-0)  
                            + ((iy*grid->ny) + k-1) * (grid->nx*global->topology_x)
                            + skip);
                k1  = INDEX_FORTRAN_3(j+1,k+1,1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
                k2  = INDEX_FORTRAN_3(j+1,k+1,2,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
                e1  = field(k2).ex;
                e2  = field(k2).ey;
                cb1 = 0.5*(field(k1).cbx+field(k2).cbx);
                cb2 = 0.5*(field(k1).cby+field(k2).cby);
                pvec[index]   = ( e1*cb2-e2*cb1 )*norm;
                e1vec[index]  = e1;
                e2vec[index]  = e2;
                cb1vec[index] = cb1;
                cb2vec[index] = cb2;
              }
            }
          }
          skip+=ncells_x;

          // Upper z face
          if ( iz==global->topology_z-1 ) {
            for ( j=0; j< grid->nx; ++j ) {
              for ( k=1; k<=grid->ny; ++k ) {
                float e1, e2, cb1, cb2;
                index = int(  ((ix*grid->nx) + j-0)
                            + ((iy*grid->ny) + k-1) * (grid->nx*global->topology_x)
                            + skip);
                k1  = INDEX_FORTRAN_3(j+1,k+1,grid->nz-1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
                k2  = INDEX_FORTRAN_3(j+1,k+1,grid->nz  ,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
                e1  = field(k2).ex;
                e2  = field(k2).ey;
                cb1 = 0.5*(field(k1).cbx+field(k2).cbx);
                cb2 = 0.5*(field(k1).cby+field(k2).cby);
                pvec[index]   = ( e1*cb2-e2*cb1 )*norm;
                e1vec[index]  = e1;
                e2vec[index]  = e2;
                cb1vec[index] = cb1;
                cb2vec[index] = cb2;
              } // for
            } // for
          } // if                        
        } // if
      } // if

      std::tuple<Kokkos::View<double*[5]>, Kokkos::View<double[6]>> poynting_data = poynting_flux(poynting_flags, global->emax);
      auto poynting_vec = std::get<Kokkos::View<double*[5]>>(poynting_data);
      auto poynting_sum = std::get<Kokkos::View<double[6]>>(poynting_data);
      auto poynting_sum_h = Kokkos::create_mirror_view(poynting_sum);
      Kokkos::deep_copy(poynting_sum_h, poynting_sum);


      // Sum poynting flux on surface
      skip=0;

      // Lower x face
      for ( i=0, psum[0]=0; i<ncells_z; ++i ) psum[0]+=pvec[i+skip];
      // Upper x face
      skip+=ncells_z;
      for ( i=0, psum[1]=0; i<ncells_z; ++i ) psum[1]+=pvec[i+skip];
      // Lower z face
      skip+=ncells_z;
      for ( i=0, psum[2]=0; i<ncells_x; ++i ) psum[2]+=pvec[i+skip];
      // Upper z face
      skip+=ncells_x;
      for ( i=0, psum[3]=0; i<ncells_x; ++i ) psum[3]+=pvec[i+skip];

      // Sum over all surfaces
      mp_allsum_d(psum, gpsum, sum_stride);

      // Divide by number of mesh points summed over
      gpsum[0] /= static_cast<double>(ncells_z);
      if ( global->write_backscatter_only==0 ) {
        gpsum[1] /= ncells_z;
        gpsum[2] /= ncells_x;
        gpsum[3] /= ncells_x;
      } // if

      REQUIRE_THAT(gpsum[0], Catch::Matchers::WithinRel(poynting_sum_h(0), 0.0000001) );
      REQUIRE_THAT(gpsum[1], Catch::Matchers::WithinRel(poynting_sum_h(1), 0.0000001) );
      REQUIRE_THAT(gpsum[2], Catch::Matchers::WithinRel(poynting_sum_h(4), 0.0000001) );
      REQUIRE_THAT(gpsum[3], Catch::Matchers::WithinRel(poynting_sum_h(5), 0.0000001) );
    } // if
  } END_PRIMITIVE;
# endif // switch for poynting diagnostic 

#if DEBUG_SYNCHRONIZE_IO
  // DEBUG
  mp_barrier();
  sim_log( "All diagnostics done in begin_diagnostics" );
#endif
}


begin_particle_injection {
  // No particle injection for this simulation
}


begin_current_injection {
  // No current injection for this simulation
}

begin_particle_collisions {
  // the collision operator that used to be here was the old one & needs to be replaced. 
}

begin_field_injection { 
  // No field injection for this simulation
} 

TEST_CASE( "Verify Poynting flux diagnostics are correct", "[PoyntingFlux]" )
{
  // Init and run sim
  vpic_simulation simulation = vpic_simulation();

  // TODO: We should do this in a safer manner
  simulation.initialize( 0, NULL );

  while( simulation.advance() );

  simulation.finalize();

  if( world_rank==0 ) log_printf( "normal exit\n" );
}

int main(int argc, char** argv) {

    // Setup
    boot_services( &argc, &argv );

    int result = Catch::Session().run( argc, argv );

    // clean-up...
    halt_services();

    return result;
}
