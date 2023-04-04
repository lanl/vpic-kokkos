/*
 * Written by:
 *   Brandon M. Medina, Ph.D.
 *   Plasma Theory and Applications (XCP-6)
 *   Weapons Physics
 *   Los Alamos National Lab
 * March 2023

========================================================================
  This input deck simulates the 1st benchmark problem found in:
     J. Derouillat, A. Beck, F. Pérez, T. Vinci, M. Chiaramello, A. Grassi,
     M. Flé, G. Bouchard, I. Plotnikov, N. Aunai, J. Dargent, C. Riconda, M. Grech,
     Smilei : A collaborative, open-source, multi-purpose particle-in-cell code for plasma simulation,
     Computer Physics Communications,Volume 222,2018,Pages 351-373,ISSN 0010-4655,
     https://doi.org/10.1016/j.cpc.2017.09.024

  This simulation consists of irradiating a thin (1 cell long) neutral
  material (hydrogen) with a short (few optical-cycle long) laser with
  wavelength lambda_0 = 0.8 microns.

  The laser intensity is kept constant at I_L = 1e14 W/cm^2, corresponding
  to a normalized vector potential a_0 ~ 6.81e-3, over 10 optical cycles.
  
========================================================================
*/


#include <ctime>
using namespace std;

// This is probably unnecessary on modern parallel file systems, e.g., lustre
#define NUM_TURNSTILES 3000  

#if 0
#  define DIAG_LOG(MSG)                                          \
   {                                                             \
     FILE *fptmp=NULL;                                           \
     char fnametmp[256];                                         \
     sprintf( fnametmp, "log/diag_log.%i", rank() );             \
     if ( !(fptmp=fopen(fnametmp, "a")) ) ERROR(("Cannot open file %s",\
                 fnametmp));        \
     fprintf( fptmp, "At time step %i: %s\n", step(), MSG ); \
     fclose( fptmp );                                            \
   }
#else
#  define DIAG_LOG(MSG)
#endif


// From grid/partition.c: used to determine which domains are on edge
#define RANK_TO_INDEX(rank,ix,iy,iz) BEGIN_PRIMITIVE {                    \
    int _ix, _iy, _iz;                                                    \
    _ix  = (rank);                        /* ix = ix+gpx*( iy+gpy*iz ) */ \
    _iy  = _ix/int(global->topology_x);   /* iy = iy+gpy*iz */            \
    _ix -= _iy*int(global->topology_x);   /* ix = ix */                   \
    _iz  = _iy/int(global->topology_y);   /* iz = iz */                   \
    _iy -= _iz*int(global->topology_y);   /* iy = iy */                   \
    (ix) = _ix;                                                           \
    (iy) = _iy;                                                           \
    (iz) = _iz;                                                           \
  } END_PRIMITIVE 

// General purpose allocation macro
#ifndef ALLOCATE
#define ALLOCATE(A,LEN,TYPE)                                                  \
  if ( !((A)=(TYPE *)malloc((size_t)(LEN)*sizeof(TYPE))) )                    \
    ERROR(("Cannot allocate."));
#endif // ALLOCATE

begin_globals {
  double emax;                   // E0 of the laser
  double omega_0;                // w0/wpe
  int energies_interval;        // how frequently to dump energies
  int    field_interval;         // how frequently to dump field built-in
                                 // diagnostic
  int    restart_interval; 	 // how frequently to write restart file. 
  int    quota_check_interval;   // how often to check if quote exceeded
  int    mobile_ions;	         // flag: 0 if ions are not to be pushed
  int    I1_present;             // flag nonzero when H ions are present. 
  int    I2_present;             // flag nonzero when He ions are present.  
  int    launch_wave;            // whether or not to propagate a laser from
                                 // y-z boundary

  int FIELD_IONIZATION;
  int    particle_interval;
  int    load_particles;         // Flag to turn off particle load for testing
                                 //wave launch. 

  double xmin,xmax,ymin,ymax,zmin,zmax; // Global geometry

  // Parameters for 2d and 3d Gaussian wave launch
  double lambda;
  double waist;                  // how wide the focused beam is
  double width;
  double zcenter;		 // center of beam at boundary in z
  double ycenter;		 // center of beam at boundary in y
  double xfocus;                 // how far from boundary to focus
  double mask;			 // # gaussian widths from beam center where I nonzero

  int    pulse_shape;            // 0 for steady, indefinite pulse, 1 for
                                 // square pulse, 2 for sin^2
  double pulse_FWHM;
  double nu_c;
  double pulse_start;

  double quota_sec;              // Run quota in seconds
  int    rtoggle;         // Enables save of last two restart dumps for safety

  double topology_x;
  double topology_y;
  double topology_z;

  // Dump parameters for standard VPIC output formats
  DumpParameters fdParams;
  DumpParameters hedParams;
  DumpParameters hI1dParams;
  DumpParameters hI2dParams;
  std::vector<DumpParameters *> outputParams;


};

#define VAC 5e-6*0
// Updated by Scott V. Luedtke, XCP-6
// Density function helpers for defining how dense the plasma is as a function
// of position.  Uses SI units to avoid confusion at the expense of some extra
// conversions.  The initialization function is responsible for feeding SI
// units.  Returns a value between 0 and 1 which corresponds to the percetnage
// of n_0 the density should be for position (x,y,z)

/*
double density_func(double x, double y, double z, double xmin, double xmax,
        double ymin, double ymax, double zmin, double zmax){
    double dens = 1.;
    //Start with a nice vacuum inside the boundaries
    // 3DCHANGE --- this needs to change when in 3D
    //if( y<ymin+VAC || y > ymax-VAC) return 0.;
    if( z<zmin+VAC || z > zmax-VAC) return 0.;
    double ramp_length = 10e-6;
    double ramp1_min = xmin+VAC;
    double ramp1_max = ramp1_min+ramp_length;
    double ramp2_max = xmax-VAC-10e-6;// More space for the pulse to pass
                                      // electrons, maybe
    double ramp2_min = ramp2_max-ramp_length;
    if( x<ramp1_min || x>ramp2_max) return 0.;

    if( x>=ramp1_min && x<ramp1_max )
        dens *= (1. + cos( (x-ramp1_min)*M_PI/ramp_length - M_PI))/2.;

    if( x>=ramp2_min && x<ramp2_max )
        dens *= (1. + cos((x-ramp2_min)*M_PI/ramp_length))/2.;

    return dens;
}
*/

// A simple slab of thickness length starting at the origin
static inline double slab(double x, double y, double z, double xstart,
        double length, double zmin, double zmax, double ymin, double ymax){
    //Start with a nice vacuum inside the boundaries
    // 3DCHANGE --- this needs to change when in 3D
    //if( y<ymin+VAC || y > ymax-VAC) return 0.;
    if( z<zmin+VAC || z > zmax-VAC) return 0.;
	if (x>=xstart && x < (xstart+length))
		return 1.;
	return 0.;
}

#undef VAC

begin_initialization {
  // Alright, I'm switching this all to SI, and I'm making it consistent with
  // reality.  Dimensionless numbers should be accurate.

  // These are exact
#define h_SI (6.62607015e-34) /* J s */
#define e_SI (1.602176634e-19) /* C */
#define k_B_SI (1.380649e-23) /* J / K */
#define c_SI (2.99792458e8) /* m / s */

  // These are not exact
#define fine_structure (7.2973525693e-3) /* Unitless, true in all unit systems*/
#define m_e_SI (9.1093837015e-31) /* kg */

  //Derived
  double eps0_SI = e_SI*e_SI/(2.*fine_structure*h_SI*c_SI);
  double hbar_SI = h_SI/(2.*M_PI);

  // Now set code units
  double e_c = 1;
  double c_c = 1;
  double eps0_c = 1;
  double m_e_c = 1;
  // We can immediately set some conversion factors
  double charge_to_SI = e_SI/e_c;
  double mass_to_SI = m_e_SI/m_e_c;
  // This constrains Plank's constant
  double hbar_c = e_c*e_c/(4.*M_PI*eps0_c*c_c*fine_structure);
  // With those set, we can make many conversion factors.
  // We can arrange constants to make a length
  double length_c = hbar_c/(m_e_c*c_c);
  double length_SI = hbar_SI/(m_e_SI*c_SI);
  double length_to_SI = length_SI/length_c;
  // Similarly for time
  double time_c = hbar_c/(m_e_c*c_c*c_c);
  double time_SI = hbar_SI/(m_e_SI*c_SI*c_SI);
  double time_to_SI = time_SI/time_c;
  // This gives us one more base unit
  double current_to_SI = charge_to_SI/time_to_SI;
  // That should be all the base unit conversions we need.  For convenience:
  double vel_to_SI = c_SI/c_c;
  double energy_to_SI = mass_to_SI*pow(length_to_SI,2)/pow(time_to_SI,2);
  double momentum_to_SI = mass_to_SI*length_to_SI/time_to_SI;
  double E_to_SI = mass_to_SI*length_to_SI/(pow(time_to_SI,2)*charge_to_SI);



  // Physical parameters
  double n_e_over_n_crit       = 90;       // n_e/n_crit in solid slab
  double laser_intensity_W_cm2 = 1e14;      // units of W/cm^2
  double laser_intensity_SI    = laser_intensity_W_cm2/1e-4; // units of W/m^2
  double laser_E_SI = sqrt( (2*laser_intensity_SI)/(c_SI * eps0_SI) ); // Laser E
  double laser_E_c = laser_E_SI / E_to_SI;
  
  double lambda_SI  = 0.8e-6; // 1.058e-6;
  double w0_SI = 1.25e-6; // Beam waist

  double Lx_SI         = 16e-6; // Simulation box size
  double Ly_SI         =  w0_SI*sqrt(M_PI/2.);  // 3DCHANGE
  double Lz_SI         = 16e-6;
  //double t_stop = 1.2e-12 / time_to_SI; // Simulation run time

  double T_e = 5. * e_SI; // Technically, this is k_B*T.  e_SI is eV to J.
  double T_i = 5. * e_SI;
  float dfrac               = 0.0; // fraction of charge density for n_Al12/ne

  double dist_to_focus = 1;// For this run, put the focus at the back boundary

  // Simulation parameters
  // These are floating point to avoid a lot of casting
  // Increase resolution to ~3000 for physical results
  double nx = 1200;
  double ny = 1;
  double nz = 200;

  double nppc = 75;  // Average number of macro particles/cell of each species

  int topology_x = 8;
  int topology_y = 1;
  int topology_z = 1;
  double quota = 1;             // Run quota in hours.  
  double quota_sec = quota*3600;  // Run quota in seconds. 

  // Reduce the size of the grid data when writing to disk?
  // Must divide nx/topology_x
  int stride_x = 1, stride_y = 1, stride_z = 1;

  int rng_seed = 9818272;  // random number generator seed.
  double iv_thick = 1;  // Thickness of impermeable vacuum (in cells)
  double cfl_req = 0.98; // How close to Courant should we try to run
  double damp    = 0.0; // Level of radiation damping

  // Debug options and big switches TODO: Does the rest of the deck honor these?
  int launch_wave  = 1; // whether or not to launch wave from y-z plane
  int load_particles = 1;         // Flag to turn off particle load for testing
                                  // wave launch. William Daughton.
  int mobile_ions         = 1;           // whether or not to push ions
  int FIELD_IONIZATION    = 0;
  // For the first run particle_tracing=1, and particle_tracing=2 for the
  // second run

  // Derived quantities
  double omega_L_SI = 2.*M_PI*c_SI/lambda_SI;
  double ncr_SI = pow(omega_L_SI,2)*eps0_SI*m_e_SI/pow(e_SI,2);
  double wpe_SI = sqrt(n_e_over_n_crit*ncr_SI*pow(e_SI,2)/(eps0_SI*m_e_SI));
  double skin_depth_SI = c_SI/wpe_SI; // meters
  double n_e_SI = n_e_over_n_crit*ncr_SI;
  double debye_SI = sqrt(eps0_SI*T_e/(n_e_SI*e_SI*e_SI));


#define mp_me 1836.15267343
  double A_I1    = 1;              // proton
  double A_I2    = 2;             // neutral hydrogen
  double Z_I1    = 1;
  double Z_I2    = 1;
  double q_I2    = 1e-30;   // physical charge in code units, vpic doesnt like when charge is zero
  double m_I1_SI = A_I1*mp_me*m_e_SI;
  double m_I2_SI = A_I2*mp_me*m_e_SI;
  double m_I1_c = m_I1_SI/mass_to_SI;
  double m_I2_c = m_I2_SI/mass_to_SI;

  double c2 = c_SI*c_SI;
  // In 3 dimensions, the average energy is 3 halves the temperature
  double E_e = T_e*1.5;
  double E_i = T_i*1.5;
  // Relativistically corrected average momentum in each dimension
  double px_e_SI = sqrt(1./3.)*sqrt(E_e*E_e+2.*E_e*m_e_SI*c2)/c_SI;
  double px_I1_SI = sqrt(1./3.)*sqrt(E_i*E_i+2.*E_i*m_I1_SI*c2)/c_SI;
  double px_I2_SI = sqrt(1./3.)*sqrt(E_i*E_i+2.*E_i*m_I2_SI*c2)/c_SI;
  // VPIC uses normalized momentum, not momentum in code units.
  double px_e_norm = 0; //px_e_SI/(m_e_SI*c_SI);
  double px_I1_norm = 0; //px_I1_SI/(m_I1_SI*c_SI);
  double px_I2_norm = px_I2_SI/(m_I2_SI*c_SI);

  // Code units
  double dx = Lx_SI/nx / length_to_SI;
  double dy = Ly_SI/ny / length_to_SI;
  double dz = Lz_SI/nz / length_to_SI;

  // Code units.  We calculate the simulation box dimensions this way, rather
  // than just doing an SI to code conversion on the physical parameter,
  // because each processor gets a number of cells and a cell size, not a set
  // length and a number of cells.  Since the deck is in double precision and
  // the code is single, this probably doesn't mater.
  double Lx = nx*dx;
  double Ly = ny*dy;
  double Lz = nz*dz;

  global->xmin = -dist_to_focus / length_to_SI;
  global->xmax = global->xmin + Lx;
  global->ymin = -.5*Ly;
  global->ymax = global->ymin + Ly;
  global->zmin = -.5*Lz;
  global->zmax = global->zmin + Lz;


  double particles_alloc = nppc*ny*nz*nx;

  double dt = cfl_req*courant_length(Lx, Ly, Lz, nx, ny, nz);

  
  // Laser parameters
  int pulse_shape=1;                   // square pulse

  int cycles = 10;
  double nu = c_SI/lambda_SI;
  double nu_c = nu*time_to_SI;
  double pulse_FWHM = ((cycles/nu) / time_to_SI); // pulse duration
  double pulse_period = 1 / nu; 
  // How far in front of the boundary should the peak start?
  double pulse_start = 350e-15 / time_to_SI; // time in code units
  double lambda    = lambda_SI / length_to_SI;  // Wavelength
  double xfocus    = dist_to_focus / length_to_SI; // Distance from boundary to
                                                   // focus
  double f_number  = 1.5; // Not used!    // f number of beam
  double waist     = w0_SI / length_to_SI;  // width of beam at focus
  double ycenter   = 0;         // spot centered in y on lhs boundary
  double zcenter   = 0;         // spot centered in z on lhs boundary
  double mask      = 2.8;       // Set drive I>0 if r>mask*width at boundary.
  double Rayleigh = M_PI*waist*waist/lambda;
  double width = waist*sqrt(1.+pow(xfocus/Rayleigh,2));
  double omega_0 = omega_L_SI * time_to_SI;
  double emax = sqrt(2.*laser_intensity_SI/(c_SI*eps0_SI)) / E_to_SI; // code units
// if plane wave:
  //emax = emax*sqrt(waist/width); // at entrance if 2D Gaussian
  //emax = emax*(waist/width); // at entrance if 3D Gaussian 3DCHANGE

  
  double t_stop = pulse_FWHM; // Simulation runtime

  // Diagnostics intervals.  
  int energies_interval = 50;
  int field_interval    = 10;//int(5./omega_L_SI / time_to_SI / dt);
  int particle_interval = 10*field_interval;
  int restart_interval = 400;
  int quota_check_interval = 200;
  int spectra_interval = int(pulse_FWHM/dt);



  double Ne    = nppc*nx*ny*nz;             // Number of macro electrons in box
  Ne = trunc_granular(Ne, nproc());         // Make Ne divisible by number of
                                            // processors       
  double Npe   = n_e_SI * Lx_SI*Ly_SI*Lz_SI; // Number of physical electrons in
                                             // box
  double qe    = -Npe/Ne;                   // Charge per macro electron


  // Parameters for the ions (note it is the same box as for electrons)
  double n_I2_SI = 1e20; // Density of I2
  double N_I2    = nppc*nx*ny*nz; //Number of macro I2 in box 
  N_I2 = trunc_granular(N_I2, nproc()); // make divisible by # processors
  double NpI2    = n_I2_SI * Lx_SI*Ly_SI*Lz_SI; // Number of physical I2 in box
  double w_I2    = NpI2/N_I2;
  int I1_present = 1;
  int I2_present = 1;


  

  // Print stuff that I need for plotters and such, and with enough sig figs!
  // Be very careful modifying this.  Plotters depend on explicit locations of
  // some of these numbers.  Generally speaking, add lines at the end only.
  if(rank() == 0){
    FILE * out;
    out = fopen("params.txt", "w");
    fprintf(out, "# Parameter file used for plotters.\n");
    fprintf(out, "%.17e   Code pulse start\n", pulse_start);
    fprintf(out, "%d   pulse start in # of timesteps\n", int(pulse_start/dt));
    fprintf(out, "%.17e   Code pulse FWHM\n", pulse_FWHM);
    fprintf(out, "%d   pulse FWHM in # of timesteps\n", int(pulse_FWHM/dt));
    fprintf(out, "%.17e   Code E to SI conversion factor\n", E_to_SI);
    fprintf(out, "%.17e   Code time to SI conversion factor\n", time_to_SI);
    fprintf(out, "%.17e   Code length to SI conversion factor\n", length_to_SI);
    fprintf(out, "%.17e   Code mass to SI conversion factor\n", mass_to_SI);
    fprintf(out, "%.17e   Code charge to SI conversion factor\n", charge_to_SI);
    fprintf(out, "%.17e   Time step (dt), SI\n", dt*time_to_SI);
    fprintf(out, "%.17e   Time the simulation stops in SI units\n",t_stop*time_to_SI);
    fprintf(out, "%.17e   Time the simulation stops in code units\n",t_stop);
    fprintf(out, "%d   Number of steps in the entire simulation\n",
            int(t_stop/(dt)));
    fprintf(out, "%.17e   Laser wavelength, SI\n", lambda_SI);
    fprintf(out, "%.17e   Ratio of electron to critical density\n",
            n_e_over_n_crit);
    fprintf(out, "%.17e   Box size x, meters\n", Lx_SI);
    fprintf(out, "%.17e   Box size y, meters\n", Ly_SI);
    fprintf(out, "%.17e   Box size z, meters\n", Lz_SI);
    fprintf(out, "%.17e   Low box corner x, meters\n",
            global->xmin*length_to_SI);
    fprintf(out, "%.17e   Low box corner y, meters\n",
            global->ymin*length_to_SI);
    fprintf(out, "%.17e   Low box corner z, meters\n",
            global->zmin*length_to_SI);
    fprintf(out, "%d   Number of cells in x\n", int(nx));
    fprintf(out, "%d   Number of cells in y\n", int(ny));
    fprintf(out, "%d   Number of cells in z\n", int(nz));
    fprintf(out, "%d   Number of domains in x\n", topology_x);
    fprintf(out, "%d   Number of domains in y\n", topology_y);
    fprintf(out, "%d   Number of domains in z\n", topology_z);
    fprintf(out, "%d   Grid data output stride in x\n", stride_x);
    fprintf(out, "%d   Grid data output stride in y\n", stride_y);
    fprintf(out, "%d   Grid data output stride in z\n", stride_z);
    fprintf(out, "%.17e   N_I2\n", N_I2);
    fprintf(out, "%.17e   NpI2\n", NpI2);
    fprintf(out, "%.17e   w_I2\n", w_I2);
    fprintf(out, "%d   Field interval\n", field_interval);
    fprintf(out, "%d   Energies interval\n", energies_interval);
    fprintf(out, "%d   Tracer interval\n", 0);
    fprintf(out, "%d   Restart interval\n", restart_interval);
    fprintf(out, "%d   Number of tracers per species\n", 0);
    fprintf(out, "%d   Number of tracer species\n", 0);
    fprintf(out, "%d   Number of variables per tracer (possibly wrong)\n",
            0);
    fprintf(out, "%d   Number of ranks\n", nproc());
    fprintf(out, "%d   Number of bins in the spectra\n", 0);
    fprintf(out, "%.17e   Spec max for electron, code units (gamma-1)\n",
            0);
    fprintf(out, "%.17e   Spec max for I2, code units (gamma-1)\n", 0);
    fprintf(out, "%d   Spectra Interval\n", spectra_interval);
    fprintf(out, "%d   This is my rank\n", rank());
    fclose(out);
  }
  
  // PRINT SIMULATION PARAMETERS 
    
  sim_log("***** Simulation parameters *****");
  sim_log("* Processors:                    "<<nproc());
  sim_log("* Time step, max time, nsteps =  "<<dt*time_to_SI<<" "
          <<t_stop*time_to_SI<<" "<<int(t_stop/(dt)));
  sim_log("* Debye length (SI) =            "<<debye_SI);
  sim_log("* Electron skin depth (SI) =     "<<skin_depth_SI);
  sim_log("* dx, dy, dz (SI) =              "<<dx*length_to_SI<<" "<<dy
          *length_to_SI<<" "<<dz*length_to_SI);
  sim_log("* Field points per Debye =       "<<debye_SI/(dx*length_to_SI)<<" "
          <<debye_SI/(dy*length_to_SI)<<" "<<debye_SI/(dz*length_to_SI));
  sim_log("* Field points per skin depth =  "<<skin_depth_SI/(dx*length_to_SI)
          <<" "<<skin_depth_SI/(dy*length_to_SI)<<" "<<skin_depth_SI/(dz
              *length_to_SI));
  sim_log("* Field points per wavelength =  "<<lambda/dx<<" "<<lambda/dy<<" "
          <<lambda/dz);
  sim_log("* Lx, Ly, Lz =                   "<<Lx*length_to_SI<<" "<<Ly
          *length_to_SI<<" "<<Lz*length_to_SI);
  sim_log("* nx, ny, nz =                   "<<nx<<" "<<ny<<" "<<nz);
  sim_log("* Physical/macro electron =      "<<fabs(qe));
  sim_log("* Physical I2/macro I2 =         "<<w_I2);
  //sim_log("* Physical I1/macro I1 =         "<<qi_I1);
  sim_log("* particles_alloc =              "<<particles_alloc);
  sim_log("* Average I2 particles/processor:   "<<N_I2/nproc());
  sim_log("* Average particles/cell:        "<<nppc);
  sim_log("* Do we have mobile ions?        "<<(mobile_ions ? "Yes" : "No"));
  sim_log("* Are we launching a laser?      "<<(launch_wave ? "Yes" : "No"));
  sim_log("* Omega_0, Omega_pe:             "<<(omega_0/time_to_SI)<<" "
          <<wpe_SI);
  sim_log("* Plasma density(m^-3), ne/nc:   "<<Npe<<" "<<n_e_over_n_crit);
  sim_log("* I2 density(m^-3):   "<<NpI2);
  sim_log("* Vac wavelength, I_laser:       "<<lambda_SI<<" "
          <<laser_intensity_SI);
  sim_log("* T_e, T_i (eV):                 "<<T_e<<" "<<T_i<<" "<<m_e_c<<" "
          <<m_I1_c<<" "<<m_I2_c);
  sim_log("* m_e, m_I1, m_I2 (code units):  "<<m_e_c<<" "<<m_I1_c<<" "<<m_I2_c);
  sim_log("* Radiation damping:             "<<damp);
  sim_log("* Fraction of courant limit:     "<<cfl_req);
  sim_log("* emax at entrance, waist:       "<<emax<<" "<<emax/sqrt(waist
              /width));
  sim_log("* energies_interval:             "<<energies_interval);
  sim_log("* field_interval:                "<<field_interval);
  sim_log("* restart interval:              "<<restart_interval);
  sim_log("* random number base seed:       "<<rng_seed);
  sim_log("* waist, width, xfocus:          "<<waist<<" "<<width<<" "<<xfocus);
  sim_log("* ycenter, zcenter, mask:        "<<ycenter<<" "<<zcenter<<" "
          <<mask);
  sim_log("* tracer_interval:               "<<0);
  sim_log("* tracer2_interval:              "<<0); 
  sim_log("*********************************");

  // SETUP HIGH-LEVEL SIMULATION PARMETERS
  // FIXME : proper normalization in these units for: xfocus, ycenter, zcenter,
  // waist
  sim_log("Setting up high-level simulation parameters. "); 
  num_step             = int(t_stop/(dt)); 
  status_interval      = 100; 
//?????????????????????????????????????????????????????????????????????????????
  sync_shared_interval = status_interval/5;
  clean_div_e_interval = status_interval/5;
  clean_div_b_interval = status_interval/5;
  verbose = 0;
  // Kokkos change
  kokkos_field_injection = true;
  field_injection_interval = 1;
  particle_injection_interval = -1;
  current_injection_interval = -1;

  global->energies_interval        = energies_interval;
  global->field_interval           = field_interval; 
  global->restart_interval         = restart_interval;
  global->quota_check_interval = quota_check_interval;
  global->emax                     = emax; 
  global->omega_0                  = omega_0;
  global->FIELD_IONIZATION         = FIELD_IONIZATION;
  global->mobile_ions              = mobile_ions; 
  global->I1_present                = I1_present;
  global->I2_present               = I2_present;
  global->launch_wave              = launch_wave; 
  global->lambda                   = lambda;
  global->waist                    = waist;
  global->width                    = width;
  global->mask                     = mask;
  global->xfocus                   = xfocus; 
  global->ycenter                  = ycenter; 
  global->zcenter                  = zcenter; 

  global->quota_sec                = quota_sec;
  global->rtoggle                  = 0;

  global->topology_x               = topology_x;
  global->topology_y               = topology_y;
  global->topology_z               = topology_z;

  global->particle_interval          = particle_interval; 
  global->load_particles           = load_particles; 

  global->pulse_shape              = pulse_shape; 
  global->pulse_FWHM               = pulse_FWHM;
  global->nu_c                     = nu_c;
  global->pulse_start              = pulse_start;

  
  global->I1_present           = I1_present; 
  global->I2_present           = I2_present; 

  // SETUP THE GRID
  sim_log("Setting up computational grid."); 
  grid->dx = dx;
  grid->dy = dy;
  grid->dz = dz;
  grid->dt = dt;
  grid->cvac = c_c;
  grid->eps0 = eps0_c;

  // Partition a periodic box among the processors sliced uniformly in z: 
  define_absorbing_grid( global->xmin,global->ymin,global->zmin,  // Low corner
                        global->xmax,global->ymax,global->zmax,  // High corner
                        nx,      ny,       nz,        // Resolution
                        topology_x, topology_y, topology_z, // Topology
                        reflect_particles ); // Default particle boundary
                                             // condition

  // SETUP THE SPECIES - N.B. OUT OF ORDER WITH GRID SETUP IN CANONICAL ORDERING


  sim_log("Setting up electrons. ");

//???????????????????????????????????????????????????????????????????????
  // How oversized should the particle buffers be in case of non-uniform plasma?
  // This will not resize automatically; your run will crash if overflowed
  double over_alloc_fac = 3;
  double max_local_np_e            = over_alloc_fac*particles_alloc/nproc();
  double max_local_np_i1            = max_local_np_e;
  double max_local_np_i2            = max_local_np_e;
  // The movers are NOT resized and must be set big enough here.
  double max_local_nm_e            = max_local_np_e / 8.0;
  double max_local_nm_i1            = max_local_nm_e;
  double max_local_nm_i2            = max_local_nm_e;  

  species_t *ion_I1, *ion_I2;
  if ( mobile_ions ) {
  sim_log("Setting up ions. ");
    if ( I1_present && FIELD_IONIZATION==0 ) {ion_I1 = define_species("I1", Z_I1*e_c, m_I1_c, max_local_np_i1, max_local_nm_i1, 80, 0);}
    else if(I1_present && FIELD_IONIZATION) {ion_I1 = define_species("I1", Z_I1*e_c, m_I1_c, max_local_np_i1, max_local_nm_i1, 80, 0);}
    if ( I2_present && FIELD_IONIZATION==0 ) {
        ion_I2 = define_species("I2", q_I2, m_I2_c, max_local_np_i2, max_local_nm_i2, 80, 0);
        ion_I2->pb_diag->write_ux = 1;
        ion_I2->pb_diag->write_uy = 1;
        ion_I2->pb_diag->write_uz = 1;
        ion_I2->pb_diag->write_weight = 1;
        ion_I2->pb_diag->write_posx = 1;
        ion_I2->pb_diag->write_posy = 1;
        ion_I2->pb_diag->write_posz = 1;
        finalize_pb_diagnostic(ion_I2);
    }
    else if ( I2_present && FIELD_IONIZATION ) {
        ion_I2 = define_species("I2", q_I2, m_I2_c, max_local_np_i2, max_local_nm_i2, 80, 0);
        ion_I2->pb_diag->write_ux = 1;
        ion_I2->pb_diag->write_uy = 1;
        ion_I2->pb_diag->write_uz = 1;
        ion_I2->pb_diag->write_weight = 1;
        ion_I2->pb_diag->write_posx = 1;
        ion_I2->pb_diag->write_posy = 1;
        ion_I2->pb_diag->write_posz = 1;
        finalize_pb_diagnostic(ion_I2);
    }
  }

  if (FIELD_IONIZATION) {cout << "FIELD_IONIZATION is ON" << FIELD_IONIZATION << endl;}
  else { cout << "FIELD_IONIZATION is OFF" << endl; }


  // Electrons need to be defined last in input deck when field ionization is enabled
  species_t * electron = define_species("electron", -1.*e_c, m_e_c,
          max_local_np_e, max_local_nm_e, 20, 0);
  electron->pb_diag->write_ux = 1;
  electron->pb_diag->write_uy = 1;
  electron->pb_diag->write_uz = 1;
  electron->pb_diag->write_weight = 1;
  electron->pb_diag->write_posx = 1;
  electron->pb_diag->write_posy = 1;
  electron->pb_diag->write_posz = 1;
  finalize_pb_diagnostic(electron);  


  // SETUP THE MATERIALS
  // Note: the semantics of Kevin's boundary handler for systems where the
  // particles are absorbed requires that the field array be defined prior to
  // setting up the custom boundary handlers

  sim_log("Setting up materials. "); 
  define_material( "vacuum", 1 );
  define_field_array( NULL, damp ); 


  // Paint the simulation volume with materials and boundary conditions
# define iv_region ( x<global->xmin + dx*iv_thick || x>global->xmax-dx*iv_thick\
        /*3DCHANGE*/  \
	    /* || y<-global->ymin+dy*iv_thick || y>global->ymax-dy*iv_thick*/  \
        || z<global->zmin+dz*iv_thick || z>global->zmax-dz*iv_thick )
  /* all boundaries are i.v. */


  set_region_bc( iv_region, reflect_particles, reflect_particles,
  reflect_particles);
  //set_region_bc( iv_region, absorb_particles, absorb_particles,
  //absorb_particles);

  // LOAD PARTICLES

  // Load particles using rejection method (p. 290 Num. Recipes in C 2ed, Press
  // et al.)  

  if ( load_particles!=0 ) {
    sim_log( "Loading particles" );
  
    seed_entropy( rng_seed );  // Kevin said it should be this way
    // Fast load of particles
    double xmin = grid->x0;
    double xmax = (grid->x0+grid->nx*grid->dx);
    double ymin = grid->y0;
    double ymax = (grid->y0+grid->ny*grid->dy);
    double zmin = grid->z0;
    double zmax = (grid->z0+grid->nz*grid->dz);

    /*
    cout << "xmin = " << xmin << endl;
    cout << "xmax = " << xmax << endl;
    cout << "ymin = " << ymin << endl;
    cout << "ymax = " << ymax << endl;
    cout << "zmin = " << zmin << endl;
    cout << "zmax = " << zmax << endl;
    */

    //cout << "(N_I2)/(topology_x*topology_y*topology_z) = " << (N_I2)/(topology_x*topology_y*topology_z) << endl;
    
    repeat( (N_I2)/(topology_x*topology_y*topology_z) ) {
      double x = uniform( rng(0), xmin, xmax );
      double y = uniform( rng(0), ymin, ymax );   
      double z = uniform( rng(0), zmin, zmax );

      // if ( iv_region ) continue;   // Particle fell in iv_region.  Don't load.

      // Rejection method, based on user-defined density function
      if ( uniform( rng(0), 0, 1 ) < slab(x*length_to_SI, y*length_to_SI,
                  z*length_to_SI, global->xmin*length_to_SI, grid->dx*length_to_SI*1,
                  global->zmin*length_to_SI, global->zmax*length_to_SI,
                  global->ymin*length_to_SI, global->ymax*length_to_SI) ) {
      // third to last arg is "weight," a positive number
          //std::cout<< " injecting electron " << std::endl;
	
	//inject_particle( electron, x, y, z, 1,0,0,fabs(qe),qe,0,0);
        //            qe     normal( rng(0), 0, px_e_norm ),
        //                 normal( rng(0), 0, px_e_norm ),
        //                 normal( rng(0), 0, px_e_norm ), fabs(qe), 0, 0 );
	
	
        if ( mobile_ions ) {
	  inject_particle( ion_I2, x, y, z, 0, 0, 0, w_I2, q_I2,0,0);
	  inject_particle( ion_I1, x, y, z, 0, 0, 0, w_I2, q_I2,0,0);
	  

        }
	
 

      }
    }
 
    
 } // if load_particles

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
  sim_log("Setting up hydro and field diagnostics.");

  global->fdParams.format = band;
  sim_log ( "Field output format          : band" );

  global->hedParams.format = band;
  sim_log ( "Electron hydro output format : band" );

  global->hI1dParams.format = band;
  sim_log ( "I1 hydro output format : band" );

  global->hI2dParams.format = band;
  sim_log ( "I2 hydro output format   : band" );


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

  // Strides for field and hydro arrays.  Note that here we have defined them
  // the same for fields and all hydro species; if desired, we could use
  // different strides for each.   Also note that strides must divide evenly
  // into the number of cells in a given domain. 

  // Define strides and test that they evenly divide into grid->nx, ny, nz
  if (int(grid->nx)%stride_x) ERROR(("Stride doesn't evenly divide grid->nx."));
  if (int(grid->ny)%stride_y) ERROR(("Stride doesn't evenly divide grid->ny."));
  if (int(grid->nz)%stride_z) ERROR(("Stride doesn't evenly divide grid->nz."));

  //----------------------------------------------------------------------
  // Fields

  // relative path to fields data from global header
  sprintf(global->fdParams.baseDir, "field");

  // base file name for fields output
  sprintf(global->fdParams.baseFileName, "fields");

  // set field strides
  global->fdParams.stride_x = stride_x;
  global->fdParams.stride_y = stride_y;
  global->fdParams.stride_z = stride_z;
  sim_log ( "Fields x-stride " << global->fdParams.stride_x );
  sim_log ( "Fields y-stride " << global->fdParams.stride_y );
  sim_log ( "Fields z-stride " << global->fdParams.stride_z );

  // add field parameters to list
  global->outputParams.push_back(&global->fdParams);

  //----------------------------------------------------------------------
  // Electron hydro

  // relative path to electron species data from global header
  sprintf(global->hedParams.baseDir, "ehydro");

  // base file name for fields output
  sprintf(global->hedParams.baseFileName, "e_hydro");

  // set electron hydro strides
  global->hedParams.stride_x = stride_x;
  global->hedParams.stride_y = stride_y;
  global->hedParams.stride_z = stride_z;
  sim_log ( "Electron species x-stride " << global->hedParams.stride_x );
  sim_log ( "Electron species y-stride " << global->hedParams.stride_y );
  sim_log ( "Electron species z-stride " << global->hedParams.stride_z );

  // add electron hydro parameters to list
  global->outputParams.push_back(&global->hedParams);

  //----------------------------------------------------------------------
  // ion I1 hydro

  // relative path to electron species data from global header
  sprintf(global->hI1dParams.baseDir, "I1hydro");

  // base file name for fields output
  sprintf(global->hI1dParams.baseFileName, "I1_hydro");

  // set hydrogen hydro strides
  global->hI1dParams.stride_x = stride_x;
  global->hI1dParams.stride_y = stride_y;
  global->hI1dParams.stride_z = stride_z;
  sim_log ( "Ion species x-stride " << global->hI1dParams.stride_x );
  sim_log ( "Ion species y-stride " << global->hI1dParams.stride_y );
  sim_log ( "Ion species z-stride " << global->hI1dParams.stride_z );

  // add hydrogen hydro parameters to list
  global->outputParams.push_back(&global->hI1dParams);

  //----------------------------------------------------------------------
  // ion I2 hydro

  // relative path to electron species data from global header
  sprintf(global->hI2dParams.baseDir, "I2hydro");

  // base file name for fields output
  sprintf(global->hI2dParams.baseFileName, "I2_hydro");

  // set helium hydro strides
  global->hI2dParams.stride_x = stride_x;
  global->hI2dParams.stride_y = stride_y;
  global->hI2dParams.stride_z = stride_z;
  sim_log ( "Ion species x-stride " << global->hI2dParams.stride_x );
  sim_log ( "Ion species y-stride " << global->hI2dParams.stride_y );
  sim_log ( "Ion species z-stride " << global->hI2dParams.stride_z );

  // add helium hydro parameters to list
  global->outputParams.push_back(&global->hI2dParams);

 /*-----------------------------------------------------------------------
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

  //global->fdParams.output_variables( all );
  global->fdParams.output_variables( electric | magnetic );

  //global->hedParams.output_variables( all );
  global->hedParams.output_variables(  current_density  | charge_density |
                                       momentum_density | ke_density |
                                       stress_tensor );
  global->hI1dParams.output_variables(  current_density  | charge_density |
                                       momentum_density | ke_density |
                                       stress_tensor );
  global->hI2dParams.output_variables( current_density  | charge_density |
                                       momentum_density | ke_density |
                                       stress_tensor );

 /*--------------------------------------------------------------------------
  * Convenience functions for simlog output
  *------------------------------------------------------------------------*/
  char varlist[256];

  create_field_list(varlist, global->fdParams);
  sim_log ( "Fields variable list: " << varlist );

  create_hydro_list(varlist, global->hedParams);
  sim_log ( "Electron species variable list: " << varlist );

  create_hydro_list(varlist, global->hI1dParams);
  sim_log ( "I1 species variable list: " << varlist );

  create_hydro_list(varlist, global->hI2dParams);
  sim_log ( "I2 species variable list: " << varlist );

 /*------------------------------------------------------------------------*/

  // Set static field
  //set_region_field( everywhere, 0, 0, 0,                    // Electric field
  //                  0, -global->emax*1e-2, 0 ); // Magnetic field


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


begin_diagnostics {

  //if ( step()%1==0 ) sim_log("Time step: "<<step()); 

# define should_dump(x) \
  (global->x##_interval>0 && remainder(step(),global->x##_interval)==0)

  if ( step()==0 ) {
    // A grid dump contains all grid parameters, field boundary conditions,
    // particle boundary conditions and domain connectivity information. This
    // is stored in a binary format. Each rank makes a grid dump
    //dump_grid("grid");

    // A materials dump contains all the materials parameters. This is in a
    // text format. Only rank 0 makes the materials dump
    //dump_materials("materials");

    // A species dump contains the physics parameters of a species. This is in
    // a text format. Only rank 0 makes the species dump
    //dump_species("species");

    if ( rank()==0 ) {
    dump_mkdir("rundata");
    dump_mkdir("fft");
    dump_mkdir("field");
    dump_mkdir("ehydro");
    dump_mkdir("I1hydro");
    dump_mkdir("I2hydro");
    dump_mkdir("restart");
    dump_mkdir("particle");
    dump_mkdir("pb_diagnostic");

    // Turn off rundata for now
    // dump_grid("rundata/grid");
    // dump_materials("rundata/materials");
    // dump_species("rundata/species");
    global_header("global", global->outputParams);

    } // if 

  }

  // energy in various fields/particles 
  if( should_dump(energies) ) {
            dump_energies( "rundata/energies", step() ==0 ? 0 : 1 );
  } //if

  if ( should_dump(field) ) {
    field_array->copy_to_host();

    field_dump( global->fdParams );

    if ( global->load_particles ) {
      species_t * sp;
      LIST_FOR_EACH(sp, species_list){
          sp->copy_to_host();
      }
      hydro_dump( "electron", global->hedParams );
      if ( global->mobile_ions ) {
        if ( global->I1_present ) hydro_dump( "I1", global->hI1dParams );
        if ( global->I2_present ) hydro_dump( "I2", global->hI2dParams );
      }
    }

    // This is also a good time to write the pb_diag buffers to disk
    species_t * sp;
    LIST_FOR_EACH(sp, species_list){
        if(sp->pb_diag) pbd_buff_to_disk(sp->pb_diag);
    }

  }

  if(step()==num_step){
  // Dump all remaining particles into the pb_diagnostic for convienience
    // This spits out a bunch of warnings
    // TODO: Make this a nice function for the user
    species_t * sp;
    LIST_FOR_EACH(sp, species_list){
      if(sp->pb_diag){
        sp->copy_to_host();
          for(int p_index=0; p_index<sp->np; p_index++){
            pbd_write_to_buffer(sp, sp->k_p_h, sp->k_p_i_h, p_index);
          }
          // Flush the buffers
          pbd_buff_to_disk(sp->pb_diag);
      }
    }
  }

  // Particle dump data
#if 1
  if ( should_dump(particle) && global->load_particles ) {
    dump_particles( "electron", "particle/eparticle" );
    if ( global->mobile_ions ) {
      if (global->I1_present) dump_particles( "I1", "particle/I1particle" );
      if (global->I2_present) dump_particles( "I2", "particle/I2particle" );
    }
  }
#endif



  // Restart dump filenames are by default tagged with the current timestep.
  // If you do not want to do this add "0" to the dump_restart arguments. One
  // reason to not tag the dumps is if you want only one restart dump (the most
  // recent one) to be stored at a time to conserve on file space. Note: A
  // restart dump should be the _very_ _last_ dump made in a diagnostics
  // routine. If not, the diagnostics performed after the restart dump but
  // before the next timestep will be missed on a restart. Restart dumps are
  // in a binary format. Each rank makes a restart dump.

  // Restarting from restart files is accomplished by running the executable
  // with "restart restart" as additional command line arguments.  The
  // executable must be identical to that used to generate the restart dumps or
  // else the behavior may be unpredictable. 

  // Note:  restart is not currently working with custom boundary conditions
  // (such as the reflux condition) and has not been tested with emission 
  // turned on.  
  
  // if ( step()!=0 && !(step()%global->restart_interval) )
  // dump_restart("restart",0); 


  // Shut down simulation if wall clock time exceeds global->quota_sec.
  // Note that the mp_elapsed() is guaranteed to return the same value for all
  // processors (elapsed time on proc #0), and therefore the abort should
  // be synchronized across processors.

#if 1
  if ( global->quota_check_interval &&
          (step()%global->quota_check_interval)==0 ) {
    if ( uptime() > global->quota_sec ) {

    BEGIN_TURNSTILE(NUM_TURNSTILES) {
//    checkpt( "restart/restart", step() );
      checkpt( "restart/restart", global->rtoggle );
    } END_TURNSTILE;

      // SVL - This barrier needs to be before the sim_log, else the sim_log is
      // probably a lie.
      mp_barrier();
      if (world_rank==0) {
        // We are done checkpointing.  Leave a note saying what the newest
        // checkpoint is.
        FILE * newckp = fopen("newest_checkpoint.txt", "w");
        fprintf(newckp, "restart/restart.%i", global->rtoggle);
        fclose(newckp);
      }
      sim_log( "Restart dump invoked by quota completed." );
      sim_log( "Allowed runtime exceeded for this job.  Terminating." );
      mp_barrier(); // Just to be safe
      halt_mp();
      exit(0);
    }
  }
#endif

#if 1
  if ( global->restart_interval>0 &&
          ((step()%global->restart_interval)==0) ) {
    //double dumpstart = uptime();

    // NOTE: If you want an accurate start time, you need an MPI barrier.
    // This can feasibly slow down the simulation, so you might not want it
    // unless you are benchmarking the dump.
    //mp_barrier();
    std::time_t dumpstart = std::time(NULL);

//???????????????????????????????????????????????????????????????????????
    BEGIN_TURNSTILE(NUM_TURNSTILES) {
//    checkpt( restart_fbase[global->rtoggle], step() );
      checkpt("restart/restart", global->rtoggle);
 } END_TURNSTILE;

    // SVL - This is a kind of messy way to calculate this.
    //double dumpelapsed = uptime() - dumpstart;

    // Do put a barrier here so we don't say we're done before we are
    mp_barrier();

    if (world_rank==0) {
      double dumpelapsed = std::difftime(std::time(NULL), dumpstart);
      sim_log("Restart duration "<<dumpelapsed<<" seconds");

      // We are done checkpointing.  Leave a note saying what the newest
      // checkpoint is.
      FILE * newckp = fopen("newest_checkpoint.txt", "w");
      fprintf(newckp, "restart/restart.%i", global->rtoggle);
      fclose(newckp);
    }

    global->rtoggle^=1;
  }
#endif

}


begin_particle_injection {
  // No particle injection for this simulation
}


begin_current_injection {
  // No current injection for this simulation
}

begin_field_injection { 
  // Inject a light wave from lhs boundary with E aligned along y
  // For 2d, 3d, use scalar diffraction theory for the Gaussian beam source. 
  // See, e.g., Lin's notes on this or Brian's notes of 12 March 2005. 
  // (Note the sqrt(2) discrepancy in the two regarding definition of waist. 
  // The driver below assumes Brian's definition). 

  // Note, for quiet startup (i.e., so that we don't propagate a delta-function
  // noise pulse at time t=0) we multiply by a constant phase term exp(i phi)
  // where: 
  //   phi = k*global->xfocus+atan(h)/2    (2D)
  //   phi = k*global->xfocus+atan(h)      (3D)

//# define loop_over_left_boundary \
//    for ( int iz=1; iz<=grid->nz+1; ++iz ) for ( int iy=1; iy<=grid->ny; ++iy )
//# define DY    ( grid->y0 + (iy-0.5)*grid->dy - global->ycenter )
//# define DZ    ( grid->z0 +  (iz-1) *grid->dz - global->zcenter )
//# define R2    ( DY*DY+DZ*DZ )
//# define PHASE ( global->omega_0*t + h*R2/(global->width*global->width) )
//# define MASK  ( R2<=pow(global->mask*global->width,2) ? 1 : 0 )


  if ( global->launch_wave == 0 ) return;

  if ( grid->x0==float(global->xmin) ) { // Node is on left boundary
    double alpha = grid->cvac*grid->dt/grid->dx;
    double emax_coeff = ((4/(1+alpha))*global->omega_0*grid->dt*global->emax);
    double t=grid->dt*step();

#if 1
    double pulse_shape_factor=1;
    if ( global->pulse_shape>=1 ) pulse_shape_factor=( t<global->pulse_FWHM ? 1
            : 0 );
    if ( global->pulse_shape==2 ) {
    float sin_t_tau = sin(t*M_PI/global->pulse_FWHM);
      pulse_shape_factor =( t<global->pulse_FWHM ? sin_t_tau : 0 );
  }
    // Hyperbolic secant profile
    if (global->pulse_shape==3){
        double fac = 2.*log(1.+sqrt(2.))/global->pulse_FWHM;
        // This is actually a time, not a distance, since the pulse_FWHM is time
        double z = (t - global->pulse_start);
        z *= fac;
        pulse_shape_factor = 2./(exp(z)+exp(-z));
    }

    // square wave
    int    square_wave        = 1;
    double square_wave_factor = 1;
    if ( square_wave>=1 ) square_wave_factor=( sin( (t*global->nu_c)  * 2.0 * M_PI)>=0.0 ? 1 : -1 );

#endif 

    //double prefactor = emax_coeff*sqrt(2.0/M_PI); // Wave norm at edge of box
    double prefactor = global->emax;  // Wave norm at edge of box
    // Rayleigh length
    double rl     = M_PI*global->waist*global->waist/global->lambda;
    double h = global->xfocus/rl;                 // distance/Rayleigh length

// Kokkos Port
    int ny = grid->ny;
    int nz = grid->nz;
    float dy = grid->dy;
    float dz = grid->dz;
    float y0 = grid->y0;
    float z0 = grid->z0;
    float ycenter = global->ycenter;
    float zcenter = global->zcenter;
    float width = global->width;
    float mask = global->mask;
    float omega_0 = global->omega_0;
    float dy_offset = y0 - ycenter;
    float dz_offset = z0 - zcenter;
    int sy = grid->sy;
    int sz = grid->sz;
    //printf("Injecting\n");

    k_field_t& kfield = field_array->k_f_d;

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> left_edge({1, 1}, {nz+2, ny+1});
    Kokkos::parallel_for("Field injection", left_edge, KOKKOS_LAMBDA(const int iz, const int iy) {
        auto DY =( (iy-0.5)*dy + dy_offset );
        if(ny==1) DY = 0.;
        auto DZ =( (iz-1  )*dz + dz_offset );
        auto R2   =( DY*DY + DZ*DZ );
        auto PHASE=( omega_0*t + h*R2/(width*width) );
        auto MASK =( R2<=pow(mask*width,2) ? 1 : 0 );
        //kfield(1+sy*iy+sz*iz, field_var::ey) += (prefactor * cos(PHASE) * exp(-R2/(width*width)) * MASK * pulse_shape_factor);
	kfield(1+sy*iy+sz*iz, field_var::ey) = (prefactor * sin(PHASE));

	//	std::cout << "square_wave_factor = " << square_wave_factor << std::endl;
    });

  }

} 

begin_particle_collisions {
  // No particle collisions for this simulation
}

