//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"

// TODO: this import may ultimately be a bad idea, but it lets you paste an input deck in...

#include "deck/wrapper.h"

#include "src/species_advance/species_advance.h"
#include "src/vpic/vpic.h"

#include "linear_regression.cpp"

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
  int    particle_interval;
  int    load_particles;         // Flag to turn off particle load for testing
                                 //wave launch. 

  double xmin,xmax,ymin,ymax,zmin,zmax; // Global geometry

  double dt,time_to_SI;

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



void vpic_simulation::user_diagnostics() {}

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

void
vpic_simulation::user_initialization( int num_cmdline_arguments,
                                      char ** cmdline_argument )
{
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

  double nppc = 150;  // Average number of macro particles/cell of each species

  int topology_x = nproc();
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
  int mobile_ions           = 1;           // whether or not to push ions
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

  global->dt = dt;
  global->time_to_SI = time_to_SI;
  
  // Laser parameters
  int pulse_shape=1;                   // square pulse

  int cycles = 3;
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
  int I1_present = 0;
  int I2_present = 1;

  if(rank() == 0){
    FILE * out;
    out = fopen("oldparams.txt", "w");
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
  
  
  
  

  // Print stuff that I need for plotters and such, and with enough sig figs!
  // Be very careful modifying this.  Plotters depend on explicit locations of
  // some of these numbers.  Generally speaking, add lines at the end only.
  if(rank() == 0){
    FILE * out;
    out = fopen("params.txt", "w");
    fprintf(out, "%.17e, Time step (dt) SI\n", dt*time_to_SI);
    fclose(out);
  }
  
  // PRINT SIMULATION PARAMETERS 
  /*    
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
  */
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
    if ( I1_present ) {
      #if defined(FIELD_IONIZATION)
        ion_I1 = define_species("I1", Z_I1*e_c, m_I1_c, max_local_np_i1, max_local_nm_i1, 80, 0);
      #else
        ion_I1 = define_species("I1", Z_I1*e_c, m_I1_c, max_local_np_i1, max_local_nm_i1, 80, 0);
      #endif
    }
    if ( I2_present ) {
      #if defined(FIELD_IONIZATION)
        ion_I2 = define_species("I2", q_I2, m_I2_c, max_local_np_i2, max_local_nm_i2, 80, 0);
      #else
        ion_I2 = define_species("I2", q_I2, m_I2_c, max_local_np_i2, max_local_nm_i2, 80, 0);
      #endif
	ion_I2->pb_diag->write_ux = 1;
        ion_I2->pb_diag->write_uy = 1;
        ion_I2->pb_diag->write_uz = 1;
        ion_I2->pb_diag->write_weight = 1;
        ion_I2->pb_diag->write_posx = 1;
        ion_I2->pb_diag->write_posy = 1;
        ion_I2->pb_diag->write_posz = 1;
        finalize_pb_diagnostic(ion_I2);
    }
  } // if mobile_ions

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
	  #if defined(FIELD_IONIZATION)
	      inject_particle( ion_I2, x, y, z, 0, 0, 0, w_I2, q_I2,0,0);
	  #else
	      inject_particle( ion_I2, x, y, z, 0, 0, 0, w_I2,0,0);
	  #endif   
        } // if mobile ions
	
      } // if uniform < slab
    } // repeat    
 } // if load_particles
  
 
} // user_initialization


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
    });

  }

} // begin_field_injection

begin_particle_injection {

  // No particle injection for this simulation

}

begin_current_injection {

  // No current injection for this simulation

}


begin_particle_collisions{

  // No collisions for this simulation

}

TEST_CASE( "Check if field ionization agrees with analytic solution", "[average charge state]" )
{

    // Init and run sim
    vpic_simulation simulation = vpic_simulation();
    simulation.initialize( 0, NULL );
    while( simulation.advance() );
    simulation.finalize();
    if( world_rank==0 ) log_printf( "normal exit\n" );

    // Get parameters from file
    double dt_to_SI;
    std::ifstream file("params.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open the params.txt file." << std::endl;
    }
    std::string line;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string value;
        if (std::getline(iss, value, ',')) {
	  dt_to_SI = std::stod(value);
        }
    }
    

    // Define the analytic result
    double gamma_analytic = 2.55e12; // [s^-1] from cycle-averaged ADK model
    gamma_analytic = gamma_analytic*dt_to_SI; // in terms of timestep not seconds (analytic slope)

    
    // Read data from the file
    bool flag = 0;
    std::string filename = "Photoelectrons.txt";
    std::vector<DataPoint> data;
    std::ifstream inputFile(filename);
    if (inputFile.is_open()) {
        std::string line;
        while (std::getline(inputFile, line)) {
            std::istringstream iss(line);
            std::string token;
            DataPoint point;

            // Split the line by commas and extract the x and y values
	    std::getline(iss, token, ',');
            point.x = std::stod(token); // this is the timestep
            std::getline(iss, token, ',');
            float N_0 = std::stod(token); // this is the initial number of atoms
	    std::getline(iss, token, ','); 
            float N_ions = std::stod(token); // number of photoelectrons/ions
	    point.y = N_ions/N_0; // average charge state
            data.push_back(point);
        }
        inputFile.close();

        // Fit a line to the data
        double slope, intercept, stdErrorSlope, stdErrorIntercept,fitUncertainty;
        fitLineToData(data, slope, intercept, stdErrorSlope, stdErrorIntercept,fitUncertainty);

	// Check if the simulation agrees with the analytic result
	flag = fitAgreement(data, gamma_analytic, 0, slope, intercept, fitUncertainty);

        // Print the results
	//std::cout << "Analytic equation: y = " << gamma_analytic << "x + " << 0 << std::endl; 
        //std::cout << "Line equation: y = " << slope << "x + " << intercept << std::endl;
	//std:: cout << "fitUncertainty: " << fitUncertainty << std::endl;
	//std::cout << "flag: " << flag << std::endl;


    } else {
        std::cerr << "Failed to open the file: " << filename << std::endl;
    }

    REQUIRE(flag);
    

} // TEST_CASE



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
