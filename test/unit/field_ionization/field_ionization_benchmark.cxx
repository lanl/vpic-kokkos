//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"

// TODO: this import may ultimately be a bad idea, but it lets you paste an input deck in...

#include "deck/wrapper.h"

#include "src/species_advance/species_advance.h"
#include "src/vpic/vpic.h"

begin_globals {
  double emax;                   // E0 of the laser
  double omega_0;                // w0/wpe
  int energies_interval;        // how frequently to dump energies
  int ionization_states_interval; // how frequently to dump ionization states
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

  // Parameters for 2d and 3d Gaussian wave launch
  double lambda;
  double pulse_FWHM;
  double pulse_sigma;
  double pulse_mean;
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


void vpic_simulation::user_diagnostics() {
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
    } // if 

  }

  // ioization states
 #ifdef FIELD_IONIZATION
  if( should_dump(ionization_states) ) {
            dump_ionization_states( "rundata/ionization_states", step() ==0 ? 0 : 1 );
  } //if
 #endif
}

#define VAC 5e-6*0

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
  int I1_present = 0; // carbon
  int I2_present = 1; // hydrogen

  double n_e_over_n_crit       = 90;       // n_e/n_crit in solid slab
  double laser_intensity_W_cm2;
  if (I2_present){
    laser_intensity_W_cm2 = 1e14;      // units of W/cm^2
  } else if (I1_present){
    laser_intensity_W_cm2 = 1.e20;      // units of W/cm^2
  }
  double laser_intensity_SI    = laser_intensity_W_cm2/1e-4; // units of W/m^2
  double laser_E_SI = sqrt( (2*laser_intensity_SI)/(c_SI * eps0_SI) ); // Laser E
  double laser_E_c = laser_E_SI / E_to_SI;
  
  double lambda_SI  = 0.8e-6;
  double w0_SI = 1.25e-6; // Beam waist
  double nu = c_SI/lambda_SI;
  double nu_c = nu*time_to_SI;
  int cycles;
  double pulse_FWHM;
  double pulse_mean;
  double pulse_sigma;
  if ( I2_present ) {
    cycles = 1;
  } else if ( I1_present ) {
    cycles = 10;
    pulse_FWHM = 5/nu_c;
    pulse_mean  = cycles/nu_c; // need to shift the gaussian (want the max at the end of the sim)
    pulse_sigma = pulse_FWHM/( 2*sqrt(2*log(2) )); // sigma for gaussian function
  }

  double Lx_SI         = cycles*lambda_SI; // Simulation box size
  double Ly_SI         = w0_SI*sqrt(M_PI/2.);  // 3DCHANGE
  double Lz_SI         = cycles*lambda_SI;
  //double t_stop = 1.2e-12 / time_to_SI; // Simulation run time

  double T_e = 5. * e_SI; // Technically, this is k_B*T.  e_SI is eV to J.
  double T_i = 5. * e_SI;
  float dfrac               = 0.0; // fraction of charge density for n_Al12/ne

  // Simulation parameters
  // These are floating point to avoid a lot of casting
  // Increase resolution to ~3000 for physical results
  double nx = (60/lambda_SI)*Lx_SI; // 50 cells per wavelength (500 cells)
  double ny = 1;
  double nz = (6/lambda_SI)*Lx_SI; // 6 cells per wavelength (60 cells)

  double nppc = 5000;  // Average number of macro particles/cell of each species

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
  double A_I1    = 12;   // neutral carbon, mass number
  double A_I2    = 1;    // neutral hydrogen, mass number
  double Z_I1    = 6;    
  double Z_I2    = 1;
  double q_I1    = 1e-30;
  double q_I2    = 1e-30;   // physical charge in code units, vpic doesnt like when charge is zero
  double m_I1_SI = A_I1*mp_me*m_e_SI;
  double m_I2_SI = A_I2*mp_me*m_e_SI;
  double m_I1_c = m_I1_SI/mass_to_SI;
  double m_I2_c = m_I2_SI/mass_to_SI;
  double qn     = 1; // quantum numbers of the species
  double qm     = 0;
  double ql     = 0;
  
  // I1 - carbon
  const int num_elements_I1 = 6;
  Kokkos::View<double*> ionization_energy_I1("my_kokkos_view", num_elements_I1);
  double ionization_energy_I1_values[] = {11.26030, 24.38332, 47.8878, 64.4939, 392.087, 489.99334}; // in eV
  for (int i = 0; i < num_elements_I1; ++i) {
      ionization_energy_I1(i) = ionization_energy_I1_values[i];
  }
  // I2 - hydrogen
  const int num_elements_I2 = 1;
  Kokkos::View<double*> ionization_energy_I2("my_kokkos_view", num_elements_I2);
  double ionization_energy_I2_values[] = {13.6}; // in eV
  for (int i = 0; i < num_elements_I2; ++i) {
      ionization_energy_I2(i) = ionization_energy_I2_values[i];
  }
  // electron
  const int num_elements_electron = 1;
  Kokkos::View<double*> ionization_energy_electron("my_kokkos_view", num_elements_electron);
  double ionization_energy_electron_values[] = {0}; // in eV
  for (int i = 0; i < num_elements_electron; ++i) {
      ionization_energy_electron(i) = ionization_energy_electron_values[i];
  }

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
  double px_I2_norm = 0;

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

  global->xmin = 0;
  global->xmax = global->xmin + Lx;
  global->ymin = -.5*Ly;
  global->ymax = global->ymin + Ly;
  global->zmin = -.5*Lz;
  global->zmax = global->zmin + Lz;


  double particles_alloc = nppc*ny*nz*nx;

  double dt = cfl_req*courant_length(Lx, Ly, Lz, nx, ny, nz);
  
  // Laser parameters
  double pulse_period = 1 / nu; 
  double pulse_start = 350e-15 / time_to_SI; // time in code units
  double lambda    = lambda_SI / length_to_SI;  // Wavelength
  double omega_0 = omega_L_SI * time_to_SI;
  double emax = sqrt(2.*laser_intensity_SI/(c_SI*eps0_SI)) / E_to_SI; // code units
  
  double t_stop = cycles/nu_c; // Simulation runtime

  // Diagnostics intervals.  
  int energies_interval = 50;
  int ionization_states_interval = 1;
  int field_interval    = 10;//int(5./omega_L_SI / time_to_SI / dt);
  int particle_interval = 10*field_interval;
  int restart_interval = 400;
  int quota_check_interval = 200;
  //  int spectra_interval = int(pulse_FWHM/dt);



  double Ne    = nppc*nx*ny*nz;             // Number of macro electrons in box
  Ne = trunc_granular(Ne, nproc());         // Make Ne divisible by number of
                                            // processors       
  double Npe   = n_e_SI * Lx_SI*Ly_SI*Lz_SI; // Number of physical electrons in
                                             // box
  double qe    = -Npe/Ne;                   // Charge per macro electron


  // Parameters for the ions (note it is the same box as for electrons)
  double n_I1_SI = 1e20; // Density of I1
  double n_I2_SI = 1e20; // Density of I2
  double N_I1    = nppc*nz; //Number of macro I1 in box
  double N_I2    = nppc*nz; //Number of macro I2 in box
  N_I1 = trunc_granular(N_I1, nproc()); // make divisible by # processors
  N_I2 = trunc_granular(N_I2, nproc()); // make divisible by # processors
  double NpI1    = n_I1_SI * Lx_SI*Ly_SI*Lz_SI; // Number of physical I1 in box
  double NpI2    = n_I2_SI * Lx_SI*Ly_SI*Lz_SI; // Number of physical I2 in box
  double w_I1    = NpI1/N_I1;
  double w_I2    = NpI2/N_I2;

//  if(rank() == 0){
//    FILE * out;
//    out = fopen("oldparams.txt", "w");
//    fprintf(out, "# Parameter file used for plotters.\n");
//    fprintf(out, "%.17e   Code pulse start\n", pulse_start);
//    fprintf(out, "%d   pulse start in # of timesteps\n", int(pulse_start/dt));
//    fprintf(out, "%.17e   Code pulse FWHM\n", pulse_FWHM);
//    fprintf(out, "%d   pulse FWHM in # of timesteps\n", int(pulse_FWHM/dt));
//    fprintf(out, "%.17e   Code E to SI conversion factor\n", E_to_SI);
//    fprintf(out, "%.17e   Code time to SI conversion factor\n", time_to_SI);
//    fprintf(out, "%.17e   Code length to SI conversion factor\n", length_to_SI);
//    fprintf(out, "%.17e   Code mass to SI conversion factor\n", mass_to_SI);
//    fprintf(out, "%.17e   Code charge to SI conversion factor\n", charge_to_SI);
//    fprintf(out, "%.17e   Time step (dt), SI\n", dt*time_to_SI);
//    fprintf(out, "%.17e   Time the simulation stops in SI units\n",t_stop*time_to_SI);
//    fprintf(out, "%.17e   Time the simulation stops in code units\n",t_stop);
//    fprintf(out, "%d   Number of steps in the entire simulation\n",
//            int(t_stop/(dt)));
//    fprintf(out, "%.17e   Laser wavelength, SI\n", lambda_SI);
//    fprintf(out, "%.17e   Ratio of electron to critical density\n",
//            n_e_over_n_crit);
//    fprintf(out, "%.17e   Box size x, meters\n", Lx_SI);
//    fprintf(out, "%.17e   Box size y, meters\n", Ly_SI);
//    fprintf(out, "%.17e   Box size z, meters\n", Lz_SI);
//    fprintf(out, "%.17e   Low box corner x, meters\n",
//            global->xmin*length_to_SI);
//    fprintf(out, "%.17e   Low box corner y, meters\n",
//            global->ymin*length_to_SI);
//    fprintf(out, "%.17e   Low box corner z, meters\n",
//            global->zmin*length_to_SI);
//    fprintf(out, "%d   Number of cells in x\n", int(nx));
//    fprintf(out, "%d   Number of cells in y\n", int(ny));
//    fprintf(out, "%d   Number of cells in z\n", int(nz));
//    fprintf(out, "%d   Number of domains in x\n", topology_x);
//    fprintf(out, "%d   Number of domains in y\n", topology_y);
//    fprintf(out, "%d   Number of domains in z\n", topology_z);
//    fprintf(out, "%d   Grid data output stride in x\n", stride_x);
//    fprintf(out, "%d   Grid data output stride in y\n", stride_y);
//    fprintf(out, "%d   Grid data output stride in z\n", stride_z);
//    fprintf(out, "%.17e   N_I2\n", N_I2);
//    fprintf(out, "%.17e   NpI2\n", NpI2);
//    fprintf(out, "%.17e   w_I2\n", w_I2);
//    fprintf(out, "%d   Field interval\n", field_interval);
//    fprintf(out, "%d   Energies interval\n", energies_interval);
//    fprintf(out, "%d   Tracer interval\n", 0);
//    fprintf(out, "%d   Restart interval\n", restart_interval);
//    fprintf(out, "%d   Number of tracers per species\n", 0);
//    fprintf(out, "%d   Number of tracer species\n", 0);
//    fprintf(out, "%d   Number of variables per tracer (possibly wrong)\n",
//            0);
//    fprintf(out, "%d   Number of ranks\n", nproc());
//    fprintf(out, "%d   Number of bins in the spectra\n", 0);
//    fprintf(out, "%.17e   Spec max for electron, code units (gamma-1)\n",
//            0);
//    fprintf(out, "%.17e   Spec max for I2, code units (gamma-1)\n", 0);
//    fprintf(out, "%d   This is my rank\n", rank());
//    fprintf(out, "%.17e   Pulse FWHM, code units\n",pulse_FWHM);
//    fprintf(out, "%.17e   Pulse Sigma, code units\n",pulse_sigma);
//    fprintf(out, "%.17e   Pulse mean, code units\n",pulse_mean);
//    fprintf(out, "%.17e   Laser max E, SI\n",laser_E_SI);
//    fprintf(out, "%.17e   Laser emax, code units\n",emax);
//    fprintf(out, "%d   nppc\n",int(nppc));
//    fclose(out);
//  }
  
  
  
  

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
  global->ionization_states_interval = ionization_states_interval;
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

  global->quota_sec                = quota_sec;
  global->rtoggle                  = 0;

  global->topology_x               = topology_x;
  global->topology_y               = topology_y;
  global->topology_z               = topology_z;

  global->particle_interval          = particle_interval; 
  global->load_particles           = load_particles; 

  global->pulse_FWHM               = pulse_FWHM;
  global->pulse_sigma               = pulse_sigma;
  global->pulse_mean               = pulse_mean;
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
  grid->lambda = lambda_SI;
  grid->t_to_SI = time_to_SI;
  grid->l_to_SI = length_to_SI;
  grid->q_to_SI = charge_to_SI;
  grid->m_to_SI = mass_to_SI;

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
       ion_I1 = define_species("I1", q_I1, ionization_energy_I1, qn,qm,ql, m_I1_c, max_local_np_i1, max_local_nm_i1, 80, 0);
      #else
        ion_I1 = define_species("I1", Z_I1*e_c, m_I1_c, max_local_np_i1, max_local_nm_i1, 80, 0);
      #endif
    }
    if ( I2_present ) {
      #if defined(FIELD_IONIZATION)
       ion_I2 = define_species("I2", q_I2, ionization_energy_I2, qn,qm,ql, m_I2_c, max_local_np_i2, max_local_nm_i2, 80, 0);
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
  species_t *electron;
  #if defined(FIELD_IONIZATION)
  electron = define_species("electron", -1.*e_c, ionization_energy_electron, 0,0,0, m_e_c,max_local_np_e, max_local_nm_e, 20, 0);
  #else  
    electron = define_species("electron", -1.*e_c, m_e_c, max_local_np_e, max_local_nm_e, 20, 0);
  #endif
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

    double x = global->xmin + grid->dx;
    double y = (ymin+ymax)/2.; 
    repeat( (N_I2)/(topology_x*topology_y*topology_z) ) {  
      double z = uniform( rng(0), zmin, zmax );
	
        if ( mobile_ions ) {

	  if ( I2_present ) {
           #if defined(FIELD_IONIZATION)
               inject_particle( ion_I2, x, y, z, 0, 0, 0, w_I2, q_I2,0,0);
           #else
               inject_particle( ion_I2, x, y, z, 0, 0, 0, w_I2,0,0);
           #endif
         } // if I2 present
         
         if ( I1_present ) {
            #if defined(FIELD_IONIZATION)
                inject_particle( ion_I1, x, y, z, 0, 0, 0, w_I1, q_I1,0,0);
            #else
                inject_particle( ion_I1, x, y, z, 0, 0, 0, w_I1,0,0);
            #endif
          } // if I1 present
	  
        } // if mobile ions
    } // repeat    
 } // if load_particles
  
 
} // user_initialization


begin_field_injection { 

  if ( global->launch_wave == 0 ) return;
  
  if ( grid->x0==float(global->xmin) ) { // Node is on left boundary
    double t=grid->dt*step();
    int ny = grid->ny;
    int nz = grid->nz;;
    int sy = grid->sy;
    int sz = grid->sz;
    k_field_t& kfield = field_array->k_f_d;

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> left_edge({1, 1}, {nz+2, ny+1});
    Kokkos::parallel_for("Field injection", left_edge, KOKKOS_LAMBDA(const int iz, const int iy) {
        if ( global->I2_present==1 ){
	  kfield(1+sy*iy+sz*iz, field_var::ey) = (global->emax * cos(global->omega_0*t));
	} else if (global->I1_present==1 ){
	  kfield(1+sy*iy+sz*iz, field_var::ey) = (global->emax * cos(global->omega_0*t)) * exp(-(t-global->pulse_mean)*(t-global->pulse_mean)/(2.*global->pulse_sigma*global->pulse_sigma));
	}
    });

  }

} // begin_field_injection



begin_particle_injection {}
begin_current_injection {}
begin_particle_collisions{}

TEST_CASE( "Check if field ionization agrees with numerical solution", "[average charge state]" )
{
    // Structure to hold the data points
    struct DataPoint {
      double timestep;
      std::vector<double> ionizationStates;
    };

    // Parameters    
    bool flag = 0; // Flag that decides if test passes
    double L2_THRESHOLD = 0.01; 
    
    // Init and run sim
    vpic_simulation simulation = vpic_simulation();
    simulation.initialize( 0, NULL );
    while( simulation.advance() );
    simulation.finalize();
    if( world_rank==0 ) log_printf( "normal exit\n" );

    // Open parameters file
    double dt_to_SI;
    std::ifstream file("params.txt");
    bool paramsRead = file.is_open();
    
    // Open VPIC file
    std::string filename = "rundata/ionization_states_I2";
    std::vector<DataPoint> data;
    std::ifstream ionizationFile(filename);
    bool ionizationRead = ionizationFile.is_open(); // is params.txt opened
    
    // Open gold file
    std::string gold_file_name = GOLD_FILE;
    std::vector<DataPoint> data_analytic;
    std::ifstream goldFile(gold_file_name);
    bool goldRead = goldFile.is_open();

    
    // Only continue with calculation if both files are open
    if (paramsRead && ionizationRead && goldRead){
      // get dt_to_SI
      std::string line;
      if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string value;
        if (std::getline(iss, value, ',')) {
	  dt_to_SI = std::stod(value);
        }
      }

      // Read the VPIC data
      double initial0Plus = 0.0; // initial number of particles in the 0+ state
      bool initial0PlusSet = false;
      bool initial0PlusAdded = false; // Track if a data point with initial 0+ has been added
      std::string line1;
      while (std::getline(ionizationFile, line1)) {
	// skip comment lines
	if (line1[0] == '%') { continue; }
    
        std::istringstream iss(line1);
        DataPoint dp;
        if (!(iss >> dp.timestep)) { break; } // Error

        double ionState;
	int stateIndex = 0;
	bool isInitial0Plus = true; // Assume it's the initial 0+ state until proven otherwise
        while (iss >> ionState) { 
	  // Set initial0Plus for the first timestep (0+ state)
          if (!initial0PlusSet && stateIndex == 0) {
              initial0Plus = ionState;
              initial0PlusSet = true;
          }

	  // Check if current state is different from initial 0+ state
	  // particles are 1 grid cell away from boundary so there is a
	  // lag between ionization start in vpic and the numerical solution
          if (stateIndex == 0 && ionState != initial0Plus) {
              isInitial0Plus = false;
          }
	  
	  // Calculate relative ionization state population
          dp.ionizationStates.push_back(ionState / initial0Plus);
	  
	  stateIndex++;
        }

	// Only add this data point if it's not an initial 0+ state or if it's the first occurrence
	// in other words, remove the lad from the vpic data
        if (!isInitial0Plus || (isInitial0Plus && !initial0PlusAdded)) {
            data.push_back(dp);
            if (isInitial0Plus) {
                initial0PlusAdded = true; // Mark that the initial 0+ state has been added
            }
        }

      } // while
      ionizationFile.close();


      // Read the numerical solution from the gold file
      std::string line2;
      while (std::getline(goldFile, line2)) {
        std::istringstream iss(line2);
        std::string token;
        DataPoint dp_a;

	// First, read the timestep
        if (!std::getline(iss, token, ',')) {
            std::cerr << "Error reading timestep from gold file" << std::endl;
            break; // or handle the error as appropriate
        }
        dp_a.timestep = std::stod(token);

	// Read relative ionization states
        while (std::getline(iss, token, ',')) {
	  dp_a.ionizationStates.push_back(std::stod(token));
        }
	data_analytic.push_back(dp_a);
      }
      goldFile.close();


      // Find residuals and calculate L2 error
      int shorterSize = std::min(data.size(), data_analytic.size());
      int N_states = data[0].ionizationStates.size();
      int N_states_analytic = data_analytic[0].ionizationStates.size();
      
      // Check that datasets are not empty and that the number of ionization states is equal
      if (shorterSize == 0 || N_states != N_states_analytic) {
	flag = 0;
        std::cerr << "Error: issue with one of the datasets." << std::endl;
      } else {
	std::vector<double> l2Error(N_states, 0.0);
        for (int i = 0; i < shorterSize; i++) {
	  for (size_t j = 0; j < N_states; ++j) {
      	    double residual = data[i].ionizationStates[j] - data_analytic[i].ionizationStates[j];
            l2Error[j] += residual * residual;
	  }
        }
	
        // Check if L2 values for each state are below threshold
	bool allBelowThreshold = true; // Variable to check if all errors are below the threshold
	for (size_t k = 0; k < N_states; ++k) {
          l2Error[k] = sqrt(l2Error[k]);
          std::cout << "L2 Error for ionization state " << k << ": " << l2Error[k] << std::endl;
          // Check against a threshold for each state
          if (l2Error[k] > L2_THRESHOLD) {
	    allBelowThreshold = false; // If any state fails, set to false
	    break;
	  }
        } // k loop

	std::cout << "allBelowThreshold: " << allBelowThreshold << std::endl;

	// Check if the test passes
	if (allBelowThreshold) {
            flag = 1; // Set flag to 1 if all errors are below the threshold
        } else {
            flag = 0; // Set flag to 0 if any error is above the threshold
        }

      } // else datasets arent empty and match

    } else { // if files couldnt be opened
      flag = 0; 
      std::cerr << "Failed to open required files." << std::endl;
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
