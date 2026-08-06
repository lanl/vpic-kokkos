/////////////////////////////////////////////////////
//
//   2D mirror with neutral beam injection. Two beams are
//   injected near the x-axis perpendicular to device axis (z-axis).
//
//   Deck and plasma conditions are based off of Aaron and Kai's
//   deck called ANVIL_DD_cylinder.cxx (last updated Feb 2026)
//   and Ari and Adam's deck mirror-2D-K.cxx
//
//////////////////////////////////////////////////////
//
// History:
//   2023 Sep/Oct - original deck from Ari Le (LANL)
//   2023 Nov - major cleanup by Aaron Tran (UW--Madison)
//   2023 Nov - forked GDT-3d to target WHAM-phase2-3d parameters
//   2024 Feb/Jun - forked to target BEAM parameters
//   2025 Mar - remove extraneous code to simplify deck for new users
//   2025 Dec/2026 Feb - convert to 2D cylinder & use ANVIL-DD parameters
//   2026 Mar - add NBI and chemistry, M Lavell (LANL)
//   2026 Apr - add fusion reactions and Python inputs, M Lavell (LANL)
//
//////////////////////////////////////////////////////

// ============================================================================
// User configuration includes and macros
// ============================================================================

#define NUM_TURNSTILES 256

#define DUMP_WITH_HDF5

// #ifdef DUMP_WITH_HDF5
// #ifndef VPIC_ENABLE_HDF5
// #error "VPIC_ENABLE_HDF5" is required
// #endif
// #endif

// #include "sigma.h"
// #include "dsdOmega.h"
// #include "dump_info.cxx"
#include <tr1/cmath>
#define MATHLIB() std::tr1::

enum BEAM_INJECTION_PLANE { X, Y, Z };

const double ERG_PER_KEV = 1.602e-9;

template<typename T>
T SQR(T val) { return val*val; }

// Employ turnstiles to partially serialize the high-volume file writes.
// In this case, the restart dumps.  Set NUM_TURNSTILES to be the desired
// number of simultaneous writes.
#define NUM_TURNSTILES 256

#define ENABLE_FUSION_RXS 1
#define ENABLE_ANISOTROPIC_FUSION 0
#define INCLUDE_TRITIUM_FUEL 0
#define CYL_FIELDS 1

#define DO_COLLISIONS 0
#define INJECT_BEAMS 0

// ============================================================================
// Other macros
// ============================================================================

// Determine which domains are along the boundaries
// using macros from grid/partition.c
# define RANK_TO_INDEX(rank,ix,iy,iz) BEGIN_PRIMITIVE {           \
    int _ix, _iy, _iz;                                            \
    _ix  = (rank);                /* ix = ix+gpx*( iy+gpy*iz ) */ \
    _iy  = _ix/int(topology_x);   /* iy = iy+gpy*iz */            \
    _ix -= _iy*int(topology_x);   /* ix = ix */                   \
    _iz  = _iy/int(topology_y);   /* iz = iz */                   \
    _iy -= _iz*int(topology_y);   /* iy = iy */                   \
    (ix) = _ix;                                                   \
    (iy) = _iy;                                                   \
    (iz) = _iz;                                                   \
  } END_PRIMITIVE

#define INDEX_TO_RANK(ix,iy,iz,rank) do {                                                \
    int _ix = (ix), _iy = (iy), _iz = (iz);                                              \
    /* Wrap processor index periodically */                                              \
    while(_ix>=int(topology_x)) _ix-=int(topology_x); while(_ix<0) _ix+=int(topology_x); \
    while(_iy>=int(topology_y)) _iy-=int(topology_y); while(_iy<0) _iy+=int(topology_y); \
    while(_iz>=int(topology_z)) _iz-=int(topology_z); while(_iz<0) _iz+=int(topology_z); \
    /* Compute the rank */                                                               \
    (rank) = _ix + int(topology_x)*( _iy + int(topology_y)*_iz );                        \
  } while(0)

// Apply field-injection boundary conditions
// using macro from local.c
#define XYZ_LOOP(xl,xh,yl,yh,zl,zh)             \
  for( z=zl; z<=zh; z++ )                       \
    for( y=yl; y<=yh; y++ )                     \
      for( x=xl; x<=xh; x++ )

// Dummy struct for Coulomb collision model
struct CCModel {
  KOKKOS_INLINE_FUNCTION float
  operator() (float vr, float Z1, float Z2=0.0) const 
  {
    return 1.0;
  }
};

// ============================================================================
// begin_globals
// ============================================================================
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
// ============================================================================
begin_globals {

  // Intervals for various outputs and checks
  int restart_interval;
  int fields_interval;
  int particle_interval;
  int energies_interval;

  // Restart dumps and simulation termination from input deck
  int rtoggle;
  int quota_check_interval;
  double quota_sec;

  // Variables for output dumps
  DumpParameters fieldDumpParams;
  DumpParameters hydroDSeedDumpParams;
  DumpParameters hydroDBeamDumpParams;  
  DumpParameters hydroNeutronDumpParams;
  DumpParameters hydroHelium3DumpParams;
  DumpParameters hydroProtonDumpParams;
  DumpParameters hydroTritiumDumpParams;
  DumpParameters hydroHelium4DumpParams;
  std::vector<DumpParameters *> outputParams;

  int particle_stride;

  // Plasma reference values (used in beam injection)
  double ref_di;
  double ref_vA0;
  double ref_E0;
  double ref_n0;

  // Domain size and discretization
  double Lx, Ly, Lz;
  double nx, ny, nz;
  double topology_x, topology_y, topology_z;

  // Variables for beam injection
#define NUM_BEAMS (2)
  int n_beams;
  
  // User inputs
  //double n_beam[NUM_BEAMS];    // density (units of n0)
  double E_beam[NUM_BEAMS];      // energy (keV)
  double P_beam[NUM_BEAMS];      // power (MW)
  double T_beam_para[NUM_BEAMS]; // thermal energy (units of keV)
  double T_beam_perp[NUM_BEAMS]; // thermal energy (units of keV)
  double x_beam[NUM_BEAMS][3];   // center (units of di)
  double r_beam[NUM_BEAMS];      // radius (units of di)
  double theta_beam[NUM_BEAMS];  // polar angle
  double phi_beam[NUM_BEAMS];    // azimuthal angle
  int ppc_beam[NUM_BEAMS];       // particles-per-cell per injection step
  BEAM_INJECTION_PLANE beam_inj_plane[NUM_BEAMS]; // plane from which particles are injected

  // Derived beam quantities
  int icell_beam[NUM_BEAMS][6];              // range of cells for injection (ix0, ix1, iy0, iy1, iz0, iz1)
  double w_beam[NUM_BEAMS];                  // particle weights (derived from P_beam, E_beam, area, volume, ppc)
  double v_mag[NUM_BEAMS];                   // velocity magnitude (derived from E_beam)
  double v_thermal[NUM_BEAMS][2];            // thermal velocity (derived from T_para, T_perp)
  double rotation_matrix[NUM_BEAMS][3][3];   // rotation transform (derived from theta, phi)
  bool contains_injection_region[NUM_BEAMS]; // specify if rank domain includes injection region

}; // end global_params


// ============================================================================
// begin_initialization
// ============================================================================
// At this point, there is an empty grid and the random number generator is
// seeded with the rank. The grid, materials, species need to be defined.
// Then the initial non-zero fields need to be loaded at time level 0 and the
// particles (position and momentum both) need to be loaded at time level 0.
// ============================================================================
begin_initialization {

  // --------------------------------------------------------------------------
  // Plasma parameters - unit system
  // --------------------------------------------------------------------------

  // Define the system of code units for this problem
  double ec   = 1;                // Charge normalization
  double mi   = 1;                // Mass normalization
  double c    = 1;                // Velocity normalization (speed of light)
  double di   = 1;                // Length normalization (ion inertial length)
  double eps0 = 1;                // Permittivity of space

  // Define the system of physical units for this problem
  //
  // Hybrid-PIC simulations without collisions are scale free.
  // Each simulation represents a two-parameter family of solutions for
  // arbitrary values of reference (B_0,n_0), which correspond to B_code=1 and
  // n_code=1 respectively.
  //
  // "Alfvenic" unit scheme (velocity normalized to vAi, time normalized to ion
  // cyclotron frequency) is used to convert from code to physical units.
  //
  // We specify reference values in physical units because:
  // 1. Decks are designed for specific (B_0,n_0).
  // 2. CQL3D-m/Pleiades initial conditions must be converted from physical
  //    units to code units.

  double ref_b0  = 3.0e4;      // Central cell magnetic field in Gauss
  double ref_n0  = 1e14;       // Central cell reference density in cm^-3
  double ref_mp  = 1.6726e-24; // Proton mass in grams
  // double ref_mi  = 3.344e-24;  // Ion mass in grams (deuterium)
  double ref_mi  = 1.6605e-24;  // Atomic mass unit in grams
  double ref_me  = 9.109e-28;  // Electron mass in grams
  double ref_qi  = 4.803e-10;  // Ion charge in esu (Gaussian CGS unit)
  double ref_c   = 2.998e10;   // Speed of light in cm/s
  double ref_vA0 = ref_b0/sqrt(4*M_PI*ref_n0*ref_mi);  // Alfven speed in cm/s
  double ref_di = ref_c/sqrt(4*M_PI*ref_n0*ref_qi*ref_qi/ref_mi);  // Ion skin depth in cm
  double ref_E0 = ref_mi*ref_vA0*ref_vA0; // Reference energy in erg
  double ref_wci = ref_vA0 / ref_di;
  double ref_wce = ref_wci * ref_mi / ref_me;

  if (world_rank == 0) {
    MESSAGE(("Reference ion skin depth di = %e cm", ref_di));
    MESSAGE(("Reference ion Alfven speed vA0 = %e cm", ref_vA0));
  }

  // --------------------------------------------------------------------------
  // Plasma parameters - particle space and velocity distributions
  // --------------------------------------------------------------------------

  // Maxwellian w/ Gaussian radial density profile
  double Te_erg = 1 * ERG_PER_KEV;  // Electron temperature in erg (CGS unit)
  double Ti_erg = 1 * ERG_PER_KEV;  // Ion temperature in erg (CGS unit)
  double ni = 30000000000000.0;

  double rsigma = 5;  // radial profile's gaussian width in code units

  // Density spatial distribution
  // fillvac = true -> spawn particles in all vacuum regions of simulation
  //                   domain, up to level of Ohm's law density floor
  // fillvac = false -> do not put particles in vacuum regions
  bool fillvac = true;

  double b0_G = 30000.0;

  // --------------------------------------------------------------------------
  // Hybrid-PIC solver parameters
  // --------------------------------------------------------------------------

  double hyb_b0  = b0_G/ref_b0;     // Constant By perpendicular to 1D domain
  double hyb_te  = Te_erg/ref_E0;   // Electron temperature kB*Te/(mi*vA0^2)
  double hyb_den = ni/ref_n0;       // Density corresponding to hyb_te (for hyb_gamma!=1)
  double hyb_den_floor_ohm = 0.2; // Density floor for Ohm's law update
  double hyb_den_floor_pe  = 0.1; // Density floor for electron pressure update
  double hyb_eta      = 0;          // Resistivity
  double hyb_hypereta = 1e-2;       // Hyper-resistivity
  double hyb_gamma    = 1.0;        // Electron fluid adiabatic index
  double hyb_nsub     = 10;        // Number of field subcycles
  int hyb_nsm         = 3;          // Smoothing passes per timestep for ion moments
  int hyb_nsmb        = 0;          // B-field smoothing interval; 0 disables
                                    // WARNING this diffuses energy, use only
                                    // if you know what you're doing

  // --------------------------------------------------------------------------
  // Simulation domain parameters
  // --------------------------------------------------------------------------

  double Lx = 36*di; // double size of box in x dimension
  double Ly = 2.0*M_PI; // size of box in y dimension
  double Lz = 360*di; // size of box in z dimension, 36/320 = 0.1125

  // Plasma region
  double Lxp = Lx/4.5;
  double Lzp = Lz/4.5;

  // Use double datatype (not int) for n{x,y,z} and topology_{x,y,z} to work
  // with FileIO write to "info.bin" for the Fortran "translate.f90"
  // postprocessing script.  The translate script parses "info.bin" as an
  // unformatted binary stream and so cannot currently cope with mixed double
  // and int datatypes. --ATr,2023nov17
  double nx = 256/4;          // Number of cells in x, y, and z
  double ny = 16;
  double nz = 1024/2;

  double topology_x = 1;    // Number of domains in x, y, and z
  double topology_y = 1;
  double topology_z = 4;

  // --------------------------------------------------------------------------
  // Simulation magnetic geometry and external force fields
  // --------------------------------------------------------------------------

  // Initialize "external" magnetic-field geometry
  // by reading HDF5 file from Pleiades diamagnetic equilibrium solver
  // user should verify that Pleiades data spans the VPIC domain.

  // The Pleaides field init uses vacuum fields on a large domain and
  // diamagnetic fields on a smaller domain.
  // Specify r/z cylinder boundary where the two field solutions should be
  // stitched together.
  double rmax_diamag = 0.50;  // meters  ANVIL-DD from Kai
  double zmax_diamag = 6.0;   // meters

  // --------------------------------------------------------------------------
  // Simulation timestepping, duration, and output parameters
  // --------------------------------------------------------------------------

  // Time step (units of omega_ci^-1)
  double dt = 0.01;

  // Simulation runtime
  double Lt = 1000.0;  // Target simulation duration in omega_ci^-1
  double num_step_user = (int)(Lt / dt);   // ACTUAL simulation runtime in steps

  // Wall-clock runtime
  double quota     = 9.85;          // Wall-clock quota in hours
  double quota_sec = quota*3600;    // Wall-clock quota in seconds

  // Output dumps, restart dumps, and other checks
  int status_interval_user  = 100;      // Stdout/stderr timer reports
  int quota_check_interval  = 100;      // Wall-clock runtime quota check
  int restart_interval      = 100000;     // Simulation restart dumps
  int fields_interval       = 500;     // Fields/hydro dumps
  int particle_interval     = 10000;  // Particle dumps; 0 to disable
  int energies_interval     = 100;      // Scalar diagnostics
  int fields_stride         = 1;        // Stride for field/hydro dumps
  int particle_stride       = 100;      // Stride for particle dumps

  // HDF5 or (old) binary GDA dumps?  Binary is default
  //enable_binary_dump();
  //enable_hdf5_dump();

  // User injection parameters (Kokkos); taken from shock/shock-hyb.cxx
  field_injection_interval = -1;  // -1 to disable
  current_injection_interval = -1;
  particle_injection_interval = 20;
  kokkos_field_injection = false;  // True = inject on device (skip host<->device copy)
  kokkos_current_injection = false;
  kokkos_particle_injection = true;

  // Choose which MPI rank will print timer information.
  // Here, INDEX_TO_RANK finds the rank at domain center which is expected to
  // get a representative balance of particle- vs. field-advance cost.
  int status_timers_rank_user;
  INDEX_TO_RANK(
      int(topology_x/2),
      int(topology_y/2),
      int(topology_z/2),
      status_timers_rank_user
  );

  if (status_timers_rank_user >= nproc()) {
    MESSAGE(("Error: invalid status_timers_rank_user"));
    mp_abort(1000);
  }

  // --------------------------------------------------------------------------
  // Particle sampling and weights
  // --------------------------------------------------------------------------
  double nppc = 400;            // Average number of macro particle per cell per species
  double sort_interval = 20; // Sort interval for particles, type double to match src/vpic/vpic.h
  double Npart  = nppc*nx*ny*nz;          // total macro electrons in box
  Npart = trunc_granular(Npart,nproc());  // Make divisible by number of processors; disabled to avoid int overflow risk --ATr,2025aug11
  double wi = ni/ref_n0*Lx*Ly*Lz/Npart;


  // --------------------------------------------------------------------------
  // Initialize high level simulation parameters and globals
  // --------------------------------------------------------------------------

  // Globals in vpic/vpic.h that are directly initialized by user
  num_step                      = num_step_user;
  status_interval               = status_interval_user;
  status_timers_rank            = status_timers_rank_user;

  // Intervals for various outputs and checks
  global->restart_interval      = restart_interval;
  global->fields_interval       = fields_interval;
  global->particle_interval     = particle_interval;
  global->energies_interval     = energies_interval; 

  // Restart dumps and simulation termination from input deck
  global->rtoggle               = 0;  // not meant to be set by user
  global->quota_check_interval  = quota_check_interval;
  global->quota_sec             = quota_sec;

  // Domain paramters
  global->nx = nx;
  global->ny = ny;
  global->nz = nz;
  global->Lx = Lx;
  global->Ly = Ly;
  global->Lz = Lz;
  global->topology_x = topology_x;
  global->topology_y = topology_y;
  global->topology_z = topology_z;
    
  // Reference values
  global->ref_di = ref_di;
  global->ref_vA0 = ref_vA0;
  global->ref_E0 = ref_E0;
  global->ref_n0 = ref_n0;

  // --------------------------------------------------------------------------
  // Initialize NBI paramters
  // --------------------------------------------------------------------------

  // Parameters for neutral beam injection
  global->n_beams = NUM_BEAMS;

  // First beam injecting from top of domain near x-axis in +z direction at 45 degree angle
  //   beam 1: -x/+z, rotate(v0, phi=0, theta=-3/4*np.pi)
  global->E_beam[0]         = 25.0;  // energy (keV)
  global->P_beam[0]         = 1.0;   // power (MW)
  global->T_beam_para[0]    = 1.0;   // thermal energy (keV)
  global->T_beam_perp[0]    = 1.0;   // thermal energy (keV)
  global->x_beam[0][0]      = Lx / 3.0;   // x-center
  global->x_beam[0][1]      = 0.0;   // y-center
  global->x_beam[0][2]      = 0.0;   // z-center
  global->r_beam[0]         = di * 5.0;    // radius
  global->theta_beam[0]     = -3.0 / 4.0 * M_PI; // polar angle
  global->phi_beam[0]       = 0.0; // azimuthal angle
  global->ppc_beam[0]       = 40; // particles-per-cell
  global->beam_inj_plane[0] = BEAM_INJECTION_PLANE::Z; // injeciton plane

  // Second beam injecting bottom of domain near x-axis in -z direction at 45 degree angle
  //   beam 2: +x/-z, rotate(v0, phi=0, theta=1/4*np.pi)
  global->E_beam[1]         = 25.0;  // energy (keV)
  global->P_beam[1]         = 1.0;   // power (MW)
  global->T_beam_para[1]    = 1.0;   // thermal energy (keV)
  global->T_beam_perp[1]    = 1.0;   // thermal energy (keV)
  global->x_beam[1][0]      = -Lx / 3.0;   // x-center
  global->x_beam[1][1]      = 0.0;   // y-center
  global->x_beam[1][2]      = 0.0;   // z-center
  global->r_beam[1]         = di * 5.0;    // radius
  global->theta_beam[1]     = 1.0 / 4.0 * M_PI; // polar angle
  global->phi_beam[1]       = 0.0; // azimuthal angle
  global->ppc_beam[1]       = 40; // particles-per-cell
  global->beam_inj_plane[1] = BEAM_INJECTION_PLANE::Z; // injeciton plane

  // --------------------------------------------------------------------------
  // Initialize grid
  // --------------------------------------------------------------------------

  // Setup basic grid parameters
  //grid->dx = Lx/nx;
  //grid->dy = Ly/ny;
  //grid->dz = Lz/nz;
  grid->dt = dt;
  grid->cvac = c;
  grid->eps0 = eps0;

  grid->eos_den = hyb_den;
  grid->den_floor_ohm = hyb_den_floor_ohm;
  grid->den_floor_pe  = hyb_den_floor_pe;
  grid->eta = hyb_eta;
  grid->hypereta = hyb_hypereta;

  grid->eos_gamma = hyb_gamma;
  grid->nsub  = hyb_nsub;
  grid->nsm   = hyb_nsm;
  grid->nsmb  = hyb_nsmb;

  // Partition a periodic box among the processors sliced uniformly along x,y,z.
  // Inner radial boundary is offset half a cell off the axis (r0 = 0.5*dr) so the
  // innermost cell face never reaches r=0 -- this bounds the 1/r metric and kills
  // the near-axis high-density/E spikes without needing a floor. The axis BC is
  // applied here; reflecting across r=0.5*dr instead of exactly 0 is a negligible
  // (half-cell) approximation.
  define_periodic_grid( 0.01*Lx, -0.5*Ly, -0.5*Lz,   // Low corner (r0 = dr/2)
                        0.5*Lx,           0.5*Ly,  0.5*Lz,    // High corner
                        nx, ny, nz,                           // Resolution
                        topology_x, topology_y, topology_z);  // Topology
  grid->init_cylindrical_grid();

  // Identify boundary domains
  int ix, iy, iz;
  RANK_TO_INDEX( int(rank()), ix, iy, iz );

  // Override some of the boundary conditions (default is periodic)
  sim_log("Conducting fields on all boundaries");
  // Inner-r is the cylindrical AXIS (r=0): R- and theta- vector field components
  // flip sign across it (theta->theta+pi); z-components and scalars unchanged.
  if ( ix==0 )            set_domain_field_bc( BOUNDARY(-1,0,0), cylindrical_axis_fields );
  if ( ix==topology_x-1 ) set_domain_field_bc( BOUNDARY( 1,0,0), pec_fields );
  // if ( iy==0 )            set_domain_field_bc( BOUNDARY(0,-1,0), pec_fields );
  // if ( iy==topology_y-1 ) set_domain_field_bc( BOUNDARY(0, 1,0), pec_fields );
  if ( iz==0 )            set_domain_field_bc( BOUNDARY(0,0,-1), pec_fields );
  if ( iz==topology_z-1 ) set_domain_field_bc( BOUNDARY(0,0, 1), pec_fields );

  // Absorbing particle boundaries
  sim_log("Absorb particles on all boundaries");
  // Inner-r is the axis: particles crossing r=0 are remapped to theta+pi.
  if ( ix==0 )            set_domain_particle_bc( BOUNDARY(-1,0,0), cylindrical_axis_particles );
  if ( ix==topology_x-1 ) set_domain_particle_bc( BOUNDARY( 1,0,0), absorb_particles );
  // if ( iy==0 )            set_domain_particle_bc( BOUNDARY(0,-1,0), absorb_particles );
  // if ( iy==topology_y-1 ) set_domain_particle_bc( BOUNDARY(0, 1,0), absorb_particles );
  if ( iz==0 )            set_domain_particle_bc( BOUNDARY(0,0,-1), absorb_particles );
  if ( iz==topology_z-1 ) set_domain_particle_bc( BOUNDARY(0,0, 1), absorb_particles );

  // --------------------------------------------------------------------------
  // Initialize materials and field arrays
  // --------------------------------------------------------------------------

  // define_material(...) takes optional arguments to specify dielectric media.
  // Said arguments and materials are not used in Hybrid-VPIC.
  // But, still need >=1 material (vacuum) for simulation code to work.

  material_t * vacuum = define_material( "vacuum", 1 );

  // If you pass NULL to define field array, the standard field array will
  // be used (if damp is not provided, no radiation damping will be used).
  //
  // Prerequisites: grid and all materials must be defined before the field
  // array can be defined.

  define_field_array(NULL);

  // The macro set_region_eta_multipliers( region, tcax, tcay, tcaz )
  // is used to specify spatially-varying numerical dissipation as follows:
  // * tcax multiplies hyper-resistivity in hyb_hypereta.cc
  // * tcay multiplies resistivity in hyb_advance_e.cc
  // * tcaz multiplies all E-field components in hyb_advance_e.cc
  //        and also multiplies hyper-resistivity in hyb_hypereta.cc
  // set_region_eta_multipliers(everywhere, 1., 1., 1.);  // vacuum

  // Set electron temperature on the grid and electron fluid for ionization
  set_region_te(everywhere, hyb_te);

  // Reflecting inner BC
  
  double z_P1 = -0.35*Lz;
  double z_P2 =  0.35*Lz;
  double x_P = Lx/2;
  double R_P = 0.25*Lx;

#define R2P1 ( 0.02*(z-z_P1)*(z-z_P1) + (x-x_P)*(x-x_P) )
#define R2P2 ( 0.02*(z-z_P1)*(z-z_P1) + (x+x_P)*(x+x_P) )
#define R2P3 ( 0.02*(z-z_P2)*(z-z_P2) + (x-x_P)*(x-x_P) )
#define R2P4 ( 0.02*(z-z_P2)*(z-z_P2) + (x+x_P)*(x+x_P) )

# define INSIDE_COIL1 (R2P1 < R_P*R_P )
# define INSIDE_COIL2 (R2P2 < R_P*R_P )
# define INSIDE_COIL3 (R2P3 < R_P*R_P )
# define INSIDE_COIL4 (R2P4 < R_P*R_P )


  set_region_bc( INSIDE_COIL1, reflect_particles, reflect_particles, reflect_particles );
  set_region_bc( INSIDE_COIL2, reflect_particles, reflect_particles, reflect_particles );
  set_region_bc( INSIDE_COIL3, reflect_particles, reflect_particles, reflect_particles );
  set_region_bc( INSIDE_COIL4, reflect_particles, reflect_particles, reflect_particles );


#define R21 ( 0.02*(z-z_P1)*(z-z_P1) + (x-x_P)*(x-x_P) )
#define R22 ( 0.02*(z-z_P1)*(z-z_P1) + (x+x_P)*(x+x_P) )
#define R23 ( 0.02*(z-z_P2)*(z-z_P2) + (x-x_P)*(x-x_P) )
#define R24 ( 0.02*(z-z_P2)*(z-z_P2) + (x+x_P)*(x+x_P) )

#define INSIDE_LAYER1 ( (R21 < 1.5*R_P*R_P) && (R2P1 > R_P*R_P) )
#define INSIDE_LAYER2 ( (R22 < 1.5*R_P*R_P) && (R2P2 > R_P*R_P) )
#define INSIDE_LAYER3 ( (R23 < 1.5*R_P*R_P) && (R2P3 > R_P*R_P) )
#define INSIDE_LAYER4 ( (R24 < 1.5*R_P*R_P) && (R2P4 > R_P*R_P) )

  // Set resistive layer multipliers. 
  // set_region_eta_multipliers(REGION, hyper_eta mult., eta mult., E field mult.)

  set_region_eta_multipliers( INSIDE_LAYER1, 10.0, 1., 1. );
  set_region_eta_multipliers( INSIDE_LAYER2, 10.0, 1., 1. );
  set_region_eta_multipliers( INSIDE_LAYER3, 10.0, 1., 1. );
  set_region_eta_multipliers( INSIDE_LAYER4, 10.0, 1., 1. );
  
  set_region_eta_multipliers( x> 0.45*Lx, 1., 1., 0. );
  set_region_eta_multipliers( x<-0.45*Lx, 1., 1., 0. );
  
  set_region_eta_multipliers( z<-0.485*Lz, 1., 1., 0. );
  set_region_eta_multipliers( z> 0.485*Lz, 1., 1., 0. );
   
  set_region_eta_multipliers( INSIDE_COIL1, 1., 1., 0. );
  set_region_eta_multipliers( INSIDE_COIL2, 1., 1., 0. );
  set_region_eta_multipliers( INSIDE_COIL3, 1., 1., 0. );
  set_region_eta_multipliers( INSIDE_COIL4, 1., 1., 0. );


  // --------------------------------------------------------------------------
  // Load electromagnetic fields
  // --------------------------------------------------------------------------
  // Note: everywhere is a region that encompasses the entire simulation
  // In general, regions are specifed as logical equations (i.e. x>0 && x+y<2)
  //
  // The field macros
  //     set_region_field(...)
  //     set_region_bext(...)
  // take logical expressions phrased using global coordinates (x,y,z)
  // to initialize electric and magnetic fields.
  // --------------------------------------------------------------------------
  
  
#if CYL_FIELDS  
#define RHO() (sqrt(x*x + y*y))
#define ALPHA(zc,rc) ( rc*rc + x*x + y*y + (z-zc)*(z-zc) - 2.0*rc*RHO() )
#define BETA(zc,rc)  ( rc*rc + x*x + y*y + (z-zc)*(z-zc) + 2.0*rc*RHO() )
#define K2(zc,rc)    ( sqrt(1.0 - ALPHA(zc,rc)/BETA(zc,rc)) )
#define ELLIPK(zc,rc)( MATHLIB() comp_ellint_1 (K2(zc,rc)) )
#define ELLIPE(zc,rc)( MATHLIB() comp_ellint_2 (K2(zc,rc)) )


#define BXC(zc,rc,Ic) (2.0*Ic*rc/M_PI*(z-zc)/RHO()*x/RHO()/( 2.0*ALPHA(zc,rc)*sqrt(BETA(zc,rc)) )*( (rc*rc+x*x+y*y+(z-zc)*(z-zc))*ELLIPE(zc,rc) - ALPHA(zc,rc)*ELLIPK(zc,rc) ) ) 
#define BYC(zc,rc,Ic) (2.0*Ic*rc/M_PI*(z-zc)/RHO()*y/RHO()/( 2.0*ALPHA(zc,rc)*sqrt(BETA(zc,rc)) )*( (rc*rc+x*x+y*y+(z-zc)*(z-zc))*ELLIPE(zc,rc) - ALPHA(zc,rc)*ELLIPK(zc,rc) ) )
#define BZC(zc,rc,Ic) (2.0*Ic*rc/M_PI                     /( 2.0*ALPHA(zc,rc)*sqrt(BETA(zc,rc)) )*( (rc*rc+x*x+y*y+(z-zc)*(z-zc))*ELLIPE(zc,rc) + ALPHA(zc,rc)*ELLIPK(zc,rc) ) ) 

  double zcoil1 = 0.3*Lz;
  double zcoil2 = -0.3*Lz;
  double rcoil  = 0.71*Lx;
  
  double B0=0.5;
  double B1 = 0.1;
  double BZ0 = B1/1.13;
  double Icoil = 1.03*(B0-B1);

#define BX ( BXC(zcoil1,rcoil,Icoil) +  BXC(zcoil2,rcoil,Icoil) )
#define BY ( BYC(zcoil1,rcoil,Icoil) +  BYC(zcoil2,rcoil,Icoil) )
#define BZ ( BZC(zcoil1,rcoil,Icoil) +  BZC(zcoil2,rcoil,Icoil) )

  sim_log( "Loading fields" );
  set_region_field_cart( everywhere, 0, 0, 0,       // Electric field
  		                0, 0 ,0 );    // Magnetic field

  // External B: project the physical Cartesian coil field onto the per-cell
  // orthonormal basis (e_1,e_2,e_3), then divide by the scale factors to store
  // true CONTRAVARIANT components cb0_i = (B_cart . e_i)/h_i. This matches the
  // set_region_bext_cart / dump / solver contravariant convention, but is
  // correct at every theta (the macro treats Cartesian x as radial, only valid
  // at theta=0; with ny>1 that is wrong off-axis). Done deck-local so the shared
  // macro (used by pcai/whistler) is untouched.
  {
    const double _c = grid->cvac;
    for( int _k=0; _k<grid->nz+2; _k++ ) {
    for( int _j=0; _j<grid->ny+2; _j++ ) {
    for( int _i=0; _i<grid->nx+2; _i++ ) {
      double x, y, z;
      int _voxel = VOXEL(_i, _j, _k, grid->nx, grid->ny, grid->nz);
      grid->geom().local_to_global_cart(_voxel, 0.0, 0.0, 0.0, x, y, z);
      // Physical Cartesian external field at this cell center
      double bx = ( BX );
      double by = ( BY );
      double bz = ( BZ+BZ0 );
      // Per-cell orthonormal basis vectors (Cartesian components) and scale factors
      int _m = GRID_TO_MESH(_i, _j, _k, grid->nx, grid->ny, grid->nz);
      double e1x = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::e_1_u);
      double e1y = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::e_1_v);
      double e1z = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::e_1_w);
      double e2x = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::e_2_u);
      double e2y = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::e_2_v);
      double e2z = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::e_2_w);
      double e3x = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::e_3_u);
      double e3y = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::e_3_v);
      double e3z = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::e_3_w);
      double h1 = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::h_1);
      double h2 = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::h_2);
      double h3 = grid->k_curvilinear_mesh_h(_m, curv_mesh_var::h_3);
      if(h1 == 0.0) h1 = 1.0;
      if(h2 == 0.0) h2 = 1.0;  // defensive on axis
      if(h3 == 0.0) h3 = 1.0;
      // Store contravariant components (physical projection / h_i)
      field(_i,_j,_k).cbx0 = _c*( bx*e1x + by*e1y + bz*e1z )/h1;
      field(_i,_j,_k).cby0 = _c*( bx*e2x + by*e2y + bz*e2z )/h2;
      field(_i,_j,_k).cbz0 = _c*( bx*e3x + by*e3y + bz*e3z )/h3;
    }}}
  }
  
#else
  double Lcoil1 = 0.6*Lz;
  double Lcoil2 = 0.1*Lz;

  double x1 = 0.55*Lx;
  double z00 = -0.5*Lz;
  double z1 = -0.3*Lz;
  double z2 =  0.3*Lz;
  double z3 = -0.4*Lz;
  double z4 =  0.4*Lz;

  double B0 = 1.0*hyb_b0; // 0.5
  double B1 = 0.2*hyb_b0; // 0.1
  double I1 = 1.45*B1;
  double I2 = 2.0*B0-0.5*B1;
  double I4 = 0.53*B1;

#define BZC(I,xc,zc,L) (-2.0*I/8.0/atan(L/2.0/xc)*(atan((z-zc)/(x-xc))+atan((L-z+zc)/(x-xc))-atan((z-zc)/(x+xc))-atan((L-z+zc)/(x+xc))) )
#define BXC(I,xc,zc,L) ( I/8.0/atan(L/2.0/xc)*log(((z-zc)*(z-zc)+(x-xc)*(x-xc))*((L-z+zc)*(L-z+zc)+(x+xc)*(x+xc))/((((L-z+zc)*(L-z+zc)+(x-xc)*(x-xc)))*(((z-zc)*(z-zc)+(x+xc)*(x+xc))))) )

#define BX ( BXC(I1,x1,z1,Lcoil1) + BXC(I2,x1,z2,Lcoil2) + BXC(I2,x1,z3,Lcoil2) + BXC(I4,x1,z4,Lcoil2) +  BXC(I4,x1,z00,Lcoil2) )
#define BZ ( BZC(I1,x1,z1,Lcoil1) + BZC(I2,x1,z2,Lcoil2) + BZC(I2,x1,z3,Lcoil2) + BZC(I4,x1,z4,Lcoil2) +  BZC(I4,x1,z00,Lcoil2) )


  sim_log( "Loading fields" );
  set_region_field_cart( everywhere, 0, 0, 0,    // Electric field
  		                          0, 0, 0 );  // Magnetic field
  set_region_bext_cart( everywhere, 0, 0, BZ0 ); // External Magnetic field
#endif 

  // --------------------------------------------------------------------------
  // Initialize species
  // --------------------------------------------------------------------------

  double nmax = Npart/nproc();
  double nmovers = 0.1*nmax;
  double nmax_prod = 10.0*Npart/nproc();
  double nmovers_prod = 0.1*nmax_prod;
  double sort_method = 1; // 0=in place, 1=out of place

  if (nmax > 2e9 || nmax_prod > 2e9) {
    MESSAGE(("Error: nmax %.3e overflows int32 in particle buffer init", nmax));
    mp_abort(1000);
  }

  // Define particle mass in units of atomic mass
  double m_D   = 2.014102 * mi;
  double m_T   = 3.016049 * mi;
  double m_n   = 1.008665 * mi;
  double m_p   = 1.007276 * mi;
  double m_He3 = 3.016029 * mi;
  double m_He4 = 4.002603 * mi;

  species_t *D_seed = define_species( "D_seed", ec, m_D, nmax, nmovers, sort_interval, sort_method );
  species_t *D_beam = define_species( "D_beam", ec, m_D, 2, 2, sort_interval, sort_method );

  species_t *T   = define_species("Tritium",   ec, m_T,   2, 2, sort_interval, sort_method);
  species_t *n   = define_species("Neutron",   ec, m_n,   2, 2, sort_interval, sort_method);
  species_t *p   = define_species("Proton",    ec, m_p,   2, 2, sort_interval, sort_method);
  species_t *He3 = define_species("Helium3",   ec, m_He3, 2, 2, sort_interval, sort_method);
  species_t *He4 = define_species("Helium4",   ec, m_He4, 2, 2, sort_interval, sort_method);

  // Create electron fluid species (use in electron impact ionization)
  float me = 1.0/1837.0;
  fluid_species_t * e_fl = define_fluid_species( "e_fluid", -ec, me/mi );
  set_region_fluid( everywhere, "e_fluid", hyb_den, hyb_te, hyb_den*hyb_te );

  // --------------------------------------------------------------------------
  // Setup collisions
  // --------------------------------------------------------------------------

  bool var_wt = true;
  int collision_interval = (int)1*sort_interval; // interval for performing collisions

#if DO_COLLISIONS

  // Charge exchange (particle-particle)
  // A^+N + B^+M -> A^(+N+dq) + B^(+M-dq)  
  int dq_cex = -1;
  DD_cex dd_cex_cs;
  define_collision_op(binary_charge_exchange( "DsDb_cex", D_seed, D_beam, dq_cex, dd_cex_cs, collision_interval, var_wt ));
  define_collision_op(binary_charge_exchange( "DsDs_cex", D_seed, D_seed, dq_cex, dd_cex_cs, collision_interval, var_wt ));
  define_collision_op(binary_charge_exchange( "DbDb_cex", D_beam, D_beam, dq_cex, dd_cex_cs, collision_interval, var_wt ));

  // Ion impact ionization (particle-particle)
  // A^+N + B^+M -> A^+N + B^(+M+1) + e^-
  double dE_ioniz_keV = 13.6 * 1.0e-3;
  double dE_ioniz = dE_ioniz_keV * ERG_PER_KEV / ref_E0;
  DD_ii_ioniz dd_ioniz_cs;
  define_collision_op(binary_ion_impact_ioniz( "DsDb_ioniz", D_seed, D_beam, dE_ioniz, dd_ioniz_cs, collision_interval, var_wt ));
  define_collision_op(binary_ion_impact_ioniz( "DsDs_ioniz", D_seed, D_seed, dE_ioniz, dd_ioniz_cs, collision_interval, var_wt ));
  define_collision_op(binary_ion_impact_ioniz( "DbDb_ioniz", D_beam, D_beam, dE_ioniz, dd_ioniz_cs, collision_interval, var_wt ));

  // Electron impact ionization (particle-electron fluid)
  // A^+N + e^- -> A^(+N+1) + 2e^-
  DD_ei_ioniz dd_eioniz_cs;
  define_collision_op(electron_impact_ionization( "eDb_ioniz", D_beam, e_fl, dE_ioniz, dd_eioniz_cs, collision_interval, field_array ));
  define_collision_op(electron_impact_ionization( "eDs_ioniz", D_seed, e_fl, dE_ioniz, dd_eioniz_cs, collision_interval, field_array ));
  
  // Coulomb collisions: ion-ion
  // cvar0 = q1^2 * q2^2 * lnL / (8 * pi)
  // in SI: cvar0 = (e^4*n0*Lambda)/(8*pi*eps0^2*m_e^2*c^3)
  // Computed with nu normalized by w_pe and simulation units are
  // normalized by w_ci, so multiply nu by (w_pe / w_ci) where w_ci is the 
  // ion cyclotron frequency for singly-ionized oxygen
  double ln_Lambda = 10.0;
  double lnL_8pi = ln_Lambda / (8.0 * M_PI);
  double wpe_wci = ref_wce / ref_wci;
  // double cvar0_sb = (D_seed->q * D_seed->q) * (D_beam->q * D_beam->q) * lnL_8pi * wpe_wci;
  double cvar0 = lnL_8pi * wpe_wci; // multiply cvar by qi^2*qj^2 in code for VARIABLE_CHARGE

  CCModel ccm; // Dummy struct doesn't do anything but needed for constructor...
  define_collision_op(binary_coulomb( "ta_sb", D_seed, D_beam, cvar0, ccm, collision_interval, var_wt ));
  define_collision_op(binary_coulomb( "ta_ss", D_seed, D_seed, cvar0, ccm, collision_interval, var_wt ));
  define_collision_op(binary_coulomb( "ta_bb", D_beam, D_beam, cvar0, ccm, collision_interval, var_wt ));

  // This uses G. Chen's new binary Coulomb collision model
  // define_collision_op(takizuka_abe( "ta_sb", D_seed, D_beam, cvar0_sb, collision_interval, var_wt));
  // define_collision_op(takizuka_abe( "ta_ss", D_seed, D_seed, cvar0_ss, collision_interval, var_wt));
  // define_collision_op(takizuka_abe( "ta_bb", D_beam, D_beam, cvar0_bb, collision_interval, var_wt));

  // Coulomb collisions: electron-ion (still testing)
  // double cvar0_es = lnL_8pi * wpe_wci;
  // double cvar0_eb = lnL_8pi * wpe_wci;
  // define_collision_op(lemons( "coulomb_es", D_seed, e_fl, cvar0_es, collision_interval, field_array ));
  // define_collision_op(lemons( "coulomb_eb", D_beam, e_fl, cvar0_eb, collision_interval, field_array ));


  // --------------------------------------------------------------------------
  // Setup fusion reactions
  // --------------------------------------------------------------------------

#if ENABLE_FUSION_RXS == 1

  int fusion_interval = (int)1*sort_interval; // interval for performing collisions
  float pmult = 1000.0; // production multiplier (increase number of product macroparticles)
  float pmult_DT = std::max(1.0, pmult / 10.0); // DT is more reactive so don't need as big of a multiplier

  // Flag for creating one pair of products at center of mass of reactants or two pairs with 
  // one pair at each of the locations of the reactants (latter improves charge conservation).
  // Default is true.
  bool one_product_pair = true; 

  // D + D -> n + He3 + 3.269e6 eV
  DD_nHe3_cs DD_nHe3_cs_model;
  float dE_DD_nHe3_eV = 3.269e6;
  float dE_DD_nHe3 = dE_DD_nHe3_eV * 1e-3 * ERG_PER_KEV / ref_E0;

  // D + D -> p + T + 4.03e6 eV
  DD_pT_cs DD_pT_cs_model;
  float dE_DD_pT_eV = 4.03e6;
  float dE_DD_pT = dE_DD_pT_eV * 1e-3 * ERG_PER_KEV / ref_E0;

  // D + T -> n + He4 + 17.589e6 eV
  DT_nHe4_cs DT_nHe4_cs_model;
  float dE_DT_nHe4_eV = 17.589e6;
  float dE_DT_nHe4 = dE_DT_nHe4_eV * 1e-3 * ERG_PER_KEV / ref_E0;

#if ENABLE_ANISOTROPIC_FUSION == 1

  dsdOmega DD_dsdOmega_model("DD_dsdomega_coefs.csv");
  dsdOmega DT_dsdOmega_model("DT_dsdomega_coefs.csv");

  // D + D -> n + He3 + 3.269e6 eV
  define_collision_op(binary_fusion( "DsDs_nHe3_fusion", D_seed, D_seed, n, He3, 
    dE_DD_nHe3, pmult, fusion_interval, one_product_pair, DD_nHe3_cs_model, DD_dsdOmega_model ));
  define_collision_op(binary_fusion( "DbDb_nHe3_fusion", D_beam, D_beam, n, He3, 
    dE_DD_nHe3, pmult, fusion_interval, one_product_pair, DD_nHe3_cs_model, DD_dsdOmega_model ));
  define_collision_op(binary_fusion( "DbDs_nHe3_fusion", D_beam, D_seed, n, He3, 
    dE_DD_nHe3, pmult, fusion_interval, one_product_pair, DD_nHe3_cs_model, DD_dsdOmega_model ));

  // D + D -> p + T + 4.03e6 eV
  define_collision_op(binary_fusion( "DsDs_pT_fusion", D_seed, D_seed, p, T, 
    dE_DD_pT, pmult, fusion_interval, one_product_pair, DD_pT_cs_model, DD_dsdOmega_model ));
  define_collision_op(binary_fusion( "DbDb_pT_fusion", D_beam, D_beam, p, T, 
    dE_DD_pT, pmult, fusion_interval, one_product_pair, DD_pT_cs_model, DD_dsdOmega_model ));
  define_collision_op(binary_fusion( "DbDs_pT_fusion", D_beam, D_seed, p, T, 
    dE_DD_pT, pmult, fusion_interval, one_product_pair, DD_pT_cs_model, DD_dsdOmega_model ));

  // D + T -> n + He4 + 17.589e6 eV
  define_collision_op(binary_fusion( "DsT_nHe4_fusion", D_seed, T, n, He4,
    dE_DT_nHe4, pmult, fusion_interval, one_product_pair, DT_nHe4_cs_model, DT_dsdOmega_model ));
  define_collision_op(binary_fusion( "DbT_nHe4_fusion", D_beam, T, n, He4,
    dE_DT_nHe4, pmult, fusion_interval, one_product_pair, DT_nHe4_cs_model, DT_dsdOmega_model ));

#else // isotropic emission (default option, don't pass dsdOmega model)

  // D + D -> n + He3 + 3.269e6 eV
  define_collision_op(binary_fusion( "DsDs_nHe3_fusion", D_seed, D_seed, n, He3, 
    dE_DD_nHe3, pmult, fusion_interval, one_product_pair, DD_nHe3_cs_model ));
  define_collision_op(binary_fusion( "DbDb_nHe3_fusion", D_beam, D_beam, n, He3, 
    dE_DD_nHe3, pmult, fusion_interval, one_product_pair, DD_nHe3_cs_model ));
  define_collision_op(binary_fusion( "DbDs_nHe3_fusion", D_beam, D_seed, n, He3, 
    dE_DD_nHe3, pmult, fusion_interval, one_product_pair, DD_nHe3_cs_model ));

  // D + D -> p + T + 4.03e6 eV
  define_collision_op(binary_fusion( "DsDs_pT_fusion", D_seed, D_seed, p, T, 
    dE_DD_pT, pmult, fusion_interval, one_product_pair, DD_pT_cs_model ));
  define_collision_op(binary_fusion( "DbDb_pT_fusion", D_beam, D_beam, p, T, 
    dE_DD_pT, pmult, fusion_interval, one_product_pair, DD_pT_cs_model ));
  define_collision_op(binary_fusion( "DbDs_pT_fusion", D_beam, D_seed, p, T, 
    dE_DD_pT, pmult, fusion_interval, one_product_pair, DD_pT_cs_model ));

  // D + T -> n + He4 + 17.589e6 eV
  define_collision_op(binary_fusion( "DsT_nHe4_fusion", D_seed, T, n, He4,
    dE_DT_nHe4, pmult, fusion_interval, one_product_pair, DT_nHe4_cs_model ));
  define_collision_op(binary_fusion( "DbT_nHe4_fusion", D_beam, T, n, He4,
    dE_DT_nHe4, pmult, fusion_interval, one_product_pair, DT_nHe4_cs_model ));

#endif // anisotropic emission
#endif // enable fusion reactions

  D_beam->last_indexed = -1;
  D_seed->last_indexed = -1;
  T->last_indexed = -1;
  n->last_indexed = -1;
  p->last_indexed = -1;
  He3->last_indexed = -1;
  He4->last_indexed = -1;
  
#endif //DO_COLLISIONS
  
  // --------------------------------------------------------------------------
  // Load particles
  // --------------------------------------------------------------------------

  sim_log( "Loading particles" );

  // seed_entropy(...) internally ensures that random number generators have a
  // different sequence on each MPI rank
  int rng_seed = 1;  // Random number seed
  seed_entropy( rng_seed );

  // Convert dimensionful user input to dimensionless
  double vth_D = sqrt(Ti_erg / (m_D * ref_E0));
  double w_D = wi;

#if INCLUDE_TRITIUM_FUEL == 1
  // Introduce tritium - weights are set so that the density remains constant
  double vth_T = sqrt(Ti_erg / (m_T * ref_E0));
  double w_T = wi / (1.0 + 1);
  w_D = wi - w_T;

  repeat ( Npart/nproc() ) {
    double x, y, z, ux, uy, uz;
    x = uniform( rng(0), grid->x0, grid->x1 );
    y = uniform( rng(0), grid->y0, grid->y1 );
    z = uniform( rng(0), grid->z0, grid->z1 );

    if ( abs(z) < Lzp && (abs(x) < Lxp) ) {
      ux = normal( rng(0), 0, vth_T );
      uy = normal( rng(0), 0, vth_T );
      uz = normal( rng(0), 0, vth_T );

      inject_particle( T, x, y, z, ux, uy, uz, w_T, 0, 0, T->q );
    }
  }
#endif // include tritium

  repeat ( Npart/nproc() ) {
    double x, y, z, ux, uy, uz;
    x = uniform( rng(0), grid->x0, grid->x1 );
    y = uniform( rng(0), grid->y0, grid->y1 );
    z = uniform( rng(0), grid->z0, grid->z1 );

    if ( abs(z) < Lzp && (abs(x) < Lxp) ) {
      ux = normal( rng(0), 0, vth_D );
      uy = normal( rng(0), 0, vth_D );
      uz = normal( rng(0), 0, vth_D );

      inject_particle( D_seed, x, y, z, ux, uy, uz, w_D*x, 0, 0, D_seed->q );
    }
  }
  sim_log( "Finished loading particles" );

  // --------------------------------------------------------------------------
  // Log diagnostic information about this simulation
  // --------------------------------------------------------------------------

  if (rank()==status_timers_rank_user) {
    MESSAGE(("Reporting timers for rank %d domain x0,x1=(%f,%f) y0,y1=(%f,%f) z0,z1=(%f,%f)",
      rank(), grid->x0, grid->x1, grid->y0, grid->y1, grid->z0, grid->z1
    ));
  }

  sim_log ( "***********************************************" );
  sim_log ( "topology_x = " << topology_x );
  sim_log ( "topology_y = " << topology_y );
  sim_log ( "topology_z = " << topology_z );
  sim_log ( "num_step = " << num_step );
  sim_log ( "Lx/di = " << Lx/di );
  sim_log ( "Ly/di = " << Ly/di );
  sim_log ( "Lz/di = " << Lz/di );
  sim_log ( "nx = " << nx );
  sim_log ( "ny = " << ny );
  sim_log ( "nz = " << nz );
  sim_log ( "nproc = " << nproc () );
  sim_log ( "nppc = " << nppc );
  sim_log ( "di = " << di );
  sim_log ( "dt = " << dt );
  sim_log ( "dx/di = " << Lx/(di*nx) );
  sim_log ( "dy/di = " << Ly/(di*ny) );
  sim_log ( "dz/di = " << Lz/(di*nz) );
  sim_log ( "energies_interval = " << energies_interval );
  sim_log ( "fields_interval = " << fields_interval );
  sim_log ( "particle_interval = " << particle_interval );
  sim_log ( "restart_interval = " << restart_interval );
  sim_log ( "sort_interval = " << sort_interval );
  sim_log ( "collision_interval = " << collision_interval );
  sim_log ( "fields_stride = " << fields_stride );
  sim_log ( "particle_stride = " << particle_stride );
  sim_log ( "***********************************************" );

  // Dump simulation information to file "info"
  if (rank() == 0 ) {
    FILE *fp_info;
    if ( ! (fp_info=fopen("info.dat", "w")) ) ERROR(("Cannot open file."));
    fprintf( fp_info, "***** Simulation parameters *****\n" );
    fprintf( fp_info, "topology_x =           %g\n", topology_x );
    fprintf( fp_info, "topology_y =           %g\n", topology_y );
    fprintf( fp_info, "topology_z =           %g\n", topology_z );
    fprintf( fp_info, "num_step =             %i\n", num_step );
    fprintf( fp_info, "Lx/di =                %g\n", Lx/di );
    fprintf( fp_info, "Ly/di =                %g\n", Ly/di );
    fprintf( fp_info, "Lz/di =                %g\n", Lz/di );
    fprintf( fp_info, "nx =                   %g\n", nx );
    fprintf( fp_info, "ny =                   %g\n", ny );
    fprintf( fp_info, "nz =                   %g\n", nz );
    fprintf( fp_info, "nproc =                %d\n", nproc() );
    fprintf( fp_info, "nppc =                 %g\n", nppc );
    fprintf( fp_info, "di =                   %g\n", di );
    fprintf( fp_info, "dt =                   %g\n", dt );
    fprintf( fp_info, "dx/di =                %g\n", Lx/(di*nx) );
    fprintf( fp_info, "dy/di =                %g\n", Ly/(di*ny) );
    fprintf( fp_info, "dz/di =                %g\n", Lz/(di*nz) );
    fprintf( fp_info, "energies_interval =    %d\n", energies_interval );
    fprintf( fp_info, "fields_interval =      %d\n", fields_interval );
    fprintf( fp_info, "particle_interval =    %d\n", particle_interval );
    fprintf( fp_info, "restart_interval =     %d\n", restart_interval );
    fprintf( fp_info, "sort_interval =        %g\n", sort_interval ); // double not int
    fprintf( fp_info, "collision_interval =   %g\n", collision_interval );
    fprintf( fp_info, "fields_stride =        %d\n", fields_stride );
    fprintf( fp_info, "particle_stride =      %d\n", particle_stride );
    fprintf( fp_info, "*********************************\n" );
    fclose(fp_info);
  }

  // Dump simulation information to file "info.bin" for translate script
  // which also gets passed to IDL plotting routines, so be careful
  // about changing this too much...
  if (rank() == 0 ) {

    // write binary info file
    FileIO fp_info;
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

    fp_info.write(&mi, 1 ); // Translate script expects this number
    //fp_info.write(&vthi, 1 );
    //fp_info.write(&status_interval, 1 );
    fp_info.close();
  }

  // Dump simulation information to file "info.hdf5"
  if (rank() == 0 ) {
    std::vector<int> ivs;
    std::vector<double> dvs;
    std::vector<const char*> inms, dnms;

    // Int parameters
    // ... domain grid and timestepping
    inms.push_back("/num_step"); ivs.push_back( num_step );
    inms.push_back("/nx"); ivs.push_back( nx ); // different from grid->nx !!
    inms.push_back("/ny"); ivs.push_back( ny );
    inms.push_back("/nz"); ivs.push_back( nz );
    inms.push_back("/topology_x"); ivs.push_back( (int)topology_x );
    inms.push_back("/topology_y"); ivs.push_back( (int)topology_y );
    inms.push_back("/topology_z"); ivs.push_back( (int)topology_z );
    // ... dump intervals and stride
    inms.push_back("/fields_interval"); ivs.push_back( fields_interval );
    inms.push_back("/fields_stride"); ivs.push_back( fields_stride );
    inms.push_back("/particle_stride"); ivs.push_back( particle_stride );
    inms.push_back("/energies_interval"); ivs.push_back( energies_interval );
    inms.push_back("/particle_interval"); ivs.push_back( particle_interval );
    inms.push_back("/restart_interval"); ivs.push_back( restart_interval );
    inms.push_back("/sort_interval"); ivs.push_back( sort_interval );

    // Double/float parameters
    // ... reference B field, ion density, mass, charge in Gaussian CGS units
    dnms.push_back("/ref_b0"); dvs.push_back( ref_b0 );
    dnms.push_back("/ref_n0"); dvs.push_back( ref_n0 );
    dnms.push_back("/ref_mi"); dvs.push_back( ref_mi );
    dnms.push_back("/ref_qi"); dvs.push_back( ref_qi );
    // ... domain grid and timestepping
    dnms.push_back("/dt"); dvs.push_back( dt );
    dnms.push_back("/dx"); dvs.push_back( grid->dx );
    dnms.push_back("/dy"); dvs.push_back( grid->dy );
    dnms.push_back("/dz"); dvs.push_back( grid->dz );
    dnms.push_back("/Lx"); dvs.push_back( Lx );
    dnms.push_back("/Ly"); dvs.push_back( Ly );
    dnms.push_back("/Lz"); dvs.push_back( Lz );
    dnms.push_back("/x0"); dvs.push_back( grid->x0 ); // rank=0 gives lower left corner for full domain
    dnms.push_back("/y0"); dvs.push_back( grid->y0 );
    dnms.push_back("/z0"); dvs.push_back( grid->z0 );
    // ... particle and fluid properties
    dnms.push_back("/hyb_te"); dvs.push_back( hyb_te );
    dnms.push_back("/hyb_den"); dvs.push_back( hyb_den );
    dnms.push_back("/hyb_den_floor_ohm"); dvs.push_back( hyb_den_floor_ohm);
    dnms.push_back("/hyb_den_floor_pe"); dvs.push_back( hyb_den_floor_pe);
    dnms.push_back("/hyb_eta"); dvs.push_back( hyb_eta );
    dnms.push_back("/hyb_hypereta"); dvs.push_back( hyb_hypereta );
    dnms.push_back("/hyb_gamma"); dvs.push_back( hyb_gamma );
    // ... particle initialization and inject
    dnms.push_back("/nppc"); dvs.push_back( nppc );
      // ... beam parameters
    dnms.push_back("/E_beam"); dvs.push_back( global->E_beam[0] );
    dnms.push_back("/P_beam"); dvs.push_back( global->P_beam[0] );
    dnms.push_back("/T_beam_para"); dvs.push_back( global->T_beam_para[0] );
    dnms.push_back("/T_beam_perp"); dvs.push_back( global->T_beam_perp[0] );
    dnms.push_back("/x_beam_x"); dvs.push_back( global->x_beam[0][0] );
    dnms.push_back("/x_beam_y"); dvs.push_back( global->x_beam[0][1] );
    dnms.push_back("/x_beam_z"); dvs.push_back( global->x_beam[0][2] );
    dnms.push_back("/r_beam"); dvs.push_back( global->r_beam[0] );
    dnms.push_back("/theta_beam"); dvs.push_back( global->theta_beam[0] );
    dnms.push_back("/phi_beam"); dvs.push_back( global->phi_beam[0] );
    dnms.push_back("/ppc_beam"); dvs.push_back( global->ppc_beam[0] );
    dnms.push_back("/beam_inj_plane"); dvs.push_back( global->beam_inj_plane[0] );

    // Almost done, write out file to disk
    // dump_info("info.hdf5", ivs, inms, dvs, dnms);
  }

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

	global->fieldDumpParams.format = band;
	sim_log ( "Fields output format = band" );

	global->hydroDSeedDumpParams.format = band;
	sim_log ( "Ion species output format = band" );

	global->hydroDBeamDumpParams.format = band;
	sim_log ( "Beam species output format = band" );

  global->hydroNeutronDumpParams.format = band;
	sim_log ( "Neutron species output format = band" );

  global->hydroHelium3DumpParams.format = band;
	sim_log ( "Helium3 species output format = band" );
  
	global->hydroTritiumDumpParams.format = band;
	sim_log ( "Tritium species output format = band" );

  global->hydroProtonDumpParams.format = band;
	sim_log ( "Proton species output format = band" );

  global->hydroHelium4DumpParams.format = band;
	sim_log ( "Helium4 species output format = band" );

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

  // fields output path and stride
  sprintf(global->fieldDumpParams.baseDir, "fields");
  sprintf(global->fieldDumpParams.baseFileName, "fields");
  global->fieldDumpParams.stride_x       = fields_stride;
  global->fieldDumpParams.stride_y       = fields_stride;
  global->fieldDumpParams.stride_z       = fields_stride;
  global->outputParams.push_back(&global->fieldDumpParams);

  // seed-ion hydro output path and stride
  sprintf(global->hydroDSeedDumpParams.baseDir, "hydro");
  sprintf(global->hydroDSeedDumpParams.baseFileName, "Hhydro");
  global->hydroDSeedDumpParams.stride_x      = fields_stride;
  global->hydroDSeedDumpParams.stride_y      = fields_stride;
  global->hydroDSeedDumpParams.stride_z      = fields_stride;
  global->outputParams.push_back(&global->hydroDSeedDumpParams);

  // beam-ion hydro output path and stride
  sprintf(global->hydroDBeamDumpParams.baseDir, "hydro");
  sprintf(global->hydroDBeamDumpParams.baseFileName, "HBhydro");
  global->hydroDBeamDumpParams.stride_x      = fields_stride;
  global->hydroDBeamDumpParams.stride_y      = fields_stride;
  global->hydroDBeamDumpParams.stride_z      = fields_stride;
  global->outputParams.push_back(&global->hydroDBeamDumpParams);

  // Neutron-ion hydro output path and stride
  sprintf(global->hydroNeutronDumpParams.baseDir, "hydro");
  sprintf(global->hydroNeutronDumpParams.baseFileName, "nhydro");
  global->hydroNeutronDumpParams.stride_x      = fields_stride;
  global->hydroNeutronDumpParams.stride_y      = fields_stride;
  global->hydroNeutronDumpParams.stride_z      = fields_stride;
  global->outputParams.push_back(&global->hydroNeutronDumpParams);

  // Helium3-ion hydro output path and stride
  sprintf(global->hydroHelium3DumpParams.baseDir, "hydro");
  sprintf(global->hydroHelium3DumpParams.baseFileName, "He3hydro");
  global->hydroHelium3DumpParams.stride_x      = fields_stride;
  global->hydroHelium3DumpParams.stride_y      = fields_stride;
  global->hydroHelium3DumpParams.stride_z      = fields_stride;
  global->outputParams.push_back(&global->hydroHelium3DumpParams);

  // Tritium-ion hydro output path and stride
  sprintf(global->hydroTritiumDumpParams.baseDir, "hydro");
  sprintf(global->hydroTritiumDumpParams.baseFileName, "Thydro");
  global->hydroTritiumDumpParams.stride_x      = fields_stride;
  global->hydroTritiumDumpParams.stride_y      = fields_stride;
  global->hydroTritiumDumpParams.stride_z      = fields_stride;
  global->outputParams.push_back(&global->hydroTritiumDumpParams);

  // Proton-ion hydro output path and stride
  sprintf(global->hydroProtonDumpParams.baseDir, "hydro");
  sprintf(global->hydroProtonDumpParams.baseFileName, "phydro");
  global->hydroProtonDumpParams.stride_x      = fields_stride;
  global->hydroProtonDumpParams.stride_y      = fields_stride;
  global->hydroProtonDumpParams.stride_z      = fields_stride;
  global->outputParams.push_back(&global->hydroProtonDumpParams);

  // Helium4-ion hydro output path and stride
  sprintf(global->hydroHelium4DumpParams.baseDir, "hydro");
  sprintf(global->hydroHelium4DumpParams.baseFileName, "He4hydro");
  global->hydroHelium4DumpParams.stride_x      = fields_stride;
  global->hydroHelium4DumpParams.stride_y      = fields_stride;
  global->hydroHelium4DumpParams.stride_z      = fields_stride;
  global->outputParams.push_back(&global->hydroHelium4DumpParams);

  // all species raw particle dump stride
  global->particle_stride = particle_stride;

  sim_log ( "Fields x-stride " << global->fieldDumpParams.stride_x );
  sim_log ( "Fields y-stride " << global->fieldDumpParams.stride_y );
  sim_log ( "Fields z-stride " << global->fieldDumpParams.stride_z );

  sim_log ( "DSeed-ion species x-stride " << global->hydroDSeedDumpParams.stride_x );
  sim_log ( "DSeed-ion species y-stride " << global->hydroDSeedDumpParams.stride_y );
  sim_log ( "DSeed-ion species z-stride " << global->hydroDSeedDumpParams.stride_z );

  sim_log ( "DBeam-ion species x-stride " << global->hydroDBeamDumpParams.stride_x );
  sim_log ( "DBeam-ion species y-stride " << global->hydroDBeamDumpParams.stride_y );
  sim_log ( "DBeam-ion species z-stride " << global->hydroDBeamDumpParams.stride_z );

  sim_log ( "Neutron-ion species x-stride " << global->hydroNeutronDumpParams.stride_x );
  sim_log ( "Neutron-ion species y-stride " << global->hydroNeutronDumpParams.stride_y );
  sim_log ( "Neutron-ion species z-stride " << global->hydroNeutronDumpParams.stride_z );

  sim_log ( "Helium3-ion species x-stride " << global->hydroHelium3DumpParams.stride_x );
  sim_log ( "Helium3-ion species y-stride " << global->hydroHelium3DumpParams.stride_y );
  sim_log ( "Helium3-ion species z-stride " << global->hydroHelium3DumpParams.stride_z );

  sim_log ( "Tritium-ion species x-stride " << global->hydroTritiumDumpParams.stride_x );
  sim_log ( "Tritium-ion species y-stride " << global->hydroTritiumDumpParams.stride_y );
  sim_log ( "Tritium-ion species z-stride " << global->hydroTritiumDumpParams.stride_z );

  sim_log ( "Proton-ion species x-stride " << global->hydroProtonDumpParams.stride_x );
  sim_log ( "Proton-ion species y-stride " << global->hydroProtonDumpParams.stride_y );
  sim_log ( "Proton-ion species z-stride " << global->hydroProtonDumpParams.stride_z );

  sim_log ( "Helium4-ion species x-stride " << global->hydroHelium4DumpParams.stride_x );
  sim_log ( "Helium4-ion species y-stride " << global->hydroHelium4DumpParams.stride_y );
  sim_log ( "Helium4-ion species z-stride " << global->hydroHelium4DumpParams.stride_z );

  sim_log ( "All species particle stride " << global->particle_stride);

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

  // Standard VPIC
  output_variables( electric | div_e_err | magnetic | div_b_err |
                    tca      | rhob      | current  | rhof |
                    emat     | nmat      | fmat     | cmat );

  // HybridVPIC
  output_variables(   electric |   magnetic | magnetic0 | ... );

  output_variables( current_density  | charge_density |
                    momentum_density | mass_density   | stress_tensor );
  */

  // These have just the most useful things turned on
  //global->fdParams.output_variables( electric | magnetic | current | div_e_err );
  //global->hedParams.output_variables( current_density | charge_density | mass_density | stress_tensor);
  //global->hHdParams.output_variables( current_density | charge_density | mass_density | stress_tensor);

  global->fieldDumpParams.output_variables( allvars );
  global->hydroDSeedDumpParams.output_variables( allvars );
  global->hydroDBeamDumpParams.output_variables( allvars );
  global->hydroNeutronDumpParams.output_variables( allvars );
  global->hydroHelium3DumpParams.output_variables( allvars );
  global->hydroTritiumDumpParams.output_variables( allvars );
  global->hydroProtonDumpParams.output_variables( allvars );
  global->hydroHelium4DumpParams.output_variables( allvars );

  /*--------------------------------------------------------------------------
   * Convenience functions for simlog output
   *------------------------------------------------------------------------*/

  char varlist[512];
  create_field_list(varlist, global->fieldDumpParams);
  sim_log ( "Fields variable list: " << varlist );

  create_hydro_list(varlist, global->hydroDSeedDumpParams);
  sim_log ( "DSeed-ion species variable list: " << varlist );

  create_hydro_list(varlist, global->hydroDBeamDumpParams);
  sim_log ( "DBeam-ion species variable list: " << varlist );

  create_hydro_list(varlist, global->hydroNeutronDumpParams);
  sim_log ( "Neutron-ion species variable list: " << varlist );

  create_hydro_list(varlist, global->hydroHelium3DumpParams);
  sim_log ( "Helium3-ion species variable list: " << varlist );

  create_hydro_list(varlist, global->hydroTritiumDumpParams);
  sim_log ( "Tritium-ion species variable list: " << varlist );

  create_hydro_list(varlist, global->hydroProtonDumpParams);
  sim_log ( "Proton-ion species variable list: " << varlist );

  create_hydro_list(varlist, global->hydroHelium4DumpParams);
  sim_log ( "Helium4-ion species variable list: " << varlist );

  sim_log("*** Finished with user-specified initialization ***");

  // STANDARD VPIC IS DESCRIBED BELOW, HybridVPIC EVOLUTION LOOP DIFFERS.
  //
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

} // end of begin_initialization



// ============================================================================
// begin_diagnostics
// ============================================================================
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
   * DIRECTORY PATHS SHOULD BE RELATIVE TO
   * THE LOCATION OF THE GLOBAL HEADER!!!
   * (not sure if this is still required... --ATr,2023nov16)
   *------------------------------------------------------------------------*/

  if(step()==0) {
      if (dump_strategy_id == DUMP_STRATEGY_BINARY) {
        dump_mkdir("fields");
        dump_mkdir("hydro");
        dump_mkdir("particle");
      }
      dump_mkdir("restore0");
      dump_mkdir("restore1");  // 1st backup
      dump_mkdir("restore2");  // 2nd backup
      dump_mkdir("rundata");

      global_header("global", global->outputParams);
      //dump_grid("rundata/grid");
      //dump_materials("rundata/materials");
      dump_species("rundata/species");
  }

  /*--------------------------------------------------------------------------
   * Normal output dump
   *------------------------------------------------------------------------*/

  const int restart_interval  = global->restart_interval;
  const int fields_interval   = global->fields_interval;
  const int particle_interval = global->particle_interval;
  const int energies_interval = global->energies_interval;

  // Scalar diagnostics
  if (energies_interval > 0 && step() % energies_interval == 0) {
    int append = step() == 0 ? 0 : 1;
    dump_energies("rundata/energies", append);
    //dump_particle_counts("rundata/particle_counts", append);
  }

  if (fields_interval > 0 && step() % fields_interval == 0) {
    field_dump(global->fieldDumpParams);
    hydro_dump("D_seed", global->hydroDSeedDumpParams);
    hydro_dump("D_beam", global->hydroDBeamDumpParams);

#if ENABLE_FUSION_RXS == 1
    hydro_dump("Tritium", global->hydroTritiumDumpParams);
    hydro_dump("Neutron", global->hydroNeutronDumpParams);
    hydro_dump("Proton",  global->hydroProtonDumpParams);
    hydro_dump("Helium3", global->hydroHelium3DumpParams);
    hydro_dump("Helium4", global->hydroHelium4DumpParams);
#endif
  }

  if (particle_interval > 0 && step() % particle_interval == 0) {

    char subdir[36];

    if (dump_strategy_id == DUMP_STRATEGY_BINARY) {
      sprintf(subdir,"particle/T.%ld",step());
      dump_mkdir(subdir);
    }

    // filename argument here only affects binary dumps
    // HDF5 particle dump filenames are hard-coded in HVPIC-K source
    sprintf(subdir,"particle/T.%ld/D_seed",step());
    dump_particles("D_seed", subdir, global->particle_stride);

    sprintf(subdir,"particle/T.%ld/D_beam",step());
    dump_particles("D_beam", subdir, global->particle_stride);

#if ENABLE_FUSION_RXS == 1
    sprintf(subdir,"particle/T.%ld/Tritium",step());
    dump_particles("Tritium", subdir, global->particle_stride);

    sprintf(subdir,"particle/T.%ld/Neutron",step());
    dump_particles("Neutron", subdir, global->particle_stride);

    sprintf(subdir,"particle/T.%ld/Proton",step());
    dump_particles("Proton", subdir, global->particle_stride);

    sprintf(subdir,"particle/T.%ld/Helium3",step());
    dump_particles("Helium3", subdir, global->particle_stride);

    sprintf(subdir,"particle/T.%ld/Helium4",step());
    dump_particles("Helium4", subdir, global->particle_stride);
#endif
  }

  /*--------------------------------------------------------------------------
   * Restart dump
   *------------------------------------------------------------------------*/

  if (step() > 0 && step() % restart_interval == 0) {
    if (!global->rtoggle) {
      global->rtoggle = 1;
      BEGIN_TURNSTILE(NUM_TURNSTILES){
        checkpt("restore1/restore", 0);
      } END_TURNSTILE;
    } else {
      global->rtoggle = 0;
      BEGIN_TURNSTILE(NUM_TURNSTILES){
        checkpt("restore2/restore", 0);
      } END_TURNSTILE;
    }
  }

  /*--------------------------------------------------------------------------
   * Abort simulation when wall clock time exceeds global->quota_sec
   *------------------------------------------------------------------------*/

  const int quota_check_interval  = global->quota_check_interval;
  const double quota_sec          = global->quota_sec;

  // Synchronize the abort across processors by using mp_elapsed(), which is
  // guaranteed to return the same value for all ranks.
  // Check infrequently to avoid costly ALL_REDUCE in mp_elapsed().

  if (step() > 0 && quota_check_interval > 0 && step() % quota_check_interval == 0) {
    if (uptime() > quota_sec) {
      sim_log("Allowed runtime exceeded for this job.  Terminating....\n");

      BEGIN_TURNSTILE(NUM_TURNSTILES){
        checkpt("restore0/restore",0);
      } END_TURNSTILE;

      sim_log("Restart dump completed.");
      exit(0); // Exit or abort?
    }
  }

} // end of begin_diagnostics


// ============================================================================
// begin_current_injection
// ============================================================================
begin_current_injection {
  // No current injection for this simulation
}


// ============================================================================
// begin_field_injection
// ============================================================================
begin_field_injection {
  // No field injection for this simulation
}


// ============================================================================
// begin_particle_collisions
// ============================================================================

begin_particle_collisions {
  // No particle collisions for this simulation
  // Can be provided by other include files
}


// ============================================================================
// begin_particle_injection
// ============================================================================


begin_particle_injection {
#if INJECT_BEAMS

  const double dt = grid->dt;
  const double hx = grid->dx;
  const double hy = grid->dy;
  const double hz = grid->dz;

  const int n_beams = global->n_beams;
  const int age = 0;

  // Beam injection user-parameters in keV and MW (be careful normalizing)
  const double mi_kg = 1.67262193e-27;
  const double c_ms  = 2.99792458e8;
  const double e_C   = 1.60217663e-19;

  const double n0 = global->ref_n0 * 1.0e6; // m^-3
  const double l0 = global->ref_di * 0.01;  // m
  const double v0 = global->ref_vA0 * 0.01; // m/s
  const double area0 = l0 * l0;
  const double vol0 = l0 * l0 * l0;

  // species_t * species = global->D_beam; // all beams populate same species  
  species_t * species = find_species_id(1, species_list);  
  double charge = 0.0; // neutral beam

  // Initialize beam variables (determine range of cells and particles weights)
  static int initted = 0;
  if ( !initted ) {
    initted = 1;

    for (int i_beam = 0; i_beam < n_beams; i_beam++) {
      double x = global->x_beam[i_beam][0];  // x-center
      double y = global->x_beam[i_beam][1];  // y-center
      double z = global->x_beam[i_beam][2];  // z-center
      double r = global->r_beam[i_beam];     // radius
      BEAM_INJECTION_PLANE plane = global->beam_inj_plane[i_beam];

      // Find beam center
      int ix_center = int((x / global->Lx + 0.5) * global->nx - 1);
      int iy_center = int((y / global->Ly + 0.5) * global->ny - 1);
      int iz_center = int((z / global->Lz + 0.5) * global->nz - 1);

      // Initialize range of cell indices
      global->icell_beam[i_beam][0] = ix_center; // ix0
      global->icell_beam[i_beam][1] = ix_center; // ix1
      global->icell_beam[i_beam][2] = iy_center; // iy0
      global->icell_beam[i_beam][4] = iy_center; // iy1
      global->icell_beam[i_beam][4] = iz_center; // iz0
      global->icell_beam[i_beam][5] = iz_center; // iz1

      switch(plane) {
        case BEAM_INJECTION_PLANE::X:
        {
          global->icell_beam[i_beam][0] = int(((x-r) / global->Lx + 0.5) * global->nx - 1); // ix0
          global->icell_beam[i_beam][1] = int(((x+r) / global->Lx + 0.5) * global->nx - 1); // ix1
          break;
        }
        case BEAM_INJECTION_PLANE::Y:
        {
          global->icell_beam[i_beam][2] = int(((y-r) / global->Ly + 0.5) * global->ny - 1); // iy0
          global->icell_beam[i_beam][3] = int(((y+r) / global->Ly + 0.5) * global->ny - 1); // iy1
          break;
        }
        case BEAM_INJECTION_PLANE::Z:
        {
          global->icell_beam[i_beam][4] = int(((z-r) / global->Lz + 0.5) * global->nz - 1); // iz0
          global->icell_beam[i_beam][5] = int(((z+r) / global->Lz + 0.5) * global->nz - 1); // iz1
          break;
        }
      } // endswitch

      // Number of cells for injection region
      double nx_inj = global->icell_beam[i_beam][1] - global->icell_beam[i_beam][0] + 1;
      double ny_inj = global->icell_beam[i_beam][3] - global->icell_beam[i_beam][2] + 1;
      double nz_inj = global->icell_beam[i_beam][5] - global->icell_beam[i_beam][4] + 1;
      
      // Determine beam velocity based on direction and energy
      // (E_beam in keV, need v_mag in vpic units)
      double E_beam_J = global->E_beam[i_beam] * e_C * 1.0e3; // convert from keV to J
      double m_sp_kg = species->m * mi_kg;
      double v_beam_ms = c_ms * std::sqrt(1.0 - 1.0 / SQR(1.0 + E_beam_J / (m_sp_kg * SQR(c_ms))));
      global->v_mag[i_beam] = v_beam_ms / v0;

      // Determine thermal velocity parallel and perpendicular to beam
      double T_J = global->T_beam_para[i_beam] * e_C * 1.0e3; // convert from keV to J
      double vth_ms = sqrt(3.0 * T_J / m_sp_kg);
      global->v_thermal[i_beam][0] = vth_ms / v0;

      T_J = global->T_beam_perp[i_beam] * e_C * 1.0e3; // convert from keV to J
      vth_ms = sqrt(3.0 * T_J / m_sp_kg);
      global->v_thermal[i_beam][1] = vth_ms / v0;

      double area_inject, dx_cell; 

      switch(plane) {
        case BEAM_INJECTION_PLANE::X:
        {
          area_inject = (nz_inj * hz) * (ny_inj * hy);
          dx_cell = hx;
          break;
        }
        case BEAM_INJECTION_PLANE::Y:
        {
          area_inject = (nx_inj * hx) * (nz_inj * hz);
          dx_cell = hy;
          break;
        }
        case BEAM_INJECTION_PLANE::Z:
        {
          area_inject = (nx_inj * hx) * (ny_inj * hy);
          dx_cell = hz;
          break;
        }
      } // endswitch

      // Determine particle weights based on dt, power, and energy
      double P_beam_W = global->P_beam[i_beam] * 1.0e6;

      // double tau_inject_s = particle_injection_interval * grid->dt / w_ci_s; // injection interval, s

      double n_cells = nx_inj * ny_inj * nz_inj;
      double n_particles = n_cells * global->ppc_beam[i_beam];

      double vol_inject_m3 = n_cells * hx * hy * hz * vol0;
      double area_inject_m2 = area_inject * area0;
      double n_beam_m3 = P_beam_W / (E_beam_J * v_beam_ms * area_inject_m2);

      // Modify weight to account for time it takes for beam to cross cell
      double w_adjust_time = 2.0 * global->v_mag[i_beam] * particle_injection_interval * grid->dt / dx_cell;

      // Modify weight to account for injecting in 2D rather than 3D
      // // area_1D = dy*dz, area_3D = pi*r^2, ratio = dy*dz / pi*r^2
      // // double w_adjust_area = hy * hz / (M_PI * r * r);
      // area_2D = 2*r*dy, area_3D = pi*r^2, ratio = 2*dy / np*r
      double w_adjust_area = 2.0 * hy / (M_PI * r);

      global->w_beam[i_beam] = w_adjust_time * w_adjust_area * \
        (n_beam_m3 / n0) * (vol_inject_m3 / vol0) / n_particles;
      
      // Calculate rotation matrix, rotate x_hat by polar angle theta and azimuthal angle phi
      double theta = global->theta_beam[i_beam];
      double   phi = global->phi_beam[i_beam];

      global->rotation_matrix[i_beam][0][0] = cos(phi) * cos(theta);
      global->rotation_matrix[i_beam][0][1] = -sin(phi);
      global->rotation_matrix[i_beam][0][2] = cos(phi) * sin(theta);

      global->rotation_matrix[i_beam][1][0] = sin(phi) * cos(theta);
      global->rotation_matrix[i_beam][1][1] = cos(phi);
      global->rotation_matrix[i_beam][1][2] = sin(phi) * sin(theta);

      global->rotation_matrix[i_beam][2][0] = -sin(theta);
      global->rotation_matrix[i_beam][2][1] = 0.0;
      global->rotation_matrix[i_beam][2][2] = cos(theta);

      // Determine if rank contains injection region
      // Initialize rank as not containing injection region
      global->contains_injection_region[i_beam] = false;

      // Loop through global zones and test if within rank
      int ix0 = global->icell_beam[i_beam][0]; 
      int ix1 = global->icell_beam[i_beam][1];
      int iy0 = global->icell_beam[i_beam][2]; 
      int iy1 = global->icell_beam[i_beam][3];
      int iz0 = global->icell_beam[i_beam][4]; 
      int iz1 = global->icell_beam[i_beam][5];

      int ix0l = (int)1e8, iy0l = (int)1e8, iz0l = (int)1e8; // initialize to large number
      int ix1l = -1, iy1l = -1, iz1l = -1; // initialize to small number

      for (int iz=iz0; iz<=iz1; iz++) {
        for (int iy=iy0; iy<=iy1; iy++) {
          for (int ix=ix0; ix<=ix1; ix++) {

            // Map zone to rank index
            int irx = int(ix / grid->nx);
            int iry = int(iy / grid->ny);
            int irz = int(iz / grid->nz);
            int target_rank = irx + global->topology_x*(iry + global->topology_y*irz);

            if (int(rank()) == target_rank) {
              // Mark as containing injection zone
              global->contains_injection_region[i_beam] = true;

              // Map global indices of beam injection region to local indices of rank
              ix0l = std::min(ix % grid->nx, ix0l);
              ix1l = std::max(ix % grid->nx, ix1l);
              
              iy0l = std::min(iy % grid->ny, iy0l);
              iy1l = std::max(iy % grid->ny, iy1l);
              
              iz0l = std::min(iz % grid->nz, iz0l);
              iz1l = std::max(iz % grid->nz, iz1l);
            } // endif(rank=target_rank)

          } // endfor(ix)
        } // endfor(iy)
      } // endfor(iz)

      // Assign to global variable
      global->icell_beam[i_beam][0] = ix0l; 
      global->icell_beam[i_beam][1] = ix1l;
      global->icell_beam[i_beam][2] = iy0l; 
      global->icell_beam[i_beam][3] = iy1l;
      global->icell_beam[i_beam][4] = iz0l; 
      global->icell_beam[i_beam][5] = iz1l;
      
    } // endfor(i_beam)
  } // endif(!initted)

  // Loop over beams and inject particles
  for (int i_beam = 0; i_beam < n_beams; i_beam++) {

    if (!global->contains_injection_region[i_beam]) { continue; }

    int ix0 = global->icell_beam[i_beam][0]; 
    int ix1 = global->icell_beam[i_beam][1];
    int iy0 = global->icell_beam[i_beam][2]; 
    int iy1 = global->icell_beam[i_beam][3];
    int iz0 = global->icell_beam[i_beam][4]; 
    int iz1 = global->icell_beam[i_beam][5];

    int ppc = global->ppc_beam[i_beam];
    double weight = global->w_beam[i_beam];
    double v_mag = global->v_mag[i_beam];
    double *v_th = global->v_thermal[i_beam];
    double rm[3][3];

    std::copy(&global->rotation_matrix[i_beam][0][0],
              &global->rotation_matrix[i_beam][0][0] + 9,
              &rm[0][0]);

    double x, y, z, vx, vy, vz, vx_th, vy_th, vz_th;

    for (int iz=iz0; iz<=iz1; iz++) {
      for (int iy=iy0; iy<=iy1; iy++) {
        for (int ix=ix0; ix<=ix1; ix++) {

          repeat(ppc) {
            // Sample location within cell
            x = grid->x0 + hx * (ix + uniform(rng(0), 0 , 1));
            y = grid->y0 + hy * (iy + uniform(rng(0), 0 , 1));
            z = grid->z0 + hz * (iz + uniform(rng(0), 0 , 1));
                
            // Sample thermal velocity and add to beam velocity
            vx_th = normal(rng(0), 0, v_th[0]); // v_th from T_para
            vy_th = normal(rng(0), 0, v_th[1]); // v_th from T_perp
            vz_th = normal(rng(0), 0, v_th[1]); // v_th from T_perp

            // Rotate velocity
            vx = rm[0][0] * (v_mag + vx_th) + rm[0][1] * vy_th + rm[0][2] * vz_th;
            vy = rm[1][0] * (v_mag + vx_th) + rm[1][1] * vy_th + rm[1][2] * vz_th;
            vz = rm[2][0] * (v_mag + vx_th) + rm[2][1] * vy_th + rm[2][2] * vz_th;

            inject_particle_r(species, x, y, z, vx, vy, vz, weight, age, charge);
          } // end repeat

        } // endfor(ix)
      } // endfor(iy)
    } // endfor(iz)

  } // endfor(i_beam)
#endif
} // end particle injection
