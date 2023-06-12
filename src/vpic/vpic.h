/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Heavily revised and extended from earlier V4PIC versions
 *
 * snell - revised to add new dumps, 20080310
 *
 */

#ifndef vpic_h
#define vpic_h

#include <vector>
#include <cmath>

#include "../boundary/boundary.h"
#include "../collision/collision.h"
#include "../emitter/emitter.h"
// FIXME: INCLUDES ONCE ALL IS CLEANED UP
#include "../util/io/FileIO.h"
#include "../util/bitfield.h"
#include "../util/checksum.h"
#include "../util/system.h"

#ifndef USER_GLOBAL_SIZE
#define USER_GLOBAL_SIZE 16384
#endif

#ifndef NVARHISMX
#define NVARHISMX 250
#endif
//  #include "dumpvars.h"

typedef FileIO FILETYPE;

const uint32_t electric		(1<<0 | 1<<1 | 1<<2);
const uint32_t div_e_err	(1<<3);
const uint32_t magnetic		(1<<4 | 1<<5 | 1<<6);
const uint32_t div_b_err	(1<<7);
const uint32_t tca			(1<<8 | 1<<9 | 1<<10);
const uint32_t rhob			(1<<11);
const uint32_t current		(1<<12 | 1<<13 | 1<<14);
const uint32_t rhof			(1<<15);
const uint32_t emat			(1<<16 | 1<<17 | 1<<18);
const uint32_t nmat			(1<<19);
const uint32_t fmat			(1<<20 | 1<<21 | 1<<22);
const uint32_t cmat			(1<<23);

const size_t total_field_variables(24);
const size_t total_field_groups(12); // this counts vectors, tensors etc...
// These bits will be tested to determine which variables to output
const size_t field_indeces[12] = { 0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23 };

struct FieldInfo {
	char name[128];
	char degree[128];
	char elements[128];
	char type[128];
	size_t size;
}; // struct FieldInfo

const uint32_t current_density	(1<<0 | 1<<1 | 1<<2);
const uint32_t charge_density	(1<<3);
const uint32_t momentum_density	(1<<4 | 1<<5 | 1<<6);
const uint32_t ke_density		(1<<7);
const uint32_t stress_tensor	(1<<8 | 1<<9 | 1<<10 | 1<<11 | 1<<12 | 1<<13);
/* May want to use these instead
const uint32_t stress_diagonal 		(1<<8 | 1<<9 | 1<<10);
const uint32_t stress_offdiagonal	(1<<11 | 1<<12 | 1<<13);
*/

enum DumpVar {
  GlobalPos = (1<<0 | 1<<1 | 1<<2),
  Efield = (1<<3 | 1<<4 | 1<<5),
  Bfield = (1<<6 | 1<<7 | 1<<8),
  CurrentDensity = (1<<9 | 1<<10 | 1<<11),
  ChargeDensity = (1<<12),
  MomentumDensity = (1<<13 | 1<<14 | 1<<15),
  KEDensity = (1<<16),
  StressTensor = (1<<17 | 1<<18 | 1<<19 | 1<<20 | 1<<21 | 1<<22),
  All = 0xFFFFFFFF
};

const size_t total_hydro_variables(14);
const size_t total_hydro_groups(5); // this counts vectors, tensors etc...
// These bits will be tested to determine which variables to output
const size_t hydro_indeces[5] = { 0, 3, 4, 7, 8 };

struct HydroInfo {
	char name[128];
	char degree[128];
	char elements[128];
	char type[128];
	size_t size;
}; // struct FieldInfo

/*----------------------------------------------------------------------------
 * DumpFormat Enumeration
----------------------------------------------------------------------------*/
enum DumpFormat {
  band = 0,
  band_interleave = 1
}; // enum DumpFormat

/*----------------------------------------------------------------------------
 * DumpParameters Struct
----------------------------------------------------------------------------*/
struct DumpParameters {

  void output_variables(uint32_t mask) {
    output_vars.set(mask);
  } // output_variables

  BitField output_vars;

  size_t stride_x;
  size_t stride_y;
  size_t stride_z;

  DumpFormat format;

  char name[128];
  char baseDir[128];
  char baseFileName[128];

}; // struct DumpParameters

class vpic_simulation {
public:
  vpic_simulation();
  ~vpic_simulation();
  void initialize( int argc, char **argv );
  void modify( const char *fname );
  int advance( void );
  void finalize( void );
  void print_run_details( void );

  // Directly initialized by user

  int verbose;              // Should system be verbose
  int num_step;             // Number of steps to take
  int num_comm_round;       // Num comm round
  int status_interval;      // How often to print status messages
  int clean_div_e_interval; // How often to clean div e
  int num_div_e_round;      // How many clean div e rounds per div e interval
  int clean_div_b_interval; // How often to clean div b
  int num_div_b_round;      // How many clean div b rounds per div b interval
  int sync_shared_interval; // How often to synchronize shared faces

  // Track whether injection functions necessary
  int field_injection_interval = -1;
  int current_injection_interval = -1;
  int particle_injection_interval = -1;
  // Track whether injection functions are ported to Kokkos
  bool kokkos_field_injection = false;
  bool kokkos_current_injection = false;
  bool kokkos_particle_injection = false;

  // FIXME: THESE INTERVALS SHOULDN'T BE PART OF vpic_simulation
  // THE BIG LIST FOLLOWING IT SHOULD BE CLEANED UP TOO

  double quota;
  int checkpt_interval;
  int hydro_interval;
  int field_interval;
  int particle_interval;

  size_t nxout, nyout, nzout;
  size_t px, py, pz;
  float dxout, dyout, dzout;

  int ndfld;
  int ndhyd;
  int ndpar;
  int ndhis;
  int ndgrd;
  int head_option;
  int istride;
  int jstride;
  int kstride;
  int stride_option;
  int pstride;
  int nprobe;
  int ijkprobe[NVARHISMX][4];
  float xyzprobe[NVARHISMX][3];
  int block_dump;
  int stepdigit;
  int rankdigit;
  int ifenergies;

  // Helper initialized by user

  /* There are enough synchronous and local random number generators
     to permit the host thread plus all the pipeline threads for one
     dispatcher to simultaneously produce both synchronous and local
     random numbers.  Keeping the synchronous generators in sync is
     the generator users responsibility. */

  rng_pool_t           * entropy;            // Local entropy pool
  rng_pool_t           * sync_entropy;       // Synchronous entropy pool
  grid_t               * grid;               // define_*_grid et al
  material_t           * material_list;      // define_material
  field_array_t        * field_array;        // define_field_array
  interpolator_array_t * interpolator_array; // define_interpolator_array
  hydro_array_t        * hydro_array;        // define_hydro_array
  species_t            * species_list;       // define_species /
                                             // species helpers
  species_t            * tracers_list;       // define_tracers /
                                             // species helpers
  particle_bc_t        * particle_bc_list;   // define_particle_bc /
                                             // boundary helpers
  emitter_t            * emitter_list;       // define_emitter /
                                             // emitter helpers
  collision_op_t       * collision_op_list;  // collision helpers

  // User defined checkpt preserved variables
  // Note: user_global is aliased with user_global_t (see deck_wrapper.cxx)

  char user_global[USER_GLOBAL_SIZE];

  /*----------------------------------------------------------------------------
   * Diagnostics
   ---------------------------------------------------------------------------*/
  double poynting_flux(double e0);

  /*----------------------------------------------------------------------------
   * Check Sums
   ---------------------------------------------------------------------------*/
#if defined(ENABLE_OPENSSL)
  void output_checksum_fields();
  void checksum_fields(CheckSum & cs);
  void output_checksum_species(const char * species);
  void checksum_species(const char * species, CheckSum & cs);
#endif // ENABLE_OPENSSL

  void print_available_ram() {
    SystemRAM::print_available();
  } // print_available_ram

  ///////////////
  // Dump helpers

  int dump_mkdir(const char * dname);
  int dump_cwd(char * dname, size_t size);

  // Text dumps
  void dump_energies( const char *fname, int append = 1 );
  void dump_materials( const char *fname );
  void dump_species( const char *fname );
  void dump_tracers_buffered_csv( const char *sp_name, uint32_t dump_vars, 
                                  const char *fbase, int append = 1,
                                  int fname_tag = 1 );
  void dump_tracers_csv( const char *sp_name, uint32_t dump_vars, const char *fbase, int append = 1,
                                  int fname_tag = 1 );
  void dump_particles_csv( const char *sp_name,
                           uint32_t dump_vars,
                           const char *fbase,
                           const int append=1,
                           int ftag=1 );

  // Binary dumps
  void dump_grid( const char *fbase );
  void dump_fields( const char *fbase, int fname_tag = 1 );
  void dump_hydro( const char *sp_name, const char *fbase,
                   int fname_tag = 1 );
  void dump_particles( const char *sp_name, const char *fbase,
                       int fname_tag = 1 );

  // HDF5 dumps
  void dump_tracers_hdf5(const char* sp_name, const uint32_t dump_vars, const char*fbase, int append=1);
  void dump_tracers_buffered_hdf5( const char *sp_name, uint32_t dump_vars, const char *fbase, const int append=1 );

  // convenience functions for simlog output
  void create_field_list(char * strlist, DumpParameters & dumpParams);
  void create_hydro_list(char * strlist, DumpParameters & dumpParams);

  void print_hashed_comment(FileIO & fileIO, const char * comment);
  void global_header(const char * base,
  	std::vector<DumpParameters *> dumpParams);

  void field_header(const char * fbase, DumpParameters & dumpParams);
  void hydro_header(const char * speciesname, const char * hbase,
    DumpParameters & dumpParams);

  void field_dump(DumpParameters & dumpParams);
  void hydro_dump(const char * speciesname, DumpParameters & dumpParams);

  ///////////////////
  // Useful accessors

  inline int
  rank() { return world_rank; }

  inline int
  nproc() { return world_size; }

  inline void
  barrier() { mp_barrier(); }

  inline double
  time() {
    return grid->t0 + (double)grid->dt*(double)grid->step;
  }

  inline int64_t &
  step() {
   return grid->step;
  }

  inline field_t &
  field( const int v ) {
    return field_array->f[ v ];
  }

  inline int
  voxel( const int ix, const int iy, const int iz ) {
    return ix + grid->sy*iy + grid->sz*iz;
  }

  inline int
  voxel( const int ix, const int iy, const int iz, const int sy, const int sz ) {
    return ix + sy*iy +sz*iz;
  }

  inline field_t &
  field( const int ix, const int iy, const int iz ) {
    return field_array->f[ voxel(ix,iy,iz) ];
  }

  inline k_field_t& get_field() {
      return field_array->k_f_d;
  }

  inline float& k_field(const int ix, const int iy, const int iz, field_var::f_v member) {
      return field_array->k_f_d(voxel(ix,iy,iz), member);
  }

  inline interpolator_t &
  interpolator( const int v ) {
    return interpolator_array->i[ v ];
  }

  inline interpolator_t &
  interpolator( const int ix, const int iy, const int iz ) {
    return interpolator_array->i[ voxel(ix,iy,iz) ];
  }

  inline hydro_t &
  hydro( const int v ) {
    return hydro_array->h[ v ];
  }

  inline hydro_t &
  hydro( const int ix, const int iy, const int iz ) {
    return hydro_array->h[ voxel(ix,iy,iz) ];
  }

  inline rng_t *
  rng( const int n ) {
    return entropy->rng[n];
  }

  inline rng_t *
  sync_rng( const int n ) {
    return sync_entropy->rng[n];
  }

  ///////////////
  // Grid helpers

  inline void
  define_units( float cvac,
                float eps0 ) {
    grid->cvac = cvac;
    grid->eps0 = eps0;
  }

  inline void
  define_timestep( float dt, double t0 = 0, int64_t step = 0 ) {
    grid->t0   = t0;
    grid->dt   = (float)dt;
    grid->step = step;
  }

  // The below functions automatically create partition simple grids with
  // simple boundary conditions on the edges.

  inline void
  define_periodic_grid( double xl,  double yl,  double zl,
                        double xh,  double yh,  double zh,
                        double gnx, double gny, double gnz,
                        double gpx, double gpy, double gpz )
  {
      px = size_t(gpx); py = size_t(gpy); pz = size_t(gpz);
      partition_periodic_box( grid, xl, yl, zl, xh, yh, zh,
              (int)gnx, (int)gny, (int)gnz,
              (int)gpx, (int)gpy, (int)gpz );
  }

  inline void
  define_absorbing_grid( double xl,  double yl,  double zl,
                         double xh,  double yh,  double zh,
                         double gnx, double gny, double gnz,
                         double gpx, double gpy, double gpz, int pbc )
  {
      px = size_t(gpx); py = size_t(gpy); pz = size_t(gpz);
      partition_absorbing_box( grid, xl, yl, zl, xh, yh, zh,
              (int)gnx, (int)gny, (int)gnz,
              (int)gpx, (int)gpy, (int)gpz,
              pbc );
  }

  inline void
  define_reflecting_grid( double xl,  double yl,  double zl,
                          double xh,  double yh,  double zh,
                          double gnx, double gny, double gnz,
                          double gpx, double gpy, double gpz )
  {
      px = size_t(gpx); py = size_t(gpy); pz = size_t(gpz);
      partition_metal_box( grid, xl, yl, zl, xh, yh, zh,
              (int)gnx, (int)gny, (int)gnz,
              (int)gpx, (int)gpy, (int)gpz );
  }

  // The below macros allow custom domains to be created

  // Creates a particle reflecting metal box in the local domain
  inline void
  size_domain( double lnx, double lny, double lnz ) {
    size_grid(grid,(int)lnx,(int)lny,(int)lnz);
  }

  // Attaches a local domain boundary to another domain
  inline void join_domain( int boundary, double rank ) {
    join_grid( grid, boundary, (int)rank );
  }

  // Sets the field boundary condition of a local domain boundary
  inline void set_domain_field_bc( int boundary, int fbc ) {
    set_fbc( grid, boundary, fbc );
  }

  // Sets the particle boundary condition of a local domain boundary
  inline void set_domain_particle_bc( int boundary, int pbc ) {
    set_pbc( grid, boundary, pbc );
  }

  ///////////////////
  // Material helpers

  inline material_t *
  define_material( const char * name,
                   double eps,
                   double mu = 1,
                   double sigma = 0,
                   double zeta = 0 ) {
    return append_material( material( name,
                                      eps,   eps,   eps,
                                      mu,    mu,    mu,
                                      sigma, sigma, sigma,
                                      zeta,  zeta,  zeta ), &material_list );
  }

  inline material_t *
  define_material( const char * name,
                   double epsx,        double epsy,       double epsz,
                   double mux,         double muy,        double muz,
                   double sigmax,      double sigmay,     double sigmaz,
		   double zetax = 0 ,  double zetay = 0,  double zetaz = 0 ) {
    return append_material( material( name,
                                      epsx,   epsy,   epsz,
                                      mux,    muy,    muz,
                                      sigmax, sigmay, sigmaz,
                                      zetax,  zetay,  zetaz ), &material_list );
  }

  inline material_t *
  lookup_material( const char * name ) {
    return find_material_name( name, material_list );
  }

  inline material_t *
  lookup_material( material_id id ) {
    return find_material_id( id, material_list );
  }

  //////////////////////
  // Field array helpers

  // If fa is provided, define_field_advance will use it (and take ownership
  // of the it).  Otherwise the standard field array will be used with the
  // optionally provided radition damping parameter.

  inline void
  define_field_array( field_array_t * fa = NULL, double damp = 0 ) {
    int nx1 = grid->nx + 1, ny1 = grid->ny+1, nz1 = grid->nz+1;

    if( grid->nx<1 || grid->ny<1 || grid->nz<1 )
      ERROR(( "Define your grid before defining the field array" ));
    if( !material_list )
      ERROR(( "Define your materials before defining the field array" ));

    field_array        = fa ? fa :
                         new_standard_field_array( grid, material_list, damp );
    interpolator_array = new_interpolator_array( grid );
    hydro_array        = new_hydro_array( grid );

    // Pre-size communications buffers. This is done to get most memory
    // allocation over with before the simulation starts running

    mp_size_recv_buffer(grid->mp,BOUNDARY(-1, 0, 0),ny1*nz1*sizeof(hydro_t));
    mp_size_recv_buffer(grid->mp,BOUNDARY( 1, 0, 0),ny1*nz1*sizeof(hydro_t));
    mp_size_recv_buffer(grid->mp,BOUNDARY( 0,-1, 0),nz1*nx1*sizeof(hydro_t));
    mp_size_recv_buffer(grid->mp,BOUNDARY( 0, 1, 0),nz1*nx1*sizeof(hydro_t));
    mp_size_recv_buffer(grid->mp,BOUNDARY( 0, 0,-1),nx1*ny1*sizeof(hydro_t));
    mp_size_recv_buffer(grid->mp,BOUNDARY( 0, 0, 1),nx1*ny1*sizeof(hydro_t));

    mp_size_send_buffer(grid->mp,BOUNDARY(-1, 0, 0),ny1*nz1*sizeof(hydro_t));
    mp_size_send_buffer(grid->mp,BOUNDARY( 1, 0, 0),ny1*nz1*sizeof(hydro_t));
    mp_size_send_buffer(grid->mp,BOUNDARY( 0,-1, 0),nz1*nx1*sizeof(hydro_t));
    mp_size_send_buffer(grid->mp,BOUNDARY( 0, 1, 0),nz1*nx1*sizeof(hydro_t));
    mp_size_send_buffer(grid->mp,BOUNDARY( 0, 0,-1),nx1*ny1*sizeof(hydro_t));
    mp_size_send_buffer(grid->mp,BOUNDARY( 0, 0, 1),nx1*ny1*sizeof(hydro_t));
  }

  // Other field helpers are provided by macros in deck_wrapper.cxx

  //////////////////
  // Species helpers

  // FIXME: SILLY PROMOTIONS
  inline species_t *
  define_species( const char *name,
                  double q,
                  double m,
                  double max_local_np,
                  double max_local_nm,
                  double sort_interval,
                  double sort_out_of_place ) {
    // Compute a reasonble number of movers if user did not specify
    // Based on the twice the number of particles expected to hit the boundary
    // of a wpdt=0.2 / dx=lambda species in a 3x3x3 domain
    if( max_local_nm<0 ) {
      max_local_nm = 2*max_local_np/25;
      if( max_local_nm<16*(MAX_PIPELINE+1) )
        max_local_nm = 16*(MAX_PIPELINE+1);
    }
    return append_species( species( name, (float)q, (float)m,
                                    (int)max_local_np, (int)max_local_nm,
                                    (int)sort_interval, (int)sort_out_of_place,
                                    grid ), &species_list );
  }

#if defined( VPIC_ENABLE_PARTICLE_ANNOTATIONS ) || defined( VPIC_ENABLE_TRACER_PARTICLES )
  /**
   * @brief Create empty tracer species 
   *
   * It is up to the user to provide any additional annotations and create the 
   * tracer particles. Automatically adds TracerID 64-bit integer annotation 
   * for tracers.  
   * Tracer data can be buffered before I/O operations. The size of the buffer
   * is user controllable and defaults to storing 10x the maximum number of local
   * particles.
   * User can allocate additional memory for safety by supplting a multiplicative 
   * factor for the maximum number of local particles. If not factor is supplied 
   * then the code will default to allocating an additional 10% of the particles.
   * Species is automatically added to the tracer list.
   *
   * @param name The name of the tracer species
   * @param q Species particle charge
   * @param m Species particle rest mass
   * @param max_local_np Maximum number of particles for a single process
   * @param max_local_nm Maximum number of movers for a single process
   * @param sort_interval Number of time steps between particle sorting
   * @param sort_out_of_place Whether or not to sort out of place 
   * @param num_particles_buffer Number of particles to buffer before writing (Optional)
   * @param over_alloc_fact Multiplicative factor for allocating additional memory (Optional)
   * @param annotations Additional annotation variables (Optional)
   *
   * @return Pointer to tracer species
   */
  inline species_t * 
  define_tracer_species(const char* name,
                        const float q,
                        const float m,
                        const int max_local_np,
                        const int max_local_nm,
                        const int sort_interval,
                        const int sort_out_of_place,
                        const int num_particles_buffer = -1,
                        const float over_alloc_factor = 1.1,
                        annotation_vars_t annotations = annotation_vars_t()) {
    // Create tracer species based on the original species
    species_t* tracers = species( name, 
                                  q, m, 
                                  static_cast<int>(max_local_np*over_alloc_factor), 
                                  static_cast<int>(max_local_nm*over_alloc_factor), 
                                  sort_interval, sort_out_of_place, 
                                  grid);
    // Mark species as tracer
    tracers->is_tracer = true;

    // Add annotations for global tracer ID
    annotations.add_annotation<int64_t>(std::string("TracerID"));
    tracers->init_annotations(max_local_np, max_local_nm, annotations);

    // Initialize IO buffers
    int buffer_size = num_particles_buffer;
    if(num_particles_buffer == -1)
      buffer_size = max_local_np * 10;
    tracers->init_io_buffers(buffer_size, over_alloc_factor);

    return append_species(tracers, &tracers_list); 
//    return append_species(tracers, &species_list); 
  }

  /**
   * @brief Create empty tracer species based on existing species
   *
   * It is up to the user to provide any additional annotations and create the 
   * tracer particles. Automatically adds TracerID 64-bit integer annotation 
   * for tracers.  
   * Tracer data can be buffered before I/O operations. The size of the buffer
   * is user controllable and defaults to storing 10x the maximum number of local
   * particles.
   * User can allocate additional memory for safety by supplting a multiplicative 
   * factor for the maximum number of local particles. If not factor is supplied 
   * then the code will default to allocating an additional 10% of the particles.
   * Species is automatically added to the tracer list.
   *
   * @param name The name of the tracer species
   * @param original_species Parent species that the tracer is based on
   * @param max_local_np Maximum number of particles for a single process
   * @param max_local_nm Maximum number of movers for a single process
   * @param num_particles_buffer Number of particles to buffer before writing (Optional)
   * @param over_alloc_fact Multiplicative factor for allocating additional memory (Optional)
   * @param annotations Additional annotation variables (Optional)
   *
   * @return Pointer to tracer species
   */
  inline species_t * 
  define_tracer_species(const char* name,
                        species_t* original_species, 
                        const int max_local_np,
                        const int max_local_nm,
                        const int num_particles_buffer = -1,
                        const float over_alloc_factor = 1.1,
                        annotation_vars_t annotations = annotation_vars_t()) {
    // Create tracer species based on the original species
    species_t* tracers = species( name, 
                                  original_species->q, original_species->m, 
                                  static_cast<int>(max_local_np*over_alloc_factor), 
                                  static_cast<int>(max_local_nm*over_alloc_factor), 
                                  original_species->sort_interval, original_species->sort_out_of_place, 
                                  grid);
    // Mark species as tracer
    tracers->is_tracer = true;

    // Add annotations for global tracer ID
    annotations.add_annotation<int64_t>(std::string("TracerID"));
    tracers->init_annotations(max_local_np, max_local_nm, annotations);

    // Initialize IO buffers
    int buffer_size = num_particles_buffer;
    if(num_particles_buffer == -1)
      buffer_size = max_local_np * 10;
    tracers->init_io_buffers(buffer_size, over_alloc_factor);

    // Set parent species pointer
    tracers->parent_species = original_species;

    return append_species(tracers, &tracers_list); 
//    return append_species(tracers, &species_list); 
  }

  /**
   * @brief Create tracer species and copy/move every Nth particle from the parent
   *
   * Copies/Moves every Nth particle from the parent to the tracer species.
   * Automatically adds TracerID 64-bit integer annotation for tracers.  
   * It is up to the user to provide any additional annotations. 
   * Tracer data can be buffered before I/O operations. The size of the buffer
   * is user controllable and defaults to storing 10x the maximum number of local
   * particles.
   * User can allocate additional memory for safety by supplting a multiplicative 
   * factor for the maximum number of local particles. If not factor is supplied 
   * then the code will default to allocating an additional 10% of the particles.
   * Species is automatically added to the tracer list.
   *
   * @param name The name of the tracer species
   * @param original_species Parent species that the tracer is based on
   * @param tracer_type Decide whether to move or copy particles from parent
   * @param skip Number of particles to skip between copying/moving tracers
   * @param num_particles_buffer Number of particles to buffer before writing (Optional)
   * @param over_alloc_fact Multiplicative factor for allocating additional memory (Optional)
   * @param annotations Additional annotation variables (Optional)
   *
   * @return Pointer to tracer species
   */
  inline species_t * 
  define_tracer_species_by_nth( const char* name, 
                                species_t* original_species, 
                                const TracerType tracer_type, 
                                const float skip,
                                const int num_particles_buffer = -1,
                                const float over_alloc_factor = 1.1,
                                annotation_vars_t annotations = annotation_vars_t()) {

    // Adjust amount of local particles/movers for tracers
    int max_local_np = (static_cast<int>(static_cast<float>(original_species->max_np) * over_alloc_factor) / skip) + 1;
    int max_local_nm = (static_cast<int>(static_cast<float>(original_species->max_nm) * over_alloc_factor) / skip) + 1;
    
    // Create tracer species based on the original species
    species_t* tracers = species( name, 
                                  original_species->q, original_species->m, 
                                  max_local_np, max_local_nm, 
                                  original_species->sort_interval, original_species->sort_out_of_place, 
                                  grid);
    
    // Mark species as tracer
    tracers->is_tracer = true;

    // Add annotations for globas tracer ID
    annotations.add_annotation<int64_t>(std::string("TracerID"));
    if(tracer_type == TracerType::Copy)
      annotations.add_annotation<float>(std::string("Weight"));

    // Copy any annotations from the parent species
    if(original_species->using_annotations) {
      annotations.combine(original_species->annotation_vars);
    }

    int buffer_size = num_particles_buffer;
    if(num_particles_buffer == -1)
      buffer_size = max_local_np * 10;

    tracers->init_annotations(max_local_np, max_local_nm, annotations);
    tracers->init_io_buffers(buffer_size, over_alloc_factor);

    // Set parent species pointer
    tracers->parent_species = original_species;

    // Set tracer type
    tracers->tracer_type = tracer_type;

    // Copy of move particles to tracers
    tracers->create_tracers_by_nth(original_species, tracer_type, skip, rank());

    return append_species(tracers, &tracers_list); 
//    return append_species(tracers, &species_list); 
  }

  /**
   * @brief Create tracer species and copy/move N particles from the parent
   *
   * Creates N tracers by selecting N evenly spaced particles from the parent.
   * Automatically adds TracerID 64-bit integer annotation for tracers.  
   * It is up to the user to provide any additional annotations. 
   * Tracer data can be buffered before I/O operations. The size of the buffer
   * is user controllable and defaults to storing 10x the maximum number of local
   * particles.
   * User can allocate additional memory for safety by supplting a multiplicative 
   * factor for the maximum number of local particles. If not factor is supplied 
   * then the code will default to allocating an additional 10% of the particles.
   * Species is automatically added to the tracer list.
   *
   * @param name The name of the tracer species
   * @param original_species Parent species that the tracer is based on
   * @param tracer_type Decide whether to move or copy particles from parent
   * @param ntracers Number of particles to copy/move tracers
   * @param num_particles_buffer Number of particles to buffer before writing (Optional)
   * @param over_alloc_fact Multiplicative factor for allocating additional memory (Optional)
   * @param annotations Additional annotation variables (Optional)
   *
   * @return Pointer to tracer species
   */
  inline species_t * 
  define_tracer_species_with_n( const char* name, 
                                species_t* original_species, 
                                const TracerType tracer_type, 
                                const float ntracers,
                                const int num_particles_buffer = -1,
                                const float over_alloc_factor = 1.1,
                                annotation_vars_t annotations = annotation_vars_t()) {
    // Verify # of tracer is acceptable
    if(ntracers < 1.0 || static_cast<int>(ntracers) > original_species->np)
      ERROR(( "%f is a bad number of tracers. Should be in [%d,%d]", ntracers, 1, original_species->np));
    return define_tracer_species_by_nth(name, original_species, tracer_type, original_species->np / ntracers, num_particles_buffer, over_alloc_factor, annotations);
  }

  /**
   * @brief Create tracer species and N percent of the particles from the parent
   *
   * Creates tracers by selecting N percent of the particles from the parent 
   * and copying/moving them to the tracers.
   * Automatically adds TracerID 64-bit integer annotation for tracers.  
   * It is up to the user to provide any additional annotations. 
   * Tracer data can be buffered before I/O operations. The size of the buffer
   * is user controllable and defaults to storing 10x the maximum number of local
   * particles.
   * User can allocate additional memory for safety by supplting a multiplicative 
   * factor for the maximum number of local particles. If not factor is supplied 
   * then the code will default to allocating an additional 10% of the particles.
   * Species is automatically added to the tracer list.
   *
   * @param name The name of the tracer species
   * @param original_species Parent species that the tracer is based on
   * @param tracer_type Decide whether to move or copy particles from parent
   * @param skip Number of particles to skip between copying/moving tracers
   * @param num_particles_buffer Number of particles to buffer before writing (Optional)
   * @param over_alloc_fact Multiplicative factor for allocating additional memory (Optional)
   * @param annotations Additional annotation variables (Optional)
   *
   * @return Pointer to tracer species
   */
  inline species_t * 
  define_tracer_species_by_percentage(const char* name,
                                      species_t* original_species, 
                                      const TracerType tracer_type, 
                                      const float percentage, 
                                      const int num_particles_buffer = -1,
                                      const float over_alloc_factor = 1.1,
                                      annotation_vars_t annotations = annotation_vars_t()) {
    // Check if input percentage is valid
    if((percentage <= 0.0) || (percentage >= 100.0))
      ERROR(( "Percentage (%f) is not in [0,100]", percentage));
    
    int ntracers = static_cast<float>(original_species->np) * (percentage/100.0);
    return define_tracer_species_by_nth(name, original_species, tracer_type, original_species->np / ntracers, num_particles_buffer, over_alloc_factor, annotations);
  }

  /**
   * @brief Create tracer species from the parent by selecting based on the provided predicate 
   *
   * Copy/Move tracers from the parent based on the supplied filter function.
   * Automatically adds TracerID 64-bit integer annotation for tracers.  
   * It is up to the user to provide any additional annotations. 
   * Tracer data can be buffered before I/O operations. The size of the buffer
   * is user controllable and defaults to storing 10x the maximum number of local
   * particles.
   * User can allocate additional memory for safety by supplting a multiplicative 
   * factor for the maximum number of local particles. If not factor is supplied 
   * then the code will default to allocating an additional 10% of the particles.
   * Species is automatically added to the tracer list.
   *
   * @param name The name of the tracer species
   * @param original_species Parent species that the tracer is based on
   * @param tracer_type Decide whether to move or copy particles from parent
   * @param filter Filter function for selecting which particles to take from the parent 
   * @param num_particles_buffer Number of particles to buffer before writing (Optional)
   * @param over_alloc_fact Multiplicative factor for allocating additional memory (Optional)
   * @param annotations Additional annotation variables (Optional)
   *
   * @return Pointer to tracer species
   */
  inline species_t * 
  define_tracer_species_by_predicate( const char* name, 
                                      species_t* original_species, 
                                      const TracerType tracer_type, 
                                      std::function <bool (particle_t)> filter,
                                      const int num_particles_buffer = -1,
                                      const float over_alloc_factor = 1.1,
                                      annotation_vars_t annotations = annotation_vars_t()) {

    // Adjust amount of local particles/movers for tracers
    const size_t count_true = std::count_if(original_species->p, original_species->p + original_species->np, filter);
    const size_t max_local_np = ceil(original_species->max_np * over_alloc_factor * count_true/float(original_species->np)) + 1;
    const size_t max_local_nm = ceil(original_species->max_nm * over_alloc_factor * count_true/float(original_species->nm)) + 1;
    
    // Create tracer species based on the original species
    species_t* tracers = species( name, 
                                  original_species->q, original_species->m, 
                                  max_local_np, max_local_nm, 
                                  original_species->sort_interval, original_species->sort_out_of_place, 
                                  grid);
    
    // Mark species as tracer
    tracers->is_tracer = true;

    // Add annotations for globas tracer ID
    annotations.add_annotation<int64_t>(std::string("TracerID"));
    if(tracer_type == TracerType::Copy)
      annotations.add_annotation<float>(std::string("Weight"));
    // Copy any annotations from the parent species
    if(original_species->using_annotations) {
      annotations.combine(original_species->annotation_vars);
    }

    int buffer_size = num_particles_buffer;
    if(num_particles_buffer == -1)
      buffer_size = max_local_np * 10;

    tracers->init_annotations(max_local_np, max_local_nm, annotations);
    tracers->init_io_buffers(buffer_size, over_alloc_factor);

    // Set parent species pointer
    tracers->parent_species = original_species;

    // Set tracer type
    tracers->tracer_type = tracer_type;

    // Copy of move particles to tracers
    tracers->create_tracers_by_predicate(original_species, tracer_type, filter, rank());

    return append_species(tracers, &tracers_list); 
//    return append_species(tracers, &species_list); 
  }

#endif

  inline species_t *
  find_species( const char *name ) {
     return find_species_name( name, species_list );
  }

  inline species_t *
  find_species( int32_t id ) {
     return find_species_id( id, species_list );
  }

  ///////////////////
  // Particle helpers

  // Note: Don't use injection with aging during initialization

  // Defaults in the declaration below enable backwards compatibility.

  void
  inject_particle( species_t * sp,
                   double x,  double y,  double z,
                   double ux, double uy, double uz,
                   double w,  double age = 0, int update_rhob = 1 );

  // Inject particle raw is for power users!
  // No nannyism _at_ _all_:
  // - Availability of free stoarge is _not_ checked.
  // - Particle displacements and voxel index are _not_ for validity.
  // - The rhob field is _not_ updated.
  // - Injection with displacment may use up movers (i.e. don't use
  //   injection with displacement during initialization).
  // This injection is _ultra_ _fast_.

  inline void
  inject_particle_raw( species_t * RESTRICT sp,
                       float dx, float dy, float dz, int32_t i,
                       float ux, float uy, float uz, float w ) {
    particle_t * RESTRICT p = sp->p + (sp->np++);
    p->dx = dx; p->dy = dy; p->dz = dz; p->i = i;
    p->ux = ux; p->uy = uy; p->uz = uz; p->w = w;
  }

  // This variant does a raw inject and moves the particles

  inline void
  inject_particle_raw( species_t * RESTRICT sp,
                       float dx, float dy, float dz, int32_t i,
                       float ux, float uy, float uz, float w,
                       float dispx, float dispy, float dispz,
                       int update_rhob ) {
    particle_t       * RESTRICT p  = sp->p  + (sp->np++);
    particle_mover_t * RESTRICT pm = sp->pm + sp->nm;
    p->dx = dx; p->dy = dy; p->dz = dz; p->i = i;
    p->ux = ux; p->uy = uy; p->uz = uz; p->w = w;
    pm->dispx = dispx; pm->dispy = dispy; pm->dispz = dispz; pm->i = sp->np-1;
    if( update_rhob ) accumulate_rhob( field_array->f, p, grid, -sp->q );
    sp->nm += move_p( sp->p, pm, field_array->k_jf_accum_h, grid, sp->q );
  }

  //////////////////////////////////
  // Random number generator helpers

  // seed_rand seed the all the random number generators.  The seed
  // used for the individual generators is based off the user provided
  // seed such each local generator in each process (rng[0:r-1]) gets
  // a unique seed.  Each synchronous generator (sync_rng[0:r-1]) gets a
  // unique seed that does not overlap with the local generators
  // (common across each process).  Lastly, all these seeds are such
  // that, no individual generator seeds are reused across different
  // user seeds.
  // FIXME: MTRAND DESPERATELY NEEDS A LARGER SEED SPACE!

  inline void seed_entropy( int base ) {
    seed_rng_pool( entropy,      base, 0 );
    seed_rng_pool( sync_entropy, base, 1 );
  }

  // Uniform random number on (low,high) (open interval)
  // FIXME: IS THE INTERVAL STILL OPEN IN FINITE PRECISION
  //        AND IS THE OPEN INTERVAL REALLY WHAT USERS WANT??
  inline double uniform( rng_t * rng, double low, double high ) {
    double dx = drand( rng );
    return low*(1-dx) + high*dx;
  }

  // Normal random number with mean mu and standard deviation sigma
  inline double normal( rng_t * rng, double mu, double sigma ) {
    return mu + sigma*drandn( rng );
  }

  /////////////////////////////////
  // Emitter and particle bc helpers

  // Note that append_emitter is hacked to silently returne if the
  // emitter is already in the list.  This allows things like:
  //
  // define_surface_emitter( my_emitter( ... ), rgn )
  // ... or ...
  // my_emit_t * e = my_emit( ... )
  // define_surface_emitter( e, rgn )
  // ... or ...
  // my_emit_t * e = define_emitter( my_emit( ... ) )
  // define_surface_emitter( e, rng )
  // ...
  // All to work.  (Nominally, would like define_surface_emitter
  // to evaluate to the value of e.  But, alas, the way
  // define_surface_emitter works and language limitations of
  // strict C++ prevent this.)

  inline emitter_t *
  define_emitter( emitter_t * e ) {
    return append_emitter( e, &emitter_list );
  }

  inline particle_bc_t *
  define_particle_bc( particle_bc_t * pbc ) {
    return append_particle_bc( pbc, &particle_bc_list );
  }

  inline collision_op_t *
  define_collision_op( collision_op_t * cop ) {
    return append_collision_op( cop, &collision_op_list );
  }

  ////////////////////////
  // Miscellaneous helpers

  inline void abort( double code ) {
    nanodelay(2000000000); mp_abort((((int)code)<<17)+1);
  }

  // Truncate "a" to the nearest integer multiple of "b"
  inline double trunc_granular( double a, double b ) { return b*int(a/b); }

  // Compute the remainder of a/b
  inline double remainder( double a, double b ) { return std::remainder(a,b); }
  // remainder(a,b);

  // Compute the Courant length on a regular mesh
  inline double courant_length( double lx, double ly, double lz,
				double nx, double ny, double nz ) {
    double w0, w1 = 0;
    if( nx>1 ) w0 = nx/lx, w1 += w0*w0;
    if( ny>1 ) w0 = ny/ly, w1 += w0*w0;
    if( nz>1 ) w0 = nz/lz, w1 += w0*w0;
    return sqrt(1/w1);
  }

  //////////////////////////////////////////////////////////
  // These friends are used by the checkpt / restore service

  friend void checkpt_vpic_simulation( const vpic_simulation * vpic );
  friend vpic_simulation * restore_vpic_simulation( void );
  friend void reanimate_vpic_simulation( vpic_simulation * vpic );

  ////////////////////////////////////////////////////////////
  // User input deck provided functions (see deck_wrapper.cxx)

  void user_initialization( int argc, char **argv );
  void user_particle_injection(void);
  void user_current_injection(void);
  void user_field_injection(void);
  void user_diagnostics(void);
  void user_particle_collisions(void);

  /**
   * @brief Copy all field data to the host, if it has not already been copied
   * this step
   *
   * This does not guarantee that the particles are truly up to date, since it
   * checks only if a copy has already been done at some point during the
   * current step, but it will always work in user_diagnostics unless the loop
   * is modified or a user modifies particles during user_diagnostics.
   *
   */
  void user_diagnostics_copy_field_mem_to_host()
  {
      if (step() > field_array->last_copied)
        field_array->copy_to_host();
  }

  /**
   * @brief Copy all available particle memory from device to host, for a given
   * species, if it has not been copied this step
   *
   * This does not guarantee that the particles are truly up to date, since it
   * checks only if a copy has already been done at some point during the
   * current step, but it will always work in user_diagnostics unless the loop
   * is modified or a user modifies particles during user_diagnostics.
   *
   * @param speciesname the name of the species to copy
   */
  void user_diagnostics_copy_particles_mem_to_host(const char * speciesname)
  {
      species_t * sp = find_species_name(speciesname, species_list);
      if(!sp) ERROR(( "Invalid Species name: %s", speciesname ));

      if(step() > sp->last_copied)
        sp->copy_to_host();
  }

  /**
   * @brief Copy all available particle memory from host to device, for a given
   * list of species, if it has not been copied this step
   *
   * This does not guarantee that the particles are truly up to date, since it
   * checks only if a copy has already been done at some point during the
   * current step, but it will always work in user_diagnostics unless the loop
   * is modified or a user modifies particles during user_diagnostics.
   */
  void user_diagnostics_copy_all_particles_mem_to_host(species_t* species_list)
  {
      auto* sp = species_list;
      LIST_FOR_EACH( sp, species_list ) {
          if(step() > sp->last_copied)
            sp->copy_to_host();
      }
  }

};


/**
 * @brief After a checkpoint restore, we must move the data back over to the
 * Kokkos objects. This currently must be done for all views
 */
void restore_kokkos(vpic_simulation& simulation);
// TODO: would this make more sense as a member function on vpic_simulation_t


#endif // vpic_h
