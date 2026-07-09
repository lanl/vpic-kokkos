/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version (data structures based on earlier
 *                    V4PIC versions)
 *
 */

#ifndef _species_advance_h_
#define _species_advance_h_

#include <iostream>

#include "../sf_interface/sf_interface.h"
#include "Kokkos_DualView.hpp"

#ifdef VPIC_ENABLE_HDF5
#include "hdf5.h"
#endif
#ifdef VPIC_ENABLE_HDF5_ASYNC
#include "h5_async_vol.h"
#endif
#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
#include "standard/annotations.h"
#endif

typedef int32_t species_id; // Must be 32-bit wide for particle_injector_t

// FIXME: Eventually particle_t (definitely) and ther other formats
// (maybe) should be opaque and specific to a particular
// species_advance implementation

typedef struct particle {
  float dx, dy, dz; // Particle position in cell coordinates (on [-1,1])
  int32_t i;        // Voxel containing the particle.  Note that
  /**/              // particled awaiting processing by boundary_p
  /**/              // have actually set this to 8*voxel + face where
  /**/              // face is the index of the face they interacted
  /**/              // with (on 0:5).  This limits the local number of
  /**/              // voxels to 2^28 but emitter handling already
  /**/              // has a stricter limit on this (2^26).
  float ux, uy, uz; // Particle normalized momentum
  float w;          // Particle weight (number of physical particles)
#ifdef VARIABLE_CHARGE
  float qp;     // Particle charge
#endif

} particle_t;
 
// WARNING: FUNCTIONS THAT USE A PARTICLE_MOVER ASSUME THAT EVERYBODY
// WHO USES THAT PARTICLE MOVER WILL HAVE ACCESS TO PARTICLE ARRAY

typedef struct particle_mover {
  float dispx, dispy, dispz; // Displacement of particle
  size_t i;                 // Index of the particle to move
  float _pad[3];
} particle_mover_t;

// NOTE: THE LAYOUT OF A PARTICLE_INJECTOR _MUST_ BE COMPATIBLE WITH
// THE CONCATENATION OF A PARTICLE_T AND A PARTICLE_MOVER!

typedef struct particle_injector {
  float dx, dy, dz;          // Particle position in cell coords (on [-1,1])
  int32_t i;                 // Index of cell containing the particle
  float ux, uy, uz;          // Particle normalized momentum
  float w;                   // Particle weight (number of physical particles)
#ifdef VARIABLE_CHARGE
  float qp;     // Particle charge
#endif
  float dispx, dispy, dispz; // Displacement of particle
  species_id sp_id;          // Species of particle
} particle_injector_t;

// Seems like this belongs in boundary.h
class species_t;
typedef struct pb_diagnostic {

    int         enable; // Wether or not to use this diagnostic
    int         enable_user; // Will the user write additional values?
    // TODO: Do I need this circular reference?
    species_t   *sp; // Pointer to the species
    char        *fname;
    int         file_counter; // How many files has this rank written?
    size_t      store_counter; // How many floats need to be written from the buffer
    size_t      write_counter; // How many floats have been written to the current file
    size_t      bufflen; // Size of the memory buffer in sizeof(float)
    float       *buff; // The buffer that stores data to be written

    int         num_user_writes; // Number of floats the user stores per particle
    int         num_writes; // Total number of floats stored per particle

    int         write_ux;
    int         write_uy;
    int         write_uz;
    int         write_momentum_magnitude;
    int         write_posx;
    int         write_posy;
    int         write_posz;
    int         write_weight;
    int         write_pq;

} pb_diagnostic_t;

enum class TracerType { Copy, Move };

class species_t {
    public:

        char * name;                        // Species name
        float q;                            // Species particle charge
        float m;                            // Species particle rest mass

        size_t np = 0, max_np = 0;             // Number and max local particles
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
        particle_t * ALIGNED(128) p;        // Array of particles for the species
#endif

        // TODO: these could be unsigned?
        size_t nm = 0, max_nm = 0;             // Number and max local movers in use

        particle_mover_t * ALIGNED(128) pm; // Particle movers

        int64_t last_sorted;                // Step when the particles were last sorted.
        int64_t last_indexed;               // Step when the particles were last indexed.    
        int sort_interval;                  // How often to sort the species
        int sort_out_of_place;              // Sort method
        int * ALIGNED(128) partition;       // Static array indexed 0:
        /**/                                // (nx+2)*(ny+2)*(nz+2).  Each value
        /**/                                // corresponds to the associated particle
        /**/                                // array index of the first particle in
        /**/                                // the cell.  Array is allocated and
        /**/                                // values computed in sort_p.  Purpose is
        /**/                                // for implementing collision models
        /**/                                // This is given in terms of the
        /**/                                // underlying's grids space filling
        /**/                                // curve indexing.  Thus, immediately
        /**/                                // after a sort:
        /**/                                //   sp->p[sp->partition[g->sfc[i]  ]:
        /**/                                //         sp->partition[g->sfc[i]+1]-1]
        /**/                                // are all the particles in voxel
        /**/                                // with local index i, while:
        /**/                                //   sp->p[ sp->partition[ j   ]:
        /**/                                //          sp->partition[ j+1 ] ]
        /**/                                // are all the particles in voxel
        /**/                                // with space filling curve index j.
        /**/                                // Note: SFC NOT IN USE RIGHT NOW THUS
        /**/                                // g->sfc[i]=i ABOVE.

        k_particle_partition_t k_partition_d;
        k_particle_partition_t::HostMirror k_partition_h;

        // Used for indirect sorts.
        k_particle_sortindex_t k_sortindex_d;
        k_particle_sortindex_t::HostMirror k_sortindex_h;

        grid_t * g;                         // Underlying grid
        species_id id;                      // Unique identifier for a species
        species_t* next = NULL;             // Next species in the list

        // Particle boundary diagnostic.
        pb_diagnostic_t * pb_diag = NULL;

        // Tracer type
        TracerType tracer_type;
        species_t* parent_species = NULL;


        //// END CHECKPOINTED DATA, START KOKKOS //////


        k_particles_t k_p_d;                 // kokkos particles view on device
        k_particles_i_t k_p_i_d;             // kokkos particles view on device

        k_particles_t::HostMirror k_p_h;     // kokkos particles view on host
        k_particles_i_t::HostMirror k_p_i_h; // kokkos particles view on host

        k_particle_copy_t k_pc_d;            // kokkos particles copy for movers view on device
        k_particle_i_copy_t k_pc_i_d;        // kokkos particles copy for movers view on device

        k_particle_copy_t::HostMirror k_pc_h;      // kokkos particles copy for movers view on host
        k_particle_i_copy_t::HostMirror k_pc_i_h;  // kokkos particles i copy for movers view on host

        // Only need host versions
        k_particle_copy_t::HostMirror k_pr_h;      // kokkos particles copy for received particles
        k_particle_i_copy_t::HostMirror k_pr_i_h;  // kokkos particles i copy for received particles

        k_particle_movers_t k_pm_d;         // kokkos particle movers on device
        k_particle_i_movers_t k_pm_i_d;         // kokkos particle movers on device

        k_particle_movers_t::HostMirror k_pm_h;  // kokkos particle movers on host
        k_particle_i_movers_t::HostMirror k_pm_i_h;  // kokkos particle movers on host

        // TODO: what is an iterator here??
        k_counter_t k_nm_d;               // nm iterator
        k_counter_t::HostMirror k_nm_h;

#if defined(VPIC_ENABLE_PARTICLE_ANNOTATIONS) || defined(VPIC_ENABLE_TRACER_PARTICLES)
#ifdef VPIC_ENABLE_HDF5_ASYNC
        hid_t es_id;
#endif
        bool is_tracer = false;
        bool using_annotations = false;
        annotation_vars_t annotation_vars;
        annotations_t<Kokkos::DefaultExecutionSpace>     annotations_d;
        annotations_t<Kokkos::DefaultHostExecutionSpace> annotations_h;
        annotations_t<Kokkos::DefaultExecutionSpace>     annotations_copy_d;
        annotations_t<Kokkos::DefaultHostExecutionSpace> annotations_copy_h;
        annotations_t<Kokkos::DefaultHostExecutionSpace> annotations_recv_h;

        size_t np_buffered=0;
        size_t np_buffered_max=0;
        std::vector<std::pair<int64_t,int64_t>> np_per_ts; // (np, ts)

        k_particles_t                                particle_io_buffer_d;
        k_particles_i_t                              particle_cell_io_buffer_d;
        annotations_t<Kokkos::DefaultExecutionSpace> annotations_io_buffer_d;
        Kokkos::View<float**, Kokkos::LayoutLeft>    tracer_buffer_d;

        k_particles_t::HostMirror                               particle_io_buffer_h;
        k_particles_i_t::HostMirror                             particle_cell_io_buffer_h;
        annotations_t<Kokkos::DefaultHostExecutionSpace>        annotations_io_buffer_h;
        Kokkos::View<float**, Kokkos::LayoutLeft>::HostMirror   tracer_buffer_h;
#endif


        // TODO: this should ultimatley be removeable.
        // This tracks the number of particles we need to move back to the device
        // And is basically the same as nm at certain times?
        size_t num_to_copy = 0;

        // Step when the species was last copied to to the host.  The copy can
        // take place at any time during the step, so checking
        // last_copied==step() does not mean that the host and device
        // data are the same.  Typically, copy is called immediately after the
        // step is incremented and before or during user_diagnostics.  Checking
        // last_copied==step() in these circumstances does mean the host
        // is up to date, unless you do unusual stuff in user_diagnostics.
        //
        // This number is tracked on the host only, and may be inaccurate on
        // the device.
        int64_t last_copied = -1;

        // Static allocations for the compressor
        Kokkos::View<size_t*> unsafe_index;
        Kokkos::View<size_t> clean_up_to_count;
        Kokkos::View<size_t> clean_up_from_count;
        Kokkos::View<size_t>::HostMirror clean_up_from_count_h;
        Kokkos::View<size_t*> clean_up_from;
        Kokkos::View<size_t*> clean_up_to;

        // Init Kokkos Particle Arrays
        species_t() = default;

        species_t(size_t n_particles, size_t n_pmovers)
        {
           init_kokkos_particles(n_particles, n_pmovers);
        }

        void init_kokkos_particles()
        {
            init_kokkos_particles(max_np, max_nm);
        }
        void init_kokkos_particles(size_t n_particles, size_t n_pmovers)
        {
            k_p_d = k_particles_t("k_particles", n_particles);
            k_p_i_d = k_particles_i_t("k_particles_i", n_particles);
            k_pc_d = k_particle_copy_t("k_particle_copy_for_movers", n_pmovers);
            k_pc_i_d = k_particle_i_copy_t("k_particle_copy_for_movers_i", n_pmovers);
            k_pr_h = k_particle_copy_t::HostMirror("k_particle_send_for_movers", n_pmovers);
            k_pr_i_h = k_particle_i_copy_t::HostMirror("k_particle_send_for_movers_i", n_pmovers);
            k_pm_d = k_particle_movers_t("k_particle_movers", n_pmovers);
            k_pm_i_d = k_particle_i_movers_t("k_particle_movers_i", n_pmovers);
            k_nm_d = k_counter_t("k_nm"); // size 1 encoded in type
            unsafe_index = Kokkos::View<size_t*>("safe index", 2*n_pmovers);
            clean_up_to_count = Kokkos::View<size_t>("clean up to count");
            clean_up_from_count = Kokkos::View<size_t>("clean up from count");
            clean_up_from = Kokkos::View<size_t*>("clean up from", n_pmovers);
            clean_up_to = Kokkos::View<size_t*>("clean up to", n_pmovers);

            k_p_h = Kokkos::create_mirror_view(k_p_d);
            k_p_i_h = Kokkos::create_mirror_view(k_p_i_d);

            k_pc_h = Kokkos::create_mirror_view(k_pc_d);
            k_pc_i_h = Kokkos::create_mirror_view(k_pc_i_d);

            k_pm_h = Kokkos::create_mirror_view(k_pm_d);
            k_pm_i_h = Kokkos::create_mirror_view(k_pm_i_d);

            k_nm_h = Kokkos::create_mirror_view(k_nm_d);

            clean_up_from_count_h = Kokkos::create_mirror_view(clean_up_from_count);
        }

        /**
         * @brief Copies all the outbound particles and movers to the host.
         */
        void copy_outbound_to_host();

        /**
         * @brief Copies all the particles and movers from the device to the host.
         */
        void copy_to_host();

        /**
         * @brief Copies all the particles and movers from the host to the device.
         */
        void copy_to_device();

        /**
         * @brief Copies all the inbound particles from the host to the device.
         */
        void copy_inbound_to_device();

#if defined(VPIC_ENABLE_PARTICLE_ANNOTATIONS) || defined(VPIC_ENABLE_TRACER_PARTICLES)
        /**
         *  @brief Create tracer particle from existing species
         *
         *  @param src_species    Species to create particle from
         *  @param index          Index of particle to use
         */
        void create_tracer_from(species_t* src_species, const size_t index); 

        /**
         *  @brief Allocate memory for IO buffering tracers
         *
         *  @param N_particles       Number of particles to buffer before dumping
         *  @param over_alloc_factor Multiplier for over allocating space
         */
        void init_io_buffers(const size_t N_particles, const float over_alloc_factor);
        void init_io_buffers(const size_t N_particles);

        /**
         * @brief Add additional per particle annotations. 
         *
         * @param num_particles Number of particles using annotations
         * @param num_movers    Number of movers that need annotations
         * @param vars          Annotation variables
         */
        void init_annotations( const size_t num_particles, const size_t num_movers, annotation_vars_t& vars );

        /**
         * Create tracer particles from parent species using a predicate
         *
         * @param parent_species Species of particles to create tracers from
         * @param tracer_type    Whether to Copy or Move particles to tracers
         * @param filter         Generic function to decide whether particle is a tracer
         */
        void create_tracers_by_predicate( species_t* parent_species,
                                          const TracerType tracer_type,
                                          std::function <bool (particle_t)> filter, const int rank ); 

        /**
         * Create tracer particles from parent species. Select every Nth particle
         *
         * @param parent_species Species of particles to create tracers from
         * @param tracer_type    Whether to Copy or Move particles to tracers
         * @param skip           Amount of particles to skip between selections
         */
        void create_tracers_by_nth( species_t* parent_species,
                                    const TracerType tracer_type,
                                    float skip, int rank); 
#endif
};

// In species_advance.c

int
num_species( const species_t * sp_list );

void
delete_species_list( species_t * sp_list );

species_t *
find_species_id( species_id id,
                 species_t * sp_list );

species_t *
find_species_name( const char * name,
                   species_t * sp_list );

species_t *
append_species( species_t * sp,
                species_t ** sp_list );

species_t *
species( const char * name,
         float q,
         float m,
         size_t max_local_np,
         size_t max_local_nm,
         int sort_interval,
         int sort_out_of_place,
         grid_t * g );

// FIXME: TEMPORARY HACK UNTIL THIS SPECIES_ADVANCE KERNELS
// CAN BE CONSTRUCTED ANALOGOUS TO THE FIELD_ADVANCE KERNELS
// (THESE FUNCTIONS ARE NECESSARY FOR HIGHER LEVEL CODE)

// In sort_p.c

void
sort_p( species_t * RESTRICT sp );

// In advance_p.cxx

void
advance_p( /**/  species_t            * RESTRICT sp,
                 interpolator_array_t * RESTRICT ia,
                 field_array_t* RESTRICT fa );

// In center_p.cxx

// This does a half advance field advance and a half Boris rotate on
// the particles.  As such particles with r at the time step and u
// half a step stale is moved second order accurate to have r and u on
// the time step.

void
center_p( /**/  species_t            * RESTRICT sp,
          const interpolator_array_t * RESTRICT ia );

// In uncenter_p.cxx

// This is the inverse of center_p.  Thus, particles with r and u at
// the time step are adjusted to have r at the time step and u half a
// step stale.

void
uncenter_p( /**/  species_t            * RESTRICT sp,
            const interpolator_array_t * RESTRICT ia );

// In energy.cxx

// This computes the kinetic energy stored in the particles.  The
// calculation is done numerically robustly.  All nodes get the same
// result.

double
energy_p( const species_t            * RESTRICT sp,
          const interpolator_array_t * RESTRICT ia );

double
energy_p_kokkos( const species_t            * RESTRICT sp,
          const interpolator_array_t * RESTRICT ia );

// In rho_p.cxx

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
void
accumulate_rho_p( /**/  field_array_t * RESTRICT fa,
                  const species_t     * RESTRICT sp );
#endif

void
accumulate_rhob( field_t          * RESTRICT ALIGNED(128) f,
                 const particle_t * RESTRICT ALIGNED(32)  p,
                 const grid_t     * RESTRICT              g,
                 const float                              qsp );
void
k_accumulate_rho_p( /**/  field_array_t * RESTRICT fa,
                  const species_t     * RESTRICT sp,
                const grid_t* g,
              float gdx,
    float gdy,
    float gdz,
    float gdt,
    const int nx,
    const int ny,
    const int nz);

void k_accumulate_rhob(
            k_field_t& kfield,
            k_particles_t& kpart,
            k_particle_movers_t& kpart_movers,
            const grid_t* RESTRICT g,
            const float qsp,
            const size_t nm);

void k_accumulate_rhob_single_cpu(
            k_field_t& kfield,
            k_particles_t& kpart,
            k_particles_i_t& kpart_i,
            const int i,
            const grid_t* g,
            const float qsp
);

// In hydro_p.c

void
accumulate_hydro_p( /**/  hydro_array_t        * RESTRICT ha,
                    const species_t            * RESTRICT sp,
                    const interpolator_array_t * RESTRICT ia );

void accumulate_hydro_p_kokkos(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_hydro_t k_hydro,
        k_interpolator_t& k_interp,
        const species_t            * RESTRICT sp
);

void accumulate_hydro_p_kokkos_nomove_ngp(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_hydro_t k_hydro,
        k_interpolator_t& k_interp,
        const species_t            * RESTRICT sp
);

// In move_p.cxx
int
move_p( particle_t       * ALIGNED(128) p0,
        particle_mover_t * ALIGNED(16)  pm,
        //accumulator_t    * ALIGNED(128) a0,
        k_jf_accum_t::HostMirror& k_jf_accum,
        const grid_t     *              g,
        const float                     qsp );

// move_p_kokkos now supports curvilinear coordinates
template<class particle_view_t, class particle_i_view_t, class neighbor_view_t, class scatter_view_t>
int
KOKKOS_INLINE_FUNCTION
move_p_kokkos(
    const particle_view_t& k_particles,
    const particle_i_view_t& k_particles_i,
    particle_mover_t* ALIGNED(16)  pm,
    scatter_view_t scatter_view,
    const grid_t* g,
    neighbor_view_t& d_neighbor,
    int64_t rangel,
    int64_t rangeh,
    const float qsp,
    float gdx,
    float gdy,
    float gdz,
    float gdt,
    const int nx,
    const int ny,
    const int nz
)
{

  #define p_dx    k_particles(pi, particle_var::dx)
  #define p_dy    k_particles(pi, particle_var::dy)
  #define p_dz    k_particles(pi, particle_var::dz)
  #define p_ux    k_particles(pi, particle_var::ux)
  #define p_uy    k_particles(pi, particle_var::uy)
  #define p_uz    k_particles(pi, particle_var::uz)
  #define p_w     k_particles(pi, particle_var::w)
#ifdef VARIABLE_CHARGE
  #define p_q     k_particles(pi, particle_var::qp)
#endif
  #define pii     k_particles_i(pi)

  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, q;
  int axis, face;
  int64_t neighbor;
  size_t pi = pm->i;
  float ux,uy,uz,x_half,y_half,z_half,fracdt;
  constexpr float one=1.;

  auto scatter_access = scatter_view.access();

#ifdef VARIABLE_CHARGE
  q = p_q*p_w;
#else
  q = qsp*p_w;
#endif

  for(;;) {
    int ii = pii;
    s_midx = p_dx;
    s_midy = p_dy;
    s_midz = p_dz;

    ux = p_ux;
    uy = p_uy;
    uz = p_uz;

    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    float grad_xi_x, grad_xi_y, grad_xi_z;
    float grad_eta_x, grad_eta_y, grad_eta_z;
    float grad_mu_x, grad_mu_y, grad_mu_z;
    float jac;

    compute_reciprocal_basis(
        g,
        s_midx, s_midy, s_midz, ii, nx, ny, nz,
        gdx, gdy, gdz,
        grad_xi_x, grad_xi_y, grad_xi_z,
        grad_eta_x, grad_eta_y, grad_eta_z,
        grad_mu_x, grad_mu_y, grad_mu_z,
        jac);

    // Transform Cartesian velocities to contravariant logical velocities
    float d_xi_dt  = ux * grad_xi_x  + uy * grad_xi_y  + uz * grad_xi_z;
    float d_eta_dt = ux * grad_eta_x + uy * grad_eta_y + uz * grad_eta_z;
    float d_mu_dt  = ux * grad_mu_x  + uy * grad_mu_y  + uz * grad_mu_z;

    // Find position of particle at t_n+1/2
    v0 = (ux==0) ? 0.0 : s_dispx/d_xi_dt/gdt*2.;
    v1 = (uy==0) ? 0.0 : s_dispy/d_eta_dt/gdt*2.;
    v2 = (uz==0) ? 0.0 : s_dispz/d_mu_dt/gdt*2.;

    //this is equivalent in cart:
    // float d_xi_dt  = ux * grad_xi_x  + uy * grad_xi_y  + uz * grad_xi_z;
    // float d_eta_dt = ux * grad_eta_x + uy * grad_eta_y + uz * grad_eta_z;
    // float d_mu_dt  = ux * grad_mu_x  + uy * grad_mu_y  + uz * grad_mu_z;

    // float h_xi  = 1.0f / sqrtf(grad_xi_x*grad_xi_x   + grad_xi_y*grad_xi_y   + grad_xi_z*grad_xi_z);
    // float h_eta = 1.0f / sqrtf(grad_eta_x*grad_eta_x + grad_eta_y*grad_eta_y + grad_eta_z*grad_eta_z);
    // float h_mu  = 1.0f / sqrtf(grad_mu_x*grad_mu_x   + grad_mu_y*grad_mu_y   + grad_mu_z*grad_mu_z);

    // float u_phys_xi  = d_xi_dt  * h_xi;
    // float u_phys_eta = d_eta_dt * h_eta;
    // float u_phys_mu  = d_mu_dt  * h_mu;


    // // Find position of particle at t_n+1/2
    // v0 = (ux==0) ? 0.0 : s_dispx/u_phys_xi/gdt*gdx;
    // v1 = (uy==0) ? 0.0 : s_dispy/u_phys_eta/gdt*gdy;
    // v2 = (uz==0) ? 0.0 : s_dispz/u_phys_mu/gdt*gdz;

    // WARNING(("gdx %e h_xi %e", gdx, h_xi));
    fracdt = v0;
    if(v1>fracdt) fracdt=v1;
    if(v2>fracdt) fracdt=v2;
    fracdt = 2.0*(fracdt-0.5);
    
    if(fracdt>0){
      
      x_half = s_midx + fracdt*d_xi_dt*gdt/2.;
      y_half = s_midy + fracdt*d_eta_dt*gdt/2.; 
      z_half = s_midz + fracdt*d_mu_dt*gdt/2.;
      
      if( x_half<=one &&  y_half<=one &&  z_half<=one &&
         -x_half<=one && -y_half<=one && -z_half<=one) {
        
        // Compute reciprocal basis vectors at half-step position
        float grad_xi_x, grad_xi_y, grad_xi_z;
        float grad_eta_x, grad_eta_y, grad_eta_z;
        float grad_mu_x, grad_mu_y, grad_mu_z;
        float jac;
        
        compute_reciprocal_basis(
            g,
            x_half, y_half, z_half, ii, nx, ny, nz,
            gdx, gdy, gdz,
            grad_xi_x, grad_xi_y, grad_xi_z,
            grad_eta_x, grad_eta_y, grad_eta_z,
            grad_mu_x, grad_mu_y, grad_mu_z,
            jac);
        
        // Transform Cartesian velocity to contravariant logical velocity
        float d_xi_dt  = ux * grad_xi_x  + uy * grad_xi_y  + uz * grad_xi_z;
        float d_eta_dt = ux * grad_eta_x + uy * grad_eta_y + uz * grad_eta_z;
        float d_mu_dt  = ux * grad_mu_x  + uy * grad_mu_y  + uz * grad_mu_z;
        
        // Compute proper volume scaling (inverse Jacobian)
        float inv_jac = 1.0f / jac;
        
#ifdef SHAPE_NGP
        // Deposit contravariant logical current with proper volume weighting
        // Note: Factor of 0.125 = 1/8 accounts for the cell volume normalization
        // in logical coordinates (cell spans [-1,1]^3 = volume of 8)
        float qfactor = q * 0.125f * inv_jac;
        
        scatter_access(ii, field_var::jfx)  += qfactor * d_xi_dt;
        scatter_access(ii, field_var::jfy)  += qfactor * d_eta_dt;
        scatter_access(ii, field_var::jfz)  += qfactor * d_mu_dt;
        scatter_access(ii, field_var::rhof) += qfactor;
#endif // SHAPE_NGP
        
      } //if inbnds
      
    } //if more than half dt left
    
    s_dir[0] = (s_dispx>0) ? 1 : -1;
    s_dir[1] = (s_dispy>0) ? 1 : -1;
    s_dir[2] = (s_dispz>0) ? 1 : -1;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak
    /**/      v3=2,  axis=3;
    if(v0<v3) v3=v0, axis=0;
    if(v1<v3) v3=v1, axis=1;
    if(v2<v3) v3=v2, axis=2;
    v3 *= 0.5;

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;
    
    // Compute the remaining particle displacement
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    // Compute the new particle offset
    p_dx += s_dispx+s_dispx;
    p_dy += s_dispy+s_dispy;
    p_dz += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)
    if( axis==3 ) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    v0 = s_dir[axis];
    k_particles(pi, particle_var::dx + axis) = v0;
    face = axis; if( v0>0 ) face += 3;

    neighbor = d_neighbor( 6*ii + face );

    if( neighbor==reflect_particles ) {
      // Hit a reflecting boundary condition
      k_particles(pi, particle_var::ux + axis) = -k_particles(pi, particle_var::ux + axis);
      float* disp = static_cast<float*>(&(pm->dispx));
      disp[axis] = -disp[axis];
      continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
      // Cannot handle the boundary condition here
      pii = 8*pii + face;
      return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel
    pii = neighbor - rangel;
    k_particles(pi, particle_var::dx + axis) = -v0;
  }
  
  #undef p_dx
  #undef p_dy
  #undef p_dz
  #undef p_ux
  #undef p_uy
  #undef p_uz
  #undef p_w
#ifdef VARIABLE_CHARGE
  #undef p_q
#endif
  #undef pii

  return 0; // Return "mover not in use"
}

// this has no data race protection for write into the accumulators
// this has no data race protection for write into the accumulators
template<class particle_view_t, class particle_i_view_t, class neighbor_view_t, class accum_view_t>
int
move_p_kokkos_host_serial(
    const particle_view_t& k_particles,
    const particle_i_view_t& k_particles_i,
    particle_mover_t* ALIGNED(16) pm,
    accum_view_t& k_jf_accum,
    const grid_t* g,
    neighbor_view_t& d_neighbor,
    int64_t rangel,
    int64_t rangeh,
    const float qsp
)
{
  const int nx = g->nx;
  const int ny = g->ny;
  const int nz = g->nz;

  float ux,uy,uz,x_half,y_half,z_half,fracdt;
  constexpr float one=1., one_twelfth=1./12.;
  const float gdx=g->dx, gdy=g->dy, gdz=g->dz, gdt=g->dt;
  const float rV = g->rdx * g->rdy * g->rdz;
  const float rV12 = rV*one_twelfth;

  #define p_dx    k_particles(pi, particle_var::dx)
  #define p_dy    k_particles(pi, particle_var::dy)
  #define p_dz    k_particles(pi, particle_var::dz)
  #define p_ux    k_particles(pi, particle_var::ux)
  #define p_uy    k_particles(pi, particle_var::uy)
  #define p_uz    k_particles(pi, particle_var::uz)
  #define p_w     k_particles(pi, particle_var::w)
#ifdef VARIABLE_CHARGE
  #define p_q     k_particles(pi, particle_var::qp)
#endif
  #define pii     k_particles_i(pi)

  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, q;
  float w0, wx, wy, wz, wmx, wmy, wmz;
  int axis, face;
  int64_t neighbor;
  size_t pi = pm->i;

#ifdef VARIABLE_CHARGE
  q = p_q*p_w;
#else
  q = qsp*p_w;
#endif

  for(;;) {
    int ii = pii;
    s_midx = p_dx;
    s_midy = p_dy;
    s_midz = p_dz;

    ux = p_ux;
    uy = p_uy;
    uz = p_uz;
    
    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    float grad_xi_x, grad_xi_y, grad_xi_z;
    float grad_eta_x, grad_eta_y, grad_eta_z;
    float grad_mu_x, grad_mu_y, grad_mu_z;
    float jac;

    compute_reciprocal_basis(
        g,
        s_midx, s_midy, s_midz, ii, nx, ny, nz,
        gdx, gdy, gdz,
        grad_xi_x, grad_xi_y, grad_xi_z,
        grad_eta_x, grad_eta_y, grad_eta_z,
        grad_mu_x, grad_mu_y, grad_mu_z,
        jac);

    // Transform Cartesian velocities to contravariant logical velocities
    float d_xi_dt  = ux * grad_xi_x  + uy * grad_xi_y  + uz * grad_xi_z;
    float d_eta_dt = ux * grad_eta_x + uy * grad_eta_y + uz * grad_eta_z;
    float d_mu_dt  = ux * grad_mu_x  + uy * grad_mu_y  + uz * grad_mu_z;

    // Find position of particle at t_n+1/2
    v0 = (ux==0) ? 0.0 : s_dispx/d_xi_dt/gdt*2.;
    v1 = (uy==0) ? 0.0 : s_dispy/d_eta_dt/gdt*2.;
    v2 = (uz==0) ? 0.0 : s_dispz/d_mu_dt/gdt*2.;

    fracdt = v0;
    if(v1>fracdt) fracdt=v1;
    if(v2>fracdt) fracdt=v2;
    fracdt = 2.0*(fracdt-0.5);

    if(fracdt>0){

      x_half = s_midx + fracdt*d_xi_dt*gdt/2.;
      y_half = s_midy + fracdt*d_eta_dt*gdt/2.; 
      z_half = s_midz + fracdt*d_mu_dt*gdt/2.;
      
      if( x_half<=one &&  y_half<=one &&  z_half<=one && 
         -x_half<=one && -y_half<=one && -z_half<=one) {
        
#ifdef SHAPE_NGP
        // Compute reciprocal basis vectors at half-step position
        float grad_xi_x, grad_xi_y, grad_xi_z;
        float grad_eta_x, grad_eta_y, grad_eta_z;
        float grad_mu_x, grad_mu_y, grad_mu_z;
        float jac;
        
        compute_reciprocal_basis(
            g,
            x_half, y_half, z_half, ii, nx, ny, nz,
            gdx, gdy, gdz,
            grad_xi_x, grad_xi_y, grad_xi_z,
            grad_eta_x, grad_eta_y, grad_eta_z,
            grad_mu_x, grad_mu_y, grad_mu_z,
            jac);
        
        // Transform Cartesian velocity to contravariant logical velocity
        float d_xi_dt  = ux * grad_xi_x  + uy * grad_xi_y  + uz * grad_xi_z;
        float d_eta_dt = ux * grad_eta_x + uy * grad_eta_y + uz * grad_eta_z;
        float d_mu_dt  = ux * grad_mu_x  + uy * grad_mu_y  + uz * grad_mu_z;
        
        // Compute proper volume scaling (inverse Jacobian)
        float inv_jac = 1.0f / jac;
        
        // Deposit contravariant logical current with proper volume weighting
        // Note: Factor of 0.125 = 1/8 accounts for the cell volume normalization
        // in logical coordinates (cell spans [-1,1]^3 = volume of 8)
        float qfactor = q * 0.125f * inv_jac;
        
        k_jf_accum(ii, accumulator_var::jx)  += qfactor * d_xi_dt;
        k_jf_accum(ii, accumulator_var::jy)  += qfactor * d_eta_dt;
        k_jf_accum(ii, accumulator_var::jz)  += qfactor * d_mu_dt;
        k_jf_accum(ii, accumulator_var::rho) += qfactor;

#elif defined( SHAPE_QS )
        // stencil coefficients
        w0 =  q*rV12*2.f*( 3.f - x_half*x_half - y_half*y_half - z_half*z_half );
        wx =  q*rV12*( x_half + 1.f )*( x_half + 1.f );
        wy =  q*rV12*( y_half + 1.f )*( y_half + 1.f );
        wz =  q*rV12*( z_half + 1.f )*( z_half + 1.f );
        wmx = q*rV12*( x_half - 1.f )*( x_half - 1.f );
        wmy = q*rV12*( y_half - 1.f )*( y_half - 1.f );
        wmz = q*rV12*( z_half - 1.f )*( z_half - 1.f );

        // Voxel indices
        int iii = ii;
        int zi = iii/((nx+2)*(ny+2));
        iii -= zi*(nx+2)*(ny+2);
        int yi = iii/(nx+2);
        int xi = iii-yi*(nx+2);
        // Neighboring voxel 1D (flattened) indices
        int iix = VOXEL(xi+1,yi,zi,nx,ny,nz);
        int iiy = VOXEL(xi,yi+1,zi,nx,ny,nz);
        int iiz = VOXEL(xi,yi,zi+1,nx,ny,nz);
        int iimx = VOXEL(xi-1,yi,zi,nx,ny,nz);
        int iimy = VOXEL(xi,yi-1,zi,nx,ny,nz);
        int iimz = VOXEL(xi,yi,zi-1,nx,ny,nz);

        k_jf_accum(ii, accumulator_var::jx)  += w0*ux;
        k_jf_accum(ii, accumulator_var::jy)  += w0*uy;
        k_jf_accum(ii, accumulator_var::jz)  += w0*uz;
        k_jf_accum(ii, accumulator_var::rho) += w0;

        k_jf_accum(iix, accumulator_var::jx)  += wx*ux;
        k_jf_accum(iix, accumulator_var::jy)  += wx*uy;
        k_jf_accum(iix, accumulator_var::jz)  += wx*uz;
        k_jf_accum(iix, accumulator_var::rho) += wx;

        k_jf_accum(iiy, accumulator_var::jx)  += wy*ux;
        k_jf_accum(iiy, accumulator_var::jy)  += wy*uy;
        k_jf_accum(iiy, accumulator_var::jz)  += wy*uz;
        k_jf_accum(iiy, accumulator_var::rho) += wy;

        k_jf_accum(iiz, accumulator_var::jx)  += wz*ux;
        k_jf_accum(iiz, accumulator_var::jy)  += wz*uy;
        k_jf_accum(iiz, accumulator_var::jz)  += wz*uz;
        k_jf_accum(iiz, accumulator_var::rho) += wz;

        k_jf_accum(iimx, accumulator_var::jx)  += wmx*ux;
        k_jf_accum(iimx, accumulator_var::jy)  += wmx*uy;
        k_jf_accum(iimx, accumulator_var::jz)  += wmx*uz;
        k_jf_accum(iimx, accumulator_var::rho) += wmx;

        k_jf_accum(iimy, accumulator_var::jx)  += wmy*ux;
        k_jf_accum(iimy, accumulator_var::jy)  += wmy*uy;
        k_jf_accum(iimy, accumulator_var::jz)  += wmy*uz;
        k_jf_accum(iimy, accumulator_var::rho) += wmy;

        k_jf_accum(iimz, accumulator_var::jx)  += wmz*ux;
        k_jf_accum(iimz, accumulator_var::jy)  += wmz*uy;
        k_jf_accum(iimz, accumulator_var::jz)  += wmz*uz;
        k_jf_accum(iimz, accumulator_var::rho) += wmz;
#endif // defined(SHAPE_QS)

      } //if inbds
      
    } //if more than half dt left

    s_dir[0] = (s_dispx>0) ? 1 : -1;
    s_dir[1] = (s_dispy>0) ? 1 : -1;
    s_dir[2] = (s_dispz>0) ? 1 : -1;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak
    /**/      v3=2,  axis=3;
    if(v0<v3) v3=v0, axis=0;
    if(v1<v3) v3=v1, axis=1;
    if(v2<v3) v3=v2, axis=2;
    v3 *= 0.5;

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;

    // Compute the remaining particle displacement
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    // Compute the new particle offset
    p_dx += s_dispx+s_dispx;
    p_dy += s_dispy+s_dispy;
    p_dz += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)

    if( axis==3 ) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.

    v0 = s_dir[axis];
    k_particles(pi, particle_var::dx + axis) = v0;
    face = axis; if( v0>0 ) face += 3;

    neighbor = d_neighbor( 6*ii + face );

    if( neighbor==reflect_particles ) {
      // Hit a reflecting boundary condition
      k_particles(pi, particle_var::ux + axis) = -k_particles(pi, particle_var::ux + axis);
      float* disp = static_cast<float*>(&(pm->dispx));
      disp[axis] = -disp[axis];
      continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
      // Cannot handle the boundary condition here
      pii = 8*pii + face;
      return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel
    pii = neighbor - rangel;
    k_particles(pi, particle_var::dx + axis) = -v0;
  }
  
  #undef p_dx
  #undef p_dy
  #undef p_dz
  #undef p_ux
  #undef p_uy
  #undef p_uz
  #undef p_w
#ifdef VARIABLE_CHARGE
  #undef p_q
#endif
  #undef pii

  return 0; // Return "mover not in use"
}

// TODO: this bascially duplicates funcitonality in rho_p.cc and should be DRY'd
template<typename kf_t, typename kp_t, typename kpi_t> // k_field_t, k_particles_t, k_particles_i_t
void k_accumulate_rhob_single_cpu(
        kf_t& k_rhob_accum,
        kp_t& kpart,
        kpi_t& kpart_i,
        const int i,
        const grid_t* g,
        const float qsp
)
{
    // Extract grid vars
    const float r8V = g->r8V;
    const int nx = g->nx;
    const int ny = g->ny;
    const int nz = g->nz;
    const int sy = g->sy;
    const int sz = g->sz;

    // Kernel
    //float w0 = p->dx, w1 = p->dy, w2, w3, w4, w5, w6, w7, dz = p->dz;
    //int v = p->i, x, y, z, sy = g->sy, sz = g->sz;
    //w7 = (qsp*g->r8V)*p->w;
    float w0 = kpart(i, particle_var::dx);
    float w1 = kpart(i, particle_var::dy);
    float w7 = (qsp * r8V) * kpart(i, particle_var::w);
    float dz = kpart(i, particle_var::dz);
    int v = kpart_i(i);
    //printf("\n Vars are %g, %g, %g %g\n", w0, w1, w7, dz);

    float w6 = w7 - w0 * w7;
    w7 = w7 + w0 * w7;
    float w4 = w6 - w1 * w6;
    float w5 = w7 - w1 * w7;
    w6 = w6 + w1 * w6;
    w7 = w7 + w1 * w7;
    w0 = w4 - dz * w4;
    w1 = w5 - dz * w5;
    float w2 = w6 - dz * w6;
    float w3 = w7 - dz * w7;
    w4 = w4 + dz * w4;
    w5 = w5 + dz * w5;
    w6 = w6 + dz * w6;
    w7 = w7 + dz * w7;

    int x = v;
    int z = x/sz;
    if(z == 1) {
        w0 += w0;
        w1 += w1;
        w2 += w2;
        w3 += w3;
    }
    if(z == nz) {
        w4 += w4;
        w5 += w5;
        w6 += w6;
        w7 += w7;
    }
    x -= sz * z;
    int y = x/sy;
    if(y == 1) {
        w0 += w0;
        w1 += w1;
        w4 += w4;
        w5 += w5;
    }
    if(y == ny) {
        w2 += w2;
        w3 += w3;
        w6 += w6;
        w7 += w7;
    }
    x -= sy * y;
    if(x == 1) {
        w0 += w0;
        w2 += w2;
        w4 += w4;
        w6 += w6;
    }
    if(x == nx) {
        w1 += w1;
        w3 += w3;
        w5 += w5;
        w7 += w7;
    }
    //printf("Absorbing %d into %d for %e %e %e %e %e %e %e %e \n", i, v, w0, w1, w2, w3, w4, w5, w6, w7);
    // Save the bound charge to an accumulator array to be added to rhob on the
    // device later
    k_rhob_accum(v) += w0;
    k_rhob_accum(v+1) += w1;
    k_rhob_accum(v+sy) += w2;
    k_rhob_accum(v+sy+1) += w3;
    k_rhob_accum(v+sz) += w4;
    k_rhob_accum(v+sz+1) += w5;
    k_rhob_accum(v+sz+sy) += w6;
    k_rhob_accum(v+sz+sy+1) += w7;
}

#endif // _species_advance_h_