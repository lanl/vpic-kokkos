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
#include "../vpic/kokkos_simd_extensions.h"
#include "Kokkos_DualView.hpp"

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
} particle_t;

// WARNING: FUNCTIONS THAT USE A PARTICLE_MOVER ASSUME THAT EVERYBODY
// WHO USES THAT PARTICLE MOVER WILL HAVE ACCESS TO PARTICLE ARRAY

typedef struct particle_mover {
  float dispx, dispy, dispz; // Displacement of particle
  int32_t i;                 // Index of the particle to move
} particle_mover_t;

// NOTE: THE LAYOUT OF A PARTICLE_INJECTOR _MUST_ BE COMPATIBLE WITH
// THE CONCATENATION OF A PARTICLE_T AND A PARTICLE_MOVER!

typedef struct particle_injector {
  float dx, dy, dz;          // Particle position in cell coords (on [-1,1])
  int32_t i;                 // Index of cell containing the particle
  float ux, uy, uz;          // Particle normalized momentum
  float w;                   // Particle weight (number of physical particles)
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

} pb_diagnostic_t;

class species_t {
    public:

        char * name;                        // Species name
        float q;                            // Species particle charge
        float m;                            // Species particle rest mass

        int np = 0, max_np = 0;             // Number and max local particles
        particle_t * ALIGNED(128) p;        // Array of particles for the species

        // TODO: these could be unsigned?
        int nm = 0, max_nm = 0;             // Number and max local movers in use

        particle_mover_t * ALIGNED(128) pm; // Particle movers

        int64_t last_sorted;                // Step when the particles were last
        // sorted.
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

        grid_t * g;                         // Underlying grid
        species_id id;                      // Unique identifier for a species
        species_t* next = NULL;             // Next species in the list

        // Particle boundary diagnostic.
        pb_diagnostic_t * pb_diag = NULL;


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

        // TODO: this should ultimatley be removeable.
        // This tracks the number of particles we need to move back to the device
        // And is basically the same as nm at certain times?
        int num_to_copy = 0;

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
        Kokkos::View<int*> unsafe_index;
        Kokkos::View<int> clean_up_to_count;
        Kokkos::View<int> clean_up_from_count;
        Kokkos::View<int>::HostMirror clean_up_from_count_h;
        Kokkos::View<int*> clean_up_from;
        Kokkos::View<int*> clean_up_to;

        // Init Kokkos Particle Arrays
        species_t(int n_particles, int n_pmovers)
        {
           init_kokkos_particles(n_particles, n_pmovers);
        }

        void init_kokkos_particles()
        {
            init_kokkos_particles(max_np, max_nm);
        }
        void init_kokkos_particles(int n_particles, int n_pmovers)
        {
#ifdef TILED
            auto ntiles = n_particles / 64;
            if(ntiles*64 < n_particles)
              ntiles += 1;
printf("SIMD_LEN: %d, Particle vars: %d, ntiles: %d\n", 64, PARTICLE_VAR_COUNT, ntiles);
            k_p_d = k_particles_t("k_particles", 64, PARTICLE_VAR_COUNT, ntiles);
#else
            k_p_d = k_particles_t("k_particles", n_particles);
#endif
            k_p_i_d = k_particles_i_t("k_particles_i", n_particles);
            k_pc_d = k_particle_copy_t("k_particle_copy_for_movers", n_pmovers);
            k_pc_i_d = k_particle_i_copy_t("k_particle_copy_for_movers_i", n_pmovers);
            k_pr_h = k_particle_copy_t::HostMirror("k_particle_send_for_movers", n_pmovers);
            k_pr_i_h = k_particle_i_copy_t::HostMirror("k_particle_send_for_movers_i", n_pmovers);
            k_pm_d = k_particle_movers_t("k_particle_movers", n_pmovers);
            k_pm_i_d = k_particle_i_movers_t("k_particle_movers_i", n_pmovers);
            k_nm_d = k_counter_t("k_nm"); // size 1 encoded in type
            unsafe_index = Kokkos::View<int*>("safe index", 2*n_pmovers);
            clean_up_to_count = Kokkos::View<int>("clean up to count");
            clean_up_from_count = Kokkos::View<int>("clean up from count");
            clean_up_from = Kokkos::View<int*>("clean up from", n_pmovers);
            clean_up_to = Kokkos::View<int*>("clean up to", n_pmovers);

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
         int max_local_np,
         int max_local_nm,
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

void
accumulate_rho_p( /**/  field_array_t * RESTRICT fa,
                  const species_t     * RESTRICT sp );

void
accumulate_rhob( field_t          * RESTRICT ALIGNED(128) f,
                 const particle_t * RESTRICT ALIGNED(32)  p,
                 const grid_t     * RESTRICT              g,
                 const float                              qsp );
void
k_accumulate_rho_p( /**/  field_array_t * RESTRICT fa,
                  const species_t     * RESTRICT sp );

void k_accumulate_rhob(
            k_field_t& kfield,
            k_particles_t& kpart,
            k_particle_movers_t& kpart_movers,
            const grid_t* RESTRICT g,
            const float qsp,
            const int nm);

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
        k_hydro_d_t k_hydro,
        k_interpolator_t& k_interp,
        const species_t            * RESTRICT sp
);

template<class CurrentView, typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION 
accumulate_simd(CurrentView& current, size_t active_lanes, SIMDFloatMask mask,
                SIMDInt32& ii,  const int nx,  const int ny,  const int nz,
                SIMDFloat& v0,  SIMDFloat& v1,  SIMDFloat& v2,  SIMDFloat& v3,
                SIMDFloat& v4,  SIMDFloat& v5,  SIMDFloat& v6,  SIMDFloat& v7,
                SIMDFloat& v8,  SIMDFloat& v9,  SIMDFloat& v10, SIMDFloat& v11
)
{
#ifdef VPIC_ENABLE_ACCUMULATORS 
  namespace KokkosSIMD = Kokkos::Experimental;
  using element_aligned_tag_t = KokkosSIMD::element_aligned_tag;
  SIMDFloat v12 = 0.0, v13 = 0.0, v14 = 0.0, v15 = 0.0;
  SIMDFloat* v[16];
  v[0]  = &v0;  v[1]  = &v1;  v[2]  = &v2;  v[3]  = &v3;
  v[4]  = &v4;  v[5]  = &v5;  v[6]  = &v6;  v[7]  = &v7;
  v[8]  = &v8;  v[9]  = &v9;  v[10] = &v10; v[11] = &v11;
  v[12] = &v12; v[13] = &v13; v[14] = &v14; v[15] = &v15;

  // Determine whether all particles are in the same cell
  SIMDInt32 temp = ii[0];
  auto same = temp == ii;

  // Accumulators are LayoutLeft (Column major)
  if constexpr (std::is_same<Kokkos::LayoutLeft, typename CurrentView::array_layout>::value) {
    if((SIMDInt32::size() == active_lanes) && KokkosSIMD::all_of(same)) {
      for(int i=0; i<12; i++) {
        current(ii[0],  i) += KokkosSIMD::reduce(KokkosSIMD::where(mask,  *v[i]), std::plus{}); 
      }
    } else {
      for(int i=0; i<12; i++) {
        KokkosSIMD::where(mask, *v[i]).scatter_to( &(current(0, i)), ii);
      }
    }
  } else if constexpr (std::is_same<Kokkos::LayoutRight, typename CurrentView::array_layout>::value) {
    // Accumulators are LayoutRight (Row major)
    if constexpr(SIMD_LEN == 16) {
      // All particles in the same cell
      if((SIMDInt32::size() == active_lanes) && KokkosSIMD::all_of(same)) {
        transpose(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
                   v0 += v1;  v0 += v2;  v0 += v3;
        v0 += v4;  v0 += v5;  v0 += v6;  v0 += v7;
        v0 += v8;  v0 += v9;  v0 += v10; v0 += v11;
        v0 += v12; v0 += v13; v0 += v14; v0 += v15;
        increment(&(current(ii[0], 0)), v0);
      } else { // Particles belong to different cells
        transpose(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
        for(int i=0; i<active_lanes; i++) {
          increment( &(current(ii[ i], 0)), *v[i] );
        }
      }
    } else if constexpr (SIMD_LEN == 8) {
      // All particles in the same cell
      if((SIMDInt32::size() == active_lanes) && KokkosSIMD::all_of(same)) {
        transpose(v0, v1, v2, v3, v4, v5, v6, v7);
                   v0 += v1;  v0 += v2;  v0 += v3;
        v0 += v4;  v0 += v5;  v0 += v6;  v0 += v7;
        increment(&(current(ii[0], 0)), v0);
        transpose(v8, v9, v10, v11, v12, v13, v14, v15);
                   v8 += v9;  v8 += v10; v8 += v11;
        v8 += v12; v8 += v13; v8 += v14; v8 += v15;
        increment(&(current(ii[0], 8)), v8);
      } else { // Particles belong to different cells
        transpose(v0, v1, v2, v3, v4, v5, v6, v7);
        for(int i=0; i<active_lanes; i++) {
          increment( &(current(ii[ i], 0)), *v[i] );
        }
        transpose(v8, v9, v10, v11, v12, v13, v14, v15);
        for(int i=0; i<active_lanes; i++) {
          increment( &(current(ii[ i], 8)), *v[i+8] );
        }
      }
    } else if constexpr (SIMD_LEN == 4) {
      if((SIMDInt32::size() == active_lanes) && KokkosSIMD::all_of(same)) {
        transpose(v0, v1, v2, v3);
        v0 += v1;  v0 += v2;  v0 += v3;
        increment(&(current((int)ii[0], 0)), v0);
        transpose(v4, v5, v6, v7);
        v4 += v5;  v4 += v6;  v4 += v7;
        increment(&(current((int)ii[0], 4)), v4);
        transpose(v8, v9, v10, v11);
        v8 += v9;  v8 += v10; v8 += v11;
        increment(&(current((int)ii[0], 8)), v8);
      } else { // Particles belong to different cells
        transpose(v0, v1, v2, v3);
        for(int i=0; i<active_lanes; i++) {
          increment( &(current((int)ii[ i], 0)), *(v[i]) );
        }
        transpose(v4, v5, v6, v7);
        for(int i=0; i<active_lanes; i++) {
          increment( &(current((int)ii[ i], 4)), *(v[i+4]) );
        }
        transpose(v8, v9, v10, v11);
        for(int i=0; i<active_lanes; i++) {
          increment( &(current((int)ii[ i], 8)), *(v[i+8]) );
        }
      }
    } else {
      for(size_t idx=0; idx<active_lanes; idx++) {
        current((int)ii[idx], 0)  += v0[idx];
        current((int)ii[idx], 1)  += v1[idx];
        current((int)ii[idx], 2)  += v2[idx];
        current((int)ii[idx], 3)  += v3[idx];
        current((int)ii[idx], 4)  += v4[idx];
        current((int)ii[idx], 5)  += v5[idx];
        current((int)ii[idx], 6)  += v6[idx];
        current((int)ii[idx], 7)  += v7[idx];
        current((int)ii[idx], 8)  += v8[idx];
        current((int)ii[idx], 9)  += v9[idx];
        current((int)ii[idx], 10) += v10[idx];
        current((int)ii[idx], 11) += v11[idx];
      }
    }
  }
#else
  // Update current
  for(size_t idx=0; idx<active_lanes; idx++) {
    int iii = ii[idx];
    int zi = iii/((nx+2)*(ny+2));
    iii -= zi*(nx+2)*(ny+2);
    int yi = iii/(nx+2);
    int xi = iii - yi*(nx+2);
    
    current((int)ii[idx], field_var::jfx)                      += v0[idx];
    current(VOXEL(xi,yi+1,zi,nx,ny,nz),   field_var::jfx) += v1[idx];
    current(VOXEL(xi,yi,zi+1,nx,ny,nz),   field_var::jfx) += v2[idx];
    current(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += v3[idx];
    
    current((int)ii[idx], field_var::jfy)                      += v4[idx];
    current(VOXEL(xi,yi,zi+1,nx,ny,nz),   field_var::jfy) += v5[idx];
    current(VOXEL(xi+1,yi,zi,nx,ny,nz),   field_var::jfy) += v6[idx];
    current(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += v7[idx];
    
    current((int)ii[idx], field_var::jfz)                      += v8[idx];
    current(VOXEL(xi+1,yi,zi,nx,ny,nz),   field_var::jfz) += v9[idx];
    current(VOXEL(xi,yi+1,zi,nx,ny,nz),   field_var::jfz) += v10[idx];
    current(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += v11[idx];
  }
#endif
}

// In move_p.cxx
int
move_p( particle_t       * ALIGNED(128) p0,
        particle_mover_t * ALIGNED(16)  pm,
        //accumulator_t    * ALIGNED(128) a0,
        k_jf_accum_t::HostMirror& k_jf_accum,
        const grid_t     *              g,
        const float                     qsp );

template<class particle_view_t, class particle_i_view_t, typename scatter_view_t, class neighbor_view_t>
simd_float_mask_t
KOKKOS_INLINE_FUNCTION
move_p_kokkos_simd_a(
    const particle_view_t& k_particles,
    const particle_i_view_t& k_particles_i,
    simd_float_t& dispx,
    simd_float_t& dispy,
    simd_float_t& dispz,
    simd_int32_t& pm_i,
    simd_float_mask_t mask,
    size_t active_lanes,
    scatter_view_t& scatter_access,
    const grid_t* g,
    neighbor_view_t& d_neighbor,
    int64_t rangel,
    int64_t rangeh,
    const float qsp,
    float cx, float cy, float cz,
    const int nx, const int ny, const int nz
)
{
  simd_float_mask_t lane_mask([active_lanes](std::size_t i) { return i < active_lanes; });
  simd_float_mask_t res(false), not_done=mask && lane_mask;;
  const simd_float_t one_third = (1./3.);
  simd_float_t s_midx, s_midy, s_midz;
  simd_float_t s_dispx, s_dispy, s_dispz;
  simd_float_t dx, dy, dz;
  simd_float_t v0, v1, v2, v3, v4, v5, q;
  simd_float_t v6, v7, v8, v9, v10, v11, v12, v13;
  simd_float_t s_dir[3];
  simd_float_t r[3], dr[3];
  simd_int32_t axis, face;
  simd_int32_t ii([k_particles_i, pm_i, not_done](int i){if(not_done[i]) return k_particles_i(pm_i[i]); else return 0;});
  int64_t neighbor;

  float* mem_dx = &(k_particles(0, particle_var::dx));
  float* mem_dy = &(k_particles(0, particle_var::dy));
  float* mem_dz = &(k_particles(0, particle_var::dz));
  float* mem_ux = &(k_particles(0, particle_var::ux));
  float* mem_uy = &(k_particles(0, particle_var::uy));
  float* mem_uz = &(k_particles(0, particle_var::uz));
  float* mem_w  = &(k_particles(0, particle_var::w ));
  int*   mem_ii = &(k_particles_i(0));

  simd_int32_mask_t not_done_int([not_done](int i){return not_done[i];});
  KokkosSIMD::where(not_done_int, ii).gather_from(mem_ii, pm_i);
  if constexpr (std::is_same<Kokkos::LayoutLeft, k_particles_t::array_layout>::value) {
    KokkosSIMD::where(not_done, r[0]).gather_from(mem_dx, pm_i);
    KokkosSIMD::where(not_done, r[1]).gather_from(mem_dy, pm_i);
    KokkosSIMD::where(not_done, r[2]).gather_from(mem_dz, pm_i);
    KokkosSIMD::where(not_done, q).gather_from(mem_w, pm_i);
  } else {
    simd_int32_t indices([pm_i](std::size_t i) { return pm_i[i]*PARTICLE_VAR_COUNT; });
    KokkosSIMD::where(not_done, r[0]).gather_from(mem_dx, indices);
    KokkosSIMD::where(not_done, r[1]).gather_from(mem_dy, indices);
    KokkosSIMD::where(not_done, r[2]).gather_from(mem_dz, indices);
    KokkosSIMD::where(not_done, q).gather_from(mem_w, indices);
  }
//  q = qsp*p_w;
  KokkosSIMD::where(not_done, q) = q*qsp;
  dr[0] = dispx;
  dr[1] = dispy;
  dr[2] = dispz;


  while(KokkosSIMD::any_of(not_done)) {
//  int pi = pm_i[idx];
//    int ii = pii;
//    s_midx = p_dx;
//    s_midy = p_dy;
//    s_midz = p_dz;
    s_midx = r[0];
    s_midy = r[1];
    s_midz = r[2];

//    s_dispx = dispx[idx];
//    s_dispy = dispy[idx];
//    s_dispz = dispz[idx];
    s_dispx = dr[0];
    s_dispy = dr[1];
    s_dispz = dr[2];

    //printf("pre axis %d x %e y %e z %e \n", axis, p_dx, p_dy, p_dz);

    //printf("disp x %e y %e z %e \n", s_dispx, s_dispy, s_dispz);

//    s_dir[0] = (s_dispx>0) ? 1 : -1;
//    s_dir[1] = (s_dispy>0) ? 1 : -1;
//    s_dir[2] = (s_dispz>0) ? 1 : -1;
    s_dir[0] = -1;
    s_dir[1] = -1;
    s_dir[2] = -1;
    KokkosSIMD::where(s_dispx>0, s_dir[0]) = 1;
    KokkosSIMD::where(s_dispy>0, s_dir[1]) = 1;
    KokkosSIMD::where(s_dispz>0, s_dir[2]) = 1;
//    s_dir[0] = KokkosSIMD::condition(s_dispx>0, 1, -1);
//    s_dir[1] = KokkosSIMD::condition(s_dispy>0, 1, -1);
//    s_dir[2] = KokkosSIMD::condition(s_dispz>0, 1, -1);

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
//    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
//    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
//    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;
    v0 = (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dir[2]-s_midz)/s_dispz;
    KokkosSIMD::where(s_dispx==0, v0) = 3.4e38f;
    KokkosSIMD::where(s_dispy==0, v1) = 3.4e38f;
    KokkosSIMD::where(s_dispz==0, v2) = 3.4e38f;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
//    /**/      v3=2,  axis=3;
//    if(v0<v3) v3=v0, axis=0;
//    if(v1<v3) v3=v1, axis=1;
//    if(v2<v3) v3=v2, axis=2;
//    v3 *= 0.5;
    v3=2.0f; axis=simd_int32_t(3);
    simd_int32_mask_t axis_mask([v0,v3](int i){return v0[i]<v3[i];});
    KokkosSIMD::where(axis_mask, axis) = 0;
    KokkosSIMD::where(v0<v3, v3) = v0;
    axis_mask = simd_int32_mask_t([v1,v3](int i){return v1[i]<v3[i];});
    KokkosSIMD::where(axis_mask, axis) = 1;
    KokkosSIMD::where(v1<v3, v3) = v1;
    axis_mask = simd_int32_mask_t([v2,v3](int i){return v2[i]<v3[i];});
    KokkosSIMD::where(axis_mask, axis) = 2;
    KokkosSIMD::where(v2<v3, v3) = v2;
    v3 *= 0.5f;

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
//    v5 = q*s_dispx*s_dispy*s_dispz*(1.f/3.f);
    v13 = q*s_dispx*s_dispy*s_dispz*one_third;

    //a = (float *)(&d_accumulators[ci]);

#   define accumulate_j(X,Y,Z,v0,v1,v2,v3)                                        \
    v12 = q*s_disp##X;     /* v2 = q ux                            */  \
    v1  = v12*s_mid##Y;    /* v1 = q ux dy                         */  \
    v0  = v12-v1;          /* v0 = q ux (1-dy)                     */  \
    v1 += v12;             /* v1 = q ux (1+dy)                     */  \
    v12 = 1+s_mid##Z;      /* v4 = 1+dz                            */  \
    v2  = v0*v12;          /* v2 = q ux (1-dy)(1+dz)               */  \
    v3  = v1*v12;          /* v3 = q ux (1+dy)(1+dz)               */  \
    v12  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
    v0 *= v12;             /* v0 = q ux (1-dy)(1-dz)               */  \
    v1 *= v12;             /* v1 = q ux (1+dy)(1-dz)               */  \
    v0 += v13;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
    v1 -= v13;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
    v2 -= v13;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
    v3 += v13;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \

    accumulate_j(x,y,z,v0,v1,v2,v3);
    v0  *= cx; v1  *= cx; v2  *= cx; v3  *= cx;
    accumulate_j(y,z,x,v4,v5,v6,v7);
    v4  *= cy; v5  *= cy; v6  *= cy; v7  *= cy;
    accumulate_j(z,x,y,v8,v9,v10,v11);
    v8  *= cz; v9  *= cz; v10 *= cz; v11 *= cz;
//    accumulate_simd(scatter_access, active_lanes, not_done,
//                    ii,  nx,  ny,  nz,
//                    v0,  v1,  v2,  v3,
//                    v4,  v5,  v6,  v7,
//                    v8,  v9,  v10, v11);

#if defined( VPIC_ENABLE_ACCUMULATORS )
//    simd_float_t v12 = 0.0, v13 = 0.0, v14 = 0.0, v15 = 0.0;
//    transpose(v0, v1, v2, v3, v4, v5, v6, v7);
//    transpose(v8, v9, v10, v11, v12, v13, v14, v15);
//    if(not_done[0]) {
//      increment(&(scatter_access(ii[ 0], 0)), v0);
//      increment(&(scatter_access(ii[ 0], 8)), v8);
//    }
//    if(not_done[1]) {
//      increment(&(scatter_access(ii[ 1], 0)), v1);
//      increment(&(scatter_access(ii[ 1], 8)), v9);
//    }
//    if(not_done[2]) {
//      increment(&(scatter_access(ii[ 2], 0)), v2);
//      increment(&(scatter_access(ii[ 2], 8)), v10);
//    }
//    if(not_done[3]) {
//      increment(&(scatter_access(ii[ 3], 0)), v3);
//      increment(&(scatter_access(ii[ 3], 8)), v11);
//    }
//    if(not_done[4]) {
//      increment(&(scatter_access(ii[ 4], 0)), v4);
//      increment(&(scatter_access(ii[ 4], 8)), v12);
//    }
//    if(not_done[5]) {
//      increment(&(scatter_access(ii[ 5], 0)), v5);
//      increment(&(scatter_access(ii[ 5], 8)), v13);
//    }
//    if(not_done[6]) {
//      increment(&(scatter_access(ii[ 6], 0)), v6);
//      increment(&(scatter_access(ii[ 6], 8)), v14);
//    }
//    if(not_done[7]) {
//      increment(&(scatter_access(ii[ 7], 0)), v7);
//      increment(&(scatter_access(ii[ 7], 8)), v15);
//    }

    for(size_t idx=0; idx<active_lanes; idx++) {
      if(not_done[idx]) {
        scatter_access(ii[idx], 0)  += v0[idx];
        scatter_access(ii[idx], 1)  += v1[idx];
        scatter_access(ii[idx], 2)  += v2[idx];
        scatter_access(ii[idx], 3)  += v3[idx];
        scatter_access(ii[idx], 4)  += v4[idx];
        scatter_access(ii[idx], 5)  += v5[idx];
        scatter_access(ii[idx], 6)  += v6[idx];
        scatter_access(ii[idx], 7)  += v7[idx];
        scatter_access(ii[idx], 8)  += v8[idx];
        scatter_access(ii[idx], 9)  += v9[idx];
        scatter_access(ii[idx], 10) += v10[idx];
        scatter_access(ii[idx], 11) += v11[idx];
      }
    }
#else
    for(size_t idx=0; idx<active_lanes; idx++) {
      if(not_done[idx]) {
        int iii = ii[idx];
        int zi = iii/((nx+2)*(ny+2));
        iii -= zi*(nx+2)*(ny+2);
        int yi = iii/(nx+2);
        int xi = iii-yi*(nx+2);
        scatter_access(ii[idx], field_var::jfx)                      += v0[idx];
        scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx)   += v1[idx];
        scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx)   += v2[idx];
        scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += v3[idx];
    
        scatter_access(ii[idx], field_var::jfy)                      += v4[idx];
        scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy)   += v5[idx];
        scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy)   += v6[idx];
        scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += v7[idx];
    
        scatter_access(ii[idx], field_var::jfz)                      += v8[idx];
        scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz)   += v9[idx];
        scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz)   += v10[idx];
        scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += v11[idx];
      }
    }
#endif

#   undef accumulate_j

    // Compute the remaining particle displacment
    dr[0] -= s_dispx;
    dr[1] -= s_dispy;
    dr[2] -= s_dispz;
//    dispx -= s_dispx;
//    dispy -= s_dispy;
//    dispz -= s_dispz;

    //printf("pre axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    // Compute the new particle offset
//    p_dx += s_dispx+s_dispx;
//    p_dy += s_dispy+s_dispy;
//    p_dz += s_dispz+s_dispz;
    r[0] += s_dispx+s_dispx;
    r[1] += s_dispy+s_dispy;
    r[2] += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)
    //printf("axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
//    if( axis==3 ) {
//      not_done[idx] = false;
//      continue;
//    }
//
    simd_float_mask_t temp([axis](int i) {return axis[i] == 3;});
    if constexpr (std::is_same<Kokkos::LayoutLeft, k_particles_t::array_layout>::value) {
      KokkosSIMD::where(temp, r[0]).scatter_to(mem_dx, pm_i);
      KokkosSIMD::where(temp, r[1]).scatter_to(mem_dy, pm_i);
      KokkosSIMD::where(temp, r[2]).scatter_to(mem_dz, pm_i);
//      KokkosSIMD::where(temp, ii).scatter_to(mem_ii, pm_i);
    } else {
      simd_int32_t indices([pm_i](std::size_t i) { return pm_i[i]*PARTICLE_VAR_COUNT; });
      KokkosSIMD::where(temp, r[0]).scatter_to(mem_dx, indices);
      KokkosSIMD::where(temp, r[1]).scatter_to(mem_dy, indices);
      KokkosSIMD::where(temp, r[2]).scatter_to(mem_dz, indices);
//      KokkosSIMD::where(temp, ii).scatter_to(mem_ii, pm_i);
    }
//    KokkosSIMD::where((axis == 3) && not_done, r[0]).scatter_to(mem_dx, pm_i);
//    KokkosSIMD::where((axis == 3) && not_done, r[1]).scatter_to(mem_dy, pm_i);
//    KokkosSIMD::where((axis == 3) && not_done, r[2]).scatter_to(mem_dz, pm_i);
    KokkosSIMD::where((axis == 3) && not_done_int, ii  ).scatter_to(mem_ii, pm_i);
    not_done = not_done && !temp;

//for(size_t idx=0; idx<active_lanes; idx++) {
//  if(not_done[idx]) {
//    if(axis[idx] == 3) {
////      k_particles( pm_i[idx], particle_var::dx) = r[0][idx];
////      k_particles( pm_i[idx], particle_var::dy) = r[1][idx];
////      k_particles( pm_i[idx], particle_var::dz) = r[2][idx];
//      k_particles_i(pm_i[idx]) = ii[idx];
//      not_done[idx] = false;
//      continue;
//    }
//  }
//}

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

//    v0 = s_dir[axis];
//    k_particles( pi, particle_var::dx + axis) = v0; // Avoid roundoff fiascos--put the particle
//                           // _exactly_ on the boundary.
//    face = axis; if( v0>0 ) face += 3;

//    v0[idx] = s_dir[axis[idx]][idx];
//    k_particles( pm_i[idx], particle_var::dx + axis[idx]) = v0[idx];
//    face[idx] = axis[idx]; if(v0[idx]>0) face[idx] += 3;

    simd_float_t axis_float([axis](int i) {return static_cast<float>(axis[i]);});
    KokkosSIMD::where(simd_float_mask_t(axis_float == 0.f), v0) = s_dir[0];
    KokkosSIMD::where(simd_float_mask_t(axis_float == 1.f), v0) = s_dir[1];
    KokkosSIMD::where(simd_float_mask_t(axis_float == 2.f), v0) = s_dir[2];
    face = axis;
    simd_int32_mask_t face_update([v0](int i){return v0[i] > 0;});
    KokkosSIMD::where(face_update, face) = face + 3;
//    KokkosSIMD::where(not_done, v0).scatter_to(mem_dx, pm_i);
//    KokkosSIMD::where(not_done, v0).scatter_to(mem_dy, pm_i);
//    KokkosSIMD::where(not_done, v0).scatter_to(mem_dz, pm_i);

    int64_t neighbors[simd_int32_t::size()];
    int32_t _new_voxel[simd_int32_t::size()];
    for(size_t idx=0; idx<active_lanes; idx++) {
      neighbors[idx] = d_neighbor( 6*ii[idx] + face[idx] );
      _new_voxel[idx] = neighbors[idx] - rangel;
      if(not_done[idx]) {
        k_particles( pm_i[idx], particle_var::dx + axis[idx]) = v0[idx];
      }
    }
    simd_int32_t new_voxel;
    new_voxel.copy_from(_new_voxel, KokkosSIMD::element_aligned_tag());
    simd_float_mask_t reflect([neighbors](int i) {
      return neighbors[i] == reflect_particles;
    });
    simd_float_mask_t boundary([neighbors, rangel, rangeh](int i) {
      return (neighbors[i] < rangel || neighbors[i] > rangeh);
    });
    KokkosSIMD::where(not_done && reflect && axis_float == 0, dr[0]) = -dr[0];
    KokkosSIMD::where(not_done && reflect && axis_float == 1, dr[1]) = -dr[1];
    KokkosSIMD::where(not_done && reflect && axis_float == 2, dr[2]) = -dr[2];
    KokkosSIMD::where(not_done && boundary && axis_float == 0, dispx) = dr[0];
    KokkosSIMD::where(not_done && boundary && axis_float == 1, dispy) = dr[1];
    KokkosSIMD::where(not_done && boundary && axis_float == 2, dispz) = dr[2];
    res = res || (not_done && boundary);
    not_done = not_done || (not_done && boundary);
    simd_int32_mask_t update_voxel([not_done, reflect, boundary](int i){return not_done[i] && !reflect[i] && !boundary[i];});
    KokkosSIMD::where(update_voxel, ii) = new_voxel;
    KokkosSIMD::where(not_done && !reflect && !boundary && axis_float == 0, r[0]) = -r[0];
    KokkosSIMD::where(not_done && !reflect && !boundary && axis_float == 1, r[1]) = -r[1];
    KokkosSIMD::where(not_done && !reflect && !boundary && axis_float == 2, r[2]) = -r[2];
    
//    for(size_t idx=0; idx<active_lanes; idx++) {
//      if(not_done[idx]) {
//        k_particles( pm_i[idx], particle_var::dx + axis[idx]) = v0[idx];
//
//        // TODO: clean this fixed index to an enum
//        //neighbor = g->neighbor[ 6*ii + face ];
//        neighbor = d_neighbor( 6*ii[idx] + face[idx] );
//
//        // TODO: these two if statements used to be marked UNLIKELY,
//        // but that intrinsic doesn't work on GPU.
//        // for performance portability, maybe specialize UNLIKELY
//        // for CUDA mode and put it back
//
//        if( neighbor==reflect_particles ) {
//          // Hit a reflecting boundary condition.  Reflect the particle
//          // momentum and remaining displacement and keep moving the
//          // particle.
//          k_particles( pm_i[idx], particle_var::ux + axis[idx]) = -k_particles( pm_i[idx], particle_var::ux + axis[idx]);
//          // Clearer and works with AMD GPUs
////          float* disp = static_cast<float*>(&(pm->dispx));
////          disp[axis] = -disp[axis];
//          dr[axis[idx]][idx] = -dr[axis[idx]][idx];
//          continue;
//        }
//
//        if( neighbor<rangel || neighbor>rangeh ) {
//          // Cannot handle the boundary condition here.  Save the updated
//          // particle position, face it hit and update the remaining
//          // displacement in the particle mover.
////          pii = 8*pii + face;
////          return 1; // Return "mover still in use"
//
//          k_particles(pm_i[idx], particle_var::dx) = r[0][idx];
//          k_particles(pm_i[idx], particle_var::dy) = r[1][idx];
//          k_particles(pm_i[idx], particle_var::dz) = r[2][idx];
//          k_particles_i(pm_i[idx]) = 8*k_particles_i(pm_i[idx]) + face[idx];
//          dispx[idx] = dr[0][idx];
//          dispy[idx] = dr[1][idx];
//          dispz[idx] = dr[2][idx];
//          res[idx] = true;
//          not_done[idx] = false;
//          continue;
//        }
//
//        // Crossed into a normal voxel.  Update the voxel index, convert the
//        // particle coordinate system and keep moving the particle.
//
////        pii = neighbor - rangel;
////        k_particles_i(pm_i[idx]) = neighbor - rangel;
//        ii[idx] = neighbor - rangel;
//        /**/                         // Note: neighbor - rangel < 2^31 / 6
////        k_particles( pm_i[idx], particle_var::dx + axis[idx]) = -v0[idx];      // Convert coordinate system
//        r[axis[idx]][idx] = -r[axis[idx]][idx];
//      }
//    }
  }

//  #undef p_dx
//  #undef p_dy
//  #undef p_dz
//  #undef p_ux
//  #undef p_uy
//  #undef p_uz
//  #undef p_w
//  #undef pii

//  return 0; // Return "mover not in use"
  return res;
}

template<class particle_view_t, class particle_i_view_t, class neighbor_view_t, typename scatter_view_t>
int
KOKKOS_FUNCTION
move_p_kokkos_simd_b(
    const particle_view_t& k_particles,
    const particle_i_view_t& k_particles_i,
    particle_mover_t* ALIGNED(16)  pm,
    scatter_view_t& scatter_access,
    const grid_t* g,
    neighbor_view_t& d_neighbor,
    int64_t rangel,
    int64_t rangeh,
    const float qsp,
    float cx, float cy, float cz,
    const int nx, const int ny, const int nz
)
{
  #define p_dx    k_particles( pi, particle_var::dx)
  #define p_dy    k_particles( pi, particle_var::dy)
  #define p_dz    k_particles( pi, particle_var::dz)
  #define p_ux    k_particles( pi, particle_var::ux)
  #define p_uy    k_particles( pi, particle_var::uy)
  #define p_uz    k_particles( pi, particle_var::uz)
  #define p_w     k_particles( pi, particle_var::w)
  #define pii     k_particles_i(pi)

  int64_t neighbor;
  int pi = pm->i;

  float f0;
  int axis, face;
  simd_float32x4_t s_mid, s_disp, s_dir;
  simd_float32x4_t v0, v1, v2, v3, v4, v5;
  simd_float32x4_t q(qsp * p_w);
  const simd_float32x4_t one(1.f), large_num(3.4e38f);
  constexpr float one_third = 1.f/3.f;

    //printf("in move %d \n", pi);

  int ii = pii;
  simd_float32x4_t r([k_particles, pi](std::size_t i) {
    if(i==0) {
      return p_dx;
    } else if(i==1) {
      return p_dy;
    } else if(i==2) {
      return p_dz;
    } else {
      return 0.0f;
    }
  }); 
  simd_float32x4_t dr;
  dr.copy_from((float*)(&pm[0]), KokkosSIMD::element_aligned_tag());

  for(;;) {
    // At this point:
    //   r     = current particle position in local voxel coordinates
    //           (note the current voxel is on [-1,1]^3.
    //   dr    = remaining particle displacment
    //           (note: this is in voxel edge lengths!)
    //   voxel = local voxel of particle
    // Thus, in the local coordinate system, it is desired to move the
    // particle through all points in the local coordinate system:
    //   streak_r(s) = r + 2 disp s for s in [0,1]
    //
    // Determine the fractional length and type of current
    // streak made by the particle through this voxel.  The streak
    // ends on either the first voxel face intersected by the
    // particle track or at the end of the particle track.
    //
    // Note: a divide by zero cannot occur below due to the shift of
    // the denominator by tiny.  Also, the shift by tiny is large
    // enough that the divide will never overflow when dr is tiny
    // (|sgn_dr-r|<=2 => 2/tiny = 2e+37 < FLT_MAX = 3.4e38).
    // Likewise, due to speed of light limitations, generally dr
    // cannot get much larger than 1 or so and the numerator, if not
    // zero, can generally never be smaller than FLT_EPS/2.  Thus,
    // likewise, the divide will never underflow either.

    s_mid = r;
    s_disp = dr;
    s_dir = KokkosSIMD::condition(s_disp > 0, 1, -1);

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = KokkosSIMD::condition(s_disp==0, large_num, (s_dir - s_mid)/s_disp);

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
    /**/      
                 f0=2.0f,  axis=3;
    if(v0[0]<f0) f0=v0[0], axis=0;
    if(v0[1]<f0) f0=v0[1], axis=1;
    if(v0[2]<f0) f0=v0[2], axis=2;
    f0 *= 0.5;

    simd_float32x4_t sign_flip([axis](int i){return i==axis ? -1.0f : 1.0f;});

    // Compute the midpoint and the normalized displacement of the streak
    s_disp *= simd_float32x4_t(f0);
    s_mid += s_disp;
    // Compute the remaining particle displacment
    dr -= s_disp;
    // Compute the new particle offset
    r += s_disp + s_disp;

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
    v5 = simd_float32x4_t((float)q[0]*(float)s_disp[0]*s_disp[1]*s_disp[2]*one_third); // q( ux*uy*uz*(1/3) )
    v4  = q*s_disp;                    // v4 = q( ux, uy, uz, D/C)
    v1  = v4*shuffle<1,2,0,3>(s_mid);  // v1 = q( uxdy, uydz, uzdx, D/C )
    v0  = v4-v1;                       // v0 = q( ux(1-dy), uy(1-dz), uz(1-dx), D/C )
    v1 += v4;                          // v1 = q( ux(1+dy), uy(1+dz), uz(1+dx), D/C )
    v4  = one+shuffle<2,0,1,3>(s_mid); // v4 = 1+dz, 1+dx, 1+dy, D/C
    v2  = v0*v4;                       // v2 = q( ux(1-dy)(1+dz), uy(1-dz)(1+dx), uz(1-dx)(1+dy), D/C )
    v3  = v1*v4;                       // v3 = q( ux(1+dy)(1+dz), uy(1+dz)(1+dx), uz(1+dx)(1+dy), D/C )
    v4  = one-shuffle<2,0,1,3>(s_mid); // v4 = 1-dz, 1-dx, 1-dy, D/C
    v0 *= v4;                          // v0 = q( ux(1-dy)(1-dz), uy(1-dz)(1-dx), uz(1-dx)(1-dy), D/C )
    v1 *= v4;                          // v1 = q( ux(1+dy)(1-dz), uy(1+dz)(1-dx), uz(1+dx)(1-dy), D/C )
    v0 += v5;                          // v0 = q( ux( (1-dy)(1-dz) + uy*uz/3 ), uy( (1-dz)(1-dx) + uxuz/3 ), uz( (1-dx)(1-dy) + uxuy/3 ), D/C )
    v1 -= v5;                          // v1 = q( ux( (1+dy)(1-dz) - uy*uz/3 ), uy( (1+dz)(1-dx) - uxuz/3 ), uz( (1+dx)(1-dy) - uxuy/3 ), D/C ) 
    v2 -= v5;                          // v2 = q( ux( (1-dy)(1+dz) - uy*uz/3 ), uy( (1-dz)(1+dx) - uxuz/3 ), uz( (1-dx)(1+dy) - uxuy/3 ), D/C ) 
    v3 += v5;                          // v3 = q( ux( (1+dy)(1+dz) + uy*uz/3 ), uy( (1+dz)(1+dx) + uxuz/3 ), uz( (1+dx)(1+dy) + uxuy/3 ), D/C ) 
    transpose(v0, v1, v2, v3);
    v0 *= cx;
    v1 *= cy;
    v2 *= cz;
    
#ifdef VPIC_ENABLE_ACCUMULATORS 
    increment((float*)(&scatter_access(ii, 0)), v0);
    increment((float*)(&scatter_access(ii, 4)), v1);
    increment((float*)(&scatter_access(ii, 8)), v2);
#else
    int iii = ii;
    int zi = iii/((nx+2)*(ny+2));
    iii -= zi*(nx+2)*(ny+2);
    int yi = iii/(nx+2);
    int xi = iii-yi*(nx+2);

    scatter_access(ii, field_var::jfx) += v0[0];
    scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += v0[1];
    scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += v0[2];
    scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += v0[3];

    scatter_access(ii, field_var::jfy) += v1[0];
    scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += v1[1];
    scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += v1[2];
    scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += v1[3];

    scatter_access(ii, field_var::jfz) += v2[0];
    scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += v2[1];
    scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += v2[2];
    scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += v2[3];
#endif

#   undef accumulate_j

    // If an end streak, return success (should be ~50% of the time)
    if( axis==3 ) {
      p_dx = r[0]; p_dy = r[1]; p_dz = r[2];
      pii = ii;
      break;
    }

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.
    f0 = s_dir[axis];
    k_particles( pi, particle_var::dx + axis) = f0; // Avoid roundoff fiascos--put the particle
    
    // _exactly_ on the boundary.
    face = axis; if( f0>0 ) face += 3;

    // TODO: clean this fixed index to an enum
    //neighbor = g->neighbor[ 6*ii + face ];
    neighbor = d_neighbor( 6*ii + face );

    // TODO: these two if statements used to be marked UNLIKELY,
    // but that intrinsic doesn't work on GPU.
    // for performance portability, maybe specialize UNLIKELY
    // for CUDA mode and put it back
    if( neighbor==reflect_particles ) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      k_particles( pi, particle_var::ux + axis) = -k_particles( pi, particle_var::ux + axis);
      dr *= sign_flip;
//      dr[axis] = -dr[axis];     

      continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
      p_dx = r[0]; p_dy = r[1]; p_dz = r[2];
      pii = 8*pii + face;
      dr.copy_to((float*)(&pm[0]), KokkosSIMD::element_aligned_tag());
      pm->i = pi;
      return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.
    /**/                         // Note: neighbor - rangel < 2^31 / 6
    ii = neighbor - rangel;
    r *= sign_flip;
//    r[axis] = -r[axis];
  }
  #undef p_dx
  #undef p_dy
  #undef p_dz
  #undef p_ux
  #undef p_uy
  #undef p_uz
  #undef p_w
  #undef pii
  return 0; // Return "mover not in use"
}

//template<class particle_view_t, class particle_i_view_t, class neighbor_view_t, class scatter_view_t>
//int
//KOKKOS_INLINE_FUNCTION
//move_p_kokkos(
//    const particle_view_t& k_particles,
//    const particle_i_view_t& k_particles_i,
//    const k_particle_movers_t&  pm,
//    const k_particle_i_movers_t&  pmi,
//    const int idx,
//    //accumulator_sa_t k_accumulators_sa,
//    scatter_view_t& scatter_view,
//    const grid_t* g,
//    neighbor_view_t& d_neighbor,
//    int64_t rangel,
//    int64_t rangeh,
//    const float qsp,
//    //field_array_t* RESTRICT fa,
//    //field_view_t& k_field,
//    float cx,
//    float cy,
//    float cz,
//    const int nx,
//    const int ny,
//    const int nz
//)
//{
//
//  #define p_dx    k_particles( pi, particle_var::dx)
//  #define p_dy    k_particles( pi, particle_var::dy)
//  #define p_dz    k_particles( pi, particle_var::dz)
//  #define p_ux    k_particles( pi, particle_var::ux)
//  #define p_uy    k_particles( pi, particle_var::uy)
//  #define p_uz    k_particles( pi, particle_var::uz)
//  #define p_w     k_particles( pi, particle_var::w)
//  #define pii     k_particles_i(pi)
//
//  //#define local_pm_dispx  k_local_particle_movers(0, particle_mover_var::dispx)
//  //#define local_pm_dispy  k_local_particle_movers(0, particle_mover_var::dispy)
//  //#define local_pm_dispz  k_local_particle_movers(0, particle_mover_var::dispz)
//  //#define local_pm_i      k_local_particle_movers(0, particle_mover_var::pmi)
//
//
//  //k_field_t& k_field = fa->k_f_d;
//  float s_midx, s_midy, s_midz;
//  float s_dispx, s_dispy, s_dispz;
//  float s_dir[3];
//  float v0, v1, v2, v3, v4, v5, q;
//  int axis, face;
//  int64_t neighbor;
//  //int pi = int(local_pm_i);
//  int pi = pmi(idx);
////  auto  k_field_scatter_access = k_f_sa.access();
////  auto accum_sa = accum_sv.access();
//#if defined( VPIC_ENABLE_ACCUMULATORS )
//  auto scatter_access = scatter_view.access();
//#else
//  auto& scatter_access = scatter_view;
//#endif
//
//  q = qsp*p_w;
//
//    //printf("in move %d \n", pi);
//
//  for(;;) {
//    int ii = pii;
//    s_midx = p_dx;
//    s_midy = p_dy;
//    s_midz = p_dz;
//
//
//    s_dispx = pm(idx, particle_mover_var::dispx);
//    s_dispy = pm(idx, particle_mover_var::dispy);
//    s_dispz = pm(idx, particle_mover_var::dispz);
//
//    //printf("pre axis %d x %e y %e z %e \n", axis, p_dx, p_dy, p_dz);
//
//    //printf("disp x %e y %e z %e \n", s_dispx, s_dispy, s_dispz);
//
//    s_dir[0] = (s_dispx>0) ? 1 : -1;
//    s_dir[1] = (s_dispy>0) ? 1 : -1;
//    s_dir[2] = (s_dispz>0) ? 1 : -1;
//
//    // Compute the twice the fractional distance to each potential
//    // streak/cell face intersection.
//    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
//    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
//    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;
//
//    // Determine the fractional length and axis of current streak. The
//    // streak ends on either the first face intersected by the
//    // particle track or at the end of the particle track.
//    //
//    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
//    //   axis 3        ... streak ends at end of the particle track
//    /**/      v3=2,  axis=3;
//    if(v0<v3) v3=v0, axis=0;
//    if(v1<v3) v3=v1, axis=1;
//    if(v2<v3) v3=v2, axis=2;
//    v3 *= 0.5;
//
//    // Compute the midpoint and the normalized displacement of the streak
//    s_dispx *= v3;
//    s_dispy *= v3;
//    s_dispz *= v3;
//    s_midx += s_dispx;
//    s_midy += s_dispy;
//    s_midz += s_dispz;
//
//    // Accumulate the streak.  Note: accumulator values are 4 times
//    // the total physical charge that passed through the appropriate
//    // current quadrant in a time-step
//    v5 = q*s_dispx*s_dispy*s_dispz*(1.f/3.f);
//
//    //a = (float *)(&d_accumulators[ci]);
//
//#   define accumulate_j(X,Y,Z)                                        \
//    v4  = q*s_disp##X;    /* v2 = q ux                            */  \
//    v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
//    v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
//    v1 += v4;             /* v1 = q ux (1+dy)                     */  \
//    v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
//    v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
//    v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
//    v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
//    v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
//    v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
//    v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
//    v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
//    v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
//    v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \
//
//    //Kokkos::atomic_add(&a[0], v0);
//    //Kokkos::atomic_add(&a[1], v1);
//    //Kokkos::atomic_add(&a[2], v2);
//    //Kokkos::atomic_add(&a[3], v3);
//
//    if constexpr (std::is_same<scatter_view_t,k_field_sv_t>::value) {
//      int iii = ii;
//      int zi = iii/((nx+2)*(ny+2));
//      iii -= zi*(nx+2)*(ny+2);
//      int yi = iii/(nx+2);
//      int xi = iii-yi*(nx+2);
//      accumulate_j(x,y,z);
//      scatter_access(ii, field_var::jfx) += cx*v0;
//      scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
//      scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
//      scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;
//
//      accumulate_j(y,z,x);
//      scatter_access(ii, field_var::jfy) += cy*v0;
//      scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
//      scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
//      scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;
//
//      accumulate_j(z,x,y);
//      scatter_access(ii, field_var::jfz) += cz*v0;
//      scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
//      scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
//      scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;
//    } else if constexpr (std::is_same<scatter_view_t,Kokkos::View<float*[12]>>::value) {
//      accumulate_j(x,y,z);
//      scatter_access(ii, 0) += cx*v0;
//      scatter_access(ii, 1) += cx*v1;
//      scatter_access(ii, 2) += cx*v2;
//      scatter_access(ii, 3) += cx*v3;
//
//      accumulate_j(y,z,x);
//      scatter_access(ii, 4) += cy*v0;
//      scatter_access(ii, 5) += cy*v1;
//      scatter_access(ii, 6) += cy*v2;
//      scatter_access(ii, 7) += cy*v3;
//
//      accumulate_j(z,x,y);
//      scatter_access(ii, 8) += cz*v0;
//      scatter_access(ii, 9) += cz*v1;
//      scatter_access(ii, 10) += cz*v2;
//      scatter_access(ii, 11) += cz*v3;
//    }
//
//#   undef accumulate_j
//
//    // Compute the remaining particle displacment
//    pm(idx, particle_mover_var::dispx) -= s_dispx;
//    pm(idx, particle_mover_var::dispy) -= s_dispy;
//    pm(idx, particle_mover_var::dispz) -= s_dispz;
//
//    //printf("pre axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
//    // Compute the new particle offset
//    p_dx += s_dispx+s_dispx;
//    p_dy += s_dispy+s_dispy;
//    p_dz += s_dispz+s_dispz;
//
//    // If an end streak, return success (should be ~50% of the time)
//    //printf("axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
//
//    if( axis==3 ) break;
//
//    // Determine if the particle crossed into a local cell or if it
//    // hit a boundary and convert the coordinate system accordingly.
//    // Note: Crossing into a local cell should happen ~50% of the
//    // time; hitting a boundary is usually a rare event.  Note: the
//    // entry / exit coordinate for the particle is guaranteed to be
//    // +/-1 _exactly_ for the particle.
//
//    v0 = s_dir[axis];
//    k_particles( pi, particle_var::dx + axis) = v0; // Avoid roundoff fiascos--put the particle
//                           // _exactly_ on the boundary.
//    face = axis; if( v0>0 ) face += 3;
//
//    // TODO: clean this fixed index to an enum
//    //neighbor = g->neighbor[ 6*ii + face ];
//    neighbor = d_neighbor( 6*ii + face );
//
//    // TODO: these two if statements used to be marked UNLIKELY,
//    // but that intrinsic doesn't work on GPU.
//    // for performance portability, maybe specialize UNLIKELY
//    // for CUDA mode and put it back
//
//
//    if( neighbor==reflect_particles ) {
//      // Hit a reflecting boundary condition.  Reflect the particle
//      // momentum and remaining displacement and keep moving the
//      // particle.
//      k_particles( pi, particle_var::ux + axis) = -k_particles( pi, particle_var::ux + axis);
//      // Clearer and works with AMD GPUs
//      float* disp = static_cast<float*>(&(pm(idx, particle_mover_var::dispx)));
//      disp[axis] = -disp[axis];
//
//      continue;
//    }
//
//    if( neighbor<rangel || neighbor>rangeh ) {
//      // Cannot handle the boundary condition here.  Save the updated
//      // particle position, face it hit and update the remaining
//      // displacement in the particle mover.
//      pii = 8*pii + face;
//      return 1; // Return "mover still in use"
//      }
//
//    // Crossed into a normal voxel.  Update the voxel index, convert the
//    // particle coordinate system and keep moving the particle.
//
//    pii = neighbor - rangel;
//    /**/                         // Note: neighbor - rangel < 2^31 / 6
//    k_particles( pi, particle_var::dx + axis) = -v0;      // Convert coordinate system
//  }
//  #undef p_dx
//  #undef p_dy
//  #undef p_dz
//  #undef p_ux
//  #undef p_uy
//  #undef p_uz
//  #undef p_w
//  #undef pii
//
//  //#undef local_pm_dispx
//  //#undef local_pm_dispy
//  //#undef local_pm_dispz
//  //#undef local_pm_i
//  return 0; // Return "mover not in use"
//}

template<class particle_view_t, class particle_i_view_t, class neighbor_view_t, class scatter_view_t>
int
KOKKOS_INLINE_FUNCTION
move_p_kokkos(
    const particle_view_t& k_particles,
    const particle_i_view_t& k_particles_i,
    particle_mover_t* ALIGNED(16)  pm,
    //accumulator_sa_t k_accumulators_sa,
    scatter_view_t& scatter_access,
    const grid_t* g,
    neighbor_view_t& d_neighbor,
    int64_t rangel,
    int64_t rangeh,
    const float qsp,
    //field_array_t* RESTRICT fa,
    //field_view_t& k_field,
    float cx,
    float cy,
    float cz,
    const int nx,
    const int ny,
    const int nz
)
{

  #define p_dx    k_particles( pi, particle_var::dx)
  #define p_dy    k_particles( pi, particle_var::dy)
  #define p_dz    k_particles( pi, particle_var::dz)
  #define p_ux    k_particles( pi, particle_var::ux)
  #define p_uy    k_particles( pi, particle_var::uy)
  #define p_uz    k_particles( pi, particle_var::uz)
  #define p_w     k_particles( pi, particle_var::w)
  #define pii     k_particles_i(pi)

  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, v4, v5, q;
  int axis, face;
  int64_t neighbor;
  int pi = pm->i;

  int ii = pii;
  float r[3], dr[3];
  r[0] = p_dx; r[1] = p_dy; r[2] = p_dz;
  dr[0] = pm->dispx; dr[1] = pm->dispy; dr[2] = pm->dispz;
  q = qsp*p_w;

  for(;;) {
    s_midx = r[0];
    s_midy = r[1];
    s_midz = r[2];

    s_dispx = dr[0];
    s_dispy = dr[1];
    s_dispz = dr[2];

    s_dir[0] = (s_dispx>0) ? 1 : -1;
    s_dir[1] = (s_dispy>0) ? 1 : -1;
    s_dir[2] = (s_dispz>0) ? 1 : -1;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
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

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
    v5 = q*s_dispx*s_dispy*s_dispz*(1.f/3.f);

#   define accumulate_j(X,Y,Z)                                        \
    v4  = q*s_disp##X;    /* v2 = q ux                            */  \
    v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
    v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
    v1 += v4;             /* v1 = q ux (1+dy)                     */  \
    v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
    v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
    v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
    v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
    v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
    v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
    v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
    v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
    v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
    v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \

#if defined( VPIC_ENABLE_ACCUMULATORS )
    accumulate_j(x,y,z);
    scatter_access(ii, 0) += cx*v0;
    scatter_access(ii, 1) += cx*v1;
    scatter_access(ii, 2) += cx*v2;
    scatter_access(ii, 3) += cx*v3;

    accumulate_j(y,z,x);
    scatter_access(ii, 4) += cy*v0;
    scatter_access(ii, 5) += cy*v1;
    scatter_access(ii, 6) += cy*v2;
    scatter_access(ii, 7) += cy*v3;

    accumulate_j(z,x,y);
    scatter_access(ii, 8)  += cz*v0;
    scatter_access(ii, 9)  += cz*v1;
    scatter_access(ii, 10) += cz*v2;
    scatter_access(ii, 11) += cz*v3;
#else
    int iii = ii;
    int zi = iii/((nx+2)*(ny+2));
    iii -= zi*(nx+2)*(ny+2);
    int yi = iii/(nx+2);
    int xi = iii-yi*(nx+2);
    accumulate_j(x,y,z);
    scatter_access(ii, field_var::jfx) += cx*v0;
    scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
    scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
    scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;

    accumulate_j(y,z,x);
    scatter_access(ii, field_var::jfy) += cy*v0;
    scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
    scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
    scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;

    accumulate_j(z,x,y);
    scatter_access(ii, field_var::jfz) += cz*v0;
    scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
    scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
    scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;
#endif

#   undef accumulate_j

    // Compute the remaining particle displacment
    dr[0] -= s_dispx;
    dr[1] -= s_dispy;
    dr[2] -= s_dispz;

    // Compute the new particle offset
    r[0] += s_dispx+s_dispx;
    r[1] += s_dispy+s_dispy;
    r[2] += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)
    if( axis==3 ) {
      p_dx = r[0];
      p_dy = r[1];
      p_dz = r[2];
      pii = ii;
      break;
    }

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    // Avoid roundoff fiascos -- put the particle exactly on the boundary.
    v0 = s_dir[axis];

    r[axis] = v0;
    face = axis; if( v0>0 ) face += 3;

    // TODO: clean this fixed index to an enum
    neighbor = d_neighbor( 6*ii + face );

    // TODO: these two if statements used to be marked UNLIKELY,
    // but that intrinsic doesn't work on GPU.
    // for performance portability, maybe specialize UNLIKELY
    // for CUDA mode and put it back
    if( neighbor==reflect_particles ) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      dr[axis] = -dr[axis];
      continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
      p_dx = r[0]; p_dy = r[1]; p_dz = r[2];
      pii = 8*pii + face;
      pm->dispx = dr[0];
      pm->dispy = dr[1];
      pm->dispz = dr[2];
      pm->i = pi;
      return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    ii = neighbor - rangel;
    /**/                         // Note: neighbor - rangel < 2^31 / 6
    r[axis] = -r[axis];
  }
  #undef p_dx
  #undef p_dy
  #undef p_dz
  #undef p_ux
  #undef p_uy
  #undef p_uz
  #undef p_w
  #undef pii

  //#undef local_pm_dispx
  //#undef local_pm_dispy
  //#undef local_pm_dispz
  //#undef local_pm_i
  return 0; // Return "mover not in use"
}

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
  float cx = 0.25 * g->rdy * g->rdz / g->dt;
  float cy = 0.25 * g->rdz * g->rdx / g->dt;
  float cz = 0.25 * g->rdx * g->rdy / g->dt;

  #define p_dx    k_particles(pi, particle_var::dx)
  #define p_dy    k_particles(pi, particle_var::dy)
  #define p_dz    k_particles(pi, particle_var::dz)
  #define p_ux    k_particles(pi, particle_var::ux)
  #define p_uy    k_particles(pi, particle_var::uy)
  #define p_uz    k_particles(pi, particle_var::uz)
  #define p_w     k_particles(pi, particle_var::w)
  #define pii     k_particles_i(pi)

  //#define local_pm_dispx  k_local_particle_movers(0, particle_mover_var::dispx)
  //#define local_pm_dispy  k_local_particle_movers(0, particle_mover_var::dispy)
  //#define local_pm_dispz  k_local_particle_movers(0, particle_mover_var::dispz)
  //#define local_pm_i      k_local_particle_movers(0, particle_mover_var::pmi)


  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, v4, v5, q;
  int axis, face;
  int64_t neighbor;
  //int pi = int(local_pm_i);
  int pi = pm->i;

  q = qsp*p_w;

    //printf("in move %d \n", pi);

  for(;;) {
    int ii = pii;
    s_midx = p_dx;
    s_midy = p_dy;
    s_midz = p_dz;


    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    //printf("pre axis %d x %e y %e z %e \n", axis, p_dx, p_dy, p_dz);

    //printf("disp x %e y %e z %e \n", s_dispx, s_dispy, s_dispz);

    s_dir[0] = (s_dispx>0) ? 1 : -1;
    s_dir[1] = (s_dispy>0) ? 1 : -1;
    s_dir[2] = (s_dispz>0) ? 1 : -1;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
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

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
    v5 = q*s_dispx*s_dispy*s_dispz*(1.f/3.f);

    //a = (float *)(&d_accumulators[ci]);

#   define accumulate_j(X,Y,Z)                                        \
    v4  = q*s_disp##X;    /* v2 = q ux                            */  \
    v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
    v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
    v1 += v4;             /* v1 = q ux (1+dy)                     */  \
    v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
    v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
    v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
    v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
    v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
    v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
    v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
    v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
    v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
    v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */ 

    int iii = ii;
    int zi = iii/((nx+2)*(ny+2));
    iii -= zi*(nx+2)*(ny+2);
    int yi = iii/(nx+2);
    int xi = iii - yi*(nx+2);
    accumulate_j(x,y,z);
    k_jf_accum(ii, accumulator_var::jx) += cx*v0;
    k_jf_accum(VOXEL(xi,yi+1,zi,nx,ny,nz), accumulator_var::jx) += cx*v1;
    k_jf_accum(VOXEL(xi,yi,zi+1,nx,ny,nz), accumulator_var::jx) += cx*v2;
    k_jf_accum(VOXEL(xi,yi+1,zi+1,nx,ny,nz), accumulator_var::jx) += cx*v3;

    accumulate_j(y,z,x);
    k_jf_accum(ii, accumulator_var::jy) += cy*v0;
    k_jf_accum(VOXEL(xi,yi,zi+1,nx,ny,nz), accumulator_var::jy) += cy*v1;
    k_jf_accum(VOXEL(xi+1,yi,zi,nx,ny,nz), accumulator_var::jy) += cy*v2;
    k_jf_accum(VOXEL(xi+1,yi,zi+1,nx,ny,nz), accumulator_var::jy) += cy*v3;

    accumulate_j(z,x,y);
    k_jf_accum(ii, accumulator_var::jz) += cz*v0;
    k_jf_accum(VOXEL(xi+1,yi,zi,nx,ny,nz), accumulator_var::jz) += cz*v1;
    k_jf_accum(VOXEL(xi,yi+1,zi,nx,ny,nz), accumulator_var::jz) += cz*v2;
    k_jf_accum(VOXEL(xi+1,yi+1,zi,nx,ny,nz), accumulator_var::jz) += cz*v3;

#   undef accumulate_j

    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    //printf("pre axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    // Compute the new particle offset
    p_dx += s_dispx+s_dispx;
    p_dy += s_dispy+s_dispy;
    p_dz += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)
    //printf("axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);

    if( axis==3 ) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    v0 = s_dir[axis];
    k_particles(pi, particle_var::dx + axis) = v0; // Avoid roundoff fiascos--put the particle
                           // _exactly_ on the boundary.
    face = axis; if( v0>0 ) face += 3;

    // TODO: clean this fixed index to an enum
    //neighbor = g->neighbor[ 6*ii + face ];
    neighbor = d_neighbor( 6*ii + face );

    // TODO: these two if statements used to be marked UNLIKELY,
    // but that intrinsic doesn't work on GPU.
    // for performance portability, maybe specialize UNLIKELY
    // for CUDA mode and put it back


    if( neighbor==reflect_particles ) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      k_particles(pi, particle_var::ux + axis) = -k_particles(pi, particle_var::ux + axis);
      // Clearer and works with AMD GPUs
      float* disp = static_cast<float*>(&(pm->dispx));
      disp[axis] = -disp[axis];

      continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
      pii = 8*pii + face;
      return 1; // Return "mover still in use"
      }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    pii = neighbor - rangel;
    /**/                         // Note: neighbor - rangel < 2^31 / 6
    k_particles(pi, particle_var::dx + axis) = -v0;      // Convert coordinate system
  }
  #undef p_dx
  #undef p_dy
  #undef p_dz
  #undef p_ux
  #undef p_uy
  #undef p_uz
  #undef p_w
  #undef pii

  //#undef local_pm_dispx
  //#undef local_pm_dispy
  //#undef local_pm_dispz
  //#undef local_pm_i
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
