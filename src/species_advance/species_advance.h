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

#include "../grid/grid.h"       // Include for get_neighbor_index
#include "../sf_interface/sf_interface.h"
#include "Kokkos_DualView.hpp"

// Smallest power of 2 that's an upper bound of 
// the neighborhood size (27).
static constexpr unsigned upper_bound_of_neighbors = 32;
// This is needed for move_p_kokkos
static constexpr float two = 2.f;

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

class species_t {
    public:

        char * name;                        // Species name
        float q;                            // Species particle charge
        float m;                            // Species particle rest mass

        int np = 0, max_np = 0;             // Number and max local particles
        particle_t * ALIGNED(128) p;        // Array of particles for the species

        int nm, max_nm;                     // Number and max local movers in use
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
        k_iterator_t k_nm_d;               // nm iterator
        k_iterator_t::HostMirror k_nm_h;

        // TODO: this should ultimatley be removeable.
        // This tracks the number of particles we need to move back to the device
        // And is basically the same as nm at certain times?
        int num_to_copy = 0;

        // Step when the species was last copied to to the host.  The copy can
        // take place at any time during the step, so checking
        // species_copy_last==step() does not mean that the host and device
        // data are the same.  Typically, copy is called immediately after the
        // step is incremented and before or during user_diagnostics.  Checking
        // species_copy_last==step() in these circumstances does mean the host
        // is up to date, unless you do unusual stuff in user_diagnostics.
        //
        // This number is tracked on the host only, and may be inaccurate on
        // the device.
        int64_t species_copy_last = -1;

        // Init Kokkos Particle Arrays
        species_t(int n_particles, int n_pmovers) :
            k_p_d("k_particles", n_particles),
            k_p_i_d("k_particles_i", n_particles),
            k_pm_d("k_particle_movers", n_pmovers),
            k_pm_i_d("k_particle_movers_i", n_pmovers),
            k_pc_d("k_particle_copy_for_movers", n_pmovers),
            k_pc_i_d("k_particle_copy_for_movers_i", n_pmovers),
            k_pr_h("k_particle_send_for_movers", n_pmovers),
            k_pr_i_h("k_particle_send_for_movers_i", n_pmovers),
            k_nm_d("k_nm") // size 1
    {
        k_p_h = Kokkos::create_mirror_view(k_p_d);
        k_p_i_h = Kokkos::create_mirror_view(k_p_i_d);

        k_pc_h = Kokkos::create_mirror_view(k_pc_d);
        k_pc_i_h = Kokkos::create_mirror_view(k_pc_i_d);

        k_pm_h = Kokkos::create_mirror_view(k_pm_d);
        k_pm_i_h = Kokkos::create_mirror_view(k_pm_i_d);

        k_nm_h = Kokkos::create_mirror_view(k_nm_d);
    }

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
           /**/  accumulator_array_t  * RESTRICT aa,
                 interpolator_array_t * RESTRICT ia );

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

void 
k_accumulate_rhob(k_field_t& kfield, 
                  k_particles_t& kpart, 
                  k_particle_movers_t& kpart_movers, 
                  const grid_t* RESTRICT g, 
                  const float qsp,
                  const int nm);

// In hydro_p.c

void
accumulate_hydro_p( /**/  hydro_array_t        * RESTRICT ha,
                    const species_t            * RESTRICT sp,
                    const interpolator_array_t * RESTRICT ia );

// In move_p.cxx

int
move_p( particle_t       * ALIGNED(128) p0,    // Particle array
        particle_mover_t * ALIGNED(16)  m,     // Particle mover to apply
        accumulator_t    * ALIGNED(128) a0,    // Accumulator to use
        const grid_t     *              g,     // Grid parameters
        const float                     qsp ); // Species particle charge

// Make an axis enumeration for the s_dir
enum Axis_Label {
    x = 0, y = 1, z = 2, num_axes = 3
};

// Add neighbor index dumping for testing zigzag.
#if VPIC_DUMP_NEIGHBORS
  #include "vpic_dump_neighbors.hpp"
#endif

template<class particle_view_t, class particle_i_view_t, class accumulator_sa_t, class neighbor_view_t>
int
KOKKOS_INLINE_FUNCTION
move_p_kokkos(
    const particle_view_t& k_particles,
    const particle_i_view_t& k_particles_i,
    particle_mover_t* ALIGNED(16)  pm,
    accumulator_sa_t k_accumulators_sa,
    const grid_t* g,
    neighbor_view_t& d_neighbor,
    int64_t rangel,
    int64_t rangeh,
    const float qsp
)
{
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
  int axis, neighbor_index;
  int64_t neighbor;
  //int pi = int(local_pm_i);
  int pi = pm->i;
  auto k_accumulators_scatter_access = k_accumulators_sa.access();
 
  const int planes_per_axis = g->PLANES_PER_AXIS;
  const int num_neighbors = g->NUM_NEIGHBORS;


  // TODO: Get rid of these. They are only for debugging
  float old_fx = p_dx + 2 * pm->dispx;
  float old_fy = p_dy + 2 * pm->dispy;
  float old_fz = p_dz + 2 * pm->dispz;

  // Declare some useful variables for zigzag
  // These are the final particle position in the
  // zigzag trajectory
  float zig_finalx = 0.;
  float zig_finaly = 0.;
  float zig_finalz = 0.;
    
  // These are these "relay points" calculated
  // directly in the Zigzag algorithm. (See Umeda et al. 2002)
  float xr = 0.;
  float yr = 0.;
  float zr = 0.;

#if VPIC_DUMP_NEIGHBORS
  DUMP_NEIGHBORS<int, float> print_neighbor("neighbor_indices.txt", 
                                            "neighbor_planes.txt",
                                            "neighbor_connections.txt");
#endif

  q = qsp*p_w;

    //printf("in move %d \n", pi);

    // Move the particles and accumulate the 
    // currents using either the standard way
    // or the new and flashy zigzag way.
#if ACCUMULATE_J_ZIGZAG
    #include "accumulate_j_zigzag.hpp"
#else
    // TODO: THIS IS NO LONGER VALID
    // WITHOUT AN OUTER FOR LOOP
    #include "accumulate_j.hpp"
#endif
 
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
template<class particle_view_t, class particle_i_view_t, class accumulator_t, class neighbor_view_t>
int
move_p_kokkos_host_serial(
    const particle_view_t& k_particles,
    const particle_i_view_t& k_particles_i,
    particle_mover_t* ALIGNED(16) pm,
    accumulator_t k_accumulators,
    const grid_t* g,
    neighbor_view_t& d_neighbor,
    int64_t rangel,
    int64_t rangeh,
    const float qsp
)
{

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
  //auto k_accumulators_scatter_access = k_accumulators_sa.access();

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
    v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \
    //Kokkos::atomic_add(&a[0], v0); \
    //Kokkos::atomic_add(&a[1], v1); \
    //Kokkos::atomic_add(&a[2], v2); \
    //Kokkos::atomic_add(&a[3], v3);

    accumulate_j(x,y,z);
    k_accumulators(ii, accumulator_var::jx, 0) += v0;
    k_accumulators(ii, accumulator_var::jx, 1) += v1;
    k_accumulators(ii, accumulator_var::jx, 2) += v2;
    k_accumulators(ii, accumulator_var::jx, 3) += v3;

    accumulate_j(y,z,x);
    k_accumulators(ii, accumulator_var::jy, 0) += v0;
    k_accumulators(ii, accumulator_var::jy, 1) += v1;
    k_accumulators(ii, accumulator_var::jy, 2) += v2;
    k_accumulators(ii, accumulator_var::jy, 3) += v3;

    accumulate_j(z,x,y);
    k_accumulators(ii, accumulator_var::jz, 0) += v0;
    k_accumulators(ii, accumulator_var::jz, 1) += v1;
    k_accumulators(ii, accumulator_var::jz, 2) += v2;
    k_accumulators(ii, accumulator_var::jz, 3) += v3;

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

      // TODO: make this safer
      //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
      //k_local_particle_movers(0, particle_mover_var::dispx + axis) = -k_local_particle_movers(0, particle_mover_var::dispx + axis);
      // TODO: replace this, it's horrible
      (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];


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
#endif // _species_advance_h_
