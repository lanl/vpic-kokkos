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
#include "../vpic/kokkos_helpers.h"

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

        // Track where particles are located, either on the host or the device.
        // On creation and restart, particles must be located on the host.
        // Since the only way they should be moved between the host and device is
        // through copy_to_host and copy_to_device, we know where they are.
        bool on_device = false;

        // Static allocations for the compressor
        Kokkos::View<int*> unsafe_index;
        Kokkos::View<int> clean_up_to_count;
        Kokkos::View<int> clean_up_from_count;
        Kokkos::View<int>::HostMirror clean_up_from_count_h;
        Kokkos::View<int*> clean_up_from;
        Kokkos::View<int*> clean_up_to;


        // Standard species advance kernels

        /**
         * @brief Sorts the particles.
         */
        void sort();

        /**
         * @brief Advances all particles by a single timestep and loads movers.
         */
        void advance( accumulator_array_t  * RESTRICT aa,
                      interpolator_array_t * RESTRICT ia );

        /**
         * @brief Half advance of the particles.
         */
        void center( const interpolator_array_t * RESTRICT ia );

        /**
         * @brief Inverse of center.
         */
        void uncenter( const interpolator_array_t * RESTRICT ia );

        /**
         * @brief Computes the total energy in the species, reduced across ranks.
         */
        double energy( const interpolator_array_t * RESTRICT ia );

        /**
         * @brief Adds this species contribution to the free charge density.
         */
        void accumulate_rhof( field_array_t * RESTRICT fa );

        /**
         * @brief Adds this species contribution to the hydro moments.
         */
        void accumulate_hydro( /**/  hydro_array_t        * RESTRICT ha,
                               const interpolator_array_t * RESTRICT ia );

        /**
         * @brief Copies all the outbound particles and movers  to the host.
         */
        void copy_outbound_to_host();

        /**
         * @brief Copies all the particles and movers from the device to the host.
         */
        void copy_to_host(bool force=false);

        /**
         * @brief Copies all the particles and movers from the host to the device.
         */
        void copy_to_device(bool force=false);

        /**
         * @brief Copies all the inbound particles from the host to the device.
         */
        void copy_inbound_to_device();

        // TODO: Move sorter and compress from particle_operations to species_advance

        /**
         * @brief Sorts the particles. Only works on device right now.
         */
        template<class sort_t>
        void sort(sort_t& sorter, int num_bins)
        {
          log_printf("Performance sorting \"%s\"", name );
          sorter.sort( k_p_d, k_p_i_d, np, num_bins);
        }

        /**
         * @brief Compress the device particles.
         */
        template<class compress_t>
        void compress(compress_t& compressor)
        {

          const int nm_host = k_nm_h(0);
          compressor.compress(
            k_p_d,
            k_p_i_d,
            k_pm_i_d,
            nm_host,
            np,
            this
          );
          np -= nm_host;

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

/**
 * @brief Accumulates a single particle to the bound charge density.
 * NOTE: Only works on the host. Avoid if possible.
 */
void
accumulate_rhob( field_t          * RESTRICT ALIGNED(128) f,
                 const particle_t * RESTRICT ALIGNED(32)  p,
                 const grid_t     * RESTRICT              g,
                 const float                              qsp );

/**
 * @brief Legacy function for host-side moves of particles.
 * NOTE: Only works on the host. Avoid if possible.
 */
int
move_p( particle_t          * ALIGNED(128) p0,
        particle_mover_t    * ALIGNED(16)  pm,
        accumulator_array_t *              aa,
        const grid_t        *              g,
        const float                        qsp );

typedef struct particle_bc particle_bc_t;

/**
 * @brief Exchnages passing particles across domain boundaries.
 * An entire species list is lumped together to minimize communication.
 */
void
boundary_p( particle_bc_t       * RESTRICT pbc_list,
            species_t           * RESTRICT sp_list,
            field_array_t       * RESTRICT fa,
            accumulator_array_t * RESTRICT aa );

#endif // _species_advance_h_
