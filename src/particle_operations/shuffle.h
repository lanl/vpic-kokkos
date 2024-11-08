#ifndef PARTICLE_SHUFFLE_POLICY_H
#define PARTICLE_SHUFFLE_POLICY_H

#include "../util/rng_policy.h"
#include "../species_advance/species_advance.h"
#include "../vpic/kokkos_helpers.h"

#define _RANK_TO_INDEX(rank,ix,iy,iz,nx,ny,nz) do {       \
    int _ix, _iy, _iz;                                    \
    _ix  = (rank);   /* ix = ix + nx*( iy + ny*iz )   */  \
    _iy  = _ix/(nx); /* iy = iy + gpy*iz */               \
    _ix -= _iy*(nx); /* ix = ix */                        \
    _iz  = _iy/(ny); /* iz = iz */                        \
    _iy -= _iz*(ny); /* iy = iy */                        \
    (ix) = _ix;                                           \
    (iy) = _iy;                                           \
    (iz) = _iz;                                           \
  } while(0)


// TODO : Move these to util/swap.h ?

/**
 * @brief Default swap implementation.
 */
template<class view_type>
KOKKOS_INLINE_FUNCTION
void swap(const view_type& view, size_t i, size_t j) {
  auto t  = view(i);
  view(i) = view(j);
  view(j) = t;
}

/**
 * @brief Particle swap implementation.
 */
template<>
KOKKOS_INLINE_FUNCTION
void swap(const k_particles_t& particles, size_t i, size_t j) {

  auto dx_i = particles(i, particle_var::dx);
  auto dy_i = particles(i, particle_var::dy);
  auto dz_i = particles(i, particle_var::dz);
  auto ux_i = particles(i, particle_var::ux);
  auto uy_i = particles(i, particle_var::uy);
  auto uz_i = particles(i, particle_var::uz);
  auto w_i  = particles(i, particle_var::w);

  particles(i, particle_var::dx) = particles(j, particle_var::dx);
  particles(i, particle_var::dy) = particles(j, particle_var::dy);
  particles(i, particle_var::dz) = particles(j, particle_var::dz);
  particles(i, particle_var::ux) = particles(j, particle_var::ux);
  particles(i, particle_var::uy) = particles(j, particle_var::uy);
  particles(i, particle_var::uz) = particles(j, particle_var::uz);
  particles(i, particle_var::w)  = particles(j, particle_var::w);

  particles(j, particle_var::dx) = dx_i;
  particles(j, particle_var::dy) = dy_i;
  particles(j, particle_var::dz) = dz_i;
  particles(j, particle_var::ux) = ux_i;
  particles(j, particle_var::uy) = uy_i;
  particles(j, particle_var::uz) = uz_i;
  particles(j, particle_var::w)  = w_i;

}

/**
 * @brief Serial Fisher-Yates shuffle within bins.
 */
struct FisherYatesShuffle {

    /**
     * @brief Generic Fisher-Yates shuffle.
     */
    template<class view_type>
    static void fisher_yates(
      const view_type& view,
      grid_t * g,
      k_particle_partition_t_ra partition,
      kokkos_rng_pool_t& rp
    )
    {
      const int nx = g->nx;
      const int ny = g->ny;
      const int nz = g->nz;

      using Space=Kokkos::DefaultExecutionSpace;
      using member_type=Kokkos::TeamPolicy<Space>::member_type;

      // Using a TeamPolicy here gives a massive (~4x) speedup over
      // RangePolicy or MDRangePolicy ... but I don't really understand why
      // since the algorithm is serial within voxels. I guess if RangePolicy or
      // MDRangePolicy don't use L1 effectively then this could make sense
      // because then the implementation below would be very ineffcieint.

      // TODO : If L1 usage is an issue, we should explicitly use warp
      //        shared mem and shuffle a permute vector for better
      //        compatibility with pre-V100.

      Kokkos::parallel_for("FisherYatesShuffle::fisher_yates",
        Kokkos::TeamPolicy<Space>(nx*ny*nz, 1),
        KOKKOS_LAMBDA (member_type team_member)
        {

            int ix, iy, iz;
            _RANK_TO_INDEX(team_member.league_rank(), ix, iy, iz, nx, ny, nz);
            const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
            const auto i0 = partition(v);
            const auto ni = partition(v+1) - i0;


            auto rg = rp.get_state();

            Kokkos::single(Kokkos::PerTeam(team_member),
            [&]() {

              for(int i=0 ; i < ni-1 ; ++i ) {

                if (ni == 0) { return; }
                int j = rg.urand(i, ni); // [i, ni)
                swap(view, i0+i, i0+j);

              }

            });

            rp.free_state(rg);

        });

      // Important!
      Kokkos::fence();

    }

    static void shuffle(
      species_t* sp,
      kokkos_rng_pool_t& rp,
      bool direct=true
    )
    {

      if( !sp || !sp->g )
      {
        ERROR(("Bad args."));
      }

      if( direct ) {

        if( sp->last_sorted != sp->g->step )
        {
          ERROR(("Particles must be sorted before shuffling."));
        }

        // We do not need to shuffle k_p_i_d since shuffling
        // occurs within the same voxel.
        fisher_yates(
          sp->k_p_d,
          sp->g,
          sp->k_partition_d,
          rp
        );

      } else {

        if( sp->last_indexed != sp->g->step )
        {
          ERROR(("Particles must be indexed before shuffling."));
        }

        fisher_yates(
          sp->k_sortindex_d,
          sp->g,
          sp->k_partition_d,
          rp
        );

      }

    }

};

template <typename Policy = FisherYatesShuffle>
struct ParticleShuffler : private Policy {
    using Policy::shuffle;
};

#endif //guard
