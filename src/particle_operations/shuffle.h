#ifndef PARTICLE_SHUFFLE_POLICY_H
#define PARTICLE_SHUFFLE_POLICY_H

#include "../util/rng_policy.h"
#include "../species_advance/species_advance.h"
#include "../vpic/kokkos_helpers.h"
#include "Kokkos_Sort.hpp"
#include "Kokkos_Bitset.hpp"

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
#ifdef VARIABLE_CHARGE
  auto qp_i  = particles(i, particle_var::qp);
#endif

  particles(i, particle_var::dx) = particles(j, particle_var::dx);
  particles(i, particle_var::dy) = particles(j, particle_var::dy);
  particles(i, particle_var::dz) = particles(j, particle_var::dz);
  particles(i, particle_var::ux) = particles(j, particle_var::ux);
  particles(i, particle_var::uy) = particles(j, particle_var::uy);
  particles(i, particle_var::uz) = particles(j, particle_var::uz);
  particles(i, particle_var::w)  = particles(j, particle_var::w);
#ifdef VARIABLE_CHARGE
  particles(i, particle_var::qp) = particles(j, particle_var::qp);
#endif

  particles(j, particle_var::dx) = dx_i;
  particles(j, particle_var::dy) = dy_i;
  particles(j, particle_var::dz) = dz_i;
  particles(j, particle_var::ux) = ux_i;
  particles(j, particle_var::uy) = uy_i;
  particles(j, particle_var::uz) = uz_i;
  particles(j, particle_var::w)  = w_i;
#ifdef VARIABLE_CHARGE
  particles(j, particle_var::qp) = qp_i;
#endif

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

      //view_type
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

              for(uint32_t i=0 ; i < ni-1 ; ++i ) {

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

/**
 * @brief MergeShuffle: A Very Fast, Parallel Random Permutation Algorithm
 * Axel Bacher, Olivier, Bodini, Alexandros Hollender, and Jeremie Lumbroso 2015
 */
struct MergeShuffle {

    /**
     * @brief Generic Fisher-Yates shuffle
     */
    template<class view_type, class rand_gen>
    KOKKOS_INLINE_FUNCTION
    static void fisher_yates(const view_type& view, rand_gen rg) {
      uint32_t n = view.extent(0);
      for(uint32_t i=0; i<n; i++) {
        uint32_t j = rg.urand(i, n); // [i, n)
        swap(view, i, j);
      }
    }

    /**
     * @brief Merge shuffled subviews
     */
    template<class view_type, class rand_gen>
    KOKKOS_INLINE_FUNCTION
    static void merge(const view_type& view, rand_gen rg, uint32_t mid, uint32_t end) {
      size_t u = 0;
      size_t v = mid;
      size_t w = end;
      while(true) {
        if(rg.urand(0,2)) {
          if(v == w) break;
          swap(view, u,v++);
        } else if(u == v) {
          break;
        } 
        u++;
      }
      while(u < w) {
        uint32_t i = rg.urand(0, u+1);
        swap(view, i, u++);
      }
    }

    /**
     * @brief Shuffle subsets in parallel and merge (similar to merge sort).
     */
    template<class view_type>
    static void merge_shuffle(
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

      constexpr uint64_t cutoff = 0x10000;

      Kokkos::parallel_for("MergeShuffle::merge_shuffle",
        Kokkos::TeamPolicy<Space>(nx*ny*nz, Kokkos::AUTO()),
        KOKKOS_LAMBDA (member_type team_member) {
        int ix, iy, iz;
        _RANK_TO_INDEX(team_member.league_rank(), ix, iy, iz, nx, ny, nz);
        const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
        const auto i0 = partition(v);
        const auto ni = partition(v+1) - i0;

        uint32_t c = 0;
        while( (ni >> c) > cutoff ) {
          c++;
        }
        uint32_t q = 1 << c;
        uint64_t nn = ni;

        auto rg = rp.get_state();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, q), 
          [=] (uint32_t i) {
          uint64_t j = nn * i >> c;
          uint64_t k = nn * (i+1) >> c;
          auto subview = Kokkos::subview(view, Kokkos::make_pair(i0+j, i0+k));
          fisher_yates(subview, rg);
        });

        for(uint32_t p = 1; p < q; p += p) {
          uint32_t niters = q / 2*p;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, niters),
            [=] (uint32_t i) {
            uint64_t j = nn * i >> c;
            uint64_t k = nn * (i+p) >> c;
            uint64_t l = nn * (i+2*p) >> c;
            auto subview = Kokkos::subview(view, Kokkos::make_pair(i0+j, i0+l));
            merge(subview, rg, k-j, l-j);
          });
        }
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
        //merge_shuffle(
        //  sp->k_p_d,
        //  sp->g,
        //  sp->k_partition_d,
        //  rp
        //);

      } else {

        if( sp->last_indexed != sp->g->step )
        {
          ERROR(("Particles must be indexed before shuffling."));
        }

        merge_shuffle(
          sp->k_sortindex_d,
          sp->g,
          sp->k_partition_d,
          rp
        );

      }

    }
};

/**
 * @brief SortShuffle: Shuffle particles using randomly generated keys
 */
struct SortShuffle {
  /**
   * @brief Generate random keys and sort particles
   */
  template<class view_type>
  static void sort_shuffle(
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

    Kokkos::View<uint32_t*> keys("Keys", view.extent(0));
    Kokkos::parallel_for("SortShuffle::Generate random keys", 
      Kokkos::RangePolicy<size_t>(0,view.extent(0)), 
      KOKKOS_LAMBDA(const size_t i) {
      auto rg = rp.get_state();
      keys(i) = rg.urand();
      rp.free_state(rg);
    });

    Kokkos::parallel_for("SortShuffle::sort_shuffle",
      Kokkos::TeamPolicy<Space>(nx*ny*nz, Kokkos::AUTO()),
      KOKKOS_LAMBDA (member_type team_member) {
        int ix, iy, iz;
        _RANK_TO_INDEX(team_member.league_rank(), ix, iy, iz, nx, ny, nz);
        const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
        const auto i0 = partition(v);
        const auto ni = partition(v+1) - i0;

        auto key_view = Kokkos::subview(keys, Kokkos::make_pair(i0, i0+ni));
//        if constexpr (view_type::rank() == 1) {
          auto val_view = Kokkos::subview(view, Kokkos::make_pair(i0, i0+ni));
          Kokkos::Experimental::sort_by_key_team(team_member, key_view, val_view);
//        } else {
//          auto val_view = Kokkos::subview(view, Kokkos::make_pair(i0, i0+ni), Kokkos::ALL());
//          Kokkos::Experimental::sort_by_key_team(team_member, key_view, val_view);
//        }
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
      //sort_shuffle(
      //  sp->k_p_d,
      //  sp->g,
      //  sp->k_partition_d,
      //  rp
      //);

    } else {

      if( sp->last_indexed != sp->g->step )
      {
        ERROR(("Particles must be indexed before shuffling."));
      }

      sort_shuffle(
        sp->k_sortindex_d,
        sp->g,
        sp->k_partition_d,
        rp
      );

    }

  }
};

/**
 * @brief LCG
 */
struct ReverseBijectiveFunction {
  uint64_t capacity;

  template<class RandGen>
  KOKKOS_INLINE_FUNCTION
  void setup(uint64_t _capacity, RandGen rng) {
    capacity = _capacity;
  }

  KOKKOS_INLINE_FUNCTION
  uint64_t operator()(const uint64_t val) {
    return capacity-val-1;
  }
};

/**
 * @brief LCG
 */
struct LCGBijectiveFunction {
  uint64_t modulus;
  uint64_t multiplier;
  uint64_t addition;

  KOKKOS_INLINE_FUNCTION
  uint64_t roundUpPower2( uint64_t a ) {
    if( a & ( a - 1 ) ) {
      uint64_t i;
      for( i = 0; a > 1; i++ ) {
        a >>= 1ull;
      }
      return 1ull << ( i + 1ull );
    }
    return a;
  }
  
  template<class RandGen>
  KOKKOS_INLINE_FUNCTION
  void setup(uint64_t capacity, RandGen rng) {
    modulus = roundUpPower2( capacity );
    // Must be odd so it's coprime to modulus
    multiplier = ( rng.urand64() * 2 + 1 ) % modulus;
    addition = rng.urand64() % modulus;
  }

  KOKKOS_INLINE_FUNCTION
  uint64_t operator()(const uint64_t val) {
    // Modulus must be power of 2
    return ( ( val * multiplier ) + addition ) & ( modulus - 1 );
  }
};

/**
 * @brief Variable Philox function
 */
struct PhiloxBijectiveFunction {
  static constexpr uint64_t num_rounds = 24;
  static constexpr uint64_t M0 = 0xD2B74407B1CE6E93;
  uint64_t left_side_bits;
  uint64_t right_side_bits;
  uint64_t left_side_mask;
  uint64_t right_side_mask;
  uint32_t key[num_rounds];
  
  template<class RandGen>
  KOKKOS_INLINE_FUNCTION
  void setup(uint64_t capacity, RandGen rng) {
    // Get Cipher bits
    uint64_t total_bits;
    if( capacity == 0 ) {
      total_bits = 0;
    } else {
      uint64_t i = 0;
      capacity--;
      while(capacity != 0) {
        i++;
        capacity >>= 1;
      }
      total_bits = i < uint64_t( 4 ) ? uint64_t(4) : i;
    }

    // half bits rounded down
    left_side_bits = total_bits / 2;
    left_side_mask = ( 1ull << left_side_bits ) - 1;
    // half bits rounded up
    right_side_bits = total_bits - left_side_bits;
    right_side_mask = ( 1ull << right_side_bits ) - 1;
    // setup cipher keys
    for(uint64_t i=0; i<num_rounds; i++)
      key[i] = rng.urand();
  }

  /**
   * @brief Perform 64-bit integer multiplication and 
   * return upper and lower 32-bits
   */
  KOKKOS_INLINE_FUNCTION
  uint32_t mulhilo( uint64_t a, uint32_t b, uint32_t& hip ) {
    uint64_t product = a * uint64_t(b);
    hip = product >> 32;
    return uint32_t( product );
  }

  /**
   * @brief Compute bijective mapping of input value
   */
  KOKKOS_INLINE_FUNCTION
  uint64_t
  operator()(const uint64_t val) {
    uint32_t state[2] = { uint32_t(val >> right_side_bits), 
                          uint32_t(val &  right_side_mask) };
    for(uint64_t i=0; i<num_rounds; i++) {
      uint32_t hi;
      uint32_t lo = mulhilo(M0, state[0], hi);
      lo = (lo << (right_side_bits - left_side_bits)) | 
           state[1] >> left_side_bits;
      state[0] = ((hi ^ key[i]) ^ state[1]) & left_side_mask;
      state[1] = lo & right_side_mask;
    }
    return (uint64_t)state[0] << right_side_bits | (uint64_t)state[1];
  }
};

/**
 * @brief BijectiveShuffle: Bandwidth-Optimal Random Shuffling for GPUs
 * Rory Mitchell, Daniel Stokes, Eibe Frank, and Geoffrey Holmes
 * Increases per thread arithmetic while reducing global memory access.
 */
template<class BijectiveFunction = PhiloxBijectiveFunction>
struct BijectiveShuffle {

  /**
   * @brief Shuffle particles using a bijective function
   * 
   * Bijective functions produce a one to one mapping between inputs and
   * outputs. VariablePhilox produces the mapping for any power of 2 sequence.
   * The view is padded to a power of 2 and the flag filters out any padded
   * elements. An exclusive scan is used on the flag to get the final mapping.
   */
  template<class view_type>
  static void bijective_shuffle(
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

    Kokkos::View<size_t*> padded_partition("Padded partitioning", partition.extent(0));
    size_t padded_len = 0;
    Kokkos::parallel_scan("Compute padded partition", Kokkos::RangePolicy<size_t>(0,partition.extent(0)-1),
    KOKKOS_LAMBDA(const size_t i, size_t& partial_sum, bool is_final) {
      auto size = partition(i+1) - partition(i);
      size_t padded_cell_len = 0;
      size_t j = 0;
      if(size > 0) {
        size--;
        while(size != 0) {
          j++;
          size >>= 1;
        }
        padded_cell_len = size_t(1) << j;
      }

      if(is_final) {
        padded_partition(i) = partial_sum; 
        if(i == partition.extent(0)-2)
          padded_partition(i+1) = partial_sum + padded_cell_len;
        partial_sum += padded_cell_len;
      }
      partial_sum += padded_cell_len;
    }, padded_len);

    view_type copy("View copy", view.extent(0));
    Kokkos::deep_copy(copy, view);
    Kokkos::View<size_t*> bijection("Bijection mapping", padded_len);
    Kokkos::Bitset<Kokkos::DefaultExecutionSpace> flags_bitset(padded_len);
    flags_bitset.reset();
    Kokkos::View<size_t*> out_idx("Output indices", padded_len);

    size_t scratch_size = sizeof(BijectiveFunction);
    auto team_policy = Kokkos::TeamPolicy<Space>(nx*ny*nz, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(scratch_size));
    Kokkos::parallel_for("BijectiveShuffle::bijective_shuffle",
      team_policy, KOKKOS_LAMBDA (member_type team_member) {
      int ix, iy, iz;
      const int cell = team_member.league_rank();
      _RANK_TO_INDEX(cell, ix, iy, iz, nx, ny, nz);
      const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
      const size_t i0 = partition(v);
      const size_t ni = partition(v+1) - i0;
      const size_t padded_i0 = padded_partition(v);
      const size_t padded_ni = padded_partition(v+1) - padded_i0;
      const size_t niters = padded_ni;

      auto rg = rp.get_state();

      auto slice = Kokkos::make_pair(i0, i0+ni);
      auto padded_slice = Kokkos::make_pair(padded_i0, padded_i0+padded_ni);

      auto bijection_map = Kokkos::subview(bijection, padded_slice);
      auto out_subview = Kokkos::subview(out_idx, padded_slice);

      auto subview = Kokkos::subview(view, slice);
      auto copy_subview = Kokkos::subview(copy, slice);

      team_member.team_barrier();

      BijectiveFunction* bijective_func = (BijectiveFunction*) team_member.team_shmem().get_shmem(sizeof(BijectiveFunction));
      Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
        bijective_func->setup(padded_ni, rg);
      });

      team_member.team_barrier();

      // Compute bijection
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, niters), 
        [=] (const size_t idx) {
        const uint64_t b = (*bijective_func)(idx);
        if(b < ni)
          flags_bitset.set(padded_i0+idx);
        bijection_map(idx) = b;
      });

      team_member.team_barrier();

      // Get output indices
      Kokkos::parallel_scan(Kokkos::TeamThreadRange(team_member, niters),
        [=] (const size_t i, size_t& partial_sum, bool is_final) {
        if(is_final) 
          out_subview(i) = partial_sum;
        partial_sum += size_t( flags_bitset.test(padded_i0+i) );
      });

      team_member.team_barrier();

      // Shuffle data
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, niters), 
        [=] (const size_t i) {
        if(bijection_map(i) < ni) {
          subview(out_subview(i)) = copy_subview(bijection_map(i));
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
      //bijective_shuffle(
      //  sp->k_p_d,
      //  sp->g,
      //  sp->k_partition_d,
      //  rp
      //);

    } else {

      if( sp->last_indexed != sp->g->step )
      {
        ERROR(("Particles must be indexed before shuffling."));
      }

      bijective_shuffle(
        sp->k_sortindex_d,
        sp->g,
        sp->k_partition_d,
        rp
      );

    }

  }
};

// Use Merge shuffle on CPUs and Philox shuffle on GPUs
using DefaultShuffle = std::conditional<std::is_same_v<Kokkos::DefaultExecutionSpace,
                                                       Kokkos::DefaultHostExecutionSpace>, 
                                        MergeShuffle, 
                                        BijectiveShuffle<PhiloxBijectiveFunction> >::type;

template <typename Policy = DefaultShuffle>
struct ParticleShuffler : private Policy {
    using Policy::shuffle;
};

#endif //guard
