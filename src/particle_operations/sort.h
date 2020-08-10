#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

#include <Kokkos_Sort.hpp>
#include <Kokkos_DualView.hpp>
#include "../vpic/kokkos_helpers.h"
#include "../species_advance/species_advance.h"
#include "shuffle.h"

/**
 * @brief Bin sort that minimizes memory allocations.
 *
 * For indirect sorts, this is about 10-20% faster than Kokkos bin sort
 * and will use less memory. For direct sorts, this uses max_np out of
 * place particles, but only does a single copy loop. Memory usage and
 * performance depend on max_np vs np.
 */
struct BinSort {

    static void shuffle_sort(
      species_t* sp,
      kokkos_rng_pool_t& rng
    ) 
    {
      ERROR(("Not implemented."));
    }

    static void sort(
        species_t* sp,
        const bool direct = true
    );

};

/**
 * @brief Simple bin sort using Kokkos inbuilt sort
 */
struct DefaultSort {

    static void shuffle_sort(
      species_t* sp,
      kokkos_rng_pool_t& rng
    )
    {

      // TODO: fisher_yates has been tested and verified, but the complete
      //       shuffle_sort has not. Only possible issue is if the fence
      //       in fisher_yates gets removed (I think).

      // Try grab the index's for a permute key
      //int pi = particle_var::pi; // FIXME: can you really not pass an enum in??
      //auto keys = Kokkos::subview(particles, Kokkos::ALL, pi);
      auto keys = sp->k_p_i_d;
      const int num_bins = sp->g->nv;

      // TODO: we can tighten the bounds on this to avoid ghosts

      using key_type = decltype(keys);
      using Comparator = Kokkos::BinOp1D<key_type>;
      Comparator comp(num_bins, 0, num_bins);

      int sort_within_bins = 0;
      Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, sp->np, comp, sort_within_bins);
      bin_sort.create_permute_vector();

      sp->k_partition_d = bin_sort.get_bin_offsets();
      sp->k_partition_h = Kokkos::create_mirror_view(sp->k_partition_d);

      FisherYatesShuffle::fisher_yates(
        bin_sort.get_permute_vector(),
        sp->g,
        sp->k_partition_d,
        rng
      );

      // TODO: do we ever touch this data on the host?
      // Kokkos::deep_copy(sp->k_partition_h, sp->k_partition_d);

      bin_sort.sort(sp->k_p_d);
      bin_sort.sort(sp->k_p_i_d);
      sp->last_sorted = sp->g->step;

    }

    static void sort(
        species_t* sp,
        const bool direct = true
    )
    {

        // Try grab the index's for a permute key
        //int pi = particle_var::pi; // FIXME: can you really not pass an enum in??
        //auto keys = Kokkos::subview(particles, Kokkos::ALL, pi);
        auto keys = sp->k_p_i_d;

        const int step = sp->g->step;
        const int num_bins = sp->g->nv;

        // TODO: we can tighten the bounds on this to avoid ghosts

        using key_type = decltype(keys);
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(num_bins, 0, num_bins);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, sp->np, comp, sort_within_bins );
        bin_sort.create_permute_vector();

        sp->k_partition_d = bin_sort.get_bin_offsets();
        //sp->k_partition_h = Kokkos::create_mirror_view(sp->k_partition_d);

        // TODO: do we ever touch this data on the host?
        // Kokkos::deep_copy(sp->k_partition_h, sp->k_partition_d);

        if( direct ) {

          bin_sort.sort(sp->k_p_d);
          bin_sort.sort(sp->k_p_i_d);
          sp->last_sorted = step;

        } else {

          sp->k_sortindex_d = bin_sort.get_permute_vector();
          //sp->k_sortindex_h = Kokkos::create_mirror_view(sp->k_sortindex_d);
          sp->last_indexed  = step;

          // TODO: do we ever touch this data on the host? Seems expensive.
          // Kokkos::deep_copy(sp->k_sortindex_h, sp->k_sortindex_d);

        }

    }

};

template <typename Policy = DefaultSort>
struct ParticleSorter : private Policy {
    using Policy::sort;
    using Policy::shuffle_sort;
};

#endif //guard
