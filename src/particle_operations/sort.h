#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

#include <Kokkos_Sort.hpp>
#include <Kokkos_DualView.hpp>
#include "../vpic/kokkos_helpers.h"

/**
 * @brief Simple bin sort using Kokkos inbuilt sort
 */
struct DefaultSort {
    static void sort(
            k_particles_t particles,
            const int32_t np,
            const int32_t num_bins
    )
    {
        // Try grab the index's for a permute key
        int pi = particle_var::pi; // FIXME: can you really not pass an enum in??
        auto keys = Kokkos::subview(particles, Kokkos::ALL, pi);

        // TODO: we can tighten the bounds on this to avoid ghosts

        using key_type = decltype(keys);
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(num_bins, 0, num_bins);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
        bin_sort.create_permute_vector();
        bin_sort.sort(particles);
    }

};

template <typename Policy = DefaultSort>
struct ParticleSorter : private Policy {
    using Policy::sort;
};

#endif //guard
