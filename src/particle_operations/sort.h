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
            /*
            k_particles_t particles,
            k_particles_i_t particles_i,
            const int32_t np,
            */
            species_t* sp,
            const int32_t num_bins,
            int step
    )
    {

        auto& particles = sp->k_p_d;
        auto& particles_i = sp->k_p_i_d;

        // Try grab the index's for a permute key
        //int pi = particle_var::pi; // FIXME: can you really not pass an enum in??
        //auto keys = Kokkos::subview(particles, Kokkos::ALL, pi);
        auto keys = particles_i;

        // TODO: we can tighten the bounds on this to avoid ghosts

        using key_type = decltype(keys);
        //using Comparator = Casted_BinOp1D<key_type>;
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(num_bins, 0, num_bins);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, sp->np, comp, sort_within_bins );
        bin_sort.create_permute_vector();
        bin_sort.sort(particles);
        bin_sort.sort(particles_i);

        sp->k_partition_d = bin_sort.get_bin_offsets();
        sp->k_partition_h = Kokkos::create_mirror_view(sp->k_partition_d);
        Kokkos::deep_copy(sp->k_partition_d, sp->k_partition_h);

        // Track the step on which we sorted the particles
        sp->last_sorted = step;
    }

};

template <typename Policy = DefaultSort>
struct ParticleSorter : private Policy {
    using Policy::sort;
};

#endif //guard
