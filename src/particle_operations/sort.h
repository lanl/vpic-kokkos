#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

// TODO: document
struct DefaultSort {
    static void sort(
            k_particles_t particles,
            const int32_t np,
            species_t* sp
    )
    {
        // Try grab the index's for a permute key
        int pi = particle_var::pi; // TODO: can you really not pass an enum in??
        auto keys = Kokkos::subview(sp->k_p_d, Kokkos::ALL, pi);
        using key_type = decltype(keys);

        // TODO: we can tighten the bounds on this
        int max = accumulator_array->na;
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(max, 0, max);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, sp->np, comp, sort_within_bins );
        bin_sort.create_permute_vector();
        bin_sort.sort(sp->k_p_d);
    }

};

struct SortCompress {
    static void compress(
            k_particles_t particles,
            k_particle_movers_t particle_movers,
            const int32_t nm,
            const int32_t np,
            species_t* sp
            )
    {
    }

};

template <typename Policy = DefaultSort>
struct ParticleSorter : private Policy {
    using Policy::sort;
};

#endif //guard
