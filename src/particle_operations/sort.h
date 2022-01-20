#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

#include <Kokkos_Sort.hpp>
#include <Kokkos_DualView.hpp>
#include "../vpic/kokkos_helpers.h"

struct min_max_functor {
  typedef Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> minmax_scalar;
  Kokkos::View<int*> view;
  min_max_functor(const Kokkos::View<int*>& view_) : view(view_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i, minmax_scalar& minmax) const {
    if(view(i) < minmax.min_val && view(i) != 0) minmax.min_val = view(i);
    if(view(i) > minmax.max_val && view(i) != 0) minmax.max_val = view(i);
  }
};

/**
 * @brief Simple bin sort using Kokkos inbuilt sort
 */
struct DefaultSort {
    // TODO: should the sort interface just take the sp?
    static void sort(
            k_particles_t particles,
            k_particles_i_t particles_i,
            const int32_t np,
            const int32_t num_bins
    )
    {
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
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
        bin_sort.create_permute_vector();
        bin_sort.sort(particles);
        bin_sort.sort(particles_i);
    }

    static void strided_sort(
            k_particles_t particles,
            k_particles_i_t particles_i,
            const int32_t np,
            const int32_t num_bins
    )
    {
        // Create permute view by taking index view and adding offsets such that we get
        // 1,2,3,1,2,3,1,2,3 instead of 1,1,1,2,2,2,3,3,3 
        Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> result;
        Kokkos::MinMax<Kokkos::View<int*>::non_const_value_type> reducer(result);
        // Find max and min particle index
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(particles_i), reducer);
        Kokkos::View<int*> bin_counter("Counter for updating keys", num_bins);
        Kokkos::deep_copy(bin_counter, 0);
        // Count number of particles in each cell and add an offset 
        // (current number of particles in cell multiplied by the largest index)
        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
          int count = Kokkos::atomic_fetch_add(&(bin_counter(particles_i(i))), 1);
          particles_i(i) += count*(result.max_val+1);
        });
        // Save the max particle index to undo the offset after sorting
		int max_val = result.max_val+1;
        // Get the new max index
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(particles_i), reducer);
        auto keys = particles_i;

        // Create Comparator(number of bins, lowest val, highest val)
        using key_type = decltype(keys);
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(np, result.min_val, result.max_val);

        // Create permutation View
        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
        bin_sort.create_permute_vector();

        // Sort particle data. 
        // If using LayoutLeft we can save memory by sorting each particle variable separately.
		if(std::is_same<Kokkos::LayoutLeft, k_particles_t::array_layout>::value) {
			for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
				auto sub_view = Kokkos::subview(particles, Kokkos::ALL, i);
				bin_sort.sort(sub_view);
			}
		} else {
          bin_sort.sort(particles);
		}
        // Sort particle indices
        bin_sort.sort(particles_i);

        // Remove offset from indices
        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
			while(particles_i(i) > max_val)
				particles_i(i) -= max_val;
		});
    }
};

template <typename Policy = DefaultSort>
struct ParticleSorter : private Policy {
    using Policy::sort;
    using Policy::strided_sort;
};

#endif //guard
