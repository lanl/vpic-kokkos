#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

#include <Kokkos_Sort.hpp>
#include <Kokkos_DualView.hpp>
#include "../vpic/kokkos_helpers.h"
#include "../vpic/kokkos_tuning.hpp"

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

struct min_max_functor_u64 {
  typedef Kokkos::MinMaxScalar<Kokkos::View<uint64_t*>::non_const_value_type> minmax_scalar;
  Kokkos::View<uint64_t*> view;
  min_max_functor_u64(const Kokkos::View<uint64_t*>& view_) : view(view_) {}
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
    static void standard_sort(
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

        // Create comparator
        using key_type = decltype(keys);
        //using Comparator = Casted_BinOp1D<key_type>;
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(num_bins, 0, num_bins);

        // Sort and make permutation View
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
    }

    static void species_sort(
            species_t* sp,
            const int32_t num_bins
    )
    {
        k_particles_t& particles = sp->k_p_d;
        k_particles_i_t& particles_i = sp->k_p_i_d;
        int np = sp->np;

        // Try grab the index's for a permute key
        //int pi = particle_var::pi; // FIXME: can you really not pass an enum in??
        //auto keys = Kokkos::subview(particles, Kokkos::ALL, pi);
        auto keys = particles_i;

        // TODO: we can tighten the bounds on this to avoid ghosts

        // Create comparator
        using key_type = decltype(keys);
        //using Comparator = Casted_BinOp1D<key_type>;
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(num_bins, 0, num_bins);

        // Sort and make permutation View
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

        // Sort annotations
        for(int i=0; i<sp->num_annotations.nint_vars; i++) {
          auto sub_view = Kokkos::subview(sp->annotations_d.i32, Kokkos::ALL, i);
          bin_sort.sort(sub_view);
        }
        for(int i=0; i<sp->num_annotations.nint64_vars; i++) {
          auto sub_view = Kokkos::subview(sp->annotations_d.i64, Kokkos::ALL, i);
          bin_sort.sort(sub_view);
        }
        for(int i=0; i<sp->num_annotations.nfloat_vars; i++) {
          auto sub_view = Kokkos::subview(sp->annotations_d.f32, Kokkos::ALL, i);
          bin_sort.sort(sub_view);
        }
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
        Kokkos::View<uint64_t*> keys("Temp keys", particles_i.extent(0));
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
          keys(i) = static_cast<uint64_t>(particles_i(i)) + count*(result.max_val+1);
        });
        // Save the max particle index to undo the offset after sorting
        // Get the new max index
        Kokkos::MinMaxScalar<Kokkos::View<uint64_t*>::non_const_value_type> result_u64;
        Kokkos::MinMax<Kokkos::View<uint64_t*>::non_const_value_type> reducer_u64(result_u64);
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor_u64(keys), reducer_u64);

        // Create Comparator(number of bins, lowest val, highest val)
        using key_type = decltype(keys);
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(np, result_u64.min_val, result_u64.max_val);

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
    }

    static void tiled_sort(
            k_particles_t particles,
            k_particles_i_t particles_i,
            const int32_t np,
            const int32_t num_bins,
            const int32_t tile_size   // # of cells per tile
    )
    {
        // Create permute view by taking index view and adding offsets such that we get
        // 1,1,2,2,3,3,1,1,2,2,3,3 
        Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> result;
        Kokkos::MinMax<Kokkos::View<int*>::non_const_value_type> reducer(result);
        Kokkos::View<int*> key_view("sorting keys", particles_i.extent(0));
        Kokkos::View<int*> bin_counter("Counter for updating keys", num_bins);
        Kokkos::deep_copy(key_view, particles_i);
        Kokkos::deep_copy(bin_counter, 0);
        // Find max and min particle index
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(particles_i), reducer);
        // Count number of particles in each cell and add an offset 
        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
          int count = Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1);
          key_view(i) += (result.max_val+1)*(count/tile_size);
        });
        // Get the new max index
        Kokkos::parallel_reduce("Get min/max bin post update", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(key_view), reducer);
        auto keys = key_view;

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
        }

    static void tiled_strided_sort(
            k_particles_t particles,
            k_particles_i_t particles_i,
            const int32_t np,
            const int32_t num_bins,
            const int32_t tile_size   // # of cells per tile
    )
    {
        // Create permute view by taking index view and adding offsets such that we get
        // 1,2,3,1,2,3,1,2,3 
        Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> result;
        Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> nppc_result;
        Kokkos::MinMax<Kokkos::View<int*>::non_const_value_type> reducer(result);
        Kokkos::MinMax<Kokkos::View<int*>::non_const_value_type> nppc_reducer(nppc_result);
        Kokkos::View<int*> key_view("sorting keys", particles_i.extent(0));
        Kokkos::View<int*> bin_counter("Counter for updating keys", num_bins);
        // Find max and min particle index
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(particles_i), reducer);
        Kokkos::deep_copy(key_view, particles_i);
        Kokkos::deep_copy(bin_counter, 0);
        // Count number of particles in each cell
        Kokkos::parallel_for("get max nppc", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
          Kokkos::atomic_increment(&(bin_counter(key_view(i))));
        });
        // Find the max and min number of particles per cell
        Kokkos::parallel_reduce("Get max/min nppc", Kokkos::RangePolicy<>(0,num_bins), 
          min_max_functor(bin_counter), nppc_reducer); 
        // Reset bin_counter
        Kokkos::deep_copy(bin_counter, 0);
        // Update particle indices 
        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
          int count = Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1);
          int chunk_size = tile_size*nppc_result.max_val;
          int chunk = (key_view(i)-result.min_val)/tile_size;
          int min_idx = result.min_val + chunk*tile_size;
          int offset = count*nppc_result.max_val;
          key_view(i) += chunk*chunk_size + offset - min_idx + 1;
        });
        // Find smallest and largest index
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(key_view), reducer);
        auto keys = key_view;

        // Create comparator
        using key_type = decltype(keys);
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(np, result.min_val, result.max_val);

        // Sort and create permutation View
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
    }
};

template <typename Policy = DefaultSort>
struct ParticleSorter : private Policy {
  using Policy::standard_sort;
  using Policy::strided_sort;
  using Policy::tiled_sort;
  using Policy::tiled_strided_sort;
  using Policy::species_sort;

  void sort(k_particles_t particles, k_particles_i_t particles_i, const int32_t np, const int num_bins) {
#ifdef SORT_TILE_SIZE // strided_tiled_sort or tiled_strided_sort
    SORT(particles, particles_i, np, num_bins, SORT_TILE_SIZE);
#else // standard_sort or strided_sort
    SORT(particles, particles_i, np, num_bins);
#endif
  }

  void sort(species_t* species, const int num_bins) {
    species_sort(species, num_bins);
  }
};

#endif //guard
