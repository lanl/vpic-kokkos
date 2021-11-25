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
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(particles_i), reducer);
        Kokkos::View<int*> key_view("sorting keys", particles_i.extent(0));
        Kokkos::View<int*> bin_counter("Counter for updating keys", num_bins);
        Kokkos::deep_copy(key_view, particles_i);
        Kokkos::deep_copy(bin_counter, 0);
        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
          int count = Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1);
          key_view(i) += count*result.max_val;
        });
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(key_view), reducer);
        auto keys = key_view;

        using key_type = decltype(keys);
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(result.max_val-result.min_val+1, result.min_val, result.max_val);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
        bin_sort.create_permute_vector();

        bin_sort.sort(particles);
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
        Kokkos::deep_copy(key_view, particles_i);
        Kokkos::View<int*> bin_counter("Counter for updating keys", num_bins);
        Kokkos::parallel_for("init bin counters", Kokkos::RangePolicy<>(0,num_bins), KOKKOS_LAMBDA(const int i) {
          bin_counter(i) = 0;
        });
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(key_view), reducer);
        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
          int count = Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1);
          key_view(i) += result.max_val*(count/tile_size);
        });
        Kokkos::parallel_reduce("Get min/max bin post update", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(key_view), reducer);
        auto keys = key_view;

        using key_type = decltype(keys);
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(result.max_val-result.min_val+1, result.min_val, result.max_val);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
        bin_sort.create_permute_vector();

        bin_sort.sort(particles);
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
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(particles_i), reducer);
        Kokkos::deep_copy(key_view, particles_i);
        Kokkos::deep_copy(bin_counter, 0);
        Kokkos::parallel_for("get max nppc", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
          Kokkos::atomic_increment(&(bin_counter(key_view(i))));
        });
        Kokkos::parallel_reduce("Get max/min nppc", Kokkos::RangePolicy<>(0,num_bins), 
          min_max_functor(bin_counter), nppc_reducer); 
        Kokkos::deep_copy(bin_counter, 0);
        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
          int count = Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1);
          int chunk_size = tile_size*nppc_result.max_val;
          int chunk = (key_view(i)-result.min_val)/tile_size;
          int min_idx = result.min_val + chunk*tile_size;
          int offset = count*nppc_result.max_val;
          key_view(i) = chunk*chunk_size + offset + key_view(i) - min_idx + 1;
        });
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
          min_max_functor(key_view), reducer);
        auto keys = key_view;

        using key_type = decltype(keys);
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(result.max_val-result.min_val+1, result.min_val, result.max_val);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
        bin_sort.create_permute_vector();

        bin_sort.sort(particles);
        bin_sort.sort(particles_i);
    }
};

template <typename Policy = DefaultSort>
struct ParticleSorter : private Policy {
  using Policy::standard_sort;
  using Policy::strided_sort;
  using Policy::tiled_sort;
  using Policy::tiled_strided_sort;
  void sort(k_particles_t particles, k_particles_i_t particles_i, const int32_t np, const int num_bins) {
#ifdef SORT_TILE_SIZE // strided_tiled_sort or tiled_strided_sort
    SORT(particles, particles_i, np, num_bins, SORT_TILE_SIZE);
#else // standard_sort or strided_sort
    SORT(particles, particles_i, np, num_bins);
#endif
  }
};

#endif //guard
