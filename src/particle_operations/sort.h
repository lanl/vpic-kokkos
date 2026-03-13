#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

#include <Kokkos_Sort.hpp>
#include <Kokkos_DualView.hpp>
#include "../vpic/kokkos_helpers.h"
#include "../vpic/kokkos_tuning.hpp"
#include "shuffle.h"
#include "custom_binsort.hpp"

/**
 * @brief Find min and max value in a 1D View
 */
template<typename T>
struct min_max_functor {
  typedef Kokkos::MinMaxScalar<T> minmax_scalar;
  Kokkos::View<T*> view;
  min_max_functor(const Kokkos::View<T*>& view_) : view(view_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i, minmax_scalar& minmax) const {
    if(view(i) < minmax.min_val && view(i) != 0) minmax.min_val = view(i);
    if(view(i) > minmax.max_val && view(i) != 0) minmax.max_val = view(i);
  }
};

/**
 * @brief Sort particles using CustomBinSort 
 *
 * CustomBinSort is a copy of Kokkos::BinSort with modifications to use size_t
 * for indexing. Kokkos::BinSort uses int which limits the number of elements 
 * to 2^31. 
 */
template<typename KeyViewType>
struct DefaultSorter {
  using Comparator = CustomBinOp1D<KeyViewType>;

  // No need to resize anything. Everything is allocated on the fly
  void resize(const size_t np, const size_t nbins) {}

  // TODO: should the sort interface just take the sp?
  void sort(KeyViewType key_view,
            k_particles_t& particles,
            k_particles_i_t& particles_i,
            const size_t np,
            const size_t num_bins,
            Comparator comp,
            bool sort_within_bins=false
  )
  {
    // Get subset of particle indices as keys
    auto keys = Kokkos::subview(key_view, Kokkos::make_pair<size_t,size_t>(0, np));
    
    // Sort and make permutation View
    CustomBinSort<KeyViewType, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins);
    bin_sort.create_permute_vector();

    // Sort particle data. 
    for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
      auto sub_view = Kokkos::subview(particles, Kokkos::ALL, i);
      bin_sort.sort(sub_view, 0, keys.extent(0));
    }
    // Sort particle indices
    bin_sort.sort(particles_i, 0, keys.extent(0));
  }
};

/**
 * @brief Sort particles using CustomBinSort while preallocating buffers
 *
 * CustomBinSort is a copy of Kokkos::BinSort with modifications to use size_t
 * for indexing. Kokkos::BinSort uses integers which limits the number of 
 * elements to 2^31. This sorter also allows the preallocation of temporary
 * buffers by keeping the the internal sorter on the heap and letting the user
 * resize as needed.
 */
template<typename KeyViewType>
struct PreAllocSorter {
  using Comparator = CustomBinOp1D<KeyViewType>;

  CustomBinSort<KeyViewType, Comparator>* bin_sort;
  Kokkos::View<float*, k_particles_i_t::device_type> f32_scratch;
  Kokkos::View<int*, k_particles_i_t::device_type>   i32_scratch;

  PreAllocSorter() {
    f32_scratch = Kokkos::View<float*>("Float scratch", 1);
    i32_scratch = Kokkos::View<int*>("Int32 scratch", 1);
    KeyViewType temp("temp particles_i", 1);
    CustomBinOp1D<KeyViewType> temp_bin_op(1, 0, 1);
    bin_sort = new CustomBinSort<KeyViewType, Comparator>(temp, 0, 1, temp_bin_op);
  }

  ~PreAllocSorter() {
    delete bin_sort;
  }

  /**
   * @brief Resize scratch and sorting Views
   */
  void resize(const size_t np, const size_t nbins) {
    if(f32_scratch.extent(0) < np) {
        Kokkos::resize(f32_scratch, np);
        Kokkos::resize(i32_scratch, np);
    }
    Comparator comp(nbins, 0, nbins);
    KeyViewType temp("temp particles_i", 1);
    bin_sort->reset(Kokkos::DefaultExecutionSpace(), temp, 0, np, comp, 0);
  }

  // TODO: should the sort interface just take the sp?
  void sort(
            KeyViewType& key_view,
            k_particles_t& particles,
            k_particles_i_t& particles_i,
            const size_t np,
            const size_t num_bins,
            Comparator comp,
            bool sort_within_bins=false
  )
  {
    // Resize scratch views
    resize(np, num_bins);
    
    // Get subset of particle indices as keys
    auto keys = Kokkos::subview(key_view, Kokkos::make_pair<size_t,size_t>(0, np));

    // Sort and make permutation View
    bin_sort->reset(Kokkos::DefaultExecutionSpace(), keys, static_cast<size_t>(0), np, comp, sort_within_bins);
    bin_sort->create_permute_vector();

    // Sort particle data. 
    for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
      auto sub_view = Kokkos::subview(particles, Kokkos::ALL, i);
      bin_sort->sort_scratch(Kokkos::DefaultExecutionSpace(), sub_view, f32_scratch, 0, np);
    }

    // Sort particle indices
    bin_sort->sort_scratch(Kokkos::DefaultExecutionSpace(), particles_i, i32_scratch, 0, np);
  }
};

/**
 * @brief Standard sort doesn't need a special particle reordering 
 */
struct StandardSortOrder {
  size_t min_val = 0;
  size_t max_val = 0;
  size_t num_bin = 0;

  using key_type = k_particles_i_t;
  using Comparator = CustomBinOp1D<key_type>;

  void resize(const size_t np, const size_t nbins) {}

  // TODO: should the sort interface just take the sp?
  k_particles_i_t reorder(
                          k_particles_t particles,
                          k_particles_i_t particles_i,
                          const size_t np,
                          const size_t num_bins
  )
  {
    min_val = 0;
    max_val = num_bins;
    num_bin = num_bins;
    return particles_i;
  }

  Comparator get_bin_op() {
    return Comparator(num_bin, min_val, max_val);
  }
};

/**
 * @brief Create permute view by taking index view and adding offsets such that
 *  we get 1,2,3,1,2,3,1,2,3 instead of 1,1,1,2,2,2,3,3,3 
 */
struct StridedSortOrder {
  size_t min_val;
  size_t max_val;
  size_t num_bin;
  Kokkos::View<size_t*> sort_keys;
  Kokkos::View<int*> bin_counter;

  using key_type = Kokkos::View<size_t*>;
  using Comparator = CustomBinOp1D<key_type>;

  StridedSortOrder() {
    sort_keys   = Kokkos::View<size_t*>("Keys", 1);
    bin_counter = Kokkos::View<int*>("bin counter", 1);
  }

  void resize(const size_t np, const size_t nbins) {
    if(sort_keys.extent(0) < np) 
      Kokkos::resize(sort_keys, np);
    if(bin_counter.extent(0) < nbins)
      Kokkos::resize(bin_counter, nbins+1);
  }

  // TODO: should the sort interface just take the sp?
  Kokkos::View<size_t*> reorder(
                                k_particles_t particles,
                                k_particles_i_t particles_i,
                                const size_t np,
                                const size_t num_bins
  )
  {
    // Resize scratch views
    resize(np, num_bins);

    Kokkos::MinMaxScalar<int> result;
    // Find max particle index
    size_t max_cell = 0;
    Kokkos::parallel_reduce("Get max cell ID", Kokkos::RangePolicy<size_t>(0,np), 
    KOKKOS_LAMBDA(const size_t& i, size_t& max_cell_id) {
      if(particles_i(i) > max_cell_id)
        max_cell_id = particles_i(i);
    }, Kokkos::Max<size_t>(max_cell));

    Kokkos::deep_copy(bin_counter, 0);
    // Count number of particles in each cell and add an offset 
    // (current number of particles in cell multiplied by the largest index)
    Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<size_t>(0, np), KOKKOS_CLASS_LAMBDA(const size_t i) {
      size_t count = Kokkos::atomic_fetch_inc(&(bin_counter(particles_i(i))));
      sort_keys(i) = static_cast<size_t>(particles_i(i)) + count*(max_cell+1);
    });
    // Save the max particle index to undo the offset after sorting
    // Get the new max index
    Kokkos::MinMaxScalar<size_t> result_u64;
    Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<size_t>(0,particles_i.extent(0)), 
      min_max_functor(sort_keys), Kokkos::MinMax<size_t>(result_u64));
    min_val = result_u64.min_val;
    max_val = result_u64.max_val;
    num_bin = np;
    return sort_keys;
  }

  Comparator get_bin_op() {
    return Comparator(num_bin, min_val, max_val);
  }
};

/**
 * @brief Create permute view by taking index view and adding offsets such that
 *  we get 1,1,2,2,3,3,1,1,2,2,3,3  instead of 1,1,1,1,2,2,2,2,3,3,3,3
 */
struct TiledSortOrder {
  size_t min_val;
  size_t max_val;
  size_t num_bin;
  Kokkos::View<size_t*> sort_keys;
  Kokkos::View<int*> bin_counter;

  using key_type = Kokkos::View<size_t*>;
  using Comparator = CustomBinOp1D<key_type>;

  TiledSortOrder() {
    sort_keys   = Kokkos::View<size_t*>("Keys", 1);
    bin_counter = Kokkos::View<int*>("bin counter", 1);
  }

  void resize(const size_t np, const size_t nbins) {
    if(sort_keys.extent(0) < np) 
      Kokkos::resize(sort_keys, np);
    if(bin_counter.extent(0) < nbins)
      Kokkos::resize(bin_counter, nbins);
  }

  // TODO: should the sort interface just take the sp?
  Kokkos::View<size_t*> reorder(
          k_particles_t particles,
          k_particles_i_t particles_i,
          const size_t np,
          const size_t num_bins,
          const uint32_t tile_size   // # of cells per tile
  )
  {
    // Resize scratch views
    resize(np, num_bins);

    // Find max particle index
    size_t max_cell = 0;
    Kokkos::parallel_reduce("Get max cell ID", Kokkos::RangePolicy<size_t>(0,np), 
    KOKKOS_LAMBDA(const size_t& i, size_t& max_cell_id) {
      if(particles_i(i) > max_cell_id)
        max_cell_id = particles_i(i);
    }, Kokkos::Max<size_t>(max_cell));

    Kokkos::deep_copy(bin_counter, 0);
    // Count number of particles in each cell and add an offset 
    Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<size_t>(0, np), 
    KOKKOS_CLASS_LAMBDA(const size_t i) {
      size_t count = Kokkos::atomic_fetch_add(&(bin_counter(particles_i(i))), 1);
      sort_keys(i) = static_cast<size_t>(particles_i(i)) + (max_cell+1)*(count/tile_size);
    });
    // Get the new max index
    Kokkos::MinMaxScalar<size_t> key_bounds;
    Kokkos::parallel_reduce("Get min/max bin post update", Kokkos::RangePolicy<size_t>(0,np), 
      min_max_functor(sort_keys), Kokkos::MinMax<size_t>(key_bounds));

    min_val = key_bounds.min_val;
    max_val = key_bounds.max_val;
    num_bin = np;
    return sort_keys;
  }

  Comparator get_bin_op() {
    return Comparator(num_bin, min_val, max_val);
  }
};

/**
 * @brief Create permute view by taking index view and adding offsets such that
 *  we get 1,2,1,2,3,4,3,4  instead of 1,1,2,2,3,3,4,4
 */
struct TiledStridedSortOrder {
  size_t min_val;
  size_t max_val;
  size_t num_bin;
  Kokkos::View<size_t*> sort_keys;
  Kokkos::View<int*> bin_counter;

  using key_type = Kokkos::View<size_t*>;
  using Comparator = CustomBinOp1D<key_type>;

  TiledStridedSortOrder() {
    sort_keys   = Kokkos::View<size_t*>("Keys", 1);
    bin_counter = Kokkos::View<int*>("bin counter", 1);
  }

  void resize(const size_t np, const size_t nbins) {
    if(sort_keys.extent(0) < np) 
      Kokkos::resize(sort_keys, np);
    if(bin_counter.extent(0) < nbins)
      Kokkos::resize(bin_counter, nbins);
  }

  // TODO: should the sort interface just take the sp?
  Kokkos::View<size_t*> reorder(
          k_particles_t particles,
          k_particles_i_t particles_i,
          const size_t np,
          const size_t num_bins,
          const uint32_t tile_size   // # of cells per tile
  )
  {
    // Resize scratch views
    resize(np, num_bins);

    auto range_policy = Kokkos::RangePolicy<size_t>(0, np);
    auto bin_range_policy = Kokkos::RangePolicy<size_t>(0, num_bins);
    Kokkos::MinMaxScalar<int> cell_id, cell_size;
    // Find max and min particle index
    Kokkos::parallel_reduce("Get min/max bin", range_policy, 
      min_max_functor(particles_i), Kokkos::MinMax<int>(cell_id));
    const int min_cell = cell_id.min_val;
    Kokkos::deep_copy(bin_counter, 0);
    // Count number of particles in each cell
    Kokkos::parallel_for("get max nppc", range_policy, 
    KOKKOS_CLASS_LAMBDA(const size_t i) {
      Kokkos::atomic_inc(&(bin_counter(particles_i(i))));
    });
    // Find the max and min number of particles per cell
    Kokkos::parallel_reduce("Get max/min nppc", bin_range_policy, 
      min_max_functor(bin_counter), Kokkos::MinMax<int>(cell_size)); 
    const size_t chunk_size = tile_size * static_cast<size_t>(cell_size.max_val + 1);
    // Reset bin_counter
    Kokkos::deep_copy(bin_counter, 0);
    // Update particle indices 
    Kokkos::parallel_for("Update keys", range_policy, 
    KOKKOS_CLASS_LAMBDA(const size_t i) {
      const size_t count = Kokkos::atomic_fetch_inc(&(bin_counter(particles_i(i))));
      const size_t chunk_idx = static_cast<size_t>(particles_i(i) - min_cell) / tile_size;
      sort_keys(i) = static_cast<size_t>(particles_i(i) - min_cell)  
                   + chunk_idx*chunk_size + count*tile_size;
    });

    // Find smallest and largest index
    Kokkos::MinMaxScalar<size_t> new_keys;
    Kokkos::parallel_reduce("Get min/max bin", range_policy, 
      min_max_functor(sort_keys), Kokkos::MinMax<size_t>(new_keys));

    min_val = new_keys.min_val;
    max_val = new_keys.max_val;
    num_bin = np;
    return sort_keys;
  }

  Comparator get_bin_op() {
    return Comparator(num_bin, min_val, max_val);
  }
};

template < typename SortOrder = DEFAULT_SORT_ORDER, 
           typename Sorter = PreAllocSorter<typename SortOrder::key_type> >
struct ParticleSorter {
  SortOrder order;
  Sorter sorter;

  void resize(const size_t np, const size_t nbins) {
    order.resize(np, nbins);
    sorter.resize(np, nbins);
  }

  void sort(k_particles_t particles, k_particles_i_t particles_i, 
            const size_t np, const size_t num_bins, const size_t tile_size=1) {
    if constexpr (std::is_same_v<SortOrder,TiledSortOrder> ||
                  std::is_same_v<SortOrder,TiledStridedSortOrder>) {
      auto keys = order.reorder(particles, particles_i, np, num_bins, tile_size);
      auto binop = order.get_bin_op();
      bool sort_bins = true;
      sorter.sort(keys, particles, particles_i, np, num_bins, binop, sort_bins);
    } else {
      auto keys = order.reorder(particles, particles_i, np, num_bins);
      auto binop = order.get_bin_op();
      sorter.sort(keys, particles, particles_i, np, num_bins, binop);
    }
  }
};
#endif //guard
