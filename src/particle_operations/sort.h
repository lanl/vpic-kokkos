#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

#include <Kokkos_Sort.hpp>
#include "../vpic/kokkos_helpers.h"
#include "../vpic/kokkos_tuning.hpp"
#include "custom_binsort.hpp"
//#include <concepts>

///** @concept SorterConcept
// *  @brief Concept outlining the required member functions for a Sorter class 
// *
// *  Requires resize(...) to adjust any internal View data. Important for cases 
// *  where memory allocations/deallocations are expensive. The sort(...) 
// *  function performs the actual particle sorting.
// */
//template<typename T, typename KeyViewType>
//concept SorterConcept = requires(T a, KeyViewType& _key_view, 
//                                 k_particles_t& part, 
//                                 k_particles_i_t& part_i, 
//                                 const size_t _np, const size_t _num_bins, 
//                                 CustomBinOp1D<KeyViewType> _comp,
//                                 bool _sort_in_bins) {
//  { a.sort(_key_view, part, part_i, _np, _num_bins, _comp, _sort_in_bins) };
//  { a.resize(_np, _num_bins) };
//};
//
///** @concept ParticleReorderer
// *  @brief Concept outlining the required member functions for reordering keys
// *
// *  Requires resize(...) to adjust any internal View data. Important for cases 
// *  where memory allocations/deallocations are expensive. The reorder(...) 
// *  function takes particle data and returns a key View with adjusted keys for 
// *  more control over sorting. The get_bin_op() function returns a comparator 
// *  to use with Kokkos sort.
// */
//template<typename T>
//concept ParticleReorderer = requires(T a, k_particles_t _part, 
//                                     k_particles_i_t _part_i, 
//                                     const size_t _np, const size_t _nbins,
//                                     const uint32_t _tile_size) {
//  { a.resize(_np, _nbins) };
//  { a.reorder(_part, _part_i, 
//              _np, _nbins, _tile_size) } -> std::same_as<typename T::key_type>;
//  { a.get_bin_op() } -> std::same_as<CustomBinOp1D<typename T::key_type> >;
//};

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
    if(view(i) < minmax.min_val) minmax.min_val = view(i);
    if(view(i) > minmax.max_val) minmax.max_val = view(i);
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

    // Sort particle data using subviews to reduce memory usage. 
    for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
      auto sub_view = Kokkos::subview(particles, Kokkos::ALL, i);
      bin_sort.sort(sub_view, 0, np);
    }
    // Sort particle indices
    auto cell_ids = Kokkos::subview(particles_i, Kokkos::make_pair<size_t,size_t>(0, np));
    bin_sort.sort(cell_ids);
  }
};

/**
 * @brief Sort particles using the sort_by_key interface
 *
 * Allocates a permutation View and uses sort_by_key to get the correct
 * permutation indices. Applies permutation to remaining particle data.
 * The sort_by_key function will use backend specific sort functions like
 * thrust.
 */
template<typename KeyViewType>
struct SortByKeySorter {
  using Comparator = CustomBinOp1D<KeyViewType>;

  // No need to resize anything. Everything is allocated on the fly
  void resize(const size_t np, const size_t nbins) {}

  // TODO: should the sort interface just take the sp?
  void sort(KeyViewType& key_view,
            k_particles_t& particles,
            k_particles_i_t& particles_i,
            const size_t np,
            const size_t num_bins,
            Comparator comp,
            bool sort_within_bins=false
  )
  {
    auto np_range = Kokkos::make_pair<size_t,size_t>(0, np);
    auto np_policy = Kokkos::RangePolicy<size_t>(0, np);

    // Get subset of particle indices as keys
    auto keys = Kokkos::subview(key_view, np_range);

    Kokkos::View<size_t*> permute_view("Permutation view", np);
    Kokkos::parallel_for("Iota", np_policy, KOKKOS_LAMBDA(const size_t i) {
      permute_view(i) = i;
    });

    Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), 
                                      keys, permute_view);

    // Sort particle data. 
    Kokkos::View<float*> f32_scratch("particles scratch", np);
    for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
      auto sub_view = Kokkos::subview(particles, np_range, i);
      Kokkos::deep_copy(f32_scratch, sub_view);
      Kokkos::parallel_for("Permute particle var", np_policy, 
        KOKKOS_LAMBDA(const size_t idx) {
        particles(idx, i) = f32_scratch(permute_view(idx));
      });
    }

    // Sort particle indices. If the keys haven't been reordered then 
    // particles_i will be sorted by the initial sort_by_key call.
    if constexpr (!std::is_same_v<KeyViewType, k_particles_i_t>) {
      auto cell_ids = Kokkos::subview(particles_i, np_range);
      k_particles_i_t scratch("particles_i scratch", np);
      Kokkos::deep_copy(scratch, cell_ids);
      Kokkos::parallel_for("Permute cell indices", np_policy, 
        KOKKOS_LAMBDA(const size_t i) {
        particles_i(i) = scratch(permute_view(i));
      });
    }
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
  using device_type = k_particles_i_t::device_type;
  using exec_space = typename KeyViewType::execution_space;
  using Comparator = CustomBinOp1D<KeyViewType>;

  CustomBinSort<KeyViewType, Comparator>* bin_sort;
  Kokkos::View<float*, device_type> scratch;
  KeyViewType temp_keys;

  PreAllocSorter() {
    scratch = Kokkos::View<float*>("Float scratch", 1);
    temp_keys = KeyViewType("temp particles_i", 1);
    CustomBinOp1D<KeyViewType> bin_op(1, 0, 1);
    bin_sort = new CustomBinSort<KeyViewType, Comparator>(temp_keys, 0, 1, bin_op);
  }

  ~PreAllocSorter() {
    delete bin_sort;
  }

  /**
   * @brief Resize scratch and sorting Views
   */
  void resize(const size_t np, const size_t nbins) {
    if(scratch.extent(0) < np) {
      Kokkos::resize(scratch, np);
    }
    Comparator comp(nbins, 0, np);
    bin_sort->reset(exec_space(), temp_keys, 0, np, comp, 0);
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
    auto subview_pair = Kokkos::make_pair<size_t,size_t>(0, np);
    auto keys = Kokkos::subview(key_view, subview_pair);

    // Sort and make permutation View
    bin_sort->reset(exec_space(), keys, 0, np, comp, sort_within_bins);
    bin_sort->create_permute_vector();

    // Sort particle data. 
    for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
      auto sub_view = Kokkos::subview(particles, subview_pair, i);
      bin_sort->sort_scratch(exec_space(), sub_view, scratch, 0, np);
    }

    // Sort particle indices
    using i32_scratch=Kokkos::View<int*, device_type, Kokkos::MemoryUnmanaged>;
    i32_scratch int_scratch((int*)(scratch.data()), np);
    bin_sort->sort_scratch(exec_space(), particles_i, int_scratch, 0, np);
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
                          const size_t num_bins,
                          const uint32_t tile_size=1   // # of cells per tile
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
    if(bin_counter.extent(0) < nbins+1)
      Kokkos::resize(bin_counter, nbins+1);
  }

  // TODO: should the sort interface just take the sp?
  Kokkos::View<size_t*> reorder(
                                k_particles_t particles,
                                k_particles_i_t particles_i,
                                const size_t np,
                                const size_t num_bins,
                                const uint32_t tile_size=1 // # cells per tile
  )
  {
    // Resize scratch views
    resize(np, num_bins);

    auto np_range = Kokkos::RangePolicy<size_t>(0,np);

    Kokkos::MinMaxScalar<int> result;
    // Find max particle index
    size_t max_cell = 0;
    Kokkos::parallel_reduce("Get max cell ID", np_range, 
      KOKKOS_LAMBDA(const size_t& i, size_t& max_cell_id) {
      if(particles_i(i) > max_cell_id)
        max_cell_id = particles_i(i);
    }, Kokkos::Max<size_t>(max_cell));

    Kokkos::deep_copy(bin_counter, 0);
    // Count number of particles in each cell and add an offset 
    // (current number of particles in cell multiplied by the largest index)
    Kokkos::parallel_for("Update keys", np_range, 
      KOKKOS_CLASS_LAMBDA(const size_t i) {
      size_t count = Kokkos::atomic_fetch_inc(&(bin_counter(particles_i(i))));
      sort_keys(i) = static_cast<size_t>(particles_i(i)) + count*(max_cell+1);
    });
    // Save the max particle index to undo the offset after sorting
    // Get the new max index
    Kokkos::MinMaxScalar<size_t> result_u64;
    Kokkos::parallel_reduce("Get min/max bin", np_range, 
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
          k_particles_t part,
          k_particles_i_t part_i,
          const size_t np,
          const size_t num_bins,
          const uint32_t tile_size   // # of cells per tile
  )
  {
    // Resize scratch views
    resize(np, num_bins);

    auto np_range = Kokkos::RangePolicy<size_t>(0,np);

    // Find max particle index
    size_t max_cell = 0;
    Kokkos::parallel_reduce("Get max cell ID", np_range, 
    KOKKOS_LAMBDA(const size_t& i, size_t& max_cell_id) {
      if(part_i(i) > max_cell_id)
        max_cell_id = part_i(i);
    }, Kokkos::Max<size_t>(max_cell));

    Kokkos::deep_copy(bin_counter, 0);
    // Count number of particles in each cell and add an offset 
    Kokkos::parallel_for("Update keys", np_range, 
    KOKKOS_CLASS_LAMBDA(const size_t i) {
      size_t count = Kokkos::atomic_fetch_add(&(bin_counter(part_i(i))), 1);
      sort_keys(i) = static_cast<size_t>(part_i(i)) 
                   + (max_cell+1)*(count/tile_size);
    });
    // Get the new max index
    Kokkos::MinMaxScalar<size_t> key_bounds;
    Kokkos::parallel_reduce("Get min/max bin post update", np_range, 
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
          k_particles_t part,
          k_particles_i_t part_i,
          const size_t np,
          const size_t num_bins,
          const uint32_t tile_size   // # of cells per tile
  )
  {
    auto range_policy = Kokkos::RangePolicy<size_t>(0, np);
    auto bin_range_policy = Kokkos::RangePolicy<size_t>(0, num_bins);
    // Resize scratch views
    resize(np, num_bins);
    Kokkos::MinMaxScalar<int> cell_id, cell_size;
    // Find max and min particle index
    Kokkos::parallel_reduce("Get min/max bin", range_policy, 
      min_max_functor(part_i), Kokkos::MinMax<int>(cell_id));
    const int min_cell = cell_id.min_val;
    Kokkos::deep_copy(bin_counter, 0);
    // Count number of particles in each cell
    Kokkos::parallel_for("get max nppc", range_policy, 
    KOKKOS_CLASS_LAMBDA(const size_t i) {
      Kokkos::atomic_inc(&(bin_counter(part_i(i))));
    });
    // Find the max and min number of particles per cell
    Kokkos::parallel_reduce("Get max/min nppc", bin_range_policy, 
      min_max_functor(bin_counter), Kokkos::MinMax<int>(cell_size)); 
    const size_t chunk_size = tile_size 
                            * static_cast<size_t>(cell_size.max_val + 1);
    // Reset bin_counter
    Kokkos::deep_copy(bin_counter, 0);
    // Update particle indices 
    Kokkos::parallel_for("Update keys", range_policy, 
    KOKKOS_CLASS_LAMBDA(const size_t i) {
      const size_t count = Kokkos::atomic_fetch_inc(&(bin_counter(part_i(i))));
      const size_t chunk_idx = static_cast<size_t>(part_i(i) - min_cell) 
                             / tile_size;
      sort_keys(i) = static_cast<size_t>(part_i(i) - min_cell)  
                   + chunk_idx*chunk_size + count*tile_size;
    });

    // Find smallest and largest index
    Kokkos::MinMaxScalar<size_t> new_keys;
    Kokkos::parallel_reduce("Get min/max bin", range_policy, 
      min_max_functor(sort_keys), Kokkos::MinMax<size_t>(new_keys));

    min_val = new_keys.min_val;
    max_val = new_keys.max_val;
    num_bin = max_val - min_val + 1;
    return sort_keys;
  }

  Comparator get_bin_op() {
    return Comparator(num_bin, min_val, max_val);
  }
};

//template < ParticleReorderer SortOrder = DEFAULT_SORT_ORDER, 
//           SorterConcept<typename SortOrder::key_type> Sorter 
//             = DefaultSorter<typename SortOrder::key_type> >
template < typename SortOrder = DEFAULT_SORT_ORDER, 
           typename Sorter = PreAllocSorter<typename SortOrder::key_type> >
struct ParticleSorter {
  SortOrder order;
  Sorter sorter;

  void resize(const size_t np, const size_t nbins) {
    order.resize(np, nbins);
    sorter.resize(np, nbins);
  }

  void sort(k_particles_t part, k_particles_i_t part_i, 
            const size_t np, const size_t num_bins, const size_t tile_size=1) {
    auto keys = order.reorder(part, part_i, np, num_bins, tile_size);
    auto binop = order.get_bin_op();
    const bool sort_bins = false;
    sorter.sort(keys, part, part_i, np, num_bins, binop, sort_bins);
  }
};
#endif //guard
