#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

#include <Kokkos_Sort.hpp>
#include <Kokkos_DualView.hpp>
#include "../vpic/kokkos_helpers.h"
#include "../vpic/kokkos_tuning.hpp"
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
std::cout << "key_view len: " << key_view.extent(0) << std::endl;
std::cout << "particles_i len: " << particles_i.extent(0) << std::endl;
std::cout << "np: " << np << std::endl;
std::cout << "num_bins: " << num_bins << std::endl;
std::cout << "f32_scratch len: " << f32_scratch.size() << std::endl;
std::cout << "i32_scratch len: " << i32_scratch.size() << std::endl;
std::cout << "BinOp1D: (" << comp.max_bins_ << ", " << comp.mul_ << ", " << comp.min_ << ")\n";
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

//struct DefaultSort {
//    enum SortMethod {
//        standard_sort;
//        strided_sort;
//        tiled_sort;
//        tiled_strided_sort;
//    };
//
//    constexpr SortMethod method = SORT;
//    if constexpr (method == SortMethod::standard_sort) {
//      using key_type = k_particles_i_t;
//    } else {
//      using key_type = Kokkos::View<size_t*>;
//    }
////#if SORT == standard_sort
////    using key_type = k_particles_i_t;
////#else
////    using key_type = Kokkos::View<size_t*>;
////#endif
//    using Comparator = CustomBinOp1D<key_type>;
//
//    Kokkos::View<size_t*> sort_keys;
//    Kokkos::View<int*> bin_counter;
//    int max_bins = 0;
//    size_t min_val = ULLONG_MAX;
//    size_t max_val = 0;
//    CustomBinSort<key_type, Comparator>* bin_sort;
//    Kokkos::View<float*, k_particles_i_t::device_type> f32_scratch;
//    Kokkos::View<int*, k_particles_i_t::device_type>   i32_scratch;
//
//    DefaultSort() {
//        sort_keys   = Kokkos::View<size_t*>("Keys", 1);
//        bin_counter = Kokkos::View<int*>("bin counter", 1);
//        f32_scratch = Kokkos::View<float*>("Float scratch", 1);
//        i32_scratch = Kokkos::View<int*>("Int32 scratch", 1);
//#if SORT == standard_sort
//        bin_sort = new CustomBinSort<key_type, Comparator>(k_particles_i_t("temp particles_i", 1), CustomBinOp1D<key_type>(1, 0, 1));
//#else
//        bin_sort = new CustomBinSort<key_type, Comparator>(sort_keys, CustomBinOp1D<key_type>(1, 0, 1));
//#endif
//    }
//
//    ~DefaultSort() {
//        delete bin_sort;
//    }
//
//    void resize(const size_t np, const size_t nbins) {
//        if(sort_keys.extent(0) < np) {
//            Kokkos::resize(sort_keys, np);
//            Kokkos::resize(f32_scratch, np);
//            Kokkos::resize(i32_scratch, np);
//        }
//        if(bin_counter.extent(0) < nbins)
//            Kokkos::resize(bin_counter, nbins);
//        Comparator comp(nbins, 0, nbins);
//#if SORT == standard_sort
//        bin_sort->reset(Kokkos::DefaultExecutionSpace(), k_particles_i_t("temp particles_i", 1), 0, np, comp, 0);
//#else
//        bin_sort->reset(Kokkos::DefaultExecutionSpace(), sort_keys, 0, np, comp, 0);
//#endif
//    }
//
//    // TODO: should the sort interface just take the sp?
//    void standard_sort(
//            k_particles_t particles,
//            k_particles_i_t particles_i,
//            const size_t np,
//            const size_t num_bins
//    )
//    {
//        // Resize scratch views
//        resize(np, num_bins);
//
//        // Get subset of particle indices as keys
//        auto keys = Kokkos::subview(particles_i, Kokkos::make_pair<size_t,size_t>(0, np));
//        
//        // Create comparator
//        Comparator comp(num_bins, 0, num_bins);
//
//        // Sort and make permutation View
//        int sort_within_bins = 0;
//        bin_sort->reset(Kokkos::DefaultExecutionSpace(), keys, 0, np, comp, sort_within_bins);
//        bin_sort->create_permute_vector();
//
//        // Sort particle data. 
//        // If using LayoutLeft we can save memory by sorting each particle variable separately.
//        auto f32_subview = Kokkos::subview(f32_scratch, Kokkos::make_pair<size_t,size_t>(0, np));
//        for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
//          auto sub_view = Kokkos::subview(particles, Kokkos::ALL, i);
//          bin_sort->sort_scratch(Kokkos::DefaultExecutionSpace(), sub_view, f32_subview, 0, keys.extent(0));
//        }
//
//        // Sort particle indices
//        auto i32_subview = Kokkos::subview(i32_scratch, Kokkos::make_pair<size_t,size_t>(0, np));
//        bin_sort->sort_scratch(Kokkos::DefaultExecutionSpace(), particles_i, i32_subview, 0, keys.extent(0));
//    }
//
//    void strided_sort(
//            k_particles_t particles,
//            k_particles_i_t particles_i,
//            const size_t np,
//            const size_t num_bins
//    )
//    {
//        // Create permute view by taking index view and adding offsets such that we get
//        // 1,2,3,1,2,3,1,2,3 instead of 1,1,1,2,2,2,3,3,3 
//        
//        // Resize scratch views
//        resize(np, num_bins);
//
//        //Kokkos::View<uint64_t*> keys("Temp keys", particles_i.extent(0));
//        Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> result;
//        Kokkos::MinMax<Kokkos::View<int*>::non_const_value_type> reducer(result);
//        // Find max and min particle index
//        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
//          min_max_functor(particles_i), reducer);
//        Kokkos::View<size_t*> bin_counter("Counter for updating keys", num_bins);
//        Kokkos::deep_copy(bin_counter, 0);
//        // Count number of particles in each cell and add an offset 
//        // (current number of particles in cell multiplied by the largest index)
//        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<size_t>(0, np), KOKKOS_LAMBDA(const size_t i) {
//          size_t count = Kokkos::atomic_fetch_add(&(bin_counter(particles_i(i))), 1);
//          sort_keys(i) = static_cast<size_t>(particles_i(i)) + count*(result.max_val+1);
//        });
//        // Save the max particle index to undo the offset after sorting
//        // Get the new max index
//        Kokkos::MinMaxScalar<Kokkos::View<size_t*>::non_const_value_type> result_u64;
//        Kokkos::MinMax<Kokkos::View<size_t*>::non_const_value_type> reducer_u64(result_u64);
//        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<size_t>(0,particles_i.extent(0)), 
//          min_max_functor(sort_keys), reducer_u64);
//
//        // Create Comparator(number of bins, lowest val, highest val)
//        using key_type = decltype(sort_keys);
//        using Comparator = Kokkos::BinOp1D<key_type>;
//        Comparator comp(np, result_u64.min_val, result_u64.max_val);
//
//        // Create permutation View
//        int sort_within_bins = 0;
//        Kokkos::BinSort<key_type, Comparator> bin_sort(sort_keys, 0, np, comp, sort_within_bins );
//        bin_sort.create_permute_vector();
//
//        // Sort particle data. 
//        // If using LayoutLeft we can save memory by sorting each particle variable separately.
//        if(std::is_same<Kokkos::LayoutLeft, k_particles_t::array_layout>::value) {
//          for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
//            auto sub_view = Kokkos::subview(particles, Kokkos::ALL, i);
//            bin_sort.sort(sub_view);
//          }
//        } else {
//          bin_sort.sort(particles);
//        }
//        // Sort particle indices
//        bin_sort.sort(particles_i);
//    }
//
//    void tiled_sort(
//            k_particles_t particles,
//            k_particles_i_t particles_i,
//            const size_t np,
//            const size_t num_bins,
//            const int32_t tile_size   // # of cells per tile
//    )
//    {
//        // Create permute view by taking index view and adding offsets such that we get
//        // 1,1,2,2,3,3,1,1,2,2,3,3 
//        Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> result;
//        Kokkos::MinMax<Kokkos::View<int*>::non_const_value_type> reducer(result);
//        Kokkos::View<int*> key_view("sorting keys", particles_i.extent(0));
//        Kokkos::View<size_t*> bin_counter("Counter for updating keys", num_bins);
//        Kokkos::deep_copy(key_view, particles_i);
//        Kokkos::deep_copy(bin_counter, 0);
//        // Find max and min particle index
//        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
//          min_max_functor(particles_i), reducer);
//        // Count number of particles in each cell and add an offset 
//        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const size_t i) {
//          size_t count = Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1);
//          key_view(i) += (result.max_val+1)*(count/tile_size);
//        });
//        // Get the new max index
//        Kokkos::parallel_reduce("Get min/max bin post update", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
//          min_max_functor(key_view), reducer);
//        auto keys = key_view;
//
//        // Create Comparator(number of bins, lowest val, highest val)
//        using key_type = decltype(keys);
//        using Comparator = Kokkos::BinOp1D<key_type>;
//        Comparator comp(np, result.min_val, result.max_val);
//
//        // Create permutation View
//        int sort_within_bins = 0;
//        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
//        bin_sort.create_permute_vector();
//
//        // Sort particle data. 
//        // If using LayoutLeft we can save memory by sorting each particle variable separately.
//        if(std::is_same<Kokkos::LayoutLeft, k_particles_t::array_layout>::value) {
//          for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
//            auto sub_view = Kokkos::subview(particles, Kokkos::ALL, i);
//            bin_sort.sort(sub_view);
//          }
//        } else {
//          bin_sort.sort(particles);
//        }
//        // Sort particle indices
//        bin_sort.sort(particles_i);
//    }
//
//    void tiled_strided_sort(
//            k_particles_t particles,
//            k_particles_i_t particles_i,
//            const size_t np,
//            const size_t num_bins,
//            const int32_t tile_size   // # of cells per tile
//    )
//    {
//        // Create permute view by taking index view and adding offsets such that we get
//        // 1,2,3,1,2,3,1,2,3 
//        Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> result;
//        Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> nppc_result;
//        Kokkos::MinMax<Kokkos::View<int*>::non_const_value_type> reducer(result);
//        Kokkos::MinMax<Kokkos::View<int*>::non_const_value_type> nppc_reducer(nppc_result);
//        Kokkos::View<int*> key_view("sorting keys", particles_i.extent(0));
//        Kokkos::View<int*> bin_counter("Counter for updating keys", num_bins);
//        // Find max and min particle index
//        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
//          min_max_functor(particles_i), reducer);
//        Kokkos::deep_copy(key_view, particles_i);
//        Kokkos::deep_copy(bin_counter, 0);
//        // Count number of particles in each cell
//        Kokkos::parallel_for("get max nppc", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const size_t i) {
//          Kokkos::atomic_inc(&(bin_counter(key_view(i))));
//        });
//        // Find the max and min number of particles per cell
//        Kokkos::parallel_reduce("Get max/min nppc", Kokkos::RangePolicy<>(0,num_bins), 
//          min_max_functor(bin_counter), nppc_reducer); 
//        const size_t chunk_size = tile_size*(nppc_result.max_val+1);
//        // Reset bin_counter
//        Kokkos::deep_copy(bin_counter, 0);
//        // Update particle indices 
//        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const size_t i) {
//          const size_t count = Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1);
//          const size_t chunk_idx = (key_view(i)-(result.min_val))/tile_size;
//          key_view(i) += chunk_idx*chunk_size + count*tile_size - result.min_val;
//        });
//        // Find smallest and largest index
//        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,particles_i.extent(0)), 
//          min_max_functor(key_view), reducer);
//        auto keys = key_view;
//
//        // Create comparator
//        using key_type = decltype(keys);
//        using Comparator = Kokkos::BinOp1D<key_type>;
//        Comparator comp(np, result.min_val, result.max_val);
//
//        // Sort and create permutation View
//        int sort_within_bins = 1;
//        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
//        bin_sort.create_permute_vector();
//
//        // Sort particle data. 
//        // If using LayoutLeft we can save memory by sorting each particle variable separately.
//        if(std::is_same<Kokkos::LayoutLeft, k_particles_t::array_layout>::value) {
//          for(int i=0; i<PARTICLE_VAR_COUNT; i++) {
//            auto sub_view = Kokkos::subview(particles, Kokkos::ALL, i);
//            bin_sort.sort(sub_view);
//          }
//        } else {
//          bin_sort.sort(particles);
//        }
//          // Sort particle indices
//          bin_sort.sort(particles_i);
//        }
//};

////template <typename Policy = DefaultSort>
////template <typename Policy = PreAllocSorter<k_particles_i_t>>
//template <typename Policy = DefaultSorter<k_particles_i_t>>
//struct ParticleSorter : public Policy {
//  //using Policy::standard_sort;
//  //using Policy::strided_sort;
//  //using Policy::tiled_sort;
//  //using Policy::tiled_strided_sort;
//
//  void sort(k_particles_t particles, k_particles_i_t particles_i, const size_t np, const size_t num_bins) {
//    //Policy::sort(particles, particles_i, np, num_bins);
//    Policy::sort(particles_i, particles, particles_i, np, num_bins);
////#ifdef SORT_TILE_SIZE // strided_tiled_sort or tiled_strided_sort
////    SORT(particles, particles_i, np, num_bins, SORT_TILE_SIZE);
////#else // standard_sort or strided_sort
////    SORT(particles, particles_i, np, num_bins);
////#endif
//  }
//};

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
