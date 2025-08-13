#ifndef SCATTER_GATHER_SORT_H
#define SCATTER_GATHER_SORT_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

enum SortMode {
  Standard,
  Strided,
  Tiled,
  TiledStrided
};

template<class ViewType>
struct min_max_functor {
  using minmax_scalar = Kokkos::MinMaxScalar< typename ViewType::non_const_value_type >;
  ViewType view;
  min_max_functor(const ViewType& view_) : view(view_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i, minmax_scalar& minmax) const {
    if(view(i) < minmax.min_val) minmax.min_val = view(i);
    if(view(i) > minmax.max_val) minmax.max_val = view(i);
  }
};

//struct min_max_functor_u64 {
//  typedef Kokkos::MinMaxScalar<Kokkos::View<uint64_t*>::non_const_value_type> minmax_scalar;
//  Kokkos::View<uint64_t*> view;
//  min_max_functor_u64(const Kokkos::View<uint64_t*>& view_) : view(view_) {}
//  KOKKOS_INLINE_FUNCTION
//  void operator()(const size_t& i, minmax_scalar& minmax) const {
//    if(view(i) < minmax.min_val) minmax.min_val = view(i);
//    if(view(i) > minmax.max_val) minmax.max_val = view(i);
//  }
//};

template<typename KeyView>
bool verify_sort(KeyView& key_view,
                 const std::string& mode_str,
                 const uint64_t n_unique,
                 const int32_t tile_size) {
  auto key_view_host = Kokkos::create_mirror_view(key_view);
  Kokkos::deep_copy(key_view_host, key_view);
  bool success = true;
  int max = 0;
  int min = INT_MAX;
  for(size_t i=0; i<key_view.size(); i++) {
    if(key_view_host(i) > max)
      max = key_view_host(i);
    if(key_view_host(i) < min)
      min = key_view_host(i);
  }
//  printf("Max key: %d\n", max);
//  printf("Min key: %d\n", min);
  if(mode_str.compare("standard") == 0) {
    for(size_t i=1; i<key_view.size(); i++) {
      if(key_view_host(i-1) > key_view_host(i)) {
printf("Failure: (%zu): %d > %d\n", i, key_view_host(i-1), key_view_host(i));
printf("%d, %d, %d\n", key_view_host(i-1), key_view_host(i), key_view_host(i+1));
        success = false;
        break;
      }
    }
  } else if (mode_str.compare("strided") == 0) {
    for(size_t i=1; i<key_view.size(); i++) {
      if( (i % n_unique != 0) && (key_view_host(i-1) >= key_view_host(i)) ) {
        success = false;
        break;
      } else if( i % n_unique == 0 ) {
        if(key_view_host(i) != key_view_host(0)) {
          success = false;
          break;
        }
      }
    }
  } else if (mode_str.compare("tiled") == 0) {
    size_t ntiles = key_view.size() / tile_size;
    if(ntiles*tile_size < key_view.size())
      ntiles += 1;
    for(size_t i=1; i<key_view.size(); i++) {
      //size_t tile = i / tile_size;
      size_t key = i % tile_size;
      if(key == 0) {
        if(key_view_host(i-1) < key_view_host(i)) {
          success = false;
          break;
        }
      } else {
        if(key_view_host(i-1) != key_view_host(i)) {
          success = false;
          break;
        }
      }
    }
  } else if (mode_str.compare("tiled-strided") == 0) {
    size_t ntiles = key_view.size() / tile_size;
    if(ntiles*tile_size < key_view.size())
      ntiles += 1;
    for(size_t i=1; i<key_view.size(); i++) {
      size_t tile = i / tile_size;
      size_t key = i % tile_size;
      if(key == 0) {
        if((key_view_host(i) != key_view_host(i-tile_size)) && (key_view_host(i-1) >= key_view_host(i)) && (key_view_host(i-1) != max)) {
printf("Failure: (%zu) : key_view_host(i) != key_view_host(i-tile_size)) && (key_view_host(i-1) >= key_view_host(i)) \t : %d != %d && %d != %d\n", i, key_view_host(i), key_view_host(i-tile_size), key_view_host(i-1), key_view_host(i));
printf("%d, %d, %d\n", key_view_host(i-1), key_view_host(i), key_view_host(i+1));
          success = false;
          break;
        }
      } else {
        if( (key_view_host(i-1) >= key_view_host(i)) && (max-key_view_host(tile*tile_size) >= tile_size) ) {
printf("Failure: (%zu) : key_view_host(i-1) >= key_view_host(i)\t : %d >= %d\n", i, key_view_host(i-1), key_view_host(i));
printf("%d, %d, %d\n", key_view_host(i-1), key_view_host(i), key_view_host(i+1));
          success = false;
          break;
        }
      }
    }
  }
  if(success) {
    printf("%s was successful!\n", mode_str.c_str());
  } else {
    printf("%s was not successful!\n", mode_str.c_str());
  }
  return success;
}

// TODO: should the sort interface just take the sp?
template<typename KeyView>
void standard_sort(
        KeyView& key_view,
        const uint64_t n,
        const uint64_t num_bins,
        const int32_t tile_size   // # of cells per tile
)
{
    Kokkos::sort(key_view);

//    //auto vals = Kokkos::subview(val_view, Kokkos::pair(0,n));
//    //Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), key_view, vals);
//
//    // Try to grab the indices for a permute key
//    //auto keys = Kokkos::subview(key_view, Kokkos::pair(0, n));
//    auto keys = key_view;
//
//    // Create comparator
//    using key_type = decltype(keys);
//    using Comparator = Kokkos::BinOp1D<key_type>;
//    Comparator comp(num_bins, 0, num_bins);
//
//    // Sort and make permutation View
//    int sort_within_bins = 1;
//    Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, n, comp, sort_within_bins );
//    bin_sort.create_permute_vector();
//
//    // Sort keys
//    bin_sort.sort(key_view);
}

template<typename KeyView>
void strided_sort(
        KeyView& key_view,
        const uint64_t np,
        const uint64_t num_bins,
        const int32_t tile_size   // # of cells per tile
)
{
    // Create permute view by taking index view and adding offsets such that we get
    // 1,2,3,1,2,3,1,2,3 instead of 1,1,1,2,2,2,3,3,3 
//printf("Allocating %lu GB for Temp keys\n", sizeof(uint64_t)*key_view.extent(0)/1000000000LLU);
    Kokkos::View<uint64_t*> keys("Temp keys", key_view.extent(0));
    {
    Kokkos::MinMaxScalar<Kokkos::View<uint64_t*>::non_const_value_type> result;
    Kokkos::MinMax<Kokkos::View<uint64_t*>::non_const_value_type> reducer(result);
    // Find max and min key 
    Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<uint64_t>(0,key_view.extent(0)), 
      min_max_functor(key_view), reducer);
//printf("Allocating %lu GB for bin counter\n", sizeof(uint16_t)*num_bins/1000000000LLU);
    Kokkos::View<uint16_t*> bin_counter("Counter for updating keys", num_bins);
    Kokkos::deep_copy(bin_counter, 0);
    // Count number of val_view in each cell and add an offset 
    // (current number of val_view in cell multiplied by the largest key)
    Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<uint64_t>(0, np), KOKKOS_LAMBDA(const uint64_t i) {
      uint64_t count = static_cast<uint64_t>(Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1));
      keys(i) = static_cast<uint64_t>(key_view(i)) + count*(result.max_val+1);
    });
    }
    Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), keys, key_view);

//    // Save the max key to undo the offset after sorting
//    // Get the new max key 
//    Kokkos::MinMaxScalar<Kokkos::View<uint64_t*>::non_const_value_type> result_u64;
//    Kokkos::MinMax<Kokkos::View<uint64_t*>::non_const_value_type> reducer_u64(result_u64);
//    Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,key_view.extent(0)), 
//      min_max_functor_u64(keys), reducer_u64);
//
//    // Create Comparator(number of bins, lowest val, highest val)
//    using key_type = decltype(keys);
//    using Comparator = Kokkos::BinOp1D<key_type>;
//    Comparator comp(np, result_u64.min_val, result_u64.max_val);
//
//    // Create permutation View
//    int sort_within_bins = 0;
//    Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
//    bin_sort.create_permute_vector();
//
//    // Sort keys
//    bin_sort.sort(key_view);
}

template<typename KeyView>
void tiled_sort(
        KeyView& key_view,
        const uint64_t np,
        const uint64_t num_bins,
        const int32_t tile_size   // # of cells per tile
)
{
    // Create permute view by taking index view and adding offsets such that we get
    // 1,1,2,2,3,3,1,1,2,2,3,3 
    Kokkos::MinMaxScalar<Kokkos::View<uint64_t*>::non_const_value_type> result;
    Kokkos::MinMax<Kokkos::View<uint64_t*>::non_const_value_type> reducer(result);
    Kokkos::View<uint64_t*> keys("sorting keys", key_view.extent(0));
    {
    Kokkos::View<uint16_t*> bin_counter("Counter for updating keys", num_bins);
    Kokkos::deep_copy(keys, key_view);
    Kokkos::deep_copy(bin_counter, 0);
    // Find max and min key
    Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<uint64_t>(0,keys.extent(0)), 
      min_max_functor(keys), reducer);
    // Count number of val_view in each cell and add an offset 
    Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<uint64_t>(0, np), KOKKOS_LAMBDA(const uint64_t i) {
      uint64_t count = static_cast<uint64_t>(Kokkos::atomic_fetch_add(&(bin_counter(keys(i))), 1));
      keys(i) += (result.max_val+1)*(count/tile_size);
    });
    }
    Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), keys, key_view);

//    // Get the new max index
//    Kokkos::parallel_reduce("Get min/max bin post update", Kokkos::RangePolicy<>(0,keys.extent(0)), 
//      min_max_functor(keys), reducer);
//
//    // Create Comparator(number of bins, lowest val, highest val)
//    using key_type = decltype(keys);
//    using Comparator = Kokkos::BinOp1D<key_type>;
//    Comparator comp(np, result.min_val, result.max_val);
//
//    // Create permutation View
//    int sort_within_bins = 0;
//    Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
//    bin_sort.create_permute_vector();
//
//    // Sort keys
//    bin_sort.sort(key_view);
}

template<typename KeyView>
void tiled_strided_sort(
        KeyView& key_view,
        const uint64_t np,
        const uint64_t num_bins,
        const int32_t tile_size   // # of cells per tile
)
{
    // Create permute view by taking index view and adding offsets such that we get
    // 1,2,3,1,2,3,1,2,3 
    Kokkos::MinMaxScalar<typename KeyView::non_const_value_type> result;
    Kokkos::MinMaxScalar<uint16_t> nppc_result;
    Kokkos::MinMax<typename KeyView::non_const_value_type> reducer(result);
    Kokkos::MinMax<uint16_t> nppc_reducer(nppc_result);
    Kokkos::View<uint64_t*> keys("sorting keys", key_view.extent(0));
    {
    Kokkos::View<uint16_t*> bin_counter("Counter for updating keys", num_bins);
//    Kokkos::View<uint64_t*> mapping("Mapping for removing gaps", num_bins);
//    Kokkos::deep_copy(mapping, 0);
    // Find max and min key
    Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<uint64_t>(0,key_view.extent(0)), 
      min_max_functor(key_view), reducer);
//printf("Min,Max keys: %d,%d\n", result.min_val, result.max_val);
//    Kokkos::deep_copy(keys, key_view);
    Kokkos::deep_copy(bin_counter, 0);
    // Count number of val_view in each cell
    Kokkos::parallel_for("get max nppc", Kokkos::RangePolicy<uint64_t>(0, np), KOKKOS_LAMBDA(const uint64_t i) {
      Kokkos::atomic_inc(&(bin_counter(key_view(i))));
//      mapping(key_view(i)) = 1;
    });
//    Kokkos::parallel_scan("create mapping", Kokkos::RangePolicy<uint64_t>(0, num_bins), KOKKOS_LAMBDA(const uint64_t i, uint64_t& update, const bool final) {
//      const uint64_t val_i = mapping(i);
//      if(final) {
//        mapping(i) = update;
//      }
//      update += val_i;
//    });
//Kokkos::parallel_for("Print mapping", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int idx) {
//  printf("Mapping: ");
//  for(int i=0; i<mapping.size(); i++) {
//    //if(i != mapping(i)) 
//      printf("%d:%d\t", i, mapping(i));
//  }
//  printf("\n");
//});
    // Find the max and min number of val_view per cell
    Kokkos::parallel_reduce("Get max/min nppc", Kokkos::RangePolicy<uint64_t>(0,num_bins), 
      min_max_functor(bin_counter), nppc_reducer); 
//printf("Bin count Min,Max keys: %d,%d\n", nppc_result.min_val, nppc_result.max_val);
    const uint64_t chunk_size = tile_size*(nppc_result.max_val);
//printf("Chunk size: %lu\n", chunk_size);
    // Reset bin_counter
    Kokkos::deep_copy(bin_counter, 0);
    // Update keys 
    Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<uint64_t>(0, np), KOKKOS_LAMBDA(const uint64_t i) {
      const uint64_t count = Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1);
      const uint64_t chunk_idx = key_view(i)/tile_size;
      keys(i) = chunk_idx*chunk_size + count*tile_size + static_cast<uint64_t>(key_view(i));
//      const uint64_t chunk_idx = mapping(key_view(i))/tile_size;
//      keys(i) = chunk_idx*chunk_size + count*tile_size + static_cast<uint64_t>(mapping(key_view(i)));
//      const uint64_t chunk_idx = (key_view(i)-(result.min_val))/tile_size;
//      keys(i) = chunk_idx*chunk_size + count*tile_size + static_cast<uint64_t>(key_view(i)-result.min_val);
//if(count < 2 && (key_view(i) == 26890 || key_view(i) == 10507 || key_view(i) == 10508) ) {
//  printf("Key val %d, i=%d, count=%lu, chunk_idx=%lu, keys(i)=%lu\n", key_view(i), i, count, chunk_idx, keys(i));
//}
    });

    }
    Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), keys, key_view);

//    // Find smallest and largest key
//    Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,key_view.extent(0)), 
//      min_max_functor(keys), reducer);
//
//    // Create comparator
//    using key_type = decltype(keys);
//    using Comparator = Kokkos::BinOp1D<key_type>;
//    Comparator comp(np, result.min_val, result.max_val);
//
//    // Sort and create permutation View
//    int sort_within_bins = 1;
//    Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
//    bin_sort.create_permute_vector();
//
//    // Sort keys
//    bin_sort.sort(key_view);
}

#endif //guard
