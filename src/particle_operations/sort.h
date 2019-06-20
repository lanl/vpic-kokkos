#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

#include <Kokkos_Sort.hpp>
#include <Kokkos_DualView.hpp>
#include "../vpic/kokkos_helpers.h"

// We need this because we use the bits in particle_var::pi as ints but store
// them as floats. It should really go away and we should move pi into it's own
// struct
template<class KeyViewType> struct Casted_BinOp1D {
  int max_bins_;
  double mul_;
  typename KeyViewType::const_value_type range_;
  typename KeyViewType::const_value_type min_;

  Casted_BinOp1D():max_bins_(0),mul_(0.0),
            range_(typename KeyViewType::const_value_type()),
            min_(typename KeyViewType::const_value_type()) {}

  //Construct BinOp with number of bins, minimum value and maxuimum value
  Casted_BinOp1D(int max_bins__, typename KeyViewType::const_value_type min,
                               typename KeyViewType::const_value_type max )
     :max_bins_(max_bins__+1),mul_(1.0*max_bins__/(max-min)),range_(max-min),min_(min) {}

  //Determine bin index from key value
  template<class ViewType>
  KOKKOS_INLINE_FUNCTION
  int bin(ViewType& keys, const int& i) const {
    return int(mul_*(keys(i)-min_));
  }

  //Return maximum bin index + 1
  KOKKOS_INLINE_FUNCTION
  int max_bins() const {
    return max_bins_;
  }

  //Compare to keys within a bin if true new_val will be put before old_val
  template<class ViewType, typename iType1, typename iType2>
  KOKKOS_INLINE_FUNCTION
  bool operator()(ViewType& keys, iType1& i1, iType2& i2) const {
      int a = keys(i1);
      int b = keys(i2);
      int _a = reinterpret_cast<int&>(a);
      int _b = reinterpret_cast<int&>(b);
      return a < b;
  }
};

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
        using Comparator = Casted_BinOp1D<key_type>;
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
