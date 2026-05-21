#ifndef CUSTOM_BINSORT_HPP_
#define CUSTOM_BINSORT_HPP_

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <iostream>

template <class KeyViewType>
struct CustomBinOp1D {
  size_t max_bins_ = {};
  double mul_   = {};
  double min_   = {};

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED CustomBinOp1D() = default;
#else
  CustomBinOp1D() = delete;
#endif

  // Construct BinOp with number of bins, minimum value and maximum value
  CustomBinOp1D(size_t max_bins, typename KeyViewType::const_value_type min,
          typename KeyViewType::const_value_type max)
      : max_bins_(max_bins + 1),
        // Cast to double to avoid possible overflow when using integer
        mul_(static_cast<double>(max_bins) /
             (static_cast<double>(max) - static_cast<double>(min))),
        min_(static_cast<double>(min)) {
    // For integral types the number of bins may be larger than the range
    // in which case we can exactly have one unique value per bin
    // and then don't need to sort bins.
    if (std::is_integral_v<typename KeyViewType::const_value_type> &&
        (static_cast<double>(max) - static_cast<double>(min)) <=
            static_cast<double>(max_bins)) {
      mul_ = 1.;
    }
  }

  // Determine bin index from key value
  template <class ViewType>
  KOKKOS_INLINE_FUNCTION size_t bin(ViewType& keys, const size_t& i) const {
    return static_cast<size_t>(mul_ * (static_cast<double>(keys(i)) - min_));
  }

  // Return maximum bin index + 1
  KOKKOS_INLINE_FUNCTION
  size_t max_bins() const { return max_bins_; }

  // Compare to keys within a bin if true new_val will be put before old_val
  template <class ViewType, typename iType1, typename iType2>
  KOKKOS_INLINE_FUNCTION bool operator()(ViewType& keys, iType1& i1,
                                         iType2& i2) const {
    return keys(i1) < keys(i2);
  }
};

template <class DstViewType, class SrcViewType, int Rank = DstViewType::rank>
struct CopyOp;

template <class DstViewType, class SrcViewType>
struct CopyOp<DstViewType, SrcViewType, 1> {
  KOKKOS_INLINE_FUNCTION
  static void copy(DstViewType const& dst, size_t i_dst, SrcViewType const& src,
                   size_t i_src) {
    dst(i_dst) = src(i_src);
  }
};

template <class DstViewType, class SrcViewType>
struct CopyOp<DstViewType, SrcViewType, 2> {
  KOKKOS_INLINE_FUNCTION
  static void copy(DstViewType const& dst, size_t i_dst, SrcViewType const& src,
                   size_t i_src) {
    for (size_t j = 0; j < (size_t)dst.extent(1); j++) dst(i_dst, j) = src(i_src, j);
  }
};

template <class DstViewType, class SrcViewType>
struct CopyOp<DstViewType, SrcViewType, 3> {
  KOKKOS_INLINE_FUNCTION
  static void copy(DstViewType const& dst, size_t i_dst, SrcViewType const& src,
                   size_t i_src) {
    for (size_t j = 0; j < dst.extent(1); j++)
      for (size_t k = 0; k < dst.extent(2); k++)
        dst(i_dst, j, k) = src(i_src, j, k);
  }
};

template <class KeyViewType, class BinSortOp,
          class Space    = typename KeyViewType::device_type,
          class SizeType = typename KeyViewType::memory_space::size_type>
class CustomBinSort {
 public:
  template <class DstViewType, class SrcViewType>
  struct copy_functor {
    using src_view_type = typename SrcViewType::const_type;

    using copy_op = CopyOp<DstViewType, src_view_type>;

    DstViewType dst_values;
    src_view_type src_values;
    size_t dst_offset;

    copy_functor(DstViewType const& dst_values_, size_t const& dst_offset_,
                 SrcViewType const& src_values_)
        : dst_values(dst_values_),
          src_values(src_values_),
          dst_offset(dst_offset_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t& i) const {
      copy_op::copy(dst_values, i + dst_offset, src_values, i);
    }
  };

  template <class DstViewType, class PermuteViewType, class SrcViewType>
  struct copy_permute_functor {
    // If a Kokkos::View then can generate constant random access
    // otherwise can only use the constant type.

    using src_view_type = std::conditional_t<
        Kokkos::is_view<SrcViewType>::value,
        Kokkos::View<typename SrcViewType::const_data_type,
                     typename SrcViewType::array_layout,
                     typename SrcViewType::device_type
#if !defined(KOKKOS_COMPILER_NVHPC) || (KOKKOS_COMPILER_NVHPC >= 230700)
                     ,
                     Kokkos::MemoryTraits<Kokkos::RandomAccess>
#endif
                     >,
        typename SrcViewType::const_type>;

    using perm_view_type = typename PermuteViewType::const_type;

    using copy_op = CopyOp<DstViewType, src_view_type>;

    DstViewType dst_values;
    perm_view_type sort_order;
    src_view_type src_values;
    size_t src_offset;

    copy_permute_functor(DstViewType const& dst_values_,
                         PermuteViewType const& sort_order_,
                         SrcViewType const& src_values_, size_t const& src_offset_)
        : dst_values(dst_values_),
          sort_order(sort_order_),
          src_values(src_values_),
          src_offset(src_offset_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t& i) const {
      copy_op::copy(dst_values, i, src_values, src_offset + sort_order(i));
    }
  };

  // Naming this alias "execution_space" would be problematic since it would be
  // considered as execution space for the various functors which might use
  // another execution space through sort() or create_permute_vector().
  using exec_space  = typename Space::execution_space;
  using bin_op_type = BinSortOp;

  struct bin_count_tag {};
  struct bin_offset_tag {};
  struct bin_binning_tag {};
  struct bin_sort_bins_tag {};

 public:
  using size_type  = size_t; //SizeType;
  using value_type = size_type;

  using offset_type    = Kokkos::View<size_type*, Space>;
  using bin_count_type = Kokkos::View<const size_t*, Space>;

  using const_key_view_type = typename KeyViewType::const_type;

  // If a Kokkos::View then can generate constant random access
  // otherwise can only use the constant type.

  using const_rnd_key_view_type = std::conditional_t<
      Kokkos::is_view<KeyViewType>::value,
      Kokkos::View<typename KeyViewType::const_data_type,
                   typename KeyViewType::array_layout,
                   typename KeyViewType::device_type,
                   Kokkos::MemoryTraits<Kokkos::RandomAccess> >,
      const_key_view_type>;

  using non_const_key_scalar = typename KeyViewType::non_const_value_type;
  using const_key_scalar     = typename KeyViewType::const_value_type;

  using bin_count_atomic_type =
      Kokkos::View<size_t*, Space, Kokkos::MemoryTraits<Kokkos::Atomic> >;

 private:
  const_key_view_type keys;
  const_rnd_key_view_type keys_rnd;

 public:
  BinSortOp bin_op;
  offset_type bin_offsets;
  bin_count_atomic_type bin_count_atomic;
  bin_count_type bin_count_const;
  offset_type sort_order;

  size_t range_begin;
  size_t range_end;
  bool sort_within_bins;

 public:
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED CustomBinSort() = default;
#else
  CustomBinSort() = delete;
#endif

  //----------------------------------------
  // Constructor: takes the keys, the binning_operator and optionally whether to
  // sort within bins (default false)
  template <typename ExecutionSpace>
  CustomBinSort(const ExecutionSpace& exec, const_key_view_type keys_,
          size_t range_begin_, size_t range_end_, BinSortOp bin_op_,
          bool sort_within_bins_ = false)
      : keys(keys_),
        keys_rnd(keys_),
        bin_op(bin_op_),
        bin_offsets(),
        bin_count_atomic(),
        bin_count_const(),
        sort_order(),
        range_begin(range_begin_),
        range_end(range_end_),
        sort_within_bins(sort_within_bins_) {
    static_assert(
        Kokkos::SpaceAccessibility<ExecutionSpace,
                                   typename Space::memory_space>::accessible,
        "The provided execution space must be able to access the memory space "
        "CustomBinSort was initialized with!");
    if (bin_op.max_bins() <= 0)
      Kokkos::abort(
          "The number of bins in the BinSortOp object must be greater than 0!");
    bin_count_atomic = Kokkos::View<size_t*, Space>(
        "Kokkos::SortImpl::CustomBinSortFunctor::bin_count", bin_op.max_bins());
    bin_count_const = bin_count_atomic;
    bin_offsets =
        offset_type(view_alloc(exec, Kokkos::WithoutInitializing,
                               "Kokkos::SortImpl::CustomBinSortFunctor::bin_offsets"),
                    bin_op.max_bins());
    sort_order =
        offset_type(view_alloc(exec, Kokkos::WithoutInitializing,
                               "Kokkos::SortImpl::CustomBinSortFunctor::sort_order"),
                    range_end - range_begin);
  }

  CustomBinSort(const_key_view_type keys_, size_t range_begin_, size_t range_end_,
          BinSortOp bin_op_, bool sort_within_bins_ = false)
      : CustomBinSort(exec_space{}, keys_, range_begin_, range_end_, bin_op_,
                sort_within_bins_) {}

  template <typename ExecutionSpace>
  CustomBinSort(const ExecutionSpace& exec, const_key_view_type keys_,
          BinSortOp bin_op_, bool sort_within_bins_ = false)
      : CustomBinSort(exec, keys_, 0, keys_.extent(0), bin_op_, sort_within_bins_) {}

  CustomBinSort(const_key_view_type keys_, BinSortOp bin_op_,
          bool sort_within_bins_ = false)
      : CustomBinSort(exec_space{}, keys_, bin_op_, sort_within_bins_) {}

  //----------------------------------------
  // Reset sorter with new keys, range, and bin op
  template <typename ExecutionSpace>
  void reset(const ExecutionSpace& exec, const_key_view_type keys_,
             size_t range_begin_, size_t range_end_, BinSortOp bin_op_,
             bool sort_within_bins_ = false) {
    static_assert(
        Kokkos::SpaceAccessibility<ExecutionSpace,
                                   typename Space::memory_space>::accessible,
        "The provided execution space must be able to access the memory space "
        "CustomBinSort was initialized with!");
    if (bin_op.max_bins() <= 0)
      Kokkos::abort(
          "The number of bins in the BinSortOp object must be greater than 0!");

    keys = keys_;
    keys_rnd = keys_;
    bin_op = bin_op_;
    range_begin = range_begin_;
    range_end = range_end_;
    sort_within_bins = sort_within_bins_;

    if(bin_count_atomic.extent(0) < bin_op.max_bins()) {
      Kokkos::resize(bin_count_atomic, bin_op.max_bins());
    }
    Kokkos::deep_copy(bin_count_atomic, 0);
    bin_count_const = bin_count_atomic;
    if(bin_offsets.extent(0) < bin_op.max_bins()) {
      Kokkos::resize(bin_offsets, bin_op.max_bins());
    }
    Kokkos::deep_copy(bin_offsets, 0);
    if(sort_order.extent(0) < range_end - range_begin) {
      Kokkos::resize(sort_order, range_end - range_begin);
    }
    Kokkos::deep_copy(sort_order, 0);
  }

  //----------------------------------------
  // Create the permutation vector, the bin_offset array and the bin_count
  // array. Can be called again if keys changed
  template <class ExecutionSpace>
  void create_permute_vector(const ExecutionSpace& exec) {
    static_assert(
        Kokkos::SpaceAccessibility<ExecutionSpace,
                                   typename Space::memory_space>::accessible,
        "The provided execution space must be able to access the memory space "
        "CustomBinSort was initialized with!");

    const size_t len = range_end - range_begin;
    Kokkos::parallel_for(
        "Kokkos::Sort::BinCount",
        Kokkos::RangePolicy<ExecutionSpace, bin_count_tag>(exec, 0, len),
        *this);
    Kokkos::parallel_scan("Kokkos::Sort::BinOffset",
                          Kokkos::RangePolicy<ExecutionSpace, bin_offset_tag>(
                              exec, 0, bin_op.max_bins()),
                          *this);

    Kokkos::deep_copy(exec, bin_count_atomic, 0);
    Kokkos::parallel_for(
        "Kokkos::Sort::BinBinning",
        Kokkos::RangePolicy<ExecutionSpace, bin_binning_tag>(exec, 0, len),
        *this);

    if (sort_within_bins)
      Kokkos::parallel_for(
          "Kokkos::Sort::CustomBinSort",
          Kokkos::RangePolicy<ExecutionSpace, bin_sort_bins_tag>(
              exec, 0, bin_op.max_bins()),
          *this);
  }

  // Create the permutation vector, the bin_offset array and the bin_count
  // array. Can be called again if keys changed
  void create_permute_vector() {
    Kokkos::fence("Kokkos::Binsort::create_permute_vector: before");
    exec_space e{};
    create_permute_vector(e);
    e.fence("Kokkos::Binsort::create_permute_vector: after");
  }

  // Sort a subset of a view with respect to the first dimension using the
  // permutation array and supplied scratch Views
  template <class ExecutionSpace, class ValuesViewType>
  void sort_scratch(const ExecutionSpace& exec, ValuesViewType const& values,
            Kokkos::View<typename ValuesViewType::data_type,
                         typename ValuesViewType::device_type> sorted_values,
            size_t values_range_begin, size_t values_range_end) const {
    if (values.extent(0) == 0) {
      return;
    }

    static_assert(
        Kokkos::SpaceAccessibility<ExecutionSpace,
                                   typename Space::memory_space>::accessible,
        "The provided execution space must be able to access the memory space "
        "CustomBinSort was initialized with!");
    static_assert(
        Kokkos::SpaceAccessibility<
            ExecutionSpace, typename ValuesViewType::memory_space>::accessible,
        "The provided execution space must be able to access the memory space "
        "of the View argument!");

    const size_t len        = range_end - range_begin;
    const size_t values_len = values_range_end - values_range_begin;
    if (len != values_len) {
      Kokkos::abort(
          "CustomBinSort::sort: values range length != permutation vector length");
    }

    using scratch_view_type =
        Kokkos::View<typename ValuesViewType::data_type,
                     typename ValuesViewType::device_type>;

    {
      copy_permute_functor<scratch_view_type /* DstViewType */
                           ,
                           offset_type /* PermuteViewType */
                           ,
                           ValuesViewType /* SrcViewType */
                           >
          functor(sorted_values, sort_order, values,
                  values_range_begin - range_begin);

      parallel_for("Kokkos::Sort::CopyPermute",
                   Kokkos::RangePolicy<ExecutionSpace, size_t>(exec, 0, len), functor);
    }

    {
      copy_functor<ValuesViewType, scratch_view_type> functor(
          values, range_begin, sorted_values);

      parallel_for("Kokkos::Sort::Copy",
                   Kokkos::RangePolicy<ExecutionSpace, size_t>(exec, 0, len), functor);
    }
  }

  // Sort a subset of a view with respect to the first dimension using the
  // permutation array
  template <class ExecutionSpace, class ValuesViewType>
  void sort(const ExecutionSpace& exec, ValuesViewType const& values,
            size_t values_range_begin, size_t values_range_end) const {
    if (values.extent(0) == 0) {
      return;
    }

    static_assert(
        Kokkos::SpaceAccessibility<ExecutionSpace,
                                   typename Space::memory_space>::accessible,
        "The provided execution space must be able to access the memory space "
        "CustomBinSort was initialized with!");
    static_assert(
        Kokkos::SpaceAccessibility<
            ExecutionSpace, typename ValuesViewType::memory_space>::accessible,
        "The provided execution space must be able to access the memory space "
        "of the View argument!");

    const size_t len        = range_end - range_begin;
    const size_t values_len = values_range_end - values_range_begin;
    if (len != values_len) {
      Kokkos::abort(
          "CustomBinSort::sort: values range length != permutation vector length");
    }

    using scratch_view_type =
        Kokkos::View<typename ValuesViewType::data_type,
                     typename ValuesViewType::device_type>;
    scratch_view_type sorted_values(
        view_alloc(exec, Kokkos::WithoutInitializing,
                   "Kokkos::SortImpl::CustomBinSortFunctor::sorted_values"),
        values.rank_dynamic > 0 ? len : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        values.rank_dynamic > 1 ? values.extent(1)
                                : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        values.rank_dynamic > 2 ? values.extent(2)
                                : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        values.rank_dynamic > 3 ? values.extent(3)
                                : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        values.rank_dynamic > 4 ? values.extent(4)
                                : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        values.rank_dynamic > 5 ? values.extent(5)
                                : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        values.rank_dynamic > 6 ? values.extent(6)
                                : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        values.rank_dynamic > 7 ? values.extent(7)
                                : KOKKOS_IMPL_CTOR_DEFAULT_ARG);

    {
      copy_permute_functor<scratch_view_type /* DstViewType */
                           ,
                           offset_type /* PermuteViewType */
                           ,
                           ValuesViewType /* SrcViewType */
                           >
          functor(sorted_values, sort_order, values,
                  values_range_begin - range_begin);

      parallel_for("Kokkos::Sort::CopyPermute",
                   Kokkos::RangePolicy<ExecutionSpace>(exec, 0, len), functor);
    }

    {
      copy_functor<ValuesViewType, scratch_view_type> functor(
          values, range_begin, sorted_values);

      parallel_for("Kokkos::Sort::Copy",
                   Kokkos::RangePolicy<ExecutionSpace>(exec, 0, len), functor);
    }
  }

  // Sort a subset of a view with respect to the first dimension using the
  // permutation array
  template <class ValuesViewType>
  void sort(ValuesViewType const& values, size_t values_range_begin,
            size_t values_range_end) const {
    Kokkos::fence("Kokkos::Binsort::sort: before");
    exec_space exec;
    sort(exec, values, values_range_begin, values_range_end);
    exec.fence("Kokkos::CustomBinSort:sort: after");
  }

  template <class ExecutionSpace, class ValuesViewType>
  void sort(ExecutionSpace const& exec, ValuesViewType const& values) const {
    this->sort(exec, values, 0, /*values.extent(0)*/ range_end - range_begin);
  }

  template <class ValuesViewType>
  void sort(ValuesViewType const& values) const {
    this->sort(values, 0, /*values.extent(0)*/ range_end - range_begin);
  }

  // Get the permutation vector
  KOKKOS_INLINE_FUNCTION
  offset_type get_permute_vector() const { return sort_order; }

  // Get the start offsets for each bin
  KOKKOS_INLINE_FUNCTION
  offset_type get_bin_offsets() const { return bin_offsets; }

  // Get the count for each bin
  KOKKOS_INLINE_FUNCTION
  bin_count_type get_bin_count() const { return bin_count_const; }

 public:
  KOKKOS_INLINE_FUNCTION
  void operator()(const bin_count_tag& /*tag*/, const size_t i) const {
    const size_t j = range_begin + i;
    bin_count_atomic(bin_op.bin(keys, j))++;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const bin_offset_tag& /*tag*/, const size_t i,
                  value_type& offset, const bool& final) const {
    if (final) {
      bin_offsets(i) = offset;
    }
    offset += bin_count_const(i);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const bin_binning_tag& /*tag*/, const size_t i) const {
    const size_t j     = range_begin + i;
    const size_t bin   = bin_op.bin(keys, j);
    const size_t count = bin_count_atomic(bin)++;

    sort_order(bin_offsets(bin) + count) = j;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const bin_sort_bins_tag& /*tag*/, const size_t i) const {
    auto bin_size = bin_count_const(i);
    if (bin_size <= 1) return;
    constexpr bool use_std_sort =
        std::is_same_v<typename exec_space::memory_space, Kokkos::HostSpace>;
    size_t lower_bound = bin_offsets(i);
    size_t upper_bound = lower_bound + bin_size;
    // Switching to std::sort for more than 10 elements has been found
    // reasonable experimentally.
    if (use_std_sort && bin_size > 10) {
      KOKKOS_IF_ON_HOST(
          (std::sort(sort_order.data() + lower_bound,
                     sort_order.data() + upper_bound,
                     [this](size_t p, size_t q) { return bin_op(keys_rnd, p, q); });))
    } else {
      for (size_t k = lower_bound + 1; k < upper_bound; ++k) {
        size_t old_idx = sort_order(k);
        size_t j       = k - 1;
        while (j >= lower_bound) {
          size_t new_idx = sort_order(j);
          if (!bin_op(keys_rnd, old_idx, new_idx)) break;
          sort_order(j + 1) = new_idx;
          --j;
        }
        sort_order(j + 1) = old_idx;
      }
    }
  }
};

#endif //CUSTOM_BINSORT_HPP_
