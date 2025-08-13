#ifndef __SIMD_HELPERS_HPP__
#define __SIMD_HELPERS_HPP__

#include <Kokkos_Core.hpp>

namespace Kokkos {
  /** \brief  Intra-thread vector parallel_for. Executes lambda(iType i) for each
   * i=0..N-1.
   *
   * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
   */
  template <template <typename iType, class ThreadsExecTeamMember> class ThreadVectorRangeBoundariesStruct, 
            typename iType, class ThreadsExecTeamMember, class Lambda>
  KOKKOS_INLINE_FUNCTION void parallel_for_simd(
      const ThreadVectorRangeBoundariesStruct<
          iType, ThreadsExecTeamMember>& loop_boundaries,
      const Lambda& lambda) {
    #pragma omp simd
    for (iType i = loop_boundaries.start; i < loop_boundaries.end;
         i += loop_boundaries.increment)
      lambda(i);
  }

  /** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i,
   * ValueType & val) for each i=0..N-1.
   *
   * The range i=0..N-1 is mapped to all vector lanes of the the calling thread
   * and a summation of val is performed and put into result.
   */
  template <template <typename iType, class ThreadsExecTeamMember> class ThreadVectorRangeBoundariesStruct, 
            typename iType, class ThreadsExecTeamMember, class Lambda, typename ValueType>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<!Kokkos::is_reducer<ValueType>::value>::type
      parallel_reduce_simd_sum(const ThreadVectorRangeBoundariesStruct<
                          iType, ThreadsExecTeamMember>& loop_boundaries,
                      const Lambda& lambda, ValueType& result) {
    result = ValueType();
    //#pragma omp for simd reduction(+:result) schedule(simd:static)
    #pragma omp simd reduction(+:result)
    for (iType i = loop_boundaries.start; i < loop_boundaries.end;
         i += loop_boundaries.increment) {
      lambda(i, result);
    }
  }
}

#endif
