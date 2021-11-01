#ifndef _kokkos_tuning_h_
#define _kokkos_tuning_h_

// Sorting
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  #define SORT strided_sort
#else
  #define SORT standard_sort
#endif

#endif // _kokkos_tuning_h_


