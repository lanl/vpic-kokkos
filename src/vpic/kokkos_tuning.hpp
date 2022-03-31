#ifndef _kokkos_tuning_h_
#define _kokkos_tuning_h_

// Hierarchical parallelism
#ifdef VPIC_ENABLE_HIERARCHICAL
  #ifdef KOKKOS_ARCH_VOLTA70
    #ifdef VPIC_ENABLE_TEAM_REDUCTION
      #define LEAGUE_SIZE 65536
      #define TEAM_SIZE 32
    #else
      #define LEAGUE_SIZE 4096
      #define TEAM_SIZE 512
    #endif
  #else
    #define LEAGUE_SIZE ((g->nv)/4)
    #define TEAM_SIZE Kokkos::AUTO
  #endif
#endif

// Sorting
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // Check if using team reduction optimization
  #if defined(VPIC_ENABLE_TEAM_REDUCTION) || defined(VPIC_ENABLE_HIERARCHICAL)
    #define SORT standard_sort
  #else
    #define SORT strided_sort
  #endif
#else
  #define SORT standard_sort
//  #define SORT strided_sort
//  #define SORT tiled_sort
//  #define SORT_TILE_SIZE 16
#endif

#endif // _kokkos_tuning_h_

