#ifndef _kokkos_tuning_h_
#define _kokkos_tuning_h_

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_CUDA_UVM) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET) 
  #define USE_GPU
#endif

#if defined(KOKKOS_ENABLE_THREADS) || defined(KOKKOS_ENABLE_OPENMP)
  #define USE_CPU
#endif

// Vectorization (CPU only)
#if defined( VPIC_ENABLE_VECTORIZATION ) && defined( USE_GPU )
  #undef VPIC_ENABLE_VECTORIZATION
#endif
#if defined( VPIC_ENABLE_VECTORIZATION ) && !defined( USE_GPU )
  #define LANE lane
  #define BEGIN_THREAD_BLOCK Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_iters), [&] (int lane) {
  #define END_THREAD_BLOCK });
  #define BEGIN_VECTOR_BLOCK Kokkos::parallel_for_simd(Kokkos::ThreadVectorRange(team_member, num_iters), [&] (int lane) {
  #define END_VECTOR_BLOCK });
#else
  #define LANE 0
  #define BEGIN_THREAD_BLOCK
  #define END_THREAD_BLOCK
  #define BEGIN_VECTOR_BLOCK
  #define END_VECTOR_BLOCK
#endif

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
#endif

#endif // _kokkos_tuning_h_

