#ifndef _kokkos_tuning_h_
#define _kokkos_tuning_h_

#ifdef VPIC_ENABLE_HIERARCHICAL
  #ifdef KOKKOS_ARCH_VOLTA70
    #define LEAGUE_SIZE 2048
    #define TEAM_SIZE 512
  #else
    #define LEAGUE_SIZE ((g->nv)/4)
    #define TEAM_SIZE Kokkos::AUTO
  #endif
#endif

#endif // _kokkos_tuning_h_

