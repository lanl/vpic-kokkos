#ifndef _kokkos_tuning_h_
#define _kokkos_tuning_h_


#include "../util/util_base.h"
#include <Kokkos_Core.hpp>
#include <stdlib.h>

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
    #define LEAGUE_SIZE (na/4)
    #define TEAM_SIZE Kokkos::AUTO
  #endif
#endif

// Sorting
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // Check if using team reduction optimization
  #if defined(VPIC_ENABLE_TEAM_REDUCTION) || defined(VPIC_ENABLE_HIERARCHICAL)
    #define SORT standard_sort
    #define SORT_TYPE StandardSort
  #else
    #define SORT strided_sort
    #define SORT_TYPE StridedSort
  #endif
#else
  #define SORT standard_sort
  #define SORT_TYPE StandardSort
#endif

typedef enum SortType {
  StandardSort = 0,
  StridedSort = 1,
  TiledSort = 2,
  TiledStridedSort = 3
} SortType;


typedef struct OptimizationSettings {
  bool enable_hierarchical;
  bool enable_team_reduction;
  SortType sort_type;
  uint32_t sort_tile_size;
  uint32_t advance_p_league_size;
  uint32_t advance_p_team_size;

  OptimizationSettings() {
#ifdef VPIC_ENABLE_HIERARCHICAL
    enable_hierarchical = true;
#else 
    enable_hierarchical = false;
#endif
#ifdef VPIC_ENABLE_TEAM_REDUCTION
    enable_team_reduction = true;
#else
    enable_team_reduction = false;
#endif
    sort_type = SORT_TYPE;
    sort_tile_size = 0;
    advance_p_league_size = 0;
    advance_p_team_size = 0;
  }

  OptimizationSettings(int na) {
#ifdef VPIC_ENABLE_HIERARCHICAL
    enable_hierarchical = true;
#else 
    enable_hierarchical = false;
#endif
#ifdef VPIC_ENABLE_TEAM_REDUCTION
    enable_team_reduction = true;
#else
    enable_team_reduction = false;
#endif
    sort_type = SORT_TYPE;
    sort_tile_size = 0;
    advance_p_league_size = 0;
    advance_p_team_size = 0;
#ifdef VPIC_ENABLE_HIERARCHICAL
    Kokkos::TeamPolicy<> team_policy(LEAGUE_SIZE, TEAM_SIZE);
    advance_p_league_size = team_policy.league_size();
    advance_p_team_size = team_policy.team_size();
#endif

  }

  void initialize(int na) {
#ifdef VPIC_ENABLE_HIERARCHICAL
    enable_hierarchical = true;
#else 
    enable_hierarchical = false;
#endif
#ifdef VPIC_ENABLE_TEAM_REDUCTION
    enable_team_reduction = true;
#else
    enable_team_reduction = false;
#endif
    sort_type = SORT_TYPE;
    sort_tile_size = 0;
    advance_p_league_size = 0;
    advance_p_team_size = 0;
#ifdef VPIC_ENABLE_HIERARCHICAL
    Kokkos::TeamPolicy<> team_policy(LEAGUE_SIZE, TEAM_SIZE);
    advance_p_league_size = team_policy.league_size();
    advance_p_team_size = team_policy.team_size();
#endif
  }

  void init_from_cmdline(int argc, char** argv) {
    for(int i=0; i<argc; i++) {
      const char* arg = argv[i];
      if(strcmp(arg, "--enable-hierarchical") == 0) {
        enable_hierarchical = true;
      } else if(strcmp(arg, "--disable-hierarchical") == 0) {
        enable_hierarchical = false;
        enable_team_reduction = false;
      } else if(strcmp(arg, "--enable-team-reduction") == 0) {
        enable_hierarchical = true;
        enable_team_reduction = true;
      } else if(strcmp(arg, "--disable-team-reduction") == 0) {
        enable_team_reduction = false;
      } else if(strcmp(arg, "--league-size") == 0) {
        advance_p_league_size = atoi(argv[i+1]);
      } else if(strcmp(arg, "--team-size") == 0) {
        advance_p_team_size = atoi(argv[i+1]);
      } else if(strcmp(arg, "--sort-mode") == 0) {
        if(strcmp(argv[i+1], "StandardSort") == 0) {
          sort_type = StandardSort;
        } else if(strcmp(argv[i+1], "StridedSort") == 0) {
          sort_type = StridedSort;
        } else if(strcmp(argv[i+1], "TiledSort") == 0) {
          sort_type = TiledSort;
        } else if(strcmp(argv[i+1], "TiledStridedSort") == 0) {
          sort_type = TiledStridedSort;
        }
      }
    }
  }
  
  void print() {
    log_printf("**********Optimiztion settings**********\n");
    log_printf("Hierarchical parallelism enabled: %s\n", enable_hierarchical ? "true" : "false");
    log_printf("Team reduction enabled: %s\n", enable_team_reduction ? "true" : "false");
    if(advance_p_league_size == 0 && enable_hierarchical) {
      log_printf("League size: ((g->nv)/4)\n");
    } else if(enable_hierarchical) {
      log_printf("League size: %d\n", advance_p_league_size);
    }
    if(advance_p_team_size == 0 && enable_hierarchical) {
      log_printf("Team size: Kokkos::AUTO\n");
    } else if(enable_hierarchical) {
      log_printf("Team size: %d\n", advance_p_team_size);
    }
    if(sort_type == StandardSort) {
      log_printf("Sorting order: StandardSort\n");
    } else if(sort_type == StridedSort) {
      log_printf("Sorting order: StridedSort\n");
    } else if(sort_type == TiledSort) {
      log_printf("Sorting order: TiledSort\n");
    } else if(sort_type == TiledStridedSort) {
      log_printf("Sorting order: TiledStridedSort\n");
    }
    if(sort_tile_size == 0) {
    } else {
      log_printf("Sort tile size: %d\n", sort_tile_size);
    }
    log_printf("****************************************\n");
  }
} OptimizationSettings;


#endif // _kokkos_tuning_h_

