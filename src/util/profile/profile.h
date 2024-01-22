#ifndef _profile_h_
#define _profile_h_

#include "../util_base.h"
#include <chrono>

// To add a named timer to the profile, add a line to this macro in
// the position you want the times to appear in the profile dumps.  To
// keep dumps prettily formatted, only the first 16 characters of the
// timer name is printed on profile dumps.

#ifdef FIELD_IONIZATION
#define PROFILE_TIMERS(_) \
  _( clear_accumulators ) \
  _( sort_p            ) \
  _( collision_model   ) \
  _( advance_p         ) \
  _( reduce_accumulators ) \
  _( emission_model    ) \
  _( boundary_p        ) \
  _( clear_jf          ) \
  _( unload_accumulator ) \
  _( synchronize_jf    ) \
  _( advance_b         ) \
  _( advance_e         ) \
  _( clear_rhof        ) \
  _( accumulate_rho_p  ) \
  _( synchronize_rho   ) \
  _( compute_div_e_err ) \
  _( compute_rms_div_e_err ) \
  _( clean_div_e       ) \
  _( compute_div_b_err ) \
  _( compute_rms_div_b_err ) \
  _( clean_div_b       ) \
  _( synchronize_tang_e_norm_b ) \
  _( load_interpolator ) \
  _( compute_curl_b    ) \
  _( compute_rhob      ) \
  _( uncenter_p        ) \
  _( user_initialization ) \
  _( user_particle_collisions ) \
  _( user_particle_injection ) \
  _( user_current_injection ) \
  _( user_field_injection ) \
  _( sort_particles ) \
  _( field_sa_contributions ) \
  _( dump_energies ) \
  _( dump_ionization_states ) \
  _( FIELD_DATA_MOVEMENT ) \
  _( PARTICLE_DATA_MOVEMENT ) \
  _( JF_ACCUM_DATA_MOVEMENT ) \
  _( INTERPOLATOR_DATA_MOVEMENT ) \
  _( BACKFILL ) \
  _( BACKFILL_COMPRESS ) \
  _( user_data_movement ) \
  _( user_diagnostics  )
#else
#define PROFILE_TIMERS(_) \
  _( clear_accumulators ) \
  _( sort_p            ) \
  _( collision_model   ) \
  _( advance_p         ) \
  _( reduce_accumulators ) \
  _( emission_model    ) \
  _( boundary_p        ) \
  _( clear_jf          ) \
  _( unload_accumulator ) \
  _( synchronize_jf    ) \
  _( advance_b         ) \
  _( advance_e         ) \
  _( clear_rhof        ) \
  _( accumulate_rho_p  ) \
  _( synchronize_rho   ) \
  _( compute_div_e_err ) \
  _( compute_rms_div_e_err ) \
  _( clean_div_e       ) \
  _( compute_div_b_err ) \
  _( compute_rms_div_b_err ) \
  _( clean_div_b       ) \
  _( synchronize_tang_e_norm_b ) \
  _( load_interpolator ) \
  _( compute_curl_b    ) \
  _( compute_rhob      ) \
  _( uncenter_p        ) \
  _( user_initialization ) \
  _( user_particle_collisions ) \
  _( user_particle_injection ) \
  _( user_current_injection ) \
  _( user_field_injection ) \
  _( sort_particles ) \
  _( field_sa_contributions ) \
  _( dump_energies ) \
  _( FIELD_DATA_MOVEMENT ) \
  _( PARTICLE_DATA_MOVEMENT ) \
  _( JF_ACCUM_DATA_MOVEMENT ) \
  _( INTERPOLATOR_DATA_MOVEMENT ) \
  _( BACKFILL ) \
  _( BACKFILL_COMPRESS ) \
  _( user_data_movement ) \
  _( user_diagnostics  )
#endif

enum profile_internal_use_only_timers {
  profile_internal_use_only_invalid_timer = -1,
# define PROFILE_INTERNAL_USE_ONLY( timer ) profile_internal_use_only_##timer,
  PROFILE_TIMERS( PROFILE_INTERNAL_USE_ONLY )
# undef PROFILE_INTERNAL_USE_ONLY
  profile_internal_use_only_n_timer
};
// TIC / TOC are used to update the timing profile.  For example:
//
//   TIC { for( n=0; n<n_iter; n++ ) foo(); } TOC( foo, n_iter );
//
// A TIC/TOC block is semantically a single statement (so it works
// fine as the body of a for loop or an if statement.

#define TIC                                                           \
  do {                                                                \
    double _profile_tic = wallclock();                                \
    do

#define TOC(timer,n_calls)                                            \
    while(0);                                                         \
    profile_internal_use_only[profile_internal_use_only_##timer].t += \
    wallclock() - _profile_tic;                                       \
    profile_internal_use_only[profile_internal_use_only_##timer].n += \
      (n_calls);                                                      \
  } while(0);

// TODO: these unsafe macros should be removed, but I didn't want to fight with all the extra while loop and scoping.
//#define KOKKOS_TIC()
//  do {
//    std::chrono::high_resolution_clock::time_point _profile_tic = std::chrono::high_resolution_clock::now();
#define KOKKOS_TIC()                                                  \
  do {                                                                \
    double _profile_tic = wallclock();

// This macro:
// 1) is more flexible but stronger scoped than the normal TIC
// 2) calls kokkos::fence, and thus has performance over head
//#define KOKKOS_TOC_(timer,n_calls, should_barrier)
//    profile_internal_use_only[profile_internal_use_only_##timer].t +=
//      (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - _profile_tic)).count();
//    profile_internal_use_only[profile_internal_use_only_##timer].n +=
//      (n_calls);
//  } while(0);
//  if (should_barrier) Kokkos::fence();

#define KOKKOS_TOC_(timer,n_calls, should_barrier)                     \
    if (should_barrier) Kokkos::fence(); \
    profile_internal_use_only[profile_internal_use_only_##timer].t += \
      wallclock() - _profile_tic;                                     \
    profile_internal_use_only[profile_internal_use_only_##timer].n += \
      (n_calls);                                                      \
  } while(0);

#define KOKKOS_TOC(timer,n_calls) KOKKOS_TOC_(timer, n_calls, 1)
// N for no barrier
#define KOKKOS_TOCN(timer,n_calls) KOKKOS_TOC_(timer, n_calls, 0)

// Do not touch these


typedef struct profile_internal_use_only_timer {
  const char * name;
  double t, t_total;
  int n, n_total;
} profile_internal_use_only_timer_t;

extern profile_internal_use_only_timer_t profile_internal_use_only[];

// Updates the cumulative profile, resets the local profile and, if
// dump is true, writes the local and cumulative profiles to the log.

void
update_profile( int dump );

// Returns a local wallclock in seconds.  Only relative values are
// accurate, and then only within same "short run".

double
wallclock( void );

#endif // _profile_h_
