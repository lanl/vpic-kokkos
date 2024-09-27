#include "profile.h"
#include "sys/time.h"
#include "../mp/mp.h"

profile_internal_use_only_timer_t profile_internal_use_only[] = {
# define PROFILE_TIMER_INIT( timer ) { #timer, 0., 0., 0, 0, -1 },
  PROFILE_TIMERS( PROFILE_TIMER_INIT )
# undef PROFILE_TIMER_INIT
  { NULL, 0., 0., 0, 0, -1 }
};

profile_internal_use_only_timer_t profile_internal_use_only_mean[] = {
# define PROFILE_TIMER_INIT( timer ) { #timer, 0., 0., 0, 0, -1 },
  PROFILE_TIMERS( PROFILE_TIMER_INIT )
# undef PROFILE_TIMER_INIT
  { NULL, 0., 0., 0, 0, -1 }
};

profile_internal_use_only_timer_t profile_internal_use_only_min[] = {
# define PROFILE_TIMER_INIT( timer ) { #timer, 0., 0., 0, 0, -1 },
  PROFILE_TIMERS( PROFILE_TIMER_INIT )
# undef PROFILE_TIMER_INIT
  { NULL, 0., 0., 0, 0, -1 }
};

profile_internal_use_only_timer_t profile_internal_use_only_max[] = {
# define PROFILE_TIMER_INIT( timer ) { #timer, 0., 0., 0, 0, -1 },
  PROFILE_TIMERS( PROFILE_TIMER_INIT )
# undef PROFILE_TIMER_INIT
  { NULL, 0., 0., 0, 0, -1 }
};

void
update_profile( int dump ) {
  profile_internal_use_only_timer_t * p;
  double sum = 0, sum_total = 0;

  for( p=profile_internal_use_only; p->name; p++ ) {
    p->t_total += p->t;
    p->n_total += p->n;
    sum        += p->t;
    sum_total  += p->t_total;
  }

  if( dump ) {
    log_printf( "\n" // 8901234567890123456 | xxx% x.xe+xx x.xe+xx x.xe+xx | xxx% x.xe+xx x.xe+xx x.xe+xx
                "                           |      Since Last Update       |     Since Last Restore\n"
                "    Operation              | Pct   Time    Count    Per   | Pct   Time    Count    Per\n"
                "---------------------------+------------------------------+------------------------------\n" );

    for( p=profile_internal_use_only; p->name; p++ ) {
      if( p->n==0 && p->n_total==0 ) continue;
      log_printf( "%26.26s | % 3d%% %.3e %.1e %.1e | % 3d%% %.3e %.1e %.1e\n",
                  p->name,
                  (int)( 100.*p->t/sum + 0.5 ), p->t,
                  (double)p->n,
                  p->t/(DBL_EPSILON+(double)p->n ),
                  (int)( 100.*p->t_total/sum_total + 0.5 ), p->t_total,
                  (double)p->n_total,
                  p->t_total/(DBL_EPSILON+(double)p->n_total) );
    }

    log_printf( "\n" );
  }

  for( p=profile_internal_use_only; p->name; p++ ) {
    p->t = 0;
    p->n = 0;
  }
}


void
update_profile_meanminmax( int dump ) {
  profile_internal_use_only_timer_t * p;
  //profile_internal_use_only_timer_t * pmean;
  profile_internal_use_only_timer_t * pmin;
  profile_internal_use_only_timer_t * pmax;
  double sum = 0;
  double sum_total = 0;

  for( p=profile_internal_use_only; p->name; p++ ) {
    p->t_total += p->t;
    p->n_total += p->n;
    sum        += p->t;
    sum_total  += p->t_total;
  }

  //pmean = profile_internal_use_only_mean;
  pmin  = profile_internal_use_only_min;
  pmax  = profile_internal_use_only_max;
  for( p=profile_internal_use_only; p->name; p++ ) {
    //mp_allsum_d          (&p->t, &pmean->t, 1);
    mp_allminloc_scalar_d(&p->t, &pmin->t, &pmin->rank);
    mp_allmaxloc_scalar_d(&p->t, &pmax->t, &pmax->rank);

    //pmean->t /= world_size;  // util/util_base.h macro

    //pmean++;
    pmin++;
    pmax++;
  }

  // To check absolute timing across ranks, e.g. for single time steps
  //double sum_mean = 0;
  //double sum_min = 0;
  //double sum_max = 0;
  //int dummy;
  //mp_allminloc_scalar_d(&sum, &sum_min, &dummy);
  //mp_allmaxloc_scalar_d(&sum, &sum_max, &dummy);
  //mp_allsum_d          (&sum, &sum_mean, 1);
  //sum_mean /= world_size;

  if( dump ) {
    #if defined(VPIC_PRINT_MORE_DIGITS)
    log_printf( "\n" // 8901234567890123456 | xxx% x.xxxe+xx x.xe+xx x.xxxe+xx | x.xxxe+xx x.xxxe+xx  xxx%   xxx% xxxxx | xxx% x.xxxe+xx x.xe+xx x.xxxe+xx
                "    Rank %-6d            | Since Last Update                | All Ranks                          Max | Since Last Restore\n"
                "    Operation              | Pct   Time      Count    Per     | Min       Max     Min/Max Me/Max  Rank | Pct    Time     Count     Per\n"
                "---------------------------+----------------------------------+----------------------------------------+---------------------------------\n", world_rank );
    #else
    log_printf( "\n" // 8901234567890123456 | xxx% x.xe+xx x.xe+xx x.xe+xx | x.xe+xx x.xe+xx  xxx%   xxx% xxxxx | xxx% x.xe+xx x.xe+xx x.xe+xx
                "    Rank %-6d            | Since Last Update            | All Ranks                      Max | Since Last Restore\n"
                "    Operation              | Pct   Time    Count    Per   | Min     Max   Min/Max Me/Max  Rank | Pct   Time    Count    Per\n"
                "---------------------------+------------------------------+------------------------------------+-----------------------------\n", world_rank );
    #endif

    //pmean = profile_internal_use_only_mean;
    pmin  = profile_internal_use_only_min;
    pmax  = profile_internal_use_only_max;
    for( p=profile_internal_use_only; p->name; p++ ) {
      if( p->n==0 && p->n_total==0 ) {
        //pmean++;
        pmin++;
        pmax++;
        continue;
      }

      // compare this rank to fastest/slowest
      // guard against divide by zero b/c pmax->t can be zero even when n > 0
      int min_max = 0;
      int me_max = 0;
      if (pmax->t > 0) {
        min_max = (int)( 100.*pmin->t/pmax->t + 0.5 );
        me_max = (int)( 100.*p->t/pmax->t + 0.5 );
      }

      // pct of time compared to other operations
      // guard against divide by zero b/c sum can be zero at simulation's end
      int pct = 0;
      if (sum > 0) {
        pct = (int)( 100.*p->t/sum + 0.5 );
      }

      if (p->n > 0) {
          #if defined(VPIC_PRINT_MORE_DIGITS)
          log_printf( "%26.26s | %3d%% %.3e %.1e %.3e | %.3e %.3e  %3d%%   %3d%% % 5d | %3d%% %.3e %.1e %.3e\n",
          #else
          log_printf( "%26.26s | %3d%% %.1e %.1e %.1e | %.1e %.1e  %3d%%   %3d%% % 5d | %3d%% %.1e %.1e %.1e\n",
          #endif
                      p->name,
                      // Since Last Update
                      pct,
                      p->t,
                      (double)p->n,
                      p->t/(DBL_EPSILON+(double)p->n ),
                      // All Ranks
                      pmin->t,
                      pmax->t,
                      min_max,
                      me_max,
                      pmax->rank,
                      // Since Last Restore
                      (int)( 100.*p->t_total/sum_total + 0.5 ),
                      p->t_total,
                      (double)p->n_total,
                      p->t_total/(DBL_EPSILON+(double)p->n_total) );
      } else {
          #if defined(VPIC_PRINT_MORE_DIGITS)
          log_printf( "%26.26s |                                  |                                        | %3d%% %.3e %.1e %.3e\n",
          #else
          log_printf( "%26.26s |                              |                                    | %3d%% %.1e %.1e %.1e\n",
          #endif
                      p->name,
                      // Since Last Restore
                      (int)( 100.*p->t_total/sum_total + 0.5 ),
                      p->t_total,
                      (double)p->n_total,
                      p->t_total/(DBL_EPSILON+(double)p->n_total) );
      }

      //pmean++;
      pmin++;
      pmax++;
    }
    #if defined(VPIC_PRINT_MORE_DIGITS)
    log_printf( "%26.26s | %3d%% %.3e                   |                                        | %3d%% %.3e\n",
    #else
    log_printf( "%26.26s | %3d%% %.1e                 |                                    | %3d%% %.1e\n",
    #endif
                "Total", 100, sum, 100, sum_total );
    log_printf( "\n" );
  }

  for( p=profile_internal_use_only; p->name; p++ ) {
    p->t = 0;
    p->n = 0;
  }
}


double
wallclock( void ) {
  struct timeval tv[1];
  gettimeofday( tv, NULL );
  return (double)(tv->tv_sec) + 1e-6*(double)(tv->tv_usec);
}
