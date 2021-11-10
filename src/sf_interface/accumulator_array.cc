/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

#include "sf_interface.h"

static int
aa_n_pipeline(void) {
  int                       n = serial.n_pipeline;
  if( n<thread.n_pipeline ) n = thread.n_pipeline;
  return n; /* max( {serial,thread,spu}.n_pipeline ) */
}

void
checkpt_accumulator_array( const accumulator_array_t * aa ) {
  CHECKPT( aa, 1 );
  CHECKPT_ALIGNED( aa->a, (size_t)(aa->n_pipeline+1)*(size_t)aa->stride, 128 );
  CHECKPT_PTR( aa->g );
}

accumulator_array_t *
restore_accumulator_array( void ) {
  accumulator_array_t * aa;
  RESTORE( aa );
  RESTORE_ALIGNED( aa->a );
  RESTORE_PTR( aa->g );
  if( aa->n_pipeline!=aa_n_pipeline() )
    ERROR(( "Number of accumulators restored is not the same as the number of "
            "accumulators checkpointed.  Did you change the number of threads "
            "per process between checkpt and restore?" ));
  return aa;
}

accumulator_array_t *
new_accumulator_array( grid_t * g ) {
  accumulator_array_t * aa;
  if( !g ) ERROR(( "Bad grid."));
  //MALLOC( aa, 1 );

  // TODO: this is likely too big
  //printf("Making %d copies of accumulator \n",aa_n_pipeline()+1 );
  aa = new accumulator_array_t(
          //(size_t)(aa->n_pipeline+1)*(size_t)aa->stride,
          //(size_t)(aa_n_pipeline()+1)*(size_t)(POW2_CEIL(g->nv,2))
          g->nv
  );
  aa->n_pipeline = aa_n_pipeline();
  aa->stride     = POW2_CEIL(g->nv,2);
  aa->g          = g;
  //aa->na         = (size_t)(aa->n_pipeline+1)*(size_t)aa->stride;
  MALLOC_ALIGNED( aa->a, aa->na, 128 );
  CLEAR( aa->a, aa->na);
  REGISTER_OBJECT( aa, checkpt_accumulator_array, restore_accumulator_array,
                  NULL );
  return aa;
}

void
delete_accumulator_array( accumulator_array_t * aa ) {
  if( !aa ) return;
  UNREGISTER_OBJECT( aa );
  FREE_ALIGNED( aa->a );
  FREE( aa );
}

void
accumulator_array_t::copy_to_host() {

  Kokkos::deep_copy(k_a_h, k_a_d);

  // Avoid capturing this
  auto& k_accumulators_h = k_a_h;
  accumulator_t * host_accum = a;

  Kokkos::parallel_for("copy accumulator to host",
    KOKKOS_TEAM_POLICY_HOST (na, Kokkos::AUTO),
    KOKKOS_LAMBDA (const KOKKOS_TEAM_POLICY_HOST::member_type &team_member) {

      const unsigned int i = team_member.league_rank();
      /* TODO: Do we really need a 2d loop here*/
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ACCUMULATOR_ARRAY_LENGTH), [=] (int j) {
        host_accum[i].jx[j] = k_accumulators_h(i, accumulator_var::jx, j);
        host_accum[i].jy[j] = k_accumulators_h(i, accumulator_var::jy, j);
        host_accum[i].jz[j] = k_accumulators_h(i, accumulator_var::jz, j);
      });

    });

}

void
accumulator_array_t::copy_to_device() {

  // Avoid capturing this
  auto& k_accumulators_h = k_a_h;
  accumulator_t * host_accum = a;

  Kokkos::parallel_for("copy accumulator to device",
    KOKKOS_TEAM_POLICY_HOST (na, Kokkos::AUTO),
    KOKKOS_LAMBDA (const KOKKOS_TEAM_POLICY_HOST::member_type &team_member) {

      const unsigned int i = team_member.league_rank();
      /* TODO: Do we really need a 2d loop here*/
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ACCUMULATOR_ARRAY_LENGTH), [=] (int j) {
        k_accumulators_h(i, accumulator_var::jx, j) = host_accum[i].jx[j];
        k_accumulators_h(i, accumulator_var::jy, j) = host_accum[i].jy[j];
        k_accumulators_h(i, accumulator_var::jz, j) = host_accum[i].jz[j];
      });

    });
  Kokkos::deep_copy(k_a_d, k_a_h);

}
