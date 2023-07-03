#include "rng.h"
#include "../checkpt/checkpt.h"
#include "../profile/profile.h"

/* Private API ***************************************************************/

void
checkpt_rng_pool ( rng_pool_t * rp ) {
  int n;
  uint64_t seed;

  // Set a device seed for use on restore. Access to the
  // internal Kokkos state is not allowed.
  if( rp->n_rng > 0 ) {
    seed = u64rand( rp->rng[0] );
  } else {
    seed = wallclock()*1e9;
  }

  CHECKPT( rp, 1 );
  CHECKPT( rp->rng, rp->n_rng );
  for( n=0; n<rp->n_rng; n++ ) CHECKPT_PTR( rp->rng[n] );
  CHECKPT_VAL( uint64_t, seed );

  // Reseed the device rng to maintain consistency between
  // continued and restarted version as much as possible.
  rp->k_rng_pool = kokkos_rng_pool_t(seed);

}

rng_pool_t *
restore_rng_pool( void ) {
  rng_pool_t * rp;
  int n;
  uint64_t seed;

  RESTORE( rp );
  RESTORE( rp->rng );
  for( n=0; n<rp->n_rng; n++ ) RESTORE_PTR( rp->rng[n] );
  RESTORE_VAL(uint64_t, seed);

  new(&rp->k_rng_pool) kokkos_rng_pool_t(seed);

  return rp;
}

/* Public API ****************************************************************/

rng_pool_t *
new_rng_pool( int n_rng,
              int seed,
              int sync ) {
  rng_pool_t * rp;
  int n;
  if( n_rng<1 ) ERROR(( "Bad args" ));
  rp = new rng_pool_t();
  MALLOC( rp->rng, n_rng );
  for( n=0; n<n_rng; n++ ) rp->rng[n] = new_rng( 0 );
  rp->n_rng = n_rng;
  seed_rng_pool( rp, seed, sync );
  REGISTER_OBJECT( rp, checkpt_rng_pool, restore_rng_pool, NULL );
  return rp;
}

void
delete_rng_pool( rng_pool_t * rp ) {
  int n;
  if( !rp ) return;
  UNREGISTER_OBJECT( rp );
  for( n=0; n<rp->n_rng; n++ ) delete_rng( rp->rng[n] );
  FREE( rp->rng );
  delete(rp);
}

rng_pool_t *
seed_rng_pool( rng_pool_t * RESTRICT rp,
               int seed,
               int sync ) {
  int n;
  if( !rp ) ERROR(( "Bad args" ));
  seed = (sync ? world_size : world_rank) + (world_size+1)*rp->n_rng*seed;
  rp->k_rng_pool = kokkos_rng_pool_t(seed);
  for( n=0; n<rp->n_rng; n++ ) seed_rng( rp->rng[n], seed + (world_size+1)*n );
  return rp;
}

