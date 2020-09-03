#define IN_collision
#include "takizuka_abe.h"

/* Private interface *********************************************************/

void
checkpt_takizuka_abe_collision_op(const void * cop) {
  takizuka_abe_collision_op_t * ta = (takizuka_abe_collision_op_t *) cop;
  CHECKPT(ta, 1);
  checkpt_binary_collision_op_internal( (const binary_collision_op_t *) cop );
}

void *
restore_takizuka_abe_collision_op() {
  takizuka_abe_collision_op_t * ta;
  RESTORE(ta);
  return restore_binary_collision_op_internal( (binary_collision_op_t *) ta );
}

void
apply_takizuka_abe_collision_op( collision_op_t * cop,
                                 kokkos_rng_pool_t& rng ) {
  takizuka_abe_collision_op_t * ta = (takizuka_abe_collision_op_t *) cop;
  takizuka_abe_model model(ta->cvar0);
  apply_binary_collision_model_pipeline<false>((binary_collision_op_t *) cop, model, rng);
}

void
delete_takizuka_abe_collision_op(collision_op_t * cop) {
  takizuka_abe_collision_op_t * ta = (takizuka_abe_collision_op_t *) cop;
  UNREGISTER_OBJECT(ta);
  FREE(ta);
}

/* Public interface **********************************************************/

collision_op_t *
takizuka_abe(
  const char       * name,
  /**/  species_t  * spi,
  /**/  species_t  * spj,
  const double       cvar0,
  const int          interval
)
{

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g ||
      cvar0 <= 0 || interval <= 0 )
    ERROR(("Bad args."));

  takizuka_abe_collision_op_t * ta;
  MALLOC( ta, 1);
  MALLOC( ta->name, strlen(name) +1 );
  strcpy( ta->name, name ); 
  
  ta->spi         = spi;
  ta->spj         = spj;
  ta->cvar0       = cvar0 * spi->q * spi->q * spj->q * spj->q;
  ta->interval    = interval;
  ta->apply_cop   = &apply_takizuka_abe_collision_op;
  ta->delete_cop  = &delete_takizuka_abe_collision_op;
  ta->next        = NULL;

  REGISTER_OBJECT(ta,
                  &checkpt_takizuka_abe_collision_op,
                  &restore_takizuka_abe_collision_op,
                  NULL);

  return ta;

}
