#define IN_collision
#include "lemons.h"

/* Private interface *********************************************************/

void
checkpt_lemons_collision_op(const void * cop) {
  lemons_collision_op_t * ta = (lemons_collision_op_t *) cop;
  CHECKPT(ta, 1);
  checkpt_particle_bulk_collision_op_internal( (const particle_bulk_collision_op_t *) cop );
}

void *
restore_lemons_collision_op() {
  lemons_collision_op_t * ta;
  RESTORE(ta);
  return restore_particle_bulk_collision_op_internal( (particle_bulk_collision_op_t *) ta );
}

void
apply_lemons_collision_op( collision_op_t * cop,
                                 kokkos_rng_pool_t& rng ) {
  lemons_collision_op_t * ta = (lemons_collision_op_t *) cop;
  lemons_model model(ta->cvar0);
  apply_particle_bulk_collision_model_pipeline<false>((particle_bulk_collision_op_t *) cop, model, rng);
}

void
delete_lemons_collision_op(collision_op_t * cop) {
  lemons_collision_op_t * ta = (lemons_collision_op_t *) cop;
  UNREGISTER_OBJECT(ta);
  FREE(ta);
}

/* Public interface **********************************************************/

collision_op_t *
lemons(
  const char       * name,
  /**/  species_t  * spi,
  /**/  fluid_species_t  * spj,
  const double       cvar0,
  const int          interval
)
{

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g ||
      cvar0 <= 0 || interval <= 0 )
    ERROR(("Bad args."));

  lemons_collision_op_t * ta;
  MALLOC( ta, 1);
  MALLOC( ta->name, strlen(name) +1 );
  strncpy( ta->name, name, strlen(name)+1);

  ta->spi         = spi;
  ta->spj         = spj;
  ta->cvar0       = cvar0 * spi->q * spi->q * spj->q * spj->q;
  ta->interval    = interval;
  ta->apply_cop   = &apply_lemons_collision_op;
  ta->delete_cop  = &delete_lemons_collision_op;
  ta->next        = NULL;

  REGISTER_OBJECT(ta,
                  &checkpt_lemons_collision_op,
                  &restore_lemons_collision_op,
                  NULL);

  return ta;

}
