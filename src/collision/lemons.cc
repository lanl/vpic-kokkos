#define IN_collision
#include "lemons.h"

/* Private interface *********************************************************/

void
checkpt_lemons_collision_op(const void * cop) {
  lemons_collision_op_t * le = (lemons_collision_op_t *) cop;
  CHECKPT(le, 1);
  checkpt_particle_bulk_collision_op_internal( (const particle_bulk_collision_op_t *) cop );
}

void *
restore_lemons_collision_op() {
  lemons_collision_op_t * le;
  RESTORE(le);
  return restore_particle_bulk_collision_op_internal( (particle_bulk_collision_op_t *) le );
}

void
apply_lemons_collision_op( collision_op_t * cop,
                                 kokkos_rng_pool_t& rng ) {
  lemons_collision_op_t * le = (lemons_collision_op_t *) cop;
  lemons_model model(le->cvar0);
  apply_particle_bulk_collision_model_pipeline<false>((particle_bulk_collision_op_t *) cop, model, rng);
}

void
delete_lemons_collision_op(collision_op_t * cop) {
  lemons_collision_op_t * le = (lemons_collision_op_t *) cop;
  UNREGISTER_OBJECT(le);
  FREE(le);
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

  lemons_collision_op_t * le;
  MALLOC( le, 1);
  MALLOC( le->name, strlen(name) +1 );
  strncpy( le->name, name, strlen(name)+1);

  le->spi         = spi;
  le->spj         = spj;
  le->cvar0       = cvar0 * spi->q * spi->q * spj->q * spj->q;
  le->interval    = interval;
  le->apply_cop   = &apply_lemons_collision_op;
  le->delete_cop  = &delete_lemons_collision_op;
  le->next        = NULL;

  REGISTER_OBJECT(le,
                  &checkpt_lemons_collision_op,
                  &restore_lemons_collision_op,
                  NULL);

  return le;

}
