#define IN_collision
#include "charge_exchange.h"

/* Private interface *********************************************************/

void
checkpt_cex_collision_op(const void * cop) {
  cex_collision_op_t * cex = (cex_collision_op_t *) cop;
  CHECKPT(cex, 1);
  checkpt_particle_bulk_collision_op_internal( (const particle_bulk_collision_op_t *) cop );
}

void *
restore_cex_collision_op() {
  cex_collision_op_t * cex;
  RESTORE(cex);
  return restore_particle_bulk_collision_op_internal( (particle_bulk_collision_op_t *) cex );
}

void
apply_cex_collision_op( collision_op_t * cop,
			kokkos_rng_pool_t& rng ) {
  cex_collision_op_t * cex = (cex_collision_op_t *) cop;
  cex_model model(cex->sigma_cx0);
  apply_particle_bulk_collision_model_pipeline<true>((particle_bulk_collision_op_t *) cop, model, rng); // To-do: Change MC to true!
}

void
delete_cex_collision_op(collision_op_t * cop) {
  cex_collision_op_t * cex = (cex_collision_op_t *) cop;
  UNREGISTER_OBJECT(cex);
  FREE(cex);
}

/* Public interface **********************************************************/

collision_op_t *
charge_exchange(
  const char       * name,
  /**/  species_t  * spi,
  /**/  fluid_species_t  * spj,
  //  const double       cvar0,
  float (*sigmafunc)(float,float),
  const int          interval
)
{

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g || interval <= 0 )
    ERROR(("Bad args."));

  cex_collision_op_t * cex;
  MALLOC( cex, 1);
  MALLOC( cex->name, strlen(name) +1 );
  strncpy( cex->name, name, strlen(name)+1);

  cex->spi         = spi;
  cex->spj         = spj;
  cex->sigma_cx0   = sigmafunc;
  //  ta->cvar0       = cvar0 * spi->q * spi->q * spj->q * spj->q;
  cex->interval    = interval;
  cex->apply_cop   = &apply_cex_collision_op;
  cex->delete_cop  = &delete_cex_collision_op;
  cex->next        = NULL;

  REGISTER_OBJECT(cex,
                  &checkpt_cex_collision_op,
                  &restore_cex_collision_op,
                  NULL);

  return cex;

}
