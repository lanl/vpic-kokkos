#define IN_collision
#include "drag.h"

/* Private interface *********************************************************/

void
checkpt_drag_collision_op(const void * cop) {
  drag_collision_op_t * drag = (drag_collision_op_t *) cop;
  CHECKPT(drag, 1);
  checkpt_particle_bulk_collision_op_internal( (const particle_bulk_collision_op_t *) cop );
}

void *
restore_drag_collision_op() {
  drag_collision_op_t * drag;
  RESTORE(drag);
  return restore_particle_bulk_collision_op_internal( (particle_bulk_collision_op_t *) drag );
}

void
apply_drag_collision_op( collision_op_t * cop,
			kokkos_rng_pool_t& rng ) {
  drag_collision_op_t * drag = (drag_collision_op_t *) cop;
  drag_model model(drag->stopping_cx0);
  apply_particle_bulk_collision_model_pipeline<false>((particle_bulk_collision_op_t *) cop, model, rng); // Drag: Monte Carlo is false with current method.
}

void
delete_drag_collision_op(collision_op_t * cop) {
  drag_collision_op_t * drag = (drag_collision_op_t *) cop;
  UNREGISTER_OBJECT(drag);
  FREE(drag);
}

/* Public interface **********************************************************/

collision_op_t *
drag(
  const char       * name,
  /**/  species_t  * spi,
  /**/  fluid_species_t  * spj,
  //  const double       cvar0,
  float (*stoppingfunc)(float),
  const int          interval
)
{

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g || interval <= 0 )
    ERROR(("Bad args."));

  drag_collision_op_t * drag;
  MALLOC( drag, 1);
  MALLOC( drag->name, strlen(name) +1 );
  strncpy( drag->name, name, strlen(name)+1);

  drag->spi         = spi;
  drag->spj         = spj;
  drag->stopping_cx0   = stoppingfunc;
  //  ta->cvar0       = cvar0 * spi->q * spi->q * spj->q * spj->q;
  drag->interval    = interval;
  drag->apply_cop   = &apply_drag_collision_op;
  drag->delete_cop  = &delete_drag_collision_op;
  drag->next        = NULL;

  REGISTER_OBJECT(drag,
                  &checkpt_drag_collision_op,
                  &restore_drag_collision_op,
                  NULL);

  return drag;

}
