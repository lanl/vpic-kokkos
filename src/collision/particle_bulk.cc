#define IN_collision
#include "particle_bulk.h"

/* Private interface *********************************************************/

void
checkpt_particle_bulk_collision_op_internal(const particle_bulk_collision_op_t * cop) {
  CHECKPT_PTR( cop->spi );
  CHECKPT_PTR( cop->fspj );
  checkpt_collision_op_internal( cop );
}

void *
restore_particle_bulk_collision_op_internal(particle_bulk_collision_op_t * cop) {
  RESTORE_PTR( cop->spi );
  RESTORE_PTR( cop->fspj );
  return restore_collision_op_internal( cop );
}

/* No public interface *******************************************************/
