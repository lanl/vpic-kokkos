#define IN_collision
#include "binary_neutral.h"

/* Private interface *********************************************************/

void
checkpt_binary_neutral_collision_op_internal(const binary_neutral_collision_op_t * cop) {
  CHECKPT_PTR( cop->spi );
  CHECKPT_PTR( cop->spj );
  // CHECKPT_PTR( cop->spp1 );
  // CHECKPT_PTR( cop->spp2 );
  checkpt_collision_op_internal( cop );
}

void *
restore_binary_neutral_collision_op_internal(binary_neutral_collision_op_t * cop) {
  RESTORE_PTR( cop->spi );
  RESTORE_PTR( cop->spj );
  // RESTORE_PTR( cop->spp1 );
  // RESTORE_PTR( cop->spp2 );
  return restore_collision_op_internal( cop );
}

/* No public interface *******************************************************/
