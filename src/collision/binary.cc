#define IN_collision
#include "binary.h"

/* Private interface *********************************************************/

void
checkpt_binary_collision_op_internal(const binary_collision_op_t * cop) {
  CHECKPT_PTR( cop->spi );
  CHECKPT_PTR( cop->spj );
  checkpt_collision_op_internal( cop );
}

void *
restore_binary_collision_op_internal(binary_collision_op_t * cop) {
  RESTORE_PTR( cop->spi );
  RESTORE_PTR( cop->spj );
  return restore_collision_op_internal( cop );
}

/* No public interface *******************************************************/
