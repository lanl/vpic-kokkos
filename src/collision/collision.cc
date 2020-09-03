#define IN_collision
#include "collision_private.h"

/* Private interface *********************************************************/

void
checkpt_collision_op_internal( const collision_op_t * cop ) {
  CHECKPT_STR( cop->name );
  CHECKPT_SYM( cop->apply_cop );
  CHECKPT_SYM( cop->delete_cop );
  CHECKPT_PTR( cop->next );
}

void *
restore_collision_op_internal( collision_op_t * cop ) {
  RESTORE_STR( cop->name );
  RESTORE_SYM( cop->apply_cop );
  RESTORE_SYM( cop->delete_cop );
  RESTORE_PTR( cop->next );
  return cop;
}

/* Public interface **********************************************************/

int
num_collision_op( const collision_op_t * cop_list ) {
  const collision_op_t * cop;
  int n = 0;
  LIST_FOR_EACH( cop, cop_list ) n++;
  return n;
}

void
apply_collision_op_list( collision_op_t    * cop_list,
                         kokkos_rng_pool_t & rng_pool ) {
  collision_op_t * cop;
  LIST_FOR_EACH( cop, cop_list ) cop->apply_cop( cop, rng_pool );
}

void
delete_collision_op_list( collision_op_t * cop_list ) {
  collision_op_t * cop;
  while( cop_list ) {
    cop = cop_list;
    cop_list = cop_list->next;
    cop->delete_cop( cop );
  }
}

collision_op_t *
append_collision_op( collision_op_t * cop,
                     collision_op_t ** cop_list ) {
  if( !cop || !cop_list || cop->next ) ERROR(( "Bad args" ));
  cop->next = *cop_list;
  *cop_list = cop;
  return cop;
}
