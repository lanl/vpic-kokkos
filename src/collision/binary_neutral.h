#ifndef _binary_neutral_h_
#define _binary_neutral_h_

// #include "collision_private.h"
#include "kokkos/binary_neutral_pipeline_voxel_indirect.h"


/**
 * @brief Binary neutral collision pipeline dispatch wrapper.
 */
template<bool MonteCarlo, class collision_model>
void apply_binary_neutral_collision_model_pipeline( 
  binary_neutral_collision_op_t * cop,
  collision_model& model,
  kokkos_rng_pool_t& rng )
{
  const int step = cop->spi->g->step;

  if( cop->interval<1 || (step % cop->interval) ) {
    return;
  }

  binary_neutral_collision_pipeline<MonteCarlo> pipeline(
    cop->spi,
    cop->spj,
    cop->interval,
    rng,
    cop->field
    // cop->spp1,
    // cop->spp2
  );

  pipeline.dispatch(model);
}

// In binary_neutral.cc

void
checkpt_binary_neutral_collision_op_internal(const binary_neutral_collision_op_t * cop);

void *
restore_binary_neutral_collision_op_internal(binary_neutral_collision_op_t * cop);

#endif /* _binary_neutral_h_ */
