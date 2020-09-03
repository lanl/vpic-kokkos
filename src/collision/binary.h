#ifndef _binary_h_
#define _binary_h_

#include "collision_private.h"
#include "kokkos/binary_pipeline_voxel_indirect.h"

/**
 * @brief Base collision operator for binary collsiions.
 *
 * Cannot be used directly, must be subclassed.
 */
struct binary_collision_op_t : public collision_op_t {
  species_t  * spi;
  species_t  * spj;
  int          interval;
};


/**
 * @brief Binary pipeline dispatch wrapper.
 */
template<bool MonteCarlo, class collision_model>
void apply_binary_collision_model_pipeline( binary_collision_op_t * cop,
                                            collision_model& model,
                                            kokkos_rng_pool_t& rng )
{
    const int step = cop->spi->g->step;

    if( cop->interval<1 || (step % cop->interval) ) {
        return;
    }

    std::cout << "Appling Collisions" << std::endl;

    binary_collision_pipeline<MonteCarlo> pipeline(
      cop->spi,
      cop->spj,
      cop->interval,
      rng
    );

    pipeline.dispatch(model);
}


// In binary.cc

void
checkpt_binary_collision_op_internal(const binary_collision_op_t * cop);

void *
restore_binary_collision_op_internal(binary_collision_op_t * cop);

#endif /* _binary_h_ */
