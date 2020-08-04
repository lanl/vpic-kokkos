#ifndef _binary_h_
#define _binary_h_

#include "collision_private.h"
#include "binary_collision_pipeline.h"

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
    
    // TODO: This is hardcoded, later we will want to have this enabled
    // at compile time instead.
    #ifdef COLLISION_PARTICLE_PARALLEL_POLICY
        binary_collision_pipeline< ParticleParallel<MonteCarlo> > pipeline(
          cop->spi,
          cop->spj,
          cop->interval,
          rng
        );
    #elif COLLISION_VOXEL_PARALLEL_POLICY
        binary_collision_pipeline< VoxelParallel<MonteCarlo> > pipeline(
          cop->spi,
          cop->spj,
          cop->interval,
          rng
        );
    #else
        ERROR(("Collision Parallel Policy not defined"));
    #endif

    pipeline.dispatch(model);
}


// In binary.cc

void
checkpt_binary_collision_op_internal(const binary_collision_op_t * cop);

void *
restore_binary_collision_op_internal(binary_collision_op_t * cop);

#endif /* _binary_h_ */
