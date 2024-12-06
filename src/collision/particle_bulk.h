#ifndef _particle_bulk_h_
#define _particle_bulk_h_

#include "collision_private.h"
#include "kokkos/particle_bulk_pipeline_voxel_indirect.h"

/**
 * @brief Base collision operator for particle-bulk binary collisions.
 *
 * Cannot be used directly, must be subclassed.
 */
struct particle_bulk_collision_op_t : public collision_op_t {
  species_t  * spi;
  fluid_species_t  * spj;
  int          interval;
};


/**
 * @brief Particle bulk pipeline dispatch wrapper.
 */
template<bool MonteCarlo, class collision_model>
void apply_particle_bulk_collision_model_pipeline( particle_bulk_collision_op_t * cop,
						   collision_model& model,
						   kokkos_rng_pool_t& rng )
{
    const int step = cop->spi->g->step;

    if( cop->interval<1 || (step % cop->interval) ) {
        return;
    }

    //std::cout << "Applying Collisions" << std::endl;

    particle_bulk_collision_pipeline<MonteCarlo> pipeline(
      cop->spi,
      cop->spj,
      cop->interval,
      rng
    );

    pipeline.dispatch(model);
}


// In particle_bulk.cc

void
checkpt_particle_bulk_collision_op_internal(const particle_bulk_collision_op_t * cop);

void *
restore_particle_bulk_collision_op_internal(particle_bulk_collision_op_t * cop);

#endif /* _particle_bulk_h_ */
