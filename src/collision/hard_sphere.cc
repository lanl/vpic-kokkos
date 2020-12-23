#if 0
#define IN_collision
#include "hard_sphere.h"

/* Private interface *********************************************************/

hard_sphere_collision_op_t::hard_sphere_collision_op_t(
  std::string name,
  species_t  * spi,
  species_t  * spj,
  double       radius,
  int          interval
)
  : binary_collision_op_t(name, "Hard sphere", spi, spj, interval),
    radius(radius)
{

}

/* Public interface **********************************************************/

collision_op_t *
hard_sphere(
  const char       * name,
  /**/  species_t  * spi,
  /**/  species_t  * spj,
  const double       radius,
  const int          interval
)
{

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g ||
      radius <= 0 || interval <= 0 )
    ERROR(("Bad args."));

  return new hard_sphere_collision_op_t(name, spi, spj, radius, interval);

}

void
hard_sphere_collision_op_t::apply(
  kokkos_rng_pool_t& rng
)
{

  const int step = spi->g->step;

  if( interval<1 || (step % interval) ) {
      return;
  }

  hard_sphere_model model(radius);
  binary_collision_pipeline<true> pipeline(spi, spj, interval, rng);
  pipeline.dispatch(model);

}
#endif