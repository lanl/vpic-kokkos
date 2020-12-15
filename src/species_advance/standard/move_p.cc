#define IN_spa
#include "spa_private.h"

// Legacy wrapper for host-side moves in emitters.

int
move_p( particle_t          * ALIGNED(128) p0,
        particle_mover_t    * ALIGNED(16)  pm,
        accumulator_array_t *              aa,
        const grid_t        *              g,
        const float                        qsp )
{


  ParticleViewWrapper pview(p0);

  SELECT_GEOMETRY(g->geometry, geo, {

    return move_p_kokkos(
      pview,
      pview,
      pm,
      aa->get_host_accumulator(),
      g->k_neighbor_h,
      g->rangel,
      g->rangeh,
      qsp
    );

  });

}
