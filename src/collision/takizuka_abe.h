#ifndef _takizuka_abe_h_
#define _takizuka_abe_h_

#include "binary.h"

/**
 * @brief Takizuka-Abe collision operator.
 */
struct takizuka_abe_collision_op_t : public binary_collision_op_t {
  double cvar0;
};


/**
 * @brief Takizuka-Abe binary collision model.
 */
struct takizuka_abe_model : public collision_model {
  const float cvar;

  takizuka_abe_model( float cvar ) : cvar(cvar) { };

  /**
   * @brief tan(theta/2) is normally distributed and variance scales ~ ur^-3/2.
   */
  KOKKOS_INLINE_FUNCTION
  float tan_theta_half(
    kokkos_rng_state_t& rg,
    float E,
    float nvdt
  ) const
  {
    float sigma = sqrtf(cvar*nvdt/(E*E));
    sigma = sigma > 1 ? 1 : sigma;
    return rg.normal(0, sigma);
  }

};

#endif /* _takizuka_abe_h_ */
