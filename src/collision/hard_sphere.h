#ifndef _hard_sphere_h_
#define _hard_sphere_h_

#include "binary.h"

/**
 * @brief Hard-sphere binary collision operator.
 */
struct hard_sphere_collision_op_t : public binary_collision_op_t {
  double radius;
};


/**
 * @brief Hard-sphere binary collision model.
 */
struct hard_sphere_model : public binary_collision_model {
  const float radius;

  KOKKOS_INLINE_FUNCTION
  hard_sphere_model(
    float radius
  )
  : radius(radius) {};

  /**
   * @brief The total cross section is pi*(2*r)^2.
   */
  KOKKOS_INLINE_FUNCTION
  float cross_section(
    kokkos_rng_state_t& rg,
    float E,
    float nvdt
  ) const
  {
    return 4*M_PI*radius*radius;
  }

  /**
   * @brief Hard-sphere scattering angle.
   *
   * In a hard sphere collision, if two particles are known to have
   * collided (but we know nothing about the details of the collisions),
   * we know that in the reduced mass system, the reduced mass particle
   * was incident transversely on the origin particle somewhere in the
   * circle from [0,R).
   *
   * Given this normalized impact parameter, b/R, the collision angle
   * theta in the reduced mass system is given by:
   *    sin( pi/2 - theta/2 ) = b/R
   * or:
   *    cos( theta/2 ) = b/R
   * or:
   *    tan( theta/2 ) = sqrt( 1 - (b/R)^2 ) / (b/R)
   *
   * To avoid numerical issues for small b/R, we work in double precision.
   */
  KOKKOS_INLINE_FUNCTION
  float tan_theta_half(
    kokkos_rng_state_t& rg,
    float E,
    float nvdt
  ) const
  {
    double b = rg.drand(0, 1);
    return sqrt(1 - b*b)/b;
  }

};

#endif /* _hard_sphere_h_ */
