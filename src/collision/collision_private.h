#ifndef _collision_private_h_
#define _collision_private_h_

#ifndef IN_collision
#error "Do not include collision_private.h; use collision.h"
#endif

#include "collision.h"
#include "src/util/rng_policy.h"
#include "../particle_operations/sort.h"
#include "../particle_operations/shuffle.h"

#define RANK_TO_INDEX(rank,ix,iy,iz,nx,ny,nz) do {        \
    int _ix, _iy, _iz;                                    \
    _ix  = (rank);   /* ix = ix + gpx*( iy + gpy*iz ) */  \
    _iy  = _ix/(nx); /* iy = iy + gpy*iz */               \
    _ix -= _iy*(nx); /* ix = ix */                        \
    _iz  = _iy/(ny); /* iz = iz */                        \
    _iy -= _iz*(ny); /* iy = iy */                        \
    (ix) = _ix;                                           \
    (iy) = _iy;                                           \
    (iz) = _iz;                                           \
  } while(0)

// Templated small array used for accumulation/reduction
template <class ScalarType, int N>
struct Accum {
  // We'll store partial sums in v[0..N-1]
  enum : int { n = N };
  ScalarType v[n] = { 0 };

  // Operator += for combining two partial sums
  KOKKOS_INLINE_FUNCTION
  Accum& operator+=(const Accum &b) {
    for (int i = 0; i < n; ++i) {
      v[i] += b.v[i];
    }
    return *this;
  }
};
typedef Accum<float, 5> gmomType; //0:mass, 1-3:momentum, 4-energy
typedef Accum<float, 26> gmomType26; //before+after collision for 2 species

namespace Kokkos { //required
template <>
struct reduction_identity<gmomType> {
  KOKKOS_INLINE_FUNCTION
  static gmomType sum() {
    // If gmomType() is guaranteed to construct an identity for summation (e.g. init zeros),
    // then returning a default-constructed object is fine.
    return gmomType();
  }
};

template<>
struct reduction_identity<gmomType26> {
    KOKKOS_INLINE_FUNCTION
    static gmomType26 sum() { return gmomType26(); }
};
    
}

typedef void
(*apply_collision_op_func_t)( struct collision_op * cop,
                              kokkos_rng_pool_t   & rng);

typedef void
(*delete_collision_op_func_t) ( struct collision_op * cop );

struct collision_op {
  char * name;
  apply_collision_op_func_t  apply_cop;
  delete_collision_op_func_t delete_cop;
  collision_op_t * next;
};

/**
 * @brief Base collision model
 *
 * Implements all required methods, but does nothing.
 */
struct collision_model {

  /**
   * @brief Tangent of half the polar scattering angle.
   *
   * @param rg Random number generator
   * @param E Collision energy
   * @param nvdt Areal density of particles encountered
   */
  KOKKOS_INLINE_FUNCTION
  constexpr float tan_theta_half(
    kokkos_rng_state_t& rg,
    float E,
    float nvdt
  ) const
  {
    return 0;
  }

  /**
   * @brief Scattering cross section.
   *
   * The cross-section is used only when dispatched in a Monte-Carlo pipeline.
   * A collision will occur with probability cross_section*nvdt.
   *
   * @param rg Random number generator
   * @param E Collision energy
   * @param nvdt Areal density of particles encountered
   */
  KOKKOS_INLINE_FUNCTION
  constexpr float cross_section(
    kokkos_rng_state_t& rg,
    float E,
    float nvdt
  ) const
  {
    return 0;
  }

  /**
   * @brief Coefficient of restitution for the collision.
   *
   * COR = sqrt(KE_final / KE_initial), elastic collisions have COR = 1 exactly.
   *
   * @param rg Random number generator
   * @param E Collision energy
   * @param nvdt Areal density of particles encountered
   */
  KOKKOS_INLINE_FUNCTION
  constexpr float restitution(
    kokkos_rng_state_t& rg,
    float E,
    float nvdt
  ) const
  {
    return 1;
  }

};

// In collision.cc

void
checkpt_collision_op_internal( const collision_op_t * cop );

void *
restore_collision_op_internal( collision_op_t * params );


#endif /* _collision_h_ */
