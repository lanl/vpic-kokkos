#ifndef _collision_h_
#define _collision_h_

#include "../util/rng_policy.h"
#include "../species_advance/species_advance.h"

struct collision_op;
typedef struct collision_op collision_op_t;

/* In collision.cc */

int
num_collision_op( const collision_op_t * cop_list );

void
apply_collision_op_list( collision_op_t    * cop_list,
                         kokkos_rng_pool_t & rng_pool );

void
delete_collision_op_list( collision_op_t * cop_list );

collision_op_t *
append_collision_op( collision_op_t * cop,
                     collision_op_t ** cop_list );

/* In takizuka_abe.cc */

/* The Takizuka-Abe collision model is based on Takizuka and Abe, JCP 1977
   and efficiently models small-angle Coulomb scattering by randomly pairing
   particles. On average, each particle is scattered once per call. The model
   is fully defined by a single parameter, the base collision frequency nu0.
   In SI units, nu0 is defined by

   nu0 = log(Lambda) / 8 pi sqrt(2) c^3 eps0^2

   where log(Lambda) is the Coulomb logarithm. For a thermal species with
   temperature T, normalized mass m and charge q, the self-scattering momentum
   transfer rate (i.e., "the" collision rate) is related to nu0 via

   nu_s = 4 (mc^2 / T)^3 nu0 / 3 sqrt(pi)

   The paper defines variance ~= sqrt(2)*nu_0, and here we expect the user to
   pass us the base variance, cvar0 = sqrt(2)*nu_0.
*/

collision_op_t *
takizuka_abe(
  const char       * name,
  /**/  species_t  * spi,
  /**/  species_t  * spj,
  const double       cvar0,
  const int          interval
);

/* In hard_sphere.cc */

/* Binary hard sphere collisions between particles with equal radius. */

// TODO: CPU-VPIC allowed unequal radii. Fix this.
// TODO: CPU-VPIC allowed sample != 1, here we hardcode to sample = 1.

collision_op_t *
hard_sphere(
  const char       * name,
  /**/  species_t  * spi,
  /**/  species_t  * spj,
  const double       radius,
  const int          interval
);

#endif /* _collision_h_ */
