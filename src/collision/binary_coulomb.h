#ifndef _binary_coulomb_h_
#define _binary_coulomb_h_

#include "binary_neutral.h"

/**
 * @brief Binary charge exchange collision operator.
 */
template<typename Functor>
struct binary_coulomb_collision_op_t : public binary_neutral_collision_op_t {
  double cvar0;
  bool var_wt;
  Functor sigma_cx0;
};


/**
 * @brief Binary charge exchange collision model.
 */
template<typename Functor>
struct binary_coulomb_model : public collision_model<binary_coulomb_model<Functor>> {
  CollisionType collision_type = CollisionType::BinaryCoulomb;
  const double cvar0;
  const bool var_wt;
  Functor sigma_cx;
  binary_coulomb_model( Functor op, double cvar0, bool var_wt ) : cvar0(cvar0), var_wt(var_wt), sigma_cx(op) { };


  /**
   * @brief cross_section(E,nvdt,Z1,Z2)
   */
  KOKKOS_INLINE_FUNCTION
  float cross_section(
    kokkos_rng_state_t& rg,
    float vr,      // relative velocity
    float nvdt,
    float E0,      // relative energy
    float Z1,      // Charge of first particle
    float Z2=0.0   // Charge of second particle
  ) const
  {
    return sigma_cx(vr, Z1, Z2);
  }

  // KOKKOS_INLINE_FUNCTION
  // float modify_charge( ) const
  // {
  //   float delta_charge = dq; 
  //   return delta_charge;
  // }


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
    float sigma = sqrtf(cvar0*nvdt/(E*E));
    sigma = sigma > 1 ? 1 : sigma;
    return rg.normal(0, sigma);
  }

};


/* Private interface *********************************************************/

template<typename Functor>
void
checkpt_binary_coulomb_collision_op(const void * cop) {
  binary_coulomb_collision_op_t<Functor> * coul = (binary_coulomb_collision_op_t<Functor> *) cop;
  CHECKPT(coul, 1);
  checkpt_binary_neutral_collision_op_internal( (const binary_neutral_collision_op_t *) cop );
}

template<typename Functor>
void *
restore_binary_coulomb_collision_op() {
  binary_coulomb_collision_op_t<Functor> * coul;
  RESTORE(coul);
  return restore_binary_neutral_collision_op_internal( (binary_neutral_collision_op_t *) coul );
}

template<typename Functor>
void
apply_binary_coulomb_collision_op( collision_op_t * cop,
                                           kokkos_rng_pool_t& rng ) {
  binary_coulomb_collision_op_t<Functor> * coul = (binary_coulomb_collision_op_t<Functor> *) cop;
  binary_coulomb_model model(coul->sigma_cx0, coul->cvar0, coul->var_wt);
  if(coul->var_wt)
    apply_binary_neutral_collision_model_pipeline<true>((binary_neutral_collision_op_t *) cop, model, rng);
  else
    apply_binary_neutral_collision_model_pipeline<false>((binary_neutral_collision_op_t *) cop, model, rng);
}

template<typename Functor>
void
delete_binary_coulomb_collision_op(collision_op_t * cop) {
  binary_coulomb_collision_op_t<Functor> * coul = (binary_coulomb_collision_op_t<Functor> *) cop;
  UNREGISTER_OBJECT(coul);
  FREE(coul);
}

/* Public interface **********************************************************/

template<typename Functor>
collision_op_t *
binary_coulomb(
  const char       * name,
  /**/  species_t  * spi,
  /**/  species_t  * spj,
  const double          cvar0,
  Functor            sigma_func,
  const int          interval,
  const bool         var_wt
  // species_t        * spp1=NULL,
  // species_t        * spp2=NULL
)
{

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g ||
      cvar0 <= 0 || interval <= 0 )
    ERROR(("Bad args."));

  binary_coulomb_collision_op_t<Functor> * coul;
  MALLOC( coul, 1);
  MALLOC( coul->name, strlen(name) +1 );
  strncpy( coul->name, name, strlen(name)+1);

  spi->last_indexed = -1; //to ensure sort in collisions
  spj->last_indexed = -1;
  
  coul->spi         = spi;
  coul->spj         = spj;
  // coul->spp1        = spp1;
  // coul->spp2        = spp2;
  coul->sigma_cx0   = sigma_func;
#ifdef VARIABLE_CHARGE  
  coul->cvar0       = cvar0; //charges are to be multiplied by particles
#else
  coul->cvar0       = cvar0 * spi->q * spi->q * spj->q * spj->q;
#endif  
  coul->var_wt      = var_wt;
  coul->interval    = interval;
  coul->apply_cop   = &apply_binary_coulomb_collision_op<Functor>;
  coul->delete_cop  = &delete_binary_coulomb_collision_op<Functor>;
  coul->next        = NULL;

  REGISTER_OBJECT(coul,
                  &checkpt_binary_coulomb_collision_op<Functor>,
                  &restore_binary_coulomb_collision_op<Functor>,
                  NULL);

  return coul;
}

#endif  /* _binary_coulomb_h_ */
