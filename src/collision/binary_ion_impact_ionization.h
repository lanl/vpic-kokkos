#ifndef _binary_ion_impact_ioniz_h_
#define _binary_ion_impact_ioniz_h_

#include "binary_neutral.h"

/**
 * @brief Binary ion impact ionization collision operator.
 */
template<typename Functor>
struct binary_ion_impact_ioniz_collision_op_t : public binary_neutral_collision_op_t {
  float dE;
  bool var_wt;
  Functor sigma_cx0;
};


/**
 * @brief Binary ion impact ionization collision model.
 */
template<typename Functor>
struct binary_ion_impact_ioniz_model : public collision_model<binary_ion_impact_ioniz_model<Functor>> {
  CollisionType collision_type = CollisionType::BinaryIonImpactIoniz;
  const float dE;
  const bool var_wt;
  Functor sigma_cx;
  binary_ion_impact_ioniz_model( Functor op, float dE, bool var_wt ) : sigma_cx(op), dE(dE), var_wt(var_wt) { };


  /**
   * @brief cross_section(E,nvdt,Z1,Z2)
   */
  KOKKOS_INLINE_FUNCTION
  float cross_section(
    kokkos_rng_state_t& rg,
    float vr,    // relatvie velocity
    float nvdt,
    float E0,    // relative energy
    float Z1,    // Charge of particle
    float Z2=0  // Charge of particle
  ) const
  {
    return sigma_cx(vr, Z1, Z2);
  }

  /**
   * @brief restitution returns the scale factor of relative speed
   */
  KOKKOS_INLINE_FUNCTION
  float restitution(
    kokkos_rng_state_t& rg,
    float *param
  ) const
  {
    auto E0 = param[0]; // projectile energy
    
    // Physically should not be capable of ionizing if
    // relative energy is below the ionization energy
    // if (E0 < dE) { return 0.001; }

    auto Cr = std::sqrt((E0 - dE) / E0); // scale factor for change in velocity
    // std::cout << "E0="<<E0 << " dE=" << dE << std::endl;
    // std::cout << "Cr = " << Cr << "Cr2 = " << (E0 - dE) / E0 << " dE/E0 = " << dE/E0 << std::endl;
    return Cr;
  }

  /**
   * @brief tan(theta/2) not implemented yet so no scattering
   */
  KOKKOS_INLINE_FUNCTION
  float tan_theta_half(
    kokkos_rng_state_t& rg,
    float E,
    float nvdt
  ) const
  {
    float value = 0.0;
    return value;
  }

  /**
   * @brief modify_charge(). Don't use. Always increment charge 
   * of second species by one.
   */
  // KOKKOS_INLINE_FUNCTION
  // float modify_charge( ) const
  // {
  //   float delta_charge = dq; 
  //   return delta_charge;
  // }
};


/* Private interface *********************************************************/

template<typename Functor>
void
checkpt_binary_ion_impact_ioniz_collision_op(const void * cop) {
  binary_ion_impact_ioniz_collision_op_t<Functor> * ioniz = (binary_ion_impact_ioniz_collision_op_t<Functor> *) cop;
  CHECKPT(ioniz, 1);
  checkpt_binary_neutral_collision_op_internal( (const binary_neutral_collision_op_t *) cop );
}

template<typename Functor>
void *
restore_binary_ion_impact_ioniz_collision_op() {
  binary_ion_impact_ioniz_collision_op_t<Functor> * ioniz;
  RESTORE(ioniz);
  return restore_binary_neutral_collision_op_internal( (binary_neutral_collision_op_t *) ioniz );
}

template<typename Functor>
void
apply_binary_ion_impact_ioniz_collision_op( collision_op_t * cop,
                                            kokkos_rng_pool_t& rng ) {
  binary_ion_impact_ioniz_collision_op_t<Functor> * ioniz = (binary_ion_impact_ioniz_collision_op_t<Functor> *) cop;
  binary_ion_impact_ioniz_model model(ioniz->sigma_cx0, ioniz->dE, ioniz->var_wt);
  if(ioniz->var_wt)
    apply_binary_neutral_collision_model_pipeline<true>((binary_neutral_collision_op_t *) cop, model, rng);
  else
    apply_binary_neutral_collision_model_pipeline<false>((binary_neutral_collision_op_t *) cop, model, rng);
}

template<typename Functor>
void
delete_binary_ion_impact_ioniz_collision_op(collision_op_t * cop) {
  binary_ion_impact_ioniz_collision_op_t<Functor> * ioniz = (binary_ion_impact_ioniz_collision_op_t<Functor> *) cop;
  UNREGISTER_OBJECT(ioniz);
  FREE(ioniz);
}

/* Public interface **********************************************************/

template<typename Functor>
collision_op_t *
binary_ion_impact_ioniz(
  const char       * name,
  /**/  species_t  * spi,
  /**/  species_t  * spj,
  const float        dE,
  Functor            sigma_func,
  const int          interval,
  const bool         var_wt
  // species_t        * spp1=NULL,
  // species_t        * spp2=NULL
)
{

  if ( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g || interval <= 0 ) {
    ERROR(("Bad args."));
  }

  binary_ion_impact_ioniz_collision_op_t<Functor> * ioniz;
  MALLOC( ioniz, 1);
  MALLOC( ioniz->name, strlen(name) +1 );
  strncpy( ioniz->name, name, strlen(name)+1);

  spi->last_indexed = -1; //to ensure sort in collisions
  spj->last_indexed = -1;
  
  ioniz->spi         = spi;
  ioniz->spj         = spj;
  // ioniz->spp1        = spp1;
  // ioniz->spp2        = spp2;
  ioniz->sigma_cx0   = sigma_func;
  ioniz->dE          = dE;
  ioniz->var_wt      = var_wt;
  ioniz->interval    = interval;
  ioniz->apply_cop   = &apply_binary_ion_impact_ioniz_collision_op<Functor>;
  ioniz->delete_cop  = &delete_binary_ion_impact_ioniz_collision_op<Functor>;
  ioniz->next        = NULL;

  REGISTER_OBJECT(ioniz,
                  &checkpt_binary_ion_impact_ioniz_collision_op<Functor>,
                  &restore_binary_ion_impact_ioniz_collision_op<Functor>,
                  NULL);

  return ioniz;
}

#endif  /* _binary_ion_impact_ioniz_h_ */