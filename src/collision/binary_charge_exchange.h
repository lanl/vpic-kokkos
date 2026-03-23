#ifndef _binary_charge_exchange_h_
#define _binary_charge_exchange_h_

#include "binary_neutral.h"

/**
 * @brief Binary charge exchange collision operator.
 */
template<typename Functor>
struct binary_charge_exchange_collision_op_t : public binary_neutral_collision_op_t {
  int dq;
  bool var_wt;
  Functor sigma_cx0;
};


/**
 * @brief Binary charge exchange collision model.
 */
template<typename Functor>
struct binary_charge_exchange_model : public collision_model<binary_charge_exchange_model<Functor>> {
  CollisionType collision_type = CollisionType::BinaryChargeExchange;
  const int dq;
  const bool var_wt;
  Functor sigma_cx;
  binary_charge_exchange_model( Functor op, int dq, bool var_wt ) : sigma_cx(op), dq(dq), var_wt(var_wt) { };


  /**
   * @brief cross_section(E,nvdt,Z1,Z2)
   */
  KOKKOS_INLINE_FUNCTION
  float cross_section(
    kokkos_rng_state_t& rg,
    float vr,     // Changed input variable.
    float nvdt,
    float Z1,      // Charge of first particle
    float Z2=0.0   // Charge of second particle
  ) const
  {
    return sigma_cx(vr, Z1, Z2);
  }

  KOKKOS_INLINE_FUNCTION
  float modify_charge( ) const
  {
    float delta_charge = dq; 
    return delta_charge;
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

};


/* Private interface *********************************************************/

template<typename Functor>
void
checkpt_binary_charge_exchange_collision_op(const void * cop) {
  binary_charge_exchange_collision_op_t<Functor> * cex = (binary_charge_exchange_collision_op_t<Functor> *) cop;
  CHECKPT(cex, 1);
  checkpt_binary_neutral_collision_op_internal( (const binary_neutral_collision_op_t *) cop );
}

template<typename Functor>
void *
restore_binary_charge_exchange_collision_op() {
  binary_charge_exchange_collision_op_t<Functor> * cex;
  RESTORE(cex);
  return restore_binary_neutral_collision_op_internal( (binary_neutral_collision_op_t *) cex );
}

template<typename Functor>
void
apply_binary_charge_exchange_collision_op( collision_op_t * cop,
                                           kokkos_rng_pool_t& rng ) {
  binary_charge_exchange_collision_op_t<Functor> * cex = (binary_charge_exchange_collision_op_t<Functor> *) cop;
  binary_charge_exchange_model model(cex->sigma_cx0, cex->dq, cex->var_wt);
  // if(cex->var_wt)
  apply_binary_neutral_collision_model_pipeline<true>((binary_neutral_collision_op_t *) cop, model, rng);
  //  else
  //    apply_binary_neutral_collision_model_pipeline<false>((binary_neutral_collision_op_t *) cop, model, rng);
  }

template<typename Functor>
void
delete_binary_charge_exchange_collision_op(collision_op_t * cop) {
  binary_charge_exchange_collision_op_t<Functor> * cex = (binary_charge_exchange_collision_op_t<Functor> *) cop;
  UNREGISTER_OBJECT(cex);
  FREE(cex);
}

/* Public interface **********************************************************/

template<typename Functor>
collision_op_t *
binary_charge_exchange(
  const char       * name,
  /**/  species_t  * spi,
  /**/  species_t  * spj,
  const int          dq,
  Functor            sigma_func,
  const int          interval,
  const bool         var_wt
  // species_t        * spp1=NULL,
  // species_t        * spp2=NULL
)
{

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g ||
      abs(dq) != 1 || interval <= 0 )
    ERROR(("Bad args."));

  binary_charge_exchange_collision_op_t<Functor> * cex;
  MALLOC( cex, 1);
  MALLOC( cex->name, strlen(name) +1 );
  strncpy( cex->name, name, strlen(name)+1);

  spi->last_indexed = -1; //to ensure sort in collisions
  spj->last_indexed = -1;
  
  cex->spi         = spi;
  cex->spj         = spj;
  // cex->spp1        = spp1;
  // cex->spp2        = spp2;
  cex->sigma_cx0   = sigma_func;
  cex->dq          = dq;
  cex->var_wt      = var_wt;
  cex->interval    = interval;
  cex->apply_cop   = &apply_binary_charge_exchange_collision_op<Functor>;
  cex->delete_cop  = &delete_binary_charge_exchange_collision_op<Functor>;
  cex->next        = NULL;

  REGISTER_OBJECT(cex,
                  &checkpt_binary_charge_exchange_collision_op<Functor>,
                  &restore_binary_charge_exchange_collision_op<Functor>,
                  NULL);

  return cex;

}



#endif  /* _binary_charge_exchange_h_ */