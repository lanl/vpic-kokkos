#ifndef _charge_exchange_h_
#define _charge_exchange_h_

#include "particle_bulk.h"

/**
 * @brief Charge exchange collision operator.
 */
template<typename Functor>
struct cex_collision_op_t : public particle_bulk_collision_op_t {
  float   dq0;
  Functor sigma_cx0;
  float   production_multiplier;
};

/**
 * @brief Charge exchange collision model.
 */
template<typename Functor>
struct cex_model : public collision_model<cex_model<Functor>> {
  CollisionType collision_type = CollisionType::BulkChargeExchange;
  units_SI units;
  const float dq;
  Functor sigma_cx;
  float production_multiplier;
  
  cex_model( units_SI units, Functor op, float dq, float production_multiplier ) : 
    units(units), 
    sigma_cx(op), 
    dq(dq), 
    production_multiplier{production_multiplier} 
    {};
  
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
    float sig = sigma_cx(units, vr, Z1);
    return sig;
  }
  
    
  /**
   * @brief tan(theta/2)
   */
  KOKKOS_INLINE_FUNCTION
  float tan_theta_half(
    kokkos_rng_state_t& rg,
    float * param
  ) const
  {
    float value = 0;
    return value; // No scattering for now. TO-DO: Add scattering for CEX.
  }


  KOKKOS_INLINE_FUNCTION
    float modify_charge( ) const
  {
    float delta_charge = dq;
    return delta_charge;
  }
  
  /**
   * @brief Function for modifying the likelihood of a reaction such that 
   *        more macroparticles are created (with smaller weights) and
   *        the physically correct reaction rate is maintained
   * @param P probability to scatter
   * @param pr_mult production multiplier
   */
  KOKKOS_INLINE_FUNCTION
  constexpr void modify_reaction_probability( float& P, float& pr_mult ) const
  {
    // Can tune likelihood of reaction using production_multiplier.
    // But the method requires P<1 to resolve the reaction, so here we
    // reduce the multiplier until the condition is satisfied
    pr_mult = production_multiplier;
    float p_scatter = P * pr_mult;
    while (p_scatter > 1.0) {
      pr_mult /= 2.0;
      p_scatter = P * pr_mult;
    }
    P *= pr_mult;
    return;
  } // end modify_reaction_probability()

  /**
   * @brief Implemention of upload_moment_src_impl() for charge exchange
   *        model accumulations change in density.
   * todo: add change in momentum (depends on new kinetic particle)
   */
  template <class ViewType>
  KOKKOS_INLINE_FUNCTION
  void upload_moment_src_impl( 
    const ViewType & spj_v, 
    const int v,
    const gmomType &Dm, 
    const float mi,
    const float mj,
    const float mj_ttl) const 
  {
    spj_v(v, fluid_var::ux)  += -Dm.v[1] * mi / mj_ttl; // du_2 = dp_1 / m_2
    spj_v(v, fluid_var::uy)  += -Dm.v[2] * mi / mj_ttl;
    spj_v(v, fluid_var::uz)  += -Dm.v[3] * mi / mj_ttl;

    // dT = 2/3 * dE_ave = 2/3 * dE_ttl / N,  where N = m_fluid_ttl / m_fluid_particle
    spj_v(v, fluid_var::tmp) += -Dm.v[4] * mi / (mj_ttl / mj) * 2.0 / 3.0;
    spj_v(v, fluid_var::tmp) = (spj_v(v, fluid_var::tmp) > 0.0) ? spj_v(v, fluid_var::tmp) : 0.0;

    // drho = dn * m_fluid_particle
    spj_v(v, fluid_var::den) += -Dm.v[5] * mj; 
    spj_v(v, fluid_var::den) = (spj_v(v, fluid_var::den) > 0.0) ? spj_v(v, fluid_var::den) : 0.0;
  } // end upload_moment_src_impl()
};

/* Private interface *********************************************************/

template<typename Functor>
void
checkpt_cex_collision_op(const void * cop) {
  cex_collision_op_t<Functor> * cex = (cex_collision_op_t<Functor> *) cop;
  CHECKPT(cex, 1);
  checkpt_particle_bulk_collision_op_internal( (const particle_bulk_collision_op_t *) cop );
}

template<typename Functor>
void *
restore_cex_collision_op() {
  cex_collision_op_t<Functor> * cex;
  RESTORE(cex);
  return restore_particle_bulk_collision_op_internal( (particle_bulk_collision_op_t *) cex );
}

template<typename Functor>
void
apply_cex_collision_op( collision_op_t * cop,
                        kokkos_rng_pool_t& rng ) {
  cex_collision_op_t<Functor> * cex = (cex_collision_op_t<Functor> *) cop;
  cex_model model(cex->spi->g->units, cex->sigma_cx0, cex->dq0, cex->production_multiplier);
  apply_particle_bulk_collision_model_pipeline<true>((particle_bulk_collision_op_t *) cop, model, rng); // To-do: Change MC to true!
}

template<typename Functor>
void
delete_cex_collision_op(collision_op_t * cop) {
  cex_collision_op_t<Functor> * cex = (cex_collision_op_t<Functor> *) cop;
  UNREGISTER_OBJECT(cex);
  FREE(cex);
}

/* Public interface **********************************************************/

template<typename Functor>
collision_op_t *
charge_exchange(
  const char       * name,
  /**/  species_t  * spi,
  /**/  fluid_species_t  * spj,
  const float        dq0,
  Functor            sigmafunc,
  const int          interval,
  float              production_multiplier=1.0,
  species_t        * spp=NULL
) {

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g || interval <= 0 )
    ERROR(("Bad args."));

  if( production_multiplier > 1.0 ) {
    WARNING(("Bad arg. production_multiplier valid for <=1. Reassigning to 1."));
    production_multiplier = 1.0;
  }

  cex_collision_op_t<Functor> * cex;
  MALLOC( cex, 1);
  MALLOC( cex->name, strlen(name) +1 );
  strncpy( cex->name, name, strlen(name)+1);

  cex->spi         = spi;
  cex->spj         = spj;
  cex->spp         = spp;
  cex->sigma_cx0   = sigmafunc;
  cex->dq0         = dq0;
  cex->production_multiplier = production_multiplier;
  cex->interval    = interval;
  cex->apply_cop   = &apply_cex_collision_op<Functor>;
  cex->delete_cop  = &delete_cex_collision_op<Functor>;
  cex->next        = NULL;
  cex->field       = NULL;

  REGISTER_OBJECT(cex,
                  &checkpt_cex_collision_op<Functor>,
                  &restore_cex_collision_op<Functor>,
                  NULL);

  return cex;

}

#endif /* _charge_exchange_h_ */
