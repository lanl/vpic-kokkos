#ifndef _charge_exchange_h_
#define _charge_exchange_h_

#include "particle_bulk.h"
//#include "collision_private.h"

/**
 * @brief Charge exchange collision operator.
 */
template<typename Functor>
struct cex_collision_op_t : public particle_bulk_collision_op_t {
  double dq0;
  Functor sigma_cx0;
};

/**
 * @brief Charge exchange collision model.
 */
template<typename Functor>
struct cex_model : public collision_model<cex_model<Functor>> {
  // const float cvar;
  const float dq;
  Functor sigma_cx;
  //float (*sigma_cx)(float,float);
  //takizuka_abe_model( float cvar ) : cvar(cvar) { };
  cex_model( Functor op, float dq ) : sigma_cx(op), dq(dq) { };
  //cex_model( cex_coll_func_t _sigma_cx0 ) : sigma_cx(_sigma_cx0) { };
  //cex_model( float (*sigma_cx0)(float,float) ) : sigma_cx(sigma_cx0) { };

  
  KOKKOS_INLINE_FUNCTION
  float cross_section(
    kokkos_rng_state_t& rg,
    float Z,     // Charge of particle
    float vr,    // Changed input variable.
    float nvdt
  ) const
  {
     //    float Z = 5;
    float sig = sigma_cx(vr,Z);
    //    float sig = 9999999;
    
    //    printf("Z = %f,vr = %f, sigma = %e, nvdt=%e\n",Z,vr,sig,(sig*nvdt));
    
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
    //float capture = -1;
    float delta_charge = dq;
    
    return delta_charge;
  }
  

  /**
   * @brief Implemention of upload_moment_src_impl() for charge exchange
   *         model accumulations change in density.
   * todo: add change in momentum
   */
  template <class ViewType>
  KOKKOS_INLINE_FUNCTION
  void upload_moment_src_impl( 
    const ViewType & spj_v, 
    const int v,
		const gmomType &Dm, 
    const float mi,
    const float mj) const 
  {
    spj_v(v, fluid_var::den) += -Dm.v[5];
    // spj_v(v, fluid_var::ux) += -Dm.v[1] / (mj * Dm.v[0]);
    // spj_v(v, fluid_var::uy) += -Dm.v[2] / (mj * Dm.v[0]);
    // spj_v(v, fluid_var::uz) += -Dm.v[3] / (mj * Dm.v[0]);
    // spj_v(v, fluid_var::msx) += -Dm.v[1] * mi;
    // spj_v(v, fluid_var::msy) += -Dm.v[2] * mi;
    // spj_v(v, fluid_var::msz) += -Dm.v[3] * mi;
    // spj_v(v, fluid_var::ens) += -Dm.v[4] * mi;
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
  cex_model model(cex->sigma_cx0,cex->dq0);
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
  const double       dq0,
  Functor sigmafunc,
  const int          interval
) {

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g || interval <= 0 )
    ERROR(("Bad args."));

  cex_collision_op_t<Functor> * cex;
  MALLOC( cex, 1);
  MALLOC( cex->name, strlen(name) +1 );
  strncpy( cex->name, name, strlen(name)+1);

  cex->spi         = spi;
  cex->spj         = spj;
  cex->sigma_cx0   = sigmafunc;
  cex->dq0         = dq0;
  //  ta->cvar0       = cvar0 * spi->q * spi->q * spj->q * spj->q;
  cex->interval    = interval;
  cex->apply_cop   = &apply_cex_collision_op<Functor>;
  cex->delete_cop  = &delete_cex_collision_op<Functor>;
  cex->next        = NULL;

  REGISTER_OBJECT(cex,
                  &checkpt_cex_collision_op<Functor>,
                  &restore_cex_collision_op<Functor>,
                  NULL);

  return cex;

}

#endif /* _charge_exchange_h_ */
