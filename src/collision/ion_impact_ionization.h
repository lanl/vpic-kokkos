#ifndef _ion_impact_ionization_h_
#define _ion_impact_ionization_h_

#include "particle_bulk.h"

/**
 * @brief Ion impact ionization collision operator.
 */
template<typename Functor>
struct ion_ioniz_collision_op_t : public particle_bulk_collision_op_t {
  Functor sigma_cx0;
  double dE;
};

/**
 * @brief Ion impact ionization collision model.
 */
template<typename Functor>
struct ion_ioniz_model : public collision_model<ion_ioniz_model<Functor>> {
  CollisionType collision_type = CollisionType::BulkIonImpactIoniz;
  Functor sigma_cx;
  double dE;

  ion_ioniz_model( Functor op, double dE) : 
    sigma_cx(op), dE{dE} {};

  
  KOKKOS_INLINE_FUNCTION
  float cross_section(
    kokkos_rng_state_t& rg,
    float vr,    // Changed input variable.
    float nvdt,
    float Z1,     // Charge of particle
    float Z2=0.0  // Charge of particle
  ) const
  {
    float sig = sigma_cx(vr, Z1);
    return sig;
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
    // Removing energy only from particle assumes fluid is at rest
    auto E0 = param[4]; // projectile energy
    auto Cr = std::sqrt((E0 - dE) / E0); // scale factor for change in velocity
    // std::cout << "Cr = " << Cr << "Cr2 = " << (E0 - dE_i) / E0 << " dE/E0 = " << dE_i/E0 << std::endl;
    return Cr;
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
    return value; // No scattering for now.
  }


  // Incoming ion does not change charge
  // KOKKOS_INLINE_FUNCTION
  //   float modify_charge( ) const
  // {
  //   float delta_charge = dq; 
  //   return delta_charge;
  // }
  

  /**
   * @brief Implemention of upload_moment_src_impl() for ion impact ionization
   *        model accumulations change in density.
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
    spj_v(v, fluid_var::ux)  += -Dm.v[1] * mi / mj; // du_2 = dp_1 / m_2
    spj_v(v, fluid_var::uy)  += -Dm.v[2] * mi / mj;
    spj_v(v, fluid_var::uz)  += -Dm.v[3] * mi / mj;
    spj_v(v, fluid_var::tmp) += -Dm.v[4] * mi / mj * 1.0 / 3.0; // dT ~ 2/3 dE
    spj_v(v, fluid_var::den) += -Dm.v[5];
  } // end upload_moment_src_impl()
};

/* Private interface *********************************************************/

template<typename Functor>
void
checkpt_ion_ioniz_collision_op(const void * cop) {
  ion_ioniz_collision_op_t<Functor> * ion_ioniz = (ion_ioniz_collision_op_t<Functor> *) cop;
  CHECKPT(ion_ioniz, 1);
  checkpt_particle_bulk_collision_op_internal( (const particle_bulk_collision_op_t *) cop );
}

template<typename Functor>
void *
restore_ion_ioniz_collision_op() {
  ion_ioniz_collision_op_t<Functor> * ion_ioniz;
  RESTORE(ion_ioniz);
  return restore_particle_bulk_collision_op_internal( (particle_bulk_collision_op_t *) ion_ioniz );
}

template<typename Functor>
void
apply_ion_ioniz_collision_op( collision_op_t * cop,
			kokkos_rng_pool_t& rng ) {
  ion_ioniz_collision_op_t<Functor> * ion_ioniz = (ion_ioniz_collision_op_t<Functor> *) cop;
  ion_ioniz_model model(ion_ioniz->sigma_cx0, ion_ioniz->dE);
  apply_particle_bulk_collision_model_pipeline<true>((particle_bulk_collision_op_t *) cop, model, rng); // To-do: Change MC to true!
}

template<typename Functor>
void
delete_ion_ioniz_collision_op(collision_op_t * cop) {
  ion_ioniz_collision_op_t<Functor> * ion_ioniz = (ion_ioniz_collision_op_t<Functor> *) cop;
  UNREGISTER_OBJECT(ion_ioniz);
  FREE(ion_ioniz);
}

/* Public interface **********************************************************/

template<typename Functor>
collision_op_t *
ion_impact_ionization(
  const char       * name,
  /**/  species_t  * spi,
  /**/  fluid_species_t  * spj,
  const double       dE,
  Functor            sigmafunc,
  const int          interval,
  species_t        * spp=NULL
) {

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g || interval <= 0 )
    ERROR(("Bad args."));

  ion_ioniz_collision_op_t<Functor> * ion_ioniz;
  MALLOC( ion_ioniz, 1);
  MALLOC( ion_ioniz->name, strlen(name) +1 );
  strncpy( ion_ioniz->name, name, strlen(name)+1);

  ion_ioniz->spi         = spi;
  ion_ioniz->spj         = spj;
  ion_ioniz->spp         = spp;
  ion_ioniz->sigma_cx0   = sigmafunc;
  ion_ioniz->dE          = dE;
  ion_ioniz->interval    = interval;
  ion_ioniz->apply_cop   = &apply_ion_ioniz_collision_op<Functor>;
  ion_ioniz->delete_cop  = &delete_ion_ioniz_collision_op<Functor>;
  ion_ioniz->next        = NULL;
  ion_ioniz->field       = NULL;

  REGISTER_OBJECT(ion_ioniz,
                  &checkpt_ion_ioniz_collision_op<Functor>,
                  &restore_ion_ioniz_collision_op<Functor>,
                  NULL);

  return ion_ioniz;

}

#endif /* _ion_impact_ionization_h_ */
