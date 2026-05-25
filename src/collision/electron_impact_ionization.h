#ifndef _electron_impact_ionization_h_
#define _electron_impact_ionization_h_

#include "particle_bulk.h"

/**
 * @brief Electron impact ionization collision operator.
 */
template<typename Functor>
struct electron_ioniz_collision_op_t : public particle_bulk_collision_op_t {
  Functor sigma_cx0;
  double dE;
};

/**
 * @brief Electron impact ionization collision model.
 */
template<typename Functor>
struct electron_ioniz_model : public collision_model<electron_ioniz_model<Functor>> {
  CollisionType collision_type = CollisionType::BulkElectronImpactIoniz;
  Functor sigma_cx;
  double dE;

  electron_ioniz_model( Functor op, double dE) : 
    sigma_cx(op), dE{dE} {};

  
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
    // Add extra catch to ensure ionization does not occur
    // if the relative energy is below the ionization energy.
    // (Note the cross section should be zero below the 
    // the ionization energy, but lets be safe)
    if (dE > E0) {
      return 0.0;
    }

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


  // Incoming particle does not change charge
  // KOKKOS_INLINE_FUNCTION
  //   float modify_charge( ) const
  // {
  //   float delta_charge = dq; 
  //   return delta_charge;
  // }
  

  /**
   * @brief Implemention of upload_moment_src_impl() for electron impact ionization
   *        model accumulations change in momentum and energy.
   * todo: confirm moment transfer, currently doing same as lemons model
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
    // spj_v(v, field_var::sx) += -mj * Dm.v[1];
    // spj_v(v, field_var::sy) += -mj * Dm.v[2];
    // spj_v(v, field_var::sz) += -mj * Dm.v[3];
    // spj_v(v, field_var::se) += -mj * Dm.v[4];
  } // end upload_moment_src_impl()
};

/* Private interface *********************************************************/

template<typename Functor>
void
checkpt_electron_ioniz_collision_op(const void * cop) {
  electron_ioniz_collision_op_t<Functor> * electron_ioniz = (electron_ioniz_collision_op_t<Functor> *) cop;
  CHECKPT(electron_ioniz, 1);
  checkpt_particle_bulk_collision_op_internal( (const particle_bulk_collision_op_t *) cop );
}

template<typename Functor>
void *
restore_electron_ioniz_collision_op() {
  electron_ioniz_collision_op_t<Functor> * electron_ioniz;
  RESTORE(electron_ioniz);
  return restore_particle_bulk_collision_op_internal( (particle_bulk_collision_op_t *) electron_ioniz );
}

template<typename Functor>
void
apply_electron_ioniz_collision_op( collision_op_t * cop, kokkos_rng_pool_t& rng ) {
  electron_ioniz_collision_op_t<Functor> * electron_ioniz = (electron_ioniz_collision_op_t<Functor> *) cop;
  electron_ioniz_model model(electron_ioniz->sigma_cx0, electron_ioniz->dE);
  apply_particle_bulk_collision_model_pipeline<true>((particle_bulk_collision_op_t *) cop, model, rng);
}

template<typename Functor>
void
delete_electron_ioniz_collision_op(collision_op_t * cop) {
  electron_ioniz_collision_op_t<Functor> * electron_ioniz = (electron_ioniz_collision_op_t<Functor> *) cop;
  UNREGISTER_OBJECT(electron_ioniz);
  FREE(electron_ioniz);
}

/* Public interface **********************************************************/

template<typename Functor>
collision_op_t *
electron_impact_ionization(
  const char       * name,
  /**/  species_t  * spi,
  /**/  fluid_species_t  * spj,
  const double       dE,
  Functor            sigmafunc,
  const int          interval,
  field_array_t    * field
) {

  if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g || interval <= 0 )
    ERROR(("Bad args."));

  electron_ioniz_collision_op_t<Functor> * electron_ioniz;
  MALLOC( electron_ioniz, 1);
  MALLOC( electron_ioniz->name, strlen(name) +1 );
  strncpy( electron_ioniz->name, name, strlen(name)+1);

  spi->last_indexed = -1; //to ensure sort in collisions

  if(field != NULL) {
    electron_ioniz->field = field;
  } else {
    electron_ioniz->field = NULL;
  }
  
  electron_ioniz->spi         = spi;
  electron_ioniz->spj         = spj;
  electron_ioniz->sigma_cx0   = sigmafunc;
  electron_ioniz->dE          = dE;
  electron_ioniz->interval    = interval;
  electron_ioniz->apply_cop   = &apply_electron_ioniz_collision_op<Functor>;
  electron_ioniz->delete_cop  = &delete_electron_ioniz_collision_op<Functor>;
  electron_ioniz->next        = NULL;

  REGISTER_OBJECT(electron_ioniz,
                  &checkpt_electron_ioniz_collision_op<Functor>,
                  &restore_electron_ioniz_collision_op<Functor>,
                  NULL);

  return electron_ioniz;

}

#endif /* _electron_impact_ionization_h_ */
