#ifndef _drag_h_
#define _drag_h_

#include "particle_bulk.h"

/**
 * @brief Neutral drag collision operator.
 */
template<typename Functor>
struct drag_collision_op_t : public particle_bulk_collision_op_t {
  //  double cvar0;
  Functor stopping_cx0;
};


/**
 * @brief Drag collision model.
 */
template<typename Functor>
struct drag_model : public collision_model<drag_model<Functor>> {
  // const float cvar;

  Functor stopping_cx;

  drag_model( Functor op ) : stopping_cx(op) { };

  /* 
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
  */

  /**
   * @brief restitution returns the scale factor of relative speed
   */
  KOKKOS_INLINE_FUNCTION
  float restitution(
    kokkos_rng_state_t& rg,
    float *param
  ) const
  {
      auto v0 = param[0];
      //assert(v0>0);
      if(v0==0) return 0;

      auto ndt_mi2 = param[2]; // Actually need n*dt/mi -> multiply by mi in stopping_cx.
      
      float mS = stopping_cx(v0); 

      auto Cr = 1.0 - ndt_mi2*mS/v0;
      auto Crterm2 = ndt_mi2*mS/v0;

      //if (Crterm2 > 1.0e-1) {
	//	printf("v0=%14.8e, ndt_mi2=%14.8e, mS=%14.8e, Crterm2=%14.8e, Cr=%14.8e",v0
	//	,ndt_mi2,mS,Crterm2, Cr);
      //      }
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
    return value; // No scattering for now. TO-DO: Add scattering for elastic collisions
  }

  
};


/* Private interface *********************************************************/

template<typename Functor>
void
checkpt_drag_collision_op(const void * cop) {
  drag_collision_op_t<Functor> * drag = (drag_collision_op_t<Functor> *) cop;
  CHECKPT(drag, 1);
  checkpt_particle_bulk_collision_op_internal( (const particle_bulk_collision_op_t *) cop );
}

template<typename Functor>
void *
restore_drag_collision_op() {
  drag_collision_op_t<Functor> * drag;
  RESTORE(drag);
  return restore_particle_bulk_collision_op_internal( (particle_bulk_collision_op_t *) drag );
}

template<typename Functor>
void
apply_drag_collision_op( collision_op_t * cop,
                        kokkos_rng_pool_t& rng ) {
  drag_collision_op_t<Functor> * drag = (drag_collision_op_t<Functor> *) cop;
  drag_model model(drag->stopping_cx0);
  apply_particle_bulk_collision_model_pipeline<false>((particle_bulk_collision_op_t *) cop, model, rng); // To-do: MC false for drag only (for now)
}

template<typename Functor>
void
delete_drag_collision_op(collision_op_t * cop) {
  drag_collision_op_t<Functor> * drag = (drag_collision_op_t<Functor> *) cop;
  UNREGISTER_OBJECT(drag);
  FREE(drag);
}


/* Public interface **********************************************************/

template<typename Functor>
collision_op_t *
drag(
  const char       * name,
  /**/  species_t  * spi,
  /**/  fluid_species_t  * spj,
  //  const double       cvar0,                                                                                                          
  Functor stoppingfunc,
  const int          interval
) {

   if( !name || !spi || !spj || !spi->g || !spj->g || spi->g != spj->g || interval <= 0 )
    ERROR(("Bad args."));

  drag_collision_op_t<Functor> * drag;
  MALLOC( drag, 1);
  MALLOC( drag->name, strlen(name) +1 );
  strncpy( drag->name, name, strlen(name)+1);

  drag->spi         = spi;
  drag->spj         = spj;
  drag->stopping_cx0   = stoppingfunc;
  //  ta->cvar0       = cvar0 * spi->q * spi->q * spj->q * spj->q;
  drag->interval    = interval;
  drag->apply_cop   = &apply_drag_collision_op<Functor>;
  drag->delete_cop  = &delete_drag_collision_op<Functor>;
  drag->next        = NULL;
  drag->field       = NULL;

  REGISTER_OBJECT(drag,
                  &checkpt_drag_collision_op<Functor>,
                  &restore_drag_collision_op<Functor>,
                  NULL);

  return drag;

}


#endif /* _drag_h_ */
