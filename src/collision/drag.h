#ifndef _drag_h_
#define _drag_h_

#include "particle_bulk.h"

/**
 * @brief Neutral drag collision operator.
 */
struct drag_collision_op_t : public particle_bulk_collision_op_t {
  //  double cvar0;
  float (*stopping_cx0)(float);
};


/**
 * @brief Drag collision model.
 */
struct drag_model : public collision_model<drag_model> {
  // const float cvar;

  float (*stopping_cx)(float);

  drag_model( float (*stopping_cx0)(float) ) : stopping_cx(stopping_cx0) { };

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

      if (Crterm2 > 1.0e-1) {
	//	printf("v0=%14.8e, ndt_mi2=%14.8e, mS=%14.8e, Crterm2=%14.8e, Cr=%14.8e",v0,ndt_mi2,mS,Crterm2, Cr);
      }
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

#endif /* _drag_h_ */
