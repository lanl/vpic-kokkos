#ifndef _charge_exchange_h_
#define _charge_exchange_h_

#include "particle_bulk.h"

/**
 * @brief Charge exchange collision operator.
 */
struct cex_collision_op_t : public particle_bulk_collision_op_t {
  //  double cvar0;
  float (*sigma_cx0)(float,float);
};


/**
 * @brief Charge exchange collision model.
 */
struct cex_model : public collision_model {
  // const float cvar;

  float (*sigma_cx)(float,float);
  //takizuka_abe_model( float cvar ) : cvar(cvar) { };
  cex_model( float (*sigma_cx0)(float,float) ) : sigma_cx(sigma_cx0) { };

  
  KOKKOS_INLINE_FUNCTION
  float cross_section(
    kokkos_rng_state_t& rg,
    float vr,    // Changed input variable.
    float nvdt
  ) const
  {
    float Z = 5;
    float sig = sigma_cx(vr,Z);

    printf("sigma = %f\n",sig);
    
    return 0;
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
    float capture = -1;

    return capture;
  }
  
};

#endif /* _charge_exchange_h_ */
