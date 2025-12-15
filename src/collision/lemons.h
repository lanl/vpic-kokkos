#ifndef _lemons_h_
#define _lemons_h_

#include "particle_bulk.h"

/**
 * @brief Lemons collision operator.
 */
struct lemons_collision_op_t : public particle_bulk_collision_op_t {
  double cvar0;
  //k_field_t k_field;
};


/**
 * @brief Lemons binary collision model.
 */
struct lemons_model : public collision_model<lemons_model> {
  CollisionType collision_type = CollisionType::BulkLemons;
  const float d_cvar0;
  const float d_twosqrtpi;
    lemons_model( float cvar0 ) : d_cvar0(cvar0), d_twosqrtpi(2.0 / sqrt( M_PI )) { };

  /**
   * @brief tan(theta/2) is normally distributed and variance scales ~ ur^-3/2.
   */
  KOKKOS_INLINE_FUNCTION
  float tan_theta_half(
    kokkos_rng_state_t& rg,
    float *param
  ) const
  {
      auto v = param[0];
      auto vjth = param[1];
      auto ndt_mi2 = param[2];
      auto cvar = d_cvar0*4.0*ndt_mi2;
      auto z = v/(vjth);
      auto erfz = erf( z );
      auto gz = ( erfz - z * d_twosqrtpi * exp( -z * z ) ) / ( 2.0 * z * z );
      auto gamma = cvar / ( v * v * v ) * ( erfz - gz );
      gamma = ( gamma > 0 ) ? sqrt( gamma ) : 0; // why>0?
      
    //    printf("#sigma=%e\n",sigma);
      auto dtheta = rg.normal(0, gamma);
      return tan(0.5*dtheta);
      
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
      auto v0 = param[0];
      //assert(v0>0);
      if(v0==0) return 0;
      auto v = v0;
      auto vjth = param[1];
      auto ndt_mi2 = param[2];
      auto mr = param[3]; //mi/mj
      
      auto mfact = 2.0*(1.0+mr);
      auto cvar = d_cvar0*4.0*ndt_mi2;
      auto z = v/(vjth);
      auto erfz = erf( z );
      auto gz = ( erfz - z * d_twosqrtpi * exp( -z * z ) ) / ( 2.0 * z * z );
      
      auto beta = 0.5 * cvar / ( v * v * v ) * ( gz * ( mfact * z * z + 1.0 ) - erfz );

      auto dsq = cvar * gz / v;
      dsq = ( dsq > 0 ) ? sqrt( dsq ) : 0;
      
      // Milstein method
      auto xi = rg.normal(0, 1);
      auto bbp = cvar / ( 4 * sqrt( z ) * v ) * ( z * d_twosqrtpi * exp( -z * z ) - 3 * gz );
      auto vs  = v - beta * v + dsq * xi + bbp * ( xi * xi - 1 );

      if ( 5.0 * v * v * v > ( 0.25 * mfact * mfact ) * 0.5 * cvar && vs > 0 ) {
	  v = vs;
      } else {
	  auto v2  = v * v;
	  auto gzz = ( erfz - z * d_twosqrtpi * exp( -z * z ) ) / ( 2.0 * z );
	  auto mu0 = -cvar / (vjth) *gzz * mr * 2.0;
	  auto mp0 = cvar * 2.0 / ( vjth * sqrt( 3.14 ) ) * exp( -z * z );
	  auto k1  = mu0 + mp0;
	  auto vv  = v2 + k1;
	  auto v_1 = vv > 0 ? sqrt( vv ) : v;
	  z          = v_1 / vjth;
	  erfz       = erf( z );
	  gzz        = ( erfz - z * d_twosqrtpi * exp( -z * z ) ) / ( 2.0 * z );
	  auto mu1 = -cvar / (vjth) *gzz * mr * 2.0;
	  auto mp1 = cvar * 2.0 / ( vjth * sqrt( 3.14 ) ) * exp( -z * z );
	  auto k2  = mu1 + mp1;
	  auto vv2 = v2 + k2;
	  vv         = ( vv + vv2 ) * 0.5;
	  v          = vv > 0 ? sqrt( vv ) : 0.0;
      }

      return v/v0;
      
  }
  
  template <class ViewType>
  KOKKOS_INLINE_FUNCTION
  void upload_moment_src_impl( const ViewType & spj_v, const int v,
			       const gmomType &Dm, const float mi, const float mj
			       ) const {
    if constexpr (std::is_same<ViewType, k_fluid_t>::value) {
	    // printf("#upload_moment_src_impl()  in lemons model\n");
      spj_v(v, fluid_var::msx) += -Dm.v[1]*mi;
      spj_v(v, fluid_var::msy) += -Dm.v[2]*mi;
      spj_v(v, fluid_var::msz) += -Dm.v[3]*mi;
      spj_v(v, fluid_var::ens) += -Dm.v[4]*mi;
      // printf("#msxyz=%e,%e,%e, ens=%e\n",spj_v(v, fluid_var::msx),spj_v(v, fluid_var::msy),spj_v(v, fluid_var::msz),spj_v(v, fluid_var::ens));
    } else if constexpr (std::is_same<ViewType, k_field_t>::value) {
      // FIELD implementation
	    // printf("lemons_model: uploading to field\n");
      spj_v(v, field_var::sx) += -Dm.v[1]*mi;
      spj_v(v, field_var::sy) += -Dm.v[2]*mi;
      spj_v(v, field_var::sz) += -Dm.v[3]*mi;
      spj_v(v, field_var::se) += -Dm.v[4]*mi;

      //printf("#mi=%e,wsum=%e, msxyz=%e,%e,%e, ens=%e\n",mi,Dm.v[0],spj_v(v, field_var::sx),spj_v(v, field_var::sy),spj_v(v, field_var::sz),spj_v(v, field_var::se));
    }
    
  } 
};

#endif /* _lemons_h_ */
