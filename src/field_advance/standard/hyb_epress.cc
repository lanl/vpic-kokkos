// Note: This is similar to vacuum_compute_curl_b

#define IN_sfa
#include "sfa_private.h"

typedef struct pipeline_args {
  /**/  field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
} pipeline_args_t;

#define F(ind,v) k_field(f##ind##_index, field_var::v)

#define INIT_STENCIL()							\
  size_t f0_index  = VOXEL(x,   y,   z,    nx,ny,nz);			\
  float  rho = half*( (one-hstep)*( F(0,rhof) + F(0,rhofold) ) + hstep*( three*F(0,rhof) - F(0,rhofold)) ) ; \
  /* Floor the density BEFORE the pow() below. Without this, empty/edge cells */ \
  /* with rho<=0 give pow(rho/eos_den, gamma)=NaN (gamma non-integer), which  */ \
  /* poisons pe -> E -> the whole field. */                                     \
  rho = (rho > den_floor_pe) ? rho : den_floor_pe;


void
hyb_epress( field_array_t * RESTRICT fa,
                  float frac ) {
  if( !fa     ) ERROR(( "Bad args" ));

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  k_field_t k_field = fa->k_f_d;
  //const material_coefficient_t * ALIGNED(128) m = args->p->mc;
  const grid_t                 *              g = args->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;

  const float den_floor_pe = g->den_floor_pe;

  const float hstep = frac;
  const float half = 1./2., one = 1., three = 3.;

  const float eos_den = g->eos_den;  // reference density for power-law equation of state
  const float eos_gamma = g->eos_gamma;  // power law exponent for equation of state

  Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({0, 0, 0}, {nz+2, ny+2, nx+2});

  // Avoid possibly costly power(...) if possible
  // Keep conditional branch outside the loop
  
	  if (eos_gamma == 1.0) {

    		Kokkos::parallel_for("hyb_epress", zyx_policy,
                         KOKKOS_LAMBDA(const int z, const int y, const int x) {
        		INIT_STENCIL();
        		F(0,pe) = F(0,te0) * (rho/eos_den);
    		});

  	} else {

    		Kokkos::parallel_for("hyb_epress", zyx_policy,
                         KOKKOS_LAMBDA(const int z, const int y, const int x) {
        		INIT_STENCIL();
        		F(0,pe) = F(0,te0) * pow(rho/eos_den,eos_gamma);
    		});
  	}
  
   
}
