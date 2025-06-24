// Note: This is similar to vacuum_compute_curl_b

#define IN_sfa
#include "sfa_private.h"

typedef struct pipeline_args {
  /**/  field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
} pipeline_args_t;

#define F(ind,v) k_field(f##ind##_index, field_var::v)
#define CM(ind,cv) k_curv(f##ind##_index, curv_mesh_var::cv)

#define INIT_STENCIL()							\
  size_t f0_index  = VOXEL(x,   y,   z,    nx,ny,nz);			\
  size_t fx_index  = VOXEL(x+1, y,   z,    nx,ny,nz);			\
  size_t fy_index  = VOXEL(x,   y+1, z,    nx,ny,nz);			\
  size_t fz_index  = VOXEL(x,   y,   z+1,  nx,ny,nz);			\
  size_t fmx_index = VOXEL(x-1, y,   z,    nx,ny,nz);			\
  size_t fmy_index = VOXEL(x,   y-1, z,    nx,ny,nz);			\
  size_t fmz_index = VOXEL(x,   y,   z-1,  nx,ny,nz);			\
  float  rho = half*( (one-hstep)*( F(0,rhof) + F(0,rhofold) ) + hstep*( three*F(0,rhof) - F(0,rhofold)) ) ; \
  rho = (rho > den_floor_ohm) ? rho :  den_floor_ohm;			\
  float  invrho = one/rho;						\
  float hallinvrho = (rho > den_floor_ohm) ? invrho : 0 ;		\
  float  ux = invrho*half*( (one-hstep)*( F(0,jfx) + F(0,jfxold) ) + hstep*( three*F(0,jfx) - F(0,jfxold)) ) ; \
  float  uy = invrho*half*( (one-hstep)*( F(0,jfy) + F(0,jfyold) ) + hstep*( three*F(0,jfy) - F(0,jfyold)) ) ; \
  float  uz = invrho*half*( (one-hstep)*( F(0,jfz) + F(0,jfzold) ) + hstep*( three*F(0,jfz) - F(0,jfzold)) ) ; 


#define E(x_,y_,z_)							\
  F(0,e##x_) =								\
    invrho * (F(0,cb##z_) + F(0,cb##z_##0)) * ( p##z_*( F(z_,cb##x_) - F(m##z_,cb##x_) ) - p##x_*( F(x_,cb##z_) - F(m##x_,cb##z_)) ) \
  + invrho * (F(0,cb##y_) + F(0,cb##y_##0))  * ( p##y_*( F(y_,cb##x_) - F(m##y_,cb##x_) ) - p##x_*( F(x_,cb##y_) - F(m##x_,cb##y_)) ) \
       - u##y_ * (F(0,cb##z_)+F(0,cb##z_##0))  +   u##z_ * (F(0,cb##y_)+F(0,cb##y_##0)) \
    - invrho *(1.0/CM(0,h##x_))*( p##x_*( F(x_,pe) - F(m##x_,pe)) )	\
    + eta*F(0,tcay)*( p##y_*( F(y_,cb##z_) - F(m##y_,cb##z_) ) - p##z_*( F(z_,cb##y_) - F(m##z_,cb##y_) ) );\
  F(0,e##x_) *= F(0,tcaz);
  
//* (1.0/CM(0,h##x_))
  #define FIXEDGES()\
  Kokkos::parallel_for("advance_e", x_pos, KOKKOS_LAMBDA(const int z, const int y, const int x) {\
      INIT_STENCIL();\
	E(x,y,z);\
	E(y,z,x);\
	E(z,x,y);\
    });\
  Kokkos::parallel_for("advance_e", x_neg, KOKKOS_LAMBDA(const int z, const int y, const int x) {\
      INIT_STENCIL();\
	E(x,y,z);\
	E(y,z,x);\
	E(z,x,y);\
    });\
      Kokkos::parallel_for("advance_e", y_pos, KOKKOS_LAMBDA(const int z, const int y, const int x) {\
      INIT_STENCIL();\
	E(x,y,z);\
	E(y,z,x);\
	E(z,x,y);\
    });\
      Kokkos::parallel_for("advance_e", y_neg, KOKKOS_LAMBDA(const int z, const int y, const int x) {\
      INIT_STENCIL();\
	E(x,y,z);\
	E(y,z,x);\
	E(z,x,y);\
    });\
      Kokkos::parallel_for("advance_e", z_pos, KOKKOS_LAMBDA(const int z, const int y, const int x) {\
      INIT_STENCIL();\
	E(x,y,z);\
	E(y,z,x);\
	E(z,x,y);\
    });\
      Kokkos::parallel_for("advance_e", z_neg, KOKKOS_LAMBDA(const int z, const int y, const int x) {\
      INIT_STENCIL();\
	E(x,y,z);\
	E(y,z,x);\
	E(z,x,y);\
    });\



void
hyb_advance_e( field_array_t * RESTRICT fa,
                  float frac ) {
  if( !fa     ) ERROR(( "Bad args" ));

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  k_field_t k_field = fa->k_f_d;
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;

  const grid_t                 *              g = args->g;
  const size_t nx = g->nx, ny = g->ny, nz = g->nz;
  k_curvilinear_vars_t k_curv = g->k_curvilinear_vars_d;
  
  const float px = (nx>1) ? 0.5*g->rdx : 0;
  const float py = (ny>1) ? 0.5*g->rdy : 0;
  const float pz = (nz>1) ? 0.5*g->rdz : 0;
  const float eta = g->eta;
  const float den_floor_ohm = g->den_floor_ohm;

  const float hstep = frac;
  const float half = 1./2., one = 1., three = 3.;
  size_t ind2  = 2, ind1 = 1;
  
  
  //for interior cells
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nz+1,ny+1,nx+1});
  
  //for edge faces
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_pos({ind1,ind1,  nx},{nz+ind1,ny+ind1,nx+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_neg({ind1,ind1,ind1},{nz+ind1,ny+ind1,   ind2});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_pos({ind1,ny  ,ind1},{nz+ind1,ny+ind1,nx+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_neg({ind1,ind1,ind1},{nz+ind1,ind2   ,nx+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_pos({nz  ,ind1,ind1},{nz+ind1,ny+ind1,nx+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_neg({ind1,ind1,ind1},{ind2   ,ny+ind1,nx+ind1});
  
  
    /***************************************************************************
   * Calculate electron pressure
   ***************************************************************************/
    
    hyb_epress(fa, frac);

  /***************************************************************************
   * Begin tangential B ghost setup
   ***************************************************************************/
    
    k_begin_remote_ghost_hyb_b(fa, fa->g, *(fa->fb) );
    k_end_remote_ghost_hyb_b(fa, fa->g, *(fa->fb) );
    k_hyb_local_ghost_b( fa, fa->g );
    
   /***************************************************************************
   * Update E fields
   ****************************************************************************/ 
    
    //Compute E. Interior cells correct 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1, 1, 1}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_for("hyb_e", zyx_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
	INIT_STENCIL();
	E(x,y,z);
	E(y,z,x);
	E(z,x,y);
    });

    //Fix edge cells
    
    //k_end_remote_ghost_hyb_b(fa, fa->g, *(fa->fb) );
    //k_hyb_local_ghost_b( fa, fa->g );
    //FIXEDGES()

    //Apply hypereta to E field
    
    if(fa->g->hypereta>0) hyb_heta(fa);
    
}
