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
  size_t fx_index  = VOXEL(x+1, y,   z,    nx,ny,nz);			\
  size_t fy_index  = VOXEL(x,   y+1, z,    nx,ny,nz);			\
  size_t fz_index  = VOXEL(x,   y,   z+1,  nx,ny,nz);			\
  size_t fmx_index = VOXEL(x-1, y,   z,    nx,ny,nz);			\
  size_t fmy_index = VOXEL(x,   y-1, z,    nx,ny,nz);			\
  size_t fmz_index = VOXEL(x,   y,   z-1,  nx,ny,nz);			\

//modified version of old hypereta macro curlbXYZ -> pXYZ
//note the pz here in the multiplier is NOT the curl
//#define LPL_B()\
  F(0,pex) = 4.0*( px*px*( F(x,cbx) + F(mx,cbx) - 2.0*F(0,cbx) ) +  \
		  py*py*( F(y,cbx) + F(my,cbx) - 2.0*F(0,cbx) ) +  \
		  pz*pz*( F(z,cbx) + F(mz,cbx) - 2.0*F(0,cbx) ) ); \
  F(0,pey) = 4.0*( px*px*( F(x,cby) + F(mx,cby) - 2.0*F(0,cby) ) +  \
		  py*py*( F(y,cby) + F(my,cby) - 2.0*F(0,cby) ) +  \
		  pz*pz*( F(z,cby) + F(mz,cby) - 2.0*F(0,cby) ) ); \
  F(0,pez) = 4.0*( px*px*( F(x,cbz) + F(mx,cbz) - 2.0*F(0,cbz) ) +  \
		  py*py*( F(y,cbz) + F(my,cbz) - 2.0*F(0,cbz) ) +  \
		  pz*pz*( F(z,cbz) + F(mz,cbz) - 2.0*F(0,cbz) ) ); \

#define LPL_B()\
  F(0,pex) = 4.0*(                                   \
      px2*( F(x,cbx) + F(mx,cbx) - 2.0*F(0,cbx) ) +  \
		  py2*( F(y,cbx) + F(my,cbx) - 2.0*F(0,cbx) ) +  \
		  pz2*( F(z,cbx) + F(mz,cbx) - 2.0*F(0,cbx) ) ); \
  F(0,pey) = 4.0*(                                   \
      px2*( F(x,cby) + F(mx,cby) - 2.0*F(0,cby) ) +  \
		  py2*( F(y,cby) + F(my,cby) - 2.0*F(0,cby) ) +  \
		  pz2*( F(z,cby) + F(mz,cby) - 2.0*F(0,cby) ) ); \
  F(0,pez) = 4.0*(                                   \
      px2*( F(x,cbz) + F(mx,cbz) - 2.0*F(0,cbz) ) +  \
		  py2*( F(y,cbz) + F(my,cbz) - 2.0*F(0,cbz) ) +  \
		  pz2*( F(z,cbz) + F(mz,cbz) - 2.0*F(0,cbz) ) ); \

#define CURL_LPL_B(x_,y_,z_)						\
  F(0,e##x_) -= hypereta*F(0,tcax)*F(0,tcaz)*( p##y_*( F(y_,pe##z_) - F(m##y_,pe##z_) ) \
				                                     - p##z_*( F(z_,pe##y_) - F(m##z_,pe##y_) ) )


void
hyb_heta( field_array_t * RESTRICT fa ) {
  if( !fa     ) ERROR(( "Bad args" ));

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  k_field_t k_field = fa->k_f_d;
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;
  const grid_t                 *              g = args->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;

  const float px = (nx>1) ? 0.5*g->rdx : 0;
  const float py = (ny>1) ? 0.5*g->rdy : 0;
  const float pz = (nz>1) ? 0.5*g->rdz : 0;
  const float hypereta = g->hypereta;
  //const float den_floor_ohm = g->den_floor_ohm;

  const float px2 = px*px;
  const float py2 = py*py;
  const float pz2 = pz*pz;

  // Laplace B Loop
    
  // Write: pex, pey, pez
  // Read: cbx, cby, cbz
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
  Kokkos::parallel_for("hyb_hypereta_lpl_b", xyz_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
      INIT_STENCIL();
      LPL_B();
    });
    
  // Operations on the ghost cells
  k_begin_remote_ghost_hyb_curl_lpl_b(fa); // Read: pex, pey, pez
  k_end_remote_ghost_hyb_curl_lpl_b(fa); // Write: pex, pey, pez
  k_hyb_local_ghost_lapl_b(fa, fa->g); // R/W: pex, pey, pez

  // Curl Laplace B Loop

  Kokkos::MDRangePolicy<Kokkos::Rank<3>> curl_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
  // Read: tcax, tcay, tcaz, pex, pey, pez
  // Write: ex, ey, ez
  Kokkos::parallel_for("hyb_hypereta_curl_lpl_b", curl_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
      INIT_STENCIL();
      CURL_LPL_B(x,y,z);
      CURL_LPL_B(y,z,x);
      CURL_LPL_B(z,x,y);	
    });
        
}
