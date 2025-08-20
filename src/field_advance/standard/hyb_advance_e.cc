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
  float  rho = half*( (one-hstep)*( F(0,rhof) + F(0,rhofold) ) + hstep*( three*F(0,rhof) - F(0,rhofold)) ) ; \
  rho = (rho > den_floor_ohm) ? rho :  den_floor_ohm;			\
  float  invrho = one/rho;						\
  /*float hallinvrho = (rho > den_floor_ohm) ? invrho : 0 ;*/ \
  float  ux = invrho*half*( (one-hstep)*( F(0,jfx) + F(0,jfxold) ) + hstep*( three*F(0,jfx) - F(0,jfxold)) ) ; \
  float  uy = invrho*half*( (one-hstep)*( F(0,jfy) + F(0,jfyold) ) + hstep*( three*F(0,jfy) - F(0,jfyold)) ) ; \
  float  uz = invrho*half*( (one-hstep)*( F(0,jfz) + F(0,jfzold) ) + hstep*( three*F(0,jfz) - F(0,jfzold)) ) ; 


#define E(x_,y_,z_)							\
  F(0,e##x_) =								\
    invrho * (F(0,cb##z_) + F(0,cb##z_##0)) * ( p##z_*( F(z_,cb##x_) - F(m##z_,cb##x_) ) - p##x_*( F(x_,cb##z_) - F(m##x_,cb##z_)) ) \
  + invrho * (F(0,cb##y_) + F(0,cb##y_##0)) * ( p##y_*( F(y_,cb##x_) - F(m##y_,cb##x_) ) - p##x_*( F(x_,cb##y_) - F(m##x_,cb##y_)) ) \
       - u##y_ * (F(0,cb##z_)+F(0,cb##z_##0))  +   u##z_ * (F(0,cb##y_)+F(0,cb##y_##0)) \
      - invrho * ( p##x_*( F(x_,pe) - F(m##x_,pe)) )			\
    + eta*F(0,tcay)*( p##y_*( F(y_,cb##z_) - F(m##y_,cb##z_) ) - p##z_*( F(z_,cb##y_) - F(m##z_,cb##y_) ) );\
  F(0,e##x_) *= F(0,tcaz);
  
/*
  #define FIXEDGES()\
  if(ny > 1 || nz > 1) { \
  Kokkos::parallel_for("advance_e_x_pos", x_pos, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(ny > 1 || nz > 1) { \
  Kokkos::parallel_for("advance_e_x_neg", x_neg, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(nz > 1 || nx > 1) { \
  Kokkos::parallel_for("advance_e_y_pos", y_pos, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(nz > 1 || nx > 1) { \
  Kokkos::parallel_for("advance_e_y_neg", y_neg, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(nx > 1 || ny > 1) { \
  Kokkos::parallel_for("advance_e_z_pos", z_pos, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(nx > 1 || ny > 1) { \
  Kokkos::parallel_for("advance_e_z_neg", z_neg, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
	  E(x,y,z);\
	  E(y,z,x);\
	  E(z,x,y);\
    });\
  }
*/
  
  #define FIXEDGES()\
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> xpolicy({0,1,1}, {2,ny+1,nz+1}); \
    Kokkos::parallel_for("advance_e_x_faces", xpolicy, \
    KOKKOS_LAMBDA(const int face, const int y, const int z) { \
      const int x = face == 0 ? 1 : nx; \
      INIT_STENCIL(); \
      E(x,y,z);\
      E(y,z,x);\
      E(z,x,y);\
    }); \
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> ypolicy({1,0,1}, {nx+1,2,nz+1}); \
    Kokkos::parallel_for("advance_e_y_faces", ypolicy, \
    KOKKOS_LAMBDA(const int x, const int face, const int z) { \
      const int y = face == 0 ? 1 : ny; \
      INIT_STENCIL(); \
      E(x,y,z);\
      E(y,z,x);\
      E(z,x,y);\
    }); \
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zpolicy({1,1,0}, {nz+1,ny+1,2}); \
    Kokkos::parallel_for("advance_e_z_faces", zpolicy, \
    KOKKOS_LAMBDA(const int x, const int y, const int face) { \
      const int z = face == 0 ? 1 : nz; \
      INIT_STENCIL(); \
      E(x,y,z);\
      E(y,z,x);\
      E(z,x,y);\
    }); \



void
hyb_advance_e( field_array_t * RESTRICT fa,
                  float frac ) {
  if( !fa     ) ERROR(( "Bad args" ));

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  k_field_t k_field = fa->k_f_d;
  //const material_coefficient_t * ALIGNED(128) m = args->p->mc;
  const grid_t                 *              g = args->g;
  const size_t nx = g->nx, ny = g->ny, nz = g->nz;

  const float px = (nx>1) ? 0.5*g->rdx : 0;
  const float py = (ny>1) ? 0.5*g->rdy : 0;
  const float pz = (nz>1) ? 0.5*g->rdz : 0;
  const float eta = g->eta;
  const float den_floor_ohm = g->den_floor_ohm;

  const float hstep = frac;
  constexpr float half = 1./2., one = 1., three = 3.;
  constexpr size_t ind2  = 2, ind1 = 1;
  
  
  //for interior cells
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nx+1,ny+1,nz+1});
  
  //for edge faces
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_pos(fa->pos_x_face_space, {nx  ,ind1,ind1},{nx+ind1,ny+ind1,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_neg(fa->neg_x_face_space, {ind1,ind1,ind1},{ind2   ,ny+ind1,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_pos(fa->pos_y_face_space, {ind1,ny  ,ind1},{nx+ind1,ny+ind1,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_neg(fa->neg_y_face_space, {ind1,ind1,ind1},{nx+ind1,ind2   ,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_pos(fa->pos_z_face_space, {ind1,ind1,  nz},{nx+ind1,ny+ind1,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_neg(fa->neg_z_face_space, {ind1,ind1,ind1},{nx+ind1,ny+ind1,   ind2});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_pos({nx  ,ind1,ind1},{nx+ind1,ny+ind1,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_neg({ind1,ind1,ind1},{ind2   ,ny+ind1,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_pos({ind1,ny  ,ind1},{nx+ind1,ny+ind1,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_neg({ind1,ind1,ind1},{nx+ind1,ind2   ,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_pos({ind1,ind1,  nz},{nx+ind1,ny+ind1,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_neg({ind1,ind1,ind1},{nx+ind1,ny+ind1,   ind2});
  
  
  /***************************************************************************
   * Calculate electron pressure
   ***************************************************************************/
    
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Calculate_Electron_Pressure");
  hyb_epress(fa, frac); // Read te, rhof, rhofold, Write pe
  Kokkos::Profiling::popRegion();

  /***************************************************************************
   * Begin tangential B ghost setup
   ***************************************************************************/
    
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup");
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup::Begin_Remote_Ghost_Hybrid_B");
  k_begin_remote_ghost_hyb_b( fa ); // Read: cbx, cby, cbz
  Kokkos::Profiling::popRegion();

  /***************************************************************************
   * Calculate electron pressure
   ***************************************************************************/
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Calculate_Electron_Pressure");
  hyb_epress(fa, frac); // Read te, rhof, rhofold, Write: pe
  Kokkos::Profiling::popRegion();

  //Kokkos::Profiling::pushRegion("HybridAdvanceE::Update_E_Interior::Inner");
  //const int minx = nx > 2 ? 2 : 1;
  //const int miny = ny > 2 ? 2 : 1;
  //const int minz = nz > 2 ? 2 : 1;
  //const int maxx = nx > 2 ? nx : 2;
  //const int maxy = ny > 2 ? ny : 2;
  //const int maxz = nz > 2 ? nz : 2;
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> inner_zyx_policy(fa->inner_comp_space, {minx, miny, minz}, {maxx, maxy, maxz});
  //// Write: ex,ey,ez 
  //// Read: rhof, rhofold, jfx, jfy, jfz, jfxold, jfyold, jfzold, cbx, cby, cbz, tcax, tcay, tcaz, pe
  //Kokkos::parallel_for("hyb_advance_e_inner", inner_zyx_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
  //  INIT_STENCIL();
  //  E(x,y,z);
  //  E(y,z,x);
  //  E(z,x,y);
  //});
  //Kokkos::Profiling::popRegion();

  /***************************************************************************
   * End tangential B ghost setup
   ***************************************************************************/
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup::End_Remote_Ghost_Hybrid_B");
  k_end_remote_ghost_hyb_b(fa); // Write: cbx, cby, cbz
  Kokkos::Profiling::popRegion();

  /***************************************************************************
   * Apply local hybrid ghost b
   ***************************************************************************/
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup::Hybrid_Local_Ghost_B");
  k_hyb_local_ghost_b( fa, fa->g ); // R/W: cbx, cby, cbz
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
    
  /***************************************************************************
   * Update E fields
   ***************************************************************************/ 
    
  //Compute E. Interior cells correct 
   
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Update_E_Interior");
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_inner_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
  // Write: ex,ey,ez 
  // Read: rhof, rhofold, jfx, jfy, jfz, jfxold, jfyold, jfzold, cbx, cby, cbz, tcax, tcay, tcaz, pe
  Kokkos::parallel_for("hyb_advance_e_interior", xyz_inner_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
    INIT_STENCIL();
    E(x,y,z);
    E(y,z,x);
    E(z,x,y);
  });
  Kokkos::Profiling::popRegion();

  //Fix edge cells
  
  //k_end_remote_ghost_hyb_b(fa);
  //k_hyb_local_ghost_b( fa, fa->g );
  //FIXEDGES()
  //fa->inner_comp_space.fence();
  //fa->pos_x_face_space.fence();
  //fa->neg_x_face_space.fence();
  //fa->pos_y_face_space.fence();
  //fa->neg_y_face_space.fence();
  //fa->pos_z_face_space.fence();
  //fa->neg_z_face_space.fence();

  //Apply hypereta to E field
   
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Apply_Hyper_Eta");
  // Read: cbx, cby, cbz, tcax, tcay, tcaz, ex, ey, ez
  // Write: pex, pey, pez, ex, ey, ez
  if(fa->g->hypereta>0) hyb_heta(fa);
  Kokkos::Profiling::popRegion();
    
Kokkos::fence();
}
