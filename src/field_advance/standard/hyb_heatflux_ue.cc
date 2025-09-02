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
  float dum; 								\
  float  rho = half*( (one-hstep)*( F(0,rhof) + F(0,rhofold) ) + hstep*( three*F(0,rhof) - F(0,rhofold)) ) ; \
  rho = (rho > den_floor_ohm) ? rho :  den_floor_ohm;			\
  float  invrho = one/rho;						\
  float  ux = invrho*half*( (one-hstep)*( F(0,jfx) + F(0,jfxold) ) + hstep*( three*F(0,jfx) - F(0,jfxold)) ) ; \
  float  uy = invrho*half*( (one-hstep)*( F(0,jfy) + F(0,jfyold) ) + hstep*( three*F(0,jfy) - F(0,jfyold)) ) ; \
  float  uz = invrho*half*( (one-hstep)*( F(0,jfz) + F(0,jfzold) ) + hstep*( three*F(0,jfz) - F(0,jfzold)) ) ; \
  float  ze = half*( (one-hstep)*( F(0,ze ) + F(0,zeold ) ) + hstep*( three*F(0,ze ) - F(0,zeold) ) ) ; \
  ze = (rho > den_floor_ohm) ? ze :  den_floor_ohm;			\
  float  invz = one/ze;							\
  float  zx = invz*half*( (one-hstep)*( F(0,zx) + F(0,zxold) ) + hstep*( three*F(0,zx) - F(0,zxold)) ) ; \
  float  zy = invz*half*( (one-hstep)*( F(0,zy) + F(0,zyold) ) + hstep*( three*F(0,zy) - F(0,zyold)) ) ; \
  float  zz = invz*half*( (one-hstep)*( F(0,zz) + F(0,zzold) ) + hstep*( three*F(0,zz) - F(0,zzold)) ) ; \
  

#define UE(x_,y_,z_)							\
  F(0,u##x_) = invrho * (u##x_)
  
#define BETA(z)                      					\
 ( ( b1*z *( b2*z + b3 ) ) / ( b4*z*z + b5*z +b6 ) )

#define GAMMA(z)                      					\
 ( ( g1*z *( g2*z*z*z + g3*z*z + g4*z + g5 ) ) / ( g6 * (g7*z*z*z*z + g8*z*z*z + g9*z*z + g10*z + g11 ) ) )

#define TCOEFF(z)                      					\
 ( 3.0/2.0/nuei*( t1*z*z + t2*z) / (d1*z*z + d2*z + d3))

#define JCOEFF(z)                      					\
 ( -3.0/2.0*me*( j1*z*z + j2*z) / (d1*z*z + d2*z + d3) )
 
#define UPDATE_QE()							\
  float zeff = ze/rho; 							\
  float te = F(0,pe)/rho;						\
  float nuei = (nu > 0.) ? nu*zeff*rho/pow(te,1.5)/sqrtf(me) : 0;       \
  F(0,tx) = (nu > 0.) ? GAMMA(zeff)*F(0,pe)/me/nuei : 0; /*tx: heat conductivity*/ \
  dum = half*( (one-hstep)*( F(x,rhof) + F(x,rhofold) ) + hstep*( three*F(x,rhof) - F(x,rhofold)) ) ; \
  dum = (dum>den_floor_ohm) ? dum : den_floor_ohm;			\
  float tex = F(x,pe)/dum;						\
  dum = half*( (one-hstep)*( F(mx,rhof) + F(mx,rhofold) ) + hstep*( three*F(mx,rhof) - F(mx,rhofold)) ) ; \
  dum = (dum>den_floor_ohm) ? dum : den_floor_ohm;			\
  float temx = F(mx,pe)/dum;						\
  dum = half*( (one-hstep)*( F(y,rhof) + F(y,rhofold) ) + hstep*( three*F(y,rhof) - F(y,rhofold)) ) ; \
  dum = (dum>den_floor_ohm) ? dum : den_floor_ohm;			\
  float tey = F(y,pe)/dum;						\
  dum = half*( (one-hstep)*( F(my,rhof) + F(my,rhofold) ) + hstep*( three*F(my,rhof) - F(my,rhofold)) ) ; \
  dum = (dum>den_floor_ohm) ? dum : den_floor_ohm;			\
  float temy = F(my,pe)/dum;						\
  dum = half*( (one-hstep)*( F(z,rhof) + F(z,rhofold) ) + hstep*( three*F(z,rhof) - F(z,rhofold)) ) ; \
  dum = (dum>den_floor_ohm) ? dum : den_floor_ohm;			\
  float tez = F(z,pe)/dum;						\
  dum = half*( (one-hstep)*( F(mz,rhof) + F(mz,rhofold) ) + hstep*( three*F(mz,rhof) - F(mz,rhofold)) ) ; \
  dum = (dum>den_floor_ohm) ? dum : den_floor_ohm;			\
  float temz = F(mz,pe)/dum;						\
  F(0,pex) = F(0,tx)*px*(tex-temx); 	/*store heat flux in pe_x,y,z*/	\
  F(0,pey) = F(0,tx)*py*(tey-temy); 					\
  F(0,pez) = F(0,tx)*pz*(tez-temz); 					\
  F(0,sx) = nuei/zeff*( JCOEFF(zeff)*( ux - zx ) + TCOEFF(zeff)*px*( tex - temx ) ); /*TODO: don't need until final update*/\
  F(0,sy) = nuei/zeff*( JCOEFF(zeff)*( uy - zy ) + TCOEFF(zeff)*py*( tey - temy ) );\
  F(0,sz) = nuei/zeff*( JCOEFF(zeff)*( uz - zz ) + TCOEFF(zeff)*pz*( tez - temz ) );\
  float zeold = F(0,zeold);

#define UPDATE_BETA() \
 size_t f0_index  = VOXEL(x,   y,   z,    nx,ny,nz);			\
 float  rho = half*( (one-hstep)*( F(0,rhof) + F(0,rhofold) ) + hstep*( three*F(0,rhof) - F(0,rhofold)) ) ; \
 rho = (rho > den_floor_ohm) ? rho :  den_floor_ohm;			\
 float  ze = half*( (one-hstep)*( F(0,ze ) + F(0,zeold ) ) + hstep*( three*F(0,ze ) - F(0,zeold) ) ) ; \
 ze = (rho > den_floor_ohm) ? ze :  den_floor_ohm;			\
 float zeff = ze/rho; 							\
 float te = F(0,pe)/rho;						\
 float nuei = (nu > 0.) ? nu*zeff*rho/pow(te,1.5)/sqrtf(me) : 0;        \
 F(0,ty) = (nu > 0.) ? BETA(zeff)*F(0,pe) : 0 ;                         

void
hyb_heatflux_ue( field_array_t * RESTRICT fa,
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

  const float px = (nx>1) ? 0.5*g->rdx : 0;
  const float py = (ny>1) ? 0.5*g->rdy : 0;
  const float pz = (nz>1) ? 0.5*g->rdz : 0;
  const float eta = g->eta;
  const float den_floor_ohm = g->den_floor_ohm;
  const float me = g->me;
  const float nu = g->nu;

  const float hstep = frac;
  constexpr float half = 1./2., one = 1., three = 3.;
  constexpr size_t ind2  = 2, ind1 = 1;
  constexpr float twothirds = 2./3.;
  constexpr float b1 = 30., b2 = 11., b3 = 15.*sqrt(2.);
  constexpr float b4 = 217., b5 = 604.*sqrt(2.), b6 = 288.;
  constexpr float g1 = 25., g2 = 5299888., g3 = 21559755.*sqrt(2.);
  constexpr float g4 = 17831746., g5 = 1272672.*sqrt(2.);
  constexpr float g6 = 4., g7 = 2447104., g8 = 17445571.*sqrt(2.);
  constexpr float g9 = 57670090., g10 = 16033384.*sqrt(2.);
  constexpr float g11 = 2013696.; 
  constexpr float t1 = 220.0, t2 = 150.0*sqrt(2.); 
  constexpr float j1 = 102.0, j2 = 240.0*sqrt(2.);
  constexpr float d1 = 217.0, d2 = 604.0*sqrt(2.), d3 = 288.0;
  
  
  //for interior cells
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nx+1,ny+1,nz+1});
  
  //TODO: this passes B also, just need pe
  /***************************************************************************
   * Begin tangential B ghost setup
   ***************************************************************************/
    
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup");
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup::Begin_Remote_Ghost_Hybrid_B");
  k_begin_remote_ghost_hyb_b(fa, fa->g, *(fa->fb) ); // Read: cbx, cby, cbz
  Kokkos::Profiling::popRegion();


  /***************************************************************************
   * End tangential B ghost setup
   ***************************************************************************/
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup::End_Remote_Ghost_Hybrid_B");
  k_end_remote_ghost_hyb_b(fa, fa->g, *(fa->fb) ); // Write: cbx, cby, cbz
  Kokkos::Profiling::popRegion();

  /***************************************************************************
   * Apply local hybrid ghost b
   ***************************************************************************/
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup::Hybrid_Local_Ghost_B");
  k_hyb_local_ghost_b( fa, fa->g ); // R/W: cbx, cby, cbz
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
  
   
  /***************************************************************************
   * Update ue fields
   ***************************************************************************/ 
    
  //Compute E. Interior cells correct 
   
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Update_E_Interior");
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_inner_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
  // Write: ex,ey,ez 
  // Read: rhof, rhofold, jfx, jfy, jfz, jfxold, jfyold, jfzold, cbx, cby, cbz, tcax, tcay, tcaz, pe
  Kokkos::parallel_for("hyb_advance_e_interior", xyz_inner_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
    INIT_STENCIL();
    UE(x,y,z);
    UE(y,z,x);
    UE(z,x,y);
    UPDATE_QE();
    //if(z==nz && x==nx) std::cout << "zeff  " << zeff  << "    zeold   " <<  zeold  << "   gamma   " << GAMMA(zeff) <<   std::endl  ;
    
  });
  Kokkos::Profiling::popRegion();
  
    Kokkos::Profiling::pushRegion("HybridAdvanceE::Update_Beta");
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_all_policy({0, 0, 0}, {nx+2, ny+2, nz+2});
  // Write: ex,ey,ez 
  // Read: rhof, rhofold, jfx, jfy, jfz, jfxold, jfyold, jfzold, cbx, cby, cbz, tcax, tcay, tcaz, pe
  Kokkos::parallel_for("hyb_advance_e_interior", xyz_inner_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
    UPDATE_BETA();
    
  });
  Kokkos::Profiling::popRegion();

    
Kokkos::fence();
}
