#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"
#include <Kokkos_Core.hpp>
#include <iostream>

#define F(ind,v) k_field(f##ind##_index, field_var::v)
	

#define DIVUEP() \
  ( px*(F(x,ux)*F(x,pe) - F(mx,ux)*F(mx,pe) )\
  + py*(F(y,uy)*F(y,pe) - F(my,uy)*F(my,pe) )\
  + pz*(F(z,uz)*F(z,pe) - F(mz,uz)*F(mz,pe) ) )

#define UEGRADP() \
  ( F(0,ux) * px * ( F(x,pe) - F(mx,pe) ) \
  + F(0,uy) * py * ( F(y,pe) - F(my,pe) ) \
  + F(0,uz) * pz * ( F(z,pe) - F(mz,pe) ) )

#define DIVQE() \
  ( 4.0*px*px*(2.0*F(0,pe)/rho - F(x,pe)/rhox - F(mx,pe)/rhomx)	\
  + 4.0*py*py*(2.0*F(0,pe)/rho - F(y,pe)/rhoy - F(my,pe)/rhomy)	\
  + 4.0*pz*pz*(2.0*F(0,pe)/rho - F(z,pe)/rhoz - F(mz,pe)/rhomz) )


#define INIT_STENCIL()						\
  size_t f0_index  = VOXEL(x,   y,   z,    nx,ny,nz);		\
  size_t fx_index  = VOXEL(x+1, y,   z,    nx,ny,nz);		\
  size_t fy_index  = VOXEL(x,   y+1, z,    nx,ny,nz);		\
  size_t fz_index  = VOXEL(x,   y,   z+1,  nx,ny,nz);		\
  size_t fmx_index = VOXEL(x-1, y,   z,    nx,ny,nz);		\
  size_t fmy_index = VOXEL(x,   y-1, z,    nx,ny,nz);		\
  size_t fmz_index = VOXEL(x,   y,   z-1,  nx,ny,nz);		\
  float rho    = (F(0,rhof) > denmin) ? F(0,rhof) : denmin; \
  float rhox   = (F(x,rhof) > denmin) ? F(x,rhof) : denmin; \
  float rhoy   = (F(y,rhof) > denmin) ? F(y,rhof) : denmin; \
  float rhoz   = (F(z,rhof) > denmin) ? F(z,rhof) : denmin; \
  float rhomx  = (F(mx,rhof) > denmin) ? F(mx,rhof) : denmin; \
  float rhomy  = (F(my,rhof) > denmin) ? F(my,rhof) : denmin; \
  float rhomz  = (F(mz,rhof) > denmin) ? F(mz,rhof) : denmin; \
  float dpedt  =  gamma * DIVUEP() + (gamma-1.0) * (-UEGRADP() + kappa * DIVQE());

#define UPDATE_B(delt)				\
   F(0,pe)  = (F(0,rhof) > denmin) ? F(0,oe) - delt*dpedt : F(0,te0)*F(0,rhof);

#define UPDATE1()		\
  UPDATE_B(dt2);		\
   F(0,te) = dpedt

#define UPDATE2()		\
  UPDATE_B(dt2);		\
    F(0,te) += two*dpedt;

#define UPDATE3()		\
  UPDATE_B(dt);			\
    F(0,te) += two*dpedt;
  
#define UPDATE4()		\
  UPDATE_B(dt6);		\
    F(0,pe)  -= dt6*F(0,te);	\
  F(0,pe)  -= rV*two_thirds*F(0,se);\
  F(0,se)   = 0;                 \
  F(0,pe) = (F(0,rhof)>denmin) ? F(0,pe) : F(0,te0)*F(0,rhof);\
  F(0,pe) = (F(0,pe>0)) ? F(0,pe) : 0;


void
hyb_advance_pe(field_array_t * RESTRICT fa,
          float       frac) {

  k_field_t k_field = fa->k_f_d;

  grid_t *g   = fa->g;
  size_t nx   = g->nx;
  size_t ny   = g->ny;
  size_t nz   = g->nz;
  size_t nv   = g->nv;
  const float  px   = (nx>1) ? 0.5*g->rdx : 0;
  const float  py   = (ny>1) ? 0.5*g->rdy : 0;
  const float  pz   = (nz>1) ? 0.5*g->rdz : 0;
  const float  rV   = g->rdx*g->rdy*g->rdz;

  const float isub = g->isub;
  const float nsub = g->nsub;
  const float dt=frac*(g->dt), dt6=dt/6.0, dt2=dt/2.0, two=2.0, two_thirds=2./3.;
  const float denmin = g->den_floor_pe, gamma = g->eos_gamma, kappa = g->kappa;
  
//printf("Advance_B kernel\n");
  
//Store initial B
  Kokkos::parallel_for("store pe_old", Kokkos::RangePolicy<>(0,nv),
		       KOKKOS_LAMBDA(const int v) {
			k_field(v, field_var::oe) = k_field(v, field_var::pe);
		       });

  // ----------------------------------------------------------
  // 0: Setup and smooth ion moments
  // ----------------------------------------------------------
    
  Kokkos::Profiling::pushRegion("HybridAdvancePe::Smooth_Ion_Moments");
  //Only smooth on the first subcycle
  if (isub==0) {
    
    //smooth moments
    Kokkos::Profiling::pushRegion("HybridAdvancePe::Smooth_Ion_Moments::Exchange_JF");
    k_begin_remote_ghost_hyb_jf(fa );
    k_end_remote_ghost_hyb_jf  (fa );
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("HybridAdvancePe::Smooth_Ion_Moments::Apply_Local_Ghost_JF");
    k_hyb_local_ghost_jf  (fa, fa->g);
    Kokkos::Profiling::popRegion();

    int ism = g->nsm;
    while(ism>0) {
      Kokkos::Profiling::pushRegion("HybridAdvancePe::Smooth_Ion_Moments::Smooth_Moments");
      hyb_smooth_moments( fa );
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("HybridAdvancePe::Smooth_Ion_Moments::Exchange_JF");
      k_begin_remote_ghost_hyb_jf(fa );
      k_end_remote_ghost_hyb_jf  (fa );
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("HybridAdvancePe::Smooth_Ion_Moments::Apply_Local_Ghost_JF");
      k_hyb_local_ghost_jf  (fa, fa->g);
      Kokkos::Profiling::popRegion();
      ism--;
    }
  }
  Kokkos::Profiling::popRegion();
  
  // ----------------------------------------------------------
  // 1: Calculate electric field E_n(B_n), update B=B_n+dt/2*K1
  // ----------------------------------------------------------
  
  Kokkos::Profiling::pushRegion("HybridAdvancePe::Calc_E_Update_B_K1");
  Kokkos::Profiling::pushRegion("HybridAdvancePe::Calc_E_Update_B_K1::Advance_E");
  hyb_advance_ue( fa, isub/nsub ); //sets ghost B + computes E
  Kokkos::Profiling::popRegion();
  
  Kokkos::Profiling::pushRegion("HybridAdvancePe::Calc_E_Update_B_K1::Remote");
  k_begin_remote_ghost_hyb_ue( fa );
  k_end_remote_ghost_hyb_ue( fa );
  Kokkos::Profiling::popRegion();

  //fix local BCs
  Kokkos::Profiling::pushRegion("HybridAdvancePe::Calc_E_Update_B_K1::Local");
  k_hyb_local_ghost_e( fa, fa->g );
  Kokkos::Profiling::popRegion();
  
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nx+1,ny+1,nz+1});
  
  Kokkos::Profiling::pushRegion("HybridAdvancePe::Calc_E_Update_B_K1::UpdateStencil");
  Kokkos::parallel_for("hyb_advance_pe_update1", xyz_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
      INIT_STENCIL();	  
      UPDATE1();	  
    });
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
  
  
  // ----------------------------------------------------------
  // 2: Update B=B_n+dt/2*K2 and store temp data
  // ----------------------------------------------------------
  
  Kokkos::Profiling::pushRegion("HybridAdvancePe::Update_B_K2_Store_Temp");
  hyb_advance_ue( fa, (isub+0.5)/nsub) ; //sets ghost B's
  k_begin_remote_ghost_hyb_ue( fa );
  k_end_remote_ghost_hyb_ue( fa );
  //fix local BCs
  k_hyb_local_ghost_e( fa, fa->g );
  
  Kokkos::parallel_for("hyb_advance_pe_update2", xyz_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
      INIT_STENCIL();	  
      UPDATE2();	  
    });
  Kokkos::Profiling::popRegion();
  
  // ----------------------------------------------------------
  // 3: Update B=B_n+dt*K3 and store temp data
  // ----------------------------------------------------------
  
  Kokkos::Profiling::pushRegion("HybyridAdvancePe::Update_B_K3_Store_Temp");
  hyb_advance_ue( fa, (isub+0.5)/nsub); //sets ghost B's
  
  k_begin_remote_ghost_hyb_ue( fa );
  k_end_remote_ghost_hyb_ue( fa );
  //fix local BCs
  k_hyb_local_ghost_e( fa, fa->g );
  
  Kokkos::parallel_for("hyb_advance_pe_update3", xyz_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
      INIT_STENCIL();	  
      UPDATE3();	  
    });
  Kokkos::Profiling::popRegion();
  
  
  // ----------------------------------------------------------
  // 4: Update B=B_n+dt*(K1+2*K2+2*K3+K4)/6
  // ----------------------------------------------------------
  
  Kokkos::Profiling::pushRegion("HybyridAdvancePe::Update_B_K4");
  hyb_advance_ue( fa, (isub+1.0)/nsub ); //sets ghost Bs
  
  k_begin_remote_ghost_hyb_ue( fa );
  k_end_remote_ghost_hyb_ue( fa );
  //fix local BCs
  k_hyb_local_ghost_e( fa, fa->g );
  
  Kokkos::parallel_for("hyb_advance_pe_update4", xyz_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
      INIT_STENCIL();	  	  
      UPDATE4();	  
    });
  Kokkos::Profiling::popRegion();
  
  
  // ----------------------------------------------------------
  // 5: Last update
  // ----------------------------------------------------------
  
  Kokkos::Profiling::pushRegion("HybyridAdvancePe::Update_5");
  hyb_advance_ue( fa, (isub+1.0)/nsub ); //sets ghost pe s
  //k_begin_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );//ARI add cell-centered BCs
  //k_end_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );
  //fix local BCs
  //k_hyb_local_ghost_e( fa, fa->g );
  Kokkos::Profiling::popRegion();

}
