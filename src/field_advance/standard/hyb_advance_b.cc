#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"
#include <Kokkos_Core.hpp>
#include <iostream>

#define F(ind,v) k_field(f##ind##_index, field_var::v)

// Faraday: dB^i/dt = -(curl E)^i. B is stored CONTRAVARIANT (cbx=B^1 etc) and
// E is COVARIANT (ex=E_1 etc, matching hyb_advance_e and the interpolator
// gather transform_E which uses grad xi = e/h^2). The contravariant curl of a
// covariant field is
//     (curl E)^i = (1/J) eps^{ijk} d_j E_k ,   J = h1 h2 h3 ,
// with NO inner scale factors (those belong to the physical-vector curl). The
// coordinate derivative d_j is py*(F(y,.)-F(my,.)) etc (py=0.5*rdy). On a
// CARTESIAN grid h=1 so J=1 and this reduces to the standard centered curl.
#define ROTEX()  ( inv_J * (                                       \
    py * (F(y,ez) - F(my,ez)) -                                    \
    pz * (F(z,ey) - F(mz,ey)) ) )

#define ROTEY()  ( inv_J * (                                       \
    pz * (F(z,ex) - F(mz,ex)) -                                    \
    px * (F(x,ez) - F(mx,ez)) ) )

#define ROTEZ()  ( inv_J * (                                       \
    px * (F(x,ey) - F(mx,ey)) -                                    \
    py * (F(y,ex) - F(my,ex)) ) )

#define INIT_STENCIL()                                               \
  size_t f0_index  = VOXEL(x,   y,   z,    nx,ny,nz);               \
  size_t fx_index  = VOXEL(x+1, y,   z,    nx,ny,nz);               \
  size_t fy_index  = VOXEL(x,   y+1, z,    nx,ny,nz);               \
  size_t fz_index  = VOXEL(x,   y,   z+1,  nx,ny,nz);               \
  size_t fmx_index = VOXEL(x-1, y,   z,    nx,ny,nz);               \
  size_t fmy_index = VOXEL(x,   y-1, z,    nx,ny,nz);               \
  size_t fmz_index = VOXEL(x,   y,   z-1,  nx,ny,nz);               \
  /* Curvilinear mesh data: covariant curl only needs the Jacobian at the    \
     cell center (E is already covariant, so no per-neighbor scale factors). */ \
  size_t m0_index  = GRID_TO_MESH(x,   y,   z,   nx, ny, nz);       \
  float h1_0  = k_curv_mesh(m0_index,  curv_mesh_var::h_1);          \
  float h2_0  = k_curv_mesh(m0_index,  curv_mesh_var::h_2);          \
  float h3_0  = k_curv_mesh(m0_index,  curv_mesh_var::h_3);          \
  float inv_J = 1.0f / (h1_0 * h2_0 * h3_0);  /* 1/Jacobian at cell center */

#define UPDATE_B(delt)               \
  F(0,cbx) = F(0,ox) - delt*ROTEX(); \
  F(0,cby) = F(0,oy) - delt*ROTEY(); \
  F(0,cbz) = F(0,oz) - delt*ROTEZ(); \

#define UPDATE1()    \
  UPDATE_B(dt2);     \
  F(0,tx) = ROTEX(); \
  F(0,ty) = ROTEY(); \
  F(0,tz) = ROTEZ()

#define UPDATE2()         \
  UPDATE_B(dt2);          \
  F(0,tx) += two*ROTEX(); \
  F(0,ty) += two*ROTEY(); \
  F(0,tz) += two*ROTEZ()

#define UPDATE3()         \
  UPDATE_B(dt);           \
  F(0,tx) += two*ROTEX(); \
  F(0,ty) += two*ROTEY(); \
  F(0,tz) += two*ROTEZ();

#define UPDATE4()          \
  UPDATE_B(dt6);           \
  F(0,cbx) -= dt6*F(0,tx); \
  F(0,cby) -= dt6*F(0,ty); \
  F(0,cbz) -= dt6*F(0,tz);


void
hyb_advance_b(field_array_t * RESTRICT fa,
          float       frac) {

  k_field_t k_field = fa->k_f_d;
  k_curvilinear_mesh_t k_curv_mesh = fa->g->k_curvilinear_mesh_d;  // Add this
  
  grid_t *g   = fa->g;
  size_t nx   = g->nx;
  size_t ny   = g->ny;
  size_t nz   = g->nz;
  size_t nv   = g->nv;
  float  px   = (nx>1) ? 0.5*g->rdx : 0;
  float  py   = (ny>1) ? 0.5*g->rdy : 0;
  float  pz   = (nz>1) ? 0.5*g->rdz : 0;

  const float isub = g->isub;
  const float nsub = g->nsub;
  const float dt=frac*(g->dt), dt6=dt/6.0, dt2=dt/2.0, two=2.0;
  
  //printf("Advance_B kernel\n");
  
  //Store initial B
  Kokkos::parallel_for("store b_old", Kokkos::RangePolicy<>(0,nv),
                       KOKKOS_LAMBDA(const int v) {
    k_field(v, field_var::ox) = k_field(v, field_var::cbx);
    k_field(v, field_var::oy) = k_field(v, field_var::cby);
    k_field(v, field_var::oz) = k_field(v, field_var::cbz);
  });

  // ----------------------------------------------------------
  // 0: Setup and smooth ion moments
  // ----------------------------------------------------------
    
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Smooth_Ion_Moments");
  //Only smooth on the first subcycle
  if (isub==0) {
#ifndef SHAPE_NGP
    //fix edge rho/currents for local BCs
    //adds adjacent ghost rho/current to cell and sets BC ghosts to 0
    k_hyb_local_adjust_jf(fa, fa->g);
    //fix edge rho/currents everywhere else
    //sends ghosts across ranks and adds ghosts to adjacent cells
    k_begin_remote_edge_hyb_jf(fa, fa->g, *(fa->fb) );
    k_end_remote_edge_hyb_jf  (fa, fa->g, *(fa->fb) );
#else
    // NGP accumulates only on live cells; ghosts empty.
    // So no need to fix edge rho/currents
#endif
    
    //refresh all ghosts
    Kokkos::Profiling::pushRegion("HybyridAdvanceB::Smooth_Ion_Moments::Exchange_JF");
    k_begin_remote_ghost_hyb_jf(fa);
    k_end_remote_ghost_hyb_jf  (fa);
    Kokkos::Profiling::popRegion();
    //fix ghosts for local BCs
    Kokkos::Profiling::pushRegion("HybyridAdvanceB::Smooth_Ion_Moments::Apply_Local_Ghost_JF");
    k_hyb_local_ghost_jf  (fa, fa->g);
    Kokkos::Profiling::popRegion();

    int ism = g->nsm;
    while(ism>0) {
      Kokkos::Profiling::pushRegion("HybyridAdvanceB::Smooth_Ion_Moments::Smooth_Moments");
      hyb_smooth_moments( fa );
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("HybyridAdvanceB::Smooth_Ion_Moments::Exchange_JF");
      k_begin_remote_ghost_hyb_jf(fa);
      k_end_remote_ghost_hyb_jf  (fa);
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("HybyridAdvanceB::Smooth_Ion_Moments::Apply_Local_Ghost_JF");
      k_hyb_local_ghost_jf  (fa, fa->g);
      Kokkos::Profiling::popRegion();
      ism--;
    }
  }
  Kokkos::Profiling::popRegion();
  
  // ----------------------------------------------------------
  // 1: Calculate electric field E_n(B_n), update B=B_n+dt/2*K1
  // ----------------------------------------------------------
  
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K1");
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K1::Advance_E");
  hyb_advance_e( fa, isub/nsub ); //sets ghost B + computes E
  Kokkos::Profiling::popRegion();
  
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K1::Remote");
  k_begin_remote_ghost_hyb_e( fa );//ARI add cell-centered BCs
  k_end_remote_ghost_hyb_e( fa );
  Kokkos::Profiling::popRegion();

  //fix local BCs
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K1::Local");
  k_hyb_local_ghost_e( fa, fa->g );
  Kokkos::Profiling::popRegion();
  
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nx+1,ny+1,nz+1});
  
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K1::UpdateStencil");
  Kokkos::parallel_for("hyb_advance_b_update1", xyz_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
    INIT_STENCIL();  
    UPDATE1();  
  });
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
  
  
  // ----------------------------------------------------------
  // 2: Update B=B_n+dt/2*K2 and store temp data
  // ----------------------------------------------------------
  
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K2");
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K2::Advance_E");
  hyb_advance_e( fa, (isub+0.5)/nsub) ; //sets ghost B's
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K2::Remote");
  k_begin_remote_ghost_hyb_e( fa );//ARI add cell-centered BCs
  k_end_remote_ghost_hyb_e( fa );
  Kokkos::Profiling::popRegion();
  //fix local BCs
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K2::Local");
  k_hyb_local_ghost_e( fa, fa->g );
  Kokkos::Profiling::popRegion();
  
  Kokkos::parallel_for("hyb_advance_b_update2", xyz_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
    INIT_STENCIL();  
    UPDATE2();  
  });
  Kokkos::Profiling::popRegion();
  
  // ----------------------------------------------------------
  // 3: Update B=B_n+dt*K3 and store temp data
  // ----------------------------------------------------------
  
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K3");
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K3::Advance_E");
  hyb_advance_e( fa, (isub+0.5)/nsub); //sets ghost B's
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K3::Remote");
  k_begin_remote_ghost_hyb_e( fa );//ARI add cell-centered BCs
  k_end_remote_ghost_hyb_e( fa );
  Kokkos::Profiling::popRegion();
  //fix local BCs
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K3::Local");
  k_hyb_local_ghost_e( fa, fa->g );
  Kokkos::Profiling::popRegion();
  
  Kokkos::parallel_for("hyb_advance_b_update3", xyz_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
    INIT_STENCIL();  
    UPDATE3();  
  });
  Kokkos::Profiling::popRegion();
  
  
  // ----------------------------------------------------------
  // 4: Update B=B_n+dt*(K1+2*K2+2*K3+K4)/6
  // ----------------------------------------------------------
  
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K4");
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K4::Advance_E");
  hyb_advance_e( fa, (isub+1.0)/nsub ); //sets ghost Bs
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K4::Remote");
  k_begin_remote_ghost_hyb_e( fa );//ARI add cell-centered BCs
  k_end_remote_ghost_hyb_e( fa );
  Kokkos::Profiling::popRegion();
  //fix local BCs
  Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_B_K4::Local");
  k_hyb_local_ghost_e( fa, fa->g );
  Kokkos::Profiling::popRegion();
  
  Kokkos::parallel_for("hyb_advance_b_update4", xyz_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
    INIT_STENCIL();    
    UPDATE4();  
  });
  Kokkos::Profiling::popRegion();
  
  // ----------------------------------------------------------
  // 5: Last E update last subcycle only
  // ----------------------------------------------------------
  
  if( (isub+1)==nsub ) {
    Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_E");
    Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_E::Advance_E");
    hyb_advance_e( fa, -1. ); //sets ghost Bs
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_E::Remote");
    k_begin_remote_ghost_hyb_e( fa );//ARI add cell-centered BCs
    k_end_remote_ghost_hyb_e( fa );
    Kokkos::Profiling::popRegion();
    //fix local BCs
    Kokkos::Profiling::pushRegion("HybyridAdvanceB::Update_E::Local");
    k_hyb_local_ghost_e( fa, fa->g );
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::popRegion();
    
    
    if(isub+1==nsub) {
      //Clear momentum source
      Kokkos::parallel_for("clear s", Kokkos::RangePolicy<>(0,nv),
      KOKKOS_LAMBDA(const int v) {
        k_field(v, field_var::sx) = 0;
        k_field(v, field_var::sy) = 0;
        k_field(v, field_var::sz) = 0;
      });
    }
  }
}