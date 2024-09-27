#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"
#include <Kokkos_Core.hpp>
#include <iostream>

#define F(ind,v) k_field(f##ind##_index, field_var::v)
  
#define  ROTEX()  ( py*( F(y,ez) - F(my,ez) ) - pz*( F(z,ey) - F(mz,ey) ) )	
#define  ROTEY()  ( pz*( F(z,ex) - F(mz,ex) ) - px*( F(x,ez) - F(mx,ez) ) )	
#define  ROTEZ()  ( px*( F(x,ey) - F(mx,ey) ) - py*( F(y,ex) - F(my,ex) ) )	

#define INIT_STENCIL()						\
  size_t f0_index  = VOXEL(x,   y,   z,    nx,ny,nz);		\
  size_t fx_index  = VOXEL(x+1, y,   z,    nx,ny,nz);		\
  size_t fy_index  = VOXEL(x,   y+1, z,    nx,ny,nz);		\
  size_t fz_index  = VOXEL(x,   y,   z+1,  nx,ny,nz);		\
  size_t fmx_index = VOXEL(x-1, y,   z,    nx,ny,nz);		\
  size_t fmy_index = VOXEL(x,   y-1, z,    nx,ny,nz);		\
  size_t fmz_index = VOXEL(x,   y,   z-1,  nx,ny,nz);

#define UPDATE_B(delt)				\
  F(0,cbx) = F(0,ox) - delt*ROTEX();		\
  F(0,cby) = F(0,oy) - delt*ROTEY();		\
  F(0,cbz) = F(0,oz) - delt*ROTEZ();		\

#define UPDATE1()					 \
  UPDATE_B(dt2);					 \
  F(0,tx) = ROTEX();					 \
  F(0,ty) = ROTEY();					 \
  F(0,tz) = ROTEZ()

#define UPDATE2()					 \
  UPDATE_B(dt2);					 \
  F(0,tx) += two*ROTEX();					 \
  F(0,ty) += two*ROTEY();					 \
  F(0,tz) += two*ROTEZ()

#define UPDATE3()					 \
  UPDATE_B(dt);						 \
  F(0,tx) += two*ROTEX();					 \
  F(0,ty) += two*ROTEY();					 \
  F(0,tz) += two*ROTEZ();

#define UPDATE4()					 \
  UPDATE_B(dt6);					 \
  F(0,cbx) -= dt6*F(0,tx);				 \
  F(0,cby) -= dt6*F(0,ty);				 \
  F(0,cbz) -= dt6*F(0,tz);


void
hyb_advance_b(field_array_t * RESTRICT fa,
          float       frac) {

  k_field_t k_field = fa->k_f_d;

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
    
  //Only smooth on the first subcycle
  if (isub==0) {
    
    //smooth moments
    k_begin_remote_ghost_hyb_jf(fa, fa->g, *(fa->fb) );
    k_end_remote_ghost_hyb_jf  (fa, fa->g, *(fa->fb) );
    k_hyb_local_ghost_jf  (fa, fa->g);

    int ism = g->nsm;
    while(ism>0) {
      hyb_smooth_moments( fa );
      k_begin_remote_ghost_hyb_jf(fa, fa->g, *(fa->fb) );
      k_end_remote_ghost_hyb_jf  (fa, fa->g, *(fa->fb) );
      k_hyb_local_ghost_jf  (fa, fa->g);
      ism--;
    }
  }
  
  // ----------------------------------------------------------
  // 1: Calculate electric field E_n(B_n), update B=B_n+dt/2*K1
  // ----------------------------------------------------------
  
  hyb_advance_e( fa, isub/nsub ); //sets ghost B + computes E
  
  k_begin_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );//ARI add cell-centered BCs
  k_end_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );

  //fix local BCs
  k_hyb_local_ghost_e( fa, fa->g );
  
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nz+1,ny+1,nx+1});
  
  Kokkos::parallel_for("advance_b", xyz_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
      INIT_STENCIL();	  
      UPDATE1();	  
    });
  
  
  
  // ----------------------------------------------------------
  // 2: Update B=B_n+dt/2*K2 and store temp data
  // ----------------------------------------------------------
  
  hyb_advance_e( fa, (isub+0.5)/nsub) ; //sets ghost B's
  k_begin_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );//ARI add cell-centered BCs
  k_end_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );
  //fix local BCs
  k_hyb_local_ghost_e( fa, fa->g );
  
  Kokkos::parallel_for("advance_b", xyz_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
      INIT_STENCIL();	  
      UPDATE2();	  
    });
  
  // ----------------------------------------------------------
  // 3: Update B=B_n+dt*K3 and store temp data
  // ----------------------------------------------------------
  
  hyb_advance_e( fa, (isub+0.5)/nsub); //sets ghost B's
  k_begin_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );//ARI add cell-centered BCs
  k_end_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );
  //fix local BCs
  k_hyb_local_ghost_e( fa, fa->g );
  
  Kokkos::parallel_for("advance_b", xyz_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
      INIT_STENCIL();	  
      UPDATE3();	  
    });
  
  
  // ----------------------------------------------------------
  // 4: Update B=B_n+dt*(K1+2*K2+2*K3+K4)/6
  // ----------------------------------------------------------
  
  hyb_advance_e( fa, (isub+1.0)/nsub ); //sets ghost Bs
  
  k_begin_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );//ARI add cell-centered BCs
  k_end_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );
  //fix local BCs
  k_hyb_local_ghost_e( fa, fa->g );
  
  Kokkos::parallel_for("advance_b", xyz_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
      INIT_STENCIL();	  	  
      UPDATE4();	  
    });
  
  // ----------------------------------------------------------
  // 5: Last E update
  // ----------------------------------------------------------
  
  hyb_advance_e( fa, (isub+1.0)/nsub ); //sets ghost Bs
  k_begin_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );//ARI add cell-centered BCs
  k_end_remote_ghost_hyb_e( fa, fa->g, *(fa->fb) );
  //fix local BCs
  k_hyb_local_ghost_e( fa, fa->g );

}
