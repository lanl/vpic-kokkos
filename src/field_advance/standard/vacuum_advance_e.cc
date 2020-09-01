// Note: This is similar to vacuum_compute_curl_b

#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"

typedef struct pipeline_args {
  /**/  field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
} pipeline_args_t;

#define DECLARE_STENCIL()                                                    \
  /**/  field_t                * ALIGNED(128) f = args->f;                   \
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;               \
  const grid_t                 *              g = args->g;                   \
  const int nx = g->nx, ny = g->ny, nz = g->nz;                              \
                                                                             \
  const float decayx = m->decayx, drivex = m->drivex;                        \
  const float decayy = m->decayy, drivey = m->drivey;                        \
  const float decayz = m->decayz, drivez = m->drivez;                        \
  const float damp   = args->p->damp;                                        \
  const float px_muz = ((nx>1) ? (1+damp)*g->cvac*g->dt*g->rdx : 0)*m->rmuz; \
  const float px_muy = ((nx>1) ? (1+damp)*g->cvac*g->dt*g->rdx : 0)*m->rmuy; \
  const float py_mux = ((ny>1) ? (1+damp)*g->cvac*g->dt*g->rdy : 0)*m->rmux; \
  const float py_muz = ((ny>1) ? (1+damp)*g->cvac*g->dt*g->rdy : 0)*m->rmuz; \
  const float pz_muy = ((nz>1) ? (1+damp)*g->cvac*g->dt*g->rdz : 0)*m->rmuy; \
  const float pz_mux = ((nz>1) ? (1+damp)*g->cvac*g->dt*g->rdz : 0)*m->rmux; \
  const float cj     = g->dt/g->eps0;                                        \
                                                                             \
  field_t * ALIGNED(16) f0;                                                  \
  field_t * ALIGNED(16) fx, * ALIGNED(16) fy, * ALIGNED(16) fz;              \
  int x, y, z

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

#define INIT_STENCIL()  \
  f0 = &f(x,  y,  z  ); \
  fx = &f(x-1,y,  z  ); \
  fy = &f(x,  y-1,z  ); \
  fz = &f(x,  y,  z-1)

#define NEXT_STENCIL()                \
  f0++; fx++; fy++; fz++; x++;        \
  if( x>nx ) {                        \
    /**/       y++;            x = 2; \
    if( y>ny ) z++; if( y>ny ) y = 2; \
    INIT_STENCIL();                   \
  }

KOKKOS_INLINE_FUNCTION void update_ex(const k_field_t& k_field, const size_t f0_idx, const size_t fx_idx, const size_t fy_idx, const size_t fz_idx,
                const float px_muy, const float px_muz, const float py_mux, const float py_muz, const float pz_mux, const float pz_muy,
                const float damp, const float decayx, const float drivex, const float cj) {

    const float f0_ex = k_field(f0_idx, field_var::ex);
    const float f0_cby = k_field(f0_idx, field_var::cby);
    const float f0_cbz = k_field(f0_idx, field_var::cbz);
    const float f0_tcax = k_field(f0_idx, field_var::tcax);
    const float f0_jfx = k_field(f0_idx, field_var::jfx);
    const float fy_cbz = k_field(fy_idx, field_var::cbz);
    const float fz_cby = k_field(fz_idx, field_var::cby);
    
    k_field(f0_idx, field_var::tcax) = (py_muz*(f0_cbz - fy_cbz) - pz_muy*(f0_cby - fz_cby)) - damp*f0_tcax;
    k_field(f0_idx, field_var::ex) = decayx*f0_ex + drivex*(k_field(f0_idx, field_var::tcax) - cj*f0_jfx);
}
KOKKOS_INLINE_FUNCTION void update_ey(const k_field_t& k_field, const size_t f0_idx, const size_t fx_idx, const size_t fy_idx, const size_t fz_idx,
                const float px_muy, const float px_muz, const float py_mux, const float py_muz, const float pz_mux, const float pz_muy,
                const float damp, const float decayy, const float drivey, const float cj) {

    const float f0_ey = k_field(f0_idx, field_var::ey);
    const float f0_cbx = k_field(f0_idx, field_var::cbx);
    const float f0_cbz = k_field(f0_idx, field_var::cbz);
    const float f0_tcay = k_field(f0_idx, field_var::tcay);
    const float f0_jfy = k_field(f0_idx, field_var::jfy);
    const float fx_cbz = k_field(fx_idx, field_var::cbz);
    const float fz_cbx = k_field(fz_idx, field_var::cbx);
    
    k_field(f0_idx, field_var::tcay) = (pz_mux*(f0_cbx - fz_cbx) - px_muz*(f0_cbz - fx_cbz)) - damp*f0_tcay;
    k_field(f0_idx, field_var::ey) = decayy*f0_ey + drivey*(k_field(f0_idx, field_var::tcay) - cj*f0_jfy);
}
KOKKOS_INLINE_FUNCTION void update_ez(const k_field_t& k_field, const size_t f0_idx, const size_t fx_idx, const size_t fy_idx, const size_t fz_idx,
                const float px_muy, const float px_muz, const float py_mux, const float py_muz, const float pz_mux, const float pz_muy,
                const float damp, const float decayz, const float drivez, const float cj) {

    const float f0_ez = k_field(f0_idx, field_var::ez);
    const float f0_cbx = k_field(f0_idx, field_var::cbx);
    const float f0_cby = k_field(f0_idx, field_var::cby);
    const float f0_tcaz = k_field(f0_idx, field_var::tcaz);
    const float f0_jfz = k_field(f0_idx, field_var::jfz);
    const float fx_cby = k_field(fx_idx, field_var::cby);
    const float fy_cbx = k_field(fy_idx, field_var::cbx);
    
    k_field(f0_idx, field_var::tcaz) = (px_muy*(f0_cby - fx_cby) - py_mux*(f0_cbx - fy_cbx)) - damp*f0_tcaz;
    k_field(f0_idx, field_var::ez) = decayz*f0_ez + drivez*(k_field(f0_idx, field_var::tcaz) - cj*f0_jfz);
}
#define UPDATE_EX() \
  f0->tcax = ( py_muz*(f0->cbz-fy->cbz) - pz_muy*(f0->cby-fz->cby) ) - \
             damp*f0->tcax; \
  f0->ex   = decayx*f0->ex + drivex*( f0->tcax - cj*f0->jfx )
#define UPDATE_EY() \
  f0->tcay = ( pz_mux*(f0->cbx-fz->cbx) - px_muz*(f0->cbz-fx->cbz) ) - \
             damp*f0->tcay; \
  f0->ey   = decayy*f0->ey + drivey*( f0->tcay - cj*f0->jfy )
#define UPDATE_EZ() \
  f0->tcaz = ( px_muy*(f0->cby-fx->cby) - py_mux*(f0->cbx-fy->cbx) ) - \
             damp*f0->tcaz; \
  f0->ez   = decayz*f0->ez + drivez*( f0->tcaz - cj*f0->jfz )

void
vacuum_advance_e_pipeline( pipeline_args_t * args,
                           int pipeline_rank,
                           int n_pipeline ) {
  DECLARE_STENCIL();

  int n_voxel;
  DISTRIBUTE_VOXELS( 2,nx, 2,ny, 2,nz, 16,
                     pipeline_rank, n_pipeline,
                     x, y, z, n_voxel );

  INIT_STENCIL();
  for( ; n_voxel; n_voxel-- ) {
    UPDATE_EX(); UPDATE_EY(); UPDATE_EZ(); 
    NEXT_STENCIL();
  }
}

#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

using namespace v4;

void
vacuum_advance_e_pipeline_v4( pipeline_args_t * args,
                              int pipeline_rank,
                              int n_pipeline ) {
  DECLARE_STENCIL();

  int n_voxel;
  DISTRIBUTE_VOXELS( 2,nx, 2,ny, 2,nz, 16,
                     pipeline_rank, n_pipeline,
                     x, y, z, n_voxel );

  const v4float vdecayx( decayx ), vdrivex( drivex );
  const v4float vdecayy( decayy ), vdrivey( drivey );
  const v4float vdecayz( decayz ), vdrivez( drivez );
  const v4float vdamp( damp );
  const v4float vpx_muz( px_muz ), vpx_muy( px_muy );
  const v4float vpy_mux( py_mux ), vpy_muz( py_muz );
  const v4float vpz_muy( pz_muy ), vpz_mux( pz_mux );
  const v4float vcj( cj );

  v4float save0, save1, dummy;

  v4float f0_ex,   f0_ey,   f0_ez;
  v4float f0_cbx,  f0_cby,  f0_cbz;
  v4float f0_tcax, f0_tcay, f0_tcaz;
  v4float f0_jfx,  f0_jfy,  f0_jfz;
  v4float          fx_cby,  fx_cbz;
  v4float fy_cbx,           fy_cbz;
  v4float fz_cbx,  fz_cby;

  field_t * ALIGNED(16) f00, * ALIGNED(16) f01, * ALIGNED(16) f02, * ALIGNED(16) f03; // Voxel quad
  field_t * ALIGNED(16) fx0, * ALIGNED(16) fx1, * ALIGNED(16) fx2, * ALIGNED(16) fx3; // Voxel quad +x neighbors
  field_t * ALIGNED(16) fy0, * ALIGNED(16) fy1, * ALIGNED(16) fy2, * ALIGNED(16) fy3; // Voxel quad +y neighbors
  field_t * ALIGNED(16) fz0, * ALIGNED(16) fz1, * ALIGNED(16) fz2, * ALIGNED(16) fz3; // Voxel quad +z neighbors

  // Process the bulk of the voxels 4 at a time

  INIT_STENCIL();
  for( ; n_voxel>3; n_voxel-=4 ) {
    f00 = f0; fx0 = fx; fy0 = fy; fz0 = fz; NEXT_STENCIL();
    f01 = f0; fx1 = fx; fy1 = fy; fz1 = fz; NEXT_STENCIL();
    f02 = f0; fx2 = fx; fy2 = fy; fz2 = fz; NEXT_STENCIL();
    f03 = f0; fx3 = fx; fy3 = fy; fz3 = fz; NEXT_STENCIL();

    load_4x4_tr( &f00->ex,   &f01->ex,   &f02->ex,   &f03->ex,   f0_ex,   f0_ey,   f0_ez,   save0 );
    load_4x3_tr( &f00->cbx,  &f01->cbx,  &f02->cbx,  &f03->cbx,  f0_cbx,  f0_cby,  f0_cbz         );
    load_4x4_tr( &f00->tcax, &f01->tcax, &f02->tcax, &f03->tcax, f0_tcax, f0_tcay, f0_tcaz, save1 );
    load_4x3_tr( &f00->jfx,  &f01->jfx,  &f02->jfx,  &f03->jfx,  f0_jfx,  f0_jfy,  f0_jfz         );

    load_4x3_tr( &fx0->cbx,  &fx1->cbx,  &fx2->cbx,  &fx3->cbx,  dummy,   fx_cby,  fx_cbz         );
    load_4x3_tr( &fy0->cbx,  &fy1->cbx,  &fy2->cbx,  &fy3->cbx,  fy_cbx,  dummy,   fy_cbz         );
    load_4x2_tr( &fz0->cbx,  &fz1->cbx,  &fz2->cbx,  &fz3->cbx,  fz_cbx,  fz_cby   /**/           );

    f0_tcax = fnms( vdamp,f0_tcax, fms( vpy_muz,(f0_cbz-fy_cbz), vpz_muy*(f0_cby-fz_cby) ) );
    f0_tcay = fnms( vdamp,f0_tcay, fms( vpz_mux,(f0_cbx-fz_cbx), vpx_muz*(f0_cbz-fx_cbz) ) );
    f0_tcaz = fnms( vdamp,f0_tcaz, fms( vpx_muy,(f0_cby-fx_cby), vpy_mux*(f0_cbx-fy_cbx) ) );

    f0_ex   = fma( vdecayx,f0_ex, vdrivex*fnms( vcj,f0_jfx, f0_tcax ) );
    f0_ey   = fma( vdecayy,f0_ey, vdrivey*fnms( vcj,f0_jfy, f0_tcay ) );
    f0_ez   = fma( vdecayz,f0_ez, vdrivez*fnms( vcj,f0_jfz, f0_tcaz ) );

    // Note: Unlike load_4x3 versus load_4x4, store_4x4 is much more efficient than store_4x3!
    store_4x4_tr( f0_ex,   f0_ey,   f0_ez,   save0, &f00->ex,    &f01->ex,    &f02->ex,    &f03->ex   );
    store_4x4_tr( f0_tcax, f0_tcay, f0_tcaz, save1, &f00->tcax,  &f01->tcax,  &f02->tcax,  &f03->tcax );
  }
}

#endif

void
vacuum_advance_e( field_array_t * RESTRICT fa,
                  float frac ) {
  if( !fa     ) ERROR(( "Bad args" ));
  if( frac!=1 ) ERROR(( "standard advance_e does not support frac!=1 yet" ));

//printf("Vacuum Advance E Kernel\n");

  /***************************************************************************
   * Begin tangential B ghost setup
   ***************************************************************************/
  
  begin_remote_ghost_tang_b( fa->f, fa->g );
  local_ghost_tang_b( fa->f, fa->g );

  /***************************************************************************
   * Update interior fields
   * Note: ex all (1:nx,  1:ny+1,1,nz+1) interior (1:nx,2:ny,2:nz)
   * Note: ey all (1:nx+1,1:ny,  1:nz+1) interior (2:nx,1:ny,2:nz)
   * Note: ez all (1:nx+1,1:ny+1,1:nz  ) interior (1:nx,1:ny,2:nz)
   ***************************************************************************/

  // Do majority interior in a single pass.  The host handles
  // stragglers.

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  EXEC_PIPELINES( vacuum_advance_e, args, 0 );

  // While the pipelines are busy, do non-bulk interior fields

  DECLARE_STENCIL();

  // Do left over interior ex
  for( z=2; z<=nz; z++ ) {
    for( y=2; y<=ny; y++ ) {
      f0 = &f(1,y,  z);
      fy = &f(1,y-1,z);
      fz = &f(1,y,  z-1);
      UPDATE_EX();
    }
  }

  // Do left over interior ey
  for( z=2; z<=nz; z++ ) {
    f0 = &f(2,1,z);
    fx = &f(1,1,z);
    fz = &f(2,1,z-1);
    for( x=2; x<=nx; x++ ) {
      UPDATE_EY();
      f0++;
      fx++;
      fz++;
    }
  }

  // Do left over interior ez
  for( y=2; y<=ny; y++ ) {
    f0 = &f(2,y,  1);
    fx = &f(1,y,  1);
    fy = &f(2,y-1,1);
    for( x=2; x<=nx; x++ ) {
      UPDATE_EZ();
      f0++;
      fx++;
      fy++;
    }
  }

  WAIT_PIPELINES();
  
  /***************************************************************************
   * Finish tangential B ghost setup
   ***************************************************************************/

  end_remote_ghost_tang_b( fa->f, fa->g );

  /***************************************************************************
   * Update exterior fields
   ***************************************************************************/

  // Do exterior ex
  for( y=1; y<=ny+1; y++ ) {
    f0 = &f(1,y,  1);
    fy = &f(1,y-1,1);
    fz = &f(1,y,  0);
    for( x=1; x<=nx; x++ ) {
      UPDATE_EX();
      f0++;
      fy++;
      fz++;
    }
  }
  for( y=1; y<=ny+1; y++ ) {
    f0 = &f(1,y,  nz+1);
    fy = &f(1,y-1,nz+1);
    fz = &f(1,y,  nz);
    for( x=1; x<=nx; x++ ) {
      UPDATE_EX();
      f0++;
      fy++;
      fz++;
    }
  }
  for( z=2; z<=nz; z++ ) {
    f0 = &f(1,1,z);
    fy = &f(1,0,z);
    fz = &f(1,1,z-1);
    for( x=1; x<=nx; x++ ) {
      UPDATE_EX();
      f0++;
      fy++;
      fz++;
    }
  }
  for( z=2; z<=nz; z++ ) {
    f0 = &f(1,ny+1,z);
    fy = &f(1,ny,  z);
    fz = &f(1,ny+1,z-1);
    for( x=1; x<=nx; x++ ) {
      UPDATE_EX();
      f0++;
      fy++;
      fz++;
    }
  }

  // Do exterior ey
  for( z=1; z<=nz+1; z++ ) {
    for( y=1; y<=ny; y++ ) {
      f0 = &f(1,y,z);
      fx = &f(0,y,z);
      fz = &f(1,y,z-1);
      UPDATE_EY();
    }
  }
  for( z=1; z<=nz+1; z++ ) {
    for( y=1; y<=ny; y++ ) {
      f0 = &f(nx+1,y,z);
      fx = &f(nx,  y,z);
      fz = &f(nx+1,y,z-1);
      UPDATE_EY();
    }
  }
  for( y=1; y<=ny; y++ ) {
    f0 = &f(2,y,1);
    fx = &f(1,y,1);
    fz = &f(2,y,0);
    for( x=2; x<=nx; x++ ) {
      UPDATE_EY();
      f0++;
      fx++;
      fz++;
    }
  }
  for( y=1; y<=ny; y++ ) {
    f0 = &f(2,y,nz+1);
    fx = &f(1,y,nz+1);
    fz = &f(2,y,nz  );
    for( x=2; x<=nx; x++ ) {
      UPDATE_EY();
      f0++;
      fx++;
      fz++;
    }
  }

  // Do exterior ez
  for( z=1; z<=nz; z++ ) {
    f0 = &f(1,1,z);
    fx = &f(0,1,z);
    fy = &f(1,0,z);
    for( x=1; x<=nx+1; x++ ) {
      UPDATE_EZ();
      f0++;
      fx++;
      fy++;
    }
  }
  for( z=1; z<=nz; z++ ) {
    f0 = &f(1,ny+1,z);
    fx = &f(0,ny+1,z);
    fy = &f(1,ny,  z);
    for( x=1; x<=nx+1; x++ ) {
      UPDATE_EZ();
      f0++;
      fx++;
      fy++;
    }
  }
  for( z=1; z<=nz; z++ ) {
    for( y=2; y<=ny; y++ ) {
      f0 = &f(1,y,  z);
      fx = &f(0,y,  z);
      fy = &f(1,y-1,z);
      UPDATE_EZ();
    }
  }
  for( z=1; z<=nz; z++ ) {
    for( y=2; y<=ny; y++ ) {
      f0 = &f(nx+1,y,  z);
      fx = &f(nx,  y,  z);
      fy = &f(nx+1,y-1,z);
      UPDATE_EZ();
    }
  }

  local_adjust_tang_e( fa->f, fa->g );
}

void vacuum_advance_e_interior_kokkos(k_field_t& k_field, 
                                const size_t nx, const size_t ny, const size_t nz,
                                const float px_muy, const float px_muz, const float py_mux, const float py_muz, const float pz_mux, const float pz_muy,
                                const float damp, const float decayx, const float decayy, const float decayz, const float drivex, const float drivey, const float drivez, const float cj) {
    
    // EXEC_PIPELINE
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({2, 2, 2}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_for("vacuum_advance_e: Majority of interior", zyx_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
        const int f0 = VOXEL(x,   y,   z,   nx, ny, nz);
        const int fx = VOXEL(x-1, y,   z,   nx, ny, nz);
        const int fy = VOXEL(x,   y-1, z,   nx, ny, nz);
        const int fz = VOXEL(x,   y,   z-1, nx, ny, nz);
        update_ex(k_field, f0, fx, fy, fz, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayx, drivex, cj);
        update_ey(k_field, f0, fx, fy, fz, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayy, drivey, cj);
        update_ez(k_field, f0, fx, fy, fz, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayz, drivez, cj);
    });
    
  // Do left over interior ex
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_policy({2, 2}, {nz+1, ny+1});
    Kokkos::parallel_for("vacuum_advance_e: left over interior ex", ex_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_idx = VOXEL(1, y,   z, nx, ny, nz);
        const size_t fx_idx = 0;
        const size_t fy_idx = VOXEL(1, y-1, z, nx, ny, nz);
        const size_t fz_idx = VOXEL(1, y,   z-1, nx, ny ,nz);
        update_ex(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayx, drivex, cj);
    });
  
  // Do left over interior ey
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_policy({2, 2}, {nz+1, nx+1});
    Kokkos::parallel_for("vacuum_advance_e: left over interior ey", ey_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(2, 1, z, nx, ny, nz) + (x-2);
        const size_t fx_idx = VOXEL(1, 1, z, nx, ny, nz) + (x-2);
        const size_t fy_idx = 0;
        const size_t fz_idx = VOXEL(2, 1, z-1, nx, ny ,nz) + (x-2);
        update_ey(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayy, drivey, cj);
    });
  
  // Do left over interior ez
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_policy({2, 2}, {ny+1, nx+1});
    Kokkos::parallel_for("vacuum_advance_e: left over interior ez", ez_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_idx = VOXEL(2, y,   1, nx, ny, nz) + (x-2);
        const size_t fx_idx = VOXEL(1, y,   1, nx, ny, nz) + (x-2);
        const size_t fy_idx = VOXEL(2, y-1, 1, nx, ny, nz) + (x-2);
        const size_t fz_idx = 0;
        update_ez(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayz, drivez, cj);
    });

}

void vacuum_advance_e_exterior_kokkos(k_field_t& k_field, 
                                const size_t nx, const size_t ny, const size_t nz,
                                const float px_muy, const float px_muz, const float py_mux, const float py_muz, const float pz_mux, const float pz_muy,
                                const float damp, const float decayx, const float decayy, const float decayz, const float drivex, const float drivey, const float drivez, const float cj) {
  // Do exterior ex
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_yx_policy({1, 1}, {ny+2, nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_zx_policy({2, 1}, {nz+1, nx+1});
    Kokkos::parallel_for("vacuum_advance_e: exterior ex loop 1", ex_yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_idx = VOXEL(1, y,   1,nx,ny,nz) + (x-1);
        const size_t fx_idx = 0; 
        const size_t fy_idx = VOXEL(1, y-1, 1,nx,ny,nz) + (x-1);
        const size_t fz_idx = VOXEL(1, y,   0,nx,ny,nz) + (x-1);
        update_ex(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayx, drivex, cj);
    });
    Kokkos::parallel_for("vacuum_advance_e: exterior ex loop 2", ex_yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_idx = VOXEL(1,y,  nz+1, nx,ny,nz) + (x-1);
        const size_t fx_idx = 0; 
        const size_t fy_idx = VOXEL(1,y-1,nz+1, nx,ny,nz) + (x-1);
        const size_t fz_idx = VOXEL(1,y,  nz,   nx,ny,nz) + (x-1);
        update_ex(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayx, drivex, cj);
    });
    Kokkos::parallel_for("vacuum_advance_e: exterior ex loop 3", ex_zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(1,1,z,nx,ny,nz) + (x-1);
        const size_t fx_idx = 0; // Don't care about x index, not used in update_ex anyway.
        const size_t fy_idx = VOXEL(1,0,z,nx,ny,nz) + (x-1);
        const size_t fz_idx = VOXEL(1,1,z-1,nx,ny,nz) + (x-1);
        update_ex(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayx, drivex, cj);
    });
    Kokkos::parallel_for("vacuum_advance_e: exterior ex loop 4", ex_zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(1,ny+1,z,nx,ny,nz) + (x-1);
        const size_t fx_idx = 0; // Don't care about x index, not used in update_ex anyway.
        const size_t fy_idx = VOXEL(1,ny,z,nx,ny,nz) + (x-1);
        const size_t fz_idx = VOXEL(1,ny+1,z-1,nx,ny,nz) + (x-1);
        update_ex(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayx, drivex, cj);
    });
  
  // Do exterior ey
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_zy_policy({1, 1}, {nz+2, ny+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_yx_policy({1, 2}, {ny+1, nx+1});
    Kokkos::parallel_for("vacuum_advance_e: exterior ey loop 1", ey_zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
            const size_t f0_idx = VOXEL(1,y,z,nx,ny,nz);
            const size_t fx_idx = VOXEL(0,y,z,nx,ny,nz);
            const size_t fy_idx = 0;
            const size_t fz_idx = VOXEL(1,y,z-1,nx,ny,nz);
            update_ey(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayy, drivey, cj);
    });
    Kokkos::parallel_for("vacuum_advance_e: exterior ey loop 2", ey_zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
            const size_t f0_idx = VOXEL(nx+1,y,z,nx,ny,nz);
            const size_t fx_idx = VOXEL(nx,y,z,nx,ny,nz);
            const size_t fy_idx = 0;
            const size_t fz_idx = VOXEL(nx+1,y,z-1,nx,ny,nz);
            update_ey(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayy, drivey, cj);
    });
    Kokkos::parallel_for("vacuum_advance_e: exterior ey loop 3", ey_yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
            const size_t f0_idx = VOXEL(2,y,1,nx,ny,nz) + (x-2);
            const size_t fx_idx = VOXEL(1,y,1,nx,ny,nz) + (x-2);
            const size_t fy_idx = 0;
            const size_t fz_idx = VOXEL(2,y,0,nx,ny,nz) + (x-2);
            update_ey(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayy, drivey, cj);
    });
    Kokkos::parallel_for("vacuum_advance_e: exterior ey loop 4", ey_yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
            const size_t f0_idx = VOXEL(2,y,nz+1,nx,ny,nz) + (x-2);
            const size_t fx_idx = VOXEL(1,y,nz+1,nx,ny,nz) + (x-2);
            const size_t fy_idx = 0;
            const size_t fz_idx = VOXEL(2,y,nz,nx,ny,nz) + (x-2);
            update_ey(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayy, drivey, cj);
    });

  // Do exterior ez
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zx_policy({1, 1}, {nz+1, nx+2});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zy_policy({1, 2}, {nz+1, ny+1});
    Kokkos::parallel_for("vacuum_advance_e: exterior ez loop 1", ez_zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(1,1,z,nx,ny,nz) + (x-1);
        const size_t fx_idx = VOXEL(0,1,z,nx,ny,nz) + (x-1);
        const size_t fy_idx = VOXEL(1,0,z,nx,ny,nz) + (x-1);
        const size_t fz_idx = 0;
        update_ez(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayz, drivez, cj);
    });
    Kokkos::parallel_for("vacuum_advance_e: exterior ez loop 2", ez_zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(1,ny+1,z,nx,ny,nz) + (x-1);
        const size_t fx_idx = VOXEL(0,ny+1,z,nx,ny,nz) + (x-1);
        const size_t fy_idx = VOXEL(1,ny  ,z,nx,ny,nz) + (x-1);
        const size_t fz_idx = 0;
        update_ez(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayz, drivez, cj);
    });
    Kokkos::parallel_for("vacuum_advance_e: exterior ez loop 3", ez_zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_idx = VOXEL(1,y,z,nx,ny,nz);
        const size_t fx_idx = VOXEL(0,y,z,nx,ny,nz);
        const size_t fy_idx = VOXEL(1,y-1,z,nx,ny,nz);
        const size_t fz_idx = 0;
        update_ez(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayz, drivez, cj);
    });
    Kokkos::parallel_for("vacuum_advance_e: exterior ez loop 4", ez_zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_idx = VOXEL(nx+1, y,   z,nx,ny,nz);
        const size_t fx_idx = VOXEL(nx,   y,   z,nx,ny,nz);
        const size_t fy_idx = VOXEL(nx+1, y-1, z,nx,ny,nz);
        const size_t fz_idx = 0;
        update_ez(k_field, f0_idx, fx_idx, fy_idx, fz_idx, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayz, drivez, cj);
    });

}

void
vacuum_advance_e_kokkos( field_array_t * RESTRICT fa,
                  float frac ) {
  if( !fa     ) ERROR(( "Bad args" ));
  if( frac!=1 ) ERROR(( "standard advance_e does not support frac!=1 yet" ));

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  k_field_t k_field = fa->k_f_d;
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;               
  const grid_t                 *              g = args->g;                   
  const int nx = g->nx, ny = g->ny, nz = g->nz;                              
                                                                             
  const float decayx = m->decayx, drivex = m->drivex;                        
  const float decayy = m->decayy, drivey = m->drivey;                        
  const float decayz = m->decayz, drivez = m->drivez;                        
  const float damp   = args->p->damp;                                        
  const float px_muz = ((nx>1) ? (1+damp)*g->cvac*g->dt*g->rdx : 0)*m->rmuz; 
  const float px_muy = ((nx>1) ? (1+damp)*g->cvac*g->dt*g->rdx : 0)*m->rmuy; 
  const float py_mux = ((ny>1) ? (1+damp)*g->cvac*g->dt*g->rdy : 0)*m->rmux; 
  const float py_muz = ((ny>1) ? (1+damp)*g->cvac*g->dt*g->rdy : 0)*m->rmuz; 
  const float pz_muy = ((nz>1) ? (1+damp)*g->cvac*g->dt*g->rdz : 0)*m->rmuy; 
  const float pz_mux = ((nz>1) ? (1+damp)*g->cvac*g->dt*g->rdz : 0)*m->rmux; 
  const float cj     = g->dt/g->eps0;                                        

  /***************************************************************************
   * Begin tangential B ghost setup
   ***************************************************************************/
  
//    k_begin_remote_ghost_tang_b( fa, fa->g );
    kokkos_begin_remote_ghost_tang_b(fa, fa->g, fa->fb);

    k_local_ghost_tang_b( fa, fa->g );

    vacuum_advance_e_interior_kokkos(k_field, nx, ny, nz, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayx, decayy, decayz, drivex, drivey, drivez, cj);
  
  /***************************************************************************
   * Finish tangential B ghost setup
   ***************************************************************************/

//    k_end_remote_ghost_tang_b( fa, fa->g );
    kokkos_end_remote_ghost_tang_b(fa, fa->g, fa->fb);

  /***************************************************************************
   * Update exterior fields
   ***************************************************************************/

    vacuum_advance_e_exterior_kokkos(k_field, nx, ny, nz, px_muy, px_muz, py_mux, py_muz, pz_mux, pz_muy, damp, decayx, decayy, decayz, drivex, drivey, drivez, cj);

    k_local_adjust_tang_e( fa, fa->g );
}
