#define IN_sfa
#include "sfa_private.h"

typedef struct pipeline_args {
  field_t            * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
} pipeline_args_t;

#define DECLARE_STENCIL()                                                \
  field_t                      * ALIGNED(128) f = args->f;               \
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;           \
  const grid_t                 *              g = args->g;               \
  const int nx = g->nx, ny = g->ny, nz = g->nz;                          \
                                                                         \
  const float _rdx = (nx>1) ? g->rdx : 0;                                \
  const float _rdy = (ny>1) ? g->rdy : 0;                                \
  const float _rdz = (nz>1) ? g->rdz : 0;                                \
  const float alphadt = 0.3888889/( _rdx*_rdx + _rdy*_rdy + _rdz*_rdz ); \
  const float px   = (alphadt*_rdx)*m->drivex;                           \
  const float py   = (alphadt*_rdy)*m->drivey;                           \
  const float pz   = (alphadt*_rdz)*m->drivez;                           \
                                                                         \
  field_t * ALIGNED(16) f0;                                              \
  field_t * ALIGNED(16) fx, * ALIGNED(16) fy, * ALIGNED(16) fz;          \
  int x, y, z
                     
#define f(x,y,z) f[ VOXEL(x,y,z,nx,ny,nz) ]

#define INIT_STENCIL()  \
  f0 = &f(x,  y,  z  ); \
  fx = &f(x+1,y,  z  ); \
  fy = &f(x,  y+1,z  ); \
  fz = &f(x,  y,  z+1)

#define NEXT_STENCIL()                \
  f0++; fx++; fy++; fz++; x++;        \
  if( x>nx ) {                        \
    /**/       y++;            x = 1; \
    if( y>ny ) z++; if( y>ny ) y = 1; \
    INIT_STENCIL();                   \
  }

#define MARDER_EX() f0->ex += px*(fx->div_e_err-f0->div_e_err)
#define MARDER_EY() f0->ey += py*(fy->div_e_err-f0->div_e_err)
#define MARDER_EZ() f0->ez += pz*(fz->div_e_err-f0->div_e_err)

KOKKOS_INLINE_FUNCTION void marder_ex(const k_field_t& k_field, const float px, const int f0, const int fx) {
    k_field(f0, field_var::ex) += px*(k_field(fx, field_var::div_e_err) - k_field(f0, field_var::div_e_err));
}
KOKKOS_INLINE_FUNCTION void marder_ey(const k_field_t& k_field, const float py, const int f0, const int fy) {
    k_field(f0, field_var::ey) += py*(k_field(fy, field_var::div_e_err) - k_field(f0, field_var::div_e_err));
}
KOKKOS_INLINE_FUNCTION void marder_ez(const k_field_t& k_field, const float pz, const int f0, const int fz) {
    k_field(f0, field_var::ez) += pz*(k_field(fz, field_var::div_e_err) - k_field(f0, field_var::div_e_err));
}

static void
vacuum_clean_div_e_pipeline( pipeline_args_t * args,
                             int pipeline_rank,
                             int n_pipeline ) {
  DECLARE_STENCIL();
  
  int n_voxel;
  DISTRIBUTE_VOXELS( 1,nx, 1,ny, 1,nz, 16,
                     pipeline_rank, n_pipeline,
                     x, y, z, n_voxel );

  INIT_STENCIL();
  for( ; n_voxel; n_voxel-- ) {
    MARDER_EX(); MARDER_EY(); MARDER_EZ();
    NEXT_STENCIL();
  }
}

#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

#error "Not implemented"

#endif

void
vacuum_clean_div_e_host( field_array_t * fa ) {
  if( !fa ) ERROR(( "Bad args" ));

  // Do majority of field components in single pass on the pipelines.
  // The host handles stragglers.

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  EXEC_PIPELINES( vacuum_clean_div_e, args, 0 );

  // While pipelines are busy, do left overs on the host

  DECLARE_STENCIL();
  
  // Do left over ex
  for( y=1; y<=ny+1; y++ ) {
    f0 = &f(1,y,nz+1);
    fx = &f(2,y,nz+1);
    for( x=1; x<=nx; x++ ) {
      MARDER_EX();
      f0++; fx++;
    }
  }
  for( z=1; z<=nz; z++ ) {
    f0 = &f(1,ny+1,z);
    fx = &f(2,ny+1,z);
    for( x=1; x<=nx; x++ ) {
      MARDER_EX();
      f0++; fx++;
    }
  }

  // Do left over ey
  for( z=1; z<=nz+1; z++ ) {
    for( y=1; y<=ny; y++ ) {
      f0 = &f(nx+1,y,  z);
      fy = &f(nx+1,y+1,z);
      MARDER_EY();
    }
  }
  for( y=1; y<=ny; y++ ) {
    f0 = &f(1,y,  nz+1);
    fy = &f(1,y+1,nz+1);
    for( x=1; x<=nx; x++ ) {
      MARDER_EY();
      f0++; fy++;
    }
  }

  // Do left over ez
  for( z=1; z<=nz; z++ ) {
    f0 = &f(1,ny+1,z);
    fz = &f(1,ny+1,z+1);
    for( x=1; x<=nx+1; x++ ) {
      MARDER_EZ();
      f0++; fz++;
    }
  }
  for( z=1; z<=nz; z++ ) {
    for( y=1; y<=ny; y++ ) {
      f0 = &f(nx+1,y,z);
      fz = &f(nx+1,y,z+1);
      MARDER_EZ();
    }
  }

  WAIT_PIPELINES();

  local_adjust_tang_e( fa->f, fa->g );
}

void
vacuum_clean_div_e( field_array_t * fa ) {
  if( !fa ) ERROR(( "Bad args" ));

  // Do majority of field components in single pass on the pipelines.
  // The host handles stragglers.

    const k_field_t& k_field = fa->k_f_d;
    sfa_params_t* sfa = reinterpret_cast<sfa_params_t *>(fa->params);
    const k_material_coefficient_t::HostMirror& k_mat = sfa->k_mc_h;
    const grid_t* g = fa->g;
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const float _rdx = (nx>1) ? g->rdx : 0;
    const float _rdy = (ny>1) ? g->rdy : 0;
    const float _rdz = (nz>1) ? g->rdz : 0;
    const float alphadt = 0.3888889/( _rdx*_rdx + _rdy*_rdy + _rdz*_rdz );
    const float px   = (alphadt*_rdx)*k_mat(0, material_coeff_var::drivex); 
    const float py   = (alphadt*_rdy)*k_mat(0, material_coeff_var::drivey); 
    const float pz   = (alphadt*_rdz)*k_mat(0, material_coeff_var::drivez); 

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1,1,1}, {nz+1,ny+1,nx+1});
    Kokkos::parallel_for("vacuum_clean_div_e main body", zyx_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
        const int f0 = VOXEL(x,   y,   z, nx, ny, nz);       
        const int fx = VOXEL(x+1, y,   z, nx, ny, nz);       
        const int fy = VOXEL(x,   y+1, z, nx, ny, nz);       
        const int fz = VOXEL(x,   y,   z+1, nx, ny, nz);       
        marder_ex(k_field, px, f0, fx);
        marder_ey(k_field, py, f0, fy);
        marder_ez(k_field, pz, f0, fz);
    });

  // While pipelines are busy, do left overs on the host

  // Do left over ex
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_yx_pol({1, 1}, {ny+2, nx+1});
    Kokkos::parallel_for("vacuum_clean_div_e: left over ex: yx loop", ex_yx_pol, KOKKOS_LAMBDA(const int y, const int x) {
        const int f0 = VOXEL(1, y, nz+1, nx, ny, nz) + (x-1);
        const int fx = VOXEL(2, y, nz+1, nx, ny, nz) + (x-1);
        marder_ex(k_field, px, f0, fx);
    });
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_zx_pol({1, 1}, {nz+1, nx+1});
    Kokkos::parallel_for("vacuum_clean_div_e: left over ex: zx loop", ex_zx_pol, KOKKOS_LAMBDA(const int z, const int x) {
        const int f0 = VOXEL(1, ny+1, z, nx, ny, nz) + (x-1);
        const int fx = VOXEL(2, ny+1, z, nx, ny, nz) + (x-1);
        marder_ex(k_field, px, f0, fx);
    });

  // Do left over ey
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_zy_pol({1, 1}, {nz+2, ny+1});
    Kokkos::parallel_for("vacuum_clean_div_e: left over ey: zy loop", ey_zy_pol, KOKKOS_LAMBDA(const int z, const int y) {
        const int f0 = VOXEL(nx+1, y,   z, nx, ny, nz);
        const int fy = VOXEL(nx+1, y+1, z, nx, ny, nz);
        marder_ey(k_field, py, f0, fy);
    });
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_yx_pol({1, 1}, {ny+1, nx+1});
    Kokkos::parallel_for("vacuum_clean_div_e: left over ey: yx loop", ey_yx_pol, KOKKOS_LAMBDA(const int y, const int x) {
        const int f0 = VOXEL(1, y,   nz+1, nx, ny, nz) + (x-1);
        const int fy = VOXEL(1, y+1, nz+1, nx, ny, nz) + (x-1);
        marder_ey(k_field, py, f0, fy);
    });
  
  // Do left over ez
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zx_pol({1, 1}, {nz+1, nx+2});
    Kokkos::parallel_for("vacuum_clean_div_e: left over ez: zx loop", ez_zx_pol, KOKKOS_LAMBDA(const int z, const int x) {
        const int f0 = VOXEL(1, ny+1, z,   nx, ny, nz) + (x-1);
        const int fz = VOXEL(1, ny+1, z+1, nx, ny, nz) + (x-1);
        marder_ez(k_field, pz, f0, fz);
    });
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zy_pol({1, 1}, {nz+1, ny+1});
    Kokkos::parallel_for("vacuum_clean_div_e: left over ez: zy loop", ez_zy_pol, KOKKOS_LAMBDA(const int z, const int y) {
        const int f0 = VOXEL(nx+1, y, z,   nx, ny, nz);
        const int fz = VOXEL(nx+1, y, z+1, nx, ny, nz);
        marder_ez(k_field, pz, f0, fz);
    });

  k_local_adjust_tang_e( fa, fa->g );
}

