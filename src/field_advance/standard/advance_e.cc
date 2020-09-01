// Note: This is similar to compute_curl_b

#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"
#include <Kokkos_Core.hpp>

typedef struct pipeline_args {
  field_t            * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
} pipeline_args_t;

#define DECLARE_STENCIL()                                        \
  /**/  field_t                * ALIGNED(128) f = args->f;       \
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;   \
  const grid_t                 *              g = args->g;       \
  const int nx = g->nx, ny = g->ny, nz = g->nz;                  \
                                                                 \
  const float damp = args->p->damp;                              \
  const float px   = (nx>1) ? (1+damp)*g->cvac*g->dt*g->rdx : 0; \
  const float py   = (ny>1) ? (1+damp)*g->cvac*g->dt*g->rdy : 0; \
  const float pz   = (nz>1) ? (1+damp)*g->cvac*g->dt*g->rdz : 0; \
  const float cj   = g->dt/g->eps0;                              \
                                                                 \
  field_t * ALIGNED(16) f0;                                      \
  field_t * ALIGNED(16) fx, * ALIGNED(16) fy, * ALIGNED(16) fz;  \
  int x, y, z

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

#define INIT_STENCIL()  \
  f0 = &f(x,  y,  z  ); \
  fx = &f(x-1,y,  z  ); \
  fy = &f(x,  y-1,z  ); \
  fz = &f(x,  y,  z-1)

#define NEXT_STENCIL()                \
  f0++; fx++;	fy++; fz++; x++;      \
  if( x>nx ) {                        \
    /**/       y++;            x = 2; \
    if( y>ny ) z++; if( y>ny ) y = 2; \
    INIT_STENCIL();                   \
  }

#define UPDATE_EX()						            \
  f0->tcax = ( py*(f0->cbz*m[f0->fmatz].rmuz-fy->cbz*m[fy->fmatz].rmuz) -   \
               pz*(f0->cby*m[f0->fmaty].rmuy-fz->cby*m[fz->fmaty].rmuy) ) - \
             damp*f0->tcax;                                                 \
  f0->ex   = m[f0->ematx].decayx*f0->ex +                                   \
             m[f0->ematx].drivex*( f0->tcax - cj*f0->jfx )
#define UPDATE_EY()						            \
  f0->tcay = ( pz*(f0->cbx*m[f0->fmatx].rmux-fz->cbx*m[fz->fmatx].rmux) -   \
               px*(f0->cbz*m[f0->fmatz].rmuz-fx->cbz*m[fx->fmatz].rmuz) ) - \
             damp*f0->tcay;                                                 \
  f0->ey   = m[f0->ematy].decayy*f0->ey +                                   \
             m[f0->ematy].drivey*( f0->tcay - cj*f0->jfy )
#define UPDATE_EZ()						            \
  f0->tcaz = ( px*(f0->cby*m[f0->fmaty].rmuy-fx->cby*m[fx->fmaty].rmuy) -   \
               py*(f0->cbx*m[f0->fmatx].rmux-fy->cbx*m[fy->fmatx].rmux) ) - \
             damp*f0->tcaz;                                                 \
  f0->ez   = m[f0->ematz].decayz*f0->ez +                                   \
             m[f0->ematz].drivez*( f0->tcaz - cj*f0->jfz )

KOKKOS_INLINE_FUNCTION void update_ex(const k_field_t& k_field, const k_field_edge_t& k_field_edge, const k_material_coefficient_t& k_material,
                const float damp, const float cj, size_t f0_idx,
                size_t fx_idx, size_t fy_idx, size_t fz_idx,
                const float px, const float py, const float pz) {
    float f0_ex     = k_field(f0_idx, field_var::ex);
    float f0_cby    = k_field(f0_idx, field_var::cby);
    float f0_cbz    = k_field(f0_idx, field_var::cbz);
    float f0_tcax   = k_field(f0_idx, field_var::tcax);
    float f0_jfx    = k_field(f0_idx, field_var::jfx);
    float fy_cbz    = k_field(fy_idx, field_var::cbz);
    float fz_cby    = k_field(fz_idx, field_var::cby);
    material_id f0_fmaty  = k_field_edge(f0_idx, field_edge_var::fmaty);
    material_id f0_fmatz  = k_field_edge(f0_idx, field_edge_var::fmatz);
    material_id f0_ematx  = k_field_edge(f0_idx, field_edge_var::ematx);
    material_id fy_fmatz  = k_field_edge(fy_idx, field_edge_var::fmatz);
    material_id fz_fmaty  = k_field_edge(fz_idx, field_edge_var::fmaty);

    k_field(f0_idx, field_var::tcax) = (py * (f0_cbz * k_material(f0_fmatz, material_coeff_var::rmuz) -
                                              fy_cbz * k_material(fy_fmatz, material_coeff_var::rmuz)) -
                                        pz * (f0_cby * k_material(f0_fmaty, material_coeff_var::rmuy) -
                                              fz_cby * k_material(fz_fmaty, material_coeff_var::rmuy)) ) - damp * f0_tcax;

    k_field(f0_idx, field_var::ex) = k_material(f0_ematx, material_coeff_var::decayx) * f0_ex +
                                    k_material(f0_ematx, material_coeff_var::drivex) * (k_field(f0_idx, field_var::tcax) - cj * f0_jfx);
}

KOKKOS_INLINE_FUNCTION void update_ey(const k_field_t& k_field, const k_field_edge_t& k_field_edge, const k_material_coefficient_t& k_material,
                const float damp, const float cj, size_t f0_idx,
                size_t fx_idx, size_t fy_idx, size_t fz_idx,
                const float px, const float py, const float pz) {
    float f0_ey     = k_field(f0_idx, field_var::ey);
    float f0_cbx    = k_field(f0_idx, field_var::cbx);
    float f0_cbz    = k_field(f0_idx, field_var::cbz);
    float f0_tcay   = k_field(f0_idx, field_var::tcay);
    float f0_jfy    = k_field(f0_idx, field_var::jfy);
    float fx_cbz    = k_field(fx_idx, field_var::cbz);
    float fz_cbx    = k_field(fz_idx, field_var::cbx);
    material_id f0_fmatx  = k_field_edge(f0_idx, field_edge_var::fmatx);
    material_id f0_fmatz  = k_field_edge(f0_idx, field_edge_var::fmatz);
    material_id f0_ematy  = k_field_edge(f0_idx, field_edge_var::ematy);
    material_id fx_fmatz  = k_field_edge(fx_idx, field_edge_var::fmatz);
    material_id fz_fmatx  = k_field_edge(fz_idx, field_edge_var::fmatx);

    k_field(f0_idx, field_var::tcay) = (pz * (f0_cbx * k_material(f0_fmatx, material_coeff_var::rmux) -
                                              fz_cbx * k_material(fz_fmatx, material_coeff_var::rmux)) -
                                        px * (f0_cbz * k_material(f0_fmatz, material_coeff_var::rmuz) -
                                              fx_cbz * k_material(fx_fmatz, material_coeff_var::rmuz))) -
                                        damp * f0_tcay;

    k_field(f0_idx, field_var::ey) = k_material(f0_ematy, material_coeff_var::decayy) * f0_ey +
                                     k_material(f0_ematy, material_coeff_var::drivey) * (k_field(f0_idx, field_var::tcay) - cj * f0_jfy);
}

KOKKOS_INLINE_FUNCTION void update_ez(const k_field_t& k_field, const k_field_edge_t& k_field_edge, const k_material_coefficient_t& k_material,
                const float damp, const float cj, size_t f0_idx,
                size_t fx_idx, size_t fy_idx, size_t fz_idx,
                const float px, const float py, const float pz) {
    float f0_ez     = k_field(f0_idx, field_var::ez);
    float f0_cbx    = k_field(f0_idx, field_var::cbx);
    float f0_cby    = k_field(f0_idx, field_var::cby);
    float f0_tcaz   = k_field(f0_idx, field_var::tcaz);
    float f0_jfz    = k_field(f0_idx, field_var::jfz);
    float fx_cby    = k_field(fx_idx, field_var::cby);
    float fy_cbx    = k_field(fy_idx, field_var::cbx);
    material_id f0_fmatx  = k_field_edge(f0_idx, field_edge_var::fmatx);
    material_id f0_fmaty  = k_field_edge(f0_idx, field_edge_var::fmaty);
    material_id f0_ematz  = k_field_edge(f0_idx, field_edge_var::ematz);
    material_id fx_fmaty  = k_field_edge(fx_idx, field_edge_var::fmaty);
    material_id fy_fmatx  = k_field_edge(fy_idx, field_edge_var::fmatx);


    k_field(f0_idx, field_var::tcaz) = (px * (f0_cby * k_material(f0_fmaty, material_coeff_var::rmuy) -
                                              fx_cby * k_material(fx_fmaty, material_coeff_var::rmuy)) -
                                        py * (f0_cbx * k_material(f0_fmatx, material_coeff_var::rmux) -
                                              fy_cbx * k_material(fy_fmatx, material_coeff_var::rmux))) -
                                        damp * f0_tcaz;

    k_field(f0_idx, field_var::ez) = k_material(f0_ematz, material_coeff_var::decayz) * f0_ez +
                                     k_material(f0_ematz, material_coeff_var::drivez) * (k_field(f0_idx, field_var::tcaz) - cj * f0_jfz);
}

void advance_e_interior_kokkos(k_field_t& k_field, k_field_edge_t& k_field_edge,
                                const k_material_coefficient_t&  k_material,
                                const size_t nx, const size_t ny, const size_t nz,
                                const float px, const float py, const float pz,
                                const float damp, const float cj) {

    // EXEC_PIPELINE
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({2, 2, 2}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_for("vacuum_advance_e: Majority of interior", zyx_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
        const int f0 = VOXEL(x,   y,   z,   nx, ny, nz);
        const int fx = VOXEL(x-1, y,   z,   nx, ny, nz);
        const int fy = VOXEL(x,   y-1, z,   nx, ny, nz);
        const int fz = VOXEL(x,   y,   z-1, nx, ny, nz);
        update_ex(k_field, k_field_edge, k_material, damp, cj, f0, fx, fy, fz, px, py, pz);
        update_ey(k_field, k_field_edge, k_material, damp, cj, f0, fx, fy, fz, px, py, pz);
        update_ez(k_field, k_field_edge, k_material, damp, cj, f0, fx, fy, fz, px, py, pz);
    });

  // Do left over interior ex
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_policy({2, 2}, {nz+1, ny+1});
    Kokkos::parallel_for("vacuum_advance_e: left over interior ex", ex_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_idx = VOXEL(1, y,   z, nx, ny, nz);
        const size_t fx_idx = 0;
        const size_t fy_idx = VOXEL(1, y-1, z, nx, ny, nz);
        const size_t fz_idx = VOXEL(1, y,   z-1, nx, ny ,nz);
        update_ex(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });

  // Do left over interior ey
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_policy({2, 2}, {nz+1, nx+1});
    Kokkos::parallel_for("vacuum_advance_e: left over interior ey", ey_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(2, 1, z, nx, ny, nz) + (x-2);
        const size_t fx_idx = VOXEL(1, 1, z, nx, ny, nz) + (x-2);
        const size_t fy_idx = 0;
        const size_t fz_idx = VOXEL(2, 1, z-1, nx, ny ,nz) + (x-2);
        update_ey(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });

  // Do left over interior ez
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_policy({2, 2}, {ny+1, nx+1});
    Kokkos::parallel_for("vacuum_advance_e: left over interior ez", ez_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_idx = VOXEL(2, y,   1, nx, ny, nz) + (x-2);
        const size_t fx_idx = VOXEL(1, y,   1, nx, ny, nz) + (x-2);
        const size_t fy_idx = VOXEL(2, y-1, 1, nx, ny, nz) + (x-2);
        const size_t fz_idx = 0;
        update_ez(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
}

void advance_e_exterior_kokkos(k_field_t& k_field, k_field_edge_t& k_field_edge,
                                const k_material_coefficient_t& k_material,
                                const size_t nx, const size_t ny, const size_t nz,
                                const float px, const float py, const float pz,
                                const float damp, const float cj) {
  // Do exterior ex
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_yx_policy({1, 1}, {ny+2, nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_zx_policy({2, 1}, {nz+1, nx+1});
    Kokkos::parallel_for("advance_e: exterior ex loop 1", ex_yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_idx = VOXEL(1, y,   1,nx,ny,nz) + (x-1);
        const size_t fx_idx = 0;
        const size_t fy_idx = VOXEL(1, y-1, 1,nx,ny,nz) + (x-1);
        const size_t fz_idx = VOXEL(1, y,   0,nx,ny,nz) + (x-1);
        update_ex(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
    Kokkos::parallel_for("advance_e: exterior ex loop 2", ex_yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_idx = VOXEL(1,y,  nz+1, nx,ny,nz) + (x-1);
        const size_t fx_idx = 0;
        const size_t fy_idx = VOXEL(1,y-1,nz+1, nx,ny,nz) + (x-1);
        const size_t fz_idx = VOXEL(1,y,  nz,   nx,ny,nz) + (x-1);
        update_ex(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
    Kokkos::parallel_for("advance_e: exterior ex loop 3", ex_zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(1,1,z,nx,ny,nz) + (x-1);
        const size_t fx_idx = 0; // Don't care about x index, not used in update_ex anyway.
        const size_t fy_idx = VOXEL(1,0,z,nx,ny,nz) + (x-1);
        const size_t fz_idx = VOXEL(1,1,z-1,nx,ny,nz) + (x-1);
        update_ex(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
    Kokkos::parallel_for("advance_e: exterior ex loop 4", ex_zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(1,ny+1,z,nx,ny,nz) + (x-1);
        const size_t fx_idx = 0; // Don't care about x index, not used in update_ex anyway.
        const size_t fy_idx = VOXEL(1,ny,z,nx,ny,nz) + (x-1);
        const size_t fz_idx = VOXEL(1,ny+1,z-1,nx,ny,nz) + (x-1);
        update_ex(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });

  // Do exterior ey
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_zy_policy({1, 1}, {nz+2, ny+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_yx_policy({1, 2}, {ny+1, nx+1});
    Kokkos::parallel_for("advance_e: exterior ey loop 1", ey_zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_idx = VOXEL(1,y,z,nx,ny,nz);
        const size_t fx_idx = VOXEL(0,y,z,nx,ny,nz);
        const size_t fy_idx = 0;
        const size_t fz_idx = VOXEL(1,y,z-1,nx,ny,nz);
        update_ey(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
    Kokkos::parallel_for("advance_e: exterior ey loop 2", ey_zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_idx = VOXEL(nx+1,y,z,nx,ny,nz);
        const size_t fx_idx = VOXEL(nx,y,z,nx,ny,nz);
        const size_t fy_idx = 0;
        const size_t fz_idx = VOXEL(nx+1,y,z-1,nx,ny,nz);
        update_ey(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
    Kokkos::parallel_for("advance_e: exterior ey loop 3", ey_yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_idx = VOXEL(2,y,1,nx,ny,nz) + (x-2);
        const size_t fx_idx = VOXEL(1,y,1,nx,ny,nz) + (x-2);
        const size_t fy_idx = 0;
        const size_t fz_idx = VOXEL(2,y,0,nx,ny,nz) + (x-2);
        update_ey(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
    Kokkos::parallel_for("advance_e: exterior ey loop 4", ey_yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_idx = VOXEL(2,y,nz+1,nx,ny,nz) + (x-2);
        const size_t fx_idx = VOXEL(1,y,nz+1,nx,ny,nz) + (x-2);
        const size_t fy_idx = 0;
        const size_t fz_idx = VOXEL(2,y,nz,nx,ny,nz) + (x-2);
        update_ey(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });

  // Do exterior ez
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zx_policy({1, 1}, {nz+1, nx+2});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zy_policy({1, 2}, {nz+1, ny+1});
    Kokkos::parallel_for("advance_e: exterior ez loop 1", ez_zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(1,1,z,nx,ny,nz) + (x-1);
        const size_t fx_idx = VOXEL(0,1,z,nx,ny,nz) + (x-1);
        const size_t fy_idx = VOXEL(1,0,z,nx,ny,nz) + (x-1);
        const size_t fz_idx = 0;
        update_ez(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
    Kokkos::parallel_for("advance_e: exterior ez loop 2", ez_zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_idx = VOXEL(1,ny+1,z,nx,ny,nz) + (x-1);
        const size_t fx_idx = VOXEL(0,ny+1,z,nx,ny,nz) + (x-1);
        const size_t fy_idx = VOXEL(1,ny  ,z,nx,ny,nz) + (x-1);
        const size_t fz_idx = 0;
        update_ez(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
    Kokkos::parallel_for("advance_e: exterior ez loop 3", ez_zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_idx = VOXEL(1,y,z,nx,ny,nz);
        const size_t fx_idx = VOXEL(0,y,z,nx,ny,nz);
        const size_t fy_idx = VOXEL(1,y-1,z,nx,ny,nz);
        const size_t fz_idx = 0;
        update_ez(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
    Kokkos::parallel_for("advance_e: exterior ez loop 4", ez_zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_idx = VOXEL(nx+1, y,   z,nx,ny,nz);
        const size_t fx_idx = VOXEL(nx,   y,   z,nx,ny,nz);
        const size_t fy_idx = VOXEL(nx+1, y-1, z,nx,ny,nz);
        const size_t fz_idx = 0;
        update_ez(k_field, k_field_edge, k_material, damp, cj, f0_idx, fx_idx, fy_idx, fz_idx, px, py, pz);
    });
}

void
advance_e_pipeline( pipeline_args_t * args,
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
advance_e_pipeline_v4( pipeline_args_t * args,
                       int pipeline_rank,
                       int n_pipeline ) {
  DECLARE_STENCIL();

  int n_voxel;
  DISTRIBUTE_VOXELS( 2,nx, 2,ny, 2,nz, 16,
                     pipeline_rank, n_pipeline,
                     x, y, z, n_voxel );

  const v4float vdamp( damp );
  const v4float vpx( px );
  const v4float vpy( py );
  const v4float vpz( pz );
  const v4float vcj( cj );

  v4float save0, save1, dummy;

  v4float f0_ex,   f0_ey,   f0_ez;
  v4float f0_cbx,  f0_cby,  f0_cbz;
  v4float f0_tcax, f0_tcay, f0_tcaz;
  v4float f0_jfx,  f0_jfy,  f0_jfz;
  v4float          fx_cby,  fx_cbz;
  v4float fy_cbx,           fy_cbz;
  v4float fz_cbx,  fz_cby;
  v4float m_f0_rmux, m_f0_rmuy, m_f0_rmuz;
  v4float            m_fx_rmuy, m_fx_rmuz;
  v4float m_fy_rmux,            m_fy_rmuz;
  v4float m_fz_rmux, m_fz_rmuy;
  v4float m_f0_decayx, m_f0_drivex;
  v4float m_f0_decayy, m_f0_drivey;
  v4float m_f0_decayz, m_f0_drivez;

  v4float f0_cbx_rmux, f0_cby_rmuy, f0_cbz_rmuz;

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

#   define LOAD_RMU(V,D) m_f##V##_rmu##D=v4float( m[f##V##0->fmat##D].rmu##D, \
                                                  m[f##V##1->fmat##D].rmu##D, \
                                                  m[f##V##2->fmat##D].rmu##D, \
                                                  m[f##V##3->fmat##D].rmu##D )

    LOAD_RMU(0,x); LOAD_RMU(0,y); LOAD_RMU(0,z);
    /**/           LOAD_RMU(x,y); LOAD_RMU(x,z);
    LOAD_RMU(y,x);                LOAD_RMU(y,z);
    LOAD_RMU(z,x); LOAD_RMU(z,y);

    load_4x2_tr( &m[f00->ematx].decayx, &m[f01->ematx].decayx,
                 &m[f02->ematx].decayx, &m[f03->ematx].decayx,
                 m_f0_decayx, m_f0_drivex );
    load_4x2_tr( &m[f00->ematy].decayy, &m[f01->ematy].decayy,
                 &m[f02->ematy].decayy, &m[f03->ematy].decayy,
                 m_f0_decayy, m_f0_drivey );
    load_4x2_tr( &m[f00->ematz].decayz, &m[f01->ematz].decayz,
                 &m[f02->ematz].decayz, &m[f03->ematz].decayz,
                 m_f0_decayz, m_f0_drivez );

#   undef LOAD_RMU

    f0_cbx_rmux = f0_cbx * m_f0_rmux;
    f0_cby_rmuy = f0_cby * m_f0_rmuy;
    f0_cbz_rmuz = f0_cbz * m_f0_rmuz;

    f0_tcax = fnms( vdamp,f0_tcax,
                    fms( vpy,fnms( fy_cbz,m_fy_rmuz, f0_cbz_rmuz ),
                         vpz*fnms( fz_cby,m_fz_rmuy, f0_cby_rmuy ) ) );

    f0_tcay = fnms( vdamp,f0_tcay,
                    fms( vpz,fnms( fz_cbx,m_fz_rmux, f0_cbx_rmux ),
                         vpx*fnms( fx_cbz,m_fx_rmuz, f0_cbz_rmuz ) ) );

    f0_tcaz = fnms( vdamp,f0_tcaz,
                    fms( vpx,fnms( fx_cby,m_fx_rmuy, f0_cby_rmuy ),
                         vpy*fnms( fy_cbx,m_fy_rmux, f0_cbx_rmux ) ) );

    f0_ex = fma( m_f0_decayx,f0_ex, m_f0_drivex*fnms( vcj,f0_jfx, f0_tcax ));
    f0_ey = fma( m_f0_decayy,f0_ey, m_f0_drivey*fnms( vcj,f0_jfy, f0_tcay ));
    f0_ez = fma( m_f0_decayz,f0_ez, m_f0_drivez*fnms( vcj,f0_jfz, f0_tcaz ));

    // Note: Unlike load_4x3 versus load_4x4, store_4x4 is much more efficient than store_4x3!

    store_4x4_tr( f0_ex,   f0_ey,   f0_ez,   save0, &f00->ex,    &f01->ex,    &f02->ex,    &f03->ex   );
    store_4x4_tr( f0_tcax, f0_tcay, f0_tcaz, save1, &f00->tcax,  &f01->tcax,  &f02->tcax,  &f03->tcax );
  }
}

#endif

void
advance_e( field_array_t * RESTRICT fa,
           float frac ) {
  if( !fa     ) ERROR(( "Bad args" ));
  if( frac!=1 ) ERROR(( "standard advance_e does not support frac!=1 yet" ));

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
  EXEC_PIPELINES( advance_e, args, 0 );

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

void advance_e_kokkos(field_array_t* RESTRICT fa, float frac) {
    if( !fa     ) ERROR(( "Bad args" ));
    if( frac!=1 ) ERROR(( "standard advance_e does not support frac!=1 yet" ));
    pipeline_args_t args[1];
    args->f = fa->f;
    args->p = (sfa_params_t *)fa->params;
    args->g = fa->g;

    k_field_t k_field = fa->k_f_d;
    k_field_edge_t k_field_edge = fa->k_fe_d;

    //const material_coefficient_t * ALIGNED(128) m = args->p->mc;

    const grid_t                 *              g = args->g;
    const int nx = g->nx, ny = g->ny, nz = g->nz;

    const float damp = args->p->damp;
    const float px   = (nx>1) ? (1+damp)*g->cvac*g->dt*g->rdx : 0;
    const float py   = (ny>1) ? (1+damp)*g->cvac*g->dt*g->rdy : 0;
    const float pz   = (nz>1) ? (1+damp)*g->cvac*g->dt*g->rdz : 0;
    const float cj   = g->dt/g->eps0;

    sfa_params_t* sfa = reinterpret_cast<sfa_params_t *>(fa->params);
    const k_material_coefficient_t& k_material_d = sfa->k_mc_d;

    /***************************************************************************
    * Begin tangential B ghost setup
    ***************************************************************************/

    kokkos_begin_remote_ghost_tang_b( fa, fa->g, *fa->fb );

    k_local_ghost_tang_b( fa, fa->g );

    advance_e_interior_kokkos(k_field, k_field_edge, k_material_d, nx, ny, nz, px, py, pz, damp, cj);

    /***************************************************************************
    * Finish tangential B ghost setup
    ***************************************************************************/

    kokkos_end_remote_ghost_tang_b( fa, fa->g, *fa->fb );

    /***************************************************************************
    * Update exterior fields
    ***************************************************************************/

    advance_e_exterior_kokkos(k_field, k_field_edge, k_material_d, nx, ny, nz, px, py, pz, damp, cj);

    k_local_adjust_tang_e( fa, fa->g );
}

