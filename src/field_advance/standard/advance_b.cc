#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"
#include <Kokkos_Core.hpp>
#include <iostream>

void advance_b_kokkos(k_field_t k_field, const k_curvilinear_mesh_t& k_curv,
                      const size_t nx, const size_t ny, const size_t nz, const size_t nv,
                      const float px, const float py, const float pz) {

  #define f0_cbx k_field(f0_index, field_var::cbx)
  #define f0_cby k_field(f0_index, field_var::cby)
  #define f0_cbz k_field(f0_index, field_var::cbz)

  #define f0_ex k_field(f0_index,   field_var::ex)
  #define f0_ey k_field(f0_index,   field_var::ey)
  #define f0_ez k_field(f0_index,   field_var::ez)

  #define fx_ex k_field(fx_index,   field_var::ex)
  #define fx_ey k_field(fx_index,   field_var::ey)
  #define fx_ez k_field(fx_index,   field_var::ez)

  #define fy_ex k_field(fy_index,   field_var::ex)
  #define fy_ey k_field(fy_index,   field_var::ey)
  #define fy_ez k_field(fy_index,   field_var::ez)

  #define fz_ex k_field(fz_index,   field_var::ex)
  #define fz_ey k_field(fz_index,   field_var::ey)
  #define fz_ez k_field(fz_index,   field_var::ez)

  // Curvilinear
  #define UPDATE_CBX() { \
    size_t m0 = VOXEL_TO_MESH(f0_index, nx, ny, nz); \
    size_t my = VOXEL_TO_MESH(fy_index, nx, ny, nz); \
    size_t mz = VOXEL_TO_MESH(fz_index, nx, ny, nz); \
    float h_eta = k_curv(m0, curv_mesh_var::h_2); \
    float h_mu = k_curv(m0, curv_mesh_var::h_3); \
    float h_eta_fy = k_curv(my, curv_mesh_var::h_2); \
    float h_mu_fy = k_curv(my, curv_mesh_var::h_3); \
    float h_eta_fz = k_curv(mz, curv_mesh_var::h_2); \
    float h_mu_fz = k_curv(mz, curv_mesh_var::h_3); \
    f0_cbx -= (1.0f / (h_eta * h_mu)) * ( \
        py * (h_mu_fy * fy_ez - h_mu_fz * f0_ez) - \
        pz * (h_eta_fz * fz_ey - h_eta_fy * f0_ey)); \
  }

  #define UPDATE_CBY() { \
    size_t m0 = VOXEL_TO_MESH(f0_index, nx, ny, nz); \
    size_t mx = VOXEL_TO_MESH(fx_index, nx, ny, nz); \
    size_t mz = VOXEL_TO_MESH(fz_index, nx, ny, nz); \
    float h_xi = k_curv(m0, curv_mesh_var::h_1); \
    float h_mu = k_curv(m0, curv_mesh_var::h_3); \
    float h_xi_fx = k_curv(mx, curv_mesh_var::h_1); \
    float h_mu_fx = k_curv(mx, curv_mesh_var::h_3); \
    float h_xi_fz = k_curv(mz, curv_mesh_var::h_1); \
    float h_mu_fz = k_curv(mz, curv_mesh_var::h_3); \
    f0_cby -= (1.0f / (h_mu * h_xi)) * ( \
        pz * (h_xi_fz * fz_ex - h_xi_fx * f0_ex) - \
        px * (h_mu_fx * fx_ez - h_mu_fz * f0_ez)); \
  }

  #define UPDATE_CBZ() { \
    size_t m0 = VOXEL_TO_MESH(f0_index, nx, ny, nz); \
    size_t mx = VOXEL_TO_MESH(fx_index, nx, ny, nz); \
    size_t my = VOXEL_TO_MESH(fy_index, nx, ny, nz); \
    float h_xi = k_curv(m0, curv_mesh_var::h_1); \
    float h_eta = k_curv(m0, curv_mesh_var::h_2); \
    float h_xi_fx = k_curv(mx, curv_mesh_var::h_1); \
    float h_eta_fx = k_curv(mx, curv_mesh_var::h_2); \
    float h_xi_fy = k_curv(my, curv_mesh_var::h_1); \
    float h_eta_fy = k_curv(my, curv_mesh_var::h_2); \
    f0_cbz -= (1.0f / (h_xi * h_eta)) * ( \
        px * (h_eta_fx * fx_ey - h_eta_fy * f0_ey) - \
        py * (h_xi_fy * fy_ex - h_xi_fx * f0_ex)); \
  }

  // Do the bulk of the magnetic fields in the pipelines.  The host
  // handles stragglers.
  // While the pipelines are busy, do surface fields

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nz+1,ny+1,nx+1});
    Kokkos::parallel_for("advance_b main chunk", xyz_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
        size_t f0_index = VOXEL(x,   y,   z,    nx,ny,nz);
        size_t fx_index = VOXEL(x+1, y,   z,    nx,ny,nz);
        size_t fy_index = VOXEL(x,   y+1, z,    nx,ny,nz);
        size_t fz_index = VOXEL(x,   y,   z+1,  nx,ny,nz);
        UPDATE_CBX();
        UPDATE_CBY();
        UPDATE_CBZ();
    });

  // Do left over bx
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_policy({1,1},{nz+1,ny+1});
    Kokkos::parallel_for("advance_b::bx", zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
        const size_t f0_index = VOXEL(nx+1,y,  z,  nx,ny,nz);
        const size_t fy_index = VOXEL(nx+1,y+1,z,  nx,ny,nz);
        const size_t fz_index = VOXEL(nx+1,y,  z+1,nx,ny,nz);
        UPDATE_CBX();
    });

  // Do left over by
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_policy({1,1},{nz+1,nx+1});
    Kokkos::parallel_for("advance_b::by", zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
        const size_t f0_index = VOXEL(1,ny+1, z,  nx,ny,nz) + (x-1);
        const size_t fx_index = VOXEL(2,ny+1, z,  nx,ny,nz) + (x-1);
        const size_t fz_index = VOXEL(1,ny+1, z+1,nx,ny,nz) + (x-1);
        UPDATE_CBY();
    });

  // Do left over bz
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_policy({1,1},{ny+1,nx+1});
    Kokkos::parallel_for("advance_b::bz", yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
        const size_t f0_index = VOXEL(1,y,   nz+1,  nx,ny,nz) + (x-1);
        const size_t fx_index = VOXEL(2,y,   nz+1,  nx,ny,nz) + (x-1);
        const size_t fy_index = VOXEL(1,y+1, nz+1,  nx,ny,nz) + (x-1);
        UPDATE_CBZ();
    });

}

void
advance_b(field_array_t * RESTRICT fa,
          float       frac) {

  k_field_t k_field = fa->k_f_d;

  grid_t *g   = fa->g;
  size_t nx   = g->nx;
  size_t ny   = g->ny;
  size_t nz   = g->nz;
  size_t nv   = g->nv;
  float  px   = (nx>1) ? frac*g->cvac*g->dt*g->rdx : 0;
  float  py   = (ny>1) ? frac*g->cvac*g->dt*g->rdy : 0;
  float  pz   = (nz>1) ? frac*g->cvac*g->dt*g->rdz : 0;
//printf("Advance_B kernel\n");

  advance_b_kokkos(k_field, g->k_curvilinear_mesh_d, nx, ny, nz, nv, px, py, pz);

  k_local_adjust_norm_b( fa, g );
}
