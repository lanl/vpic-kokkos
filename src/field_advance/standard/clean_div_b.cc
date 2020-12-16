#define IN_sfa
#include "sfa_private.h"

template<class geo_t>
KOKKOS_INLINE_FUNCTION void
marder_cbx(
  const geo_t& geometry,
  const k_field_t& k_field,
  const float dt,
  const int f0,
  const int fx
)
{
    k_field(f0, field_var::cbx) += dt*geometry.cell_gradient_x(
      f0,
      k_field(f0, field_var::div_b_err),
      k_field(fx, field_var::div_b_err)
    );

}

template<class geo_t>
KOKKOS_INLINE_FUNCTION void
marder_cby(
  const geo_t& geometry,
  const k_field_t& k_field,
  const float dt,
  const int f0,
  const int fy
)
{
    k_field(f0, field_var::cby) += dt*geometry.cell_gradient_y(
      f0,
      k_field(f0, field_var::div_b_err),
      k_field(fy, field_var::div_b_err)
    );

}

template<class geo_t>
KOKKOS_INLINE_FUNCTION void
marder_cbz(
  const geo_t& geometry,
  const k_field_t& k_field,
  const float dt,
  const int f0,
  const int fz
)
{
    k_field(f0, field_var::cbz) += dt*geometry.cell_gradient_z(
      f0,
      k_field(f0, field_var::div_b_err),
      k_field(fz, field_var::div_b_err)
    );

}

void
clean_div_b( field_array_t * fa ) {
  const grid_t * g = fa->g;

  if( !fa ) ERROR(( "Bad args" ));

  const int nx = g->nx;
  const int ny = g->ny;
  const int nz = g->nz;
  const float _px = (nx>1) ? g->rdx : 0;
  const float _py = (ny>1) ? g->rdy : 0;
  const float _pz = (nz>1) ? g->rdz : 0;

  // TODO: This may be non-optimal in non-Cartesian geometry.
  const float dt = 0.3888889/( _px*_px + _py*_py + _pz*_pz );


  SELECT_GEOMETRY(g->geometry, geo, {

    auto geometry = g->get_device_geometry<geo>();

    // Begin setting derr ghosts
    k_begin_remote_ghost_div_b( fa, g, *(fa->fb) );
    k_local_ghost_div_b( fa, g);

    // Have pipelines do interior of the local domain
    const k_field_t& k_field = fa->k_f_d;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({2, 2, 2}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_for("clean_div_b_kokkos", zyx_policy,
      KOKKOS_LAMBDA(const int z, const int y, const int x) {
        const int f0 = VOXEL(2, y,   z,   nx, ny, nz) + (x-2);
        const int fx = VOXEL(1, y,   z,   nx, ny, nz) + (x-2);
        const int fy = VOXEL(2, y-1, z,   nx, ny, nz) + (x-2);
        const int fz = VOXEL(2, y,   z-1, nx, ny, nz) + (x-2);
        marder_cbx(geometry, k_field, dt, f0, fx);
        marder_cby(geometry, k_field, dt, f0, fy);
        marder_cbz(geometry, k_field, dt, f0, fz);
      });

    // Do left over interior bx
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> bx_yx({1, 2}, {ny+1, nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> bx_zx({2, 2}, {nz+1, nx+1});
    Kokkos::parallel_for("clean_div_b_kokkos: interior bx: yx", bx_yx,
      KOKKOS_LAMBDA(const int y, const int x) {
        const int f0 = VOXEL(2,y,1,nx,ny,nz) + (x-2);
        const int fx = VOXEL(1,y,1,nx,ny,nz) + (x-2);
        marder_cbx(geometry, k_field, dt, f0, fx);
      });

    Kokkos::parallel_for("clean_div_b_kokkos: interior bx: zx", bx_zx,
      KOKKOS_LAMBDA(const int z, const int x) {
        const int f0 = VOXEL(2,1,z,nx,ny,nz) + (x-2);
        const int fx = VOXEL(1,1,z,nx,ny,nz) + (x-2);
        marder_cbx(geometry, k_field, dt, f0, fx);
      });

    // Left over interior by
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> by_zy({1, 2}, {nz+1, ny+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> by_yx({2, 2}, {ny+1, nx+1});
    Kokkos::parallel_for("clean_div_b_kokkos: interior by: zy", by_zy,
      KOKKOS_LAMBDA(const int z, const int y) {
        const int f0 = VOXEL(1,y,  z,nx,ny,nz);
        const int fy = VOXEL(1,y-1,z,nx,ny,nz);
        marder_cby(geometry, k_field, dt, f0, fy);
      });

    Kokkos::parallel_for("clean_div_b_kokkos: interior by: yx", by_yx,
      KOKKOS_LAMBDA(const int y, const int x) {
        const int f0 = VOXEL(2,y,  1,nx,ny,nz) + (x-2);
        const int fy = VOXEL(2,y-1,1,nx,ny,nz) + (x-2);
        marder_cby(geometry, k_field, dt, f0, fy);
      });

    // Left over interior bz
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> bz_zx({2, 1}, {nz+1, nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> bz_zy({2, 2}, {nz+1, ny+1});
    Kokkos::parallel_for("clean_div_b_kokkos: interior bz: zx", bz_zx,
      KOKKOS_LAMBDA(const int z, const int x) {
        const int f0 = VOXEL(1,1,z,  nx,ny,nz) + (x-1);
        const int fz = VOXEL(1,1,z-1,nx,ny,nz) + (x-1);
        marder_cbz(geometry, k_field, dt, f0, fz);
      });

    Kokkos::parallel_for("clean_div_b_kokkos: interior bz: zy", bz_zy,
      KOKKOS_LAMBDA(const int z, const int y) {
        const int f0 = VOXEL(1,y,z,  nx,ny,nz);
        const int fz = VOXEL(1,y,z-1,nx,ny,nz);
        marder_cbz(geometry, k_field, dt, f0, fz);
      });

    // Finish setting derr ghosts

    k_end_remote_ghost_div_b( fa, g, *(fa->fb) );

    // Do Marder pass in exterior

    // Exterior bx
    // TODO fuse kernels
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> bx_zy({1, 1}, {nz+1, ny+1});
    Kokkos::parallel_for("clean_div_b_kokkos: interior bx: zy 1", bx_zy,
      KOKKOS_LAMBDA(const int z, const int y) {
        const int f0 = VOXEL(1,y,z,nx,ny,nz);
        const int fx = VOXEL(0,y,z,nx,ny,nz);
        marder_cbx(geometry, k_field, dt, f0, fx);
      });

    Kokkos::parallel_for("clean_div_b_kokkos: interior bx: zy 2", bx_zy,
      KOKKOS_LAMBDA(const int z, const int y) {
        const int f0 = VOXEL(nx+1, y,z,nx,ny,nz);
        const int fx = VOXEL(nx,   y,z,nx,ny,nz);
        marder_cbx(geometry, k_field, dt, f0, fx);
      });

  // Exterior by
    // TODO fuse kernels
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> by_zx({1, 1}, {nz+1, nx+1});
    Kokkos::parallel_for("clean_div_b_kokkos: interior by: zx 1", by_zx,
      KOKKOS_LAMBDA(const int z, const int x) {
        const int f0 = VOXEL(1,1,z,nx,ny,nz) + (x-1);
        const int fy = VOXEL(1,0,z,nx,ny,nz) + (x-1);
        marder_cby(geometry, k_field, dt, f0, fy);
      });

    Kokkos::parallel_for("clean_div_b_kokkos: interior by: zy 2", by_zx,
      KOKKOS_LAMBDA(const int z, const int x) {
        const int f0 = VOXEL(1, ny+1, z,nx,ny,nz) + (x-1);
        const int fy = VOXEL(1, ny,   z,nx,ny,nz) + (x-1);
        marder_cby(geometry, k_field, dt, f0, fy);
      });

    // Exterior bz
    // TODO fuse kernels
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> bz_yx({1, 1}, {ny+1, nx+1});
    Kokkos::parallel_for("clean_div_b_kokkos: interior bz: yx 1", bz_yx,
      KOKKOS_LAMBDA(const int y, const int x) {
        const int f0 = VOXEL(1,y,1,nx,ny,nz) + (x-1);
        const int fz = VOXEL(1,y,0,nx,ny,nz) + (x-1);
        marder_cbz(geometry, k_field, dt, f0, fz);
      });

    Kokkos::parallel_for("clean_div_b_kokkos: interior bz: yx 2", bz_yx,
      KOKKOS_LAMBDA(const int y, const int x) {
        const int f0 = VOXEL(1, y, nz+1, nx,ny,nz) + (x-1);
        const int fz = VOXEL(1, y, nz,   nx,ny,nz) + (x-1);
        marder_cbz(geometry, k_field, dt, f0, fz);
      });

    k_local_adjust_norm_b(fa,g);

  });

}
