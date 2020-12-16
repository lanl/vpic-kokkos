#define IN_sfa
#include "sfa_private.h"

template<class geo_t, class edge_t>
KOKKOS_INLINE_FUNCTION void
marder_ex(
  const geo_t& geometry,
  const k_field_t& k_field,
  const edge_t& k_field_edge,
  const k_material_coefficient_t& k_mat,
  const float dt,
  const int f0,
  const int fx
)
{

  float drive = k_mat(k_field_edge(f0, field_edge_var::ematx), material_coeff_var::drivex);

  k_field(f0, field_var::ex) += drive * dt * geometry.node_gradient_x(
    f0, k_field(f0, field_var::div_e_err), k_field(fx, field_var::div_e_err)
  );

};

template<class geo_t, class edge_t>
KOKKOS_INLINE_FUNCTION void
marder_ey(
  const geo_t& geometry,
  const k_field_t& k_field,
  const edge_t& k_field_edge,
  const k_material_coefficient_t& k_mat,
  const float dt,
  const int f0,
  const int fy
)
{

  float drive = k_mat(k_field_edge(f0, field_edge_var::ematy), material_coeff_var::drivey);

  k_field(f0, field_var::ey) += drive * dt * geometry.node_gradient_y(
    f0, k_field(f0, field_var::div_e_err), k_field(fy, field_var::div_e_err)
  );

};

template<class geo_t, class edge_t>
KOKKOS_INLINE_FUNCTION void
marder_ez(
  const geo_t& geometry,
  const k_field_t& k_field,
  const edge_t& k_field_edge,
  const k_material_coefficient_t& k_mat,
  const float dt,
  const int f0,
  const int fz
)
{

  float drive = k_mat(k_field_edge(f0, field_edge_var::ematz), material_coeff_var::drivez);

  k_field(f0, field_var::ez) += drive * dt * geometry.node_gradient_z(
    f0, k_field(f0, field_var::div_e_err), k_field(fz, field_var::div_e_err)
  );

};

template<class geo_t, class edge_t>
void
marder_loops(
  const geo_t& geometry,
  const k_field_t& k_field,
  const edge_t& k_field_edge,
  const k_material_coefficient_t& k_mat,
  const float dt,
  const int nx,
  const int ny,
  const int nz
)
{

  // Do bulk
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1,1,1}, {nz+1,ny+1,nx+1});
  Kokkos::parallel_for("vacuum_clean_div_e main body", zyx_policy,
    KOKKOS_LAMBDA(const int z, const int y, const int x) {
      const int f0 = VOXEL(x,   y,   z, nx, ny, nz);
      const int fx = VOXEL(x+1, y,   z, nx, ny, nz);
      const int fy = VOXEL(x,   y+1, z, nx, ny, nz);
      const int fz = VOXEL(x,   y,   z+1, nx, ny, nz);
      marder_ex(geometry, k_field, k_field_edge, k_mat, dt, f0, fx);
      marder_ey(geometry, k_field, k_field_edge, k_mat, dt, f0, fy);
      marder_ez(geometry, k_field, k_field_edge, k_mat, dt, f0, fz);
    });

  // Do left over ex
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_yx_pol({1, 1}, {ny+2, nx+1});
  Kokkos::parallel_for("vacuum_clean_div_e: left over ex: yx loop", ex_yx_pol,
    KOKKOS_LAMBDA(const int y, const int x) {
      const int f0 = VOXEL(1, y, nz+1, nx, ny, nz) + (x-1);
      const int fx = VOXEL(2, y, nz+1, nx, ny, nz) + (x-1);
      marder_ex(geometry, k_field, k_field_edge, k_mat, dt, f0, fx);
    });

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_zx_pol({1, 1}, {nz+1, nx+1});
  Kokkos::parallel_for("vacuum_clean_div_e: left over ex: zx loop", ex_zx_pol,
    KOKKOS_LAMBDA(const int z, const int x) {
      const int f0 = VOXEL(1, ny+1, z, nx, ny, nz) + (x-1);
      const int fx = VOXEL(2, ny+1, z, nx, ny, nz) + (x-1);
      marder_ex(geometry, k_field, k_field_edge, k_mat, dt, f0, fx);
    });

  // Do left over ey
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_zy_pol({1, 1}, {nz+2, ny+1});
  Kokkos::parallel_for("vacuum_clean_div_e: left over ey: zy loop", ey_zy_pol,
    KOKKOS_LAMBDA(const int z, const int y) {
      const int f0 = VOXEL(nx+1, y,   z, nx, ny, nz);
      const int fy = VOXEL(nx+1, y+1, z, nx, ny, nz);
      marder_ey(geometry, k_field, k_field_edge, k_mat, dt, f0, fy);
    });

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_yx_pol({1, 1}, {ny+1, nx+1});
  Kokkos::parallel_for("vacuum_clean_div_e: left over ey: yx loop", ey_yx_pol,
    KOKKOS_LAMBDA(const int y, const int x) {
      const int f0 = VOXEL(1, y,   nz+1, nx, ny, nz) + (x-1);
      const int fy = VOXEL(1, y+1, nz+1, nx, ny, nz) + (x-1);
      marder_ey(geometry, k_field, k_field_edge, k_mat, dt, f0, fy);
    });

  // Do left over ez
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zx_pol({1, 1}, {nz+1, nx+2});
  Kokkos::parallel_for("vacuum_clean_div_e: left over ez: zx loop", ez_zx_pol,
    KOKKOS_LAMBDA(const int z, const int x) {
      const int f0 = VOXEL(1, ny+1, z,   nx, ny, nz) + (x-1);
      const int fz = VOXEL(1, ny+1, z+1, nx, ny, nz) + (x-1);
      marder_ez(geometry, k_field, k_field_edge, k_mat, dt, f0, fz);
    });

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zy_pol({1, 1}, {nz+1, ny+1});
  Kokkos::parallel_for("vacuum_clean_div_e: left over ez: zy loop", ez_zy_pol,
    KOKKOS_LAMBDA(const int z, const int y) {
      const int f0 = VOXEL(nx+1, y, z, nx, ny, nz);
      const int fz = VOXEL(nx+1, y, z+1, nx, ny, nz);
      marder_ez(geometry, k_field, k_field_edge, k_mat, dt, f0, fz);
    });

}

void
clean_div_e( field_array_t * fa ) {
  if( !fa ) ERROR(( "Bad args" ));

  // Do majority of field components in single pass on the pipelines.
  // The host handles stragglers.

  const k_field_t& k_field = fa->k_f_d;
  const k_field_edge_t& k_field_edge = fa->k_fe_d;
  sfa_params_t* sfa = reinterpret_cast<sfa_params_t *>(fa->params);
  const k_material_coefficient_t& k_mat = sfa->k_mc_d;
  const grid_t* g = fa->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  const float _rdx = (nx>1) ? g->rdx : 0;
  const float _rdy = (ny>1) ? g->rdy : 0;
  const float _rdz = (nz>1) ? g->rdz : 0;

  // TODO: This might not be optimal in non-Cartesian geometry
  const float dt = 0.3888889/( _rdx*_rdx + _rdy*_rdy + _rdz*_rdz );

  SELECT_GEOMETRY(g->geometry, geo, {

    auto geometry = g->get_device_geometry<geo>();

    if( k_mat.extent(0) == 1 ) {

      // Vacuum optimized.
      marder_loops(
        geometry,
        k_field,
        VacuumMaterialId(),
        k_mat,
        dt,
        nx,
        ny,
        nz
      );

    } else {

      marder_loops(
        geometry,
        k_field,
        k_field_edge,
        k_mat,
        dt,
        nx,
        ny,
        nz
      );

    }

  });

  k_local_adjust_tang_e( fa, fa->g );

}
