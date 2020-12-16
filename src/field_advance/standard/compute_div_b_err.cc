#define IN_sfa
#include "sfa_private.h"

void
compute_div_b_err( field_array_t * RESTRICT fa ) {

  if( !fa ) ERROR(( "Bad args" ));

  const grid_t* g = fa->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  k_field_t& k_field = fa->k_f_d;

  SELECT_GEOMETRY(g->geometry, geo, {

    auto geometry = g->get_device_geometry<geo>();

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1, 1, 1}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_for("compute_div_b_err", zyx_policy,
      KOKKOS_LAMBDA(const int z, const int y, const int x) {

        const int f0 = VOXEL(1, y,   z,   nx, ny, nz) + (x-1);
        const int fx = VOXEL(2, y,   z,   nx, ny, nz) + (x-1);
        const int fy = VOXEL(1, y+1, z,   nx, ny, nz) + (x-1);
        const int fz = VOXEL(1, y,   z+1, nx, ny, nz) + (x-1);

        k_field(f0, field_var::div_b_err) = geometry.face_divergence(
          f0,
          k_field(f0, field_var::cbx),
          k_field(fx, field_var::cbx),
          k_field(f0, field_var::cby),
          k_field(fy, field_var::cby),
          k_field(f0, field_var::cbz),
          k_field(fz, field_var::cbz)
        );

    });

  });

}
