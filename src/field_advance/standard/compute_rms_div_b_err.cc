#define IN_sfa
#include "sfa_private.h"

double
compute_rms_div_b_err( const field_array_t * fa ) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  double err = 0, local[2], global[2];

  if( !fa ) ERROR(( "Bad args"));

  SELECT_GEOMETRY(fa->g->geometry, geo, ({

    auto geometry = fa->g->get_device_geometry<geo>();

    const k_field_t& k_field = fa->k_f_d;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1, 1, 1}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_reduce("compute_rms_div_b_err",
      zyx_policy,
      KOKKOS_LAMBDA(const int z, const int y, const int x, double& error) {

        const int f0 = VOXEL(1,y,z,nx,ny,nz) + (x-1);
        const float divB = k_field(f0, field_var::div_b_err);
        error += divB*divB/geometry.inverse_voxel_volume(f0);

    }, err);

    local[0] = err;
    local[1] = geometry.domain_volume;

  }));

  mp_allsum_d( local, global, 2 );
  return fa->g->eps0*sqrt(global[0]/global[1]);
}
