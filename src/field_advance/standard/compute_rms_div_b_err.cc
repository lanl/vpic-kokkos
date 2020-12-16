#define IN_sfa
#include "sfa_private.h"

double
compute_rms_div_b_err( const field_array_t * fa ) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  double err = 0, local[2], global[2];

  if( !fa ) ERROR(( "Bad args"));

# if 0 // Original non-pipelined version
  field_t * ALIGNED(16) f0;
  int z, y, x;
  int nx = g->nx;
  int ny = g->ny;
  int nz = g->nz;

  err = 0;
  for( z=1; z<=nz; z++ ) {
    for( y=1; y<=ny; y++ ) {
      f0 = &f(1,y,z);
      for( x=1; x<=nx; x++ ) {
        err += f0->div_b_err*f0->div_b_err;
        f0++;
      }
    }
  }
# endif

    const k_field_t& k_field = fa->k_f_d;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1, 1, 1}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_reduce("compute_rms_div_b_err", zyx_policy, KOKKOS_LAMBDA(const int z, const int y, const int x, double& error) {
        const int f0 = VOXEL(1,y,z,nx,ny,nz) + (x-1);
        error += k_field(f0, field_var::div_b_err) * k_field(f0, field_var::div_b_err);
    }, err);

  local[0] = err*fa->g->dV;
  local[1] = (fa->g->nx*fa->g->ny*fa->g->nz)*fa->g->dV;
  mp_allsum_d( local, global, 2 );
  return fa->g->eps0*sqrt(global[0]/global[1]);
}
