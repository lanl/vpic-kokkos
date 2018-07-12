#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"
#include <Kokkos_Core.hpp>

KOKKOS_ENUMS();

#define DECLARE_STENCIL()                                       \
  const int   nx   = g->nx;                                     \
  const int   ny   = g->ny;                                     \
  const int   nz   = g->nz;                                     \
                                                                \
  const float px   = (nx>1) ? frac*g->cvac*g->dt*g->rdx : 0;    \
  const float py   = (ny>1) ? frac*g->cvac*g->dt*g->rdy : 0;    \
  const float pz   = (nz>1) ? frac*g->cvac*g->dt*g->rdz : 0;    \

#define f0_cbx k_field(f0_index, cbx)
#define f0_cby k_field(f0_index, cby)
#define f0_cbz k_field(f0_index, cbz)

#define f0_ex k_field(f0_index,   ex)
#define f0_ey k_field(f0_index,   ey)
#define f0_ez k_field(f0_index,   ez)

#define fx_ex k_field(fx_index,   ex)
#define fx_ey k_field(fx_index,   ey)
#define fx_ez k_field(fx_index,   ez)

#define fy_ex k_field(fy_index,   ex)
#define fy_ey k_field(fy_index,   ey)
#define fy_ez k_field(fy_index,   ez)

#define fz_ex k_field(fz_index,   ex)
#define fz_ey k_field(fz_index,   ey)
#define fz_ez k_field(fz_index,   ez)

// WTF!  Under -ffast-math, gcc-4.1.1 thinks it is okay to treat the
// below as
//   f0->cbx = ( f0->cbx + py*( blah ) ) - pz*( blah )
// even with explicit parenthesis are in there!  Oh my ...
// -fno-unsafe-math-optimizations must be used

#define UPDATE_CBX() f0_cbx -= ( py*( fy_ez-f0_ez ) - pz*( fz_ey-f0_ey ) );
#define UPDATE_CBY() f0_cby -= ( pz*( fz_ex-f0_ex ) - px*( fx_ez-f0_ez ) );
#define UPDATE_CBZ() f0_cbz -= ( px*( fx_ey-f0_ey ) - py*( fy_ex-f0_ex ) );

void
advance_b(field_array_t * RESTRICT fa,
          float       frac) {


  auto k_field = fa->k_f_d;
  auto g = fa->g;

  DECLARE_STENCIL()

  Kokkos::parallel_for(Kokkos::TeamPolicy< Kokkos::DefaultExecutionSpace>
      (nx, Kokkos::AUTO), KOKKOS_LAMBDA (const k_member_t &team_member) {
    const unsigned int x = team_member.league_rank();
    const int offset = 1;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (int y) {

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, nz), [=] (int z) {

      int f0_index = VOXEL(x+offset,  y+offset,  z+offset,   nx,ny,nz);
      int fx_index = VOXEL(x+offset+1,y+offset,  z+offset,   nx,ny,nz);
      int fy_index = VOXEL(x+offset,  y+offset+1,z+offset,   nx,ny,nz);
      int fz_index = VOXEL(x+offset,  y+offset,  z+offset+1, nx,ny,nz);

      UPDATE_CBX(); UPDATE_CBY(); UPDATE_CBZ();
      });
    });
  });


  // Do the bulk of the magnetic fields in the pipelines.  The host
  // handles stragglers.
  // While the pipelines are busy, do surface fields
  //

  // Do left over bx
  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace >
      (1, nz+1), KOKKOS_LAMBDA (int z) {
    // TODO: Parallelize this nested loop
    for(int y=1; y<=ny; y++ ) {
      int f0_index = VOXEL(nx+1,y,  z,  nx,ny,nz);
      int fy_index = VOXEL(nx+1,y+1,z,  nx,ny,nz);
      int fz_index = VOXEL(nx+1,y,  z+1,nx,ny,nz);
      UPDATE_CBX();
    }
  });

  // Do left over by
  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace >
      (1, nz+1), KOKKOS_LAMBDA (int z) {
    int f0_index = VOXEL(1,ny+1,z,  nx,ny,nz);
    int fx_index = VOXEL(2,ny+1,z,  nx,ny,nz);
    int fz_index = VOXEL(1,ny+1,z+1,nx,ny,nz);
    // TODO: Parallelize this nested loop
    for(int x=1; x<=nx; x++ ) {
      UPDATE_CBY();
      f0_index++;
      fx_index++;
      fz_index++;
    }
  });

  // Do left over bz
  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace >
      (1, ny+1), KOKKOS_LAMBDA (int y) {
    int f0_index = VOXEL(1,y,  nz+1,nx,ny,nz);
    int fx_index = VOXEL(2,y,  nz+1,nx,ny,nz);
    int fy_index = VOXEL(1,y+1,nz+1,nx,ny,nz);
    // TODO: Parallelize this nested loop
    for(int x=1; x<=nx; x++ ) {
      UPDATE_CBZ();
      f0_index++;
      fx_index++;
      fy_index++;
    }
  });

  k_local_adjust_norm_b( fa, g );
  
}
