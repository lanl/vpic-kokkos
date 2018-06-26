#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"
#include <Kokkos_Core.hpp>


enum field_var {
  ex = 0,
  ey = 1,
  ez = 2,
  div_e_err = 3,
  cbx = 4,
  cby = 5,
  cbz = 6,
  div_b_err = 7,
  tcax = 8,
  tcay = 9,
  tcaz = 10,
  rhob = 11,
  jfx = 12,
  jfy = 13,
  jfz = 14,
  rhof = 15
};

enum {
  x = 0,
  y = 1,
  z = 2,
  f0_index = 3,
  fx_index = 4,
  fy_index = 5,
  fz_index = 6,
};

#define INITIAL_KOKKOS_VIEW_BYTES 1024

#define DECLARE_STENCIL()                                       \
  const int   nx   = g->nx;                                     \
  const int   ny   = g->ny;                                     \
  const int   nz   = g->nz;                                     \
                                                                \
  const float px   = (nx>1) ? frac*g->cvac*g->dt*g->rdx : 0;    \
  const float py   = (ny>1) ? frac*g->cvac*g->dt*g->rdy : 0;    \
  const float pz   = (nz>1) ? frac*g->cvac*g->dt*g->rdz : 0;    \
                                                                \
  Kokkos::View<int[7], Kokkos::LayoutLeft, Kokkos::HostSpace>   \
      local (Kokkos::ViewAllocateWithoutInitializing("local"),  \
      INITIAL_KOKKOS_VIEW_BYTES);                               \
  local(x) = 0;                                                 \
  local(y) = 0;                                                 \
  local(z) = 0;

#define INIT_STENCIL()                                                      \
  local_d(f0_index) = VOXEL(local_d(x),local_d(y),local_d(z), nx,ny,nz);    \
  local_d(fx_index) = VOXEL(local_d(x)+1,local_d(y),local_d(z), nx,ny,nz);  \
  local_d(fy_index) = VOXEL(local_d(x),local_d(y)+1,local_d(z), nx,ny,nz);  \
  local_d(fz_index) = VOXEL(local_d(x),local_d(y),local_d(z)+1, nx,ny,nz);  \


#define NEXT_STENCIL()                \
  local_d(f0_index)++;                \
  local_d(fx_index)++;                \
  local_d(fy_index)++;                \
  local_d(fz_index)++;                \
  local_d(x)++;                       \
  if( local_d(x)>nx ) {               \
    /**/       local_d(y)++;            local_d(x) = 1; \
    if( local_d(y)>ny ) local_d(z)++; if( local_d(y)>ny ) local_d(y) = 1; \
    INIT_STENCIL();                   \
  };

#define f0_cbx k_field(local_d(f0_index), cbx)
#define f0_cby k_field(local_d(f0_index), cby)
#define f0_cbz k_field(local_d(f0_index), cbz)

#define f0_ex k_field(local_d(f0_index), ex)
#define f0_ey k_field(local_d(f0_index), ey)
#define f0_ez k_field(local_d(f0_index), ez)

#define fx_ex k_field(local_d(fx_index), ex)
#define fx_ey k_field(local_d(fx_index), ey)
#define fx_ez k_field(local_d(fx_index), ez)

#define fy_ex k_field(local_d(fy_index), ex)
#define fy_ey k_field(local_d(fy_index), ey)
#define fy_ez k_field(local_d(fy_index), ez)

#define fz_ex k_field(local_d(fz_index), ex)
#define fz_ey k_field(local_d(fz_index), ey)
#define fz_ez k_field(local_d(fz_index), ez)

// WTF!  Under -ffast-math, gcc-4.1.1 thinks it is okay to treat the
// below as
//   f0->cbx = ( f0->cbx + py*( blah ) ) - pz*( blah )
// even with explicit parenthesis are in there!  Oh my ...
// -fno-unsafe-math-optimizations must be used

#define UPDATE_CBX() f0_cbx -= ( py*( fy_ez-f0_ez ) - pz*( fz_ey-f0_ey ) );
#define UPDATE_CBY() f0_cby -= ( pz*( fz_ex-f0_ex ) - px*( fx_ez-f0_ez ) );
#define UPDATE_CBZ() f0_cbz -= ( px*( fx_ey-f0_ey ) - py*( fy_ex-f0_ex ) );


void
advance_b(
        k_field_d_t *k_field_d,
        grid_t *g,
        float                    frac) {


  auto k_field = *k_field_d;
  DECLARE_STENCIL();

  local(f0_index) = VOXEL(local(x),local(y),local(z), nx,ny,nz);
  local(fx_index) = VOXEL(local(x)+1,local(y),local(z), nx,ny,nz);
  local(fy_index) = VOXEL(local(x),local(y)+1,local(z), nx,ny,nz);
  local(fz_index) = VOXEL(local(x),local(y),local(z)+1, nx,ny,nz);

  auto local_d = Kokkos::create_mirror_view_and_copy(Kokkos::CudaSpace(), local, "local_d");

  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::Cuda >(0, g->nv), KOKKOS_LAMBDA (int i) {
    UPDATE_CBX(); UPDATE_CBY(); UPDATE_CBZ();
    NEXT_STENCIL();  
  });


  // Do the bulk of the magnetic fields in the pipelines.  The host
  // handles stragglers.

  // While the pipelines are busy, do surface fields

  // Do left over bx
  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::Cuda >(1, nz), KOKKOS_LAMBDA (int z) {
    for(int y=1; y<=ny; y++ ) {
      local_d(f0_index) = VOXEL(nx+1,y,  z,  nx,ny,nz);
      local_d(fy_index) = VOXEL(nx+1,y+1,z,  nx,ny,nz);
      local_d(fz_index) = VOXEL(nx+1,y,  z+1,nx,ny,nz);
      UPDATE_CBX();
    }
  });
  // Do left over by
  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::Cuda >(1, nz), KOKKOS_LAMBDA (int z) {
    local_d(f0_index) = VOXEL(1,ny+1,z,  nx,ny,nz);
    local_d(fy_index) = VOXEL(2,ny+1,z,  nx,ny,nz);
    local_d(fz_index) = VOXEL(1,ny+1,z+1,nx,ny,nz);
    for(int x=1; x<=nx; x++ ) {
      UPDATE_CBY();
      local_d(f0_index)++;
      local_d(fy_index)++;
      local_d(fz_index)++;
    }
  });

  // Do left over bz
  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::Cuda >(1, nz), KOKKOS_LAMBDA (int z) {
  //for( y=1; y<=ny; y++ ) {
    local_d(f0_index) = VOXEL(1,y,  nz+1,nx,ny,nz);
    local_d(fy_index) = VOXEL(2,y,  nz+1,nx,ny,nz);
    local_d(fy_index) = VOXEL(1,y+1,nz+1,nx,ny,nz);
    for(int x=1; x<=nx; x++ ) {
      UPDATE_CBZ();
      local_d(f0_index)++;
      local_d(fx_index)++;
      local_d(fy_index)++;
    }
  });

  // what does this do?
  // local_adjust_norm_b( f, g );
}
