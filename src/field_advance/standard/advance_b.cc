#define IN_sfa
#define HAS_V4_PIPELINE
#include "sfa_private.h"
#include <Kokkos_Core.hpp>
#include <iostream>

void advance_b_kokkos(k_field_t k_field, const size_t nx, const size_t ny, const size_t nz, const size_t nv,
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

  // WTF!  Under -ffast-math, gcc-4.1.1 thinks it is okay to treat the
  // below as
  //   f0->cbx = ( f0->cbx + py*( blah ) ) - pz*( blah )
  // even with explicit parenthesis are in there!  Oh my ...
  // -fno-unsafe-math-optimizations must be used

  #define UPDATE_CBX() f0_cbx -= ( py*( fy_ez-f0_ez ) - pz*( fz_ey-f0_ey ) );
  #define UPDATE_CBY() f0_cby -= ( pz*( fz_ex-f0_ex ) - px*( fx_ez-f0_ez ) );
  #define UPDATE_CBZ() f0_cbz -= ( px*( fx_ey-f0_ey ) - py*( fy_ex-f0_ex ) );

  Kokkos::parallel_for("advance b", KOKKOS_TEAM_POLICY_DEVICE
      (nz, Kokkos::AUTO),
      KOKKOS_LAMBDA
      (const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member)
  {
    const size_t z = team_member.league_rank() + 1;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (size_t yi) {
      const size_t y = yi + 1;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, nx), [=] (size_t xi) {
        const size_t x = xi + 1;
        //printf("%d %d %d %d\n", x ,y,z, ny);
        size_t f0_index = VOXEL(x,  y,  z,   nx,ny,nz);
        size_t fx_index = VOXEL(x+1,y,  z,   nx,ny,nz);
        size_t fy_index = VOXEL(x,  y+1,z,   nx,ny,nz);
        size_t fz_index = VOXEL(x,  y,  z+1, nx,ny,nz);

        UPDATE_CBX();
        UPDATE_CBY();
        UPDATE_CBZ();

        });
    });
  });
/*
    Kokkos::parallel_for(KOKKOS_TEAM_POLICY_DEVICE
        (nv, Kokkos::AUTO),
        KOKKOS_LAMBDA
        (const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
      const size_t v = team_member.league_rank();

      size_t f0_index = VOXEL(1,  1,  1,   nx,ny,nz) + v;
      size_t fx_index = VOXEL(2,  1,  1,   nx,ny,nz) + v;
      size_t fy_index = VOXEL(1,  2,  1,   nx,ny,nz) + v;
      size_t fz_index = VOXEL(1,  1,  2,   nx,ny,nz) + v;

      UPDATE_CBX();
      UPDATE_CBY();
      UPDATE_CBZ();
  });
*/
  // Do the bulk of the magnetic fields in the pipelines.  The host
  // handles stragglers.
  // While the pipelines are busy, do surface fields
  //

  // Do left over bx
  Kokkos::parallel_for("advance b bx", KOKKOS_TEAM_POLICY_DEVICE
      (nz, Kokkos::AUTO),
      KOKKOS_LAMBDA
      (const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
    const size_t z = team_member.league_rank() + 1;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (const size_t yi) {
      const size_t y = yi + 1;

      const size_t f0_index = VOXEL(nx+1,y,  z,  nx,ny,nz);
      const size_t fy_index = VOXEL(nx+1,y+1,z,  nx,ny,nz);
      const size_t fz_index = VOXEL(nx+1,y,  z+1,nx,ny,nz);
      UPDATE_CBX();
    });
  });

  // Do left over by
  Kokkos::parallel_for("advance b by", KOKKOS_TEAM_POLICY_DEVICE
      (nz, Kokkos::AUTO),
      KOKKOS_LAMBDA
      (const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
    const size_t z = team_member.league_rank() + 1;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (const size_t i) {
      const size_t f0_index = VOXEL(1,ny+1,z,  nx,ny,nz) + i;
      const size_t fx_index = VOXEL(2,ny+1,z,  nx,ny,nz) + i;
      const size_t fz_index = VOXEL(1,ny+1,z+1,nx,ny,nz) + i;

      UPDATE_CBY();

    });
  });

  // Do left over bz
  Kokkos::parallel_for("advance b bz", KOKKOS_TEAM_POLICY_DEVICE
      (ny, Kokkos::AUTO),
      KOKKOS_LAMBDA
      (const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
    const size_t y = team_member.league_rank() + 1;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (const size_t i) {
      const size_t f0_index = VOXEL(1,y,  nz+1,nx,ny,nz) + i;
      const size_t fx_index = VOXEL(2,y,  nz+1,nx,ny,nz) + i;
      const size_t fy_index = VOXEL(1,y+1,nz+1,nx,ny,nz) + i;

      UPDATE_CBZ();
    });
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
printf("Advance_B kernel\n");

  advance_b_kokkos(k_field, nx, ny, nz, nv, px, py, pz);

  k_local_adjust_norm_b( fa, g );
}
