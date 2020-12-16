/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

#include "accumulator_array.h"

static int
aa_n_pipeline(void) {
  int                       n = serial.n_pipeline;
  if( n<thread.n_pipeline ) n = thread.n_pipeline;
  return n; /* max( {serial,thread,spu}.n_pipeline ) */
}

void
checkpt_accumulator_array( const accumulator_array_t * aa ) {
  CHECKPT( aa, 1 );
  CHECKPT_ALIGNED( aa->a, (size_t)(aa->n_pipeline+1)*(size_t)aa->stride, 128 );
  CHECKPT_PTR( aa->g );
}

accumulator_array_t *
restore_accumulator_array( void ) {
  accumulator_array_t * aa;
  RESTORE( aa );
  RESTORE_ALIGNED( aa->a );
  RESTORE_PTR( aa->g );
  if( aa->n_pipeline!=aa_n_pipeline() )
    ERROR(( "Number of accumulators restored is not the same as the number of "
            "accumulators checkpointed.  Did you change the number of threads "
            "per process between checkpt and restore?" ));
  return aa;
}

accumulator_array_t *
new_accumulator_array( grid_t * g ) {
  accumulator_array_t * aa;
  if( !g ) ERROR(( "Bad grid."));
  //MALLOC( aa, 1 );

  // TODO: this is likely too big
  //printf("Making %d copies of accumulator \n",aa_n_pipeline()+1 );
  aa = new accumulator_array_t(
          //(size_t)(aa->n_pipeline+1)*(size_t)aa->stride,
          //(size_t)(aa_n_pipeline()+1)*(size_t)(POW2_CEIL(g->nv,2))
          g->nv
  );
  aa->n_pipeline = aa_n_pipeline();
  aa->stride     = POW2_CEIL(g->nv,2);
  aa->g          = g;
  //aa->na         = (size_t)(aa->n_pipeline+1)*(size_t)aa->stride;
  MALLOC_ALIGNED( aa->a, aa->na, 128 );
  CLEAR( aa->a, aa->na);
  REGISTER_OBJECT( aa, checkpt_accumulator_array, restore_accumulator_array,
                  NULL );
  return aa;
}

void
delete_accumulator_array( accumulator_array_t * aa ) {
  if( !aa ) return;
  UNREGISTER_OBJECT( aa );
  FREE_ALIGNED( aa->a );
  FREE( aa );
}

// void
// accumulator_array_t::reduce()
// {

//   const k_accumulators_t& k_accum = k_a_d;
//   auto& k_scatter = k_a_sa;

//   #define VOX(x,y,z) VOXEL(x,y,z, g->nx, g->ny, g->nz)
//   const int start = (VOX(1,1,1)/2)*2;
//   const int end = (((VOX(g->nx, g->ny, g->nz) - (VOX(1,1,1)/2)*2 + 1)+1)/2)*2;
//   #undef VOX


//   Kokkos::MDRangePolicy<Kokkos::Rank<3>> accum_policy({0, 0, start}, {4, 3, end});
//   Kokkos::parallel_for("reduce accumulator", accum_policy,
//     KOKKOS_LAMBDA(const int i, const int j, const int v) {
//       auto k_accum_sa = k_scatter.access();
//       const float next = k_accum(v+1, j, i);
//       k_accum_sa(v,j,i) += next;
//   });

//   Kokkos::Experimental::contribute(k_a_d, k_a_sa);
//   k_a_sa.reset_except(k_a_d);

// }

void
accumulator_array_t::contribute()
{

  Kokkos::Experimental::contribute(k_a_d, k_a_sa);
  k_a_sa.reset_except(k_a_d);

}

void
accumulator_array_t::combine()
{

  k_accumulators_t& k_accum_d = k_a_d;
  k_accumulators_t& k_accum_copy = k_a_d_copy;

  auto& k_accum_h = k_a_h;

  Kokkos::deep_copy(k_accum_copy, k_accum_h);

  int nv = k_accum_d.extent(0);

  Kokkos::parallel_for("combine accumulator array", nv,
    KOKKOS_LAMBDA (const int i) {
      k_accum_d(i, accumulator_var::jx, 0) += k_accum_copy(i, accumulator_var::jx, 0);
      k_accum_d(i, accumulator_var::jx, 1) += k_accum_copy(i, accumulator_var::jx, 1);
      k_accum_d(i, accumulator_var::jx, 2) += k_accum_copy(i, accumulator_var::jx, 2);
      k_accum_d(i, accumulator_var::jx, 3) += k_accum_copy(i, accumulator_var::jx, 3);

      k_accum_d(i, accumulator_var::jy, 0) += k_accum_copy(i, accumulator_var::jy, 0);
      k_accum_d(i, accumulator_var::jy, 1) += k_accum_copy(i, accumulator_var::jy, 1);
      k_accum_d(i, accumulator_var::jy, 2) += k_accum_copy(i, accumulator_var::jy, 2);
      k_accum_d(i, accumulator_var::jy, 3) += k_accum_copy(i, accumulator_var::jy, 3);

      k_accum_d(i, accumulator_var::jz, 0) += k_accum_copy(i, accumulator_var::jz, 0);
      k_accum_d(i, accumulator_var::jz, 1) += k_accum_copy(i, accumulator_var::jz, 1);
      k_accum_d(i, accumulator_var::jz, 2) += k_accum_copy(i, accumulator_var::jz, 2);
      k_accum_d(i, accumulator_var::jz, 3) += k_accum_copy(i, accumulator_var::jz, 3);
  });

}

void
accumulator_array_t::unload( field_array_t * RESTRICT fa )
{

  if( !fa || fa->g!=g ) ERROR(( "Bad args" ));

  k_field_t& k_field = fa->k_f_d;
  const k_accumulators_t& k_accum = k_a_d;
  int nx = fa->g->nx;
  int ny = fa->g->ny;
  int nz = fa->g->nz;
  float cx = 0.25 * fa->g->rdy * fa->g->rdz / fa->g->dt;
  float cy = 0.25 * fa->g->rdz * fa->g->rdx / fa->g->dt;
  float cz = 0.25 * fa->g->rdx * fa->g->rdy / fa->g->dt;

  Kokkos::MDRangePolicy<Kokkos::Rank<3>> unload_policy({1, 1, 1}, {nz+2, ny+2, nx+2});
  Kokkos::parallel_for("unload accumulator array", unload_policy,
    KOKKOS_LAMBDA(const int z, const int y, const int x) {

      int f0 = VOXEL(1, y, z, nx, ny, nz) + x-1;
      int a0 = VOXEL(1, y, z, nx, ny, nz) + x-1;
      int ax = VOXEL(0, y, z, nx, ny, nz) + x-1;
      int ay = VOXEL(1, y-1, z, nx, ny, nz) + x-1;
      int az = VOXEL(1, y, z-1, nx, ny, nz) + x-1;
      int ayz = VOXEL(1, y-1, z-1, nx, ny, nz) + x-1;
      int azx = VOXEL(0, y, z-1, nx, ny, nz) + x-1;
      int axy = VOXEL(0, y-1, z, nx, ny, nz) + x-1;
      k_field(f0, field_var::jfx) += cx*( k_accum(a0, accumulator_var::jx, 0) +
                                          k_accum(ay, accumulator_var::jx, 1) +
                                          k_accum(az, accumulator_var::jx, 2) +
                                          k_accum(ayz, accumulator_var::jx, 3) );
      k_field(f0, field_var::jfy) += cy*( k_accum(a0, accumulator_var::jy, 0) +
                                          k_accum(az, accumulator_var::jy, 1) +
                                          k_accum(ax, accumulator_var::jy, 2) +
                                          k_accum(azx, accumulator_var::jy, 3) );
      k_field(f0, field_var::jfz) += cz*( k_accum(a0, accumulator_var::jz, 0) +
                                          k_accum(ax, accumulator_var::jz, 1) +
                                          k_accum(ay, accumulator_var::jz, 2) +
                                          k_accum(axy, accumulator_var::jz, 3) );

  });

}

void
accumulator_array_t::clear() {
    Kokkos::deep_copy(k_a_d, 0.0f);
}

void
accumulator_array_t::copy_to_host() {

  Kokkos::deep_copy(k_a_h, k_a_d);

  // Avoid capturing this
  auto& k_accumulators_h = k_a_h;
  accumulator_t * host_accum = a;

  Kokkos::parallel_for("copy accumulator to host",
    KOKKOS_TEAM_POLICY_HOST (na, Kokkos::AUTO),
    KOKKOS_LAMBDA (const KOKKOS_TEAM_POLICY_HOST::member_type &team_member) {

      const unsigned int i = team_member.league_rank();
      /* TODO: Do we really need a 2d loop here*/
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ACCUMULATOR_ARRAY_LENGTH), [=] (int j) {
        host_accum[i].jx[j] = k_accumulators_h(i, accumulator_var::jx, j);
        host_accum[i].jy[j] = k_accumulators_h(i, accumulator_var::jy, j);
        host_accum[i].jz[j] = k_accumulators_h(i, accumulator_var::jz, j);
      });

    });

}

void
accumulator_array_t::copy_to_device() {

  // Avoid capturing this
  auto& k_accumulators_h = k_a_h;
  accumulator_t * host_accum = a;

  Kokkos::parallel_for("copy accumulator to device",
    KOKKOS_TEAM_POLICY_HOST (na, Kokkos::AUTO),
    KOKKOS_LAMBDA (const KOKKOS_TEAM_POLICY_HOST::member_type &team_member) {

      const unsigned int i = team_member.league_rank();
      /* TODO: Do we really need a 2d loop here*/
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ACCUMULATOR_ARRAY_LENGTH), [=] (int j) {
        k_accumulators_h(i, accumulator_var::jx, j) = host_accum[i].jx[j];
        k_accumulators_h(i, accumulator_var::jy, j) = host_accum[i].jy[j];
        k_accumulators_h(i, accumulator_var::jz, j) = host_accum[i].jz[j];
      });

    });
  Kokkos::deep_copy(k_a_d, k_a_h);

}
