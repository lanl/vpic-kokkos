// FIXME: This function assumes that the accumlator ghost values are
// zero.  Further, assumes that the ghost values of jfx, jfy, jfz are
// meaningless.  This might be changed to a more robust but slightly
// slower implementation in the near future.

#define IN_sf_interface
#include "sf_interface_private.h"

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]
#define a(x,y,z) a[ VOXEL(x,y,z, nx,ny,nz) ]

void
unload_accumulator_pipeline( unload_accumulator_pipeline_args_t * args,
			     int pipeline_rank,
                             int n_pipeline ) {
  field_t             * ALIGNED(128) f = args->f;
  const accumulator_t * ALIGNED(128) a = args->a;
  
  const accumulator_t * ALIGNED(16) a0;
  const accumulator_t * ALIGNED(16) ax,  * ALIGNED(16) ay,  * ALIGNED(16) az;
  const accumulator_t * ALIGNED(16) ayz, * ALIGNED(16) azx, * ALIGNED(16) axy;
  field_t * ALIGNED(16) f0;
  int x, y, z, n_voxel;
  
  const int nx = args->nx;
  const int ny = args->ny;
  const int nz = args->nz;

  const float cx = args->cx;
  const float cy = args->cy;
  const float cz = args->cz;

  // Process the voxels assigned to this pipeline
  
  if( pipeline_rank==n_pipeline ) return; // No need for straggler cleanup
  DISTRIBUTE_VOXELS( 1,nx+1, 1,ny+1, 1,nz+1, 1,
                     pipeline_rank, n_pipeline, x, y, z, n_voxel );

# define LOAD_STENCIL()                                                 \
  f0  = &f(x,  y,  z  );                                                \
  a0  = &a(x,  y,  z  );                                                \
  ax  = &a(x-1,y,  z  ); ay  = &a(x,  y-1,z  ); az  = &a(x,  y,  z-1);  \
  ayz = &a(x,  y-1,z-1); azx = &a(x-1,y,  z-1); axy = &a(x-1,y-1,z  )

  LOAD_STENCIL();

  for( ; n_voxel; n_voxel-- ) {

    f0->jfx += cx*( a0->jx[0] + ay->jx[1] + az->jx[2] + ayz->jx[3] );
    f0->jfy += cy*( a0->jy[0] + az->jy[1] + ax->jy[2] + azx->jy[3] );
    f0->jfz += cz*( a0->jz[0] + ax->jz[1] + ay->jz[2] + axy->jz[3] );

    f0++; a0++; ax++; ay++; az++; ayz++; azx++; axy++;

    x++;
    if( x>nx+1 ) {
      x=1, y++;
      if( y>ny+1 ) y=1, z++;
      LOAD_STENCIL();
    }

  }

# undef LOAD_STENCIL

}

#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

#error "V4 version not hooked up yet!"

#endif

void
unload_accumulator_array( /**/  field_array_t       * RESTRICT fa,
                          const accumulator_array_t * RESTRICT aa ) {
  unload_accumulator_pipeline_args_t args[1];

  if( !fa || !aa || fa->g!=aa->g ) ERROR(( "Bad args" ));

# if 0 // Original non-pipelined version

  for( z=1; z<=nz+1; z++ ) {
    for( y=1; y<=ny+1; y++ ) {

      x   = 1;
      f0  = &f(x,  y,  z  );
      a0  = &a(x,  y,  z  );
      ax  = &a(x-1,y,  z  ); ay  = &a(x,  y-1,z  ); az  = &a(x,  y,  z-1);
      ayz = &a(x,  y-1,z-1); azx = &a(x-1,y,  z-1); axy = &a(x-1,y-1,z  );

      for( x=1; x<=nx+1; x++ ) {

        f0->jfx += cx*( a0->jx[0] + ay->jx[1] + az->jx[2] + ayz->jx[3] );
        f0->jfy += cy*( a0->jy[0] + az->jy[1] + ax->jy[2] + azx->jy[3] );
        f0->jfz += cz*( a0->jz[0] + ax->jz[1] + ay->jz[2] + axy->jz[3] );

        f0++; a0++; ax++; ay++; az++; ayz++; azx++; axy++;

      }
    }
  }

# endif

  args->f  = fa->f;
  args->a  = aa->a;
  args->nx = fa->g->nx;
  args->ny = fa->g->ny;
  args->nz = fa->g->nz;
  args->cx = 0.25*fa->g->rdy*fa->g->rdz/fa->g->dt;
  args->cy = 0.25*fa->g->rdz*fa->g->rdx/fa->g->dt;
  args->cz = 0.25*fa->g->rdx*fa->g->rdy/fa->g->dt;

  EXEC_PIPELINES( unload_accumulator, args, 0 );
  WAIT_PIPELINES();
}

void
unload_accumulator_array_kokkos(field_array_t* RESTRICT fa,
                                const accumulator_array_t* RESTRICT aa) {
  if( !fa || !aa || fa->g!=aa->g ) ERROR(( "Bad args" ));

    k_field_t& k_field = fa->k_f_d;
    const k_accumulators_t& k_accum = aa->k_a_d;
    int nx = fa->g->nx;
    int ny = fa->g->ny;
    int nz = fa->g->nz;
    float cx = 0.25 * fa->g->rdy * fa->g->rdz / fa->g->dt;
    float cy = 0.25 * fa->g->rdz * fa->g->rdx / fa->g->dt;
    float cz = 0.25 * fa->g->rdx * fa->g->rdy / fa->g->dt;
    
    Kokkos::parallel_for("unload_accumulator_array", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
        const size_t z = team_member.league_rank() + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const size_t j) {
            const size_t y = j + 1;
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, nx+1), [=] (const size_t i) {
                const size_t x = i + 1;
                int f0 = VOXEL(1, y, z, nx, ny, nz) + i;
                int a0 = VOXEL(1, y, z, nx, ny, nz) + i;
                int ax = VOXEL(0, y, z, nx, ny, nz) + i;
                int ay = VOXEL(1, y-1, z, nx, ny, nz) + i;
                int az = VOXEL(1, y, z-1, nx, ny, nz) + i;
                int ayz = VOXEL(1, y-1, z-1, nx, ny, nz) + i;
                int azx = VOXEL(0, y, z-1, nx, ny, nz) + i;
                int axy = VOXEL(0, y-1, z, nx, ny, nz) + i;
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
        });
    });
}
