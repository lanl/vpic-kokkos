// FIXME: This function assumes that the accumlator ghost values are
// zero.  Further, assumes that the ghost values of jfx, jfy, jfz are
// meaningless.  This might be changed to a more robust but slightly
// slower implementation in the near future.

#define IN_sf_interface
#include "sf_interface_private.h"

//#include <iostream>
//#include <fstream>

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]
#define a(x,y,z) a[ VOXEL(x,y,z, nx,ny,nz) ]

void
hyb_unload_accumulator_pipeline( unload_accumulator_pipeline_args_t * args,
			     int pipeline_rank,
                             int n_pipeline ) {
  field_t             * ALIGNED(128) f = args->f;
  const accumulator_t * ALIGNED(128) a = args->a;
  
  const accumulator_t * ALIGNED(16) a0;
  const accumulator_t * ALIGNED(16) ax,  * ALIGNED(16) ay,  * ALIGNED(16) az;
  const accumulator_t * ALIGNED(16) amx, * ALIGNED(16) amy, * ALIGNED(16) amz;
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
  DISTRIBUTE_VOXELS( 1,nx, 1,ny, 1,nz, 1,
                     pipeline_rank, n_pipeline, x, y, z, n_voxel );

# define LOAD_STENCIL()                                                 \
  f0   = &f(x,  y,  z  );						\
  a0   = &a(x,  y,  z  );						\
  ax   = &a(x+1,y,  z  );						\
  ay   = &a(x,  y+1,z  );						\
  az   = &a(x,  y,  z+1);						\
  amx  = &a(x-1,y,  z  );						\
  amy  = &a(x,  y-1,z  );						\
  amz  = &a(x,  y,  z-1)						
  

  LOAD_STENCIL();

  for( ; n_voxel; n_voxel-- ) {

    //f0->jfx += cx*( a0->jx[0] + ay->jx[1] + az->jx[2] + ayz->jx[3] );
    //f0->jfy += cy*( a0->jy[0] + az->jy[1] + ax->jy[2] + azx->jy[3] );
    //f0->jfz += cz*( a0->jz[0] + ax->jz[1] + ay->jz[2] + axy->jz[3] );

    //f0++; a0++; ax++; ay++; az++; ayz++; azx++; axy++;

    x++;
    if( x>nx ) {
      x=1, y++;
      if( y>ny ) y=1, z++;
      LOAD_STENCIL();
    }

  }

# undef LOAD_STENCIL

}

#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

#error "V4 version not hooked up yet!"

#endif

void
hyb_unload_accumulator_array( /**/  field_array_t       * RESTRICT fa,
                          const accumulator_array_t * RESTRICT aa ) {
  unload_accumulator_pipeline_args_t args[1];

// non-pipelined version
  field_t             * ALIGNED(128) f = fa->f;
  const accumulator_t * ALIGNED(128) a = aa->a;
  
  const accumulator_t * ALIGNED(16) a0;
  const accumulator_t * ALIGNED(16) ax,  * ALIGNED(16) ay,  * ALIGNED(16) az;
  const accumulator_t * ALIGNED(16) amx, * ALIGNED(16) amy, * ALIGNED(16) amz;

  field_t * ALIGNED(16) f0;
  int x, y, z, n_voxel;
  float cx, cy, cz, cmx, cmy, cmz;
  
  const int nx = fa->g->nx;
  const int ny = fa->g->ny;
  const int nz = fa->g->nz;

  const float rV12 = 1./12.*fa->g->rdx*fa->g->rdy*fa->g->rdz;
 

  if( !fa || !aa || fa->g!=aa->g ) ERROR(( "Bad args" ));


# define LOAD_STENCIL()                                                 \
  f0   = &f(1,  y,  z  );						\
  a0   = &a(1,  y,  z  );						\
  ax   = &a(2  ,y,  z  );						\
  ay   = &a(1,  y+1,z  );						\
  az   = &a(1,  y,  z+1);						\
  amx  = &a(0,  y,  z  );						\
  amy  = &a(1,  y-1,z  );						\
  amz  = &a(1,  y,  z-1)		
  
  for( z=1; z<=nz; z++ ) {
    for( y=1; y<=ny; y++ ) { 
      
      LOAD_STENCIL();
    
      for( x=1; x<=nx; x++ ) {

	 //std::cout << "rho[0]=" << a0->rho[0] << "\n";

	f0->jfx += rV12*( a0->jx[0] + ax->jx[1] + ay->jx[2] + az->jx[3] \
			  + amx->jx[4] + amy->jx[5] + amz->jx[6] );
	f0->jfy += rV12*( a0->jy[0] + ax->jy[1] + ay->jy[2] + az->jy[3] \
			  + amx->jy[4] + amy->jy[5] + amz->jy[6] );
	f0->jfz += rV12*( a0->jz[0] + ax->jz[1] + ay->jz[2] + az->jz[3] \
			  + amx->jz[4] + amy->jz[5] + amz->jz[6] );
	f0->rhof += rV12*( a0->rho[0] + ax->rho[1] + ay->rho[2] + az->rho[3] \
			  + amx->rho[4] + amy->rho[5] + amz->rho[6] );
	
	f0++; a0++; ax++; ay++; az++; amx++; amy++; amz++;
      }
    }
  }
  
# undef LOAD_STENCIL

# define GHOST_ACCUMULATE(offset)	  \
  f0->jfx  += rV12*( a0->jx[offset]  );    \
  f0->jfy  += rV12*( a0->jy[offset]  );	  \
  f0->jfz  += rV12*( a0->jz[offset]  );	  \
  f0->rhof += rV12*( a0->rho[offset] )    

  
  //x ghost cells. 
  //Ghost densities and currents are added back to edge neighbors 
  //in hyb_advance_b (calls remote_ghost_rho) as part of field boundary conditions
  
  for( y=1; y<=ny; y++ ) {
    for( z=1; z<=nz; z++ ) {
      f0   = &f(0,  y,  z  );
      a0   = &a(1,  y,  z  );
      GHOST_ACCUMULATE(1);
    }
  }
  
  for( y=1; y<=ny; y++ ) {
    for( z=1; z<=nz; z++ ) {
      f0  = &f(nx+1,y,  z  );
      a0  = &a(nx,  y,  z  );
      GHOST_ACCUMULATE(4);
    }
  }
  
  //y ghost cells
  
  for( z=1; z<=nz; z++ ) {
    f0   = &f(1, 0, z );
    a0   = &a(1, 1, z );
    for( x=1; x<=nx; x++ ) {      
      GHOST_ACCUMULATE(2);
      f0++; a0++;
    }
  }
  
 for( z=1; z<=nz; z++ ) {
    f0   = &f(1, ny+1,z );
    a0   = &a(1, ny,  z );
    for( x=1; x<=nx; x++ ) {      
      GHOST_ACCUMULATE(5);
      f0++; a0++;
    }
  }

  //z ghost cells
  
  for( y=1; y<=ny; y++ ) {
    f0   = &f(1, y, 0 );
    a0   = &a(1, y, 1 );
    for( x=1; x<=nx; x++ ) {      
      GHOST_ACCUMULATE(3);
      f0++; a0++;
    }
  }
  
 for( y=1; y<=ny; y++ ) {
    f0   = &f(1, y, nz+1 );
    a0   = &a(1, y, nz );
    for( x=1; x<=nx; x++ ) {      
      GHOST_ACCUMULATE(6);
      f0++; a0++;
    }
  }
#undef GHOST_ACCUMULATE


#if 0 
//for pipelined
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
#endif
}
