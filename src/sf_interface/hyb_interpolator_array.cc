#define IN_sf_interface
#define HAS_V4_PIPELINE
#include "sf_interface_private.h"


#define fi(x,y,z) fi[   VOXEL(x,y,z, nx,ny,nz) ]
#define f(x,y,z)  f [   VOXEL(x,y,z, nx,ny,nz) ]
#define nb(x,y,z) nb[ 6*VOXEL(x,y,z, nx,ny,nz) ]

# define LOAD_STENCIL()    \
  pi   = &fi(x,  y,  z  ); \
  pf0  =  &f(x,  y,  z  ); \
  pfx  =  &f(x+1,y,  z  ); \
  pfy  =  &f(x,  y+1,z  ); \
  pfz  =  &f(x,  y,  z+1); \
  pfmx =  &f(x-1,y  ,z  ); \
  pfmy =  &f(x  ,y-1,z  ); \
  pfmz =  &f(x  ,y  ,z-1)

# define INTERP_FIELD(F)						\
  w0   = pf0->sm##F;							\
  wx   = pfx->sm##F;							\
  wy   = pfy->sm##F;							\
  wz   = pfz->sm##F;							\
  wmx  = pfmx->sm##F;							\
  wmy  = pfmy->sm##F;							\
  wmz  = pfmz->sm##F;							\
  pi->F       = twelfth*(six*w0 + wx + wy + wz + wmx + wmy + wmz);	\
  pi->d##F##dx    = sixth*(wx - wmx);					\
  pi->d##F##dy    = sixth*(wy - wmy);					\
  pi->d##F##dz    = sixth*(wz - wmz);				        \
  pi->d2##F##dx   = twelfth*(wx + wmx - two*w0);			\
  pi->d2##F##dy   = twelfth*(wy + wmy - two*w0);		        \
  pi->d2##F##dz   = twelfth*(wz + wmz - two*w0)	

# define INTERP_BFIELD(F)						\
  w0   = pf0->sm##F + pf0->F##0;							\
  wx   = pfx->sm##F + pfx->F##0;							\
  wy   = pfy->sm##F + pfy->F##0;							\
  wz   = pfz->sm##F + pfz->F##0;							\
  wmx  = pfmx->sm##F + pfmx->F##0;							\
  wmy  = pfmy->sm##F + pfmy->F##0;							\
  wmz  = pfmz->sm##F + pfmz->F##0;							\
  pi->F       = twelfth*(six*w0 + wx + wy + wz + wmx + wmy + wmz);	\
  pi->d##F##dx    = sixth*(wx - wmx);					\
  pi->d##F##dy    = sixth*(wy - wmy);					\
  pi->d##F##dz    = sixth*(wz - wmz);				        \
  pi->d2##F##dx   = twelfth*(wx + wmx - two*w0);			\
  pi->d2##F##dy   = twelfth*(wy + wmy - two*w0);		        \
  pi->d2##F##dz   = twelfth*(wz + wmz - two*w0)	;			
   

//ARI: doesn't work
void
hyb_load_interpolator_pipeline( load_interpolator_pipeline_args_t * args,
			    int pipeline_rank,
                            int n_pipeline ) {
  interpolator_t * ALIGNED(128) fi = args->fi;
  const field_t  * ALIGNED(128) f  = args->f;

  interpolator_t * ALIGNED(16) pi;

  const field_t  * ALIGNED(16) pf0;
  const field_t  * ALIGNED(16) pfx,  * ALIGNED(16) pfy,  * ALIGNED(16) pfz;
  const field_t  * ALIGNED(16) pfmx, * ALIGNED(16) pfmy, * ALIGNED(16) pfmz;
  int x, y, z, n_voxel;

  const int nx = args->nx;
  const int ny = args->ny;
  const int nz = args->nz;
  
  const float eighth = 0.125;
  const float fourth = 0.25;
  const float two   = 2.0;
  const float three_fourths   = 0.75;
  const float six   = 6.;
  const float twelfth   = 1./12.;
  const float sixth   = 1./6.;

  float w0, wx, wy, wz, wmx, wmy, wmz;

  // Process the voxels assigned to this pipeline
  
  if( pipeline_rank==n_pipeline ) return; // No straggler cleanup needed
  DISTRIBUTE_VOXELS( 1,nx, 1,ny, 1,nz, 1,
                     pipeline_rank, n_pipeline, x, y, z, n_voxel );

  LOAD_STENCIL();
  
  for( ; n_voxel; n_voxel-- ) {

    
    INTERP_FIELD(ex);
    INTERP_FIELD(ey);
    INTERP_FIELD(ez);
    INTERP_BFIELD(cbx);
    INTERP_BFIELD(cby);
    INTERP_BFIELD(cbz);
   
    pi++; 
    pf0++; pfx++; pfy++; pfz++; 
    pfmx++; pfmy++; pfmz++;
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

using namespace v4;

void
load_interpolator_pipeline_v4( load_interpolator_pipeline_args_t * args,
                               int pipeline_rank,
                               int n_pipeline ) {
  interpolator_t * ALIGNED(128) fi = args->fi;
  const field_t  * ALIGNED(128) f  = args->f;

  interpolator_t * ALIGNED(16) pi;

  const field_t * ALIGNED(16) pf0;
  const field_t * ALIGNED(16) pfx,  * ALIGNED(16) pfy,  * ALIGNED(16) pfz;
  const field_t * ALIGNED(16) pfyz, * ALIGNED(16) pfzx, * ALIGNED(16) pfxy;
  int x, y, z, n_voxel;

  const int nx = args->nx;
  const int ny = args->ny;
  const int nz = args->nz;

  const v4float fourth(0.25);
  const v4float half(  0.5 );

  const v4int   sgn_1_2(  0, 1<<31, 1<<31,     0 );
  const v4int   sgn_2_3(  0,     0, 1<<31, 1<<31 );
  const v4int   sgn_1_3(  0, 1<<31,     0, 1<<31 );
  const v4int   sel_0_1( -1,    -1,     0,     0 );

  v4float w0, w1, w2, w3;

  // Process the voxels assigned to this pipeline

  if( pipeline_rank==n_pipeline ) return; // No straggler cleanup needed
  DISTRIBUTE_VOXELS( 1,nx, 1,ny, 1,nz, 1,
                     pipeline_rank, n_pipeline,
                     x, y, z, n_voxel );
  
# define LOAD_STENCIL()    \
  pi   = &fi(x,  y,  z  ); \
  pf0  =  &f(x,  y,  z  ); \
  pfx  =  &f(x+1,y,  z  ); \
  pfy  =  &f(x,  y+1,z  ); \
  pfz  =  &f(x,  y,  z+1); \
  pfyz =  &f(x,  y+1,z+1); \
  pfzx =  &f(x+1,y,  z+1); \
  pfxy =  &f(x+1,y+1,z  )

  LOAD_STENCIL();
  
  for( ; n_voxel; n_voxel-- ) {

    // ex interpolation coefficients 
    w0 = toggle_bits( sgn_1_2, v4float( pf0->ex) ); // [ w0 -w0 -w0 w0 ]
    w1 =                       v4float( pfy->ex);   // [ w1  w1  w1 w1 ]
    w2 = toggle_bits( sgn_1_2, v4float( pfz->ex) ); // [ w2 -w2 -w2 w2 ]
    w3 =                       v4float(pfyz->ex);   // [ w3  w3  w3 w3 ]
    store_4x1( fourth*( ( w3 + w0 ) + toggle_bits( sgn_2_3, w1 + w2 ) ),
               &pi->ex );

    // ey interpolation coefficients 
    w0 = toggle_bits( sgn_1_2, v4float( pf0->ey) ); // [ w0 -w0 -w0 w0 ]
    w1 =                       v4float( pfz->ey);   // [ w1  w1  w1 w1 ]
    w2 = toggle_bits( sgn_1_2, v4float( pfx->ey) ); // [ w2 -w2 -w2 w2 ]
    w3 =                       v4float(pfzx->ey);   // [ w3  w3  w3 w3 ]
    store_4x1( fourth*( ( w3 + w0 ) + toggle_bits( sgn_2_3, w1 + w2 ) ),
               &pi->ey );

    // ez interpolation coefficients 
    w0 = toggle_bits( sgn_1_2, v4float( pf0->ez) ); // [ w0 -w0 -w0 w0 ]
    w1 =                       v4float( pfx->ez);   // [ w1  w1  w1 w1 ]
    w2 = toggle_bits( sgn_1_2, v4float( pfy->ez) ); // [ w2 -w2 -w2 w2 ]
    w3 =                       v4float(pfxy->ez);   // [ w3  w3  w3 w3 ]
    store_4x1( fourth*( ( w3 + w0 ) + toggle_bits( sgn_2_3, w1 + w2 ) ),
               &pi->ez );

    // bx and by interpolation coefficients 
    w0  = toggle_bits( sgn_1_3,
                       merge( sel_0_1,
                              v4float(pf0->cbx),
                              v4float(pf0->cby) ) ); // [ w0x -w0x w0y -w0y ]
    w1  =              merge( sel_0_1,
                              v4float(pfx->cbx),
                              v4float(pfy->cby) );   // [ w1x  w1x w1y  w1y ]
    store_4x1( half*( w1 + w0 ), &pi->cbx );

    // bz interpolation coefficients 
    w0  = toggle_bits( sgn_1_3, v4float(pf0->cbz) ); // [ w0 -w0 d/c d/c ]
    w1  =                       v4float(pfz->cbz);   // [ w1 -w1 d/c d/c ]
    store_4x1( half*( w1 + w0 ), &pi->cbz ); // Note: Padding after bz coeff!

    pi++; pf0++; pfx++; pfy++; pfz++; pfyz++; pfzx++; pfxy++;

    x++;
    if( x>nx ) {
      x=1, y++;
      if( y>ny ) y=1, z++;
      LOAD_STENCIL();
    }
  }

# undef LOAD_STENCIL

}

#endif

void
hyb_load_interpolator_array( /**/  interpolator_array_t * RESTRICT ia,
                         const field_array_t        * RESTRICT fa ) {
  DECLARE_ALIGNED_ARRAY( load_interpolator_pipeline_args_t, 128, args, 1 );

  if( !ia || !fa || ia->g!=fa->g ) ERROR(( "Bad args" ));

#if 1
 interpolator_t * ALIGNED(128) fi = ia->i;
  const field_t  * ALIGNED(128) f  = fa->f;

  interpolator_t * ALIGNED(16) pi;

  const field_t  * ALIGNED(16) pf0;
  const field_t  * ALIGNED(16) pfx,  * ALIGNED(16) pfy,  * ALIGNED(16) pfz;
  const field_t  * ALIGNED(16) pfmx, * ALIGNED(16) pfmy, * ALIGNED(16) pfmz;
  int x, y, z, n_voxel;

  const int nx = ia->g->nx;
  const int ny = ia->g->ny;
  const int nz = ia->g->nz;
  
  const float eighth = 0.125;
  const float fourth = 0.25;
  const float two   = 2.0;
  const float three_fourths   = 0.75;
  const float six   = 6.;
  const float twelfth   = 1./12.;
  const float sixth   = 1./6.;

  float w0, wx, wy, wz, wmx, wmy, wmz;  

for( z=1; z<=nz; z++ ) {
    for( y=1; y<=ny; y++ ) {
      pi   = &fi(1,  y,  z  );
      pf0  =  &f(1,  y,  z  );
      pfx  =  &f(2,  y,  z  );
      pfy  =  &f(1,  y+1,z  );
      pfz  =  &f(1,  y,  z+1);
      pfmx =  &f(0  ,y  ,z  );
      pfmy =  &f(1  ,y-1,z  );
      pfmz =  &f(1  ,y  ,z-1);
      
	for( x=1; x<=nx; x++ ) {

    INTERP_FIELD(ex);
    INTERP_FIELD(ey);
    INTERP_FIELD(ez);
    INTERP_BFIELD(cbx);
    INTERP_BFIELD(cby);
    INTERP_BFIELD(cbz);
   
    pi++; 
    pf0++; pfx++; pfy++; pfz++; 
    pfmx++; pfmy++; pfmz++;
      

      
      }
    }
  }

# undef INTERP_FIELD

# endif

#if 0
  args->fi = ia->i;
  args->f  = fa->f;
  args->nb = ia->g->neighbor;
  args->nx = ia->g->nx;
  args->ny = ia->g->ny;
  args->nz = ia->g->nz;

  EXEC_PIPELINES( load_interpolator, args, 0 );
  WAIT_PIPELINES();
#endif
}
