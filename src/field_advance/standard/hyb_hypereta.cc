#define IN_sfa
//#define HAS_V4_PIPELINE
#include "sfa_private.h"

#include <iostream>
//#include <fstream>

typedef struct pipeline_args {
  /**/  field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
   float                       hstep;
} pipeline_args_t;

#define DECLARE_STENCIL()                                             \
  /**/  field_t                * ALIGNED(128) f = args->f;            \
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;        \
  const grid_t                 *              g = args->g;            \
  const int nx = g->nx, ny = g->ny, nz = g->nz;                       \
  const float te = g->te;					      \
  const float eta = g->eta;					      \
                                                                      \
  float px = (nx>1) ? g->rdx : 0;				      \
  float py = (ny>1) ? g->rdx : 0;				      \
  float pz = (nz>1) ? g->rdx : 0;				      \
                                                                      \
  field_t * ALIGNED(16) f0;                                           \
  field_t * ALIGNED(16) fx, * ALIGNED(16) fy, * ALIGNED(16) fz;       \
  field_t * ALIGNED(16) fmx, * ALIGNED(16) fmy, * ALIGNED(16) fmz;    \
  int x, y, z;

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

#define INIT_STENCIL()    \
  f0  = &f(x,  y,  z  );  \
  fmx = &f(x-1,y,  z  );  \
  fmy = &f(x,  y-1,z  );  \
  fmz = &f(x,  y,  z-1);  \
  fx  = &f(x+1,y,  z  );  \
  fy  = &f(x,  y+1,z  );  \
  fz  = &f(x,  y,  z+1);  


//ARI
#define INIT_EDGE_STENCIL(i,j,k)  \
  f0  = &f(i,  j,  k  );          \
  fmx = &f(i-1,j,  k  );          \
  fmy = &f(i,  j-1,k  );          \
  fmz = &f(i,  j,  k-1);          \
  fx  = &f(i+1,j,  k  );          \
  fy  = &f(i,  j+1,k  );	  \
  fz  = &f(i,  j,  k+1); 
 

#define NEXT_STENCIL()                \
  x++; f0++;			      \
  if( x>nmx ) {                       \
    /**/       y++;            x = 2; \
    if( y>nmy) z++; if( y>nmy) y = 2; \
    INIT_STENCIL();                   \
  }

#define CURL_B()								\
  f0->curlbx = py*( fy->cbz - fmy->cbz ) - pz*( fz->cby - fmz->cby ) ;	\
  f0->curlby = pz*( fz->cbx - fmz->cbx ) - px*( fx->cbz - fmx->cbz ) ;	\
  f0->curlbz = px*( fx->cby - fmx->cby ) - py*( fy->cbx - fmy->cbx ) ;


#define LAPLACE_J()\
  f0->ex -= m[f0->cmat].epsy*m[f0->cmat].epsx*g->hypereta*( px*px*( fx->curlbx + fmx->curlbx - 2.0*f0->curlbx ) +    \
			      py*py*( fy->curlbx + fmy->curlbx - 2.0*f0->curlbx ) + \
			      pz*pz*( fz->curlbx + fmz->curlbx - 2.0*f0->curlbx ) ); \
  f0->ey -= m[f0->cmat].epsy*m[f0->cmat].epsx*g->hypereta*( px*px*( fx->curlby + fmx->curlby - 2.0*f0->curlby ) +    \
			      py*py*( fy->curlby + fmy->curlby - 2.0*f0->curlby ) + \
			      pz*pz*( fz->curlby + fmz->curlby - 2.0*f0->curlby ) ); \
  f0->ez -= m[f0->cmat].epsy*m[f0->cmat].epsx*g->hypereta*( px*px*( fx->curlbz + fmx->curlbz - 2.0*f0->curlbz ) +    \
			      py*py*( fy->curlbz + fmy->curlbz - 2.0*f0->curlbz ) + \
			      pz*pz*( fz->curlbz + fmz->curlbz - 2.0*f0->curlbz ) );

#define CURL_D2B()							\
  f0->ex -= m[f0->cmat].epsy*m[f0->cmat].epsx*g->hypereta*(py*( fy->curlbz - fmy->curlbz ) - pz*( fz->curlby - fmz->curlby ) );	\
  f0->ey -= m[f0->cmat].epsy*m[f0->cmat].epsx*g->hypereta*(pz*( fz->curlbx - fmz->curlbx ) - px*( fx->curlbz - fmx->curlbz ) );	\
  f0->ez -= m[f0->cmat].epsy*m[f0->cmat].epsx*g->hypereta*(px*( fx->curlby - fmx->curlby ) - py*( fy->curlbx - fmy->curlbx ) );

#define LAPLACE_B()\
  f0->curlbx = ( px*px*( fx->cbx + fmx->cbx - 2.0*f0->cbx ) +    \
		 py*py*( fy->cbx + fmy->cbx - 2.0*f0->cbx ) + \
		 pz*pz*( fz->cbx + fmz->cbx - 2.0*f0->cbx ) ); \
  f0->curlby = ( px*px*( fx->cby + fmx->cby - 2.0*f0->cby ) +    \
	         py*py*( fy->cby + fmy->cby - 2.0*f0->cby ) + \
		 pz*pz*( fz->cby + fmz->cby - 2.0*f0->cby ) ); \
  f0->curlbz = ( px*px*( fx->cbz + fmx->cbz - 2.0*f0->cbz ) +    \
	         py*py*( fy->cbz + fmy->cbz - 2.0*f0->cbz ) + \
	         pz*pz*( fz->cbz + fmz->cbz - 2.0*f0->cbz ) );


void
hyb_hypereta( field_array_t * RESTRICT fa ) {
  if( !fa     ) ERROR(( "Bad args" ));
  //if( frac!=1 ) ERROR(( "standard advance_e does not support frac!=1 yet" ));

  
  /***************************************************************************
   * Update interior fields CELL CENTERED
   * Note: ex all (0:nx+1,0:ny+1,0,nz+1) interior (2:nx-1,2:ny-1,2:nz-1)
   * Note: ey all (0:nx+1,0:ny+1 0:nz+1) interior (2:nx-1,2:ny-1,2:nz-1)
   * Note: ez all (0:nx+1,0:ny+1,0:nz+1) interior (2:nx-1,2:ny-1,2:nz-1)
   ***************************************************************************/

  // Do majority interior in a single pass.  The host handles
  // stragglers.

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;

  DECLARE_STENCIL();
   
  // Do all here
  for(z=1; z<=nz; z++) {
    for( y=1; y<=ny; y++ ) {
      INIT_EDGE_STENCIL(1,y,z)
      for( x=1; x<=nx; x++ ) {
	LAPLACE_B(); 
	f0++; fmx++; fmy++; fmz++;
	fx++; fy++;  fz++;	  
      }//z
    }//y
  }//x
  
  begin_remote_curlb(fa->f,fa->g);
  end_remote_curlb(fa->f,fa->g);


  // Do all here
  for(z=1; z<=nz; z++) {
    for( y=1; y<=ny; y++ ) {
      INIT_EDGE_STENCIL(1,y,z)
      for( x=1; x<=nx; x++ ) {
	CURL_D2B(); 
	f0++; fmx++; fmy++; fmz++;
	fx++; fy++;  fz++;	  
      }//z
    }//y
  }//x

}

