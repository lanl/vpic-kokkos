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
  const int nmx = nx-1;						      \
  const int nmy=  ny-1;	                        		      \
  const int nmz = nz-1;						      \
                                                                      \
  float px ;							      \
  float py ;							      \
  float pz ;							      \
                                                                      \
  field_t * ALIGNED(16) f0;                                           \
  int x, y, z;							      \
  float ux,uy,uz,rho,hstep

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

#define INIT_STENCIL()    \
  f0  = &f(x,  y,  z  );  \


//ARI
#define INIT_EDGE_STENCIL(i,j,k)  \
  f0  = &f(i,  j,  k  );          
 

#define NEXT_STENCIL()                \
  x++; f0++;			      \
  if( x>nmx ) {                       \
    /**/       y++;            x = 2; \
    if( y>nmy) z++; if( y>nmy) y = 2; \
    INIT_STENCIL();                   \
  }

//rho has floor of 0.005; hstep=0 gives rho_n, hstep=1 give rho_n+1

#define RHOHS()								\
  rho =(1.0 - hstep)*f0->rhofold + ( hstep )*(  f0->rhof );		\
  rho = (rho>0.005) ? rho : 0.005;					\
 

#define UPDATE_PE() \
  f0->pexx = te*(pow(rho/g->den,g->gamma))


void
hyb_epress( field_array_t * RESTRICT fa,
                  float frac ) {
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
  args->hstep=frac;

  DECLARE_STENCIL();
  hstep=frac;
  //have to fix for nx, ny, or nz=1
  if (nmx*nmy*nmz) {
  
  
  } 
   
  // Do all here
  for(z=0; z<=nz+1; z++) {
    for( y=0; y<=ny+1; y++ ) {
      INIT_EDGE_STENCIL(0,y,z)
	for( x=0; x<=nx+1; x++ ) {
	  RHOHS();
	  UPDATE_PE();
	  
	  //if(y==1 && x==nx) std::cout << "jzgho  " << fx->jfz << "     jz " << f0->jfz << std::endl  ;
	  //if(y==1 && x ==1) std::cout << "uz  " << uz << "      cbx " << f0->cbx << std::endl  ;
	  
	  f0++;
	  
       }
     }
   }


}

