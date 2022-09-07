#define IN_sfa
//#define HAS_V4_PIPELINE
#include "sfa_private.h"

//#include <iostream>
//#include <fstream>

typedef struct pipeline_args {
  /**/  field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
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
  const float twelfth=1.0/12.0, quarter = 0.25,two=2, six=6.0;	      \
                                                                      \
  float px ;							      \
  float py ;							      \
  float pz ;							      \
                                                                      \
  field_t * ALIGNED(16) f0;					      \
  field_t * ALIGNED(16) fx, * ALIGNED(16) fy, * ALIGNED(16) fz;	      \
  field_t * ALIGNED(16) fmx,* ALIGNED(16) fmy,* ALIGNED(16) fmz;      \
  int x, y, z;							      \
  float ux,uy,uz,rho,hstep,tmp

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

#define INIT_STENCIL()    \
  f0  = &f(x,  y,  z  );  \


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
 

# define COPY_FD(fd)                      \
  for(z=0; z<=nz+1; z++) {		  \
    for( y=0; y<=ny+1; y++ ) {		  \
    INIT_EDGE_STENCIL(0,y,z)		  \
      for( x=0; x<=nx+1; x++ ) {	  \
        f0->tmpsm = f0->fd;		  \
        f0++;				  \
      }					  \
    }					  \
  }

# define SMOOTH_FD(fd)                                                   \
  for(z=1; z<=nz; z++) {		                                 \
    for( y=1; y<=ny; y++ ) {	                                         \
    INIT_EDGE_STENCIL(1,y,z)		                                 \
      for( x=1; x<=nx; x++ ) {						 \
       f0->fd = twelfth*(six*f0->tmpsm + fx->tmpsm + fmx->tmpsm          \
               +fy->tmpsm + fmy->tmpsm + fz->tmpsm + fmz->tmpsm);        \
       f0++;fmx++;fmy++;fmz++;fx++;fy++;fz++;                            \
      }					                                 \
    }					                                 \
  }

# define COPY_B(fd)                      \
  for(z=0; z<=nz+1; z++) {		  \
    for( y=0; y<=ny+1; y++ ) {		  \
    INIT_EDGE_STENCIL(0,y,z)		  \
      for( x=0; x<=nx+1; x++ ) {	  \
        f0->tmpsm = f0->fd-f0->fd##0;	  \
        f0++;				  \
      }					  \
    }					  \
  }

# define SMOOTH_B(fd)                                                   \
  for(z=1; z<=nz; z++) {		                                 \
    for( y=1; y<=ny; y++ ) {	                                         \
    INIT_EDGE_STENCIL(1,y,z)		                                 \
      for( x=1; x<=nx; x++ ) {						 \
       f0->fd = twelfth*(six*f0->tmpsm + fx->tmpsm + fmx->tmpsm          \
               +fy->tmpsm + fmy->tmpsm + fz->tmpsm + fmz->tmpsm) + f0->fd##0;        \
       f0++;fmx++;fmy++;fmz++;fx++;fy++;fz++;                            \
      }					                                 \
    }					                                 \
  }


void
hyb_smooth_eb( field_array_t * RESTRICT fa,
	       const int smoothed) {
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
    
 	  if (smoothed) {
	    COPY_FD(smcbx); SMOOTH_FD(smcbx);
	    COPY_FD(smcby); SMOOTH_FD(smcby);
	    COPY_FD(smcbz); SMOOTH_FD(smcbz);
	    COPY_FD(smex); SMOOTH_FD(smex);
	    COPY_FD(smey); SMOOTH_FD(smey);
	    COPY_FD(smez); SMOOTH_FD(smez);
	  }
	  
	  else {
	    COPY_FD(cbx); SMOOTH_FD(smcbx);
	    COPY_FD(cby); SMOOTH_FD(smcby);
	    COPY_FD(cbz); SMOOTH_FD(smcbz);
	    COPY_FD(ex); SMOOTH_FD(smex);
	    COPY_FD(ey); SMOOTH_FD(smey);
	    COPY_FD(ez); SMOOTH_FD(smez);
	  }
     
   

  begin_remote_smooth_b( fa->f, fa->g );
  end_remote_smooth_b( fa->f, fa->g );
  hyb_local_ghost_b( fa->f, fa->g );

  begin_remote_smooth_e( fa->f, fa->g );
  end_remote_smooth_e( fa->f, fa->g );
  hyb_local_ghost_e( fa->f, fa->g );


}


void
hyb_smooth_b( field_array_t * RESTRICT fa ) {
  if( !fa     ) ERROR(( "Bad args" ));
  //if( frac!=1 ) ERROR(( "standard advance_e does not support frac!=1 yet" ));

  // Do majority interior in a single pass.  The host handles
  // stragglers.

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;

  DECLARE_STENCIL();
    
	    //COPY_FD(ex); SMOOTH_FD(ex);
	    //COPY_FD(ey); SMOOTH_FD(ey);
	    //COPY_FD(ez); SMOOTH_FD(ez);
            COPY_B(cbx); SMOOTH_B(cbx);
	    COPY_B(cby); SMOOTH_B(cby);
	    COPY_B(cbz); SMOOTH_B(cbz);

  //begin_remote_ghost_e( fa->f, fa->g );
  //end_remote_ghost_e( fa->f, fa->g );
  //hyb_local_ghost_e( fa->f, fa->g );

 begin_remote_ghost_b( fa->f, fa->g );
 end_remote_ghost_b( fa->f, fa->g );
 hyb_local_ghost_b( fa->f, fa->g );

}



void
hyb_smooth_moments( field_array_t * RESTRICT fa) {
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

  COPY_FD(jfx); SMOOTH_FD(jfx);
  COPY_FD(jfy); SMOOTH_FD(jfy);
  COPY_FD(jfz); SMOOTH_FD(jfz);
  COPY_FD(rhof); SMOOTH_FD(rhof);
  
  begin_remote_smooth_rho( fa->f, fa->g );
  end_remote_smooth_rho( fa->f, fa->g );

  hyb_local_ghost_rhof(fa->f, fa->g);
  
}
