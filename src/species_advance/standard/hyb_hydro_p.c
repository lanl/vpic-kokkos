// FIXME: THREAD THIS! HYDRO MEM SEMANTICS WILL NEED UPDATING.
// FIXME: V4 ACCELERATE THIS.  COULD BE BASED OFF ENERGY_P.

/* 
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Revised and extended from earlier V4PIC versions
 *
 */

#define IN_spa
#include "spa_private.h"

// accumulate_hydro_p adds the hydrodynamic fields associated with the
// supplied particle_list to the hydro array.  Trilinear interpolation
// is used.  hydro is known at the nodes at the same time as particle
// positions. No effort is made to fix up edges of the computational
// domain.  All particles on the list must be inbounds.  Note, the
// hydro jx,jy,jz are for diagnostic purposes only; they are not
// accumulated with a charge conserving algorithm.

void
hyb_accumulate_hydro_p( hydro_array_t              * RESTRICT ha,
                    const species_t            * RESTRICT sp,
                    const interpolator_array_t * RESTRICT ia ) {
  /**/  hydro_t        * RESTRICT ALIGNED(128) h;
  const particle_t     * RESTRICT ALIGNED(128) p;
  const interpolator_t * RESTRICT ALIGNED(128) f;
  float c, qsp, mspc, qdt_2mc, qdt_4mc2, r12V;
  int np, stride_x, stride_mx, stride_y,stride_my, stride_z, stride_mz;

  float dx, dy, dz, ux, uy, uz, w, vx, vy, vz, ke_mc;
  float w0, w1, w2, w3, w4, w5, w6, w7, t, wx, wy, wz, wmx, wmy, wmz;
  int i, n;
  const float one=1.0, two=2.0, three=3.0;

  if( !ha || !sp || !ia || ha->g!=sp->g || ha->g!=ia->g )
    ERROR(( "Bad args" ));

  h = ha->h;
  p = sp->p;
  f = ia->i;

  c        = sp->g->cvac;
  qsp      = sp->q;
  mspc     = sp->m*c;
  qdt_2mc  = (qsp*sp->g->dt)/(2*mspc);
  qdt_4mc2 = qdt_2mc / (2*c);
  //r8V      = sp->g->r8V;

  r12V      = 1.0/12./(sp->g->dx*sp->g->dy*sp->g->dz);


  np        = sp->np;
  stride_x  = VOXEL(1,0,0, sp->g->nx,sp->g->ny,sp->g->nz) -
              VOXEL(0,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  
  stride_mx = VOXEL(0,0,0, sp->g->nx,sp->g->ny,sp->g->nz) -
              VOXEL(2,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  
  stride_y  = VOXEL(1,1,0, sp->g->nx,sp->g->ny,sp->g->nz) -
              VOXEL(0,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  
  stride_my = VOXEL(0,0,0, sp->g->nx,sp->g->ny,sp->g->nz) -
              VOXEL(0,2,0, sp->g->nx,sp->g->ny,sp->g->nz);
  
  stride_z  = VOXEL(0,1,1, sp->g->nx,sp->g->ny,sp->g->nz) -
              VOXEL(0,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  
  stride_mz = VOXEL(0,0,0, sp->g->nx,sp->g->ny,sp->g->nz) -
              VOXEL(0,0,2, sp->g->nx,sp->g->ny,sp->g->nz);

  for( n=0; n<np; n++ ) {

    // Load the particle
    dx = p[n].dx;
    dy = p[n].dy;
    dz = p[n].dz;
    i  = p[n].i;
    ux = p[n].ux;
    uy = p[n].uy;
    uz = p[n].uz;
    w  = p[n].w;
    
    // Half advance E
    ux += qdt_2mc*( f[i].ex + dx*( f[i].dexdx + dx*f[i].d2exdx ) 
                            + dy*( f[i].dexdy + dy*f[i].d2exdy ) 
		            + dz*( f[i].dexdz + dz*f[i].d2exdz ) );
    uy += qdt_2mc*( f[i].ey + dx*( f[i].deydx + dx*f[i].d2eydx ) 
                            + dy*( f[i].deydy + dy*f[i].d2eydy ) 
		            + dz*( f[i].deydz + dz*f[i].d2eydz ) );
    uz += qdt_2mc*( f[i].ez + dx*( f[i].dezdx + dx*f[i].d2ezdx ) 
                            + dy*( f[i].dezdy + dy*f[i].d2ezdy ) 
		            + dz*( f[i].dezdz + dz*f[i].d2ezdz ) );

    // Boris rotation - Interpolate B field
    w5  = f[i].cbx + dx*( f[i].dcbxdx + dx*f[i].d2cbxdx ) 
                  + dy*( f[i].dcbxdy + dy*f[i].d2cbxdy ) 
                  + dz*( f[i].dcbxdz + dz*f[i].d2cbxdz );   
    w6  = f[i].cby + dx*( f[i].dcbydx + dx*f[i].d2cbydx ) 
                  + dy*( f[i].dcbydy + dy*f[i].d2cbydy ) 
                  + dz*( f[i].dcbydz + dz*f[i].d2cbydz );
    w7  = f[i].cbz + dx*( f[i].dcbzdx + dx*f[i].d2cbzdx ) 
                  + dy*( f[i].dcbzdy + dy*f[i].d2cbzdy ) 
                  + dz*( f[i].dcbzdz + dz*f[i].d2cbzdz ); 

    // Boris rotation - curl scalars (0.5 in v0 for half rotate) and
    // kinetic energy computation. Note: gamma-1 = |u|^2 / (gamma+1)
    // is the numerically accurate way to compute gamma-1
    //ke_mc = ux*ux + uy*uy + uz*uz; // ke_mc = |u|^2 (invariant)
    //vz = sqrt(1+ke_mc);            // vz = gamma    (invariant)
    //ke_mc *= c/(vz+1);             // ke_mc = c|u|^2/(gamma+1) = c*(gamma-1)
    //vz = c/vz;                     // vz = c/gamma
    w0 = qdt_4mc2;
    w1 = w5*w5 + w6*w6 + w7*w7;    // |cB|^2
    w2 = w0*w0*w1;
    w3 = w0*(1+(1./3.)*w2*(1+0.4*w2));
    w4 = w3/(1 + w1*w3*w3); w4 += w4;

    // Boris rotation - uprime
    w0 = ux + w3*( uy*w7 - uz*w6 );
    w1 = uy + w3*( uz*w5 - ux*w7 );
    w2 = uz + w3*( ux*w6 - uy*w5 );

    // Boris rotation - u
    ux += w4*( w1*w7 - w2*w6 );
    uy += w4*( w2*w5 - w0*w7 );
    uz += w4*( w0*w6 - w1*w5 );

    // Compute physical velocities
    vx  = ux;
    vy  = uy;
    vz  = uz;

    // Compute coefficients
      
      w0 =  w*two*( three - dx*dx - dy*dy - dz*dz );
      wx =  w*( dx + one )*( dx + one ); 
      wy =  w*( dy + one )*( dy + one );
      wz =  w*( dz + one )*( dz + one );
      wmx = w*( dx - one )*( dx - one );
      wmy = w*( dy - one )*( dy - one );
      wmz = w*( dz - one )*( dz - one );


    // Accumulate the hydro fields
#   define ACCUM_HYDRO( wn)                             \
    t  = qsp*wn*r12V;   /* t  = (qsp w/V) trilin_n */   \
    h[i].jx  += t*vx;                                   \
    h[i].jy  += t*vy;                                   \
    h[i].jz  += t*vz;                                   \
    h[i].rho += t;                                      \
    t  = mspc*wn*r12V;   /* t = (msp c w/V) trilin_n */	\
    dx = t*ux;          /* dx = (px w/V) trilin_n */    \
    dy = t*uy;                                          \
    dz = t*uz;                                          \
    h[i].px  += dx;                                     \
    h[i].py  += dy;                                     \
    h[i].pz  += dz;                                     \
    h[i].ke  += t;                                      \
    h[i].txx += dx*vx;                                  \
    h[i].tyy += dy*vy;                                  \
    h[i].tzz += dz*vz;                                  \
    h[i].tyz += dy*vz;                                  \
    h[i].tzx += dz*vx;                                  \
    h[i].txy += dx*vy

    /**/            ACCUM_HYDRO(w0);  // Cell i,j,k
    i += stride_x;  ACCUM_HYDRO(wx);  // Cell i+1,j,k
    i += stride_mx; ACCUM_HYDRO(wmx); // Cell i-1,j,k
    i += stride_y;  ACCUM_HYDRO(wy);  // Cell i,j+1,k
    i += stride_my; ACCUM_HYDRO(wmy); // Cell i,j-1,k
    i += stride_z;  ACCUM_HYDRO(wz);  // Cell i,j,k+1
    i += stride_mz; ACCUM_HYDRO(wmz); // Cell i,j,k-1

#   undef ACCUM_HYDRO
  }
}
