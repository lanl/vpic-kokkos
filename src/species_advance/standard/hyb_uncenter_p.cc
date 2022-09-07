#define IN_spa
//#define HAS_V4_PIPELINE
#include "spa_private.h"

#include<iostream>
void
hyb_uncenter_p_pipeline( hyb_uncenter_p_pipeline_args_t * args,
                     int pipeline_rank,
                     int n_pipeline ) {
  const interpolator_t * ALIGNED(128) f0 = args->f0;
  accumulator_t        * ALIGNED(128) a0 = args->a0;

  particle_t           * ALIGNED(32)  p;
  const interpolator_t * ALIGNED(16)  f;
  float                * ALIGNED(16)  a;


  const float qdt_2mc        =     -args->qdt_2mc; // For backward half advance
  const float qdt_4mc        = -0.5*args->qdt_2mc; // For backward half rotate
  const float one            = 1.;
  const float one_third      = 1./3.;
  const float two_fifteenths = 2./15.;
  const int accum            = args->accum; //accumulate or half advance
  const float qsp            = args->qsp;
  const float two = 2., three=3.;


  float dx, dy, dz, ux, uy, uz,q;
  float hax, hay, haz, cbx, cby, cbz;
  float v0, v1, v2, v3, v4;
  float w0,wx,wy,wz,wmx,wmy,wmz;


  int first, ii, n;

  // Determine which particles this pipeline processes

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, first, n );
  p = args->p0 + first;

 // Determine which accumulator array to use
  // The host gets the first accumulator array

  if( pipeline_rank!=n_pipeline )
    a0 += (1+pipeline_rank)*
          POW2_CEIL((args->nx+2)*(args->ny+2)*(args->nz+2),2);


  // Process particles for this pipeline

  for(;n;n--,p++) {
    dx   = p->dx;                            // Load position
    dy   = p->dy;
    dz   = p->dz;
    ii   = p->i;
    f    = f0 + ii;  
    q    = p->w;     
    ux   = p->ux;                            // Load momentum
    uy   = p->uy;
    uz   = p->uz;

    switch (accum) {

    case 0: {
                   
   // Interpolate E 
    hax  = qdt_2mc*( f->ex + dx*( f->dexdx + dx*f->d2exdx ) 
                           + dy*( f->dexdy + dy*f->d2exdy ) 
                           + dz*( f->dexdz + dz*f->d2exdz ) );                            
    hay  = qdt_2mc*( f->ey + dx*( f->deydx + dx*f->d2eydx ) 
                           + dy*( f->deydy + dy*f->d2eydy ) 
                           + dz*( f->deydz + dz*f->d2eydz ) );
    haz  = qdt_2mc*( f->ez + dx*( f->dezdx + dx*f->d2ezdx ) 
                           + dy*( f->dezdy + dy*f->d2ezdy ) 
                           + dz*( f->dezdz + dz*f->d2ezdz ) );     
    // Interpolate B
    cbx  = f->cbx + dx*( f->dcbxdx + dx*f->d2cbxdx ) 
                  + dy*( f->dcbxdy + dy*f->d2cbxdy ) 
                  + dz*( f->dcbxdz + dz*f->d2cbxdz );   
    cby  = f->cby + dx*( f->dcbydx + dx*f->d2cbydx ) 
                  + dy*( f->dcbydy + dy*f->d2cbydy ) 
                  + dz*( f->dcbydz + dz*f->d2cbydz );
    cbz  = f->cbz + dx*( f->dcbzdx + dx*f->d2cbzdx ) 
                  + dy*( f->dcbzdy + dy*f->d2cbzdy ) 
                  + dz*( f->dcbzdz + dz*f->d2cbzdz ); 
    
    v0   = qdt_4mc;///(float)sqrt(one + (ux*ux + (uy*uy + uz*uz)));
    /**/                                     // Boris - scalars
    v1   = cbx*cbx + (cby*cby + cbz*cbz);
    v2   = (v0*v0)*v1;
    v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
    v4   = v3/(one+v1*(v3*v3));
    v4  += v4;
    v0   = ux + v3*( uy*cbz - uz*cby );      // Boris - uprime
    v1   = uy + v3*( uz*cbx - ux*cbz );
    v2   = uz + v3*( ux*cby - uy*cbx );
    ux  += v4*( v1*cbz - v2*cby );           // Boris - rotation
    uy  += v4*( v2*cbx - v0*cbz );
    uz  += v4*( v0*cby - v1*cbx );
    ux  += hax;                              // Half advance E
    uy  += hay;
    uz  += haz;
    p->ux = ux;                              // Store momentum
    p->uy = uy;
    p->uz = uz;
    break;
    }

    case 1: {
      
      a  = (float *)( a0 + ii );              // Get accumulator

      q *= qsp;
      
      // Compute coefficients
      
      w0 =  q*two*(three - dx*dx - dy*dy - dz*dz );
      wx =  q*( dx + one )*( dx + one ); 
      wy =  q*( dy + one )*( dy + one );
      wz =  q*( dz + one )*( dz + one );
      wmx = q*( dx - one )*( dx - one );
      wmy = q*( dy - one )*( dy - one );
      wmz = q*( dz - one )*( dz - one );
          

      // std::cout << "unc w0=" << w0 << "\n";

    
    // Accumulate the particle charge density
   
#     define ACCUMULATE_J(X,offset)					       \
      a[0+offset]+= w0*u##X;					               \
      a[1+offset]+= wmx*u##X; a[2+offset]+= wmy*u##X; a[3+offset]+= wmz*u##X;  \
      a[4+offset]+= wx*u##X;  a[5+offset]+= wy*u##X;  a[6+offset]+= wz*u##X
       
      ACCUMULATE_J( x, 0 );
      ACCUMULATE_J( y, 7 );
      ACCUMULATE_J( z, 14 );

      a[21]+= w0; a[22]+= wmx;  a[23]+= wmy;  a[24]+= wmz;
               a[25]+= wx; a[26]+= wy; a[27]+= wz;

#     undef ACCUMULATE_J
	      break;
    }
      default: break;
    }
    }
}


#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

using namespace v4;

void
hyb_uncenter_p_pipeline_v4( center_p_pipeline_args_t * args,
                        int pipeline_rank,
                        int n_pipeline ) {
  const interpolator_t * ALIGNED(128) f0  = args->f0;

  particle_t           * ALIGNED(128) p;
  const float          * ALIGNED(16)  vp0;
  const float          * ALIGNED(16)  vp1;
  const float          * ALIGNED(16)  vp2;
  const float          * ALIGNED(16)  vp3;

  const v4float qdt_2mc(    -args->qdt_2mc); // For backward half advance
  const v4float qdt_4mc(-0.5*args->qdt_2mc); // For backward half Boris rotate
  const v4float one(1.);
  const v4float one_third(1./3.);
  const v4float two_fifteenths(2./15.);

  v4float dx, dy, dz, ux, uy, uz, q;
  v4float hax, hay, haz, cbx, cby, cbz;
  v4float v0, v1, v2, v3, v4, v5;
  v4int ii;

  int first, nq;

  // Determine which particle quads this pipeline processes

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, first, nq );
  p = args->p0 + first;
  nq >>= 2;

  // Process the particle quads for this pipeline

  for( ; nq; nq--, p+=4 ) {
    load_4x4_tr(&p[0].dx,&p[1].dx,&p[2].dx,&p[3].dx,dx,dy,dz,ii);

    // Interpolate fields
    vp0 = (const float * ALIGNED(16))(f0 + ii(0));
    vp1 = (const float * ALIGNED(16))(f0 + ii(1));
    vp2 = (const float * ALIGNED(16))(f0 + ii(2));
    vp3 = (const float * ALIGNED(16))(f0 + ii(3));
    load_4x4_tr(vp0,  vp1,  vp2,  vp3,  hax,v0,v1,v2); hax = qdt_2mc*fma( fma( dy, v2, v1 ), dz, fma( dy, v0, hax ) );
    load_4x4_tr(vp0+4,vp1+4,vp2+4,vp3+4,hay,v3,v4,v5); hay = qdt_2mc*fma( fma( dz, v5, v4 ), dx, fma( dz, v3, hay ) );
    load_4x4_tr(vp0+8,vp1+8,vp2+8,vp3+8,haz,v0,v1,v2); haz = qdt_2mc*fma( fma( dx, v2, v1 ), dy, fma( dx, v0, haz ) );
    load_4x4_tr(vp0+12,vp1+12,vp2+12,vp3+12,cbx,v3,cby,v4); cbx = fma( v3, dx, cbx );
    /**/                                                    cby = fma( v4, dy, cby );
    load_4x2_tr(vp0+16,vp1+16,vp2+16,vp3+16,cbz,v5);        cbz = fma( v5, dz, cbz );

    // Update momentum
    load_4x4_tr(&p[0].ux,&p[1].ux,&p[2].ux,&p[3].ux,ux,uy,uz,q);
    /**/                                              // Could use load_4x3_tr
    v0  = qdt_4mc*rsqrt( one + fma( ux,ux, fma( uy,uy, uz*uz ) ) );
    v1  = fma( cbx,cbx, fma( cby,cby, cbz*cbz ) );
    v2  = (v0*v0)*v1;
    v3  = v0*fma( v2, fma( v2, two_fifteenths, one_third ), one );
    v4  = v3*rcp( fma( v3*v3, v1, one ) ); v4 += v4;
    v0  = fma( fms( uy,cbz, uz*cby ), v3, ux );
    v1  = fma( fms( uz,cbx, ux*cbz ), v3, uy );
    v2  = fma( fms( ux,cby, uy*cbx ), v3, uz );
    ux  = fma( fms( v1,cbz, v2*cby ), v4, ux );
    uy  = fma( fms( v2,cbx, v0*cbz ), v4, uy );
    uz  = fma( fms( v0,cby, v1*cbx ), v4, uz );
    ux += hax;
    uy += hay;
    uz += haz;
    store_4x4_tr(ux,uy,uz,q,&p[0].ux,&p[1].ux,&p[2].ux,&p[3].ux);
    /**/                                              // Could use store_4x3_tr
  }
}

#endif

void
hyb_uncenter_p( /**/  species_t            * RESTRICT sp,
		      accumulator_array_t  * RESTRICT aa, 
		const interpolator_array_t * RESTRICT ia,
		const int                             accum) {
   DECLARE_ALIGNED_ARRAY( hyb_uncenter_p_pipeline_args_t, 128, args, 1 );
  //DECLARE_ALIGNED_ARRAY( particle_mover_seg_t, 128, seg, MAX_PIPELINE+1 );

  int rank;

  if( !sp || !aa || !ia || sp->g!=aa->g || sp->g!=ia->g )
    ERROR(( "Bad args" ));

  args->p0       = sp->p;
  args->pm       = sp->pm;
  args->a0       = aa->a;
  args->f0       = ia->i;
  args->g        = sp->g;

  args->qdt_2mc  = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
  args->cdt_dx   = sp->g->cvac*sp->g->dt*sp->g->rdx;
  args->cdt_dy   = sp->g->cvac*sp->g->dt*sp->g->rdy;
  args->cdt_dz   = sp->g->cvac*sp->g->dt*sp->g->rdz;
  args->qsp      = sp->q;

  args->np       = sp->np;
  args->max_nm   = sp->max_nm;
  args->nx       = sp->g->nx;
  args->ny       = sp->g->ny;
  args->nz       = sp->g->nz;
  args->accum    = accum;

  EXEC_PIPELINES( hyb_uncenter_p, args, 0 );
  WAIT_PIPELINES();
}
