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
accumulate_hydro_p( hydro_array_t              * RESTRICT ha,
                    const species_t            * RESTRICT sp,
                    const interpolator_array_t * RESTRICT ia ) {
  /**/  hydro_t        * RESTRICT ALIGNED(128) h;
  const particle_t     * RESTRICT ALIGNED(128) p;
  const interpolator_t * RESTRICT ALIGNED(128) f;
  float c, qsp, msp, qdt_2mc, qdt_4mc, rV;
  int np, stride_10, stride_21, stride_43;

  float dx, dy, dz, ux, uy, uz, w;
  float w0, w1, w2, w3, w4, w5, w6, w7, t;
  int i, n;

  if( !ha || !sp || !ia || ha->g!=sp->g || ha->g!=ia->g )
    ERROR(( "Bad args" ));

  h = ha->h;
  p = sp->p;
  f = ia->i;

  c        = sp->g->cvac;
  qsp      = sp->q;
  //mspc     = sp->m*c;                   // rel push
  //qdt_2mc  = (qsp*sp->g->dt)/(2*mspc);
  //qdt_4mc2 = qdt_2mc / (2*c);
  msp      = sp->m;                       // non-rel push
  qdt_2mc  = (qsp*sp->g->dt)/(2*msp*c);
  qdt_4mc  = qdt_2mc / 2;
  rV        = 1.0/(sp->g->dx*sp->g->dy*sp->g->dz);

  np        = sp->np;
  stride_10 = VOXEL(1,0,0, sp->g->nx,sp->g->ny,sp->g->nz) -
              VOXEL(0,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  stride_21 = VOXEL(0,1,0, sp->g->nx,sp->g->ny,sp->g->nz) -
              VOXEL(1,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  stride_43 = VOXEL(0,0,1, sp->g->nx,sp->g->ny,sp->g->nz) -
              VOXEL(1,1,0, sp->g->nx,sp->g->ny,sp->g->nz);

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

#ifdef SHAPE_NGP
    // Half advance E
    ux += qdt_2mc*f[i].ex;
    uy += qdt_2mc*f[i].ey;
    uz += qdt_2mc*f[i].ez;
    // Boris rotation - Interpolate B field
    w5 = f[i].cbx;
    w6 = f[i].cby;
    w7 = f[i].cbz;
#else
#ifdef SHAPE_QS
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
    w5 = f[i].cbx + dx*( f[i].dcbxdx + dx*f[i].d2cbxdx )
                  + dy*( f[i].dcbxdy + dy*f[i].d2cbxdy )
                  + dz*( f[i].dcbxdz + dz*f[i].d2cbxdz );
    w6 = f[i].cby + dx*( f[i].dcbydx + dx*f[i].d2cbydx )
                  + dy*( f[i].dcbydy + dy*f[i].d2cbydy )
                  + dz*( f[i].dcbydz + dz*f[i].d2cbydz );
    w7 = f[i].cbz + dx*( f[i].dcbzdx + dx*f[i].d2cbzdx )
                  + dy*( f[i].dcbzdy + dy*f[i].d2cbzdy )
                  + dz*( f[i].dcbzdz + dz*f[i].d2cbzdz );
#endif
#endif

    // Boris rotation - curl scalars (0.5 in v0 for half rotate) and
    // kinetic energy computation. Note: gamma-1 = |u|^2 / (gamma+1)
    // is the numerically accurate way to compute gamma-1
    //ke_mc = ux*ux + uy*uy + uz*uz; // ke_mc = |u|^2 (invariant)
    //vz = 1;//sqrt(1+ke_mc);            // vz = gamma    (invariant)
    //ke_mc *= c/(vz+1);             // ke_mc = c|u|^2/(gamma+1) = c*(gamma-1)
    //vz = c/vz;                     // vz = c/gamma
    //w0 = qdt_4mc2*vz;
    w0 = qdt_4mc;  // non-rel push
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

    // Compute physical three-velocities
    //vx  = ux*vz;  // rel push
    //vy  = uy*vz;
    //vz *= uz;
    ux *= c;        // non-rel push
    uy *= c;
    uz *= c;

    // Compute the trilinear coefficients
    //w0  = r8V*w;    // w0 = (1/8)(w/V)
    //dx *= w0;       // dx = (1/8)(w/V) x
    //w1  = w0+dx;    // w1 = (1/8)(w/V) + (1/8)(w/V)x = (1/8)(w/V)(1+x)
    //w0 -= dx;       // w0 = (1/8)(w/V) - (1/8)(w/V)x = (1/8)(w/V)(1-x)
    //w3  = 1+dy;     // w3 = 1+y
    //w2  = w0*w3;    // w2 = (1/8)(w/V)(1-x)(1+y)
    //w3 *= w1;       // w3 = (1/8)(w/V)(1+x)(1+y)
    //dy  = 1-dy;     // dy = 1-y
    //w0 *= dy;       // w0 = (1/8)(w/V)(1-x)(1-y)
    //w1 *= dy;       // w1 = (1/8)(w/V)(1+x)(1-y)
    //w7  = 1+dz;     // w7 = 1+z
    //w4  = w0*w7;    // w4 = (1/8)(w/V)(1-x)(1-y)(1+z) = (w/V) trilin_0 *Done
    //w5  = w1*w7;    // w5 = (1/8)(w/V)(1+x)(1-y)(1+z) = (w/V) trilin_1 *Done
    //w6  = w2*w7;    // w6 = (1/8)(w/V)(1-x)(1+y)(1+z) = (w/V) trilin_2 *Done
    //w7 *= w3;       // w7 = (1/8)(w/V)(1+x)(1+y)(1+z) = (w/V) trilin_3 *Done
    //dz  = 1-dz;     // dz = 1-z
    //w0 *= dz;       // w0 = (1/8)(w/V)(1-x)(1-y)(1-z) = (w/V) trilin_4 *Done
    //w1 *= dz;       // w1 = (1/8)(w/V)(1+x)(1-y)(1-z) = (w/V) trilin_5 *Done
    //w2 *= dz;       // w2 = (1/8)(w/V)(1-x)(1+y)(1-z) = (w/V) trilin_6 *Done
    //w3 *= dz;       // w3 = (1/8)(w/V)(1+x)(1+y)(1-z) = (w/V) trilin_7 *Done

    // Hybrid-VPIC NGP shape
    w0 = w*rV;

    // Accumulate the hydro fields - relativistic version
//#   define ACCUM_HYDRO( wn)                             \
//    t  = qsp*wn;        /* t  = (qsp w/V) trilin_n */   \
//    h[i].jx  += t*vx;                                   \
//    h[i].jy  += t*vy;                                   \
//    h[i].jz  += t*vz;                                   \
//    h[i].rho += t;                                      \
//    t  = mspc*wn;       /* t = (msp c w/V) trilin_n */  \
//    dx = t*ux;          /* dx = (px w/V) trilin_n */    \
//    dy = t*uy;                                          \
//    dz = t*uz;                                          \
//    h[i].px  += dx;                                     \
//    h[i].py  += dy;                                     \
//    h[i].pz  += dz;                                     \
//    h[i].ke  += t*ke_mc;                                \
//    h[i].txx += dx*vx;                                  \
//    h[i].tyy += dy*vy;                                  \
//    h[i].tzz += dz*vz;                                  \
//    h[i].tyz += dy*vz;                                  \
//    h[i].tzx += dz*vx;                                  \
//    h[i].txy += dx*vy

    // Accumulate the hydro fields - non-relativistic version
#   define ACCUM_HYDRO( wn)                             \
    t  = qsp*wn;        /* t  = (qsp w/V) trilin_n */   \
    h[i].jx  += t*ux;                                   \
    h[i].jy  += t*uy;                                   \
    h[i].jz  += t*uz;                                   \
    h[i].rho += t;                                      \
    t  = msp*wn;        /* t = (msp w/V) trilin_n */    \
    dx = t*ux;          /* dx = (px w/V) trilin_n */    \
    dy = t*uy;                                          \
    dz = t*uz;                                          \
    h[i].px  += dx;                                     \
    h[i].py  += dy;                                     \
    h[i].pz  += dz;                                     \
    h[i].rho_m  += t; /* Prev. was *ke_mc; */           \
    h[i].txx += dx*ux;                                  \
    h[i].tyy += dy*uy;                                  \
    h[i].tzz += dz*uz;                                  \
    h[i].tyz += dy*uz;                                  \
    h[i].tzx += dz*ux;                                  \
    h[i].txy += dx*uy

    /**/            ACCUM_HYDRO(w0); // Cell i,j,k
//  i += stride_10; ACCUM_HYDRO(w1); // Cell i+1,j,k
//  i += stride_21; ACCUM_HYDRO(w2); // Cell i,j+1,k
//  i += stride_10; ACCUM_HYDRO(w3); // Cell i+1,j+1,k
//  i += stride_43; ACCUM_HYDRO(w4); // Cell i,j,k+1
//  i += stride_10; ACCUM_HYDRO(w5); // Cell i+1,j,k+1
//  i += stride_21; ACCUM_HYDRO(w6); // Cell i,j+1,k+1
//  i += stride_10; ACCUM_HYDRO(w7); // Cell i+1,j+1,k+1

#   undef ACCUM_HYDRO
  }
}

void
accumulate_hydro_p_kokkos_nomove_ngp(
        //hydro_array_t              * RESTRICT ha,
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_hydro_d_t k_hydro,
        //k_hydro_sv_t k_hydro_sv, // don't need, can do locally
        k_interpolator_t& k_interp,
        const species_t            * RESTRICT sp
)
{
  k_hydro_sv_t k_hydro_sv = Kokkos::Experimental::create_scatter_view(k_hydro);

  float c, qsp, mspc, qdt_2mc, qdt_4mc2, r8V;

  int nv = sp->g->nv; // TODO: delete

  if( !sp ) {
    ERROR(( "Bad args" ));
  }

  c        = sp->g->cvac;
  qsp      = sp->q;
  mspc     = sp->m*c;
  qdt_2mc  = (qsp*sp->g->dt)/(2*mspc);
  qdt_4mc2 = qdt_2mc / (2*c);
  r8V      = sp->g->r8V;

  const int np        = sp->np;
  const int stride_10 = VOXEL(1,0,0, sp->g->nx,sp->g->ny,sp->g->nz) -
                        VOXEL(0,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  const int stride_21 = VOXEL(0,1,0, sp->g->nx,sp->g->ny,sp->g->nz) -
                        VOXEL(1,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  const int stride_43 = VOXEL(0,0,1, sp->g->nx,sp->g->ny,sp->g->nz) -
                        VOXEL(1,1,0, sp->g->nx,sp->g->ny,sp->g->nz);


  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np),
    KOKKOS_LAMBDA (size_t p_index)
    {

    // Load the particle
    double dx = k_particles(p_index, particle_var::dx);
    double dy = k_particles(p_index, particle_var::dy);
    double dz = k_particles(p_index, particle_var::dz);
    double ux = k_particles(p_index, particle_var::ux);
    double uy = k_particles(p_index, particle_var::uy);
    double uz = k_particles(p_index, particle_var::uz);
    double w  = k_particles(p_index, particle_var::w);
    int ii = k_particles_i(p_index);

    double ke_mc = static_cast<double>(ux)*static_cast<double>(ux) + static_cast<double>(uy)*static_cast<double>(uy) + static_cast<double>(uz)*static_cast<double>(uz); // ke_mc = |u|^2 (invariant)
    double vz = 1.0;//sqrt(1.0+ke_mc);            // vz = gamma    (invariant)    
    ke_mc *= c/(vz+1.0);             // ke_mc = c|u|^2/(gamma+1) = c*(gamma-1)
    
    // Compute physical velocities
    float vx  = ux*vz;
    float vy  = uy*vz;
    vz *= uz;

    float t = 0.0; // used in macro
    auto k_hydro_access = k_hydro_sv.access();

    // Accumulate the hydro fields
    #define ACCUM_HYDRO( wn, i )                        \
    t  = qsp*wn;        /* t  = (qsp w/V) trilin_n */   \
    k_hydro_access(i, hydro_var::jx)  += t*vx;                       \
    k_hydro_access(i, hydro_var::jy)  += t*vy;                       \
    k_hydro_access(i, hydro_var::jz)  += t*vz;                       \
    k_hydro_access(i, hydro_var::rho) += t;                          \
    t  = mspc*wn;       /* t = (msp c w/V) trilin_n */  \
    dx = t*ux;          /* dx = (px w/V) trilin_n */    \
    dy = t*uy;                                          \
    dz = t*uz;                                          \
    k_hydro_access(i, hydro_var::px)  += dx;                         \
    k_hydro_access(i, hydro_var::py)  += dy;                         \
    k_hydro_access(i, hydro_var::pz)  += dz;                         \
    k_hydro_access(i, hydro_var::rho_m)  += t;	/* Prev. was *ke_mc; */	     \
    k_hydro_access(i, hydro_var::txx) += dx*vx;                      \
    k_hydro_access(i, hydro_var::tyy) += dy*vy;                      \
    k_hydro_access(i, hydro_var::tzz) += dz*vz;                      \
    k_hydro_access(i, hydro_var::tyz) += dy*vz;                      \
    k_hydro_access(i, hydro_var::tzx) += dz*vx;                      \
    k_hydro_access(i, hydro_var::txy) += dx*vy;

    // TODO: this serial adding to try and save adds is a bit sad
    // TODO: This is somehow going out of bounds right now
    const int i0 = ii;
    ACCUM_HYDRO(w, i0); // Cell i,j,k
#   undef ACCUM_HYDRO
    // printf("i0-7=%d,%d,%d,%d,%d,%d,%d,%d\n",i0,i1,i2,i3,i4,i5,i6,i7);
    // printf("w0-7=%e,%e,%e,%e,%e,%e,%e,%e\n",ke_mc*w0,ke_mc*w1,ke_mc*w2,ke_mc*w3,ke_mc*w4,ke_mc*w5,ke_mc*w6,ke_mc*w7);
    //printf("interpolator=%e,%e,%e,%e,%e,%e,%e,%e\n",ex,ey,ez,dexdy,cbx,cby,cbz,dcbxdx);
  });

  Kokkos::Experimental::contribute(k_hydro, k_hydro_sv);
  Kokkos::fence(); // TODO: Check if I need this to block the contribute

  // Perform debug printing
}


void
accumulate_hydro_p_kokkos(
        //hydro_array_t              * RESTRICT ha,
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_hydro_d_t k_hydro,
        //k_hydro_sv_t k_hydro_sv, // don't need, can do locally
        k_interpolator_t& k_interp,
        const species_t            * RESTRICT sp
)
{
  k_hydro_sv_t k_hydro_sv = Kokkos::Experimental::create_scatter_view(k_hydro);

#ifdef VARIABLE_CHARGE
  float c, qsp, msp, dt_2mc, dt_4mc, rV, r12V;
#else
  float c, qsp, msp, qdt_2mc, qdt_4mc, rV, r12V;
#endif

  constexpr float one=1.0, two=2.0, three=3.0;

  //int np, stride_10, stride_21, stride_43;

  //float dx, dy, dz, ux, uy, uz, w, vx, vy, vz, ke_mc;
  //float w0, w1, w2, w3, w4, w5, w6, w7, t;
  //int i, n;
  //
  int nv = sp->g->nv; // TODO: delete

  if( !sp ) {
    ERROR(( "Bad args" ));
  }

  c        = sp->g->cvac;
  qsp      = sp->q;
  //mspc     = sp->m*c;                   // rel push
  //qdt_2mc  = (qsp*sp->g->dt)/(2*mspc);
  //qdt_4mc2 = qdt_2mc / (2*c);
  msp      = sp->m;                       // non-rel push
#ifdef VARIABLE_CHARGE
  dt_2mc   = (sp->g->dt)/(2*msp*c); // Multiply by particle q later
  dt_4mc   = dt_2mc / 2;
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace> particle_count("particle_count", nv);

  // Set initial values to min_q
  Kokkos::parallel_for("calculate_mean_q", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nv),
    KOKKOS_LAMBDA(size_t ii)
    {
        k_hydro(ii, hydro_var::min_q) = 999999999;
    });

#else
  qdt_2mc  = (qsp*sp->g->dt)/(2*msp*c);
  qdt_4mc  = qdt_2mc / 2;
#endif
  rV        = 1.0/(sp->g->dx*sp->g->dy*sp->g->dz);
  r12V      = rV/12.;

  const int np        = sp->np;
  //const int stride_10 = VOXEL(1,0,0, sp->g->nx,sp->g->ny,sp->g->nz) -
  //                      VOXEL(0,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  //const int stride_21 = VOXEL(0,1,0, sp->g->nx,sp->g->ny,sp->g->nz) -
  //                      VOXEL(1,0,0, sp->g->nx,sp->g->ny,sp->g->nz);
  //const int stride_43 = VOXEL(0,0,1, sp->g->nx,sp->g->ny,sp->g->nz) -
  //                      VOXEL(1,1,0, sp->g->nx,sp->g->ny,sp->g->nz);
  const int sy = sp->g->sy;
  const int sz = sp->g->sz;

  //for( n=0; n<np; n++ ) {
  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np),
    KOKKOS_LAMBDA (size_t p_index)
    {

    // Load the particle
    float dx = k_particles(p_index, particle_var::dx);
    float dy = k_particles(p_index, particle_var::dy);
    float dz = k_particles(p_index, particle_var::dz);
    float ux = k_particles(p_index, particle_var::ux);
    float uy = k_particles(p_index, particle_var::uy);
    float uz = k_particles(p_index, particle_var::uz);
    float w  = k_particles(p_index, particle_var::w);
#ifdef VARIABLE_CHARGE
    float qp = k_particles(p_index, particle_var::qp);
    float qdt_2mc = qp*dt_2mc;
    float qdt_4mc = qp*dt_4mc;
#endif
    int ii = k_particles_i(p_index);

    #define f_ex       k_interp(ii, interpolator_var::ex)
    #define f_dexdx    k_interp(ii, interpolator_var::dexdx)
    #define f_dexdy    k_interp(ii, interpolator_var::dexdy)
    #define f_dexdz    k_interp(ii, interpolator_var::dexdz)
    #define f_d2exdx   k_interp(ii, interpolator_var::d2exdx)
    #define f_d2exdy   k_interp(ii, interpolator_var::d2exdy)
    #define f_d2exdz   k_interp(ii, interpolator_var::d2exdz)
    #define f_ey       k_interp(ii, interpolator_var::ey)
    #define f_deydx    k_interp(ii, interpolator_var::deydx)
    #define f_deydy    k_interp(ii, interpolator_var::deydy)
    #define f_deydz    k_interp(ii, interpolator_var::deydz)
    #define f_d2eydx   k_interp(ii, interpolator_var::d2eydx)
    #define f_d2eydy   k_interp(ii, interpolator_var::d2eydy)
    #define f_d2eydz   k_interp(ii, interpolator_var::d2eydz)
    #define f_ez       k_interp(ii, interpolator_var::ez)
    #define f_dezdx    k_interp(ii, interpolator_var::dezdx)
    #define f_dezdy    k_interp(ii, interpolator_var::dezdy)
    #define f_dezdz    k_interp(ii, interpolator_var::dezdz)
    #define f_d2ezdx   k_interp(ii, interpolator_var::d2ezdx)
    #define f_d2ezdy   k_interp(ii, interpolator_var::d2ezdy)
    #define f_d2ezdz   k_interp(ii, interpolator_var::d2ezdz)
    #define f_cbx      k_interp(ii, interpolator_var::cbx)
    #define f_dcbxdx   k_interp(ii, interpolator_var::dcbxdx)
    #define f_dcbxdy   k_interp(ii, interpolator_var::dcbxdy)
    #define f_dcbxdz   k_interp(ii, interpolator_var::dcbxdz)
    #define f_d2cbxdx  k_interp(ii, interpolator_var::d2cbxdx)
    #define f_d2cbxdy  k_interp(ii, interpolator_var::d2cbxdy)
    #define f_d2cbxdz  k_interp(ii, interpolator_var::d2cbxdz)
    #define f_cby      k_interp(ii, interpolator_var::cby)
    #define f_dcbydx   k_interp(ii, interpolator_var::dcbydx)
    #define f_dcbydy   k_interp(ii, interpolator_var::dcbydy)
    #define f_dcbydz   k_interp(ii, interpolator_var::dcbydz)
    #define f_d2cbydx  k_interp(ii, interpolator_var::d2cbydx)
    #define f_d2cbydy  k_interp(ii, interpolator_var::d2cbydy)
    #define f_d2cbydz  k_interp(ii, interpolator_var::d2cbydz)
    #define f_cbz      k_interp(ii, interpolator_var::cbz)
    #define f_dcbzdx   k_interp(ii, interpolator_var::dcbzdx)
    #define f_dcbzdy   k_interp(ii, interpolator_var::dcbzdy)
    #define f_dcbzdz   k_interp(ii, interpolator_var::dcbzdz)
    #define f_d2cbzdx  k_interp(ii, interpolator_var::d2cbzdx)
    #define f_d2cbzdy  k_interp(ii, interpolator_var::d2cbzdy)
    #define f_d2cbzdz  k_interp(ii, interpolator_var::d2cbzdz)

#ifdef SHAPE_NGP
    // Half advance E
    ux += qdt_2mc * f_ex;
    uy += qdt_2mc * f_ey;
    uz += qdt_2mc * f_ez;
    // Boris rotation - Interpolate B field
    float w5 = f_cbx;
    float w6 = f_cby;
    float w7 = f_cbz;
#else
#ifdef SHAPE_QS
    // Half advance E
    ux += qdt_2mc*( f_ex + dx*( f_dexdx + dx*f_d2exdx )
                         + dy*( f_dexdy + dy*f_d2exdy )
                         + dz*( f_dexdz + dz*f_d2exdz ) );
    uy += qdt_2mc*( f_ey + dx*( f_deydx + dx*f_d2eydx )
                         + dy*( f_deydy + dy*f_d2eydy )
                         + dz*( f_deydz + dz*f_d2eydz ) );
    uz += qdt_2mc*( f_ez + dx*( f_dezdx + dx*f_d2ezdx )
                         + dy*( f_dezdy + dy*f_d2ezdy )
                         + dz*( f_dezdz + dz*f_d2ezdz ) );
    // Boris rotation - Interpolate B field
    float w5 = f_cbx + dx*( f_dcbxdx + dx*f_d2cbxdx )
                     + dy*( f_dcbxdy + dy*f_d2cbxdy )
                     + dz*( f_dcbxdz + dz*f_d2cbxdz );
    float w6 = f_cby + dx*( f_dcbydx + dx*f_d2cbydx )
                     + dy*( f_dcbydy + dy*f_d2cbydy )
                     + dz*( f_dcbydz + dz*f_d2cbydz );
    float w7 = f_cbz + dx*( f_dcbzdx + dx*f_d2cbzdx )
                     + dy*( f_dcbzdy + dy*f_d2cbzdy )
                     + dz*( f_dcbzdz + dz*f_d2cbzdz );
#endif
#endif

    // Boris rotation - curl scalars (0.5 in v0 for half rotate) and
    // kinetic energy computation. Note: gamma-1 = |u|^2 / (gamma+1)
    // is the numerically accurate way to compute gamma-1
    //float ke_mc = ux*ux + uy*uy + uz*uz; // ke_mc = |u|^2 (invariant)
    //float vz = 1; //sqrt(1.0+ke_mc);            // vz = gamma    (invariant)
    //ke_mc *= c/(vz+1.0);             // ke_mc = c|u|^2/(gamma+1) = c*(gamma-1)
    //vz = c/vz;                     // vz = c/gamma
    //float w0 = qdt_4mc2*vz;
    float w0 = qdt_4mc;  // non-rel push
    float w1 = w5*w5 + w6*w6 + w7*w7;    // |cB|^2
    float w2 = w0*w0*w1;
    float w3 = w0*(1.+(1./3.)*w2*(1.0+0.4*w2));
    float w4 = w3/(1.+ w1*w3*w3); w4 += w4;

    // Boris rotation - uprime
    w0 = ux + w3*( uy*w7 - uz*w6 );
    w1 = uy + w3*( uz*w5 - ux*w7 );
    w2 = uz + w3*( ux*w6 - uy*w5 );

    // Boris rotation - u
    ux += w4*( w1*w7 - w2*w6 );
    uy += w4*( w2*w5 - w0*w7 );
    uz += w4*( w0*w6 - w1*w5 );

    // Compute physical three-velocities
    //float vx  = ux*vz;  // rel push
    //float vy  = uy*vz;
    //vz *= uz;
    ux *= c;              // non-rel push
    uy *= c;
    uz *= c;

    // Compute the trilinear coefficients
    //w0  = r8V*w;    // w0 = (1/8)(w/V)
    //dx *= w0;       // dx = (1/8)(w/V) x
    //w1  = w0+dx;    // w1 = (1/8)(w/V) + (1/8)(w/V)x = (1/8)(w/V)(1+x)
    //w0 -= dx;       // w0 = (1/8)(w/V) - (1/8)(w/V)x = (1/8)(w/V)(1-x)
    //w3  = 1.0+dy;     // w3 = 1+y
    //w2  = w0*w3;    // w2 = (1/8)(w/V)(1-x)(1+y)
    //w3 *= w1;       // w3 = (1/8)(w/V)(1+x)(1+y)
    //dy  = 1.0-dy;     // dy = 1-y
    //w0 *= dy;       // w0 = (1/8)(w/V)(1-x)(1-y)
    //w1 *= dy;       // w1 = (1/8)(w/V)(1+x)(1-y)
    //w7  = 1.0+dz;     // w7 = 1+z
    //w4  = w0*w7;    // w4 = (1/8)(w/V)(1-x)(1-y)(1+z) = (w/V) trilin_0 *Done
    //w5  = w1*w7;    // w5 = (1/8)(w/V)(1+x)(1-y)(1+z) = (w/V) trilin_1 *Done
    //w6  = w2*w7;    // w6 = (1/8)(w/V)(1-x)(1+y)(1+z) = (w/V) trilin_2 *Done
    //w7 *= w3;       // w7 = (1/8)(w/V)(1+x)(1+y)(1+z) = (w/V) trilin_3 *Done
    //dz  = 1.0-dz;     // dz = 1-z
    //w0 *= dz;       // w0 = (1/8)(w/V)(1-x)(1-y)(1-z) = (w/V) trilin_4 *Done
    //w1 *= dz;       // w1 = (1/8)(w/V)(1+x)(1-y)(1-z) = (w/V) trilin_5 *Done
    //w2 *= dz;       // w2 = (1/8)(w/V)(1-x)(1+y)(1-z) = (w/V) trilin_6 *Done
    //w3 *= dz;       // w3 = (1/8)(w/V)(1+x)(1+y)(1-z) = (w/V) trilin_7 *Done

    // Stencil weights omit species' electric charge.
    // Charge is accounted for in ACCUM_HYDRO below, in order to distinguish
    // electric-charge outputs (j, rho) from mass-charge outputs (p, Tij)
#ifdef SHAPE_NGP
    w0 = w*rV;
#else
#ifdef SHAPE_QS
    w0 =  (w*r12V) * two*( three - dx*dx - dy*dy - dz*dz );
    float wx =  (w*r12V) * ( dx + one )*( dx + one );
    float wy =  (w*r12V) * ( dy + one )*( dy + one );
    float wz =  (w*r12V) * ( dz + one )*( dz + one );
    float wmx = (w*r12V) * ( dx - one )*( dx - one );
    float wmy = (w*r12V) * ( dy - one )*( dy - one );
    float wmz = (w*r12V) * ( dz - one )*( dz - one );
#endif
#endif

    // TODO: This could easily be a loop?

    float t = 0.0; // used in macro
    auto k_hydro_access = k_hydro_sv.access();

#ifdef VARIABLE_CHARGE
    float q = qp;

    //if ( q == 0 ) printf("qp=%e",q);
    
    Kokkos::atomic_fetch_min(&k_hydro(ii, hydro_var::min_q), q);
    Kokkos::atomic_fetch_max(&k_hydro(ii, hydro_var::max_q), q);

    Kokkos::atomic_add(&particle_count(ii), 1); // number of particles in each cell
#else
    float q = qsp;
#endif
    
    // Accumulate the hydro fields - relativistic version
//    #define ACCUM_HYDRO( wn, i )                        \
//    t  = q*wn;        /* t  = (q w/V) trilin_n */		     \
//    k_hydro_access(i, hydro_var::jx)  += t*vx;                       \
//    k_hydro_access(i, hydro_var::jy)  += t*vy;                       \
//    k_hydro_access(i, hydro_var::jz)  += t*vz;                       \
//    k_hydro_access(i, hydro_var::rho) += t;                          \
//    t  = mspc*wn;       /* t = (msp c w/V) trilin_n */  \
//    dx = t*ux;          /* dx = (px w/V) trilin_n */    \
//    dy = t*uy;                                          \
//    dz = t*uz;                                          \
//    k_hydro_access(i, hydro_var::px)  += dx;                         \
//    k_hydro_access(i, hydro_var::py)  += dy;                         \
//    k_hydro_access(i, hydro_var::pz)  += dz;                         \
//    k_hydro_access(i, hydro_var::rho_m) += t; /* changed to mass density (previously ke_mc). Nb. for non-relativistic ke can be computed through trace of pressure tensor below;)*/		     \
//    k_hydro_access(i, hydro_var::txx) += dx*vx;                      \
//    k_hydro_access(i, hydro_var::tyy) += dy*vy;                      \
//    k_hydro_access(i, hydro_var::tzz) += dz*vz;                      \
//    k_hydro_access(i, hydro_var::tyz) += dy*vz;                      \
//    k_hydro_access(i, hydro_var::tzx) += dz*vx;                      \
//    k_hydro_access(i, hydro_var::txy) += dx*vy;

    // Accumulate the hydro fields - non-relativistic version
    #define ACCUM_HYDRO( wn, i )                        \
    t  = q*wn;        /* t  = (q w/V) trilin_n */       \
    k_hydro_access(i, hydro_var::jx)  += t*ux;          \
    k_hydro_access(i, hydro_var::jy)  += t*uy;          \
    k_hydro_access(i, hydro_var::jz)  += t*uz;          \
    k_hydro_access(i, hydro_var::rho) += t;             \
    t  = msp*wn;        /* t = (msp w/V) trilin_n */    \
    dx = t*ux;          /* dx = (px w/V) trilin_n */    \
    dy = t*uy;                                          \
    dz = t*uz;                                          \
    k_hydro_access(i, hydro_var::px)  += dx;            \
    k_hydro_access(i, hydro_var::py)  += dy;            \
    k_hydro_access(i, hydro_var::pz)  += dz;            \
    k_hydro_access(i, hydro_var::rho_m) += t; /* changed to mass density (previously ke_mc). Nb. for non-relativistic ke can be computed through trace of pressure tensor below;)*/		     \
    k_hydro_access(i, hydro_var::txx) += dx*ux;         \
    k_hydro_access(i, hydro_var::tyy) += dy*uy;         \
    k_hydro_access(i, hydro_var::tzz) += dz*uz;         \
    k_hydro_access(i, hydro_var::tyz) += dy*uz;         \
    k_hydro_access(i, hydro_var::tzx) += dz*ux;         \
    k_hydro_access(i, hydro_var::txy) += dx*uy;

    // TODO: this serial adding to try and save adds is a bit sad
    // TODO: This is somehow going out of bounds right now
//    const int i0 = ii;
//    ACCUM_HYDRO(w0, i0); // Cell i,j,k
//
//    const int i1 = i0 + stride_10;
//    ACCUM_HYDRO(w1, i1); // Cell i+1,j,k
//
//    const int i2 = i1 + stride_21;
//    ACCUM_HYDRO(w2, i2); // Cell i,j+1,k
//
//    const int i3 = i2 + stride_10;
//    ACCUM_HYDRO(w3, i3); // Cell i+1,j+1,k
//
//    const int i4 = i3 + stride_43;
//    ACCUM_HYDRO(w4, i4); // Cell i,j,k+1
//
//    const int i5 = i4 + stride_10;
//    ACCUM_HYDRO(w5, i5); // Cell i+1,j,k+1
//
//    const int i6 = i5 + stride_21;
//    ACCUM_HYDRO(w6, i6); // Cell i,j+1,k+1
//
//    const int i7 = i6 + stride_10;
//    ACCUM_HYDRO(w7, i7); // Cell i+1,j+1,k+1

#ifdef SHAPE_NGP
    ACCUM_HYDRO(w0, ii); // Cell i,j,k
#else
#ifdef SHAPE_QS
    ACCUM_HYDRO(w0,  ii     ); // Cell i,j,k
    ACCUM_HYDRO(wx,  ii +  1); // Cell i+1,j,k
    ACCUM_HYDRO(wy,  ii + sy); // Cell i,j+1,k
    ACCUM_HYDRO(wz,  ii + sz); // Cell i,j,k+1
    ACCUM_HYDRO(wmx, ii -  1); // Cell i-1,j,k
    ACCUM_HYDRO(wmy, ii - sy); // Cell i,j-1,k
    ACCUM_HYDRO(wmz, ii - sz); // Cell i,j,k-1
#endif
#endif

#   undef ACCUM_HYDRO
  });

#ifdef VARIABLE_CHARGE
  // Give nan values to cells without particles
  Kokkos::parallel_for("calculate_mean_q", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nv),
    KOKKOS_LAMBDA(size_t ii)
    {
      // Calculate mean charge only if there are particles in the cell
      if (particle_count(ii) > 0) {
	//	k_hydro(ii, hydro_var::avg_q) /= static_cast<float>(particle_count(ii));
	//if (k_hydro(ii, hydro_var::min_q) == 0) printf("ii=%d, minq=%e",ii,k_hydro(ii, hydro_var::min_q));
      } else {
	//	k_hydro(ii, hydro_var::avg_q) = std::numeric_limits<double>::quiet_NaN();
	k_hydro(ii, hydro_var::min_q) = std::numeric_limits<double>::quiet_NaN();
	k_hydro(ii, hydro_var::max_q) = std::numeric_limits<double>::quiet_NaN();
      }
    });
#endif
  
  Kokkos::Experimental::contribute(k_hydro, k_hydro_sv);
  Kokkos::fence(); // TODO: Check if I need this to block the contribute

  #undef f_ex
  #undef f_dexdx
  #undef f_dexdy
  #undef f_dexdz
  #undef f_d2exdx
  #undef f_d2exdy
  #undef f_d2exdz
  #undef f_ey
  #undef f_deydx
  #undef f_deydy
  #undef f_deydz
  #undef f_d2eydx
  #undef f_d2eydy
  #undef f_d2eydz
  #undef f_ez
  #undef f_dezdx
  #undef f_dezdy
  #undef f_dezdz
  #undef f_d2ezdx
  #undef f_d2ezdy
  #undef f_d2ezdz
  #undef f_cbx
  #undef f_dcbxdx
  #undef f_dcbxdy
  #undef f_dcbxdz
  #undef f_d2cbxdx
  #undef f_d2cbxdy
  #undef f_d2cbxdz
  #undef f_cby
  #undef f_dcbydx
  #undef f_dcbydy
  #undef f_dcbydz
  #undef f_d2cbydx
  #undef f_d2cbydy
  #undef f_d2cbydz
  #undef f_cbz
  #undef f_dcbzdx
  #undef f_dcbzdy
  #undef f_dcbzdz
  #undef f_d2cbzdx
  #undef f_d2cbzdy
  #undef f_d2cbzdz

  // Perform debug printing
}
