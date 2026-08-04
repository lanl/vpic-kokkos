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

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
void
accumulate_hydro_p( hydro_array_t              * RESTRICT ha,
                    const species_t            * RESTRICT sp,
                    const interpolator_array_t * RESTRICT ia ) {
  /**/  hydro_t        * RESTRICT ALIGNED(128) h;
  const particle_t     * RESTRICT ALIGNED(128) p;
  const interpolator_t * RESTRICT ALIGNED(128) f;
  float c, qsp, msp, qdt_2mc, qdt_4mc, dt_2mc, rV;
  float dt_2c;
  size_t np, stride_10, stride_21, stride_43;

  float dx, dy, dz, ux, uy, uz, w;
  float w0, w1, w2, w3, w4, w5, w6, w7, t;
  int i; 
  size_t n;

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
  dt_2c    = (sp->g->dt)/(2*c);
  dt_2mc    = (sp->g->dt)/(2*msp*c);
  qdt_2mc  = (qsp*sp->g->dt)/(2*msp*c);
  qdt_4mc  = qdt_2mc / 2;
  rV        = 1.0/(sp->g->dx*sp->g->dy*sp->g->dz);
  const float r12V = rV/12.f;

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
#ifdef VARIABLE_CHARGE
    const float qp = p[n].qp;
    qdt_2mc = qp * dt_2mc;
    const double q = qp;
#else
    const double q = qsp;
#endif

    float hax, hay, haz, cbx, cby, cbz;
    interpolate_e(f[i], dx, dy, dz, hax, hay, haz, qdt_2mc, dt_2c); // Interpolate E
    
    // Half advance E
    ux += hax;
    uy += hay;
    uz += haz;

    interpolate_b(f[i], dx, dy, dz, cbx, cby, cbz); // Interpolate B
    w5 = cbx;
    w6 = cby;
    w7 = cbz;
    
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
#ifdef SHAPE_NGP
    w0 = w*rV;
#elif defined( SHAPE_QS )
    w0 =  (w*r12V) * 2.0f*( 3.0f - dx*dx - dy*dy - dz*dz );
    float wx =  (w*r12V) * ( dx + 1.0f )*( dx + 1.0f );
    float wy =  (w*r12V) * ( dy + 1.0f )*( dy + 1.0f );
    float wz =  (w*r12V) * ( dz + 1.0f )*( dz + 1.0f );
    float wmx = (w*r12V) * ( dx - 1.0f )*( dx - 1.0f );
    float wmy = (w*r12V) * ( dy - 1.0f )*( dy - 1.0f );
    float wmz = (w*r12V) * ( dz - 1.0f )*( dz - 1.0f );
#endif

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

//  /**/            ACCUM_HYDRO(w0); // Cell i,j,k
//  i += stride_10; ACCUM_HYDRO(w1); // Cell i+1,j,k
//  i += stride_21; ACCUM_HYDRO(w2); // Cell i,j+1,k
//  i += stride_10; ACCUM_HYDRO(w3); // Cell i+1,j+1,k
//  i += stride_43; ACCUM_HYDRO(w4); // Cell i,j,k+1
//  i += stride_10; ACCUM_HYDRO(w5); // Cell i+1,j,k+1
//  i += stride_21; ACCUM_HYDRO(w6); // Cell i,j+1,k+1
//  i += stride_10; ACCUM_HYDRO(w7); // Cell i+1,j+1,k+1

    // Accumulate the hydro fields - non-relativistic version
#   define ACCUM_HYDRO( wn, ii)                          \
    t  = q*wn;        /* t  = (q w/V) trilin_n */        \
    h[ii].jx  += t*ux;                                   \
    h[ii].jy  += t*uy;                                   \
    h[ii].jz  += t*uz;                                   \
    h[ii].rho += t;                                      \
    t  = msp*wn;        /* t = (msp w/V) trilin_n */     \
    dx = t*ux;          /* dx = (px w/V) trilin_n */     \
    dy = t*uy;                                           \
    dz = t*uz;                                           \
    h[ii].px  += dx;                                     \
    h[ii].py  += dy;                                     \
    h[ii].pz  += dz;                                     \
    h[ii].rho_m  += t; /* Prev. was *ke_mc; */           \
    h[ii].txx += dx*ux;                                  \
    h[ii].tyy += dy*uy;                                  \
    h[ii].tzz += dz*uz;                                  \
    h[ii].tyz += dy*uz;                                  \
    h[ii].tzx += dz*ux;                                  \
    h[ii].txy += dx*uy

#ifdef SHAPE_NGP
    ACCUM_HYDRO(w0, i); // Cell i,j,k
#elif defined( SHAPE_QS )
    ACCUM_HYDRO(w0,  i     ); // Cell i,j,k
    ACCUM_HYDRO(wx,  i +  1); // Cell i+1,j,k
    ACCUM_HYDRO(wy,  i + sy); // Cell i,j+1,k
    ACCUM_HYDRO(wz,  i + sz); // Cell i,j,k+1
    ACCUM_HYDRO(wmx, i -  1); // Cell i-1,j,k
    ACCUM_HYDRO(wmy, i - sy); // Cell i,j-1,k
    ACCUM_HYDRO(wmz, i - sz); // Cell i,j,k-1
#endif
#   undef ACCUM_HYDRO
  }
}
#endif

void
accumulate_hydro_p_kokkos_nomove_ngp(
                                      k_particles_t& k_particles,
                                      k_particles_i_t& k_particles_i,
                                      k_hydro_t k_hydro,
                                      k_interpolator_t& k_interp,
                                      const species_t            * RESTRICT sp
)
{
  k_hydro_sv_t k_hydro_sv = Kokkos::Experimental::create_scatter_view(k_hydro);

  float c, mspc, qdt_2mc, qdt_4mc2, r8V;

  if( !sp ) {
    ERROR(( "Bad args" ));
  }

  c        = sp->g->cvac;
  mspc     = sp->m*c;
#ifdef VARIABLE_CHARGE
  float dt_2mc  = (sp->g->dt)/(2*mspc); // Multiply by particle q later
  float dt_4mc2 = dt_2mc / (2*c);
#else
  const float qsp      = sp->q;
  qdt_2mc  = (qsp*sp->g->dt)/(2*mspc);
  qdt_4mc2 = qdt_2mc / (2*c);
#endif
  const float rV   = 1.0/(sp->g->dx*sp->g->dy*sp->g->dz);
  const float r12V = rV/12.;

  const size_t np = sp->np;
  const int sy = sp->g->sy;
  const int sz = sp->g->sz;

  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace,size_t > (0LLU, np),
    KOKKOS_LAMBDA (size_t p_index)
    {

    // Load the particle
    double dx = static_cast<double>(k_particles(p_index, particle_var::dx));
    double dy = static_cast<double>(k_particles(p_index, particle_var::dy));
    double dz = static_cast<double>(k_particles(p_index, particle_var::dz));
    double ux = static_cast<double>(k_particles(p_index, particle_var::ux));
    double uy = static_cast<double>(k_particles(p_index, particle_var::uy));
    double uz = static_cast<double>(k_particles(p_index, particle_var::uz));
    double w  = static_cast<double>(k_particles(p_index, particle_var::w ));
    int ii = k_particles_i(p_index);
    double qp  = 1.0;
#ifdef VARIABLE_CHARGE
    qp = k_particles(p_index, particle_var::qp);
    const double q = static_cast<double>(qp);
#else
    const double q = static_cast<double>(qsp);
#endif

    double ke_mc = ux*ux + uy*uy + uz*uz; // ke_mc = |u|^2 (invariant)
    double vz = 1.0;//sqrt(1.0+ke_mc);            // vz = gamma    (invariant)    
//    ke_mc *= c/(vz+1.0);             // ke_mc = c|u|^2/(gamma+1) = c*(gamma-1)
    
    // Compute physical velocities
    double vx  = ux*vz;
    double vy  = uy*vz;
    vz *= uz;

    double t = 0.0; // used in macro
    auto hydro_sa = k_hydro_sv.access();

//#ifdef SHAPE_NGP
    float w0 = w*rV;
//#elif defined( SHAPE_QS )
//    float w0 =  (w*r12V) * 2.0f*( 3.0f - dx*dx - dy*dy - dz*dz );
//    float wx =  (w*r12V) * ( dx + 1.0f )*( dx + 1.0f );
//    float wy =  (w*r12V) * ( dy + 1.0f )*( dy + 1.0f );
//    float wz =  (w*r12V) * ( dz + 1.0f )*( dz + 1.0f );
//    float wmx = (w*r12V) * ( dx - 1.0f )*( dx - 1.0f );
//    float wmy = (w*r12V) * ( dy - 1.0f )*( dy - 1.0f );
//    float wmz = (w*r12V) * ( dz - 1.0f )*( dz - 1.0f );
//#endif

    // Accumulate the hydro fields
    #define ACCUM_HYDRO( wn, i )                                 \
    t  = q*wn;        /* t  = (qsp w/V) trilin_n */              \
    hydro_sa(i, hydro_var::jx)  += t*vx;                         \
    hydro_sa(i, hydro_var::jy)  += t*vy;                         \
    hydro_sa(i, hydro_var::jz)  += t*vz;                         \
    hydro_sa(i, hydro_var::rho) += t;                            \
    t  = mspc*wn;       /* t = (msp c w/V) trilin_n */           \
    dx = t*ux;          /* dx = (px w/V) trilin_n */             \
    dy = t*uy;                                                   \
    dz = t*uz;                                                   \
    hydro_sa(i, hydro_var::px)    += dx;                         \
    hydro_sa(i, hydro_var::py)    += dy;                         \
    hydro_sa(i, hydro_var::pz)    += dz;                         \
    hydro_sa(i, hydro_var::rho_m) += t; /* Prev. was *ke_mc; */  \
    hydro_sa(i, hydro_var::txx)   += dx*vx;                      \
    hydro_sa(i, hydro_var::tyy)   += dy*vy;                      \
    hydro_sa(i, hydro_var::tzz)   += dz*vz;                      \
    hydro_sa(i, hydro_var::tyz)   += dy*vz;                      \
    hydro_sa(i, hydro_var::tzx)   += dz*vx;                      \
    hydro_sa(i, hydro_var::txy)   += dx*vy;

    // TODO: this serial adding to try and save adds is a bit sad
    // TODO: This is somehow going out of bounds right now
//    const int i0 = ii;
//    if(qp!=0) {
//      ACCUM_HYDRO(w, i0); // Cell i,j,k
//    }
    if(q!=0) {
//#ifdef SHAPE_NGP
      ACCUM_HYDRO(w0, ii); // Cell i,j,k
//#elif defined( SHAPE_QS )
//      ACCUM_HYDRO(w0,  ii     ); // Cell i,j,k
//      ACCUM_HYDRO(wx,  ii +  1); // Cell i+1,j,k
//      ACCUM_HYDRO(wy,  ii + sy); // Cell i,j+1,k
//      ACCUM_HYDRO(wz,  ii + sz); // Cell i,j,k+1
//      ACCUM_HYDRO(wmx, ii -  1); // Cell i-1,j,k
//      ACCUM_HYDRO(wmy, ii - sy); // Cell i,j-1,k
//      ACCUM_HYDRO(wmz, ii - sz); // Cell i,j,k-1
//#endif
    }
#   undef ACCUM_HYDRO
  });

  Kokkos::Experimental::contribute(k_hydro, k_hydro_sv);
  Kokkos::fence(); // TODO: Check if I need this to block the contribute
}


void
accumulate_hydro_p_kokkos(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_hydro_t k_hydro,
        k_interpolator_t& k_interp,
        const species_t            * RESTRICT sp
)
{
  k_hydro_sv_t k_hydro_sv = Kokkos::Experimental::create_scatter_view(k_hydro);
  using hydro_scalar_t = k_hydro_t::non_const_value_type;

#ifdef VARIABLE_CHARGE
  float c, msp, dt_2mc, dt_4mc, rV, r12V;
#else
  float c, msp, qdt_2mc, qdt_4mc, rV, r12V;
#endif

  constexpr float one=1.0f, two=2.0f, three=3.0f;

  size_t nv = static_cast<size_t>(sp->g->nv);

  if( !sp ) {
    ERROR(( "Bad args" ));
  }

  c        = sp->g->cvac;
  msp      = sp->m;
#ifdef VARIABLE_CHARGE
  dt_2mc   = (sp->g->dt)/(2*msp*c);
  dt_4mc   = dt_2mc / 2;
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace> particle_count("particle_count", nv);
  Kokkos::deep_copy(particle_count, 0);

  Kokkos::parallel_for("calculate_mean_q", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, size_t>(0LLU, nv),
    KOKKOS_LAMBDA(size_t ii)
    {
        k_hydro(ii, hydro_var::qmin) = std::numeric_limits<hydro_scalar_t>::max();
        k_hydro(ii, hydro_var::qmax) = std::numeric_limits<hydro_scalar_t>::min();
    });
#else
  const float qsp = sp->q;
  qdt_2mc  = (qsp*sp->g->dt)/(2*msp*c);
  qdt_4mc  = qdt_2mc / 2;
#endif
  const float dt_2c = (sp->g->dt)/(2*c);
  rV        = 1.0/(sp->g->dx*sp->g->dy*sp->g->dz);
  r12V      = rV/12.;
  const float r8V = sp->g->r8V;

  const size_t np        = sp->np;
  const int nx = sp->g->nx;
  const int ny = sp->g->ny;
  const int nz = sp->g->nz;
  const int sy = sp->g->sy;
  const int sz = sp->g->sz;
  const float gdx = sp->g->dx;
  const float gdy = sp->g->dy;
  const float gdz = sp->g->dz;
  const grid_t* g = sp->g;
  const grid::grid_geom_t geom = g->geom();

  Kokkos::parallel_for("hydro_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace,size_t > (0LLU, np),
    KOKKOS_LAMBDA (const size_t p_index)
    {

    float dx = k_particles(p_index, particle_var::dx);
    float dy = k_particles(p_index, particle_var::dy);
    float dz = k_particles(p_index, particle_var::dz);
    float ux = k_particles(p_index, particle_var::ux);
    float uy = k_particles(p_index, particle_var::uy);
    float uz = k_particles(p_index, particle_var::uz);
    float w  = k_particles(p_index, particle_var::w);
#ifdef VARIABLE_CHARGE
    const float qp = k_particles(p_index, particle_var::qp);
    const float qdt_2mc = qp*dt_2mc;
    const float qdt_4mc = qp*dt_4mc;
#endif
    const int ii = k_particles_i(p_index);

    const interpolator_t intp = read_interpolator(k_interp, ii);
    float hax, hay, haz, cbx, cby, cbz;
    
    interpolate_e(intp, dx, dy, dz, hax, hay, haz, qdt_2mc, dt_2c);
    
    ux += hax;
    uy += hay;
    uz += haz;
    
    interpolate_b(intp, dx, dy, dz, cbx, cby, cbz);
    float w5 = cbx;
    float w6 = cby;
    float w7 = cbz;

    float w0 = qdt_4mc;
    float w1 = w5*w5 + w6*w6 + w7*w7;
    float w2 = w0*w0*w1;
    float w3 = w0*(1.+(1./3.)*w2*(1.0+0.4*w2));
    float w4 = w3/(1.+ w1*(w3*w3)); w4 += w4;

    w0 = ux + w3*( uy*w7 - uz*w6 );
    w1 = uy + w3*( uz*w5 - ux*w7 );
    w2 = uz + w3*( ux*w6 - uy*w5 );

    ux += w4*( w1*w7 - w2*w6 );
    uy += w4*( w2*w5 - w0*w7 );
    uz += w4*( w0*w6 - w1*w5 );

#ifdef SHAPE_NGP
    // For curvilinear coordinates, compute reciprocal basis
    float grad_xi_x, grad_xi_y, grad_xi_z;
    float grad_eta_x, grad_eta_y, grad_eta_z;
    float grad_mu_x, grad_mu_y, grad_mu_z;
    float jac;
    
    compute_reciprocal_basis(
        geom,
        dx, dy, dz, ii, nx, ny, nz,
        gdx, gdy, gdz,
        grad_xi_x, grad_xi_y, grad_xi_z,
        grad_eta_x, grad_eta_y, grad_eta_z,
        grad_mu_x, grad_mu_y, grad_mu_z,
        jac);
    
    // Compute scale factors
    float h_xi = 1.0f / sqrtf(grad_xi_x*grad_xi_x + grad_xi_y*grad_xi_y + grad_xi_z*grad_xi_z);
    float h_eta = 1.0f / sqrtf(grad_eta_x*grad_eta_x + grad_eta_y*grad_eta_y + grad_eta_z*grad_eta_z);
    float h_mu = 1.0f / sqrtf(grad_mu_x*grad_mu_x + grad_mu_y*grad_mu_y + grad_mu_z*grad_mu_z);
    
    // Transform Cartesian velocities to physical velocities
    float u_phys_xi = (ux * grad_xi_x + uy * grad_xi_y + uz * grad_xi_z) * h_xi;
    float u_phys_eta = (ux * grad_eta_x + uy * grad_eta_y + uz * grad_eta_z) * h_eta;
    float u_phys_mu = (ux * grad_mu_x + uy * grad_mu_y + uz * grad_mu_z) * h_mu;
    
    // Use physical velocities and proper volume weighting
    ux = u_phys_xi * c;
    uy = u_phys_eta * c;
    uz = u_phys_mu * c;
    
    // NGP density = w / (physical cell volume). compute_reciprocal_basis returns
    // jac = physical_vol / logical_vol, and the logical cell [-1,1]^3 has volume 8,
    // so physical_vol = 8*jac. Hence rho = w/(8*jac). On CARTESIAN jac=gd^3/8 so
    // this reduces to w/gd^3 = rV*w (the original NGP weight); on CYLINDRICAL
    // jac=r*gd^3/8 so rho = w/(r*gd^3), i.e. weight per PHYSICAL volume r*dr*dth*dz.
    // (The previous w*rV*jac multiplied by r instead of dividing, inflating ni ~r.)
    w0 = w / (8.0f * jac);
#else
#ifdef SHAPE_QS
    // QS shape - not modified per instructions
    ux *= c;
    uy *= c;
    uz *= c;

    w *= one_twelfth;
    float v0 = dx, v1 = dy, v2 = dz;
    w0 =  w*two*( three - v0*v0 - v1*v1 - v2*v2 );
    float wx =  w*( v0 + one )*( v0 + one );
    float wy =  w*( v1 + one )*( v1 + one );
    float wz =  w*( v2 + one )*( v2 + one );
    float wmx = w*( v0 - one )*( v0 - one );
    float wmy = w*( v1 - one )*( v1 - one );
    float wmz = w*( v2 - one )*( v2 - one );
#endif
#endif

    float t = 0.0;
    auto k_hydro_access = k_hydro_sv.access();

#ifdef VARIABLE_CHARGE
    const double q = qp;

    Kokkos::atomic_min(&k_hydro(ii, hydro_var::qmin), q);
    Kokkos::atomic_max(&k_hydro(ii, hydro_var::qmax), q);

    float w_ngp = 8.0*r8V*w;
    if (q==0) {
      k_hydro_access(ii, hydro_var::n_q0) += w_ngp;
    } else if (q==1) {
      k_hydro_access(ii, hydro_var::n_q1) += w_ngp;
    } else if (q==2) {
      k_hydro_access(ii, hydro_var::n_q2) += w_ngp;
    } else if (q==3) {
      k_hydro_access(ii, hydro_var::n_q3) += w_ngp;
    } else if (q==4) {
      k_hydro_access(ii, hydro_var::n_q4) += w_ngp;
    } else if (q==5) {
      k_hydro_access(ii, hydro_var::n_q5) += w_ngp;
    }
       
    Kokkos::atomic_inc(&particle_count(ii));
#else
    float q = qsp;
#endif
    
    #define ACCUM_HYDRO( wn, i )                        \
    t  = q*wn;                                          \
    k_hydro_access(i, hydro_var::jx)  += t*ux;          \
    k_hydro_access(i, hydro_var::jy)  += t*uy;          \
    k_hydro_access(i, hydro_var::jz)  += t*uz;          \
    k_hydro_access(i, hydro_var::rho) += t;             \
    t  = msp*wn;                                        \
    dx = t*ux;                                          \
    dy = t*uy;                                          \
    dz = t*uz;                                          \
    k_hydro_access(i, hydro_var::px)  += dx;            \
    k_hydro_access(i, hydro_var::py)  += dy;            \
    k_hydro_access(i, hydro_var::pz)  += dz;            \
    k_hydro_access(i, hydro_var::rho_m) += t;           \
    k_hydro_access(i, hydro_var::txx) += dx*ux;         \
    k_hydro_access(i, hydro_var::tyy) += dy*uy;         \
    k_hydro_access(i, hydro_var::tzz) += dz*uz;         \
    k_hydro_access(i, hydro_var::tyz) += dy*uz;         \
    k_hydro_access(i, hydro_var::tzx) += dz*ux;         \
    k_hydro_access(i, hydro_var::txy) += dx*uy;

#ifdef SHAPE_NGP
    ACCUM_HYDRO(w0, ii);
#elif defined( SHAPE_QS )
    ACCUM_HYDRO(w0,  ii     );
    ACCUM_HYDRO(wx,  ii +  1);
    ACCUM_HYDRO(wy,  ii + sy);
    ACCUM_HYDRO(wz,  ii + sz);
    ACCUM_HYDRO(wmx, ii -  1);
    ACCUM_HYDRO(wmy, ii - sy);
    ACCUM_HYDRO(wmz, ii - sz);
#endif

#   undef ACCUM_HYDRO
  });

#ifdef VARIABLE_CHARGE
  Kokkos::parallel_for("calculate_mean_q", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, size_t>(0LLU, nv),
    KOKKOS_LAMBDA(size_t ii)
    {
      if (particle_count(ii) > 0) {
      } else {
        //k_hydro(ii, hydro_var::avg_q) = std::numeric_limits<double>::quiet_NaN();
        k_hydro(ii, hydro_var::qmin) = std::numeric_limits<hydro_scalar_t>::quiet_NaN();
        k_hydro(ii, hydro_var::qmax) = std::numeric_limits<hydro_scalar_t>::quiet_NaN();
      }
    });
#endif
  
  Kokkos::Experimental::contribute(k_hydro, k_hydro_sv);
  Kokkos::fence();
}
