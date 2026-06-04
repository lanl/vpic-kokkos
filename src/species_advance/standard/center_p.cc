#define IN_spa
#define HAS_V4_PIPELINE
#include "spa_private.h"

struct center_p_kernel {
  species_t* sp;
  k_particles_t::HostMirror p;
  k_particles_i_t::HostMirror p_i;
  k_interpolator_t::HostMirror f;
  float _qdt_2mc, _qdt_4mc;

  center_p_kernel(species_t* _sp,
                  k_particles_t::HostMirror& _p, 
                  const k_particles_i_t::HostMirror& _p_i, 
                  const k_interpolator_t::HostMirror& _interp,
                  const float _qdt_2mc_) : sp(_sp), p(_p), p_i(_p_i), f(_interp) {
#ifndef VARIABLE_CHARGE
    _qdt_2mc        =     _qdt_2mc_;
    _qdt_4mc        = 0.5*_qdt_2mc_; // For half Boris rotate
#endif
  }
  
  KOKKOS_INLINE_FUNCTION 
  void 
  operator() (const int n) const {
    //float dx, dy, dz, ux, uy, uz;
    float ux, uy, uz;
    float hax, hay, haz, cbx, cby, cbz;
    float v0, v1, v2, v3, v4;
    int ii;
    constexpr float one            = 1.;
    constexpr float one_third      = 1./3.;
    constexpr float two_fifteenths = 2./15.;

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
//    dx   = sp->p[n].dx; // Load position
//    dy   = sp->p[n].dy;
//    dz   = sp->p[n].dz;
#ifdef VARIABLE_CHARGE
    const float qp   = sp->p[n].qp;
    const float qdt_2mc = qp*_qdt_2mc;
    const float qdt_4mc = 0.5*qdt_2mc;
#else
    const float qdt_2mc = _qdt_2mc;
    const float qdt_4mc = _qdt_4mc;
#endif
    ii   = sp->p[n].i;
#else
//    dx   = p(n, particle_var::dx); // Load position
//    dy   = p(n, particle_var::dy);
//    dz   = p(n, particle_var::dz);
#ifdef VARIABLE_CHARGE
    const float qp   = p(n, particle_var::qp);
    const float qdt_2mc = qp*_qdt_2mc;
    const float qdt_4mc = 0.5*qdt_2mc;
#else
    const float qdt_2mc = _qdt_2mc;
    const float qdt_4mc = _qdt_4mc;
#endif
    ii   = p_i(n);
#endif
    hax  = qdt_2mc * f(ii, interpolator_var::ex);  // Interpolate E  
    hay  = qdt_2mc * f(ii, interpolator_var::ey);   
    haz  = qdt_2mc * f(ii, interpolator_var::ez);   
    cbx  = f(ii, interpolator_var::cbx); // Interpolate B
    cby  = f(ii, interpolator_var::cby); 
    cbz  = f(ii, interpolator_var::cbz); 
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
    ux   = sp->p[n].ux; // Load momentum
    uy   = sp->p[n].uy;
    uz   = sp->p[n].uz;
#else
    ux   = p(n, particle_var::ux); // Load momentum
    uy   = p(n, particle_var::uy);
    uz   = p(n, particle_var::uz);
#endif
    ux  += hax; // Half advance E
    uy  += hay;
    uz  += haz;
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
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
    sp->p[n].ux = ux;             // Store momentum
    sp->p[n].uy = uy;
    sp->p[n].uz = uz;
#else
    p(n, particle_var::ux) = ux;             // Store momentum
    p(n, particle_var::uy) = uy;
    p(n, particle_var::uz) = uz;
#endif
  }
};

void
center_p_pipeline( center_p_pipeline_args_t * args,
                   int pipeline_rank,
                   int n_pipeline ) {
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_t           * ALIGNED(32)  p;
  const interpolator_t * ALIGNED(16)  f;

#ifndef VARIABLE_CHARGE
  const float qdt_2mc        =     args->qdt_2mc;
  const float qdt_4mc        = 0.5*args->qdt_2mc; // For half Boris rotate
#else
  float qp;
  float qdt_2mc;
  float qdt_4mc;
#endif
#ifdef EXTERNAL_FORCE
  const float dt_2c = args->dt_2c;
#endif
  const float one            = 1.;
  const float one_third      = 1./3.;
  const float two_fifteenths = 2./15.;

  float dx, dy, dz, ux, uy, uz;
  float hax, hay, haz, cbx, cby, cbz;
  float v0, v1, v2, v3, v4;

  int first, ii, n;

  // Determine which particle quads this pipeline processes

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, first, n );
  p = args->p0 + first;

  // Process particles for this pipeline

  for(;n;n--,p++) {
    dx   = p->dx;                            // Load position
    dy   = p->dy;
    dz   = p->dz;
#ifdef VARIABLE_CHARGE
    qp   = p->qp;
    qdt_2mc = qp*args->qdt_2mc;
    qdt_4mc = 0.5*qdt_2mc;
#endif
    ii   = p->i;
    f    = f0 + ii;                             // Load interpolator
#ifdef SHAPE_NGP
  #ifdef EXTERNAL_FORCE
    hax  = qdt_2mc*( f->ex + f->Ex0 ) + dt_2c * f->Gx0;        // Interpolate E, E0, G0
    hay  = qdt_2mc*( f->ey + f->Ey0 ) + dt_2c * f->Gy0;
    haz  = qdt_2mc*( f->ez + f->Ez0 ) + dt_2c * f->Gz0;
  #else
    hax  = qdt_2mc*(    ( f->ex     ) );        // Interpolate E
    hay  = qdt_2mc*(    ( f->ey     ) );
    haz  = qdt_2mc*(    ( f->ez     ) );
  #endif
    cbx  = f->cbx;// + dx*f->dcbxdx;            // Interpolate B
    cby  = f->cby;// + dy*f->dcbydy;
    cbz  = f->cbz;// + dz*f->dcbzdz;
#elif defined( SHAPE_QS )
  #ifdef EXTERNAL_FORCE
    hax  = qdt_2mc*( f->ex + dx*( f->dexdx + dx*f->d2exdx )   // Interpolate E, E0
                           + dy*( f->dexdy + dy*f->d2exdy )
                           + dz*( f->dexdz + dz*f->d2exdz )
                     + f->Ex0 + dx*( f->dEx0dx + dx*f->d2Ex0dx )
                              + dy*( f->dEx0dy + dy*f->d2Ex0dy )
                              + dz*( f->dEx0dz + dz*f->d2Ex0dz ) );
    hay  = qdt_2mc*( f->ey + dx*( f->deydx + dx*f->d2eydx )
                           + dy*( f->deydy + dy*f->d2eydy )
                           + dz*( f->deydz + dz*f->d2eydz )
                     + f->Ey0 + dx*( f->dEy0dx + dx*f->d2Ey0dx )
                              + dy*( f->dEy0dy + dy*f->d2Ey0dy )
                              + dz*( f->dEy0dz + dz*f->d2Ey0dz ) );
    haz  = qdt_2mc*( f->ez + dx*( f->dezdx + dx*f->d2ezdx )
                           + dy*( f->dezdy + dy*f->d2ezdy )
                           + dz*( f->dezdz + dz*f->d2ezdz )
                     + f->Ez0 + dx*( f->dEz0dx + dx*f->d2Ez0dx )
                              + dy*( f->dEz0dy + dy*f->d2Ez0dy )
                              + dz*( f->dEz0dz + dz*f->d2Ez0dz ) );
    hax += dt_2c *( f->Gx0 + dx*( f->dGx0dx + dx*f->d2Gx0dx )   // Interpolate G0
                           + dy*( f->dGx0dy + dy*f->d2Gx0dy )
                           + dz*( f->dGx0dz + dz*f->d2Gx0dz ) );
    hay += dt_2c *( f->Gy0 + dx*( f->dGy0dx + dx*f->d2Gy0dx )
                           + dy*( f->dGy0dy + dy*f->d2Gy0dy )
                           + dz*( f->dGy0dz + dz*f->d2Gy0dz ) );
    haz += dt_2c *( f->Gz0 + dx*( f->dGz0dx + dx*f->d2Gz0dx )
                           + dy*( f->dGz0dy + dy*f->d2Gz0dy )
                           + dz*( f->dGz0dz + dz*f->d2Gz0dz ) );
  #else
    hax  = qdt_2mc*( f->ex + dx*( f->dexdx + dx*f->d2exdx )   // Interpolate E
                           + dy*( f->dexdy + dy*f->d2exdy )
                           + dz*( f->dexdz + dz*f->d2exdz ) );
    hay  = qdt_2mc*( f->ey + dx*( f->deydx + dx*f->d2eydx )
                           + dy*( f->deydy + dy*f->d2eydy )
                           + dz*( f->deydz + dz*f->d2eydz ) );
    haz  = qdt_2mc*( f->ez + dx*( f->dezdx + dx*f->d2ezdx )
                           + dy*( f->dezdy + dy*f->d2ezdy )
                           + dz*( f->dezdz + dz*f->d2ezdz ) );
  #endif
    cbx  = f->cbx + dx*( f->dcbxdx + dx*f->d2cbxdx )          // Interpolate B
                  + dy*( f->dcbxdy + dy*f->d2cbxdy )
                  + dz*( f->dcbxdz + dz*f->d2cbxdz );
    cby  = f->cby + dx*( f->dcbydx + dx*f->d2cbydx )
                  + dy*( f->dcbydy + dy*f->d2cbydy )
                  + dz*( f->dcbydz + dz*f->d2cbydz );
    cbz  = f->cbz + dx*( f->dcbzdx + dx*f->d2cbzdx )
                  + dy*( f->dcbzdy + dy*f->d2cbzdy )
                  + dz*( f->dcbzdz + dz*f->d2cbzdz );
#endif
    ux   = p->ux;                            // Load momentum
    uy   = p->uy;
    uz   = p->uz;
    ux  += hax;                              // Half advance E
    uy  += hay;
    uz  += haz;
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
    p->ux = ux;                              // Store momentum
    p->uy = uy;
    p->uz = uz;
  }
}

void
center_p( /**/  species_t            * RESTRICT sp,
          const interpolator_array_t * RESTRICT ia ) {

  if( !sp || !ia || sp->g!=ia->g ) ERROR(( "Bad args" ));

  float qdt_2mc;
#ifdef VARIABLE_CHARGE
  qdt_2mc = (sp->g->dt)/(2*sp->m*sp->g->cvac); // Multiply by qp in pipelines
#else
  qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
#endif

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  // Use legacy data structures for center_p
  DECLARE_ALIGNED_ARRAY( center_p_pipeline_args_t, 128, args, 1 );

  // Have the pipelines do the bulk of particles in quads and have the
  // host do the final incomplete quad.

  args->p0      = sp->p;
  args->f0      = ia->i;
  args->qdt_2mc = qdt_2mc;
  args->np      = sp->np;
#ifdef EXTERNAL_FORCE
  args->dt_2c = (sp->g->dt)/(2*sp->g->cvac);
#endif

  EXEC_PIPELINES( center_p, args, 0 );
  WAIT_PIPELINES();

#else
  // Use host data structures for center_p_kernel
  center_p_kernel center_particles(sp, sp->k_p_h, sp->k_p_i_h, ia->k_i_h, qdt_2mc);
  Kokkos::parallel_for("center_p", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), center_particles);
#endif
}
