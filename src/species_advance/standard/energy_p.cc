#define IN_spa
#define HAS_V4_PIPELINE
#include "spa_private.h"

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES

// This function calculates kinetic energy, normalized by c^2.
void
energy_p_pipeline( energy_p_pipeline_args_t * RESTRICT args,
                   int pipeline_rank,
                   int n_pipeline ) {
  const interpolator_t * RESTRICT ALIGNED(128) f = args->f;
  const particle_t     * RESTRICT ALIGNED(32)  p = args->p;
  const float qdt_2mc = args->qdt_2mc;
  const float msp     = args->msp;
  const float dt_2c = args->dt_2c;
  const float one     = 1;

  float dx, dy, dz;
  float v0, v1, v2;

  double en = 0;

  int i, n, n0, n1;

  // Determine which particles this pipeline processes

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, n0, n1 ); n1 += n0;

  // Process particles quads for this pipeline

  for( n=n0; n<n1; n++ ) {
    dx  = p[n].dx;
    dy  = p[n].dy;
    dz  = p[n].dz;
    i   = p[n].i;

    float hax, hay, haz, cbx, cby, cbz;
    const interpolator_t* intp = &f[i];
    
    interpolate_e(*intp, dx, dy, dz, hax, hay, haz, qdt_2mc, dt_2c); // Interpolate E

    float v0 = p[n].ux + hax;
    float v1 = p[n].uy + hay;
    float v2 = p[n].uz + haz;

    v0  = v0*v0 + v1*v1 + v2*v2;
    v0  = (msp * p[n].w) * (v0 / (one + sqrtf(one + v0)));
    en += (double)v0;
  }

  args->en[pipeline_rank] = en;
}

#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

using namespace v4;

void
energy_p_pipeline_v4( energy_p_pipeline_args_t * args,
                      int pipeline_rank,
                      int n_pipeline ) {
  const interpolator_t * RESTRICT ALIGNED(128) f = args->f;
  const particle_t     * RESTRICT ALIGNED(128) p = args->p;

  const float          * RESTRICT ALIGNED(16)  vp0;
  const float          * RESTRICT ALIGNED(16)  vp1;
  const float          * RESTRICT ALIGNED(16)  vp2;
  const float          * RESTRICT ALIGNED(16)  vp3;

  const v4float qdt_2mc(args->qdt_2mc);
  const v4float msp(args->msp);
  const v4float one(1.);

  v4float dx, dy, dz;
  v4float ex, ey, ez;
  v4float v0, v1, v2, w;
  v4int i;

  double en0 = 0, en1 = 0, en2 = 0, en3 = 0;

  int n0, nq;

  // Determine which particle quads this pipeline processes

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, n0, nq );
  p += n0;
  nq >>= 2;

  // Process the particle quads for this pipeline

  for( ; nq; nq--, p+=4 ) {
    load_4x4_tr(&p[0].dx,&p[1].dx,&p[2].dx,&p[3].dx,dx,dy,dz,i);

    // Interpolate fields

    vp0 = (float *)(f + i(0));
    vp1 = (float *)(f + i(1));
    vp2 = (float *)(f + i(2));
    vp3 = (float *)(f + i(3));
    load_4x4_tr(vp0,  vp1,  vp2,  vp3,  ex,v0,v1,v2); ex = fma( fma( dy, v2, v1 ), dz, fma( dy, v0, ex ) );
    load_4x4_tr(vp0+4,vp1+4,vp2+4,vp3+4,ey,v0,v1,v2); ey = fma( fma( dz, v2, v1 ), dx, fma( dz, v0, ey ) );
    load_4x4_tr(vp0+8,vp1+8,vp2+8,vp3+8,ez,v0,v1,v2); ez = fma( fma( dx, v2, v1 ), dy, fma( dx, v0, ez ) );

    // Update momentum to half step
    // (note Boris rotation does not change energy so it is unnecessary)

    load_4x4_tr(&p[0].ux,&p[1].ux,&p[2].ux,&p[3].ux,v0,v1,v2,w);
    v0  = fma( ex, qdt_2mc, v0 );
    v1  = fma( ey, qdt_2mc, v1 );
    v2  = fma( ez, qdt_2mc, v2 );

    // Accumulate energy

    v0 = fma( v0,v0, fma( v1,v1, v2*v2 ) );
    v0 = (msp * w) * (v0 / (one + sqrt(one + v0)));
    en0 += (double)v0(0);
    en1 += (double)v0(1);
    en2 += (double)v0(2);
    en3 += (double)v0(3);
  }

  args->en[pipeline_rank] = en0 + en1 + en2 + en3;
}

#endif

double
energy_p( const species_t            * RESTRICT sp,
          const interpolator_array_t * RESTRICT ia ) {
  DECLARE_ALIGNED_ARRAY( energy_p_pipeline_args_t, 128, args, 1 );
  DECLARE_ALIGNED_ARRAY( double, 128, en, MAX_PIPELINE+1 );
  double local, global;
  int rank;

  if( !sp || !ia || sp->g!=ia->g ) ERROR(( "Bad args" ));

  // Have the pipelines do the bulk of particles in quads and have the
  // host do the final incomplete quad.

  args->p       = sp->p;
  args->f       = ia->i;
  args->en      = en;
  args->qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
  args->msp     = sp->m;
  args->np      = sp->np;
  args->dt_2c   = (sp->g->dt)/(2*sp->g->cvac);

  EXEC_PIPELINES( energy_p, args, 0 );
  WAIT_PIPELINES();

  local = 0; for( rank=0; rank<=N_PIPELINE; rank++ ) local += en[rank];
  mp_allsum_d( &local, &global, 1 );
  return global*((double)sp->g->cvac*(double)sp->g->cvac);
}

#endif // VPIC_ENABLE_LEGACY_DATA_STRUCTURES

double
energy_p_kernel(const k_interpolator_t& k_interp, 
                const k_particles_t& k_particles, 
                const k_particles_i_t& k_particles_i, 
                const float q,
                const float dt_2mc, 
                const float dt_2c, 
                const float msp, 
                const size_t np) {
    double en = 0;
    float _qdt_2mc = q*dt_2mc;

    Kokkos::parallel_reduce(np, KOKKOS_LAMBDA(const int n, double& update) {
        float ux = k_particles(n, particle_var::ux);
        float uy = k_particles(n, particle_var::uy);
        float uz = k_particles(n, particle_var::uz);
        float dx = k_particles(n, particle_var::dx);
        float dy = k_particles(n, particle_var::dy);
        float dz = k_particles(n, particle_var::dz);
        int   ii = k_particles_i(n);
#ifdef VARIABLE_CHARGE
        const float qp = k_particles(n, particle_var::qp);
        const float qdt_2mc = qp * dt_2mc;
#else
        const float qdt_2mc = _qdt_2mc;
#endif

        float hax, hay, haz;
        const interpolator_t intp = read_interpolator(k_interp, ii); // Load interpolators
    
        interpolate_e(intp, dx, dy, dz, hax, hay, haz, qdt_2mc, dt_2c); // Interpolate E

        float v0 = ux + hax;
        float v1 = uy + hay;
        float v2 = uz + haz;
    
        v0 = v0*v0 + v1*v1 + v2*v2;
        //v0 = (msp * k_particles(n, particle_var::w)) * (v0 / (1 + sqrtf(1 + v0)));  // Relativistic kinetic energy
        v0 *= 0.5 * (msp * k_particles(n, particle_var::w));  // Non-relativistic kinetic energy
        update += static_cast<double>(v0);
    }, en);
    return en;
} // energy_p_kernel(...)

/*
double
energy_p_trilinear(const k_interpolator_t& interp, 
                   const k_particles_t& p, 
                   const k_particles_i_t& p_i, 
                   const species_t* sp) {
  const float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
  const float msp = sp->m;
  const size_t np = sp->np;
  double en = 0;

  Kokkos::parallel_reduce("energy_p_trilinear", Kokkos::RangePolicy<size_t>(0LLU, np), KOKKOS_LAMBDA(const size_t n, double& update) {
      float dx = p(n, particle_var::dx);
      float dy = p(n, particle_var::dy);
      float dz = p(n, particle_var::dz);
      int   i  = p_i(n);
      float v0 = p(n, particle_var::ux) + qdt_2mc*( ( interp(i, interpolator_var::ex   ) + dy*interp(i, interpolator_var::dexdy   ) ) +
                                                       dz*(   interp(i, interpolator_var::dexdz) + dy*interp(i, interpolator_var::d2exdydz) ) );
      float v1 = p(n, particle_var::uy) + qdt_2mc*( ( interp(i, interpolator_var::ey   ) + dz*interp(i, interpolator_var::deydz   ) ) +
                                                       dx*(   interp(i, interpolator_var::deydx) + dz*interp(i, interpolator_var::d2eydzdx) ) );
      float v2 = p(n, particle_var::uz) + qdt_2mc*( ( interp(i, interpolator_var::ez   ) + dx*interp(i, interpolator_var::dezdx   ) ) +
                                                       dy*(   interp(i, interpolator_var::dezdy) + dx*interp(i, interpolator_var::d2ezdxdy) ) );
      v0 = v0*v0 + v1*v1 + v2*v2;
      //v0 = (msp * p(n, particle_var::w)) * (v0 / (1 + sqrtf(1 + v0)));  // Relativistic kinetic energy
      v0 *= 0.5 * (msp * p(n, particle_var::w));  // Non-relativistic kinetic energy
      update += static_cast<double>(v0);
  }, en);
  return en;
}
*/

double
energy_p_kokkos(const species_t* RESTRICT sp,
                const interpolator_array_t* RESTRICT ia) {

    double local, global;
    grid_t* g = sp->g;

    if(!sp || !ia || sp->g != ia->g) ERROR(("Bad args"));

    const float dt_2mc = (sp->g->dt)/(2*sp->m*sp->g->cvac);
    const float dt_2c = (sp->g->dt)/(2*sp->g->cvac);
    const float msp = sp->m;
    const float q = sp->q;
    const size_t np = sp->np;

    local = energy_p_kernel(ia->k_i_d, sp->k_p_d, sp->k_p_i_d, q, dt_2mc, dt_2c, msp, np);
    Kokkos::fence();

    mp_allsum_d( &local, &global, 1 );
    return global*(static_cast<double>(g->cvac) * static_cast<double>(g->cvac));
}
