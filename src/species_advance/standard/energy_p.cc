#define IN_spa
#define HAS_V4_PIPELINE
#include "spa_private.h"

// This function calculates kinetic energy, normalized by c^2.
void
energy_p_pipeline( energy_p_pipeline_args_t * RESTRICT args,
                   int pipeline_rank,
                   int n_pipeline ) {
  const interpolator_t * RESTRICT ALIGNED(128) f = args->f;
  const particle_t     * RESTRICT ALIGNED(32)  p = args->p;
  const float qdt_2mc = args->qdt_2mc;
  const float msp     = args->msp;
#ifdef EXTERNAL_FORCE
  const float dt_2c = args->dt_2c;
#endif
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
#ifdef SHAPE_NGP
  #ifdef EXTERNAL_FORCE
    v0  = p[n].ux + qdt_2mc * ( f[i].ex + f[i].Ex0 ) + dt_2c * f[i].Gx0;
    v1  = p[n].uy + qdt_2mc * ( f[i].ey + f[i].Ey0 ) + dt_2c * f[i].Gy0;
    v2  = p[n].uz + qdt_2mc * ( f[i].ez + f[i].Ez0 ) + dt_2c * f[i].Gz0;
  #else
    v0  = p[n].ux + qdt_2mc * f[i].ex;
    v1  = p[n].uy + qdt_2mc * f[i].ey;
    v2  = p[n].uz + qdt_2mc * f[i].ez;
  #endif
#elif defined( SHAPE_QS )
  #ifdef EXTERNAL_FORCE
    v0 = p[n].ux + qdt_2mc*( f[i].ex + dx*( f[i].dexdx + dx*f[i].d2exdx )
                                     + dy*( f[i].dexdy + dy*f[i].d2exdy )
                                     + dz*( f[i].dexdz + dz*f[i].d2exdz )
                             + f[i].Ex0 + dx*( f[i].dEx0dx + dx*f[i].d2Ex0dx )
                                        + dy*( f[i].dEx0dy + dy*f[i].d2Ex0dy )
                                        + dz*( f[i].dEx0dz + dz*f[i].d2Ex0dz ) );
    v1 = p[n].uy + qdt_2mc*( f[i].ey + dx*( f[i].deydx + dx*f[i].d2eydx )
                                     + dy*( f[i].deydy + dy*f[i].d2eydy )
                                     + dz*( f[i].deydz + dz*f[i].d2eydz )
                             + f[i].Ey0 + dx*( f[i].dEy0dx + dx*f[i].d2Ey0dx )
                                        + dy*( f[i].dEy0dy + dy*f[i].d2Ey0dy )
                                        + dz*( f[i].dEy0dz + dz*f[i].d2Ey0dz ) );
    v2 = p[n].uz + qdt_2mc*( f[i].ez + dx*( f[i].dezdx + dx*f[i].d2ezdx )
                                     + dy*( f[i].dezdy + dy*f[i].d2ezdy )
                                     + dz*( f[i].dezdz + dz*f[i].d2ezdz )
                             + f[i].Ez0 + dx*( f[i].dEz0dx + dx*f[i].d2Ez0dx )
                                        + dy*( f[i].dEz0dy + dy*f[i].d2Ez0dy )
                                        + dz*( f[i].dEz0dz + dz*f[i].d2Ez0dz ) );
    v0 += dt_2c *( f[i].Gx0 + dx*( f[i].dGx0dx + dx*f[i].d2Gx0dx )
                            + dy*( f[i].dGx0dy + dy*f[i].d2Gx0dy )
                            + dz*( f[i].dGx0dz + dz*f[i].d2Gx0dz ) );
    v1 += dt_2c *( f[i].Gy0 + dx*( f[i].dGy0dx + dx*f[i].d2Gy0dx )
                            + dy*( f[i].dGy0dy + dy*f[i].d2Gy0dy )
                            + dz*( f[i].dGy0dz + dz*f[i].d2Gy0dz ) );
    v2 += dt_2c *( f[i].Gz0 + dx*( f[i].dGz0dx + dx*f[i].d2Gz0dx )
                            + dy*( f[i].dGz0dy + dy*f[i].d2Gz0dy )
                            + dz*( f[i].dGz0dz + dz*f[i].d2Gz0dz ) );
  #else
    v0 = p[n].ux + qdt_2mc*( f[i].ex + dx*( f[i].dexdx + dx*f[i].d2exdx )
                                     + dy*( f[i].dexdy + dy*f[i].d2exdy )
                                     + dz*( f[i].dexdz + dz*f[i].d2exdz ) );
    v1 = p[n].uy + qdt_2mc*( f[i].ey + dx*( f[i].deydx + dx*f[i].d2eydx )
                                     + dy*( f[i].deydy + dy*f[i].d2eydy )
                                     + dz*( f[i].deydz + dz*f[i].d2eydz ) );
    v2 = p[n].uz + qdt_2mc*( f[i].ez + dx*( f[i].dezdx + dx*f[i].d2ezdx )
                                     + dy*( f[i].dezdy + dy*f[i].d2ezdy )
                                     + dz*( f[i].dezdz + dz*f[i].d2ezdz ) );
  #endif
#endif
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
energy_p_kernel(const k_interpolator_t& k_interp, const k_particles_t& k_particles, const k_particles_i_t& k_particles_i, const float qdt_2mc, const float dt_2c, const float msp, const int np) {
//  const interpolator_t * RESTRICT ALIGNED(128) f = args->f;
//  const particle_t     * RESTRICT ALIGNED(32)  p = args->p;
//  const float qdt_2mc = args->qdt_2mc;
//  const float msp     = args->msp;
//  const float one     = 1;

  double en = 0;

  // Determine which particles this pipeline processes

//  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, n0, n1 );
/*
    int _N = np, _b = 16, _p = pipeline_rank, _P = n_pipeline;
    double _t = static_cast<double>(_N/_b) / static_cast<double>(_P);
    int _i = _b * static_cast<int>(_t * static_cast<double>(_p) + 0.5);
    n1 = (_p == _P) ? (_N % _b) : (_b * static_cast<int>(_t * static_cast<double>(_p+1) + 0.5) - _i;
    n0 = _i
    n1 += n0;
*/
  // Process particles quads for this pipeline
/*
  for( n=n0; n<n1; n++ ) {
    dx  = p[n].dx;
    dy  = p[n].dy;
    dz  = p[n].dz;
    i   = p[n].i;
    v0  = p[n].ux + qdt_2mc*(    ( f[i].ex    + dy*f[i].dexdy    ) +
                              dz*( f[i].dexdz + dy*f[i].d2exdydz ) );
    v1  = p[n].uy + qdt_2mc*(    ( f[i].ey    + dz*f[i].deydz    ) +
                              dx*( f[i].deydx + dz*f[i].d2eydzdx ) );
    v2  = p[n].uz + qdt_2mc*(    ( f[i].ez    + dx*f[i].dezdx    ) +
                              dy*( f[i].dezdy + dx*f[i].d2ezdxdy ) );
    v0  = v0*v0 + v1*v1 + v2*v2;
    v0  = (msp * p[n].w) * (v0 / (one + sqrtf(one + v0)));
    en += (double)v0;
  }
*/

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

    #define f_Ex0       k_interp(ii, interpolator_var::Ex0)
    #define f_dEx0dx    k_interp(ii, interpolator_var::dEx0dx)
    #define f_dEx0dy    k_interp(ii, interpolator_var::dEx0dy)
    #define f_dEx0dz    k_interp(ii, interpolator_var::dEx0dz)
    #define f_d2Ex0dx   k_interp(ii, interpolator_var::d2Ex0dx)
    #define f_d2Ex0dy   k_interp(ii, interpolator_var::d2Ex0dy)
    #define f_d2Ex0dz   k_interp(ii, interpolator_var::d2Ex0dz)
    #define f_Ey0       k_interp(ii, interpolator_var::Ey0)
    #define f_dEy0dx    k_interp(ii, interpolator_var::dEy0dx)
    #define f_dEy0dy    k_interp(ii, interpolator_var::dEy0dy)
    #define f_dEy0dz    k_interp(ii, interpolator_var::dEy0dz)
    #define f_d2Ey0dx   k_interp(ii, interpolator_var::d2Ey0dx)
    #define f_d2Ey0dy   k_interp(ii, interpolator_var::d2Ey0dy)
    #define f_d2Ey0dz   k_interp(ii, interpolator_var::d2Ey0dz)
    #define f_Ez0       k_interp(ii, interpolator_var::Ez0)
    #define f_dEz0dx    k_interp(ii, interpolator_var::dEz0dx)
    #define f_dEz0dy    k_interp(ii, interpolator_var::dEz0dy)
    #define f_dEz0dz    k_interp(ii, interpolator_var::dEz0dz)
    #define f_d2Ez0dx   k_interp(ii, interpolator_var::d2Ez0dx)
    #define f_d2Ez0dy   k_interp(ii, interpolator_var::d2Ez0dy)
    #define f_d2Ez0dz   k_interp(ii, interpolator_var::d2Ez0dz)

    #define f_Gx0       k_interp(ii, interpolator_var::Gx0)
    #define f_dGx0dx    k_interp(ii, interpolator_var::dGx0dx)
    #define f_dGx0dy    k_interp(ii, interpolator_var::dGx0dy)
    #define f_dGx0dz    k_interp(ii, interpolator_var::dGx0dz)
    #define f_d2Gx0dx   k_interp(ii, interpolator_var::d2Gx0dx)
    #define f_d2Gx0dy   k_interp(ii, interpolator_var::d2Gx0dy)
    #define f_d2Gx0dz   k_interp(ii, interpolator_var::d2Gx0dz)
    #define f_Gy0       k_interp(ii, interpolator_var::Gy0)
    #define f_dGy0dx    k_interp(ii, interpolator_var::dGy0dx)
    #define f_dGy0dy    k_interp(ii, interpolator_var::dGy0dy)
    #define f_dGy0dz    k_interp(ii, interpolator_var::dGy0dz)
    #define f_d2Gy0dx   k_interp(ii, interpolator_var::d2Gy0dx)
    #define f_d2Gy0dy   k_interp(ii, interpolator_var::d2Gy0dy)
    #define f_d2Gy0dz   k_interp(ii, interpolator_var::d2Gy0dz)
    #define f_Gz0       k_interp(ii, interpolator_var::Gz0)
    #define f_dGz0dx    k_interp(ii, interpolator_var::dGz0dx)
    #define f_dGz0dy    k_interp(ii, interpolator_var::dGz0dy)
    #define f_dGz0dz    k_interp(ii, interpolator_var::dGz0dz)
    #define f_d2Gz0dx   k_interp(ii, interpolator_var::d2Gz0dx)
    #define f_d2Gz0dy   k_interp(ii, interpolator_var::d2Gz0dy)
    #define f_d2Gz0dz   k_interp(ii, interpolator_var::d2Gz0dz)

    Kokkos::parallel_reduce(np, KOKKOS_LAMBDA(const int n, double& update) {
        float ux = k_particles(n, particle_var::ux);
        float uy = k_particles(n, particle_var::uy);
        float uz = k_particles(n, particle_var::uz);
        float dx = k_particles(n, particle_var::dx);
        float dy = k_particles(n, particle_var::dy);
        float dz = k_particles(n, particle_var::dz);
        int   ii = k_particles_i(n);
#ifdef SHAPE_NGP
  #ifdef EXTERNAL_FORCE
        float v0 = ux + qdt_2mc * (f_ex + f_Ex0) + dt_2c * f_Gx0;
        float v1 = uy + qdt_2mc * (f_ey + f_Ey0) + dt_2c * f_Gy0;
        float v2 = uz + qdt_2mc * (f_ez + f_Ez0) + dt_2c * f_Gz0;
  #else
        float v0 = ux + qdt_2mc * f_ex;
        float v1 = uy + qdt_2mc * f_ey;
        float v2 = uz + qdt_2mc * f_ez;
  #endif
#elif defined( SHAPE_QS )
        // Interpolate E
  #ifdef EXTERNAL_FORCE
        float v0 = ux + qdt_2mc*( f_ex + dx*( f_dexdx + dx*f_d2exdx )
                                       + dy*( f_dexdy + dy*f_d2exdy )
                                       + dz*( f_dexdz + dz*f_d2exdz )
                                  + f_Ex0 + dx*( f_dEx0dx + dx*f_d2Ex0dx )
                                          + dy*( f_dEx0dy + dy*f_d2Ex0dy )
                                          + dz*( f_dEx0dz + dz*f_d2Ex0dz ) );
        float v1 = uy + qdt_2mc*( f_ey + dx*( f_deydx + dx*f_d2eydx )
                                       + dy*( f_deydy + dy*f_d2eydy )
                                       + dz*( f_deydz + dz*f_d2eydz )
                                  + f_Ey0 + dx*( f_dEy0dx + dx*f_d2Ey0dx )
                                          + dy*( f_dEy0dy + dy*f_d2Ey0dy )
                                          + dz*( f_dEy0dz + dz*f_d2Ey0dz ) );
        float v2 = uz + qdt_2mc*( f_ez + dx*( f_dezdx + dx*f_d2ezdx )
                                       + dy*( f_dezdy + dy*f_d2ezdy )
                                       + dz*( f_dezdz + dz*f_d2ezdz )
                                  + f_Ez0 + dx*( f_dEz0dx + dx*f_d2Ez0dx )
                                          + dy*( f_dEz0dy + dy*f_d2Ez0dy )
                                          + dz*( f_dEz0dz + dz*f_d2Ez0dz ) );
        v0 += dt_2c *( f_Gx0 + dx*( f_dGx0dx + dx*f_d2Gx0dx )
                             + dy*( f_dGx0dy + dy*f_d2Gx0dy )
                             + dz*( f_dGx0dz + dz*f_d2Gx0dz ) );
        v1 += dt_2c *( f_Gy0 + dx*( f_dGy0dx + dx*f_d2Gy0dx )
                             + dy*( f_dGy0dy + dy*f_d2Gy0dy )
                             + dz*( f_dGy0dz + dz*f_d2Gy0dz ) );
        v2 += dt_2c *( f_Gz0 + dx*( f_dGz0dx + dx*f_d2Gz0dx )
                             + dy*( f_dGz0dy + dy*f_d2Gz0dy )
                             + dz*( f_dGz0dz + dz*f_d2Gz0dz ) );
  #else
        float v0 = ux + qdt_2mc*( f_ex + dx*( f_dexdx + dx*f_d2exdx )
                                       + dy*( f_dexdy + dy*f_d2exdy )
                                       + dz*( f_dexdz + dz*f_d2exdz ) );
        float v1 = uy + qdt_2mc*( f_ey + dx*( f_deydx + dx*f_d2eydx )
                                       + dy*( f_deydy + dy*f_d2eydy )
                                       + dz*( f_deydz + dz*f_d2eydz ) );
        float v2 = uz + qdt_2mc*( f_ez + dx*( f_dezdx + dx*f_d2ezdx )
                                       + dy*( f_dezdy + dy*f_d2ezdy )
                                       + dz*( f_dezdz + dz*f_d2ezdz ) );
  #endif
#endif
        v0 = v0*v0 + v1*v1 + v2*v2;
        //v0 = (msp * k_particles(n, particle_var::w)) * (v0 / (1 + sqrtf(1 + v0)));  // Relativistic kinetic energy
        v0 *= 0.5 * (msp * k_particles(n, particle_var::w));  // Non-relativistic kinetic energy
        update += static_cast<double>(v0);
    }, en);
    return en;

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

    #undef f_Ex0
    #undef f_dEx0dx
    #undef f_dEx0dy
    #undef f_dEx0dz
    #undef f_d2Ex0dx
    #undef f_d2Ex0dy
    #undef f_d2Ex0dz
    #undef f_Ey0
    #undef f_dEy0dx
    #undef f_dEy0dy
    #undef f_dEy0dz
    #undef f_d2Ey0dx
    #undef f_d2Ey0dy
    #undef f_d2Ey0dz
    #undef f_Ez0
    #undef f_dEz0dx
    #undef f_dEz0dy
    #undef f_dEz0dz
    #undef f_d2Ez0dx
    #undef f_d2Ez0dy
    #undef f_d2Ez0dz

    #undef f_Gx0
    #undef f_dGx0dx
    #undef f_dGx0dy
    #undef f_dGx0dz
    #undef f_d2Gx0dx
    #undef f_d2Gx0dy
    #undef f_d2Gx0dz
    #undef f_Gy0
    #undef f_dGy0dx
    #undef f_dGy0dy
    #undef f_dGy0dz
    #undef f_d2Gy0dx
    #undef f_d2Gy0dy
    #undef f_d2Gy0dz
    #undef f_Gz0
    #undef f_dGz0dx
    #undef f_dGz0dy
    #undef f_dGz0dz
    #undef f_d2Gz0dx
    #undef f_d2Gz0dy
    #undef f_d2Gz0dz
} // energy_p_kernel(...)

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
#ifdef EXTERNAL_FORCE
  args->dt_2c   = (sp->g->dt)/(2*sp->g->cvac);
#endif

  EXEC_PIPELINES( energy_p, args, 0 );
  WAIT_PIPELINES();

  local = 0; for( rank=0; rank<=N_PIPELINE; rank++ ) local += en[rank];
  mp_allsum_d( &local, &global, 1 );
  return global*((double)sp->g->cvac*(double)sp->g->cvac);
}

double
energy_p_kokkos(const species_t* RESTRICT sp,
         const interpolator_array_t* RESTRICT ia) {

    double local, global;

    if(!sp || !ia || sp->g != ia->g) ERROR(("Bad args"));

    float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
    float dt_2c = (sp->g->dt)/(2*sp->g->cvac);

    local = energy_p_kernel(ia->k_i_d, sp->k_p_d, sp->k_p_i_d, qdt_2mc, dt_2c, sp->m, sp->np);
    Kokkos::fence();

    mp_allsum_d( &local, &global, 1 );
    return global*(static_cast<double>(sp->g->cvac) * static_cast<double>(sp->g->cvac));
}
