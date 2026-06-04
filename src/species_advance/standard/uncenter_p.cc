#define IN_spa
#include "spa_private.h"

void 
uncenter_p_kokkos(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_interpolator_t& k_interp,
        size_t np,
        float qdt_2mc_c,
        float dt_2c
)
{
#ifndef VARIABLE_CHARGE
  const float qdt_2mc        =     -qdt_2mc_c; // For backward half advance
  const float qdt_4mc        = -0.5*qdt_2mc_c; // For backward half rotate
#else
  const float dt_2mc_c = qdt_2mc_c;
#endif
  const float minus_dt_2c    = -dt_2c; // For backwards half advance
  const float one            = 1.;
  const float one_third      = 1./3.;
  const float two_fifteenths = 2./15.;

  // Particle defines (p->x)
  #define p_dx    k_particles(p_index, particle_var::dx)
  #define p_dy    k_particles(p_index, particle_var::dy)
  #define p_dz    k_particles(p_index, particle_var::dz)
  #define p_ux    k_particles(p_index, particle_var::ux) // Load momentum
  #define p_uy    k_particles(p_index, particle_var::uy)
  #define p_uz    k_particles(p_index, particle_var::uz)
#ifdef VARIABLE_CHARGE
  #define p_q     k_particles(p_index, particle_var::qp)
#endif
  #define pii     k_particles_i(p_index)

  // Interpolator Defines (f->x)
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


  // this goes to np using p_index
  Kokkos::parallel_for("uncenter p", Kokkos::RangePolicy <>(0, np), 
    KOKKOS_LAMBDA (const size_t p_index) {

    int ii = pii;
    float hax, hay, haz, l_cbx, l_cby, l_cbz;
    float v0, v1, v2, v3, v4;
    
#ifdef VARIABLE_CHARGE
    float qdt_2mc = -dt_2mc_c*p_q; // For backwards half advance
    float qdt_4mc = 0.5*qdt_2mc;   // For backwards half rotation
#endif
    
#ifdef SHAPE_NGP
  #ifdef EXTERNAL_FORCE
    hax  = qdt_2mc*( f_ex + f_Ex0 ) + minus_dt_2c * f_Gx0;
    hay  = qdt_2mc*( f_ey + f_Ey0 ) + minus_dt_2c * f_Gy0;
    haz  = qdt_2mc*( f_ez + f_Ez0 ) + minus_dt_2c * f_Gz0;
  #else
    hax  = qdt_2mc*( f_ex );
    hay  = qdt_2mc*( f_ey );
    haz  = qdt_2mc*( f_ez );
  #endif
    l_cbx  = f_cbx;// + p_dx*f_dcbxdx;            // Interpolate B
    l_cby  = f_cby;// + p_dy*f_dcbydy;
    l_cbz  = f_cbz;// + p_dz*f_dcbzdz;
#elif defined( SHAPE_QS )
  #ifdef EXTERNAL_FORCE
    hax  = qdt_2mc*( f_ex + p_dx*( f_dexdx + p_dx*f_d2exdx )      // Interpolate E, E0
                          + p_dy*( f_dexdy + p_dy*f_d2exdy )
                          + p_dz*( f_dexdz + p_dz*f_d2exdz )
                     + f_Ex0 + p_dx*( f_dEx0dx + p_dx*f_d2Ex0dx )
                             + p_dy*( f_dEx0dy + p_dy*f_d2Ex0dy )
                             + p_dz*( f_dEx0dz + p_dz*f_d2Ex0dz ) );
    hay  = qdt_2mc*( f_ey + p_dx*( f_deydx + p_dx*f_d2eydx )
                          + p_dy*( f_deydy + p_dy*f_d2eydy )
                          + p_dz*( f_deydz + p_dz*f_d2eydz )
                     + f_Ey0 + p_dx*( f_dEy0dx + p_dx*f_d2Ey0dx )
                             + p_dy*( f_dEy0dy + p_dy*f_d2Ey0dy )
                             + p_dz*( f_dEy0dz + p_dz*f_d2Ey0dz ) );
    haz  = qdt_2mc*( f_ez + p_dx*( f_dezdx + p_dx*f_d2ezdx )
                          + p_dy*( f_dezdy + p_dy*f_d2ezdy )
                          + p_dz*( f_dezdz + p_dz*f_d2ezdz )
                     + f_Ez0 + p_dx*( f_dEz0dx + p_dx*f_d2Ez0dx )
                             + p_dy*( f_dEz0dy + p_dy*f_d2Ez0dy )
                             + p_dz*( f_dEz0dz + p_dz*f_d2Ez0dz ) );
    hax += minus_dt_2c *( f_Gx0 + p_dx*( f_dGx0dx + p_dx*f_d2Gx0dx )      // Interpolate G0
                                + p_dy*( f_dGx0dy + p_dy*f_d2Gx0dy )
                                + p_dz*( f_dGx0dz + p_dz*f_d2Gx0dz ) );
    hay += minus_dt_2c *( f_Gy0 + p_dx*( f_dGy0dx + p_dx*f_d2Gy0dx )
                                + p_dy*( f_dGy0dy + p_dy*f_d2Gy0dy )
                                + p_dz*( f_dGy0dz + p_dz*f_d2Gy0dz ) );
    haz += minus_dt_2c *( f_Gz0 + p_dx*( f_dGz0dx + p_dx*f_d2Gz0dx )
                                + p_dy*( f_dGz0dy + p_dy*f_d2Gz0dy )
                                + p_dz*( f_dGz0dz + p_dz*f_d2Gz0dz ) );
  #else
    hax  = qdt_2mc*( f_ex + p_dx*( f_dexdx + p_dx*f_d2exdx )      // Interpolate E
                          + p_dy*( f_dexdy + p_dy*f_d2exdy )
                          + p_dz*( f_dexdz + p_dz*f_d2exdz ) );
    hay  = qdt_2mc*( f_ey + p_dx*( f_deydx + p_dx*f_d2eydx )
                          + p_dy*( f_deydy + p_dy*f_d2eydy )
                          + p_dz*( f_deydz + p_dz*f_d2eydz ) );
    haz  = qdt_2mc*( f_ez + p_dx*( f_dezdx + p_dx*f_d2ezdx )
                          + p_dy*( f_dezdy + p_dy*f_d2ezdy )
                          + p_dz*( f_dezdz + p_dz*f_d2ezdz ) );
  #endif
    l_cbx  = f_cbx + p_dx*( f_dcbxdx + p_dx*f_d2cbxdx )             // Interpolate B
                   + p_dy*( f_dcbxdy + p_dy*f_d2cbxdy )
                   + p_dz*( f_dcbxdz + p_dz*f_d2cbxdz );
    l_cby  = f_cby + p_dx*( f_dcbydx + p_dx*f_d2cbydx )
                   + p_dy*( f_dcbydy + p_dy*f_d2cbydy )
                   + p_dz*( f_dcbydz + p_dz*f_d2cbydz );
    l_cbz  = f_cbz + p_dx*( f_dcbzdx + p_dx*f_d2cbzdx )
                   + p_dy*( f_dcbzdy + p_dy*f_d2cbzdy )
                   + p_dz*( f_dcbzdz + p_dz*f_d2cbzdz );
#endif
    v0   = qdt_4mc;///(float)sqrt(one + (p_ux*p_ux + (p_uy*p_uy + p_uz*p_uz)));
    /**/                                     // Boris - scalars
    v1    = l_cbx*l_cbx + (l_cby*l_cby + l_cbz*l_cbz);
    v2    = (v0*v0)*v1;
    v3    = v0*(one+v2*(one_third+v2*two_fifteenths));
    v4    = v3/(one+v1*(v3*v3));
    v4   += v4;
    v0    = p_ux + v3*( p_uy*l_cbz - p_uz*l_cby );      // Boris - uprime
    v1    = p_uy + v3*( p_uz*l_cbx - p_ux*l_cbz );
    v2    = p_uz + v3*( p_ux*l_cby - p_uy*l_cbx );
    p_ux += v4*( v1*l_cbz - v2*l_cby );           // Boris - rotation
    p_uy += v4*( v2*l_cbx - v0*l_cbz );
    p_uz += v4*( v0*l_cby - v1*l_cbx );
    p_ux += hax;                              // Half advance E
    p_uy += hay;
    p_uz += haz;
  });

  #undef p_dx
  #undef p_dy
  #undef p_dz
  #undef p_ux
  #undef p_uy
  #undef p_uz
#ifdef VARIABLE_CHARGE
  #undef p_q
#endif
  #undef pii

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

}

void
uncenter_p( /**/  species_t            * RESTRICT sp,
            const interpolator_array_t * RESTRICT ia ) {
  //DECLARE_ALIGNED_ARRAY( center_p_pipeline_args_t, 128, args, 1 );

  if( !sp || !ia || sp->g!=ia->g ) ERROR(( "Bad args" ));

  k_particles_t k_particles = sp->k_p_d;
  k_particles_i_t k_particles_i = sp->k_p_i_d;
  k_interpolator_t k_interp    = ia->k_i_d;
  const size_t np              = sp->np;
#ifdef VARIABLE_CHARGE
  const float qdt_2mc          = (sp->g->dt)/(2*sp->m*sp->g->cvac); // Multiply by qp in pipeline
#else
  const float qdt_2mc          = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
#endif
  const float dt_2c            = (sp->g->dt)/(2*sp->g->cvac);
  uncenter_p_kokkos(k_particles, k_particles_i, k_interp, np, qdt_2mc, dt_2c);
}
