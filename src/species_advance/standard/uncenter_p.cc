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

    const interpolator_t intp = read_interpolator(k_interp, ii); // Load interpolators

    interpolate_e(intp, p_dx, p_dy, p_dz, hax, hay, haz, qdt_2mc, minus_dt_2c); // Interpolate E
                                      
    interpolate_b(intp, p_dx, p_dy, p_dz, l_cbx, l_cby, l_cbz); // Interpolate B

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
