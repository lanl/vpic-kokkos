#define IN_spa
#include "spa_private.h"
#include <Kokkos_Core.hpp>
#include "../../vpic/kokkos_helpers.h"

void uncenter_p_kokkos(k_particles_t k_particles, k_interpolator_t k_interp, int np, float qdt_2mc_c) {
  const float qdt_2mc        =     -qdt_2mc_c; // For backward half advance
  const float qdt_4mc        = -0.5*qdt_2mc_c; // For backward half rotate
  const float one            = 1.;
  const float one_third      = 1./3.;
  const float two_fifteenths = 2./15.;

  KOKKOS_PARTICLE_ENUMS();
  // Particle defines (p->x)
  #define p_dx    k_particles(p_index, particle_var::dx) 
  #define p_dy    k_particles(p_index, particle_var::dy)
  #define p_dz    k_particles(p_index, particle_var::dz)
  #define p_ux    k_particles(p_index, particle_var::ux) // Load momentum
  #define p_uy    k_particles(p_index, particle_var::uy)
  #define p_uz    k_particles(p_index, particle_var::uz)
  #define pii     k_particles(p_index, particle_var::pi)

  KOKKOS_INTERPOLATOR_ENUMS();
  // Interpolator Defines (f->x)
  #define f_cbx k_interp(ii, interpolator_var::cbx)
  #define f_cby k_interp(ii, interpolator_var::cby)
  #define f_cbz k_interp(ii, interpolator_var::cbz)
  #define f_ex  k_interp(ii, interpolator_var::ex)
  #define f_ey  k_interp(ii, interpolator_var::ey)
  #define f_ez  k_interp(ii, interpolator_var::ez)

  #define f_dexdy    k_interp(ii, interpolator_var::dexdy)
  #define f_dexdz    k_interp(ii, interpolator_var::dexdz)

  #define f_d2exdydz k_interp(ii, interpolator_var::d2exdydz)
  #define f_deydx    k_interp(ii, interpolator_var::deydx)
  #define f_deydz    k_interp(ii, interpolator_var::deydz)

  #define f_d2eydzdx k_interp(ii, interpolator_var::d2eydzdx)
  #define f_dezdx    k_interp(ii, interpolator_var::dezdx)
  #define f_dezdy    k_interp(ii, interpolator_var::dezdy)

  #define f_d2ezdxdy k_interp(ii, interpolator_var::d2ezdxdy)
  #define f_dcbxdx   k_interp(ii, interpolator_var::dcbxdx)
  #define f_dcbydy   k_interp(ii, interpolator_var::dcbydy)
  #define f_dcbzdz   k_interp(ii, interpolator_var::dcbzdz)


  // this goes to np using p_index
  // check for off by one errors
  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace >
      (0, np), KOKKOS_LAMBDA (int p_index) {

    int ii = pii;
    float hax, hay, haz, l_cbx, l_cby, l_cbz;
    float v0, v1, v2, v3, v4;

    hax  = qdt_2mc*(      ( f_ex    + p_dy*f_dexdy    ) +
                     p_dz*( f_dexdz + p_dy*f_d2exdydz ) );
    hay  = qdt_2mc*(      ( f_ey    + p_dz*f_deydz    ) +
                     p_dx*( f_deydx + p_dz*f_d2eydzdx ) );
    haz  = qdt_2mc*(      ( f_ez    + p_dx*f_dezdx    ) +
                     p_dy*( f_dezdy + p_dx*f_d2ezdxdy ) );
    l_cbx  = f_cbx + p_dx*f_dcbxdx;            // Interpolate B
    l_cby  = f_cby + p_dy*f_dcbydy;
    l_cbz  = f_cbz + p_dz*f_dcbzdz;
    v0   = qdt_4mc/(float)sqrt(one + (p_ux*p_ux + (p_uy*p_uy + p_uz*p_uz)));
    /**/                                     // Boris - scalars
    v1    = l_cbx*l_cbx + (l_cby*l_cby + l_cbz*l_cbz);
    v2    = (v0*v0)*v1;
    v3    = v0*(one+v2*(one_third+v2*two_fifteenths));
    v4    = v3/(one+v1*(v3*v3));
    v4   += v4;
    v0    = p_ux + v3*( p_uy*l_cbz - uz*l_cby );      // Boris - uprime
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

  k_particles_t    k_particles = sp->k_p_d;
  k_interpolator_t k_interp    = ia->k_i_d;
  const int np                 = sp->np;
  const float qdt_2mc          = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
  uncenter_p_kokkos(k_particles, k_interp, np, qdt_2mc);
}
