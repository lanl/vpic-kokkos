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

  KOKKOS_ENUMS();
  // Particle defines (p->x)
  #define dx    k_particles(p_index, dx) 
  #define dy    k_particles(p_index, dy)
  #define dz    k_particles(p_index, dz)
  #define ux    k_particles(p_index, ux) // Load momentum
  #define uy    k_particles(p_index, uy)
  #define uz    k_particles(p_index, uz)

  // Interpolator Defines (f->x)
  #define f_cbx k_interp(ii, cbx)
  #define f_cby k_interp(ii, cby)
  #define f_cbz k_interp(ii, cbz)
  #define f_ex  k_interp(ii, ex)
  #define f_ey  k_interp(ii, ey)
  #define f_ez  k_interp(ii, ez)

  #define f_dexdy    k_interp(ii, dexdy)
  #define f_dexdz    k_interp(ii, dexdz)
  #define f_d2exdydz k_interp(ii, d2exdydz)
  #define f_deydx    k_interp(ii, deydx)
  #define f_deydz    k_interp(ii, deydz)
  #define f_d2eydzdx k_interp(ii, d2eydzdx)
  #define f_dezdx    k_interp(ii, dezdx)
  #define f_dezdy    k_interp(ii, dezdy)
  #define f_d2ezdxdy k_interp(ii, d2ezdxdy)
  #define f_dcbxdx   k_interp(ii, dcbxdx)
  #define f_dcbydy   k_interp(ii, dcbydy)
  #define f_dcbzdz   k_interp(ii, dcbzdz)

  // this goes to np using p_index
  // check for off by one errors
  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace >
      (0, np), KOKKOS_LAMBDA (int p_index) {

    int ii = k_particles(p_index, pi);
    float hax, hay, haz, l_cbx, l_cby, l_cbz;
    float v0, v1, v2, v3, v4;

    hax  = qdt_2mc*(    ( f_ex    + dy*f_dexdy    ) +
                     dz*( f_dexdz + dy*f_d2exdydz ) );
    hay  = qdt_2mc*(    ( f_ey    + dz*f_deydz    ) +
                     dx*( f_deydx + dz*f_d2eydzdx ) );
    haz  = qdt_2mc*(    ( f_ez    + dx*f_dezdx    ) +
                     dy*( f_dezdy + dx*f_d2ezdxdy ) );
    l_cbx  = f_cbx + dx*f_dcbxdx;            // Interpolate B
    l_cby  = f_cby + dy*f_dcbydy;
    l_cbz  = f_cbz + dz*f_dcbzdz;
    v0   = qdt_4mc/(float)sqrt(one + (ux*ux + (uy*uy + uz*uz)));
    /**/                                     // Boris - scalars
    v1   = l_cbx*l_cbx + (l_cby*l_cby + l_cbz*l_cbz);
    v2   = (v0*v0)*v1;
    v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
    v4   = v3/(one+v1*(v3*v3));
    v4  += v4;
    v0   = ux + v3*( uy*l_cbz - uz*l_cby );      // Boris - uprime
    v1   = uy + v3*( uz*l_cbx - ux*l_cbz );
    v2   = uz + v3*( ux*l_cby - uy*l_cbx );
    ux  += v4*( v1*l_cbz - v2*l_cby );           // Boris - rotation
    uy  += v4*( v2*l_cbx - v0*l_cbz );
    uz  += v4*( v0*l_cby - v1*l_cbx );
    ux  += hax;                              // Half advance E
    uy  += hay;
    uz  += haz;
  });

}

void
uncenter_p( /**/  species_t            * RESTRICT sp,
            const interpolator_array_t * RESTRICT ia ) {
  //DECLARE_ALIGNED_ARRAY( center_p_pipeline_args_t, 128, args, 1 );

  if( !sp || !ia || sp->g!=ia->g ) ERROR(( "Bad args" ));

  k_particles_t    k_particles = sp->k_p_d;
  k_interpolator_t k_interp     = ia->k_i_d;
  const int np        = sp->np;
  const float qdt_2mc      = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
  uncenter_p_kokkos(k_particles, k_interp, np, qdt_2mc);
}
