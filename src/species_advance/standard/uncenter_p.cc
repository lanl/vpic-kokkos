#define IN_spa
#include "spa_private.h"

// TODO: these should be refs?
void uncenter_p_kokkos(k_particles_soa_t k_part, k_interpolator_t k_interp, int np, float qdt_2mc_c) {
//  const float qdt_2mc        =     -qdt_2mc_c; // For backward half advance
//  const float qdt_4mc        = -0.5*qdt_2mc_c; // For backward half rotate
//  constexpr float one            = 1.;
//  constexpr float one_third      = 1./3.;
//  constexpr float two_fifteenths = 2./15.;
  const mixed_t qdt_2mc        =     static_cast<mixed_t>(-qdt_2mc_c); // For backward half advance
  const mixed_t qdt_4mc        = static_cast<mixed_t>(-0.5*qdt_2mc_c); // For backward half rotate
  const mixed_t one            = static_cast<mixed_t>(1.);
  const mixed_t one_third      = static_cast<mixed_t>(1./3.);
  const mixed_t two_fifteenths = static_cast<mixed_t>(2./15.);

// Particle defines (p->x)
  #define p_dx    k_part.dx(p_index) 
  #define p_dy    k_part.dy(p_index)
  #define p_dz    k_part.dz(p_index)
  #define p_ux    k_part.ux(p_index) // Load momentum
  #define p_uy    k_part.uy(p_index)
  #define p_uz    k_part.uz(p_index)
  #define pii     k_part.i(p_index)

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

  Kokkos::parallel_for("uncenter p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace >
      (0, np), KOKKOS_LAMBDA (int p_index) {

    int ii = pii;
//    float hax, hay, haz, l_cbx, l_cby, l_cbz;
//    float v0, v1, v2, v3, v4;
    mixed_t hax, hay, haz, l_cbx, l_cby, l_cbz;
    mixed_t v0, v1, v2, v3, v4;

    hax  = qdt_2mc*(      ( f_ex    + p_dy*f_dexdy    ) +
                     p_dz*( f_dexdz + p_dy*f_d2exdydz ) );
    hay  = qdt_2mc*(      ( f_ey    + p_dz*f_deydz    ) +
                     p_dx*( f_deydx + p_dz*f_d2eydzdx ) );
    haz  = qdt_2mc*(      ( f_ez    + p_dx*f_dezdx    ) +
                     p_dy*( f_dezdy + p_dx*f_d2ezdxdy ) );

    l_cbx  = f_cbx + p_dx*f_dcbxdx;            // Interpolate B
    l_cby  = f_cby + p_dy*f_dcbydy;
    l_cbz  = f_cbz + p_dz*f_dcbzdz;

    v0   = qdt_4mc/sqrt(one + (p_ux*p_ux + (p_uy*p_uy + p_uz*p_uz)));

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

//  // this goes to np using p_index
//  Kokkos::parallel_for("uncenter p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace >
//      (0, np/2), KOKKOS_LAMBDA (int p_index) {
//
////    int ii = pii;
//    int pi0 = k_part.i(p_index*2);
//    int pi1 = k_part.i(p_index*2+1);
//
//    packed_t hax, hay, haz, l_cbx, l_cby, l_cbz;
//    packed_t v0, v1, v2, v3, v4;
//
//    packed_t p_dx = packed_t(k_part.dx(p_index*2), k_part.dx(p_index*2+1));
//    packed_t p_dy = packed_t(k_part.dy(p_index*2), k_part.dy(p_index*2+1));
//    packed_t p_dz = packed_t(k_part.dz(p_index*2), k_part.dz(p_index*2+1));
//    packed_t p_ux = packed_t(k_part.ux(p_index*2), k_part.ux(p_index*2+1));
//    packed_t p_uy = packed_t(k_part.uy(p_index*2), k_part.uy(p_index*2+1));
//    packed_t p_uz = packed_t(k_part.uz(p_index*2), k_part.uz(p_index*2+1));
//
//    packed_t f_ex = packed_t( k_interp(pi0, interpolator_var::ex), 
//                              k_interp(pi1, interpolator_var::ex));;
//    packed_t f_ey = packed_t( k_interp(pi0, interpolator_var::ey), 
//                              k_interp(pi1, interpolator_var::ey));;
//    packed_t f_ez = packed_t( k_interp(pi0, interpolator_var::ez), 
//                              k_interp(pi1, interpolator_var::ez));;
//
//    packed_t f_cbx = packed_t(k_interp(pi0, interpolator_var::cbx), 
//                              k_interp(pi1, interpolator_var::cbx));;
//    packed_t f_cby = packed_t(k_interp(pi0, interpolator_var::cby), 
//                              k_interp(pi1, interpolator_var::cby));;
//    packed_t f_cbz = packed_t(k_interp(pi0, interpolator_var::cbz), 
//                              k_interp(pi1, interpolator_var::cbz));;
//
//    packed_t f_dexdy = packed_t(k_interp(pi0, interpolator_var::dexdy), 
//                                k_interp(pi1, interpolator_var::dexdy));
//    packed_t f_dexdz = packed_t(k_interp(pi0, interpolator_var::dexdz), 
//                                k_interp(pi1, interpolator_var::dexdz));
//    packed_t f_d2exdydz = packed_t( k_interp(pi0, interpolator_var::d2exdydz), 
//                                    k_interp(pi1, interpolator_var::d2exdydz));
//
//    packed_t f_deydx = packed_t(k_interp(pi0, interpolator_var::deydx), 
//                                k_interp(pi1, interpolator_var::deydx));
//    packed_t f_deydz = packed_t(k_interp(pi0, interpolator_var::deydz), 
//                                k_interp(pi1, interpolator_var::deydz));
//    packed_t f_d2eydzdx = packed_t( k_interp(pi0, interpolator_var::d2eydzdx), 
//                                    k_interp(pi1, interpolator_var::d2eydzdx));
//
//    packed_t f_dezdx = packed_t(k_interp(pi0, interpolator_var::dezdx), 
//                                k_interp(pi1, interpolator_var::dezdx));
//    packed_t f_dezdy = packed_t(k_interp(pi0, interpolator_var::dezdy), 
//                                k_interp(pi1, interpolator_var::dezdy));
//    packed_t f_d2ezdxdy = packed_t( k_interp(pi0, interpolator_var::d2ezdxdy), 
//                                    k_interp(pi1, interpolator_var::d2ezdxdy));
//
//    packed_t f_dcbxdx = packed_t(k_interp(pi0, interpolator_var::dcbxdx),
//                                 k_interp(pi1, interpolator_var::dcbxdx));
//    packed_t f_dcbydy = packed_t(k_interp(pi0, interpolator_var::dcbydy),
//                                 k_interp(pi1, interpolator_var::dcbydy));
//    packed_t f_dcbzdz = packed_t(k_interp(pi0, interpolator_var::dcbzdz),
//                                 k_interp(pi1, interpolator_var::dcbzdz));
//
//    hax  = packed_t(qdt_2mc)*( ( f_ex    + p_dy*f_dexdy    ) +
//                          p_dz*( f_dexdz + p_dy*f_d2exdydz ) );
//    hay  = packed_t(qdt_2mc)*( ( f_ey    + p_dz*f_deydz    ) +
//                          p_dx*( f_deydx + p_dz*f_d2eydzdx ) );
//    haz  = packed_t(qdt_2mc)*( ( f_ez    + p_dx*f_dezdx    ) +
//                          p_dy*( f_dezdy + p_dx*f_d2ezdxdy ) );
//
//    l_cbx  = f_cbx + p_dx*f_dcbxdx;            // Interpolate B
//    l_cby  = f_cby + p_dy*f_dcbydy;
//    l_cbz  = f_cbz + p_dz*f_dcbzdz;
//
//    v0   = packed_t(qdt_4mc)/sqrt(packed_t(one) + (p_ux*p_ux + (p_uy*p_uy + p_uz*p_uz)));
//
//    /**/                                     // Boris - scalars
//    v1    = l_cbx*l_cbx + (l_cby*l_cby + l_cbz*l_cbz);
//    v2    = (v0*v0)*v1;
//    v3    = v0*(packed_t(one)+v2*(packed_t(one_third)+v2*packed_t(two_fifteenths)));
//    v4    = v3/(packed_t(one)+v1*(v3*v3));
//    v4   += v4;
//
//    v0    = p_ux + v3*( p_uy*l_cbz - p_uz*l_cby );      // Boris - uprime
//    v1    = p_uy + v3*( p_uz*l_cbx - p_ux*l_cbz );
//    v2    = p_uz + v3*( p_ux*l_cby - p_uy*l_cbx );
//
//    packed_t temp_p_ux = v4*( v1*l_cbz - v2*l_cby );           // Boris - rotation
//    packed_t temp_p_uy = v4*( v2*l_cbx - v0*l_cbz );
//    packed_t temp_p_uz = v4*( v0*l_cby - v1*l_cbx );
//
//    temp_p_ux += hax;                              // Half advance E
//    temp_p_uy += hay;
//    temp_p_uz += haz;
//
//    k_part.ux(p_index*2) += temp_p_ux.low2half();
//    k_part.ux(p_index*2+1) += temp_p_ux.high2half();
//    k_part.uy(p_index*2) += temp_p_uy.low2half();
//    k_part.uy(p_index*2+1) += temp_p_uy.high2half();
//    k_part.uz(p_index*2) += temp_p_uz.low2half();
//    k_part.uz(p_index*2+1) += temp_p_uz.high2half();
//  });
}

void uncenter_p_kokkos(k_particles_soa_t k_part, k_particles_t k_particles, k_particles_i_t k_particles_i, k_interpolator_t k_interp, int np, float qdt_2mc_c) {
  const float qdt_2mc        =     -qdt_2mc_c; // For backward half advance
  const float qdt_4mc        = -0.5*qdt_2mc_c; // For backward half rotate
  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;

  // Particle defines (p->x)
//  #define p_dx    k_particles(p_index, particle_var::dx) 
//  #define p_dy    k_particles(p_index, particle_var::dy)
//  #define p_dz    k_particles(p_index, particle_var::dz)
//  #define p_ux    k_particles(p_index, particle_var::ux) // Load momentum
//  #define p_uy    k_particles(p_index, particle_var::uy)
//  #define p_uz    k_particles(p_index, particle_var::uz)
//  #define pii     k_particles_i(p_index)

  #define p_dx    k_part.dx(p_index) 
  #define p_dy    k_part.dy(p_index)
  #define p_dz    k_part.dz(p_index)
  #define p_ux    k_part.ux(p_index) // Load momentum
  #define p_uy    k_part.uy(p_index)
  #define p_uz    k_part.uz(p_index)
  #define pii     k_part.i(p_index)

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
  Kokkos::parallel_for("uncenter p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace >
      (0, np), KOKKOS_LAMBDA (int p_index) {

    int ii = pii;
//    float hax, hay, haz, l_cbx, l_cby, l_cbz;
//    float v0, v1, v2, v3, v4;
//
//    hax  = qdt_2mc*(      ( f_ex    + p_dy*f_dexdy    ) +
//                     p_dz*( f_dexdz + p_dy*f_d2exdydz ) );
//    hay  = qdt_2mc*(      ( f_ey    + p_dz*f_deydz    ) +
//                     p_dx*( f_deydx + p_dz*f_d2eydzdx ) );
//    haz  = qdt_2mc*(      ( f_ez    + p_dx*f_dezdx    ) +
//                     p_dy*( f_dezdy + p_dx*f_d2ezdxdy ) );
//    l_cbx  = f_cbx + p_dx*f_dcbxdx;            // Interpolate B
//    l_cby  = f_cby + p_dy*f_dcbydy;
//    l_cbz  = f_cbz + p_dz*f_dcbzdz;
//    v0   = qdt_4mc/(float)sqrt(one + (p_ux*p_ux + (p_uy*p_uy + p_uz*p_uz)));
//    /**/                                     // Boris - scalars
//    v1    = l_cbx*l_cbx + (l_cby*l_cby + l_cbz*l_cbz);
//    v2    = (v0*v0)*v1;
//    v3    = v0*(one+v2*(one_third+v2*two_fifteenths));
//    v4    = v3/(one+v1*(v3*v3));
//    v4   += v4;
//    v0    = p_ux + v3*( p_uy*l_cbz - p_uz*l_cby );      // Boris - uprime
//    v1    = p_uy + v3*( p_uz*l_cbx - p_ux*l_cbz );
//    v2    = p_uz + v3*( p_ux*l_cby - p_uy*l_cbx );
//    p_ux += v4*( v1*l_cbz - v2*l_cby );           // Boris - rotation
//    p_uy += v4*( v2*l_cbx - v0*l_cbz );
//    p_uz += v4*( v0*l_cby - v1*l_cbx );
//    p_ux += hax;                              // Half advance E
//    p_uy += hay;
//    p_uz += haz;

    mixed_t hax, hay, haz, l_cbx, l_cby, l_cbz;
    mixed_t v0, v1, v2, v3, v4;

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
    v0    = p_ux + v3*( p_uy*l_cbz - p_uz*l_cby );      // Boris - uprime
    v1    = p_uy + v3*( p_uz*l_cbx - p_ux*l_cbz );
    v2    = p_uz + v3*( p_ux*l_cby - p_uy*l_cbx );
    p_ux += v4*( v1*l_cbz - v2*l_cby );           // Boris - rotation
    p_uy += v4*( v2*l_cbx - v0*l_cbz );
    p_uz += v4*( v0*l_cby - v1*l_cbx );
    p_ux += hax;                              // Half advance E
    p_uy += hay;
    p_uz += haz;

//    hax  = mult(float2half(qdt_2mc),add(      add( f_ex    , mult(p_dy,f_dexdy)    ) ,
//                     mult(p_dz,add( f_dexdz , mult(p_dy,f_d2exdydz) )) ));
//    hay  = mult(float2half(qdt_2mc),add(      add( f_ey    , mult(p_dz,f_deydz)    ) ,
//                     mult(p_dx,add( f_deydx , mult(p_dz,f_d2eydzdx) )) ));
//    haz  = mult(float2half(qdt_2mc),add(      add( f_ez    , mult(p_dx,f_dezdx)    ) ,
//                     mult(p_dy,add( f_dezdy , mult(p_dx,f_d2ezdxdy) )) ));
//    l_cbx  = add(float2half(f_cbx) , mult(p_dx,f_dcbxdx));            // Interpolate B
//    l_cby  = add(float2half(f_cby) , mult(p_dy,f_dcbydy));
//    l_cbz  = add(float2half(f_cbz) , mult(p_dz,f_dcbzdz));
//    v0   = div(float2half(qdt_4mc),sqrt(add(one , add(mult(p_ux,p_ux) , add(mult(p_uy,p_uy) , mult(p_uz,p_uz))))));
//    /**/                                     // Boris - scalars
//    v1    = add(mult(l_cbx,l_cbx) , add(mult(l_cby,l_cby) , mult(l_cbz,l_cbz)));
//    v2    = mult(mult(v0,v0),v1);
//    v3    = mult(v0,add(one,mult(v2,add(one_third,mult(v2,two_fifteenths)))));
//    v4    = div(v3,add(one,mult(v1,mult(v3,v3))));
//    v4    = add(v4,v4);
//    v0    = add(p_ux , mult(v3,sub( mult(p_uy,l_cbz) , mult(p_uz,l_cby) )));      // Boris - uprime
//    v1    = add(p_uy , mult(v3,sub( mult(p_uz,l_cbx) , mult(p_ux,l_cbz) )));
//    v2    = add(p_uz , mult(v3,sub( mult(p_ux,l_cby) , mult(p_uy,l_cbx) )));
//    p_ux  = add(p_ux,mult(v4,sub( mult(v1,l_cbz) , mult(v2,l_cby) )));           // Boris - rotation
//    p_uy  = add(p_uy,mult(v4,sub( mult(v2,l_cbx) , mult(v0,l_cbz) )));
//    p_uz  = add(p_uz,mult(v4,sub( mult(v0,l_cby) , mult(v1,l_cbx) )));
//    p_ux  = add(p_ux,hax);                              // Half advance E
//    p_uy  = add(p_ux,hay);
//    p_uz  = add(p_ux,haz);
  });
}

void
uncenter_p( /**/  species_t            * RESTRICT sp,
            const interpolator_array_t * RESTRICT ia ) {
  //DECLARE_ALIGNED_ARRAY( center_p_pipeline_args_t, 128, args, 1 );

  if( !sp || !ia || sp->g!=ia->g ) ERROR(( "Bad args" ));

//  k_particles_t k_particles = sp->k_p_d;
//  k_particles_i_t k_particles_i = sp->k_p_i_d;
  k_interpolator_t k_interp    = ia->k_i_d;
  const int np                 = sp->np;
  const float qdt_2mc          = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
//  uncenter_p_kokkos(sp->k_p_soa_d, k_particles, k_particles_i, k_interp, np, qdt_2mc);
  uncenter_p_kokkos(sp->k_p_soa_d, k_interp, np, qdt_2mc);
}
