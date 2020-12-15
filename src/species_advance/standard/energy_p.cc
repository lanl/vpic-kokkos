#define IN_spa
#include "spa_private.h"

template<
  class interp_t
> double
energy_p_kokkos(
  k_particles_t& k_particles,
  k_particles_i_t& k_particles_i,
  interp_t interpolate,
  const float qdt_2mc,
  const float msp,
  const int np
)
{
  double energy = 0;
  Kokkos::parallel_reduce(np, KOKKOS_LAMBDA(const int n, double& update) {

    // Load position
    float dx = k_particles(n, particle_var::dx);
    float dy = k_particles(n, particle_var::dy);
    float dz = k_particles(n, particle_var::dz);
    int   ii = k_particles_i(n);

    // Load momentum
    float ux = k_particles(n, particle_var::ux);
    float uy = k_particles(n, particle_var::uy);
    float uz = k_particles(n, particle_var::uz);
    float q  = k_particles(n, particle_var::w);

    // Interpolate fields
    const field_vectors_t f = interpolate(ii, dx, dy, dz);

    // Half advance E
    borris_advance_e(qdt_2mc, f, ux, uy, uz);

    float v0 = ux*ux + uy*uy + uz*uz;
    v0 = (msp * q) * (v0 / (1 + sqrtf(1 + v0)));
    update += static_cast<double>(v0);

  }, energy);

  return energy;

}

double
species_t::energy( const interpolator_array_t * RESTRICT ia )
{

  if( !ia  )
  {
    ERROR(( "Bad args" ));
  }
  if( g!=ia->g )
  {
    ERROR(( "Bad args" ));
  }

  double local, global;
  float qdt_2mc = (q*g->dt)/(2*m*g->cvac);

  SELECT_GEOMETRY(g->geometry, geo, {

    local = energy_p_kokkos(
      k_p_d,
      k_p_i_d,
      ia->get_device_interpolator<geo>(),
      qdt_2mc,
      m,
      np
    );

  });

  Kokkos::fence();

  mp_allsum_d( &local, &global, 1 );
  return global*(static_cast<double>(g->cvac) * static_cast<double>(g->cvac));

}
