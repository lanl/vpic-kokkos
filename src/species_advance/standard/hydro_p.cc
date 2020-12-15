#define IN_spa
#include "spa_private.h"

template<
  class interp_t,
  class hydro_accum_t
> void
accumulate_hydro_p_kokkos(
  k_particles_t& k_particles,
  k_particles_i_t& k_particles_i,
  interp_t interpolate,
  hydro_accum_t accumulate,
  const float qdt_2mc,
  const float q,
  const float m,
  const int np
)
{

  Kokkos::parallel_for("hydro_p",
    Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np),
    KOKKOS_LAMBDA (size_t p_index)
    {

      // Load position
      float dx = k_particles(p_index, particle_var::dx);
      float dy = k_particles(p_index, particle_var::dy);
      float dz = k_particles(p_index, particle_var::dz);
      int   ii = k_particles_i(p_index);

      // Load momentum
      float ux = k_particles(p_index, particle_var::ux);
      float uy = k_particles(p_index, particle_var::uy);
      float uz = k_particles(p_index, particle_var::uz);
      float w  = k_particles(p_index, particle_var::w);

      // Interpolate fields
      const field_vectors_t f = interpolate(ii, dx, dy, dz);

      // Update momentum
      borris_advance_e(qdt_2mc, f, ux, uy, uz);
      borris_rotate_b(0.5*qdt_2mc, f, ux, uy, uz);

      // Store hydro moments
      accumulate(ii, w, q, m, dx, dy, dz, ux, uy, uz);

    });

}

void
species_t::accumulate_hydro(/**/  hydro_array_t        * RESTRICT ha,
                            const interpolator_array_t * RESTRICT ia)
{


  if( !ia  )
  {
    ERROR(( "Bad args" ));
  }
  if( !ha  )
  {
    ERROR(( "Bad args" ));
  }
  if( g!=ia->g )
  {
    ERROR(( "Bad args" ));
  }
  if( g!=ha->g )
  {
    ERROR(( "Bad args" ));
  }

  float qdt_2mc = (q*g->dt)/(2*m*g->cvac);


  SELECT_GEOMETRY(g->geometry, geo, {

      accumulate_hydro_p_kokkos(
        k_p_d,
        k_p_i_d,
        ia->get_device_interpolator<geo>(),
        ha->get_device_accumulator<geo>(),
        qdt_2mc,
        q,
        m*g->cvac,
        np
      );

  });

}
