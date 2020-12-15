#define IN_spa
#include "spa_private.h"


void
accumulate_rhob( field_t          * RESTRICT ALIGNED(128) f,
                 const particle_t * RESTRICT ALIGNED(32)  p,
                 const grid_t     * RESTRICT              g,
                 const float                              qsp ) {

  // Actual implementation is in rhob.h with the Kokkos version.

  // TODO: Placing geometry selection here might be slow. However,
  // this is only called on initialization and in emitters right
  // now. Moving initialization and emitters to device side would
  // be the optimal solution
  SELECT_GEOMETRY(g->geometry, geo, {

    accumulate_rhob_from_particle(
      f,
      p,
      g,
      qsp,
      g->get_host_geometry<geo>()
    );

  });

}

// accumulate_rho_p adds the charge density associated with the
// supplied particle array to the rhof of the fields.  Trilinear
// interpolation is used.  rhof is known at the nodes at the same time
// as particle positions.  No effort is made to fix up edges of the
// computational domain; see note in synchronize_rhob about why this
// is done this way.  All particles on the list must be inbounds.

template<class geo_type> void
accumulate_rho_p_kokkos(
    k_particles_t& k_particles,
    k_particles_i_t& k_particles_i,
    k_field_t& kfield,
    geo_type geometry,
    const int nx,
    const int ny,
    const int nz,
    const int sx,
    const int sy,
    const int sz,
    const float qsp,
    const int np
)
{

  k_field_sa_t scatter_view = Kokkos::Experimental::create_scatter_view<>(kfield);

  Kokkos::parallel_for("accumulate_rho_p",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, np),
    KOKKOS_LAMBDA(const int n) {

        int ii;
        float rho, dx, dy, dz;

        dx = k_particles(n, particle_var::dx);
        dy = k_particles(n, particle_var::dy);
        dz = k_particles(n, particle_var::dz);
        ii = k_particles_i(n);
        rho = qsp*k_particles(n, particle_var::w)*geometry.inverse_voxel_volume(ii);

        auto weighter = TrilinearWeighting(nx, ny, nz, sx, sy, sz);
        auto access = scatter_view.access();

        weighter.set_position(dx, dy, dz);
        weighter.deposit(
            access,
            ii,
            field_var::rhof,
            rho
        );

    });

  Kokkos::Experimental::contribute(kfield, scatter_view);

}

void
species_t::accumulate_rhof( field_array_t * RESTRICT fa )
{

  if( !fa )
  {
    ERROR(( "Bad args" ));
  }
  if( g!=fa->g )
  {
    ERROR(( "Bad args" ));
  }

  SELECT_GEOMETRY(g->geometry, geo, {

    accumulate_rho_p_kokkos(
      k_p_d,
      k_p_i_d,
      fa->k_f_d,
      g->get_device_geometry<geo>(),
      g->nx,
      g->ny,
      g->nz,
      g->sx,
      g->sy,
      g->sz,
      q,
      np
    );

  });

}

