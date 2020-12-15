// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa
#include "spa_private.h"

template<
  class geo_t,
  class interp_t,
  class accum_t
> void
advance_p_kokkos(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_counter_t& k_nm,
        k_neighbor_t& k_neighbors,
        geo_t geometry,
        interp_t interpolate,
        accum_t accumulate,
        const float qdt_2mc,
        const float cdt_2,
        const float qsp,
        const int np,
        const int max_nm,
        const int64_t rangel,
        const int64_t rangeh,
        const int nx,
        const int ny,
        const int nz)
{

  // zero out nm, we could probably do this earlier if we're worried about it
  // slowing things down
  Kokkos::deep_copy(k_nm, 0);

  Kokkos::parallel_for("advance_p",
    Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np),
    KOKKOS_LAMBDA (size_t p_index) {

      float v0, v1, v2, v3, v4, v5, dispx, dispy, dispz;

      // Load position
      float dx = k_particles(p_index, particle_var::dx);
      float dy = k_particles(p_index, particle_var::dy);
      float dz = k_particles(p_index, particle_var::dz);
      int   ii = k_particles_i(p_index);

      // Load momentum
      float ux = k_particles(p_index, particle_var::ux);
      float uy = k_particles(p_index, particle_var::uy);
      float uz = k_particles(p_index, particle_var::uz);
      float q  = k_particles(p_index, particle_var::w);

      // Interpolate fields
      const field_vectors_t f = interpolate(ii, dx, dy, dz);

      // Update momentum
      borris_advance_e(qdt_2mc, f, ux, uy, uz);
      borris_rotate_b(qdt_2mc, f, ux, uy, uz);
      borris_advance_e(qdt_2mc, f, ux, uy, uz);

      /**/                                      // Get norm displacement
      v0   = cdt_2/sqrtf(1.0f + (ux*ux+ (uy*uy + uz*uz)));

      dispx = ux*v0;
      dispy = uy*v0;
      dispz = uz*v0;

      geometry.convert_to_logical(ii, dx, dy, dz, dispx, dispy, dispz, ux, uy, uz);

      // Store momentum
      k_particles(p_index, particle_var::ux) = ux;
      k_particles(p_index, particle_var::uy) = uy;
      k_particles(p_index, particle_var::uz) = uz;

      v0   = dx + dispx;                           // Streak midpoint (inbnds)
      v1   = dy + dispx;
      v2   = dz + dispx;

      v3   = v0 + dispx;                           // New position
      v4   = v1 + dispx;
      v5   = v2 + dispx;

      // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
      if(  v3<=1.0f &&  v4<=1.0f &&  v5<=1.0f &&   // Check if inbnds
          -v3<=1.0f && -v4<=1.0f && -v5<=1.0f ) {

        // Common case (inbnds).

        // Store new position
        k_particles(p_index, particle_var::dx) = v3;
        k_particles(p_index, particle_var::dy) = v4;
        k_particles(p_index, particle_var::dz) = v5;

        // Accumulate current
        accumulate(ii, q*qsp, v0, v1, v2, ux, uy, uz);

      } else
      {
        // Unlikely
        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
        local_pm->dispx = dispx;
        local_pm->dispy = dispy;
        local_pm->dispz = dispz;
        local_pm->i     = p_index;

        if( move_p_kokkos( k_particles, k_particles_i, local_pm,
                          accumulate, k_neighbors, rangel, rangeh, qsp ) )
        {
          // Unlikely
          if( k_nm(0) < max_nm )
          {
              const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
              if (nm >= max_nm) Kokkos::abort("overran max_nm");

              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
              k_particle_movers_i(nm)   = local_pm->i;

              // Keep existing mover structure, but also copy the particle
              // data so we have a reduced set to move to host
              k_particle_copy(nm, particle_var::dx) = k_particles(p_index, particle_var::dx);
              k_particle_copy(nm, particle_var::dy) = k_particles(p_index, particle_var::dy);
              k_particle_copy(nm, particle_var::dz) = k_particles(p_index, particle_var::dz);
              k_particle_copy(nm, particle_var::ux) = k_particles(p_index, particle_var::ux);
              k_particle_copy(nm, particle_var::uy) = k_particles(p_index, particle_var::uy);
              k_particle_copy(nm, particle_var::uz) = k_particles(p_index, particle_var::uz);
              k_particle_copy(nm, particle_var::w)  = k_particles(p_index, particle_var::w);
              k_particle_i_copy(nm) = k_particles_i(p_index);
          }
        }
      }
    });

}

void
species_t::advance( accumulator_array_t  * RESTRICT aa,
                    interpolator_array_t * RESTRICT ia )
{

  if( !g )
  {
    ERROR(( "Bad args" ));
  }
  if( !aa )
  {
    ERROR(( "Bad args" ));
  }
  if( !ia  )
  {
    ERROR(( "Bad args" ));
  }
  if( g!=aa->g )
  {
    ERROR(( "Bad args" ));
  }
  if( g!=ia->g )
  {
    ERROR(( "Bad args" ));
  }

  float qdt_2mc = (q*g->dt)/(2*m*g->cvac);
  float cdt_2   = 0.5*g->cvac*g->dt;

  KOKKOS_TIC();

  SELECT_GEOMETRY(g->geometry, geo, {

      advance_p_kokkos(
        k_p_d,
        k_p_i_d,
        k_pc_d,
        k_pc_i_d,
        k_pm_d,
        k_pm_i_d,
        k_nm_d,
        g->k_neighbor_d,
        g->get_device_geometry<geo>(),
        ia->get_device_interpolator<geo>(),
        aa->get_device_accumulator(),
        qdt_2mc,
        cdt_2,
        q,
        np,
        max_nm,
        g->rangel,
        g->rangeh,
        g->nx,
        g->ny,
        g->nz
      );

  });

  KOKKOS_TOC( advance_p, 1);

  KOKKOS_TIC();

  // I need to know the number of movers that got populated so I can call the
  // compress. Let's copy it back
  Kokkos::deep_copy(k_nm_h, k_nm_d);
  // TODO: which way round should this copy be?

  // Copy particle mirror movers back so we have their data safe. Ready for
  // boundary_p_kokkos
  auto pc_d_subview = Kokkos::subview(k_pc_d, std::make_pair(0, k_nm_h(0)), Kokkos::ALL);
  auto pci_d_subview = Kokkos::subview(k_pc_i_d, std::make_pair(0, k_nm_h(0)));
  auto pc_h_subview = Kokkos::subview(k_pc_h, std::make_pair(0, k_nm_h(0)), Kokkos::ALL);
  auto pci_h_subview = Kokkos::subview(k_pc_i_h, std::make_pair(0, k_nm_h(0)));

  Kokkos::deep_copy(pc_h_subview, pc_d_subview);
  Kokkos::deep_copy(pci_h_subview, pci_d_subview);

  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);

}
