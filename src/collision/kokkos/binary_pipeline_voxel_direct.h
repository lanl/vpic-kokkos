#ifndef _kokkos_binary_collision_pipeline_h_
#define _kokkos_binary_collision_pipeline_h_

#include "../collision_private.h"

// Assumes single precision.
#define TAN_THETA_HALF_MAX 1.30e19f
#define PREVENT_BACKSCATTER(TAN) do  {                                          \
  if(!isfinite(TAN) || (TAN) > TAN_THETA_HALF_MAX ) (TAN) = TAN_THETA_HALF_MAX; \
} while(0)

/**
 * @brief General purpose pipeline to produce binary collisions between particles.
 *
 * Within each voxel, there will be max(ni, nj) collisions each time the
 * operator is dispatched. Collision order is deterministic, so if the pipeline
 * is dispatched multiple times, then particles should be shuffled inbetween.
 *
 * This differes from CPU-VPIC in that each particle will collide at most once
 * during each dispatch. This avoids requiring locks on particles and improves
 * performance.
 *
 * The collision properties are defined by the given operator. Each
 * operator should implement the following methods:
 *
 *    tan_theta_half(rg, E, nvdt)
 *        Computes tan(theta/2) where theta is the polar scattering angle.
 *        We use tan(theta/2) instead of theta, sin, or cos to avoid small
 *        angle precision loss issues.
 *
 *    restitution(rg, E, nvdt)
 *        Computes the coefficient of restitution for inelastic scattering,
 *        0 <= R <= 1. For elastic scattering, R = 1.
 *
 * If MonteCarlo is true, then each collision will be randomly tested to occur.
 * In this case, the operator should also define a cross_section method.
 *
 *    cross_section(rg, E, nvdt)
 *        Returns the cross-section for the collision in normalized units.
 *        The collision will occur with probability cross_section*nvdt.
 *
 * By templating this and using constexpr/consteval, good compilers should be
 * able to skip and disable unused features at compile time.
 */
template<bool MonteCarlo>
struct binary_collision_pipeline {

  using Space=Kokkos::DefaultExecutionSpace;
  using member_type=Kokkos::TeamPolicy<Space>::member_type;
  using k_density_t=Kokkos::View<float *, Space>;

  const float mi, mj, dtinterval;
  const int   nx, ny, nz;

  species_t *spi, *spj;
  kokkos_rng_pool_t& rp;
  k_density_t     spi_n,  spj_n;
  k_particles_t   spi_p,  spj_p;
  k_particles_i_t spi_i,  spj_i;

  // Random access, read-only Views
  k_particle_partition_t_ra spi_partition_ra, spj_partition_ra;

  binary_collision_pipeline(
    species_t * spi,
    species_t * spj,
    double interval,
    kokkos_rng_pool_t& rp
  )
    : spi(spi),
      spj(spj),
      rp(rp),
      mi(spi->m),
      mj(spj->m),
      dtinterval(spi->g->dt * interval),
      nx(spi->g->nx),
      ny(spi->g->ny),
      nz(spi->g->nz)
  {

    if( !spi || !spj || !spi->g || !spj->g || spi->g != spj->g || interval <= 0)
      ERROR(("Bad args."));

  }

  /**
   * @brief Dispatch a collision model on this pipeline.
   *
   * Each dispatch will test each particle for collision at least once.
   */
  template<class collision_model>
  void dispatch(
    collision_model& model
  )
  {

    ParticleSorter<> sorter;
    ParticleShuffler<> shuffler;

    // Ensure sorted and shuffled.
    if( spi->last_sorted != spi->g->step ) {
      sorter.sort( spi, true, false );
    }

    if( spj->last_sorted != spj->g->step ) {
      sorter.sort( spj, true, false );
    }

    // Always reload in case Views were invalidated.
    spi_p = spi->k_p_d;
    spj_p = spj->k_p_d;
    spi_i = spi->k_p_i_d;
    spj_i = spj->k_p_i_d;
    spi_partition_ra = spi->k_partition_d;
    spj_partition_ra = spj->k_partition_d;

    // Am I being paranoid?
    if( spi->g->nv+1 != spi_partition_ra.extent(0) )
        ERROR(("Bad spi sort products."));

    if( spj->g->nv+1 != spj_partition_ra.extent(0) )
        ERROR(("Bad spj sort products."));

    // We only need to shuffle one species to ensure random pairings.
    shuffler.shuffle( spi, rp, true, false );

    // TODO: Move this out of dispatch so we can dispatch multiple models
    //       without recomputing the density. Kokkos won't let me put it in
    //       the constructor.

    // Compute species densities using a simple histogram.
    spi_n = k_density_t("spi_n", spi->g->nv);
    spj_n = k_density_t("spj_n", spj->g->nv);

    const float rdV = 1/spi->g->dV;
    Kokkos::parallel_for("binary_collision_pipeline::spi_denisty",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, spi->np),
      KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_add(
          &spi_n(spi_i(i)),
          spi_p(i, particle_var::w)*rdV
        );
      });

    if( spi != spj ) {
      Kokkos::parallel_for("binary_collision_pipeline::spj_denisty",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, spj->np),
        KOKKOS_LAMBDA (int i) {
          Kokkos::atomic_add(
            &spj_n(spj_i(i)),
            spj_p(i, particle_var::w)*rdV
          );
        });
    }
    else {
      spj_n = spi_n;
    }

    // Do collisions.
    apply_model(model);

  }

  /**
   * @brief Loop over particles performing collisions.
   */
  template<class collision_model>
  void apply_model (
    collision_model& model
  )
  {

    Kokkos::parallel_for("binary_collision_pipeline::apply_model",
      Kokkos::TeamPolicy<Space>(nx*ny*nz, Kokkos::AUTO()),
      KOKKOS_LAMBDA (member_type team_member) {

        int ix, iy, iz;
        RANK_TO_INDEX(team_member.league_rank(), ix, iy, iz, nx, ny, nz);
        const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);

        // Find number of particles for each species.
        int i0 = spi_partition_ra(v);
        int ni = spi_partition_ra(v+1) - i0;

        int j0 = spj_partition_ra(v);
        int nj = spj_partition_ra(v+1) - j0;

        // TODO: convert this to be a more explicit check on if we have work
        if( ni <= 0 || nj <= 0 ) return; /* Nothing to do */

        // Find the real densities and compute ndt.
        const float density_i = spi_n(v);
        const float density_j = spj_n(v);
        const float density_min = density_j > density_i ? density_i : density_j;
        const float ndt = density_min*dtinterval;

        // Get a random generator. Do not leave without freeing it.
        kokkos_rng_state_t rg = rp.get_state();

        // Handle intraspecies.
        if( spi == spj ) {

            // Odd number of particles.
            if( ni%2 && ni >= 3 ) {

              Kokkos::single( Kokkos::PerTeam(team_member),
              [&]() {

                // These must be done serially to avoid atomics (same particles)

                binary_collision( model, rg, 0.5*ndt,
                    (i0),
                    (i0 + 1)
                );

                binary_collision( model, rg, 0.5*ndt,
                    (i0),
                    (i0 + 2)
                );

                binary_collision( model, rg, 0.5*ndt,
                    (i0 + 1),
                    (i0 + 2)
                );

              });

              // FIXME: is this really needed?
              team_member.team_barrier();

              ni -= 3;
              i0 += 3;

            }

            // Even number of particles.
            nj = ni = ni/2;
            j0 = i0 + ni;

        }

        // Compute collisional pairings.
        const bool ij    = ni > nj;
        const int nmax   = ij ? ni : nj;
        const int nmin   = ij ? nj : ni;
        const int ncoll  = nmin <= 0 ? 0 : nmax/nmin;
        const int remain = nmin <= 0 ? 0 : nmax - ncoll*nmin;

        // FIXME: Will combining these loops into one improve performance?

        // The first remain particles of species min will collide ncoll+1 times
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, remain),
        [&](const int& k) {

          // Inner loop must be serialized to prevent atomics.
          for(int l=0 ; l < ncoll+1 ; ++l) {

            int i = l + k*(ncoll+1) ;
            int j = k ;

            binary_collision( model, rg, ndt,
                (i0 + (ij ? i : j)),
                (j0 + (ij ? j : i))
            );

          }

        });

        // The bulk (nmin-remain) particles of species min will collide ncoll times
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nmin-remain),
        [&](const int& k) {

          // Inner loop must be serialized to prevent atomics.
          for(int l=0 ; l < ncoll ; ++l) {

            int i = l + remain*(ncoll+1) ;
            int j = k + remain ;

            binary_collision( model, rg, ndt,
                (i0 + (ij ? i : j)),
                (j0 + (ij ? j : i))
            );

          }

        });

        // We *must* free generators.
        rp.free_state(rg);

      });

  }

  /**
   * @brief Perform a collision between two particles.
   */
  template<class collision_model>
  KOKKOS_INLINE_FUNCTION
  void binary_collision (
    collision_model& model,
    kokkos_rng_state_t& rg,
    float ndt,
    int i,
    int j
  )
  {

    float dd, ur, tx, ty, tz, t0, t1, t2, stack[3];
    int d0, d1, d2;

    const float mu_i = mj / (mi + mj);
    const float mu_j = mi / (mi + mj);
    const float mu   = mu_i*mu_j;

    float uix = spi_p(i, particle_var::ux);
    float uiy = spi_p(i, particle_var::uy);
    float uiz = spi_p(i, particle_var::uz);
    float wi  = spi_p(i, particle_var::w);

    float ujx = spj_p(j, particle_var::ux);
    float ujy = spj_p(j, particle_var::uy);
    float ujz = spj_p(j, particle_var::uz);
    float wj  = spj_p(j, particle_var::w);

    // Center-of-mass velocity
    float cmx = mu_j*uix + mu_i*ujx;
    float cmy = mu_j*uiy + mu_i*ujy;
    float cmz = mu_j*uiz + mu_i*ujz;

    // Relative velocity
    float urx = uix - ujx;
    float ury = uiy - ujy;
    float urz = uiz - ujz;

    /* There are lots of ways to formulate T vector formation    */
    /* This has no branches (but uses L1 heavily)                */

    t0 = urx*urx;
    d0=0;
    d1=1;
    d2=2;
    t1=t0;
    t2=t0;

    t0 = ury*ury;
    if (t0 < t1)
    {
        d0 = 1;
        d1 = 2;
        d2 = 0;
        t1 = t0;
    }
    t2 += t0;

    t0 = urz*urz;
    if (t0 < t1)
    {
        d0 = 2;
        d1 = 0;
        d2 = 1;
    }
    t2 += t0;

    ur = sqrtf( t2 );

    // Collision parameters
    t2 *= mu;       // mu v^2  = Collision energy
    t1  = ur*ndt;   // n v dt  = Particles encountered per unit area

    // Monte-Carlo collision test
    if( MonteCarlo ) {

      // TODO : CPU VPIC warned when dd*t1 > 1 for under-resolved collisions.
      //        Would this be useful?
      dd = model.cross_section(rg, t2, t1);
      if( rg.frand() > dd*t1 ) return;

    }

    // Compute collision angle and coefficient of restitution
    const float rr = model.restitution(rg, t2, t1);
    dd = model.tan_theta_half(rg, t2, t1);
    PREVENT_BACKSCATTER(dd);

    stack[0] = urx;
    stack[1] = ury;
    stack[2] = urz;
    t1  = stack[d1];
    t2  = stack[d2];
    t0  = 1 / sqrtf( t1*t1 + t2*t2 + FLT_MIN );
    stack[d0] =  0;
    stack[d1] =  t0*t2;
    stack[d2] = -t0*t1;
    tx = stack[0];
    ty = stack[1];
    tz = stack[2];

    // Convert tan(theta/2) to sin/cos.
    t0 = 2*dd/(1+dd*dd);

    // Azimuthal angle is random.
    t1 = rg.frand(0, 2*M_PI);
    t2 = t0*sinf(t1);
    t1 = t0*ur*cosf(t1);
    t0 *= -dd;

    /* stack = (1 - cos theta) u + |u| sin theta Tperp */
    stack[0] = (t0*urx + t1*tx) + t2*( ury*tz - urz*ty );
    stack[1] = (t0*ury + t1*ty) + t2*( urz*tx - urx*tz );
    stack[2] = (t0*urz + t1*tz) + t2*( urx*ty - ury*tx );

    // Handle unequal particle weights using detailed balance.
    t0 = rg.frand(0, 1);

    if(wi*t0 <= wj) {
      spi_p(i, particle_var::ux) = (uix + mu_i*stack[0])*rr + (1-rr)*cmx;
      spi_p(i, particle_var::uy) = (uiy + mu_i*stack[1])*rr + (1-rr)*cmy;
      spi_p(i, particle_var::uz) = (uiz + mu_i*stack[2])*rr + (1-rr)*cmz;
    }

    if(wj*t0 <= wi) {
      spj_p(j, particle_var::ux) = (ujx - mu_j*stack[0])*rr + (1-rr)*cmx;
      spj_p(j, particle_var::uy) = (ujy - mu_j*stack[1])*rr + (1-rr)*cmy;
      spj_p(j, particle_var::uz) = (ujz - mu_j*stack[2])*rr + (1-rr)*cmz;
    }

  }

};

#endif
