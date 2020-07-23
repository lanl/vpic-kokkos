#ifndef _kokkos_binary_collision_pipeline_h_
#define _kokkos_binary_collision_pipeline_h_

#include "../collision_private.h"

// Assumes single precision.
// Chosen as a cutoff < sqrt(FLT_MAX) such that dd/(1+dd*dd) is always in range.
#define TAN_THETA_HALF_MAX 1.30e19f
#define PREVENT_BACKSCATTER(TAN) do  {                                          \
  if(!isfinite(TAN) || (TAN) > TAN_THETA_HALF_MAX ) (TAN) = TAN_THETA_HALF_MAX; \
} while(0)

/**
 * @brief General purpose pipeline to produce binary collisions between particles.
 *
 * Within each voxel, there will be max(ni, nj) collisions each time the
 * operator is dispatched. Collision order is deterministic, so if the pipeline
 * is dispatched multiple times, then particles will be shuffled inbetween.
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
 *        angle precision loss issues, however perfect backscattering
 *        cannot occur so tan_theta_half is limited to sqrt(FLT_MAX).
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
 * TODO : CPU-VPIC used a relativistically correct Monte-Carlo test evaluted
 *        in the frame of the scattering particle. The current implementation
 *        is purely classical and does not include relativistic effects. Do
 *        users really want relativistic collsiions?
 *
 * By templating this and using constexpr/consteval, good compilers should be
 * able to skip and disable unused features at compile time.
 *
 * ######################IMPORTANT DOCUMENTATION ##############
 * For information on use of lambdas inside struct and classes:
 *     https://github.com/kokkos/kokkos/wiki/Lambda-Dispatch 
 * ############################################################
 */

using k_density_t=Kokkos::View<float *, Kokkos::DefaultExecutionSpace>;


template <bool MonteCarlo>
struct BinaryCollision {
/**
   * @brief Perform a collision between two particles.
   */
  //Must have all struct member types passed in directly to the inline function
  //In terms of notation, all inline functions should not have variables that start
  //with the _ (underscore), because _ is used to indicate a class member before it
  //is caputred by a lambda. One lambda captured, we should refer to the variable
  //as EX: mu not _mu
  template<class collision_model>
  KOKKOS_INLINE_FUNCTION
  void binary_collision (
    float const& mu,
    float const& mu_i,
    float const& mu_j,
    k_particles_t const&   spi_p,
    k_particles_t const&   spj_p,
    collision_model& model,
    kokkos_rng_state_t& rg,
    float ndt,
    int i,
    int j
  )
  {
    
    float dd, ur, tx, ty, tz, t0, t1, t2, stack[3];
    int d0, d1, d2;

    float uix = spi_p(i, particle_var::ux);
    float uiy = spi_p(i, particle_var::uy);
    float uiz = spi_p(i, particle_var::uz);
    float wi  = spi_p(i, particle_var::w);

    float ujx = spj_p(j, particle_var::ux);
    float ujy = spj_p(j, particle_var::uy);
    float ujz = spj_p(j, particle_var::uz);
    float wj  = spj_p(j, particle_var::w);

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
    t2 *= mu;       // _mu v^2  = Collision energy
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

    // Scaled center of mass velocity.
    t1 = (1-rr);
    float cmx = t1*(mu_j*uix + mu_i*ujx);
    float cmy = t1*(mu_j*uiy + mu_i*ujy);
    float cmz = t1*(mu_j*uiz + mu_i*ujz);

    // Handle unequal particle weights using detailed balance.
    t0 = rg.frand(0, 1);

    if(wi*t0 <= wj) {
      spi_p(i, particle_var::ux) = (uix + mu_i*stack[0])*rr + cmx;
      spi_p(i, particle_var::uy) = (uiy + mu_i*stack[1])*rr + cmy;
      spi_p(i, particle_var::uz) = (uiz + mu_i*stack[2])*rr + cmz;
    }

    if(wj*t0 <= wi) {
      spj_p(j, particle_var::ux) = (ujx - mu_j*stack[0])*rr + cmx;
      spj_p(j, particle_var::uy) = (ujy - mu_j*stack[1])*rr + cmy;
      spj_p(j, particle_var::uz) = (ujz - mu_j*stack[2])*rr + cmz;
    }

  }
};

template<bool MonteCarlo>
struct VoxelParallel : BinaryCollision<MonteCarlo> {
  using BinaryCollision<MonteCarlo>::binary_collision;
  /**
   * @brief Loop over particles performing collisions.
   */
  template<class collision_model>
  void apply_model (collision_model& model, 
    float const& mu_i, 
    float const& mu_j, 
    float const& mu, 
    int const& nx, 
    int const& ny, 
    int const& nz, 
    species_t* const& spi, 
    species_t* const& spj, 
    kokkos_rng_pool_t const& rp,  
    k_particles_i_t spi_i,
    k_density_t const& spi_n, 
    k_density_t const& spj_n, 
    k_particles_t const& spi_p, 
    k_particles_t const& spj_p, 
    float const& dtinterval, 
    k_particle_sortindex_t_ra const& spi_sortindex_ra, 
    k_particle_sortindex_t_ra const& spj_sortindex_ra, 
    k_particle_partition_t_ra const& spi_partition_ra, 
    k_particle_partition_t_ra const& spj_partition_ra) 
  {  
    
    using Space=Kokkos::DefaultExecutionSpace;
    using member_type=Kokkos::TeamPolicy<Space>::member_type;
  
    Kokkos::parallel_for("binary_collision_pipeline::apply_model",
      Kokkos::TeamPolicy<Space>(nx*ny*nz, Kokkos::AUTO()),
      KOKKOS_LAMBDA (member_type team_member) {
        
        int ix, iy, iz;
        RANK_TO_INDEX(team_member.league_rank(), ix, iy, iz, nx, ny, nz);
        const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);

        // Find number of particles for each species.
        auto i0 = spi_partition_ra(v);
        auto ni = spi_partition_ra(v+1) - i0;

        auto j0 = spj_partition_ra(v);
        auto nj = spj_partition_ra(v+1) - j0;

        // TODO: convert this to be a more explicit check on if we have work
        if( ni <= 0 || nj <= 0 ) return; //Nothing to do

        // Find the real densities.
        float density_i = spi_n(v);
        float density_j = spj_n(v);

        // Compute ndt
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

                binary_collision(mu, mu_i, mu_j, spi_p, spj_p, model, rg, 0.5*ndt,
                    spi_sortindex_ra(i0),
                    spi_sortindex_ra(i0 + 1)
                );

                binary_collision(mu, mu_i, mu_j, spi_p, spj_p, model, rg, 0.5*ndt,
                    spi_sortindex_ra(i0),
                    spi_sortindex_ra(i0 + 2)
                );

                binary_collision(mu, mu_i, mu_j, spi_p, spj_p, model, rg, 0.5*ndt,
                    spi_sortindex_ra(i0 + 1),
                    spi_sortindex_ra(i0 + 2)
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

            binary_collision(mu, mu_i, mu_j, spi_p, spj_p, model, rg, ndt,
                spi_sortindex_ra(i0 + (ij ? i : j)),
                spj_sortindex_ra(j0 + (ij ? j : i))
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

            binary_collision(mu, mu_i, mu_j, spi_p, spj_p, model, rg, ndt,
                spi_sortindex_ra(i0 + (ij ? i : j)),
                spj_sortindex_ra(j0 + (ij ? j : i))
            );

          }

        });

        // We *must* free generators.
        rp.free_state(rg);
        
      });
  }
};


/**
   * @brief Loop over particles performing collisions.
   */
template<bool MonteCarlo>
struct ParticleParallel : BinaryCollision<MonteCarlo> {
  using BinaryCollision<MonteCarlo>::binary_collision;
  /**
   * @brief Loop over particles performing collisions.
   */
  template<class collision_model>
  void apply_model (collision_model& model, 
    float const& mu_i, 
    float const& mu_j, 
    float const& mu, 
    int const& nx, 
    int const& ny, 
    int const& nz, 
    species_t* const& spi, 
    species_t* const& spj, 
    kokkos_rng_pool_t const& rp,
    k_particles_i_t spi_i,
    k_density_t const& spi_n, 
    k_density_t const& spj_n, 
    k_particles_t const& spi_p, 
    k_particles_t const& spj_p, 
    float const& dtinterval, 
    k_particle_sortindex_t_ra const& spi_sortindex_ra, 
    k_particle_sortindex_t_ra const& spj_sortindex_ra, 
    k_particle_partition_t_ra const& spi_partition_ra, 
    k_particle_partition_t_ra const& spj_partition_ra) 
  {  
    
    using Space=Kokkos::DefaultExecutionSpace;
    using member_type=Kokkos::TeamPolicy<Space>::member_type;
     
    // To avoid atomics, we can only have at most sum_v min(ni(v), nj(v))
    // independent threads. This is strictly less than ni or nj, so launch
    // one thread per particles and cancel ones we don't need.

    Kokkos::parallel_for("binary_collision_pipeline::apply_model",
      Kokkos::RangePolicy<Space>(0, spi->np),
      KOKKOS_LAMBDA (const int ipart) {

        const int   v    = spi_i(ipart);        // Voxel
        const float iden = spi_n(v);            // Real density
        const float jden = spj_n(v);            // Real density
        const float ndt  = (jden > iden ? iden : jden) * dtinterval;

        // Particle counts for each species in this voxel.
        int i0 = spi_partition_ra(v);           // Voxel start index
        int ni = spi_partition_ra(v+1) - i0;    // Number of particles

        int j0 = spj_partition_ra(v);           // Voxel start index
        int nj = spj_partition_ra(v+1) - j0;    // Number of particles

        const int idx = ipart - i0;             // Intracell index

        // Handle intraspecies collisions.
        if( spi == spj ) {

            // Odd number of patticles.
            if( ni%2 && ni >= 3) {

              // These must be done serially to avoid atomics
              if( idx == 0 ) {

                // Get a random generator.
                kokkos_rng_state_t rg = rp.get_state();

               binary_collision(mu, mu_i, mu_j, spi_p, spj_p, model, rg, 0.5*ndt,
                    spi_sortindex_ra(i0),
                    spi_sortindex_ra(i0 + 1)
                );

                binary_collision(mu, mu_i, mu_j, spi_p, spj_p, model, rg, 0.5*ndt,
                    spi_sortindex_ra(i0),
                    spi_sortindex_ra(i0 + 1)
                );

                binary_collision(mu, mu_i, mu_j, spi_p, spj_p, model, rg, 0.5*ndt,
                    spi_sortindex_ra(i0),
                    spi_sortindex_ra(i0 + 1)
                );


                // Free the generator.
                rp.free_state(rg);

              }

              ni -= 3;
              i0 += 3;

            }

            // Even number of particles.
            nj = ni = ni/2;
            j0 = i0 + ni;

        }

        const bool ij  = ni > nj;
        const int nmax = ij ? ni : nj;
        const int nmin = ij ? nj : ni;

        // Nothing to do.
        if( idx >= nmin ) return;

        // Calculate collisional pairings.
        //  w.l.o.g. let ni < nj
        //  then:
        //    Let nj = ni*ncoll + remain
        //    The first remain particles from spi will collide ncoll+1 times
        //    The following (ni-remain) particles will collide ncoll times.

        int startidx;
        int ncoll  = nmin <= 0 ? 0 : nmax/nmin;
        int remain = nmin <= 0 ? 0 : nmax - ncoll*nmin;

        // If we're in the remianing list, increment the number of collisions.
        if( idx < remain )
        {
          ncoll += 1;
          startidx = idx*ncoll;
        }
        else
        {
          startidx = remain*(ncoll+1) + (idx-remain)*ncoll;
        }

        // Branchless looping.
        // if (ij) then we will collide particle j with ncoll i
        // else         we will collide particle i with ncoll j

        int i = i0 + (ij ? startidx :      idx);
        int j = j0 + (ij ?      idx : startidx);

        // Get a random generator.
        kokkos_rng_state_t rg = rp.get_state();

        for(int k=0 ; k < ncoll ; ++k) {

          binary_collision(mu, mu_i, mu_j, spi_p, spj_p, model, rg, 0.5*ndt,
                    spi_sortindex_ra(i0),
                    spi_sortindex_ra(i0 + 1)
          );

          i += ij ? 1 : 0 ;
          j += ij ? 0 : 1 ;

        }

        // Free the generator.
        rp.free_state(rg);

      });
  }
};



template<bool MonteCarlo = false, template <bool> class ParallelPolicy = VoxelParallel>
struct CollisionParallelismModel : ParallelPolicy<MonteCarlo> {
    using ParallelPolicy<MonteCarlo>::apply_model;
};

template<bool MonteCarlo = false, template <bool> class ParallelPolicy = VoxelParallel>
struct binary_collision_pipeline {


  const float _mu_i, _mu_j, _mu, _dtinterval, _rdV;
  const int   _nx, _ny, _nz;

  //Member variables start with the symbol _ and this is used 
  //to indicate they are not safe to be passed into a kokkos
  //lambda without first changing the reference type. Any var
  //starting with _ in a lambda will likely throw and illegal
  //memory error. As convention we reccomend not using _varName
  //in an inline fucntion but rather just varName
  species_t *_spi, *_spj;
  kokkos_rng_pool_t& _rp;
  k_density_t     _spi_n,  _spj_n;
  k_particles_t   _spi_p,  _spj_p;
  k_particles_i_t _spi_i,  _spj_i;

  // Random access, read-only Views
  // TODO : Does RandomAccess trait really matter?
  k_particle_sortindex_t_ra _spi_sortindex_ra, _spj_sortindex_ra;
  k_particle_partition_t_ra _spi_partition_ra, _spj_partition_ra;

  CollisionParallelismModel<MonteCarlo, ParallelPolicy> pm;

  binary_collision_pipeline(
    species_t * spi,
    species_t * spj,
    double interval,
    kokkos_rng_pool_t& rp
  )
    : _spi(spi),
      _spj(spj),
      _rp(rp),
      _mu_i(spj->m / (spi->m + spj->m)),
      _mu_j(spi->m / (spi->m + spj->m)),
      _mu(spi->m*spj->m / (spi->m + spj->m)),
      _dtinterval(spi->g->dt * interval),
      _nx(spi->g->nx),
      _ny(spi->g->ny),
      _nz(spi->g->nz),
      _rdV(1/spi->g->dV)
  {
    //TODO: is interval needed here? 
    if( !_spi || !_spj || !_spi->g || !_spj->g || _spi->g != _spj->g || interval <= 0)
      ERROR(("Bad args."));

  }

  /**
   * @brief Dispatch a collision model on this pipeline.
   *
   * Each dispatch will test each particle for collision at least once.
   */
  template<class collision_model>
  void dispatch(
    collision_model& _model
  )
  {

    ParticleSorter<BinSort> sorter;
    ParticleShuffler<> shuffler;

    // Ensure sorted and shuffled.
    if( _spi->last_indexed != _spi->g->step ) {
      sorter.sort( _spi, false );
    }

    if( _spj->last_indexed != _spj->g->step ) {
      sorter.sort( _spj, false );
    }

    // Always reload in case Views were invalidated.
    _spi_p            = _spi->k_p_d;
    _spi_i            = _spi->k_p_i_d;
    _spi_partition_ra = _spi->k_partition_d;
    _spi_sortindex_ra = _spi->k_sortindex_d;

    _spj_p            = _spj->k_p_d;
    _spj_i            = _spj->k_p_i_d;
    _spj_partition_ra = _spj->k_partition_d;
    _spj_sortindex_ra = _spj->k_sortindex_d;

    // Am I being paranoid?
    if( _spi->np      != _spi_sortindex_ra.extent(0) ||
        _spi->g->nv+1 != _spi_partition_ra.extent(0) )
        ERROR(("Bad spi sort products."));

    if( _spj->np      != _spj_sortindex_ra.extent(0) ||
        _spj->g->nv+1 != _spj_partition_ra.extent(0) )
        ERROR(("Bad spj sort products."));

    // We only need to shuffle one species to ensure random pairings.
    shuffler.shuffle( _spi, _rp, false );

    // TODO: Move this out of dispatch so we can dispatch multiple models
    //       without recomputing the density. Kokkos won't let me put it in
    //       the constructor.

    // Compute species densities using a simple histogram. Batching these
    // beforehand is much faster than doing it inline.
    _spi_n = k_density_t("spi_n", _spi->g->nv);
    _spj_n = k_density_t("spj_n", _spj->g->nv);
   
    //Not sure if this is needed to be redefined here
    const float rdV = (1/_spi->g->dV);

    // NOTE: workaround to avoid implicit capture of this
    // SEE:  kokkos lambda dispatch link at top
    auto const& spi_n = _spi_n;
    auto const& spi_p = _spi_p;
    auto const& spi_i = _spi_i;
    Kokkos::parallel_for("binary_collision_pipeline::spi_denisty",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, _spi->np),
      KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_add(
          &spi_n(spi_i(i)),
          spi_p(i, particle_var::w)*rdV
        );
    });
    
    // NOTE: workaround to avoid implicit capture of this
    // SEE:  kokkos lambda dispatch link at top
    auto const& spj_n = _spj_n;
    auto const& spj_p = _spj_p;
    auto const& spj_i = _spj_i;

    if( _spi != _spj ) {
        Kokkos::parallel_for("binary_collision_pipeline::spj_denisty",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, _spj->np),
        KOKKOS_LAMBDA (int i) {
          Kokkos::atomic_add(
            &spj_n(spj_i(i)),
            spj_p(i, particle_var::w)*rdV
          );
        });
    }
    else {
      _spj_n = _spi_n;
    }
    
    // NOTE: workaround to avoid implicit capture of this
    // SEE:  kokkos lambda dispatch link at top
    auto const& model = _model;
    auto const& mu_i = _mu_i;
    auto const& mu_j = _mu_j;
    auto const& mu = _mu;
    auto const& nx = _nx;
    auto const& ny = _ny;
    auto const& nz = _nz;
    auto const& spi = _spi;
    auto const& spj = _spj;
    auto const& rp  = _rp;
    auto const& dtinterval = _dtinterval;
    auto const& spi_sortindex_ra = _spi_sortindex_ra; 
    auto const& spj_sortindex_ra = _spj_sortindex_ra; 
    auto const& spi_partition_ra = _spi_partition_ra;
    auto const& spj_partition_ra = _spj_partition_ra; 
    


    // Do collisions.
    pm.apply_model(model, mu_i, mu_j, mu, nx, ny, nz, spi, spj, rp,
                spi_i, spi_n, spj_n, spi_p, spj_p, dtinterval, 
                spi_sortindex_ra, spj_sortindex_ra,
                spi_partition_ra, spj_partition_ra);

  }
};

#endif
