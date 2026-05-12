#ifndef _kokkos_particle_bulk_collision_pipeline_h_
#define _kokkos_particle_bulk_collision_pipeline_h_

#include "../collision_private.h"

// Assumes single precision.
// Chosen as a cutoff < sqrt(FLT_MAX) such that dd/(1+dd*dd) is always in range.
KOKKOS_INLINE_FUNCTION
void PREVENT_BACKSCATTER(float& TAN) {
  constexpr float TAN_THETA_HALF_MAX = 1.30e19f;
  if(!Kokkos::isfinite(TAN) || (TAN > TAN_THETA_HALF_MAX) )
    TAN = TAN_THETA_HALF_MAX;
}
 
/**
 * @brief General purpose pipeline to produce particle-fluid bulk collisions.
 *
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
template<bool MonteCarlo>
struct particle_bulk_collision_pipeline {

  using Space=Kokkos::DefaultExecutionSpace;
  using member_type=Kokkos::TeamPolicy<Space>::member_type;
  using k_density_t=Kokkos::View<float *, Space>;

  const float _mi, _mj, _mu_i, _mu_j, _mu, _dtinterval, _rdV;
  const int   _nx, _ny, _nz;

  //Member variables start with the symbol _ and this is used
  //to indicate they are not safe to be passed into a kokkos
  //lambda without first changing the reference type. Any var
  //starting with _ in a lambda will likely throw and illegal
  //memory error. As convention we reccomend not using _varName
  //in an inline fucntion but rather just varName
  species_t *_spi;
  kokkos_rng_pool_t& _rp;
  k_density_t     _spi_n;//,  _spj_n;
  k_particles_t   _spi_p;//,  _spj_p;
  k_particles_i_t _spi_i;//,  _spj_i;

  fluid_species_t *_spj;
  k_fluid_t _spj_fl;
  field_array_t *_field; //for electron-ion collisions
  k_field_t _spj_fd; 
  bool _use_e_field;
    
  species_t *_spp;
  k_particles_t _spp_p;
  k_particles_i_t *_spp_i;

  // Random access, read-only Views
  // TODO : Does RandomAccess trait really matter?
  k_particle_sortindex_t_ra _spi_sortindex_ra;//, _spj_sortindex_ra;
  k_particle_partition_t_ra _spi_partition_ra;//, _spj_partition_ra;

  particle_bulk_collision_pipeline(
    species_t * spi,
    fluid_species_t * spj,
    double interval,
    kokkos_rng_pool_t& rp,
    field_array_t * field,
    species_t * spp=NULL
  )
    : _mi( spi->m ),
      _mj( spj->m ),
      _mu_i(spj->m / (spi->m + spj->m)),
      _mu_j(spi->m / (spi->m + spj->m)),
      _mu(spi->m*spj->m / (spi->m + spj->m)),
      _dtinterval(spi->g->dt * interval),
      _rdV(1/spi->g->dV),
      _nx(spi->g->nx),
      _ny(spi->g->ny),
      _nz(spi->g->nz),
      _spi(spi),
      _rp(rp),
      _spj(spj),
      _field(field),
      _spp(spp)
  {
    //TODO: is interval needed here?
    if( !_spi || !_spj || !_spi->g || !_spj->g || _spi->g != _spj->g || interval <= 0)
      ERROR(("Bad args."));
    if(_field==NULL) _use_e_field = false;
    else _use_e_field = true;
    // printf("use_e_field=%d\n",_use_e_field);
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
    k_ParticleSorter<BinSort> sorter;
      //k_ParticleSorter<> sorter;      
    ParticleShuffler<> shuffler;

    // Ensure sorted and shuffled.
    if( _spi->last_indexed != _spi->g->step ) {
      sorter.sort( _spi, false );
    }

     // Always reload in case Views were invalidated.
    _spi_p            = _spi->k_p_d;
    _spi_i            = _spi->k_p_i_d;
    _spi_partition_ra = _spi->k_partition_d;
    _spi_sortindex_ra = _spi->k_sortindex_d;

    // TO-DO: NEED TO DO THIS FOR FLUID?
    _spj_fl           = _spj->k_fl_d;
    if(_use_e_field) _spj_fd = _field->k_f_d;
    // else
    //printf("Pointer _field: %p, %p, %p, _use_e_field=%d\n", _field, (void*)&_field->k_f_d, _spj_fd,_use_e_field);
    //    _spj_p            = _spj->k_p_d;
    //    _spj_i            = _spj->k_p_i_d;
    //    _spj_partition_ra = _spj->k_partition_d;
    //    _spj_sortindex_ra = _spj->k_sortindex_d;

    // Am I being paranoid?
    if( static_cast<size_t>(_spi->np)      > _spi_sortindex_ra.extent(0) || 
        static_cast<size_t>(_spi->g->nv)+1 != _spi_partition_ra.extent(0) ){
      printf("_spi->np (=%zu) ?= _spi_sortindex_ra.extent(0) (=%lu)\n",_spi->np,_spi_sortindex_ra.extent(0));
      printf("_spi->g->nv+1 (=%d) ?= _spi_partition_ra.extent(0) (=%lu)\n",_spi->g->nv+1,_spi_partition_ra.extent(0));
      ERROR(("Bad spi sort products."));
    }

    // We only need to shuffle one species to ensure random pairings.
    shuffler.shuffle( _spi, _rp, false );

    // TODO: Not sure if need to compute density from particles (target density comes from fluid).
    // TODO: Move this out of dispatch so we can dispatch multiple models
    //       without recomputing the density. Kokkos won't let me put it in
    //       the constructor.

    /*
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

    if( _spi != _spj ) {
        // NOTE: workaround to avoid implicit capture of this
        // SEE:  kokkos lambda dispatch link at top
        auto const& spj_n = _spj_n;
        auto const& spj_p = _spj_p;
        auto const& spj_i = _spj_i;
      Kokkos::parallel_for("binary_collision_pipeline::spj_denisty",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, _spj->np),
        KOKKOS_LAMBDA (size_t i) {
          Kokkos::atomic_add(
            &spj_n(spj_i(i)),
            spj_p(i, particle_var::w)*rdV
          );
        });
    }
    else {
      _spj_n = _spi_n;
    }

    */
    
    /* todo: specialize pipeline on whether products are created
      (cannot partially speciallize templated function apply_model..)
      
      template<class collision_model, bool create_products>
      void apply_model ( collision_model& _model ) {}

      template<class collision_model>
      void apply_model<collision_model, false> ( collision_model& _model) {}

      template<class collision_model>
      void apply_model<collision_model, true> ( collision_model& _model) {}

    */

    if (_spp == NULL) {
      apply_model(_model);
    } else {
      _spp_p = _spp->k_p_d;
      _spp_i = &_spp->k_p_i_d;
      apply_model_products(_model);
    }
  }

  /**
   * @brief Loop over particles performing collisions.
   */


  template<class collision_model>
  void apply_model (
    collision_model& _model
  )
  {
    // NOTE: workaround to avoid implicit capture of this
    // SEE:  kokkos lambda dispatch link at top
    auto const& model = _model;
    auto const& mi   = _mi;
    auto const& mj   = _mj;
    auto const& mu_i = _mu_i;
    auto const& mu_j = _mu_j;
    auto const& mu = _mu;
    auto const& rdV = _rdV;
    auto const& nx = _nx;
    auto const& ny = _ny;
    auto const& nz = _nz;
    //auto const& spi = _spi;
    //auto const& spj = _spj;
    auto const& rp  = _rp;
    //auto const& spi_n = _spi_n;
    //auto const& spi_i = _spi_i;
    //    auto const& spj_n = _spj_n;
    auto const& spi_p = _spi_p;
    auto const& spj_fl = _spj_fl;
    auto const& spj_fd = _spj_fd;
    //    auto const& spj_p = _spj_p;
    auto const& dtinterval = _dtinterval;
    auto const& spi_sortindex_ra = _spi_sortindex_ra;
    //    auto const& spj_sortindex_ra = _spj_sortindex_ra;
    auto const& spi_partition_ra = _spi_partition_ra;
    //    auto const& spj_partition_ra = _spj_partition_ra;
    auto const& use_e_field = _use_e_field;

    Kokkos::parallel_for("particle_fluid_collision_pipeline::apply_model",
    Kokkos::TeamPolicy<Space>(nx*ny*nz, Kokkos::AUTO()),
    KOKKOS_LAMBDA (member_type team_member) {

      int ix, iy, iz;
      RANK_TO_INDEX(team_member.league_rank(), ix, iy, iz, nx, ny, nz);
      const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);

      // Find number of particles for each species.
      auto i0 = spi_partition_ra(v);
      auto ni = spi_partition_ra(v+1) - i0;

      // TODO: convert this to be a more explicit check on if we have work
      if( ni <= 0 ) return; //Nothing to do
      
      //// Compute ndt
      //const float density_min = density_j > density_i ? density_i : density_j;
      //const float ndt = density_min*dtinterval;
      const float dt = dtinterval;

      // Get a random generator. Do not leave without freeing it.
      kokkos_rng_state_t rg = rp.get_state();

      // Extract fluid variables
      // const float n_fl   = spj_fl(v, fluid_var::den);
      // const float ux_fl  = spj_fl(v, fluid_var::ux);
      // const float uy_fl  = spj_fl(v, fluid_var::uy);
      // const float uz_fl  = spj_fl(v, fluid_var::uz);
      // const float tmp_fl = spj_fl(v, fluid_var::tmp);

      //for each cell
      gmomType Dm; 
      
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, ni),
      [&](const size_t& k, gmomType &lsum) {
      
        int i = spi_sortindex_ra(i0 + k);

#ifdef VARIABLE_CHARGE
        float up[5] = { spi_p(i, particle_var::w),
                        spi_p(i, particle_var::ux),
                        spi_p(i, particle_var::uy),
                        spi_p(i, particle_var::uz),
                        spi_p(i, particle_var::qp) };
#else
        float up[4] = { spi_p(i, particle_var::w),
                        spi_p(i, particle_var::ux),
                        spi_p(i, particle_var::uy),
                        spi_p(i, particle_var::uz) };
#endif      

        float wp   = up[0];
        float ux_n = up[1];
        float uy_n = up[2];
        float uz_n = up[3];

        float qp_n = 0.0, qp_i = 0.0;
#ifdef VARIABLE_CHARGE
        qp_n = up[4];
#endif

        bool MC_col_occurred = false;
        if( use_e_field ) {
          particle_bulk_collision(mi, mj, mu, mu_i, mu_j, up, spj_fd, model, rg, dt, v, MC_col_occurred);
        } else {      
          particle_bulk_collision(mi, mj, mu, mu_i, mu_j, up, spj_fl, model, rg, dt, v, MC_col_occurred);
        }
    
        float ux_i = up[1];
        float uy_i = up[2];
        float uz_i = up[3];
#ifdef VARIABLE_CHARGE
        qp_i = up[4];
        spi_p(i, particle_var::qp) = qp_i;
#endif
        spi_p(i, particle_var::ux) = ux_i;
        spi_p(i, particle_var::uy) = uy_i;
        spi_p(i, particle_var::uz) = uz_i;	 

        // Accumulate change in moments. Depends on collision type.
        float dn = 0.0, dux = 0.0, duy = 0.0, duz = 0.0, den = 0.0;

        switch (model.collision_type) 
        {
          case CollisionType::BulkChargeExchange:
          {
            if (!MC_col_occurred) { break; }

            // When a particle undergoes charge exchange and 
            // the projectile particle captures an electron,
            // then decrement the neutral fluid density              
            int dq = qp_i - qp_n;
            if (dq == -1) {
              // Change in neutral density is dn=w_particle/vol_cell (accumulated in reduction)
              dn = wp * rdV;
            }

            break; // end case(charge exchange)
          }
          case CollisionType::BulkElectronImpactIoniz:
          {
            if (!MC_col_occurred) { break; }
            // todo: modify electron fluid

            // Change in electron fluid energy due to the inelastic collision
            // is given by conservation of energy. It equals the ionization 
            // energy minus the energy removed from the ionizing atom
            // E_n + E_e1 = E_i + E_e1' + E_e2' + E_ionize
            // dE_e = E_e1 - (E_e1' + E_e2')
            //      = E_ionize - (E_n - E_i)
            //      = E_ionize - dE_n
            // den = wp * ( model.dE - 0.5 * mj *
            //   ( ( ux_n * ux_n + uy_n * uy_n + uz_n * uz_n ) -
            //     ( ux_i * ux_i + uy_i * uy_i + uz_i * uz_i ) ) );

            // Choose electron momentum from conservation
            // p_n + p_e1 = p_i + p_e1' + p_e2'
            // p_n - p_i = (p_e1' + p_e2') - p_e1
            // dp_n = -dp_e
            
            // dux = wp * mi / mj * (ux_i - ux_n);
            // duy = wp * mi / mj * (uy_i - uy_n);
            // duz = wp * mi / mj * (uz_i - uz_n);
            
            break; // end case(electron impact ionization)
          }
          case CollisionType::BulkDrag:
          {
            // Change in the fluid momentum and energy due to drag 
            // is due to the slowing down of the particle
            dux = ( ux_i - ux_n ) * wp;
            duy = ( uy_i - uy_n ) * wp;
            duz = ( uz_i - uz_n ) * wp;
            den = 0.5 * wp *
                ( ( ux_i * ux_i + uy_i * uy_i + uz_i * uz_i ) -
                  ( ux_n * ux_n + uy_n * uy_n + uz_n * uz_n ) );
            break; // end case(drag)
          }
          case CollisionType::BulkLemons:
          {
            dux = ( ux_i - ux_n ) * wp;
            duy = ( uy_i - uy_n ) * wp;
            duz = ( uz_i - uz_n ) * wp;
            den = 0.5 * static_cast<double>(wp)
                      * ( static_cast<double>(ux_i)*ux_i
                        + static_cast<double>(uy_i)*uy_i
                        + static_cast<double>(uz_i)*uz_i );
            break; // end case(lemons)
          }
          case CollisionType::BulkIonImpactIoniz: // only implemented for case with products
          default:
              break;
        } // end switch(model.collision_type) 
  
        lsum.add(0, wp);
        lsum.add(1, dux);
        lsum.add(2, duy);
        lsum.add(3, duz);
        lsum.add(4, den);
        lsum.add(5, dn);
      }, Dm);
      if (team_member.team_rank() == 0) {
        // Code that runs once per team leader
        if( use_e_field ) {
          // If we have a field, we upload the moment source to the field.
          // Upload the moment source to the field.
          model.upload_moment_src( spj_fd, v, Dm, mi, mj, 0.0 );
        } else {    
          float m_fluid_ttl = spj_fl(v, fluid_var::den) / rdV; // use total fluid mass = n*dV
          if (m_fluid_ttl > 0.0) {
            model.upload_moment_src( spj_fl, v, Dm, mi, mj, m_fluid_ttl );   
          }
        }
      }

        // We *must* free generators.
      rp.free_state(rg);

    });
    
    // I don't know why we need this, but without it I get an illegal memory
    // access error ... suspicious.
    Kokkos::fence();
  }

  /**
   * @brief Loop over particles performing collisions.
   *        Same as apply_model() but adds products
   *        to other particle groups.
   */
  template<class collision_model>
  void apply_model_products (
    collision_model& _model
  )
  {
    // NOTE: workaround to avoid implicit capture of this
    // SEE:  kokkos lambda dispatch link at top
    auto const& model = _model;
    auto const& mi   = _mi;
    auto const& mj   = _mj;
    auto const& mu_i = _mu_i;
    auto const& mu_j = _mu_j;
    auto const& mu = _mu;
    auto const& rdV = _rdV;
    auto const& nx = _nx;
    auto const& ny = _ny;
    auto const& nz = _nz;
    // auto const& spi = _spi;
    // auto const& spj = _spj;
    auto const& rp  = _rp;
    // auto const& spi_n = _spi_n;
    auto const& spi_i = _spi_i;
    auto const& spi_p = _spi_p;
    auto const& spj_fl = _spj_fl;
    auto const& spj_fd = _spj_fd;
    auto const& dtinterval = _dtinterval;
    auto const& spi_sortindex_ra = _spi_sortindex_ra;
    auto const& spi_partition_ra = _spi_partition_ra;
    auto const& use_e_field = _use_e_field;

    auto const& spp = _spp;
    auto const& spp_p = _spp_p;
    auto const& spp_i = *_spp_i;

    // Number of particles in product group
    const int np_products0 = spp->np;
    Kokkos::View<int*, Space::memory_space> dev_np_products("dev_np_products", 1);
    Kokkos::deep_copy(dev_np_products, 0);

    Kokkos::parallel_for("particle_fluid_collision_pipeline::apply_model",
      Kokkos::TeamPolicy<Space>(nx*ny*nz, Kokkos::AUTO()),
      KOKKOS_LAMBDA (member_type team_member) {

        int ix, iy, iz;
        RANK_TO_INDEX(team_member.league_rank(), ix, iy, iz, nx, ny, nz);
        const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);

        // Find number of particles for each species.
        auto i0 = spi_partition_ra(v);
        auto ni = spi_partition_ra(v+1) - i0;

        if( ni <= 0 ) return; // Nothing to do

	      const float dt = dtinterval;
	
        // Get a random generator. Do not leave without freeing it.
        kokkos_rng_state_t rg = rp.get_state();
	
        // Extract fluid variables
        const float n_fl   = spj_fl(v, fluid_var::den);
        const float ux_fl  = spj_fl(v, fluid_var::ux);
        const float uy_fl  = spj_fl(v, fluid_var::uy);
        const float uz_fl  = spj_fl(v, fluid_var::uz);
        const float tmp_fl = spj_fl(v, fluid_var::tmp);
        const float uth_fl = (tmp_fl > 0.0) ? sqrt(tmp_fl / mj) : 0.0;

        // Accumulate moments for each cell
        gmomType Dm; 
	
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, ni),
        [&](const int& k, gmomType &lsum) {

          int i = spi_sortindex_ra(i0 + k);

#ifdef VARIABLE_CHARGE
          float up[5] = { spi_p(i, particle_var::w),
                          spi_p(i, particle_var::ux),
                          spi_p(i, particle_var::uy),
                          spi_p(i, particle_var::uz),
                          spi_p(i, particle_var::qp) };
#else
          float up[4] = { spi_p(i, particle_var::w),
                          spi_p(i, particle_var::ux),
                          spi_p(i, particle_var::uy),
                          spi_p(i, particle_var::uz) };
#endif			      

          float wp   = up[0];
          float ux_n = up[1];
          float uy_n = up[2];
          float uz_n = up[3];

          float qp_n = 0.0, qp_i = 0.0;
#ifdef VARIABLE_CHARGE
          qp_n = up[4];
#endif

          bool MC_col_occurred = false;
          if( use_e_field ) {
            particle_bulk_collision(mi, mj, mu, mu_i, mu_j, up, spj_fd, model, rg, dt, v, MC_col_occurred);
          } else {
            particle_bulk_collision(mi, mj, mu, mu_i, mu_j, up, spj_fl, model, rg, dt, v, MC_col_occurred);
          }
	    
          float ux_i = up[1];
          float uy_i = up[2];
          float uz_i = up[3];
#ifdef VARIABLE_CHARGE
          qp_i = up[4];
          spi_p(i, particle_var::qp) = qp_i;
#endif
          spi_p(i, particle_var::ux) = ux_i;
          spi_p(i, particle_var::uy) = uy_i;
          spi_p(i, particle_var::uz) = uz_i;	 

          // Accumulate change in moments. Depends on collision type.
          float dn = 0.0, dux = 0.0, duy = 0.0, duz = 0.0, den = 0.0;

          switch (model.collision_type) {
            case CollisionType::BulkChargeExchange:
            {
              if (!MC_col_occurred) { break; }

              // When a particle undergoes charge exchange and 
              // the projectile particle captures an electron,
              // then decrement the neutral fluid density              
              int dq = qp_i - qp_n;
              if (dq != -1) { break; }
                
              // Change in neutral density is dn=w_particle/vol_cell (accumulated in reduction)
              dn = wp * rdV;

              // The new kinetic particle takes the fluid bulk velociy plus a thermal component
              float ux_pr = rg.normal(ux_fl, uth_fl);
              float uy_pr = rg.normal(uy_fl, uth_fl);
              float uz_pr = rg.normal(uz_fl, uth_fl);
              float w_pr = wp;

              // Create kinetic particle. Get particle index and incremenent number of new products
              int cntr = Kokkos::atomic_fetch_add(&dev_np_products(0), 1);
              int i_pr = np_products0 + cntr;

              spp_p(i_pr, particle_var::w)  = w_pr;
              spp_p(i_pr, particle_var::ux) = ux_pr;
              spp_p(i_pr, particle_var::uy) = uy_pr;
              spp_p(i_pr, particle_var::uz) = uz_pr;	  
              spp_p(i_pr, particle_var::dx) = spi_p(i, particle_var::dx);
              spp_p(i_pr, particle_var::dy) = spi_p(i, particle_var::dy);
              spp_p(i_pr, particle_var::dz) = spi_p(i, particle_var::dz);	  
              spp_i(i_pr) = spi_i(i);
#ifdef VARIABLE_CHARGE
              spp_p(i_pr, particle_var::qp) = 1; // spj->q - dq;
#endif

              // Decrement fluid momentum and energy based on new kinetic particle
              dux = (ux_fl - ux_pr) * w_pr;
              duy = (uy_fl - uy_pr) * w_pr;
              duz = (uz_fl - uz_pr) * w_pr;
              den = 0.5 * ( dux * dux + duy * duy + duz * duz ) / w_pr;

              break; // end case(charge exchange)
            }
            case CollisionType::BulkIonImpactIoniz:
            {
              if (!MC_col_occurred) { break; }

              // Change in neutral density is dn=w_particle/vol_cell (accumulated in reduction)
              dn = wp * rdV;

              // The new kinetic particle takes the fluid bulk velociy plus a thermal component
              float ux_pr = rg.normal(ux_fl, uth_fl);
              float uy_pr = rg.normal(uy_fl, uth_fl);
              float uz_pr = rg.normal(uz_fl, uth_fl);
              float w_pr = wp;

              // Create kinetic particle. Get particle index and incremenent number of new products
              int cntr = Kokkos::atomic_fetch_add(&dev_np_products(0), 1);
              int i_pr = np_products0 + cntr;

              spp_p(i_pr, particle_var::w)  = w_pr;
              spp_p(i_pr, particle_var::ux) = ux_pr;
              spp_p(i_pr, particle_var::uy) = uy_pr;
              spp_p(i_pr, particle_var::uz) = uz_pr;	  
              spp_p(i_pr, particle_var::dx) = spi_p(i, particle_var::dx);
              spp_p(i_pr, particle_var::dy) = spi_p(i, particle_var::dy);
              spp_p(i_pr, particle_var::dz) = spi_p(i, particle_var::dz);	  
              spp_i(i_pr) = spi_i(i);
#ifdef VARIABLE_CHARGE
              // Currently only considering ionizing neutral fluid (0->1)
              spp_p(i_pr, particle_var::qp) = 1;
#endif

              // Decrement fluid momentum and energy based on new kinetic particle
              dux = ux_pr * w_pr;
              duy = uy_pr * w_pr;
              duz = uz_pr * w_pr;
              den = 0.5 * w_pr * ( ux_pr * ux_pr + uy_pr * uy_pr + uz_pr * uz_pr );

              break; // end case(ion impact ionization)
            }
            case CollisionType::BulkElectronImpactIoniz:
            case CollisionType::BulkDrag:
            case CollisionType::BulkLemons:
            {
              break; // end case(drag,lemons,electron-ionization)
            }
            default:
              break;
          } // end switch(model.collision_type) 
    
          lsum.add(0, wp);
          lsum.add(1, dux);
          lsum.add(2, duy);
          lsum.add(3, duz);
          lsum.add(4, den);
          lsum.add(5, dn);
	      }, Dm); // end Kokkos::parallel_reduce
	
        if (team_member.team_rank() == 0) {
          // Code that runs once per team leader
          if( use_e_field ) {
            // If we have a field, we upload the moment source to the field.
            // Upload the moment source to the field.
            model.upload_moment_src( spj_fd, v, Dm, mi, mj, 0.0 );
          } else {    
            float m_fluid_ttl = spj_fl(v, fluid_var::den) / rdV; // use total fluid mass = n*dV
            if (m_fluid_ttl > 0.0) {
              model.upload_moment_src( spj_fl, v, Dm, mi, mj, m_fluid_ttl );   
            }   
          }
      	}

        // We *must* free generators.
        rp.free_state(rg);
    }); // end Kokkos::parallel_for
    
    Kokkos::fence();

    // Increment number of particles in product species
    Kokkos::View<int*, Space>::HostMirror host_np_products = Kokkos::create_mirror_view(dev_np_products);
    Kokkos::deep_copy(host_np_products, dev_np_products);
    spp->np += host_np_products(0);    

  } // end apply_model_products()


  /**
   * @brief Perform a collision between two particles.
   */
  //Must have all struct member types passed in directly to the inline function
  //In terms of notation, all inline functions should not have variables that start
  //with the _ (underscore), because _ is used to indicate a class member before it
  //is caputred by a lambda. One lambda captured, we should refer to the variable
  //as EX: mu not _mu
  template<class view_type, class collision_model>
  KOKKOS_INLINE_FUNCTION
  void particle_bulk_collision (
    const float mi,
    const float mj,
    const float mu,
    const float mu_i,
    const float mu_j,
#ifdef VARIABLE_CHARGE
    float (&up)[5],
#else
    float (&up)[4],
#endif
    const view_type& spj_f,
    collision_model& model,
    kokkos_rng_state_t& rg,
    float dt,
    int ii,
    bool& MC_collision_occurred
  )
  {

    float dd, ur, tx, ty, tz, t0, t1, t2, stack[3];
    int d0, d1, d2;

    float uix = up[1];
    float uiy = up[2];
    float uiz = up[3];
    //float wi  = up[0];

    float qi = 0;
#ifdef VARIABLE_CHARGE
    qi  = up[4];
#endif

    // Extract fluid vars
    float nj_fl, ujx_fl, ujy_fl, ujz_fl, tmp_fl;
    if constexpr (std::is_same<view_type, k_fluid_t>::value) {
      nj_fl  = spj_f(ii, fluid_var::den);
      ujx_fl = spj_f(ii, fluid_var::ux);
      ujy_fl = spj_f(ii, fluid_var::uy);
      ujz_fl = spj_f(ii, fluid_var::uz);
      tmp_fl = spj_f(ii, fluid_var::tmp);
    } else if constexpr (std::is_same<view_type, k_field_t>::value) {
      nj_fl  = spj_f(ii, field_var::rhof);
      ujx_fl = spj_f(ii, field_var::ux);
      ujy_fl = spj_f(ii, field_var::uy);
      ujz_fl = spj_f(ii, field_var::uz);
      tmp_fl = spj_f(ii, field_var::pe)/nj_fl; //nj_fl should be non-zero

      // // Use bulk electron flow for Lemon's collision and
      // // sample thermal velocity for electron impact ionization
      // if (model.collision_type == CollisionType::BulkElectronImpactIoniz) {
      //   float uth_fl = sqrt(2.0 * tmp_fl / mj);
      //   ujx_fl = rg.normal(ujx_fl, uth_fl);
      //   ujy_fl = rg.normal(ujy_fl, uth_fl);
      //   ujz_fl = rg.normal(ujz_fl, uth_fl);
      // }
    }

    // Skip if there is no more fluid present
    if (nj_fl <= 0.0) { return; }

    float ndt = nj_fl * dt;
    
    // Relative velocity
    float urx = uix - ujx_fl;
    float ury = uiy - ujy_fl;
    float urz = uiz - ujz_fl;

    // Thermal velocity
    float ujth = sqrt(2.0*tmp_fl/mj);
    
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
    } //make the smallerst |urx(,y,z)| in the d0 ( or z) direction
    t2 += t0;

    ur = sqrtf( t2 );

    // Collision parameters
    t2 *= mu;       // _mu v^2  = Collision energy
    t1  = ur*ndt;   // n v dt  = Particles encountered per unit area

    // Monte-Carlo collision test
    if( MonteCarlo ) {

      // TODO : CPU VPIC warned when dd*t1 > 1 for under-resolved collisions.
      //        Would this be useful?
      dd = model.cross_section( rg, ur, t1, qi, t2 );

      if( rg.frand() > dd*t1 ) {
        MC_collision_occurred = false;
        return;
      } else {
        MC_collision_occurred = true;
      }
    }

    // E0 is the initial energy from which energy is removed
    // during an inelastic collision. For ion impact ionization,
    // assume the fluid is at rest so E0 is the ion energy.
    // For electron impact ionization, E0 is the center-of-mass
    // energy including the sampled electron velocity. No energy
    // is removed for charge exchange or Lemon's Coulomb collision.
    //
    float E0 = 0.0; 

#ifdef VARIABLE_CHARGE
    const float dq = model.modify_charge();
    switch (model.collision_type) {
      case CollisionType::BulkChargeExchange:
      {
        up[4] += dq;
        break;
      }
      case CollisionType::BulkIonImpactIoniz:
      {
        // Note: the neutral fluid is ionized, not the
        // projectile ion so don't modify up[4]
        E0 = 0.5 * mi * ((uix*uix) + (uiy*uiy) + (uiz*uiz));
        break;
      }
      case CollisionType::BulkElectronImpactIoniz:
      {
        E0 = t2;
        up[4] += 1.0;
        break;
      }
      default:
        break;
    }
#endif

    // Compute collision angle and coefficient of restitution
    float param[5] = {ur, ujth, ndt/(mi*mi), mi/mj, E0};
    const float rr = model.restitution(rg, param);
    dd = model.tan_theta_half(rg, param);
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

    /* stack = (1 - cos theta) u + |u| sin theta + Tperp */
    stack[0] = (t0*urx + t1*tx) + t2*( ury*tz - urz*ty );
    stack[1] = (t0*ury + t1*ty) + t2*( urz*tx - urx*tz );
    stack[2] = (t0*urz + t1*tz) + t2*( urx*ty - ury*tx );

    // For electron impact ionization, we perform a binary collision
    // to determine ion momentum. The electron fluid momentum
    // is set based on momentum conservation.
    if (model.collision_type == CollisionType::BulkElectronImpactIoniz) {

      // Scaled center of mass velocity.
      t1 = (1-rr);
      float cmx = t1*(mu_j*uix + mu_i*ujx_fl);
      float cmy = t1*(mu_j*uiy + mu_i*ujy_fl);
      float cmz = t1*(mu_j*uiz + mu_i*ujz_fl);

      up[1] = (uix + mu_i*stack[0])*rr + cmx;
      up[2] = (uiy + mu_i*stack[1])*rr + cmy;
      up[3] = (uiz + mu_i*stack[2])*rr + cmz;
                
    } else {

      up[1] = ujx_fl + (urx + stack[0])*rr;
      up[2] = ujy_fl + (ury + stack[1])*rr;
      up[3] = ujz_fl + (urz + stack[2])*rr;

    }

    return;
  }

};

#endif
