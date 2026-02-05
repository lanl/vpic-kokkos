#ifndef _kokkos_binary_neutral_collision_pipeline_h_
#define _kokkos_binary_neutral_collision_pipeline_h_

#include "../collision_private.h"

// Assumes single precision.
// Chosen as a cutoff < sqrt(FLT_MAX) such that dd/(1+dd*dd) is always in range.
#define TAN_THETA_HALF_MAX 1.30e19f
#define PREVENT_BACKSCATTER(TAN) do  {                                          \
  if(!isfinite(TAN) || (TAN) > TAN_THETA_HALF_MAX ) (TAN) = TAN_THETA_HALF_MAX; \
} while(0)

 
/**
 * @brief General purpose pipeline to produce binary collisions including neutrals.
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
 *    cross_section(rg, E, nvdt)
 *        Returns the cross-section for the collision in normalized units.
 *        The collision will occur with probability cross_section*nvdt.
 *
 * ######################IMPORTANT DOCUMENTATION ##############
 * For information on use of lambdas inside struct and classes:
 *     https://github.com/kokkos/kokkos/wiki/Lambda-Dispatch
 * ############################################################
 */
template<bool VariableWeight>
struct binary_neutral_collision_pipeline {

  using Space=Kokkos::DefaultExecutionSpace;
  using member_type=Kokkos::TeamPolicy<Space>::member_type;
  using k_density_t=Kokkos::View<float *, Space>;

  const float _m_i, _m_j, _mu_i, _mu_j, _mu, _dtinterval, _dV;
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

  field_array_t *_field; //for electron-ion collisions
  k_field_t _spj_fd; 
  bool _use_e_field;
    
  // Product species for fusion reactions
  // species_t *_spp1, *_spp2;
  // k_particles_t _spp1_p, _spp2_p;
  // k_particles_i_t *_spp1_i, *_spp2_i;

  // Random access, read-only Views
  // TODO : Does RandomAccess trait really matter?
  k_particle_sortindex_t_ra _spi_sortindex_ra, _spj_sortindex_ra;
  k_particle_partition_t_ra _spi_partition_ra, _spj_partition_ra;

  binary_neutral_collision_pipeline(
    species_t * spi,
    species_t * spj,
    double interval,
    kokkos_rng_pool_t& rp,
    field_array_t * field
    // species_t * spp1=NULL,
    // species_t * spp2=NULL
  )
    : _m_i( spi->m ),
      _m_j( spj->m ),
      _mu_i(spj->m / (spi->m + spj->m)),
      _mu_j(spi->m / (spi->m + spj->m)),
      _mu(spi->m*spj->m / (spi->m + spj->m)),
      _dtinterval(spi->g->dt * interval),
      _dV(spi->g->dV),
      _nx(spi->g->nx),
      _ny(spi->g->ny),
      _nz(spi->g->nz),
      _spi(spi),
      _spj(spj),
      _rp(rp),
      _field(field)
      // _spp1(spp1),
      // _spp2(spp2)
  {
    //TODO: is interval needed here?
    if( !_spi || !_spj || !_spi->g || !_spj->g || _spi->g != _spj->g || interval <= 0)
      ERROR(("Bad args."));
    if(_field==NULL) _use_e_field = false;
    else _use_e_field = true;
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
    if( _spi->np      > _spi_sortindex_ra.extent(0) || 
        _spi->g->nv+1 != _spi_partition_ra.extent(0) ){
	printf("_spi->np (=%d) ?= _spi_sortindex_ra.extent(0) (=%d)\n",_spi->np,_spi_sortindex_ra.extent(0));
	printf("_spi->g->nv+1 (=%d) ?= _spi_partition_ra.extent(0) (=%d)\n",_spi->g->nv+1,_spi_partition_ra.extent(0));
        ERROR(("Bad spi sort products."));
    }

    // We only need to shuffle one species to ensure random pairings.
    shuffler.shuffle( _spi, _rp, false );
    if(_spi!=_spj)  shuffler.shuffle( _spj, _rp, false );

    // Compute species densities using a simple histogram. Batching these
    // beforehand is much faster than doing it inline.
    _spi_n = k_density_t("spi_n", _spi->g->nv);
    _spj_n = k_density_t("spj_n", _spj->g->nv);

    const float rdV = (1/_spi->g->dV);

    // NOTE: workaround to avoid implicit capture of this
    // SEE:  kokkos lambda dispatch link at top
    auto const& spi_n = _spi_n;
    auto const& spi_p = _spi_p;
    auto const& spi_i = _spi_i;
    Kokkos::parallel_for("binary_neutral_collision_pipeline::spi_denisty",
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
      Kokkos::parallel_for("binary_neutral_collision_pipeline::spj_denisty",
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
    
    apply_model(_model);
    // if (_spp == NULL) {
    //   apply_model(_model);
    // } else {
    //   _spp_p = _spp->k_p_d;
    //   _spp_i = &_spp->k_p_i_d;
    //   apply_model_products(_model);
    // }
  }

  /**
   * @brief Loop over particles performing collisions.
   */
  template<class collision_model>
  void apply_model (collision_model& _model)
  {
    // NOTE: workaround to avoid implicit capture of this
    // SEE:  kokkos lambda dispatch link at top
    auto const& model = _model;
    auto const& m_i   = _m_i;
    auto const& m_j   = _m_j;
    auto const& mu_i  = _mu_i;
    auto const& mu_j  = _mu_j;
    auto const& mu    = _mu;
    auto const& dV    = _dV;
    auto const& nx    = _nx;
    auto const& ny    = _ny;
    auto const& nz    = _nz;
    auto const& spi   = _spi;
    auto const& spj   = _spj;
    auto const& rp    = _rp;
    auto const& spi_n = _spi_n;
    auto const& spj_n = _spj_n;
    auto const& spi_p = _spi_p;
    auto const& spj_p = _spj_p;
    auto const& spi_i = _spi_i;
    auto const& spj_i = _spj_i;
    auto const& dtinterval = _dtinterval;
    auto const& spi_sortindex_ra = _spi_sortindex_ra;
    auto const& spj_sortindex_ra = _spj_sortindex_ra;
    auto const& spi_partition_ra = _spi_partition_ra;
    auto const& spj_partition_ra = _spj_partition_ra;
    auto const& use_e_field = _use_e_field;

    // Choose the collision function based on model.var_wt.
    // Both functions must be of the same signature.
    auto policy = Kokkos::TeamPolicy<Space>(nx*ny*nz, Kokkos::AUTO());
    if (model.var_wt) {
      constexpr int n_int   = 1;
      constexpr int n_float = 1;	
      constexpr int level   = 0; //per team shared momery
      policy = policy.set_scratch_size(level, Kokkos::PerTeam(n_int*sizeof(int)+n_float*sizeof(float)));
    }

    if constexpr (VariableWeight) {    
	    // Kokkos::parallel_for("binary_neutral_collision_pipeline::apply_model::var_wt",
			//  policy,
			//  KOKKOS_LAMBDA (member_type team_member) {
			// 	int ix, iy, iz;
			// 	RANK_TO_INDEX(team_member.league_rank(), ix, iy, iz, nx, ny, nz);
			// 	const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
				
			// 	// Find number of particles for each species.
			// 	auto i0 = spi_partition_ra(v);
			// 	auto ni = spi_partition_ra(v+1) - i0;
				
			// 	auto j0 = spj_partition_ra(v);
			// 	auto nj = spj_partition_ra(v+1) - j0;
				
			// 	if( ni <= 0 || nj <= 0 || (spi==spj && ni==1) ) return; //Nothing to do
				
			// 	// Find the real densities.
			// 	float density_i = spi_n(v);
			// 	float density_j = spj_n(v);

			// 	if(spi_p == spj_p) {
			// 		collide_self_varwt(m_i, m_j, density_i, density_j, dV, i0, j0, ni, nj, dtinterval, spi_p, spj_p, model, spi_sortindex_ra, spj_sortindex_ra, rp, team_member); 
			// 	} else {
			// 		collide_variabl_wt(m_i, m_j, density_i, density_j, dV, i0, j0, ni, nj, dtinterval, spi_p, spj_p, model, spi_sortindex_ra, spj_sortindex_ra, rp, team_member);
			// 	}
			//  });

  	} else {
	    Kokkos::parallel_for("binary_neutral_collision_pipeline::apply_model::uniform_wt",
			 policy,
			 KOKKOS_LAMBDA (member_type team_member) {
          int ix, iy, iz;
          RANK_TO_INDEX(team_member.league_rank(), ix, iy, iz, nx, ny, nz);
          const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
          
          // Find number of particles for each species.
          auto i0 = spi_partition_ra(v);
          auto ni = spi_partition_ra(v+1) - i0;
          
          auto j0 = spj_partition_ra(v);
          auto nj = spj_partition_ra(v+1) - j0;
        
          if( ni <= 0 || nj <= 0 || (spi==spj && ni==1) ) return; //Nothing to do
        
          // Find the real densities.
          float density_i = spi_n(v);
          float density_j = spj_n(v);

          collide_uniform_wt(m_i, m_j, density_i, density_j, dV, i0, j0, ni, nj, \
                             dtinterval, spi_p, spj_p, model, \
                             spi_sortindex_ra, spj_sortindex_ra, rp, team_member);
			 });
    }

    Kokkos::fence();
  } // end apply_model()


template<class collision_model>    
KOKKOS_INLINE_FUNCTION
void collide_uniform_wt(
  const float m_i, const float m_j, const float density_i, const float density_j,
  const float dV, int i0, int j0, int ni, int nj, const float dtinterval,
  const k_particles_t& spi_p, const k_particles_t& spj_p, collision_model& model,
  k_particle_sortindex_t_ra spi_sortindex_ra, k_particle_sortindex_t_ra spj_sortindex_ra,
  const kokkos_rng_pool_t& rp, const Kokkos::TeamPolicy<>::member_type & team_member)
{
  const float mu_i = m_j/(m_i+m_j);
  const float mu_j = m_i/(m_i+m_j);
  const float mu = m_i*m_j/(m_i+m_j);
    
  // Compute ndt
  const float density_max = density_i >= density_j ? density_i : density_j;
  float ndt = density_max*dtinterval;

  // Get a random generator. Do not leave without freeing it.
  kokkos_rng_state_t rg = rp.get_state();

  // Handle intraspecies.
  if( spi_p == spj_p ) {
    if(ni & 1) { //odd
  		ndt *= (float)(ni)/(float)(ni-1);
	  	// correspondingly, the collision probability decreased to (ni-1)/ni, i.e., one particle does not collide. We can use the same collision kernel.
	  }   // else ni is even, and no adjustment needed
    // Even number of particles.
    nj = ni = ni/2;
    j0 = i0 + ni;
  }
	
	const int nmin = ni < nj ? ni : nj;

	Kokkos::parallel_for(
    Kokkos::TeamThreadRange(team_member, nmin),
    [&](const int c) {
      int i = spi_sortindex_ra(i0 + c);
      int j = spj_sortindex_ra(j0 + c);

      float up[10];    
      up[0] = spi_p(i, particle_var::w);
      up[1] = spi_p(i, particle_var::ux);
      up[2] = spi_p(i, particle_var::uy);
      up[3] = spi_p(i, particle_var::uz);
      up[4] = spj_p(j, particle_var::w);
      up[5] = spj_p(j, particle_var::ux);
      up[6] = spj_p(j, particle_var::uy);
      up[7] = spj_p(j, particle_var::uz);
      up[8] = 1.0;
	    up[9] = 1.0;	
#ifdef VARIABLE_CHARGE
      up[8] = spi_p(i, particle_var::qp);
      up[9] = spj_p(j, particle_var::qp);
#endif	

      bool MC_col_occurred = binary_collision(mu, mu_i, mu_j, up, model, rg, ndt);

      spi_p(i, particle_var::ux) = up[1];
      spi_p(i, particle_var::uy) = up[2];
      spi_p(i, particle_var::uz) = up[3];	  
      spj_p(j, particle_var::ux) = up[5];
      spj_p(j, particle_var::uy) = up[6];
      spj_p(j, particle_var::uz) = up[7];	  	
#ifdef VARIABLE_CHARGE 
      spi_p(i, particle_var::qp) = up[8];
      spj_p(j, particle_var::qp) = up[9];	  	 
#endif

      // switch (model.collision_type) {
      //   case CollisionType::BulkIonImpactIoniz:
      //   {
      //     if (!MC_col_occurred) { break; }
      //     // Increase electron temperature...
      //     break;
      //   }
      //   default:
      //     break;
      // }

	});
    // We *must* free generators.
	rp.free_state(rg);
}


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
  bool binary_collision (
    const float mu,
    const float mu_i,
    const float mu_j,
    float* up,
    collision_model& model,
    kokkos_rng_state_t& rg,
    float ndt
  )
  {

    float dd, ur, tx, ty, tz, t0, t1, t2, stack[3];
    int d0, d1, d2;
    
    float wi  = up[0];
    float uix = up[1];
    float uiy = up[2];
    float uiz = up[3];
    float qi  = 1.0;
#ifdef VARIABLE_CHARGE
    qi = up[8];
#endif
    float wj  = up[4];
    float ujx = up[5];
    float ujy = up[6];
    float ujz = up[7];
    float qj  = 1.0; //
#ifdef VARIABLE_CHARGE
    qj = up[9];
#endif    

    // Relative velocity
    float urx = uix - ujx;
    float ury = uiy - ujy;
    float urz = uiz - ujz;

    /* There are lots of ways to formulate T vector formation    */
    /* This has no branches (but uses L1 heavily)                */
    bool MC_col_occurred = false;
    if (urx == 0 && ury == 0 && urz == 0) { return MC_col_occurred; }

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
    dd = model.cross_section( rg, qi, ur, t1 );

    // Determine if collision occurs, if (U > sigma * n * v * dt) then no collision
    if( rg.frand() > dd*t1 ) {
      return MC_col_occurred; 
    } else {
      MC_col_occurred = true;
    }

    // Compute collision angle and coefficient of restitution
    float param[2] = {t2,t1};
    const float rr = model.restitution(rg, param);
    dd = model.tan_theta_half(rg, t2, t1*qi*qi*qj*qj);
    PREVENT_BACKSCATTER(dd);

#ifdef VARIABLE_CHARGE
    const float dq = model.modify_charge();
    switch (model.collision_type) {
      case CollisionType::BinaryChargeExchange:
      {
        up[8] += dq;
        up[9] -= dq;
        break;
      }
      case CollisionType::BinaryIonImpactIoniz:
      {
        // Second species in collision operator constructor drops electrons
        up[9] += dq;
        break;
      }
    }
#endif

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

    up[1] = (uix + mu_i*stack[0])*rr + cmx;
    up[2] = (uiy + mu_i*stack[1])*rr + cmy;
    up[3] = (uiz + mu_i*stack[2])*rr + cmz;
               
    up[5] = (ujx - mu_j*stack[0])*rr + cmx;
    up[6] = (ujy - mu_j*stack[1])*rr + cmy;
    up[7] = (ujz - mu_j*stack[2])*rr + cmz;

    return MC_col_occurred;
  } // end binary_collision()

}; // end struct binary_neutral_collision_pipeline
#endif