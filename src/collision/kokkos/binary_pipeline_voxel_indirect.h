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
template<bool VariableWeight>
struct binary_collision_pipeline {

  using Space=Kokkos::DefaultExecutionSpace;
  using member_type=Kokkos::TeamPolicy<Space>::member_type;
  using k_density_t=Kokkos::View<float *, Space>;

  const float _mu_i, _mu_j, _mu, _dtinterval, _dV;
  const int   _nx, _ny, _nz;
  const float _m_i, _m_j;
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

  binary_collision_pipeline(
    species_t * spi,
    species_t * spj,
    float interval,
    kokkos_rng_pool_t& rp
  )
    : _mu_i(spj->m / (spi->m + spj->m)),
      _mu_j(spi->m / (spi->m + spj->m)),
      _mu(spi->m*spj->m / (spi->m + spj->m)),
      _m_i(spi->m),
      _m_j(spj->m),
      _dtinterval(spi->g->dt * interval),
      _dV(spi->g->dV),
      _nx(spi->g->nx),
      _ny(spi->g->ny),
      _nz(spi->g->nz),
      _spi(spi),
      _spj(spj),
      _rp(rp)
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

      k_ParticleSorter<BinSort> sorter;
      //k_ParticleSorter<> sorter;      
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
    if( _spj->np      > _spj_sortindex_ra.extent(0) ||
        _spj->g->nv+1 != _spj_partition_ra.extent(0) ){
	printf("_spi->np (=%d) ?= _spj_sortindex_ra.extent(0) (=%d)\n",_spj->np,_spj_sortindex_ra.extent(0));
	printf("_spj->g->nv+1 (=%d) ?= _spj_partition_ra.extent(0) (=%d)\n",_spj->g->nv+1,_spj_partition_ra.extent(0));	
        ERROR(("Bad spj sort products."));
    }
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

    if( _spi != _spj ) {
        // NOTE: workaround to avoid implicit capture of this
        // SEE:  kokkos lambda dispatch link at top
        auto const& spj_n = _spj_n;
        auto const& spj_p = _spj_p;
        auto const& spj_i = _spj_i;
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

    // Do collisions.
    apply_model(_model);

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
    auto const& mu_i = _mu_i;
    auto const& mu_j = _mu_j;
    auto const& mu = _mu;
    auto const& m_i= _m_i;
    auto const& m_j= _m_j;
    auto const& dV = _dV;
    auto const& nx = _nx;
    auto const& ny = _ny;
    auto const& nz = _nz;
    auto const& spi = _spi;
    auto const& spj = _spj;
    auto const& rp  = _rp;
    auto const& spi_n = _spi_n;
    auto const& spj_n = _spj_n;
    auto const& spi_p = _spi_p;
    auto const& spj_p = _spj_p;
    auto const& dtinterval = _dtinterval;
    auto const& spi_sortindex_ra = _spi_sortindex_ra;
    auto const& spj_sortindex_ra = _spj_sortindex_ra;
    auto const& spi_partition_ra = _spi_partition_ra;
    auto const& spj_partition_ra = _spj_partition_ra;

    // Choose the collision function based on model.var_wt.
    // Both functions must be of the same signature.
    auto policy = Kokkos::TeamPolicy<Space>(nx*ny*nz, Kokkos::AUTO());
    if(model.var_wt){
	policy = policy.set_scratch_size(1, Kokkos::PerTeam(sizeof(int)));
    }

    Kokkos::parallel_for("binary_collision_pipeline::apply_model",
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
				 
			     // TODO: convert this to be a more explicit check on if we have work
			     if( ni <= 0 || nj <= 0 || (spi==spj && ni==1) ) return; //Nothing to do
				 
			     // Find the real densities.
			     // printf("cell index =%d\n",v);
			     float density_i = spi_n(v);
			     float density_j = spj_n(v);
			     if constexpr (VariableWeight) {
				     // printf("#call collide_variabl_wt()\n");
				     collide_variabl_wt(m_i, m_j, density_i, density_j, dV, i0, j0, ni, nj, dtinterval, spi_p, spj_p, model, spi_sortindex_ra, spj_sortindex_ra, rp, team_member);
			     } else {
				     collide_uniform_wt(m_i, m_j, density_i, density_j, dV, i0, j0, ni, nj, dtinterval, spi_p, spj_p, model, spi_sortindex_ra, spj_sortindex_ra, rp, team_member);
			     }
			 });



    // I don't know why we need this, but without it I get an illegal memory
    // access error ... suspicious.
    Kokkos::fence();

  }

    
template<class collision_model>    
KOKKOS_INLINE_FUNCTION
void collide_uniform_wt(const float m_i, const float m_j, const float density_i, const float density_j, const float dV, int i0, int j0, int ni, int nj, const float dtinterval,     const k_particles_t& spi_p,  const k_particles_t& spj_p,     collision_model& model, k_particle_sortindex_t_ra spi_sortindex_ra, k_particle_sortindex_t_ra spj_sortindex_ra, const kokkos_rng_pool_t& rp, const Kokkos::TeamPolicy<>::member_type & team_member)
{
    const float mu_i = m_j/(m_i+m_j);
    const float mu_j = m_i/(m_i+m_j);
    const float mu = m_i*m_j/(m_i+m_j);
    
        // Compute ndt
        //const float density_min = density_j > density_i ? density_i : density_j;
        //const float ndt = density_min*dtinterval;
        const float density_max = density_i >= density_j ? density_i : density_j;
        float ndt = density_max*dtinterval;

        // Get a random generator. Do not leave without freeing it.
        kokkos_rng_state_t rg = rp.get_state();

        // Handle intraspecies.
        if( spi_p == spj_p ) {
	    if(ni & 1) //odd
	    {
		ndt *= (float)(ni)/(float)(ni-1);
		//correspondingly, the collision probability decreased to (ni-1)/ni, i.e., one particle does not collide. We can use the same collision kernel.
	    }   //else ni is even, and no adjustment needed
            // Even number of particles.
            nj = ni = ni/2;
            j0 = i0 + ni;
	    
        }
	
	const int nmin = std::min(ni, nj);

	Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nmin),
			     [&](const int c) {
	 int i = spi_sortindex_ra(i0 + c);
	 int j = spj_sortindex_ra(j0 + c);
	 // if(c<10) printf("i=%d, j=%d\n",i,j);
	 float up[8] = { spi_p(i, particle_var::w ),
			 spi_p(i, particle_var::ux),
			 spi_p(i, particle_var::uy),
			 spi_p(i, particle_var::uz),
			 spj_p(j, particle_var::w ),
			 spj_p(j, particle_var::ux),
			 spj_p(j, particle_var::uy),
			 spj_p(j, particle_var::uz)};
	 
	 binary_collision(mu, mu_i, mu_j, up, model, rg, ndt);

	 spi_p(i, particle_var::ux) = up[1];
	 spi_p(i, particle_var::uy) = up[2];
	 spi_p(i, particle_var::uz) = up[3];	  
	 spj_p(j, particle_var::ux) = up[5];
	 spj_p(j, particle_var::uy) = up[6];
	 spj_p(j, particle_var::uz) = up[7];	  	 
	 
	 });
        // We *must* free generators.
        rp.free_state(rg);
}

template<class collision_model>    
KOKKOS_INLINE_FUNCTION
void collide_variabl_wt(const float m_i, const float m_j, const float density_i, const float density_j, const float dV, int i_0, int j_0, int ni, int nj, const float dtinterval,     const k_particles_t& spi_p,  const k_particles_t& spj_p,   const collision_model& model, k_particle_sortindex_t_ra spi_sortindex_ra, k_particle_sortindex_t_ra spj_sortindex_ra, const kokkos_rng_pool_t& rp, const Kokkos::TeamPolicy<>::member_type & team)
{
    //printf("#in collide_variabl_wt()\n");
    const float mu_i = m_j/(m_i+m_j);
    const float mu_j = m_i/(m_i+m_j);
    const float mu = m_i*m_j/(m_i+m_j);
    
    float twt[2] = {density_i*dV,density_j*dV}; //total weight within a cell
    // Compute ndt
    const bool ij    = density_i >= density_j;
    int i0 = ij ? i_0 : j_0;
    int j0 = ij ? j_0 : i_0;
    auto mh = ij ? m_i : m_j;
    auto ml = ij ? m_j : m_i;
    auto sph_p = ij ? spi_p : spj_p;
    auto spl_p = ij ? spj_p : spi_p;
    auto sph_sortindex_ra = ij ? spi_sortindex_ra : spj_sortindex_ra;
    auto spl_sortindex_ra = ij ? spj_sortindex_ra : spi_sortindex_ra;
    auto WT = twt[ij]; 
    
    const float density_max = ij ?  density_i : density_j;
    float ndt = density_max*dtinterval;

    // Get a random generator. Do not leave without freeing it.
    kokkos_rng_state_t rg = rp.get_state();

    int nmin, nmax; // = std::min(ni, nj);
    size_t Np_c, Np_hc, Np_lc;
    
    // Handle intraspecies.
    if( spi_p == spj_p ) {
	//for simplicity everyone collides
	nj = ni/2;
	ni = ni - nj; //ni may be nj+1
	j0 = i0 + ni;
	nmin = ni<nj ? ni : nj;
	nmax = ni>nj ? ni : nj;
	Np_lc = nmin;
	Np_hc = nmax;
	Np_c  = Np_hc > Np_lc ? Np_hc : Np_lc;	
    }else{
	nmin = ij ? nj : ni; // note that in general the number of the other species can be >=< nmin
	nmax = ij ? ni : nj;	

    // Allocate one integer in team scratch memory (slot 1 is used for the shared integer).
    typedef Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> scratch_int_view_t;
    scratch_int_view_t team_first(team.team_scratch(1), 1);
    // Initialize the shared integer to a sentinel value (-1)
    Kokkos::single(Kokkos::PerTeam(team),
		   [&]() {
		       team_first(0) = -1;
		   });
    team.team_barrier();

    float cumulative = 0.0f;
    // printf("nmin=%d, nmax=%d, WT=%f, ni=%d, nj=%d, mi=%e, mj=%e, deni=%e, denj=%e, ij=%d\n", nmin, nmax, WT, ni, nj, m_i, m_j,density_i,density_j,ij);
    // Only one thread per team does the serial scan.
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      for (int j = 0; j < nmax; j++) {
        // Map the local index j to a global index.
        int i = sph_sortindex_ra(i0 + j);
	auto wp = sph_p(i, particle_var::w);
        cumulative += wp;  // accumulate weight
	// printf("j=%d,wp=%.17f,cum=%.17f\n",j,wp, cumulative);
        // Check if cumulative sum exceeds WT.
	if (cumulative == WT) {
	    team_first(0) = j+1;
        }else if (cumulative > WT) {
	    float r = rg.frand(0, 1.0);
	    int candidate = (cumulative - r * wp < WT) ? j+2 : j+1;
	    team_first(0) = candidate;
          break;
        }
      }
    });
    // Make sure all team threads see the updated local_first.
    team.team_barrier();    
    
    /*
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, nmax),
    [&] (const int j, float & update, const bool final) {
	const int i = sph_sortindex_ra(i0 + j);
	float wp = sph_p(i, particle_var::w);     // current weight
	update += wp;
	// In the final pass, if the cumulative sum exceeds WT, record this index.
	if (final && update > WT) { //The "Final" Pass
	    float r = rg.frand(0, 1.0);
	    int candidate = (update - r * wp < WT) ? j+1 : j;
	    if (candidate < 0) candidate = 0;  // guard against negative index
	    // Update team_first(0) using an atomic operation.
	    // We want the smallest j among all threads where update exceeds WT.
	    // If team_first(0) is still -1 or j is smaller than its current value, update it.
	    Kokkos::atomic_min(&team_first(0), candidate);
	}
    });    
    // Synchronize to make sure all threads see the updated team_first.
    team.team_barrier();
    */
    Np_lc = nmin;
    Np_hc = team_first(0);
    Np_c  = Np_hc > Np_lc ? Np_hc : Np_lc;
    }
    // printf("Np_lc=%d, Np_hc=%d, Np_c=%d, i0=%d, j0=%d\n", (int) Np_lc, (int) Np_hc, (int) Np_c, i0, j0);
    gmomType26 Dm;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, Np_c),
			    [&](const int c, gmomType26 &lsum) {
	 int i1 = c; 
	 int i2 = c % nmin; //it may be possible that c > nmin
	 int i = sph_sortindex_ra(i0 + i1);
	 int j = spl_sortindex_ra(j0 + i2);
	 float up[8] = { sph_p(i, particle_var::w ),
			 sph_p(i, particle_var::ux),
			 sph_p(i, particle_var::uy),
			 sph_p(i, particle_var::uz),
			 spl_p(j, particle_var::w ),
			 spl_p(j, particle_var::ux),
			 spl_p(j, particle_var::uy),
			 spl_p(j, particle_var::uz)};
	 float wp, ux, uy, uz;
	 if(c < Np_hc) {
	     wp = up[0];
	     ux = up[1];
	     uy = up[2];
	     uz = up[3];
	     lsum.v[0] += wp;
	     lsum.v[1] += wp*ux;
	     lsum.v[2] += wp*uy;
	     lsum.v[3] += wp*uz;
	     lsum.v[4] += wp*ux*ux;
	     lsum.v[5] += wp*uy*uy;
	     lsum.v[6] += wp*uz*uz;	     
	 }
	 if(c < Np_lc) {
	     wp = up[4];
	     ux = up[5];
	     uy = up[6];
	     uz = up[7];
	     lsum.v[13] += wp;
	     lsum.v[14] += wp*ux;
	     lsum.v[15] += wp*uy;
	     lsum.v[16] += wp*uz;
	     lsum.v[17] += wp*ux*ux;
	     lsum.v[18] += wp*uy*uy;
	     lsum.v[19] += wp*uz*uz;	     
	 }
	 
	 binary_collision(mu, mu_i, mu_j, up, model, rg, ndt);

	 if(c < Np_hc) {
	     wp = up[0];
	     ux = up[1];
	     uy = up[2];
	     uz = up[3];
	     sph_p(i, particle_var::ux) = ux;
	     sph_p(i, particle_var::uy) = uy;
	     sph_p(i, particle_var::uz) = uz;
	     lsum.v[7] += wp*ux;
	     lsum.v[8] += wp*uy;
	     lsum.v[9] += wp*uz;
	     lsum.v[10] += wp*ux*ux;
	     lsum.v[11] += wp*uy*uy;
	     lsum.v[12] += wp*uz*uz;
	     
	 }
	 if(c < Np_lc) {
	     wp = up[4];
	     ux = up[5];
	     uy = up[6];
	     uz = up[7];
	     spl_p(j, particle_var::ux) = ux;
	     spl_p(j, particle_var::uy) = uy;
	     spl_p(j, particle_var::uz) = uz;
	     lsum.v[20] += wp*ux;
	     lsum.v[21] += wp*uy;
	     lsum.v[22] += wp*uz;
	     lsum.v[23] += wp*ux*ux;
	     lsum.v[24] += wp*uy*uy;
	     lsum.v[25] += wp*uz*uz;
	 }
			    }, Dm);	 
    
    //correcting conservation (only for high-density species)
    float tot_ms = mh*Dm.v[0];    
    float V0_x   = (mh*Dm.v[1] + ml*Dm.v[14] - ml*Dm.v[20]) / tot_ms;
    float V0_y   = (mh*Dm.v[2] + ml*Dm.v[15] - ml*Dm.v[21]) / tot_ms;
    float V0_z   = (mh*Dm.v[3] + ml*Dm.v[16] - ml*Dm.v[22]) / tot_ms;
    float tot_En = 0.5 * (mh*( Dm.v[4] + Dm.v[5] + Dm.v[6] ) + ml*( Dm.v[17] + Dm.v[18] + Dm.v[19] ) - ml*( Dm.v[23] + Dm.v[24] + Dm.v[25] ));


    float Vp_x   = mh*Dm.v[7] / tot_ms;
    float Vp_y   = mh*Dm.v[8] / tot_ms;
    float Vp_z   = mh*Dm.v[9] / tot_ms;
    float tot_Ep = 0.5 * mh*( Dm.v[10] + Dm.v[11] + Dm.v[12] );
    
    float alph =
	sqrt( ( tot_En - 0.5 * tot_ms * ( V0_x * V0_x + V0_y * V0_y + V0_z * V0_z ) ) /
	      ( tot_Ep - 0.5 * tot_ms * ( Vp_x * Vp_x + Vp_y * Vp_y + Vp_z * Vp_z ) ) );

    auto _correction = KOKKOS_LAMBDA( const size_t c )
	{
	 int i1 = c; 
	 int i = sph_sortindex_ra(i0 + i1);

	 auto &ux_i = sph_p(i, particle_var::ux);
	 auto &uy_i = sph_p(i, particle_var::uy);
	 auto &uz_i = sph_p(i, particle_var::uz);
	 
	 ux_i = V0_x + alph * ( ux_i - Vp_x );
	 uy_i = V0_y + alph * ( uy_i - Vp_y );
	 uz_i = V0_z + alph * ( uz_i - Vp_z );
	};

    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, Np_hc ), _correction );
    

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
  void binary_collision (
    const float mu,
    const float mu_i,
    const float mu_j,
    float (&up)[8],
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

    float wj  = up[4];
    float ujx = up[5];
    float ujy = up[6];
    float ujz = up[7];


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
    /*
    // Monte-Carlo collision test
    if( MonteCarlo ) {

      // TODO : CPU VPIC warned when dd*t1 > 1 for under-resolved collisions.
      //        Would this be useful?
      dd = model.cross_section(rg, t2, t1);
      if( rg.frand() > dd*t1 ) return;

    }
    */
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

    up[1] = (uix + mu_i*stack[0])*rr + cmx;
    up[2] = (uiy + mu_i*stack[1])*rr + cmy;
    up[3] = (uiz + mu_i*stack[2])*rr + cmz;
               
    up[5] = (ujx - mu_j*stack[0])*rr + cmx;
    up[6] = (ujy - mu_j*stack[1])*rr + cmy;
    up[7] = (ujz - mu_j*stack[2])*rr + cmz;
    
  }

};

#endif
