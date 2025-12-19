#ifndef _kokkos_particle_bulk_collision_pipeline_h_
#define _kokkos_particle_bulk_collision_pipeline_h_

#include "../collision_private.h"

// Assumes single precision.
// Chosen as a cutoff < sqrt(FLT_MAX) such that dd/(1+dd*dd) is always in range.
#define TAN_THETA_HALF_MAX 1.30e19f
#define PREVENT_BACKSCATTER(TAN) do  {                                          \
  if(!isfinite(TAN) || (TAN) > TAN_THETA_HALF_MAX ) (TAN) = TAN_THETA_HALF_MAX; \
} while(0)

 
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
    
  // Random access, read-only Views
  // TODO : Does RandomAccess trait really matter?
  k_particle_sortindex_t_ra _spi_sortindex_ra;//, _spj_sortindex_ra;
  k_particle_partition_t_ra _spi_partition_ra;//, _spj_partition_ra;

  particle_bulk_collision_pipeline(
    species_t * spi,
    fluid_species_t * spj,
    double interval,
    kokkos_rng_pool_t& rp,
    field_array_t * field
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
      _spj(spj),
      _rp(rp),
      _field(field)
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
    if( _spi->np      > _spi_sortindex_ra.extent(0) || 
        _spi->g->nv+1 != _spi_partition_ra.extent(0) ){
	printf("_spi->np (=%d) ?= _spi_sortindex_ra.extent(0) (=%d)\n",_spi->np,_spi_sortindex_ra.extent(0));
	printf("_spi->g->nv+1 (=%d) ?= _spi_partition_ra.extent(0) (=%d)\n",_spi->g->nv+1,_spi_partition_ra.extent(0));
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

    */
    
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
    auto const& mi   = _mi;
    auto const& mj   = _mj;
    auto const& mu_i = _mu_i;
    auto const& mu_j = _mu_j;
    auto const& mu = _mu;
    auto const& rdV = _rdV;
    auto const& nx = _nx;
    auto const& ny = _ny;
    auto const& nz = _nz;
    auto const& spi = _spi;
    auto const& spj = _spj;
    auto const& rp  = _rp;
    auto const& spi_n = _spi_n;
    auto const& spi_i = _spi_i;
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

	//        auto j0 = spj_partition_ra(v);
	//        auto nj = spj_partition_ra(v+1) - j0;

        // TODO: convert this to be a more explicit check on if we have work
	//        if( ni <= 0 || nj <= 0 ) return; //Nothing to do
	if( ni <= 0 ) return; //Nothing to do

	//        // Find the real densities.
	//        float density_i = spi_n(v);
	//        float density_j = spj_n(v);

	//        // Compute ndt
	//        const float density_min = density_j > density_i ? density_i : density_j;
	//        const float ndt = density_min*dtinterval;
	const float dt = dtinterval;
	
        // Get a random generator. Do not leave without freeing it.
        kokkos_rng_state_t rg = rp.get_state();
	

	//// Extract fluid variables
	//	const float n_fl = spj_fl(v, fluid_var::den);

	//for each cell
	gmomType Dm; 
	
	Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, ni),
	[&](const int& k, gmomType &lsum) {

	  int i = spi_sortindex_ra(i0 + k);

#ifdef VARIABLE_CHARGE
	  float up[5] =   { spi_p(i, particle_var::w),
                            spi_p(i, particle_var::ux),
                            spi_p(i, particle_var::uy),
                            spi_p(i, particle_var::uz),
			    spi_p(i, particle_var::qp) };
#else
	  float up[4] =   { spi_p(i, particle_var::w),
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

	    if( use_e_field ) {
	    	particle_bulk_collision(mi, mj, mu, mu_i, mu_j, up, spj_fd, model, rg, dt,v);
	    } else {      
	     	particle_bulk_collision(mi, mj, mu, mu_i, mu_j, up, spj_fl, model, rg, dt,v);
	    }
	    
      float ux_i = up[1];
      float uy_i = up[2];
      float uz_i = up[3];
#ifdef VARIABLE_CHARGE
	    qp_i = up[4];
	    spi_p(i, particle_var::qp) = qp_i;
#endif

      // If the particle charge changes via charge exchange, 
      // then decrement fluid density by the particle weight and
      // assign the new kinetic particle velocity to that of the
      // fluid velocity plus a thermal component
      float dn = 0.0;
      float dq = qp_n - qp_i;

      if (model.collision_type == CollisionType::BulkChargeExchange && dq != 0.0) {
        dn = wp * rdV;
        // The new kinetic particle takes the fluid bulk velociy plus a thermal component
        float uj_thermal = sqrt(2.0 * spj_fl(v, fluid_var::tmp) / mj);
        float ux_k = rg.normal(spj_fd(v, fluid_var::ux), uj_thermal);
        float uy_k = rg.normal(spj_fd(v, fluid_var::uy), uj_thermal);
        float uz_k = rg.normal(spj_fd(v, fluid_var::uz), uj_thermal);

        // todo: create kinetic particle
        // todo: decrement fluid momentum and energy based on new kinetic particle...

      } // endif(cex)

	    spi_p(i, particle_var::ux) = ux_i;
	    spi_p(i, particle_var::uy) = uy_i;
	    spi_p(i, particle_var::uz) = uz_i;	  
	    
	    auto dux = ( ux_i - ux_n ) * wp;
	    auto duy = ( uy_i - uy_n ) * wp;
	    auto duz = ( uz_i - uz_n ) * wp;
	    auto den = 0.5 *
		( ( ux_i * ux_i + uy_i * uy_i + uz_i * uz_i ) -
		  ( ux_n * ux_n + uy_n * uy_n + uz_n * uz_n ) ) *
		wp;
    
	    lsum.v[0] += wp;
	    lsum.v[1] += dux;
	    lsum.v[2] += duy;
	    lsum.v[3] += duz;
	    // lsum.v[4] += 0.5*wp*(ux_i*ux_i+uy_i*uy_i+uz_i*uz_i); // mjl: why this instead of den?
      lsum.v[4] += den;
      lsum.v[5] += dn;
      
	    // if(k<10) 	printf("lsum=%e,%e,%e,%e,%e\n",wp,dux,duy,duz,den);
	    // if(k<10) 	printf("lsum=%e,%e,%e,%e,%e\n",lsum.v[0],lsum.v[1],lsum.v[2],lsum.v[3],lsum.v[4]);
	}, Dm);
	// printf("Dm=%e,%e,%e,%e,%e\n",Dm.v[0],Dm.v[1],Dm.v[2],Dm.v[3],Dm.v[4]);
	if (team_member.team_rank() == 0) {
	    // Code that runs once per team leader
	    if( use_e_field ) {
	    	// If we have a field, we upload the moment source to the field.
	     	// Upload the moment source to the field.
        model.upload_moment_src( spj_fd, v, Dm, mi, mj );
	    } else {    
        model.upload_moment_src( spj_fl, v, Dm, mi, mj );   
	    }
	    //printf("check: #msxyz=%e,%e,%e, ens=%e, v=%d\n",spj_fd(v, field_var::sx),spj_fd(v, field_var::sy),spj_fd(v, field_var::sz),spj_fd(v, field_var::se), v);
	    //printf("spj_fl data=%p\n", (void*)spj_fl.data());
	    // printf("[call]  spj_fd data=%p ext=(%zu,%zu) v=%d, FIELD_VAR_COUNT=%d\n",
	    // 	   (void*)spj_fd.data(), spj_fd.extent(0), spj_fd.extent(1), v, FIELD_VAR_COUNT);
	}

        // We *must* free generators.
        rp.free_state(rg);

			 });
    
    // I don't know why we need this, but without it I get an illegal memory
    // access error ... suspicious.
    Kokkos::fence();

  }

  /**
   * @brief Perform a collision between two particles.
   */
  //Must have all struct member types passed in directly to the inline function
  //In terms of notation, all inline functions should not have variables that start
  //with the _ (underscore), because _ is used to indicate a class member before it
  //is caputred by a lambda. One lambda captured, we should refer to the variable
  //as EX: mu not _mu
    template<class view_type, class collision_model>
    //template<class collision_model>
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
    //const k_particles_t&   spi_p,
    //    const k_particles_t&   spj_p,
    const view_type& spj_f,
    //const k_field_t spj_f,
    collision_model& model,
    kokkos_rng_state_t& rg,
    float dt,
    int ii
  )
  {

    float dd, ur, tx, ty, tz, t0, t1, t2, stack[3];
    int d0, d1, d2;

    float uix = up[1];
    float uiy = up[2];
    float uiz = up[3];
    float wi  = up[0];

    float qi = 0;
#ifdef VARIABLE_CHARGE
    qi  = up[4];
#endif

    //    float ujx = spj_p(j, particle_var::ux);
    //    float ujy = spj_p(j, particle_var::uy);
    //    float ujz = spj_p(j, particle_var::uz);
    //    float wj  = spj_p(j, particle_var::w);

    // Extract fluid vars
    float nj_fl, ujx_fl, ujy_fl, ujz_fl, tmp_fl;
    if constexpr (std::is_same<view_type, k_fluid_t>::value) {
	    nj_fl = spj_f(ii, fluid_var::den);
	    ujx_fl = spj_f(ii, fluid_var::ux);
	    ujy_fl = spj_f(ii, fluid_var::uy);
	    ujz_fl = spj_f(ii, fluid_var::uz);
	    tmp_fl = spj_f(ii, fluid_var::tmp);
	  } else if constexpr (std::is_same<view_type, k_field_t>::value) {
	    nj_fl = spj_f(ii, field_var::rhof);
	    ujx_fl = spj_f(ii, field_var::ux);
	    ujy_fl = spj_f(ii, field_var::uy);
	    ujz_fl = spj_f(ii, field_var::uz);
	    tmp_fl = spj_f(ii, field_var::pe)/nj_fl; //nj_fl should be non-zero
	  }
    // printf("nj_fl=%e, ujx_fl=%e, ujy_fl=%e, ujz_fl=%e, pe=%e, tmp_fl=%e\n",
    //   	   nj_fl, ujx_fl, ujy_fl, ujz_fl, spj_f(ii, field_var::pe), tmp_fl);

    float ndt = nj_fl * dt;
    //printf("n=%14.8e, dt=%14.8e, mi=%14.8e\n",nj_fl, dt, mi);
    
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
      //      dd = model.cross_section(rg, t2, t1);
      dd = model.cross_section( rg, qi, ur, t1 );
      if( rg.frand() > dd*t1 ) return;

    }

    // Compute collision angle and coefficient of restitution
    float param[4] = {ur, ujth, ndt/(mi*mi), mi/mj};
    const float rr = model.restitution(rg, param);
    dd = model.tan_theta_half(rg, param);
    PREVENT_BACKSCATTER(dd);

#ifdef VARIABLE_CHARGE
    // To-do: Check if density associated with particle > neutral background density.
    const float dq = model.modify_charge();
    //spi_p(i, particle_var::qp) += dq;
    up[4] += dq;
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

    /* stack = (1 - cos theta) u + |u| sin theta + Tperp */
    stack[0] = (t0*urx + t1*tx) + t2*( ury*tz - urz*ty );
    stack[1] = (t0*ury + t1*ty) + t2*( urz*tx - urx*tz );
    stack[2] = (t0*urz + t1*tz) + t2*( urx*ty - ury*tx );

    up[1] = ujx_fl + (urx + stack[0])*rr;
    up[2] = ujy_fl + (ury + stack[1])*rr;
    up[3] = ujz_fl + (urz + stack[2])*rr;
    
    // Scaled center of mass velocity.
    // t1 = (1-rr);
    // float cmx = t1*(mu_j*uix + mu_i*ujx_fl);
    // float cmy = t1*(mu_j*uiy + mu_i*ujy_fl);
    // float cmz = t1*(mu_j*uiz + mu_i*ujz_fl);

    // // Handle unequal particle weights using detailed balance.
    // t0 = rg.frand(0, 1);

    // TURN OF IF STATEMENT TO COMPILE (THIS CODE WILL BE REPLACED BY GY).
    //    if(wi*t0 <= wj) {
    // spi_p(i, particle_var::ux) = (uix + mu_i*stack[0])*rr + cmx;
    //   spi_p(i, particle_var::uy) = (uiy + mu_i*stack[1])*rr + cmy;
    //   spi_p(i, particle_var::uz) = (uiz + mu_i*stack[2])*rr + cmz;
      //    }

    /*    if(wj*t0 <= wi) {
      spj_p(j, particle_var::ux) = (ujx - mu_j*stack[0])*rr + cmx;
      spj_p(j, particle_var::uy) = (ujy - mu_j*stack[1])*rr + cmy;
      spj_p(j, particle_var::uz) = (ujz - mu_j*stack[2])*rr + cmz;
      }*/

  }

};

#endif
