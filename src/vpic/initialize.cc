#include "vpic.h"

#define FAK field_array->kernel

void
vpic_simulation::initialize( int argc,
                             char **argv ) {

  double err;
  species_t * sp;
  fluid_species_t * fsp;
  
  // Initialize Kokkos
  // Moved to boot servcies
  //Kokkos::initialize( argc, argv );
  kokkos_rng = new kokkos_rng_pool_t(rank()); // seed_entropy will re-seed.
  
  // Hybrid-VPIC specific parameter defaults
  grid->eta           = 0.;
  grid->hypereta      = 0.;
  grid->nsm           = 0;
  grid->nsmb          = 0;
  grid->nsub          = 1;
  grid->den_floor_ohm = 0.;
  grid->den_floor_pe  = 0.;
#ifdef HYB_USE_SEPARATE_PE
  grid->eos_gamma     = 5./3.;
#else
  grid->eos_gamma     = 1.;
#endif
  grid->eos_den       = 1.;
  grid->kappa         = 0;

  // Call the user initialize the simulation

  TIC user_initialization( argc, argv ); TOC( user_initialization, 1 );
//  user_initialization( argc, argv );

  dump_strategy = new_dump_strategy(dump_strategy_id, this);

 // -----------------------------------------------
  // Sync everything host -> device

  if( rank()==0 ) MESSAGE(( "Initializing particles, B field on device from host" ));

  // We want to call this once the neighbor is done
  auto g = species_list->g;
  auto nfaces_per_voxel = 6;
  g->init_kokkos_grid(nfaces_per_voxel*g->nv);

  KOKKOS_TIC();
  LIST_FOR_EACH( sp, species_list ) {
    sp->copy_to_device();
  }
  KOKKOS_TOCN( PARTICLE_DATA_MOVEMENT, 1);

  // Hybrid - later we will compute interpolator coeffs on device, no need to sync w/host
  //KOKKOS_TIC();
  //interpolator_array->copy_to_device();
  //KOKKOS_TOCN( INTERPOLATOR_DATA_MOVEMENT, 1);

  // Hybrid - not using accumulators
  //KOKKOS_TIC();
  //FAK->k_reduce_jf(field_array);
  //KOKKOS_TOC( JF_ACCUM_DATA_MOVEMENT, 1);

  // Hybrid - jf,rhof and jfold,rhofold not populated with valid data yet
  KOKKOS_TIC();
  field_array->copy_to_device();
  KOKKOS_TOCN( FIELD_DATA_MOVEMENT, 1);

  // -----------------------------------------------
  // Accumulate rhof,jf on device; compute E(t=0) on device

  if( rank()==0 ) MESSAGE(( "Initializing rho, J, E fields on device" ));

  // Initialize jf,rhof at t=0 (ghosts bad)
  // jf,rhof_old are garbage
  TIC FAK->clear_jf_kokkos( field_array ); TOC( clear_jf, 1 );
  LIST_FOR_EACH( sp, species_list ) TIC k_accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );

  // Fix jf,rhof ghosts
  // E,B will be garbage because jf,rhof_old not set
#ifdef HYB_USE_SEPARATE_PE
  FAK->hyb_init(field_array,0);
#else
  FAK->advance_b(field_array,0);
#endif

  // Initialize jf,rhof_old at t=0
  // Re-initialize jf,rhof at t=0 (ghosts bad)
  TIC FAK->clear_jf_kokkos( field_array ); TOC( clear_jf, 1 );
  LIST_FOR_EACH( sp, species_list ) TIC k_accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );

  // Fix jf,rhof ghosts
  // E,B will now be valid
#ifdef HYB_USE_SEPARATE_PE
  FAK->hyb_init(field_array,0);
#else
  FAK->advance_b(field_array,0);
#endif

  // -----------------------------------------------
  // Setup remaining device data for evolution loop

  if( rank()==0 ) MESSAGE(( "Initializing interpolators" ));
  if( species_list ) {
    TIC load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );
  }

  if( rank()==0 ) MESSAGE(( "Uncentering particles" ));
  LIST_FOR_EACH( sp, species_list ) {
    KOKKOS_TIC();
    uncenter_p( sp, interpolator_array );
    KOKKOS_TOC( uncenter_p, 1 );
  }

  // Let the user to perform diagnostics on the initial condition
  if( rank()==0 ) MESSAGE(( "Performing initial diagnostics" ));
  TIC user_diagnostics(); TOC( user_diagnostics, 1 );

  if( rank()==0 ) MESSAGE(( "Initialization complete" ));
  update_profile_meanminmax( rank()==status_timers_rank ); // Let the user know how initialization went
}


void
vpic_simulation::finalize( void ) {
  barrier();
  //Kokkos::finalize();
  update_profile_meanminmax( rank()==status_timers_rank );
  barrier();  // try to keep stdout/stderr from different ranks in a sensible order (see end of deck/main.cc:main(...))
}
