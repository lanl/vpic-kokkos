#include "vpic.h"

#define FAK field_array->kernel

/*
void
vpic_simulation::initialize( int argc,
                             char **argv ) {

  double err;
  species_t * sp;

  // Initialize Kokkos
  Kokkos::initialize( argc, argv );

  // Call the user initialize the simulation

  TIC user_initialization( argc, argv ); TOC( user_initialization, 1 );
//  user_initialization( argc, argv );

  // We want to call this once the neighbor is done
  // TODO: general grid handle

  auto g = species_list->g;
  auto nfaces_per_voxel = 6;
  g->init_kokkos_grid(nfaces_per_voxel*g->nv);

  // Do some consistency checks on user initialized fields

  if( rank()==0 ) MESSAGE(( "Checking interdomain synchronization" ));
  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));

  if( rank()==0 ) MESSAGE(( "Checking magnetic field divergence" ));
  TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
  TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
  TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );

  // Load fields not initialized by the user

  if( rank()==0 ) MESSAGE(( "Initializing radiation damping fields" ));
  TIC FAK->compute_curl_b( field_array ); TOC( compute_curl_b, 1 );

  if( rank()==0 ) MESSAGE(( "Initializing bound charge density" ));
  TIC FAK->clear_rhof( field_array ); TOC( clear_rhof, 1 );
  LIST_FOR_EACH( sp, species_list ) TIC accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );
  TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
  TIC FAK->compute_rhob( field_array ); TOC( compute_rhob, 1 );

  // Internal sanity checks

  if( rank()==0 ) MESSAGE(( "Checking electric field divergence" ));

  TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
  TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
  TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );

  if( rank()==0 ) MESSAGE(( "Rechecking interdomain synchronization" ));
  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));

  KOKKOS_TIC();
  KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE(species_list);
  KOKKOS_TOCN( PARTICLE_DATA_MOVEMENT, 1);

  KOKKOS_TIC(); // Time this data movement
  KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE(interpolator_array);
  KOKKOS_TOCN( INTERPOLATOR_DATA_MOVEMENT, 1);

  KOKKOS_TIC(); // Time this data movement
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE(accumulator_array);
  KOKKOS_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

  KOKKOS_TIC(); // Time this data movement
  KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
  KOKKOS_TOCN( FIELD_DATA_MOVEMENT, 1);
  
  // Do some consistency checks on user initialized fields

//  if( rank()==0 ) MESSAGE(( "Checking interdomain synchronization" ));
////  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
//  TIC err = FAK->synchronize_tang_e_norm_b_kokkos( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
//  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));
//
//  if( rank()==0 ) MESSAGE(( "Checking magnetic field divergence" ));
////  TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
//  TIC FAK->compute_div_b_err_kokkos( field_array ); TOC( compute_div_b_err, 1 );
////  TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
//  TIC err = FAK->compute_rms_div_b_err_kokkos( field_array ); TOC( compute_rms_div_b_err, 1 );
//  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
////  TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );
//  TIC FAK->clean_div_b_kokkos( field_array ); TOC( clean_div_b, 1 );
//
//  // Load fields not initialized by the user
//
//  if( rank()==0 ) MESSAGE(( "Initializing radiation damping fields" ));
////  TIC FAK->compute_curl_b( field_array ); TOC( compute_curl_b, 1 );
//  TIC FAK->compute_curl_b_kokkos( field_array ); TOC( compute_curl_b, 1 );
//
//  if( rank()==0 ) MESSAGE(( "Initializing bound charge density" ));
////  TIC FAK->clear_rhof( field_array ); TOC( clear_rhof, 1 );
//  TIC FAK->clear_rhof_kokkos( field_array ); TOC( clear_rhof, 1 );
////  LIST_FOR_EACH( sp, species_list ) TIC accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );
//  LIST_FOR_EACH( sp, species_list ) TIC k_accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );
////  TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
//  TIC FAK->k_synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
////  TIC FAK->compute_rhob( field_array ); TOC( compute_rhob, 1 );
//  TIC FAK->compute_rhob_kokkos( field_array ); TOC( compute_rhob, 1 );
//
//  // Internal sanity checks
//
//  if( rank()==0 ) MESSAGE(( "Checking electric field divergence" ));
//
//  TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
//  TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
//  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
//  TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );
//
//  if( rank()==0 ) MESSAGE(( "Rechecking interdomain synchronization" ));
//  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
//  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));

  if( species_list ) {

    if( rank()==0 ) MESSAGE(( "Uncentering particles" ));
    TIC load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );
  }
  LIST_FOR_EACH( sp, species_list ) {
      KOKKOS_TIC();
      uncenter_p( sp, interpolator_array );
      KOKKOS_TOC( uncenter_p, 1 );
  }

//  KOKKOS_TIC(); // Time this data movement
//  KOKKOS_COPY_INTERPOLATOR_MEM_TO_HOST(interpolator_array);
//  KOKKOS_TOC( INTERPOLATOR_DATA_MOVEMENT, 1);
//
//  KOKKOS_TIC();
//  KOKKOS_COPY_PARTICLE_MEM_TO_HOST(species_list);
//  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);

  if( rank()==0 ) MESSAGE(( "Performing initial diagnostics" ));

  // Let the user to perform diagnostics on the initial condition
  // field(i,j,k).jfx, jfy, jfz will not be valid at this point.
  TIC user_diagnostics(); TOC( user_diagnostics, 1 );

  if( rank()==0 ) MESSAGE(( "Initialization complete" ));
  update_profile( rank()==0 ); // Let the user know how initialization went
}
*/

void
vpic_simulation::initialize( int argc,
                             char **argv ) {

  double err;
  species_t * sp;

  // Initialize Kokkos
  Kokkos::initialize( argc, argv );

  // Call the user initialize the simulation

#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" user_initialization");
#endif
  TIC user_initialization( argc, argv ); TOC( user_initialization, 1 );
//  user_initialization( argc, argv );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // Do some consistency checks on user initialized fields

  if( rank()==0 ) MESSAGE(( "Checking interdomain synchronization" ));
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize synchronize_tang_e_norm_b 1");
#endif
  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));

  if( rank()==0 ) MESSAGE(( "Checking magnetic field divergence" ));
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize compute_div_b_err");
#endif
  TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize compute_rms_div_b_err");
#endif
  TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize clean_div_b");
#endif
  TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // Load fields not initialized by the user

  if( rank()==0 ) MESSAGE(( "Initializing radiation damping fields" ));
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize compute_curl_b");
#endif
  TIC FAK->compute_curl_b( field_array ); TOC( compute_curl_b, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  if( rank()==0 ) MESSAGE(( "Initializing bound charge density" ));
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize clear_rhof");
#endif
  TIC FAK->clear_rhof( field_array ); TOC( clear_rhof, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize accumulate_rho_p");
#endif
  LIST_FOR_EACH( sp, species_list ) TIC accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize synchronize_rho");
#endif
  TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize compute_rhob");
#endif
  TIC FAK->compute_rhob( field_array ); TOC( compute_rhob, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // Internal sanity checks

  if( rank()==0 ) MESSAGE(( "Checking electric field divergence" ));

#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize compute_div_e_err");
#endif
  TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize compute_rms_div_e_err");
#endif
  TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize clean_div_e");
#endif
  TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  if( rank()==0 ) MESSAGE(( "Rechecking interdomain synchronization" ));
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize synchronize_tang_e_norm_b 2");
#endif
  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));

  // We want to call this once the neighbor is done
  // TODO: general grid handle

  auto g = species_list->g;
  auto nfaces_per_voxel = 6;
  g->init_kokkos_grid(nfaces_per_voxel*g->nv);

#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize Particle_data_movement");
#endif
  KOKKOS_TIC();
  KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE(species_list);
  KOKKOS_TOCN( PARTICLE_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize Interpolator_data_movement");
#endif
  KOKKOS_TIC(); // Time this data movement
  KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE(interpolator_array);
  KOKKOS_TOCN( INTERPOLATOR_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize Accumulator_data_movement");
#endif
  KOKKOS_TIC(); // Time this data movement
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE(accumulator_array);
  KOKKOS_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  if( species_list ) {
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize Field_data_movement");
#endif
    KOKKOS_TIC(); // Time this data movement
    KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
    KOKKOS_TOCN( FIELD_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

    if( rank()==0 ) MESSAGE(( "Uncentering particles" ));
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize load_interpolator_array");
#endif
    TIC load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif
  }
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize uncenter_p");
#endif
  LIST_FOR_EACH( sp, species_list ) {
      KOKKOS_TIC();
      uncenter_p( sp, interpolator_array );
      KOKKOS_TOC( uncenter_p, 1 );
  }
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

//  KOKKOS_TIC(); // Time this data movement
//  KOKKOS_COPY_INTERPOLATOR_MEM_TO_HOST(interpolator_array);
//  KOKKOS_TOC( INTERPOLATOR_DATA_MOVEMENT, 1);
//
//  KOKKOS_TIC();
//  KOKKOS_COPY_PARTICLE_MEM_TO_HOST(species_list);
//  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);

  if( rank()==0 ) MESSAGE(( "Performing initial diagnostics" ));

  // Let the user to perform diagnostics on the initial condition
  // field(i,j,k).jfx, jfy, jfz will not be valid at this point.
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" initialize user_diagnostics");
#endif
  TIC user_diagnostics(); TOC( user_diagnostics, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  if( rank()==0 ) MESSAGE(( "Initialization complete" ));
  update_profile( rank()==0 ); // Let the user know how initialization went
}



void
vpic_simulation::finalize( void ) {
  barrier();
  //Kokkos::finalize();
  update_profile( rank()==0 );
}

