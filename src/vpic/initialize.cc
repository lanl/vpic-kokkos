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

//  // Do some consistency checks on user initialized fields
//
//  if( rank()==0 ) MESSAGE(( "Checking interdomain synchronization" ));
//  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
//  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));
//
//  if( rank()==0 ) MESSAGE(( "Checking magnetic field divergence" ));
//  TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
//  TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
//  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
//  TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );
//
//  // Load fields not initialized by the user
//
//  if( rank()==0 ) MESSAGE(( "Initializing radiation damping fields" ));
//  TIC FAK->compute_curl_b( field_array ); TOC( compute_curl_b, 1 );
//
//  if( rank()==0 ) MESSAGE(( "Initializing bound charge density" ));
//  TIC FAK->clear_rhof( field_array ); TOC( clear_rhof, 1 );
//  LIST_FOR_EACH( sp, species_list ) TIC accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );
//  TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
//  TIC FAK->compute_rhob( field_array ); TOC( compute_rhob, 1 );
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

  if( rank()==0 ) MESSAGE(( "Checking interdomain synchronization" ));
//  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
  TIC err = FAK->synchronize_tang_e_norm_b_kokkos( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));

  if( rank()==0 ) MESSAGE(( "Checking magnetic field divergence" ));
//  TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
  TIC FAK->compute_div_b_err_kokkos( field_array ); TOC( compute_div_b_err, 1 );
//  TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
  TIC err = FAK->compute_rms_div_b_err_kokkos( field_array ); TOC( compute_rms_div_b_err, 1 );
  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
//  TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );
  TIC FAK->clean_div_b_kokkos( field_array ); TOC( clean_div_b, 1 );

  // Load fields not initialized by the user

  if( rank()==0 ) MESSAGE(( "Initializing radiation damping fields" ));
//  TIC FAK->compute_curl_b( field_array ); TOC( compute_curl_b, 1 );
  TIC FAK->compute_curl_b_kokkos( field_array ); TOC( compute_curl_b, 1 );

  if( rank()==0 ) MESSAGE(( "Initializing bound charge density" ));
//  TIC FAK->clear_rhof( field_array ); TOC( clear_rhof, 1 );
  TIC FAK->clear_rhof_kokkos( field_array ); TOC( clear_rhof, 1 );
//  LIST_FOR_EACH( sp, species_list ) TIC accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );
  LIST_FOR_EACH( sp, species_list ) TIC k_accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );
//  TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
  TIC FAK->k_synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
//  TIC FAK->compute_rhob( field_array ); TOC( compute_rhob, 1 );
  TIC FAK->compute_rhob_kokkos( field_array ); TOC( compute_rhob, 1 );

  // Internal sanity checks

  if( rank()==0 ) MESSAGE(( "Checking electric field divergence" ));

  TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
  TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
  TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );

  if( rank()==0 ) MESSAGE(( "Rechecking interdomain synchronization" ));
  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));

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
  // Moved to boot servcies
  //Kokkos::initialize( argc, argv );

  // Call the user initialize the simulation

  TIC user_initialization( argc, argv ); TOC( user_initialization, 1 );
//  user_initialization( argc, argv );

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

  // We want to call this once the neighbor is done
  // TODO: general grid handle

  auto g = species_list->g;
  auto nfaces_per_voxel = 6;
  g->init_kokkos_grid(nfaces_per_voxel*g->nv);

  KOKKOS_TIC();
  KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE(species_list);
  KOKKOS_TOCN( PARTICLE_DATA_MOVEMENT, 1);

  KOKKOS_TIC(); // Time this data movement
  KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE(interpolator_array);
  KOKKOS_TOCN( INTERPOLATOR_DATA_MOVEMENT, 1);

  KOKKOS_TIC(); // Time this data movement
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE(accumulator_array);
  KOKKOS_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

  if( species_list ) {
    KOKKOS_TIC(); // Time this data movement
    KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
    KOKKOS_TOCN( FIELD_DATA_MOVEMENT, 1);

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


void
vpic_simulation::finalize( void ) {
  barrier();
  //Kokkos::finalize();
  update_profile( rank()==0 );
}

