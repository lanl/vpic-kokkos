#include "vpic.h"
#include "../particle_operations/compress.h"
#include "../particle_operations/sort.h"
#include <Kokkos_Sort.hpp>

#define FAK field_array->kernel

int vpic_simulation::advance(void)
{

  species_t *sp;
  double err;

  //printf("%d: Step %d \n", rank(), step());

  // Use default policy, for now
  ParticleCompressor<> compressor;
  ParticleSorter<> sorter;

  // Determine if we are done ... see note below why this is done here
  if( num_step>0 && step()>=num_step ) return 0;

  KOKKOS_TIC();

  // Sort the particles for performance if desired.
  LIST_FOR_EACH( sp, species_list )
  {
      if( (sp->sort_interval>0) && ((step() % sp->sort_interval)==0) )
      {
          if( rank()==0 ) MESSAGE(( "Performance sorting \"%s\"", sp->name ));
          //TIC sort_p( sp ); TOC( sort_p, 1 );
          sorter.sort( sp->k_p_d, sp->k_p_i_d, sp->np, accumulator_array->na);
      }
  }

  KOKKOS_TOC( sort_particles, 1);

  // At this point, fields are at E_0 and B_0 and the particle positions
  // are at r_0 and u_{-1/2}.  Further the mover lists for the particles should
  // empty and all particles should be inside the local computational domain.
  // Advance the particle lists.

  // HOST - Touches accumulators
  if( species_list )
  {
    // TIC clear_accumulator_array( accumulator_array ); TOC( clear_accumulators, 1 );
    TIC clear_accumulator_array_kokkos( accumulator_array ); TOC( clear_accumulators, 1 );
  }

  // Note: Particles should not have moved since the last performance sort
  // when calling collision operators.
  // FIXME: Technically, this placement of the collision operators only
  // yields a first order accurate Trotter factorization (not a second
  // order accurate factorization).

  if( collision_op_list )
  {
      Kokkos::abort("Collision is not supported");
      TIC apply_collision_op_list( collision_op_list ); TOC( collision_model, 1 );
  }

  // TODO: implement
  //TIC user_particle_collisions(); TOC( user_particle_collisions, 1 );

  // DEVICE function - Touches particles, particle movers, accumulators, interpolators
  LIST_FOR_EACH( sp, species_list )
  {
      // Now Times internally
      advance_p( sp, accumulator_array, interpolator_array );
  }

  // Reduce accumulator contributions into the device array
  KOKKOS_TIC();
  Kokkos::Experimental::contribute(accumulator_array->k_a_d, accumulator_array->k_a_sa);
  accumulator_array->k_a_sa.reset_except(accumulator_array->k_a_d);
  KOKKOS_TOC( accumulator_contributions, 1);

  // Copy particle movers back to host
  KOKKOS_TIC();
  KOKKOS_COPY_MOVER_MEM_TO_HOST(species_list);
  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);

  // Because the partial position push when injecting aged particles might
  // place those particles onto the guard list (boundary interaction) and
  // because advance_p requires an empty guard list, particle injection must
  // be done after advance_p and before guard list processing. Note:
  // user_particle_injection should be a stub if species_list is empty.

  // Probably needs to be on host due to user particle injection
  // May not touch memory?
  if( emitter_list )
  {
    TIC apply_emitter_list( emitter_list ); TOC( emission_model, 1 );
  }

  if((particle_injection_interval>0) && ((step() % particle_injection_interval)==0)) {
      if(!kokkos_particle_injection) {
          KOKKOS_TIC();
          KOKKOS_COPY_PARTICLE_MEM_TO_HOST(species_list);
          KOKKOS_TOC(PARTICLE_DATA_MOVEMENT, 1);
      }
      TIC user_particle_injection(); TOC( user_particle_injection, 1 );
      if(!kokkos_particle_injection) {
          KOKKOS_TIC();
          KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE(species_list);
          KOKKOS_TOC(PARTICLE_DATA_MOVEMENT, 1);
      }
  }

  bool accumulate_in_place = false; // This has to be outside the scoped timing block

  KOKKOS_TIC(); // Time this data movement
  // This could technically be done once per simulation, not every timestep
  if (accumulator_array->k_a_h.data() == accumulator_array->k_a_d.data() )
  {
      accumulate_in_place = true;
  }
  else {
      // Zero out the host accumulator
      Kokkos::deep_copy(accumulator_array->k_a_h, 0.0f);
  }

  KOKKOS_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

  // This should be after the emission and injection to allow for the
  // possibility of thread parallelizing these operations

  // HOST
  // Touches accumulator memory
  //  if( species_list )
  //    TIC reduce_accumulator_array( accumulator_array ); TOC( reduce_accumulators, 1 );
  //    TIC reduce_accumulator_array_kokkos( accumulator_array ); TOC( reduce_accumulators, 1 );

  // At this point, most particle positions are at r_1 and u_{1/2}. Particles
  // that had boundary interactions are now on the guard list. Process the
  // guard lists. Particles that absorbed are added to rhob (using a corrected
  // local accumulation).

  // HOST - Touches particle copies, particle_movers, particle_injectors,
  // accumulators (move_p), neighbors
  TIC
    for( int round=0; round<num_comm_round; round++ )
    {
      //boundary_p( particle_bc_list, species_list, field_array, accumulator_array );
      boundary_p_kokkos( particle_bc_list, species_list, field_array, accumulator_array );
    }
  TOC( boundary_p, num_comm_round );

  // currently the recv particles are in particles_recv, not particle_copy
  KOKKOS_TIC();
  LIST_FOR_EACH( sp, species_list )
  {
        auto pr_h_subview = Kokkos::subview(sp->k_pr_h,
                std::make_pair(0, sp->num_to_copy), Kokkos::ALL);
        auto pri_h_subview = Kokkos::subview(sp->k_pr_i_h,
                std::make_pair(0, sp->num_to_copy));

        auto pc_h_subview = Kokkos::subview(sp->k_pc_h,
                std::make_pair(0, sp->num_to_copy), Kokkos::ALL);
        auto pci_h_subview = Kokkos::subview(sp->k_pc_i_h,
                std::make_pair(0, sp->num_to_copy));

        Kokkos::deep_copy(pc_h_subview, pr_h_subview);
        Kokkos::deep_copy(pci_h_subview, pri_h_subview);
  }

  KOKKOS_TOCN( PARTICLE_DATA_MOVEMENT, 1);

  // Boundary_p calls move_p, so we may need to deal with the current
  // If we didn't accumulate for the host in place (as in the GPU case), do so
  if ( accumulate_in_place == false)
  {
      // Update device so we can pull it all the way back to the host
      KOKKOS_TIC(); // Time this data movement
      combine_accumulators( accumulator_array );
      KOKKOS_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);
  }

  // Clean_up once boundary p is done
  // Copy back the right data to GPU
  // Device
  // Touches particles, particle_movers
  LIST_FOR_EACH( sp, species_list )
  {
      KOKKOS_TIC(); // Time this data movement
      const int nm = sp->k_nm_h(0);

      // TODO: this can be hoisted to the end of advance_p if desired
      compressor.compress(
              sp->k_p_d,
              sp->k_p_i_d,
              sp->k_pm_i_d,
              nm,
              sp->np,
              sp
      );

      // Update np now we removed them...
      sp->np -= nm;
      KOKKOS_TOC( BACKFILL, 0);

      auto& particles = sp->k_p_d;
      auto& particles_i = sp->k_p_i_d;

      int num_to_copy = sp->num_to_copy;

      // Copy data for copies back to device
      KOKKOS_TIC();
        auto pc_d_subview = Kokkos::subview(sp->k_pc_d, std::make_pair(0, num_to_copy), Kokkos::ALL);
        auto pci_d_subview = Kokkos::subview(sp->k_pc_i_d, std::make_pair(0, num_to_copy));
        auto pc_h_subview = Kokkos::subview(sp->k_pc_h, std::make_pair(0, num_to_copy), Kokkos::ALL);
        auto pci_h_subview = Kokkos::subview(sp->k_pc_i_h, std::make_pair(0, num_to_copy));
        Kokkos::deep_copy(pc_d_subview, pc_h_subview);
        Kokkos::deep_copy(pci_d_subview, pci_h_subview);
      KOKKOS_TOCN( PARTICLE_DATA_MOVEMENT, 1);

      KOKKOS_TIC(); // Time this data movement
      auto& particle_copy = sp->k_pc_d;
      auto& particle_copy_i = sp->k_pc_i_d;
      int num_to_copy = sp->num_to_copy;
      int np = sp->np;

      // TODO: is it better to catch this during boundary p?
      if (np + sp->num_to_copy > sp->max_np)
      {
          Kokkos::abort("Species overflowed particle storage during boundary exchange");
      }

      // Append it to the particles
      Kokkos::parallel_for("append moved particles", Kokkos::RangePolicy <
              Kokkos::DefaultExecutionSpace > (0, sp->num_to_copy), KOKKOS_LAMBDA
              (int i)
      {
        int npi = np+i; // i goes from 0..n so no need for -1
        //printf("append to %d from %d \n", npi, i);
        particles(npi, particle_var::dx) = particle_copy(i, particle_var::dx);
        particles(npi, particle_var::dy) = particle_copy(i, particle_var::dy);
        particles(npi, particle_var::dz) = particle_copy(i, particle_var::dz);
        particles(npi, particle_var::ux) = particle_copy(i, particle_var::ux);
        particles(npi, particle_var::uy) = particle_copy(i, particle_var::uy);
        particles(npi, particle_var::uz) = particle_copy(i, particle_var::uz);
        particles(npi, particle_var::w)  = particle_copy(i, particle_var::w);
        particles_i(npi) = particle_copy_i(i);
      });

      // Reset this to zero now we've done the write back
      sp->np += num_to_copy;
      sp->num_to_copy = 0;
      KOKKOS_TOC( BACKFILL, 1); // Don't double count
  }

  // This copies over a val for nm, which is a lie
  LIST_FOR_EACH( sp, species_list ) {
      sp->nm = 0;
  }

  // At this point, all particle positions are at r_1 and u_{1/2}, the
  // guard lists are empty and the accumulators on each processor are current.
  // Convert the accumulators into currents.

  // HOST
  // Touches fields and accumulators
  //  TIC FAK->clear_jf( field_array ); TOC( clear_jf, 1 );
  TIC FAK->clear_jf_kokkos( field_array ); TOC( clear_jf, 1 );

  if( species_list )
  {
    TIC unload_accumulator_array_kokkos( field_array, accumulator_array );
    TOC( unload_accumulator, 1 );
  }

  //  TIC FAK->synchronize_jf( field_array ); TOC( synchronize_jf, 1 );
  TIC FAK->k_synchronize_jf( field_array ); TOC( synchronize_jf, 1 );

  // At this point, the particle currents are known at jf_{1/2}.
  // Let the user add their own current contributions. It is the users
  // responsibility to insure injected currents are consistent across domains.
  // It is also the users responsibility to update rhob according to
  // rhob_1 = rhob_0 + div juser_{1/2} (corrected local accumulation) if
  // the user wants electric field divergence cleaning to work.

  if((current_injection_interval>0) && ((step() % current_injection_interval)==0)) {
      if(!kokkos_current_injection) {
          KOKKOS_TIC();
          KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
          KOKKOS_TOC(FIELD_DATA_MOVEMENT, 1);
      }
      TIC user_current_injection(); TOC( user_current_injection, 1 );
      if(!kokkos_current_injection) {
          KOKKOS_TIC();
          KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
          KOKKOS_TOC(FIELD_DATA_MOVEMENT, 1);
      }
  }

  // DEVICE -- Touches fields
  // Half advance the magnetic field from B_0 to B_{1/2}
  KOKKOS_TIC();
  FAK->advance_b( field_array, 0.5 );
  KOKKOS_TOC( advance_b, 1 );

  // Advance the electric field from E_0 to E_1

  // Device - Touches fields
  //  TIC FAK->advance_e( field_array, 1.0 ); TOC( advance_e, 1 );
  TIC FAK->advance_e_kokkos( field_array, 1.0 ); TOC( advance_e, 1 );

  // Let the user add their own contributions to the electric field. It is the
  // users responsibility to insure injected electric fields are consistent
  // across domains.

  if ((field_injection_interval>0) && ((step() % field_injection_interval)==0)) {
      if (!kokkos_field_injection) {
          KOKKOS_TIC();
          KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
          KOKKOS_TOC(FIELD_DATA_MOVEMENT, 1);
      }
      TIC user_field_injection(); TOC( user_field_injection, 1 );
      if (!kokkos_field_injection) {
          KOKKOS_TIC();
          KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
          KOKKOS_TOC(FIELD_DATA_MOVEMENT, 1);
      }
  }

  // Half advance the magnetic field from B_{1/2} to B_1

  // DEVICE
  // Touches fields
  TIC FAK->advance_b( field_array, 0.5 ); TOC( advance_b, 1 );

  // Divergence clean e

  if( (clean_div_e_interval>0) && ((step() % clean_div_e_interval)==0) )
  {
      if( rank()==0 ) MESSAGE(( "Divergence cleaning electric field" ));

      // HOST (Device in rho_p)
      // Touches fields and particles
      // TIC FAK->clear_rhof( field_array ); TOC( clear_rhof,1 );
      TIC FAK->clear_rhof_kokkos( field_array ); TOC( clear_rhof,1 );

      if( species_list )
      {
          KOKKOS_TIC();
          LIST_FOR_EACH( sp, species_list )
          {
              //accumulate_rho_p( field_array, sp ); //TOC( accumulate_rho_p, species_list->id );
              k_accumulate_rho_p( field_array, sp );
          }
          KOKKOS_TOC( accumulate_rho_p, species_list->id );
      }

      // TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
      TIC FAK->k_synchronize_rho( field_array ); TOC( synchronize_rho, 1 );

      // HOST
      // Touches fields
      for( int round=0; round<num_div_e_round; round++ )
      {
          // TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
          TIC FAK->compute_div_e_err_kokkos( field_array ); TOC( compute_div_e_err, 1 );

          if( round==0 || round==num_div_e_round-1 ) {
              // TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
              TIC err = FAK->compute_rms_div_e_err_kokkos( field_array ); TOC( compute_rms_div_e_err, 1 );

              if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
          }
          // TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );
          TIC FAK->clean_div_e_kokkos( field_array ); TOC( clean_div_e, 1 );
      }
  }

  // Divergence clean b
  // HOST
  // Touches fields
  if( (clean_div_b_interval>0) && ((step() % clean_div_b_interval)==0) )
  {
      if( rank()==0 ) MESSAGE(( "Divergence cleaning magnetic field" ));

      for( int round=0; round<num_div_b_round; round++ )
      {
          // TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
          TIC FAK->compute_div_b_err_kokkos( field_array ); TOC( compute_div_b_err, 1 );

          if( round==0 || round==num_div_b_round-1 ) {
              // TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
              TIC err = FAK->compute_rms_div_b_err_kokkos( field_array ); TOC( compute_rms_div_b_err, 1 );
              if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
          }
          TIC FAK->clean_div_b_kokkos( field_array ); TOC( clean_div_b, 1 );
      }
  }

  // Synchronize the shared faces
  // HOST
  // Touches fields
  if( (sync_shared_interval>0) && ((step() % sync_shared_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Synchronizing shared tang e, norm b, rho_b" ));
    // TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
    TIC err = FAK->synchronize_tang_e_norm_b_kokkos( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
    if( rank()==0 ) MESSAGE(( "Domain desynchronization error = %e (arb units)", err ));
  }
  // Fields are updated ... load the interpolator for next time step and
  // particle diagnostics in user_diagnostics if there are any particle
  // species to worry about

  // DEVICE
  // Touches fields, interpolators
  if( species_list ) TIC load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );

  step()++;

  // Print out status
  if( (status_interval>0) && ((step() % status_interval)==0) ) {
      if( rank()==0 ) MESSAGE(( "Completed step %i of %i", step(), num_step ));
      update_profile( rank()==0 );
  }

  // Let the user compute diagnostics
  TIC user_diagnostics(); TOC( user_diagnostics, 1 );

  // "return step()!=num_step" is more intuitive. But if a checkpt
  // saved in the call to user_diagnostics() above, is done on the final step
  // (silly but it might happen), the test will be skipped on the restore. We
  // return true here so that the first call to advance after a restore
  // will act properly for this edge case.

#ifdef DUMP_ENERGIES
  TIC dump_energies("energies.txt", 1); TOC( dump_energies, 1);
#endif

  return 1;
}
