/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Heavily revised and extended from earlier V4PIC versions
 *
 */

#include "vpic.h"
#include "../particle_operations/compress.h"
#include "../particle_operations/sort.h"
#include <Kokkos_Sort.hpp>
#include <Kokkos_DualView.hpp>

#define FAK field_array->kernel

int vpic_simulation::advance(void) {
  species_t *sp;
  double err;

  //printf("%d: Step %d \n", rank(), step());

  // Use default policy, for now
  ParticleCompressor<> compressor;
  ParticleSorter<> sorter;

  // Determine if we are done ... see note below why this is done here

  if( num_step>0 && step()>=num_step ) return 0;

  // Sort the particles for performance if desired.
  LIST_FOR_EACH( sp, species_list )
  {
      if( (sp->sort_interval>0) && ((step() % sp->sort_interval)==0) )
      {
          if( rank()==0 ) MESSAGE(( "Performance sorting \"%s\"", sp->name ));
          //TIC sort_p( sp ); TOC( sort_p, 1 );
          UNSAFE_TIC();
          sorter.sort( sp->k_p_d, sp->k_p_i_d, sp->np, accumulator_array->na);
          UNSAFE_TOC( sort_particles, 1);

      }
  }

  // At this point, fields are at E_0 and B_0 and the particle positions
  // are at r_0 and u_{-1/2}.  Further the mover lists for the particles should
  // empty and all particles should be inside the local computational domain.
  // Advance the particle lists.

// HOST
// Touches accumulators
  if( species_list )
//    TIC clear_accumulator_array( accumulator_array ); TOC( clear_accumulators, 1 );
    TIC clear_accumulator_array_kokkos( accumulator_array ); TOC( clear_accumulators, 1 );
  UNSAFE_TIC();
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_HOST(accumulator_array);
  UNSAFE_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

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

  //TIC user_particle_collisions(); TOC( user_particle_collisions, 1 );

  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE(accumulator_array);
  UNSAFE_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);
  UNSAFE_TIC();
  KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE(interpolator_array);
  UNSAFE_TOC( INTERPOLATOR_DATA_MOVEMENT, 1);
//  KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE();

  //int lna = 180;

// DEVICE function
// Touches particles, particle movers, accumulators, interpolators
  LIST_FOR_EACH( sp, species_list )
  {
      TIC advance_p( sp, accumulator_array, interpolator_array ); TOC( advance_p, 1 );
  }

  UNSAFE_TIC();
  Kokkos::Experimental::contribute(accumulator_array->k_a_d, accumulator_array->k_a_sa);
  accumulator_array->k_a_sa.reset_except(accumulator_array->k_a_d);
  UNSAFE_TOC( accumulator_contributions, 1);

  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_HOST(accumulator_array);
  UNSAFE_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);
  UNSAFE_TIC()
  KOKKOS_COPY_INTERPOLATOR_MEM_TO_HOST(interpolator_array);
  UNSAFE_TOC( INTERPOLATOR_DATA_MOVEMENT, 1);

  UNSAFE_TIC(); // Time this data movement

  // TODO: make this into a function
  LIST_FOR_EACH( sp, species_list )
  {
    //Kokkos::deep_copy(sp->k_p_h, sp->k_p_d);
    Kokkos::deep_copy(sp->k_pm_h, sp->k_pm_d);
    Kokkos::deep_copy(sp->k_pm_i_h, sp->k_pm_i_d);
    Kokkos::deep_copy(sp->k_nm_h, sp->k_nm_d);
    //auto n_particles = sp->np;
    //auto max_pmovers = sp->max_nm;
    //k_particles_h = sp->k_p_h;
    auto& k_particle_movers_h = sp->k_pm_h;
    auto& k_particle_i_movers_h = sp->k_pm_i_h;
    auto& k_nm_h = sp->k_nm_h;
    sp->nm = k_nm_h(0);
/*
    k_pm_dual.modify_device();
    k_pm_dual.sync_host();
    auto pm_d_sub = Kokkos::subview(sp->k_pm_d, std::pair<size_t, size_t>(0, sp->nm), Kokkos::ALL());
    auto pm_h_sub = Kokkos::subview(sp->k_pm_h, std::pair<size_t, size_t>(0, sp->nm), Kokkos::ALL());
    Kokkos::deep_copy(pm_h_sub, pm_d_sub);
*/
    Kokkos::parallel_for("copy movers to host", host_execution_policy(0, sp->nm) , KOKKOS_LAMBDA (int i) {
      sp->pm[i].dispx = k_particle_movers_h(i, particle_mover_var::dispx);
      sp->pm[i].dispy = k_particle_movers_h(i, particle_mover_var::dispy);
      sp->pm[i].dispz = k_particle_movers_h(i, particle_mover_var::dispz);
      sp->pm[i].i     = k_particle_i_movers_h(i);
    });
  };
  UNSAFE_TOC( PARTICLE_DATA_MOVEMENT, 1);

  // TODO: think about if this is needed? It's done in advance_p
  //Kokkos::deep_copy(sp->k_pc_h, sp->k_pc_d);

  // Because the partial position push when injecting aged particles might
  // place those particles onto the guard list (boundary interaction) and
  // because advance_p requires an empty guard list, particle injection must
  // be done after advance_p and before guard list processing. Note:
  // user_particle_injection should be a stub if species_list is empty.

// Probably needs to be on host due to user particle injection
// May not touch memory?
  if( emitter_list )
    TIC apply_emitter_list( emitter_list ); TOC( emission_model, 1 );
  TIC user_particle_injection(); TOC( user_particle_injection, 1 );

  // This should be after the emission and injection to allow for the
  // possibility of thread parallelizing these operations

//  UNSAFE_TIC();
//  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE();
//  UNSAFE_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

// HOST
// Touches accumulator memory
  if( species_list )
    TIC reduce_accumulator_array( accumulator_array ); TOC( reduce_accumulators, 1 );
//    TIC reduce_accumulator_array_kokkos( accumulator_array ); TOC( reduce_accumulators, 1 );

  // At this point, most particle positions are at r_1 and u_{1/2}. Particles
  // that had boundary interactions are now on the guard list. Process the
  // guard lists. Particles that absorbed are added to rhob (using a corrected
  // local accumulation).

  // This should mean the kokkos accum data is up to date
  UNSAFE_TIC();
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE(accumulator_array);
  UNSAFE_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

// HOST
// Touches particle copies, particle_movers, particle_injectors, accumulators (move_p), neighbors
  TIC
    for( int round=0; round<num_comm_round; round++ )
    {
      //boundary_p( particle_bc_list, species_list, field_array, accumulator_array );
      boundary_p_kokkos( particle_bc_list, species_list, field_array, accumulator_array );
    }
  TOC( boundary_p, num_comm_round );

  // currently the recv particles are in particles_recv, not particle_copy
  LIST_FOR_EACH( sp, species_list )
  {
      Kokkos::deep_copy(sp->k_pc_h, sp->k_pr_h);
      Kokkos::deep_copy(sp->k_pc_i_h, sp->k_pr_i_h);
  }

  // Boundary_p calls move_p, so we need to deal with the current
  // TODO: this will likely break on device?
  //
  //print_fields(accumulator_array->k_a_h);

  //Kokkos::Experimental::contribute(accumulator_array->k_a_h, accumulator_array->k_a_sah);
  //accumulator_array->k_a_sa.reset_except(accumulator_array->k_a_h);

  // Update device so we can pull it all the way back to the host
  UNSAFE_TIC(); // Time this data movement
  Kokkos::deep_copy(accumulator_array->k_a_d, accumulator_array->k_a_h);
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_HOST(accumulator_array);
  UNSAFE_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

  /*
  // Move these value back to the real, on host, accum
  Kokkos::parallel_for("copy accumulator to host", KOKKOS_TEAM_POLICY_HOST \
      (na, Kokkos::AUTO),                          \
      KOKKOS_LAMBDA                                \
      (const KOKKOS_TEAM_POLICY_HOST::member_type &team_member) { \
    const unsigned int i = team_member.league_rank();              \
    \
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ACCUMULATOR_ARRAY_LENGTH), [=] (int j) { \
      accumulator_array->a[i].jx[j] = k_accumulators_h(i, accumulator_var::jx, j); \
      accumulator_array->a[i].jy[j] = k_accumulators_h(i, accumulator_var::jy, j); \
      accumulator_array->a[i].jz[j] = k_accumulators_h(i, accumulator_var::jz, j); \
    }); \
  });
  */

  // Clean_up once boundary p is done
  // Copy back the right data to GPU
  // Device
  // Touches particles, particle_movers
  LIST_FOR_EACH( sp, species_list )
  {
      UNSAFE_TIC(); // Time this data movement
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
      UNSAFE_TOC( BACKFILL, 1);

      auto& particles = sp->k_p_d;
      auto& particles_i = sp->k_p_i_d;

      int num_to_copy = sp->num_to_copy;
      //printf("Trying to append %d from particle copy where np = %d max nm %d \n", num_to_copy, sp->np, sp->max_nm);

      int np = sp->np;

      // Copy data for copies back to device
      UNSAFE_TIC();
      Kokkos::deep_copy(sp->k_pc_d, sp->k_pc_h);
      Kokkos::deep_copy(sp->k_pc_i_d, sp->k_pc_i_h);
      UNSAFE_TOC( PARTICLE_DATA_MOVEMENT, 1);

      UNSAFE_TIC(); // Time this data movement
      auto& particle_copy = sp->k_pc_d;
      auto& particle_copy_i = sp->k_pc_i_d;
      int num_to_copy = sp->num_to_copy;
      int np = sp->np;

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
      UNSAFE_TOC( BACKFILL, 0); // Don't double count
  }

  // TODO: this can be removed once the below does not rely on host memory
//  UNSAFE_TIC(); // Time this data movement
//  LIST_FOR_EACH( sp, species_list ) {\
//    Kokkos::deep_copy(sp->k_p_h, sp->k_p_d);
//    Kokkos::deep_copy(sp->k_pm_h, sp->k_pm_d);
//    Kokkos::deep_copy(sp->k_nm_h, sp->k_nm_d);
//    n_particles = sp->np;
//    max_pmovers = sp->max_nm;
//    k_particles_h = sp->k_p_h;
//    k_particle_movers_h = sp->k_pm_h;
/*
    k_nm_h = sp->k_nm_h;
    sp->nm = k_nm_h(0);
*/
//  };
//  UNSAFE_TOC( PARTICLE_DATA_MOVEMENT, 1);

  // This copies over a val for nm, which is a lie
  LIST_FOR_EACH( sp, species_list ) {
      sp->nm = 0;
  }

  /*
  LIST_FOR_EACH( sp, species_list ) {
    if( sp->nm && verbose )
      WARNING(( "Removing %i particles associated with unprocessed %s movers (increase num_comm_round)",
                sp->nm, sp->name ));
    // Drop the particles that have unprocessed movers due to a user defined
    // boundary condition. Particles of this type with unprocessed movers are
    // in the list of particles and move_p has set the voxel in the particle to
    // 8*voxel + face. This is an incorrect voxel index and in many cases can
    // in fact go out of bounds of the voxel indexing space. Removal is in
    // reverse order for back filling. Particle charge is accumulated to the
    // mesh before removing the particle.

    // Mostly parallel but probably breaks things...
    // Particles should be monotonically increasing and there is a small
    // chance that particle charge is distributed to the wrong places.

    Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Atomic> > replacements("replacement indices", sp->nm);
    remove_particles(sp->k_p_d, sp->k_pm_d, replacements, sp->nm, sp->np);
    fill_holes(sp->k_p_d, sp->k_pm_d, replacements, sp->nm);
    k_accumulate_rhob(field_array->k_f_d, sp->k_p_d, sp->k_pm_d, sp->g, sp->q, sp->nm);
    sp->np -= sp->nm;
    sp->nm = 0;

    int nm = sp->nm;
    particle_mover_t * RESTRICT ALIGNED(16)  pm = sp->pm + sp->nm - 1;
    particle_t * RESTRICT ALIGNED(128) p0 = sp->p;
    for (; nm; nm--, pm--) {
      int i = pm->i; // particle index we are removing
      p0[i].i >>= 3; // shift particle voxel down
      // accumulate the particle's charge to the mesh
      accumulate_rhob( field_array->f, p0+i, sp->g, sp->q );
//      k_accumulate_rhob( field_array->k_f_d, sp->k_p_d, i, sp->g, sp->q );
      p0[i] = p0[sp->np-1]; // put the last particle into position i
      sp->np--; // decrement the number of particles
    }
    sp->nm = 0;

  }

  KOKKOS_COPY_PARTICLE_MEM_TO_HOST();
*/
  // At this point, all particle positions are at r_1 and u_{1/2}, the
  // guard lists are empty and the accumulators on each processor are current.
  // Convert the accumulators into currents.

  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
// HOST
// Touches fields and accumulators
//  TIC FAK->clear_jf( field_array ); TOC( clear_jf, 1 );
  TIC FAK->clear_jf_kokkos( field_array ); TOC( clear_jf, 1 );

//  UNSAFE_TIC(); // Time this data movement
//  KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

  UNSAFE_TIC();
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE(accumulator_array);
  UNSAFE_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

  if( species_list )
    TIC unload_accumulator_array_kokkos( field_array, accumulator_array ); TOC( unload_accumulator, 1 );
//    TIC unload_accumulator_array( field_array, accumulator_array ); TOC( unload_accumulator, 1 );
//  UNSAFE_TIC();
//  KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
//  UNSAFE_TIC();
//  KOKKOS_COPY_ACCUMULATOR_MEM_TO_HOST();
//  UNSAFE_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

//  TIC FAK->synchronize_jf( field_array ); TOC( synchronize_jf, 1 );
  TIC FAK->k_synchronize_jf( field_array ); TOC( synchronize_jf, 1 );

  UNSAFE_TIC();
  KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
  
  // At this point, the particle currents are known at jf_{1/2}.
  // Let the user add their own current contributions. It is the users
  // responsibility to insure injected currents are consistent across domains.
  // It is also the users responsibility to update rhob according to
  // rhob_1 = rhob_0 + div juser_{1/2} (corrected local accumulation) if
  // the user wants electric field divergence cleaning to work.

  TIC user_current_injection(); TOC( user_current_injection, 1 );

  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

// DEVICE
// Touches fields
  // Half advance the magnetic field from B_0 to B_{1/2}
  TIC FAK->advance_b( field_array, 0.5 ); TOC( advance_b, 1 );

  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

  // Advance the electric field from E_0 to E_1
  
// HOST (Device in nphtan branch)
// Touches fields
  TIC FAK->advance_e( field_array, 1.0 ); TOC( advance_e, 1 );
//  TIC FAK->advance_e_kokkos( field_array, 1.0 ); TOC( advance_e, 1 );

//  UNSAFE_TIC();
//  KOKKOS_COPY_FIELD_MEM_TO_HOST();
//  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

  // Let the user add their own contributions to the electric field. It is the
  // users responsibility to insure injected electric fields are consistent
  // across domains.

// ??
  TIC user_field_injection(); TOC( user_field_injection, 1 );

  // Half advance the magnetic field from B_{1/2} to B_1

  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

// DEVICE
// Touches fields
  TIC FAK->advance_b( field_array, 0.5 ); TOC( advance_b, 1 );

  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

  // Divergence clean e

  if( (clean_div_e_interval>0) && ((step() % clean_div_e_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Divergence cleaning electric field" ));

// HOST (Device in rho_p)
// Touches fields and particles
//    TIC FAK->clear_rhof( field_array ); TOC( clear_rhof,1 );
    TIC FAK->clear_rhof_kokkos( field_array ); TOC( clear_rhof,1 );
//    UNSAFE_TIC(); // Time this data movement
//    KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//    UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
    if( species_list ) {

//        KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE();
//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

        TIC
        LIST_FOR_EACH( sp, species_list )
        {
//            accumulate_rho_p( field_array, sp ); //TOC( accumulate_rho_p, species_list->id );
            k_accumulate_rho_p( field_array, sp ); //TOC( accumulate_rho_p, species_list->id );
        }
        TOC( accumulate_rho_p, species_list->id );

//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
    }
//    UNSAFE_TIC();
//    KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//    UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

    TIC FAK->k_synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
//    TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

// HOST
// Touches fields
    for( int round=0; round<num_div_e_round; round++ ) {
//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
//      TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
      TIC FAK->compute_div_e_err_kokkos( field_array ); TOC( compute_div_e_err, 1 );
//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
      if( round==0 || round==num_div_e_round-1 ) {
//        TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
        TIC err = FAK->compute_rms_div_e_err_kokkos( field_array ); TOC( compute_rms_div_e_err, 1 );
//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
        if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
      }
//      TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );
      TIC FAK->clean_div_e_kokkos( field_array ); TOC( clean_div_e, 1 );
//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_DEVICE();
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
    }
//    UNSAFE_TIC();
//    KOKKOS_COPY_FIELD_MEM_TO_HOST();
//    UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
  }
//    UNSAFE_TIC();
//    KOKKOS_COPY_FIELD_MEM_TO_HOST();
//    UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

  // Divergence clean b
// HOST
// Touches fields
  if( (clean_div_b_interval>0) && ((step() % clean_div_b_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Divergence cleaning magnetic field" ));

    for( int round=0; round<num_div_b_round; round++ ) {
//      UNSAFE_TIC();
//      KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
//      UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
//      TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
      TIC FAK->compute_div_b_err_kokkos( field_array ); TOC( compute_div_b_err, 1 );
//      UNSAFE_TIC();
//      KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//      UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
      if( round==0 || round==num_div_b_round-1 ) {
//        TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
        TIC err = FAK->compute_rms_div_b_err_kokkos( field_array ); TOC( compute_rms_div_b_err, 1 );
//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
        if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
      }
//        UNSAFE_TIC();
//        KOKKOS_COPY_FIELD_MEM_TO_HOST();
//        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
//      TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );
      TIC FAK->clean_div_b_kokkos( field_array ); TOC( clean_div_b, 1 );
        UNSAFE_TIC();
        KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
        UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
    }
  }
//  UNSAFE_TIC();
//  KOKKOS_COPY_FIELD_MEM_TO_HOST();
//  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);

  // Synchronize the shared faces
// HOST
// Touches fields
  if( (sync_shared_interval>0) && ((step() % sync_shared_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Synchronizing shared tang e, norm b, rho_b" ));
    TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
//    TIC err = FAK->synchronize_tang_e_norm_b_kokkos( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
    if( rank()==0 ) MESSAGE(( "Domain desynchronization error = %e (arb units)", err ));
  }

  // Fields are updated ... load the interpolator for next time step and
  // particle diagnostics in user_diagnostics if there are any particle
  // species to worry about

  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
  UNSAFE_TIC();
  KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE(interpolator_array);
  UNSAFE_TOC( INTERPOLATOR_DATA_MOVEMENT, 1);

// DEVICE
// Touches fields, interpolators
  if( species_list ) TIC load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );

  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_INTERPOLATOR_MEM_TO_HOST(interpolator_array);
  UNSAFE_TOC( INTERPOLATOR_DATA_MOVEMENT, 1);
  UNSAFE_TIC();
  KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
  UNSAFE_TOC( FIELD_DATA_MOVEMENT, 1);
  UNSAFE_TIC(); // Time this data movement
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE(accumulator_array);
  UNSAFE_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);

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
  TIC dump_energies("energies.txt", 1); TOC( user_diagnostics, 1);

  return 1;
}

