#include "vpic.h"
#include "../particle_operations/compress.h"
#include "../particle_operations/sort.h"
#include <Kokkos_Sort.hpp>
#include <chrono>
#include <thread>

#define FAK field_array->kernel

int vpic_simulation::advance(void)
{
  species_t *sp;
  double err;

  std::string step_str = std::to_string(step());

#ifdef VPIC_ENABLE_PAPI
//Kokkos::Profiling::pushRegion(" " + step_str);
#endif

  //printf("%d: Step %d \n", rank(), step());

  // Use default policy, for now
  ParticleCompressor<> compressor;
  ParticleSorter<> sorter;

  // Determine if we are done ... see note below why this is done here

  if( num_step>0 && step()>=num_step ) {
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::popRegion();
#endif
    return 0;
  }

#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " Sort_particles");
#endif
  KOKKOS_TIC();

  // Sort the particles for performance if desired.
  LIST_FOR_EACH( sp, species_list )
  {
      if( (sp->sort_interval>0) && ((step() % sp->sort_interval)==0) )
      {
          if( rank()==0 ) MESSAGE(( "Performance sorting \"%s\"", sp->name ));
          //TIC sort_p( sp ); TOC( sort_p, 1 );
          sorter.sort( sp->k_p_soa_d, sp->np, accumulator_array->na);
//          sorter.strided_sort( sp->k_p_soa_d, sp->np, accumulator_array->na);
//          sorter.tiled_sort( sp->k_p_soa_d, sp->np, accumulator_array->na, 4);
//          sorter.tiled_strided_sort( sp->k_p_soa_d, sp->np, accumulator_array->na, 2048);
      }
  }

  KOKKOS_TOC( sort_particles, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // At this point, fields are at E_0 and B_0 and the particle positions
  // are at r_0 and u_{-1/2}.  Further the mover lists for the particles should
  // empty and all particles should be inside the local computational domain.
  // Advance the particle lists.

// HOST
// Touches accumulators
  if( species_list ) {
//    TIC clear_accumulator_array( accumulator_array ); TOC( clear_accumulators, 1 );
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::pushRegion(" " + step_str + " clear_accumulator_array");
#endif
    TIC clear_accumulator_array_kokkos( accumulator_array ); TOC( clear_accumulators, 1 );
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::popRegion();
#endif
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

  //TIC user_particle_collisions(); TOC( user_particle_collisions, 1 );

  //int lna = 180;

// DEVICE function
// Touches particles, particle movers, accumulators, interpolators
//  KOKKOS_TIC();

  // DEVICE function - Touches particles, particle movers, accumulators, interpolators
  LIST_FOR_EACH( sp, species_list )
  {
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " " + std::string(sp->name) + " advance_p");
      advance_p_profiling( sp, accumulator_array, interpolator_array, step() );
  Kokkos::Profiling::popRegion();
#else
      advance_p( sp, accumulator_array, interpolator_array );
#endif
  }
//  KOKKOS_TOC( advance_p, 1);

#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " accumulator_contributions");
#endif
  KOKKOS_TIC();
  Kokkos::Experimental::contribute(accumulator_array->k_a_d, accumulator_array->k_a_sa);
  accumulator_array->k_a_sa.reset_except(accumulator_array->k_a_d);
  KOKKOS_TOC( accumulator_contributions, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " Particle_Data_Movement: Copy particle movers");
#endif
  KOKKOS_TIC(); // Time this data movement
  // TODO: make this into a function
  LIST_FOR_EACH( sp, species_list )
  {
    Kokkos::deep_copy(sp->k_nm_h, sp->k_nm_d);
    {
        auto pm_h_dispx = Kokkos::subview(sp->k_pm_h, std::make_pair(0, sp->k_nm_h(0)), 0);
        auto pm_d_dispx = Kokkos::subview(sp->k_pm_d, std::make_pair(0, sp->k_nm_h(0)), 0);
        auto pm_h_dispy = Kokkos::subview(sp->k_pm_h, std::make_pair(0, sp->k_nm_h(0)), 1);
        auto pm_d_dispy = Kokkos::subview(sp->k_pm_d, std::make_pair(0, sp->k_nm_h(0)), 1);
        auto pm_h_dispz = Kokkos::subview(sp->k_pm_h, std::make_pair(0, sp->k_nm_h(0)), 2);
        auto pm_d_dispz = Kokkos::subview(sp->k_pm_d, std::make_pair(0, sp->k_nm_h(0)), 2);
        auto pm_i_h_subview = Kokkos::subview(sp->k_pm_i_h, std::make_pair(0, sp->k_nm_h(0)));
        auto pm_i_d_subview = Kokkos::subview(sp->k_pm_i_d, std::make_pair(0, sp->k_nm_h(0)));
        Kokkos::deep_copy(pm_h_dispx, pm_d_dispx);
        Kokkos::deep_copy(pm_h_dispy, pm_d_dispy);
        Kokkos::deep_copy(pm_h_dispz, pm_d_dispz);
        Kokkos::deep_copy(pm_i_h_subview, pm_i_d_subview);
    }

    auto& k_particle_movers_h = sp->k_pm_h;
    auto& k_particle_i_movers_h = sp->k_pm_i_h;
    auto& k_nm_h = sp->k_nm_h;
    sp->nm = k_nm_h(0);

    Kokkos::parallel_for("copy movers to host", host_execution_policy(0, sp->nm) , KOKKOS_LAMBDA (int i) {
      sp->pm[i].dispx = k_particle_movers_h(i, particle_mover_var::dispx);
      sp->pm[i].dispy = k_particle_movers_h(i, particle_mover_var::dispy);
      sp->pm[i].dispz = k_particle_movers_h(i, particle_mover_var::dispz);
      sp->pm[i].i     = k_particle_i_movers_h(i);
    });
  };
  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

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

    if((particle_injection_interval>0) && ((step() % particle_injection_interval)==0)) {
        if(!kokkos_particle_injection) {
#ifdef VPIC_ENABLE_PAPI
            Kokkos::Profiling::pushRegion(" " + step_str + " Particle_Data_Movement: To host before particle injection");
#endif
            KOKKOS_TIC();
            KOKKOS_COPY_PARTICLE_MEM_TO_HOST(species_list);
            KOKKOS_TOC(PARTICLE_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
            Kokkos::Profiling::popRegion();
#endif
        }
#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::pushRegion(" " + step_str + " user_particle_injection");
#endif
        TIC user_particle_injection(); TOC( user_particle_injection, 1 );
#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::popRegion();
#endif
        if(!kokkos_particle_injection) {
#ifdef VPIC_ENABLE_PAPI
            Kokkos::Profiling::pushRegion(" " + step_str + " Particle_Data_Movement: To device after particle injection");
#endif
            KOKKOS_TIC();
            KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE(species_list);
            KOKKOS_TOC(PARTICLE_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
            Kokkos::Profiling::popRegion();
#endif
        }
    }

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

  // This should mean the kokkos accum data is up to date
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " Accumulator_Data_Movement: To host before boundary_p");
#endif
  KOKKOS_TIC(); // Time this data movement
  Kokkos::deep_copy(accumulator_array->k_a_h, accumulator_array->k_a_d);
  KOKKOS_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

// HOST
// Touches particle copies, particle_movers, particle_injectors, accumulators (move_p), neighbors
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " boundary_p");
#endif
  TIC
    for( int round=0; round<num_comm_round; round++ )
    {
      //boundary_p( particle_bc_list, species_list, field_array, accumulator_array );
      boundary_p_kokkos( particle_bc_list, species_list, field_array, accumulator_array );
    }
  TOC( boundary_p, num_comm_round );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // currently the recv particles are in particles_recv, not particle_copy
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " Particle_Data_Movement: Copy recv particles to particle copy");
#endif
  KOKKOS_TIC();
  LIST_FOR_EACH( sp, species_list )
  {
        auto pc_dx_h_subview = Kokkos::subview(sp->k_pc_soa_h.dx, std::make_pair(0, sp->num_to_copy));
        auto pc_dy_h_subview = Kokkos::subview(sp->k_pc_soa_h.dy, std::make_pair(0, sp->num_to_copy));
        auto pc_dz_h_subview = Kokkos::subview(sp->k_pc_soa_h.dz, std::make_pair(0, sp->num_to_copy));
        auto pc_ux_h_subview = Kokkos::subview(sp->k_pc_soa_h.ux, std::make_pair(0, sp->num_to_copy));
        auto pc_uy_h_subview = Kokkos::subview(sp->k_pc_soa_h.uy, std::make_pair(0, sp->num_to_copy));
        auto pc_uz_h_subview = Kokkos::subview(sp->k_pc_soa_h.uz, std::make_pair(0, sp->num_to_copy));
#ifndef PARTICLE_WEIGHT_CONSTANT
        auto pc_w_h_subview = Kokkos::subview(sp->k_pc_soa_h.w, std::make_pair(0, sp->num_to_copy));
#endif
        auto pc_i_h_subview = Kokkos::subview(sp->k_pc_soa_h.i, std::make_pair(0, sp->num_to_copy));

        auto pr_dx_h_subview = Kokkos::subview(sp->k_pr_soa_h.dx, std::make_pair(0, sp->num_to_copy));
        auto pr_dy_h_subview = Kokkos::subview(sp->k_pr_soa_h.dy, std::make_pair(0, sp->num_to_copy));
        auto pr_dz_h_subview = Kokkos::subview(sp->k_pr_soa_h.dz, std::make_pair(0, sp->num_to_copy));
        auto pr_ux_h_subview = Kokkos::subview(sp->k_pr_soa_h.ux, std::make_pair(0, sp->num_to_copy));
        auto pr_uy_h_subview = Kokkos::subview(sp->k_pr_soa_h.uy, std::make_pair(0, sp->num_to_copy));
        auto pr_uz_h_subview = Kokkos::subview(sp->k_pr_soa_h.uz, std::make_pair(0, sp->num_to_copy));
#ifndef PARTICLE_WEIGHT_CONSTANT
        auto pr_w_h_subview = Kokkos::subview(sp->k_pr_soa_h.w, std::make_pair(0, sp->num_to_copy));
#endif
        auto pr_i_h_subview = Kokkos::subview(sp->k_pr_soa_h.i, std::make_pair(0, sp->num_to_copy));
  
        Kokkos::deep_copy(pc_dx_h_subview, pr_dx_h_subview);
        Kokkos::deep_copy(pc_dy_h_subview, pr_dy_h_subview);
        Kokkos::deep_copy(pc_dz_h_subview, pr_dz_h_subview);
        Kokkos::deep_copy(pc_ux_h_subview, pr_ux_h_subview);
        Kokkos::deep_copy(pc_uy_h_subview, pr_uy_h_subview);
        Kokkos::deep_copy(pc_uz_h_subview, pr_uz_h_subview);
#ifndef PARTICLE_WEIGHT_CONSTANT
        Kokkos::deep_copy(pc_w_h_subview, pr_w_h_subview);
#endif
        Kokkos::deep_copy(pc_i_h_subview, pr_i_h_subview);
  }
  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // Boundary_p calls move_p, so we need to deal with the current
  // TODO: this will likely break on device?
  //
  //print_fields(accumulator_array->k_a_h);

  //Kokkos::Experimental::contribute(accumulator_array->k_a_h, accumulator_array->k_a_sah);
  //accumulator_array->k_a_sa.reset_except(accumulator_array->k_a_h);

  // Update device so we can pull it all the way back to the host
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " Accumulator_Data_Movement: To device after boundary_p");
#endif
  KOKKOS_TIC(); // Time this data movement
  Kokkos::deep_copy(accumulator_array->k_a_d, accumulator_array->k_a_h);
  KOKKOS_TOC( ACCUMULATOR_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // Clean_up once boundary p is done
  // Copy back the right data to GPU
  // Device
  // Touches particles, particle_movers
int sp_counter = 0;
  LIST_FOR_EACH( sp, species_list )
  {
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " Backfill " + std::to_string(sp_counter) + ": Compress");
#endif
      KOKKOS_TIC(); // Time this data movement
      const int nm = sp->k_nm_h(0);

      // TODO: this can be hoisted to the end of advance_p if desired
      compressor.compress(
              sp->k_p_soa_d,
//              sp->k_p_d,
//              sp->k_p_i_d,
              sp->k_pm_i_d,
              nm,
              sp->np,
              sp
      );

      // Update np now we removed them...
      sp->np -= nm;
      KOKKOS_TOC( BACKFILL, 0);
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif

      auto& k_particles = sp->k_p_soa_d;

      int num_to_copy = sp->num_to_copy;

      int np = sp->np;

      // Copy data for copies back to device
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " Particle_Data_Movement " + std::to_string(sp_counter) + ": Update device particle copy");
#endif
      KOKKOS_TIC();
        auto pc_dx_d_subview = Kokkos::subview(sp->k_pc_soa_d.dx, std::make_pair(0, sp->num_to_copy));
        auto pc_dy_d_subview = Kokkos::subview(sp->k_pc_soa_d.dy, std::make_pair(0, sp->num_to_copy));
        auto pc_dz_d_subview = Kokkos::subview(sp->k_pc_soa_d.dz, std::make_pair(0, sp->num_to_copy));
        auto pc_ux_d_subview = Kokkos::subview(sp->k_pc_soa_d.ux, std::make_pair(0, sp->num_to_copy));
        auto pc_uy_d_subview = Kokkos::subview(sp->k_pc_soa_d.uy, std::make_pair(0, sp->num_to_copy));
        auto pc_uz_d_subview = Kokkos::subview(sp->k_pc_soa_d.uz, std::make_pair(0, sp->num_to_copy));
#ifndef PARTICLE_WEIGHT_CONSTANT
        auto pc_w_d_subview = Kokkos::subview(sp->k_pc_soa_d.w, std::make_pair(0, sp->num_to_copy));
#endif
        auto pc_i_d_subview = Kokkos::subview(sp->k_pc_soa_d.i, std::make_pair(0, sp->num_to_copy));

        auto pc_dx_h_subview = Kokkos::subview(sp->k_pc_soa_h.dx, std::make_pair(0, sp->num_to_copy));
        auto pc_dy_h_subview = Kokkos::subview(sp->k_pc_soa_h.dy, std::make_pair(0, sp->num_to_copy));
        auto pc_dz_h_subview = Kokkos::subview(sp->k_pc_soa_h.dz, std::make_pair(0, sp->num_to_copy));
        auto pc_ux_h_subview = Kokkos::subview(sp->k_pc_soa_h.ux, std::make_pair(0, sp->num_to_copy));
        auto pc_uy_h_subview = Kokkos::subview(sp->k_pc_soa_h.uy, std::make_pair(0, sp->num_to_copy));
        auto pc_uz_h_subview = Kokkos::subview(sp->k_pc_soa_h.uz, std::make_pair(0, sp->num_to_copy));
#ifndef PARTICLE_WEIGHT_CONSTANT
        auto pc_w_h_subview = Kokkos::subview(sp->k_pc_soa_h.w, std::make_pair(0, sp->num_to_copy));
#endif
        auto pc_i_h_subview = Kokkos::subview(sp->k_pc_soa_h.i, std::make_pair(0, sp->num_to_copy));
        Kokkos::deep_copy(pc_dx_d_subview, pc_dx_h_subview);
        Kokkos::deep_copy(pc_dy_d_subview, pc_dy_h_subview);
        Kokkos::deep_copy(pc_dz_d_subview, pc_dz_h_subview);
        Kokkos::deep_copy(pc_ux_d_subview, pc_ux_h_subview);
        Kokkos::deep_copy(pc_uy_d_subview, pc_uy_h_subview);
        Kokkos::deep_copy(pc_uz_d_subview, pc_uz_h_subview);
#ifndef PARTICLE_WEIGHT_CONSTANT
        Kokkos::deep_copy(pc_w_d_subview, pc_w_h_subview);
#endif
        Kokkos::deep_copy(pc_i_d_subview, pc_i_h_subview);
      KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif

#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " Backfill " + std::to_string(sp_counter) + ": Append moved particles to device particles");
#endif
      KOKKOS_TIC(); // Time this data movement
      auto& particle_copy = sp->k_pc_soa_d;
      int num_to_copy = sp->num_to_copy;
      int np = sp->np;

      // Append it to the particles
      Kokkos::parallel_for("append moved particles", Kokkos::RangePolicy <
              Kokkos::DefaultExecutionSpace > (0, sp->num_to_copy), KOKKOS_LAMBDA
              (int i)
      {
        int npi = np+i; // i goes from 0..n so no need for -1

        k_particles.dx(npi) = particle_copy.dx(i);
        k_particles.dy(npi) = particle_copy.dy(i);
        k_particles.dz(npi) = particle_copy.dz(i);
        k_particles.ux(npi) = particle_copy.ux(i);
        k_particles.uy(npi) = particle_copy.uy(i);
        k_particles.uz(npi) = particle_copy.uz(i);
#ifndef PARTICLE_WEIGHT_CONSTANT
        k_particles.w(npi)  = particle_copy.w(i);
#endif
        k_particles.i(npi)  = particle_copy.i(i);
      });

      // Reset this to zero now we've done the write back
      sp->np += num_to_copy;
      sp->num_to_copy = 0;
      KOKKOS_TOC( BACKFILL, 1); // Don't double count
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif
sp_counter ++;
  }

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
      //k_accumulate_rhob( field_array->k_f_d, sp->k_p_d, i, sp->g, sp->q );
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

// HOST
// Touches fields and accumulators
//  TIC FAK->clear_jf( field_array ); TOC( clear_jf, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " clear_jf");
#endif 
  TIC FAK->clear_jf_kokkos( field_array ); TOC( clear_jf, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  if( species_list ) {
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::pushRegion(" " + step_str + " unload_accumulator_array");
#endif
    TIC unload_accumulator_array_kokkos( field_array, accumulator_array ); TOC( unload_accumulator, 1 );
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::popRegion();
#endif
  }
//    TIC unload_accumulator_array( field_array, accumulator_array ); TOC( unload_accumulator, 1 );

//  TIC FAK->synchronize_jf( field_array ); TOC( synchronize_jf, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " synchronize_jf");
#endif
  TIC FAK->k_synchronize_jf( field_array ); TOC( synchronize_jf, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // At this point, the particle currents are known at jf_{1/2}.
  // Let the user add their own current contributions. It is the users
  // responsibility to insure injected currents are consistent across domains.
  // It is also the users responsibility to update rhob according to
  // rhob_1 = rhob_0 + div juser_{1/2} (corrected local accumulation) if
  // the user wants electric field divergence cleaning to work.

  if((current_injection_interval>0) && ((step() % current_injection_interval)==0)) {
      if(!kokkos_current_injection) {
#ifdef VPIC_ENABLE_PAPI
          Kokkos::Profiling::pushRegion(" " + step_str + " Field_Data_Movement: To host before current injection");
#endif
          KOKKOS_TIC();
          KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
          KOKKOS_TOC(FIELD_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
          Kokkos::Profiling::popRegion();
#endif
      }
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " user_current_injection");
#endif
      TIC user_current_injection(); TOC( user_current_injection, 1 );
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif
      if(!kokkos_current_injection) {
#ifdef VPIC_ENABLE_PAPI
          Kokkos::Profiling::pushRegion(" " + step_str + " Field_Data_Movement: To device after current injection");
#endif
          KOKKOS_TIC();
          KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
          KOKKOS_TOC(FIELD_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
          Kokkos::Profiling::popRegion();
#endif
      }
  }

  // DEVICE -- Touches fields
  // Half advance the magnetic field from B_0 to B_{1/2}
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " advance_b: First half step");
#endif
  KOKKOS_TIC();
  FAK->advance_b( field_array, 0.5 );
  KOKKOS_TOC( advance_b, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // Advance the electric field from E_0 to E_1
  
// HOST (Device in nphtan branch)
// Touches fields
//  TIC FAK->advance_e( field_array, 1.0 ); TOC( advance_e, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " advance_e");
#endif
  TIC FAK->advance_e_kokkos( field_array, 1.0 ); TOC( advance_e, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // Let the user add their own contributions to the electric field. It is the
  // users responsibility to insure injected electric fields are consistent
  // across domains.

    if((field_injection_interval>0) && ((step() % field_injection_interval)==0)) {
        if(!kokkos_field_injection) {
#ifdef VPIC_ENABLE_PAPI
            Kokkos::Profiling::pushRegion(" " + step_str + " Field_Data_Movement: To host before field injection");
#endif
            KOKKOS_TIC();
            KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
            KOKKOS_TOC(FIELD_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
            Kokkos::Profiling::popRegion();
#endif
        }
#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::pushRegion(" " + step_str + " user_field_injection");
#endif
        TIC user_field_injection(); TOC( user_field_injection, 1 );
#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::popRegion();
#endif
        if(!kokkos_field_injection) {
#ifdef VPIC_ENABLE_PAPI
            Kokkos::Profiling::pushRegion(" " + step_str + " Field_Data_Movement: To device after field injection");
#endif
            KOKKOS_TIC();
            KOKKOS_COPY_FIELD_MEM_TO_DEVICE(field_array);
            KOKKOS_TOC(FIELD_DATA_MOVEMENT, 1);
#ifdef VPIC_ENABLE_PAPI
            Kokkos::Profiling::popRegion();
#endif
        }
    }

  // Half advance the magnetic field from B_{1/2} to B_1

// DEVICE
// Touches fields
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " advance_b: Second half step");
#endif
  TIC FAK->advance_b( field_array, 0.5 ); TOC( advance_b, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // Divergence clean e

  if( (clean_div_e_interval>0) && ((step() % clean_div_e_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Divergence cleaning electric field" ));

// HOST (Device in rho_p)
// Touches fields and particles
//    TIC FAK->clear_rhof( field_array ); TOC( clear_rhof,1 );
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::pushRegion(" " + step_str + " clear_rhof");
#endif
    TIC FAK->clear_rhof_kokkos( field_array ); TOC( clear_rhof,1 );
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::popRegion();
#endif
    if( species_list ) {

#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::pushRegion(" " + step_str + " accumulate_rho_p");
#endif
        KOKKOS_TIC();
        LIST_FOR_EACH( sp, species_list )
        {
            //accumulate_rho_p( field_array, sp ); //TOC( accumulate_rho_p, species_list->id );
            k_accumulate_rho_p( field_array, sp );
        }
        KOKKOS_TOC( accumulate_rho_p, species_list->id );
#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::popRegion();
#endif
    }

#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::pushRegion(" " + step_str + " synchronize_rho");
#endif
    TIC FAK->k_synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::popRegion();
#endif
//    TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );

// HOST
// Touches fields
    for( int round=0; round<num_div_e_round; round++ ) {
//      TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " compute_div_e_err: Round " + std::to_string(round));
#endif
      TIC FAK->compute_div_e_err_kokkos( field_array ); TOC( compute_div_e_err, 1 );
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif
      if( round==0 || round==num_div_e_round-1 ) {
//        TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::pushRegion(" " + step_str + " compute_rms_div_e_err: Round" + std::to_string(round));
#endif
        TIC err = FAK->compute_rms_div_e_err_kokkos( field_array ); TOC( compute_rms_div_e_err, 1 );
#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::popRegion();
#endif
        if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
      }
//      TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " clean_div_e: Round" + std::to_string(round));
#endif
      TIC FAK->clean_div_e_kokkos( field_array ); TOC( clean_div_e, 1 );
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif
    }
  }

  // Divergence clean b
// HOST
// Touches fields
  if( (clean_div_b_interval>0) && ((step() % clean_div_b_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Divergence cleaning magnetic field" ));

    for( int round=0; round<num_div_b_round; round++ ) {
//      TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " compute_div_b_err: Round" + std::to_string(round));
#endif
      TIC FAK->compute_div_b_err_kokkos( field_array ); TOC( compute_div_b_err, 1 );
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif
      if( round==0 || round==num_div_b_round-1 ) {
//        TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::pushRegion(" " + step_str + " compute_rms_div_b_err: Round" + std::to_string(round));
#endif
        TIC err = FAK->compute_rms_div_b_err_kokkos( field_array ); TOC( compute_rms_div_b_err, 1 );
#ifdef VPIC_ENABLE_PAPI
        Kokkos::Profiling::popRegion();
#endif
        if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
      }
//      TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " clean_div_b: Round" + std::to_string(round));
#endif
      TIC FAK->clean_div_b_kokkos( field_array ); TOC( clean_div_b, 1 );
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif
    }
  }

  // Synchronize the shared faces
  // HOST
  // Touches fields
  if( (sync_shared_interval>0) && ((step() % sync_shared_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Synchronizing shared tang e, norm b, rho_b" ));
//    TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::pushRegion(" " + step_str + " synchronize_tang_e_norm_b");
#endif
    TIC err = FAK->synchronize_tang_e_norm_b_kokkos( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
#ifdef VPIC_ENABLE_PAPI
    Kokkos::Profiling::popRegion();
#endif
    if( rank()==0 ) MESSAGE(( "Domain desynchronization error = %e (arb units)", err ));
  }

  // Fields are updated ... load the interpolator for next time step and
  // particle diagnostics in user_diagnostics if there are any particle
  // species to worry about

// DEVICE
// Touches fields, interpolators
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " load_interpolator_array");
#endif
  if( species_list ) TIC load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  step()++;

  // Print out status
  if( (status_interval>0) && ((step() % status_interval)==0) ) {
      if( rank()==0 ) MESSAGE(( "Completed step %i of %i", step(), num_step ));
      update_profile( rank()==0 );
  }

  // Optionally move data back from the device, at user request
  if ( (particle_copy_interval > 0) && ((step() % particle_copy_interval) == 0))
  {
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " Particle_Data_Movement: To host before user_diagnostics");
#endif
      // Copy particles back
      KOKKOS_TIC(); // Time this data movement
      KOKKOS_COPY_PARTICLE_MEM_TO_HOST(species_list);
      KOKKOS_TOC( user_diagnostics, 1);
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif
  }
  if ( (field_copy_interval > 0) && ((step() % field_copy_interval) == 0))
  {
      // Copy fields back
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::pushRegion(" " + step_str + " Field_Data_Movement: To host before user diagnostics");
#endif
      KOKKOS_TIC(); // Time this data movement
      KOKKOS_COPY_FIELD_MEM_TO_HOST(field_array);
      KOKKOS_TOC( user_diagnostics, 1);
#ifdef VPIC_ENABLE_PAPI
      Kokkos::Profiling::popRegion();
#endif
  }

  // Let the user compute diagnostics
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::pushRegion(" " + step_str + " user_diagnostics");
#endif
  TIC user_diagnostics(); TOC( user_diagnostics, 1 );
#ifdef VPIC_ENABLE_PAPI
  Kokkos::Profiling::popRegion();
#endif

  // "return step()!=num_step" is more intuitive. But if a checkpt
  // saved in the call to user_diagnostics() above, is done on the final step
  // (silly but it might happen), the test will be skipped on the restore. We
  // return true here so that the first call to advance after a restore
  // will act properly for this edge case.

//#ifdef VPIC_DUMP_ENERGIES
//#ifdef VPIC_ENABLE_PAPI
//  Kokkos::Profiling::pushRegion(" " + step_str + " dump_energies");
//#endif
//  KOKKOS_TIC(); // Time this data movement
//  KOKKOS_COPY_PARTICLE_MEM_TO_HOST(species_list);
//  dump_partloc("gpu_Q1_14_pos_nx9000_2particle_1Mtimesteps_particle_locations.txt", 1, -1);
//  TIC dump_energies("gpu_Q1_14_pos_nx9000_2particle_1Mtimesteps_energies.txt", 1); TOC( dump_energies, 1);
//  dump_partloc("test3_particle_locations.txt", 1, -1);
//  TIC dump_energies("test3_energies.txt", 1); TOC( dump_energies, 1);
//  int num_part_cpy = -1;
//  num_part_cpy = 1;
//  LIST_FOR_EACH( sp, species_list ) {
//      auto dx_subview_d = Kokkos::subview(sp->k_p_soa_d.dx, Kokkos::make_pair(0,num_part_cpy));
//      auto dy_subview_d = Kokkos::subview(sp->k_p_soa_d.dy, Kokkos::make_pair(0,num_part_cpy));
//      auto dz_subview_d = Kokkos::subview(sp->k_p_soa_d.dz, Kokkos::make_pair(0,num_part_cpy));
//      auto ux_subview_d = Kokkos::subview(sp->k_p_soa_d.ux, Kokkos::make_pair(0,num_part_cpy));
//      auto uy_subview_d = Kokkos::subview(sp->k_p_soa_d.uy, Kokkos::make_pair(0,num_part_cpy));
//      auto uz_subview_d = Kokkos::subview(sp->k_p_soa_d.uz, Kokkos::make_pair(0,num_part_cpy));
//#ifndef PARTICLE_WEIGHT_CONSTANT
//      auto w_subview_d = Kokkos::subview(sp->k_p_soa_d.w, Kokkos::make_pair(0,num_part_cpy));
//#endif
//      auto i_subview_d = Kokkos::subview(sp->k_p_soa_d.i, Kokkos::make_pair(0,num_part_cpy));
//
//      auto dx_subview_h = Kokkos::subview(sp->k_p_soa_h.dx, Kokkos::make_pair(0,num_part_cpy));
//      auto dy_subview_h = Kokkos::subview(sp->k_p_soa_h.dy, Kokkos::make_pair(0,num_part_cpy));
//      auto dz_subview_h = Kokkos::subview(sp->k_p_soa_h.dz, Kokkos::make_pair(0,num_part_cpy));
//      auto ux_subview_h = Kokkos::subview(sp->k_p_soa_h.ux, Kokkos::make_pair(0,num_part_cpy));
//      auto uy_subview_h = Kokkos::subview(sp->k_p_soa_h.uy, Kokkos::make_pair(0,num_part_cpy));
//      auto uz_subview_h = Kokkos::subview(sp->k_p_soa_h.uz, Kokkos::make_pair(0,num_part_cpy));
//#ifndef PARTICLE_WEIGHT_CONSTANT
//      auto w_subview_h = Kokkos::subview(sp->k_p_soa_h.w, Kokkos::make_pair(0,num_part_cpy));
//#endif
//      auto i_subview_h = Kokkos::subview(sp->k_p_soa_h.i, Kokkos::make_pair(0,num_part_cpy));
//
//      Kokkos::deep_copy(dx_subview_h, dx_subview_d);
//      Kokkos::deep_copy(dy_subview_h, dy_subview_d);
//      Kokkos::deep_copy(dz_subview_h, dz_subview_d);
//      Kokkos::deep_copy(ux_subview_h, ux_subview_d);
//      Kokkos::deep_copy(uy_subview_h, uy_subview_d);
//      Kokkos::deep_copy(uz_subview_h, uz_subview_d);
//      Kokkos::deep_copy(i_subview_h, i_subview_d);
//#ifndef PARTICLE_WEIGHT_CONSTANT
//      Kokkos::deep_copy(w_subview_h, w_subview_d);
//#endif
//
//      auto n_particles = (num_part_cpy == -1) ? sp->np : num_part_cpy;
//
//      auto& kph = sp->k_p_soa_h;
//
//      Kokkos::parallel_for("copy particles to host", host_execution_policy(0, n_particles) , KOKKOS_LAMBDA (int i) {
//          sp->p[i].dx = kph.get_dx(i);
//          sp->p[i].dy = kph.get_dy(i);
//          sp->p[i].dz = kph.get_dz(i);
//          sp->p[i].ux = kph.ux(i);
//          sp->p[i].uy = kph.uy(i);
//          sp->p[i].uz = kph.uz(i);
//          sp->p[i].i  = kph.i(i);
//#if defined PARTICLE_WEIGHT_FLOAT
//          sp->p[i].w  = kph.w(i);
//#elif defined PARTICLE_WEIGHT_SHORT
//          sp->p[i].w  = kph.w(i) * sp->w;
//#endif
//      });
//  }
//  dump_partloc("particle_location.txt", 1, 1);
  TIC dump_energies("energies.txt", 1); TOC( dump_energies, 1);
//  KOKKOS_TOC( user_diagnostics, 1);
//#ifdef VPIC_ENABLE_PAPI
//  Kokkos::Profiling::popRegion();
//#endif
//#endif

#ifdef VPIC_ENABLE_PAPI
//Kokkos::Profiling::popRegion();
#endif
  return 1;
}

