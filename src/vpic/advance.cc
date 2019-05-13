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

#define FAK field_array->kernel

/**
 * @brief This function takes k_particle_movers as a map to tell us where gaps
 * will be in the array, and fills those gaps in parallel
 *
 * This assumes the movers will not have any repeats in indexing, otherwise
 * it's not safe to do in parallel
 *
 * @param k_particles The array to compact
 * @param k_particle_movers The array holding the packing mask
 */
void compress_particle_data(
        k_particles_t particles,
        k_particle_movers_t particle_movers,
        const int32_t nm,
        const int32_t np,
        species_t* sp
)
{
    // From particle_movers(nm, particle_mover_var::pmi), we know where the
    // gaps in particle are

    // We can safely back fill the gaps in parallel by looping over the movers,
    // which are guaranteed to be unique, and sending np-i to that index

    // WARNING: In SoA configuration this may get a bit cache thrashy?
    // TODO: this may perform better if the index in the movers are sorted first..


    // This is a O(NP) solution. There likely exists a faster O(NM) solution
    // but my attempt had a data race



    // POSSIBLE IMPROVEMENT, A better way to do is this?:
    //   Run the back fill loop but if a "pull_from" id is a gap (which can be
    //   detected by setting a special flag in it's p->i value), then skip it.
    //   Instead add the index of that guy to an (atomic) clean up list
    //
    //   Do a second pass over the cleanup list ? (care of repeated data race..)

    // This is a little slow, but not a deal breaker
    // Build a list of "safe" filling ids, to avoid data race
        // we do this for the case where a "gap" exists in the backfill region (np-nm)

    // TODO: prevent this malloc every time
    // Track is the last 2*nm particles in np are "unsafe" to pull from (i.e
    // are "gaps")
    // We want unsafe_index to map up in reverse
    // [ 0  , 1   , 2   , 3... 2nm ] is equal to
    // [np-1, np-2, np-3... np-1-2nm]
    // i.e 0 is the "last particle"
    // This is annoying, but it will give a back fill order more consistent
    // with VPIC's serial algorithm


//#define SIMPLE_COMPRESS 1

// Ignore this...
/// TODO: delete this?
#ifdef SIMPLE_COMPRESS

    // TODO: implement this as a sort
    // Do the sensible but slow version of the compress, to sanity check the
    // correctness and performance.

    // currently this is a serialized (read: slow) gpu backfill.. probably
    // don't use it. But its useful to mirror VPIC functionality

    // sort nm

    //SortView( particles_movers, 0, nm );
    //Kokkos::deep_copy( host_subarray , work_array );

    std::sort(sp->pm, sp->pm + sp->nm, compareParticleMovers<particle_mover_t>);

    Kokkos::parallel_for("copy movers to device", host_execution_policy(0, nm) , KOKKOS_LAMBDA (int i) { \
      particle_movers(i, particle_mover_var::dispx) = sp->pm[i].dispx;
      particle_movers(i, particle_mover_var::dispy) = sp->pm[i].dispy;
      particle_movers(i, particle_mover_var::dispz) = sp->pm[i].dispz;
      particle_movers(i, particle_mover_var::pmi)   = sp->pm[i].i;
    });\


    Kokkos::parallel_for("print nm", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (int _)
    {
      for (int i = 0; i < nm; i++)
      {
          printf("print nm %d has %f \n", i, particle_movers(i, particle_mover_var::pmi));
      }
    });

    Kokkos::parallel_for("particle compress", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (int n)
    {
        for (int i = nm-1; i >= 0; i++)
        {
            // VPIC backfill
            // p0[i] = p0[np];

            int pmi = static_cast<int>( particle_movers(i, particle_mover_var::pmi) );
            int pull_from = np - (nm-1-i);

            //particles(pmi) = particles(np - (nm-1-i));
            particles(pmi, particle_var::dx) = particles(pull_from, particle_var::dx);
            particles(pmi, particle_var::dy) = particles(pull_from, particle_var::dy);
            particles(pmi, particle_var::dz) = particles(pull_from, particle_var::dz);
            particles(pmi, particle_var::ux) = particles(pull_from, particle_var::ux);
            particles(pmi, particle_var::uy) = particles(pull_from, particle_var::uy);
            particles(pmi, particle_var::uz) = particles(pull_from, particle_var::uz);
            particles(pmi, particle_var::w)  = particles(pull_from, particle_var::w);
            particles(pmi, particle_var::pi) = particles(pull_from, particle_var::pi);
        }
    });


#endif


//#define SORT_COMPRESS 1
#ifdef SORT_COMPRESS


      int pi = particle_var::pi; // TODO: can you really not pass an enum in??
      auto keys = Kokkos::subview(sp->k_p_d, Kokkos::ALL, pi);
      using key_type = decltype(keys);


      // TODO: we can tighten the bounds on this
      int max = accumulator_array->na;
      using Comparator = Kokkos::BinOp1D<key_type>;
      Comparator comp(max, 0, max);

      int sort_within_bins = 0;
      Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, sp->np, comp, sort_within_bins );
      bin_sort.create_permute_vector();
      bin_sort.sort(sp->k_p_d);

      // This should leave just 0s at the start?
      auto& particles = sp->k_p_d;
      printf("SORTER \n");
      print_particles_d(particles, sp->np); // should not see any zeros


#else
    Kokkos::View<int*> unsafe_index("safe index", 2*nm);

    // TODO: prevent these allocations from being repeated.

    // Track (atomically) the id's we've tried to pull from when dealing with a
    // gap in the "danger zone"
    Kokkos::View<int> panic_counter("panic counter");

    // We use this to store a list of things we bailed out on moving. Typically because the mapping of pull_from->write_to got skipped.

    Kokkos::View<int> clean_up_to_count("clean up to count"); // todo: find an algorithm that doesn't need this
    Kokkos::View<int> clean_up_from_count("clean up from count"); // todo: find an algorithm that doesn't need this

    Kokkos::View<int>::HostMirror clean_up_to_count_h = Kokkos::create_mirror_view(clean_up_to_count);
    Kokkos::View<int>::HostMirror clean_up_from_count_h = Kokkos::create_mirror_view(clean_up_from_count);

    Kokkos::View<int*> clean_up_from("clean up from", nm);
    Kokkos::View<int*> clean_up_to("clean up to", nm);

    // Loop over 2*nm, which is enough to guarantee you `nm` non-gaps
    // Build a list of safe lookups

    // TODO: we can probably do this online while we do the advance_p
    Kokkos::parallel_for("particle compress", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, nm), KOKKOS_LAMBDA (int i)
    {
        // If the id of this particle is in the danger zone, don't add it
            // otherwise, do
        int cut_off = np-(2*nm);

        int pmi = static_cast<int>( particle_movers(i, particle_mover_var::pmi) );

        // If it's less than the cut off, it's safe
        if ( pmi >= cut_off) // danger zone
        {
          int index = ((np-1) - pmi); // Map to the reverse indexing
          unsafe_index(index) = 1; // 1 marks it as unsafe
        }
    });

    // We will use the first 0-nm of safe_index to pull from
    // We will use the nm -> 2nm range for "panic picks", if the first wasn't safe (involves atomics..)

    Kokkos::parallel_for("particle compress", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, nm), KOKKOS_LAMBDA (int n)
    {

        // TODO: is this np or np-1?
        // Doing this in the "inverse order" to match vpic
        int pull_from = (np-1) - (n); // grab a particle from the end block
        int write_to = particle_movers(nm-n-1, particle_mover_var::pmi); // put it in a gap
        int danger_zone = np - nm;

        // if they're the same, no need to do it. This can happen below in the
        // danger zone and we want to avoid "cleaning it up"
        if (pull_from == write_to) return;

        // If the "gap" is in the danger zone, no need to back fill it
        if (write_to >= danger_zone)
        {
            // TODO: we don't actually have to zero it out, we can just skip
            printf("zeroing out %d for %d, not pulling from %d\n", write_to, n, pull_from);
            particles(write_to, particle_var::dx) = 0.0;
            particles(write_to, particle_var::dy) = 0.0;
            particles(write_to, particle_var::dz) = 0.0;
            particles(write_to, particle_var::ux) = 0.0;
            particles(write_to, particle_var::uy) = 0.0;
            particles(write_to, particle_var::uz) = 0.0;
            particles(write_to, particle_var::w)  = 0.0;
            particles(write_to, particle_var::pi) = 0.0;

            // if pull from is unsafe, skip it
            if (pull_from >= danger_zone) // We only have lookup for the danger zone
            {
                if ( ! unsafe_index( (np-1) - pull_from ) )
                {
                    // FIXME: if it's not in the danger zone, someone else will
                    // fill it..but then we don't want to move it??? is this
                    // true, and a very subtle race condition?

                    // TODO: by skipping this move, we neglect to move the
                    // pull_from to somewhere sensible...  For now we put it on
                    // a clean up list..but that sucks
                    int clean_up_from_index = Kokkos::atomic_fetch_add( &clean_up_from_count(), 1 );
                    clean_up_from(clean_up_from_index) = pull_from;
                }
            }

            return;
        }

        //int safe_index_offset = (np-nm);

        // Detect if the index we want to pull from is safe
        // Want to index this 0...nm
        //if ( unsafe_index(pull_from - safe_index_offset ) )
        if ( unsafe_index( n ) )
        {
            // Instead we'll get this on the second pass
            int clean_up_to_index = Kokkos::atomic_fetch_add( &clean_up_to_count(), 1 );
            clean_up_to(clean_up_to_index) = write_to;

            return;
        }
        else {
            printf("%d is safe %d\n", n, pull_from);
        }

        printf("moving id %d %f %f %f to %d\n",
                pull_from,
                particles(pull_from, particle_var::dx),
                particles(pull_from, particle_var::dy),
                particles(pull_from, particle_var::dz),
                write_to);

        // Move the particle from np-n to pm->i
        particles(write_to, particle_var::dx) = particles(pull_from, particle_var::dx);
        particles(write_to, particle_var::dy) = particles(pull_from, particle_var::dy);
        particles(write_to, particle_var::dz) = particles(pull_from, particle_var::dz);
        particles(write_to, particle_var::ux) = particles(pull_from, particle_var::ux);
        particles(write_to, particle_var::uy) = particles(pull_from, particle_var::uy);
        particles(write_to, particle_var::uz) = particles(pull_from, particle_var::uz);
        particles(write_to, particle_var::w)  = particles(pull_from, particle_var::w);
        particles(write_to, particle_var::pi) = particles(pull_from, particle_var::pi);

        particles(pull_from, particle_var::dx) = 0.0;
        particles(pull_from, particle_var::dy) = 0.0;
        particles(pull_from, particle_var::dz) = 0.0;
        particles(pull_from, particle_var::ux) = 0.0;
        particles(pull_from, particle_var::uy) = 0.0;
        particles(pull_from, particle_var::uz) = 0.0;
        particles(pull_from, particle_var::w)  = 0.0;
        particles(pull_from, particle_var::pi) = 0.0;
    });

    Kokkos::deep_copy(clean_up_from_count_h, clean_up_from_count);
    Kokkos::deep_copy(clean_up_to_count_h, clean_up_to_count);

    Kokkos::parallel_for("compress clean up", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, clean_up_from_count_h() ), KOKKOS_LAMBDA (int n)
    {
        int write_to = clean_up_to(n);
        int pull_from = clean_up_from(n);

        printf("clean up id %d to %d \n", pull_from, write_to);
            printf("--> %f %f %f -> %f %f %f \n",
                particles(pull_from, particle_var::dx),
                particles(pull_from, particle_var::dy),
                particles(pull_from, particle_var::dz),
                particles(write_to, particle_var::dx),
                particles(write_to, particle_var::dy),
                particles(write_to, particle_var::dz)
            );

        particles(write_to, particle_var::dx) = particles(pull_from, particle_var::dx);
        particles(write_to, particle_var::dy) = particles(pull_from, particle_var::dy);
        particles(write_to, particle_var::dz) = particles(pull_from, particle_var::dz);
        particles(write_to, particle_var::ux) = particles(pull_from, particle_var::ux);
        particles(write_to, particle_var::uy) = particles(pull_from, particle_var::uy);
        particles(write_to, particle_var::uz) = particles(pull_from, particle_var::uz);
        particles(write_to, particle_var::w)  = particles(pull_from, particle_var::w);
        particles(write_to, particle_var::pi) = particles(pull_from, particle_var::pi);
    });

#endif

}

int vpic_simulation::advance(void) {
  species_t *sp;
  double err;

  // Determine if we are done ... see note below why this is done here

  printf("STEP %ld \n", step());
  if( num_step>0 && step()>=num_step ) return 0;

  // Sort the particles for performance if desired.

  LIST_FOR_EACH( sp, species_list )
    if( (sp->sort_interval>0) && ((step() % sp->sort_interval)==0) ) {
      if( rank()==0 ) MESSAGE(( "Performance sorting \"%s\"", sp->name ));
      TIC sort_p( sp ); TOC( sort_p, 1 );
    }

  // Replace sort with a kokkos sort
  // TODO: move this to a function
  // TODO: I suspect there is a simpler way to do this?
  /*
  typedef Kokkos::BinOp1D< KeyViewType > BinOp;
  BinOp bin_op(bin_max,min,max);
  Kokkos::BinSort< KeyViewType , BinOp >
  sorter(keys,bin_op,false);
  sorter.create_permute_vector();
  sorter.template sort<ViewType1>(view1);
  */


  /*
  LIST_FOR_EACH( sp, species_list )
    if( (sp->sort_interval>0) && ((step() % sp->sort_interval)==0) )
    {
      if( rank()==0 ) MESSAGE(( "Performance sorting \"%s\"", sp->name ));
          //TIC sort_p( sp ); TOC( sort_p, 1 );

      // For now we need to grab the keys or use a complex comparitor
      typedef decltype(sp->k_p_d) KeyViewType;
      typedef Kokkos::BinOp1D< KeyViewType > BinOp;

      // Pull out pi because I'm lazy
      //

      //Kokkos::BinSort<KeyViewType , BinOp > Sorter(element_,begin,end,binner,false);

      Kokkos::BinSort<KeyViewType,Comparator> bin_sort(keys, begin, end, comp, sort_within_bins );
      bin_sort.create_permute_vector();
      bin_sort.sort(element_,begin,end);
    }
    */

  // At this point, fields are at E_0 and B_0 and the particle positions
  // are at r_0 and u_{-1/2}.  Further the mover lists for the particles should
  // empty and all particles should be inside the local computational domain.
  // Advance the particle lists.

  if( species_list )
    TIC clear_accumulator_array( accumulator_array ); TOC( clear_accumulators, 1 );

  // Note: Particles should not have moved since the last performance sort
  // when calling collision operators.
  // FIXME: Technically, this placement of the collision operators only
  // yields a first order accurate Trotter factorization (not a second
  // order accurate factorization).

  if( collision_op_list )
  {
    TIC apply_collision_op_list( collision_op_list ); TOC( collision_model, 1 );
  }

  TIC user_particle_collisions(); TOC( user_particle_collisions, 1 );

  KOKKOS_INTERPOLATOR_VARIABLES();
  KOKKOS_ACCUMULATOR_VARIABLES();
  KOKKOS_PARTICLE_VARIABLES();

  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE();
  KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE();
  KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE();

  //int lna = 180;

  LIST_FOR_EACH( sp, species_list )
  {
      auto& particles = sp->k_p_d;
      print_particles_d(particles, sp->np); // should not see any zeros
      TIC advance_p( sp, accumulator_array, interpolator_array ); TOC( advance_p, 1 );
  }

  Kokkos::Experimental::contribute(accumulator_array->k_a_d, accumulator_array->k_a_sa);
  accumulator_array->k_a_sa.reset_except(accumulator_array->k_a_d);

  KOKKOS_COPY_ACCUMULATOR_MEM_TO_HOST();
  KOKKOS_COPY_PARTICLE_MEM_TO_HOST(); // TODO: this can ultimatimely be removed
  KOKKOS_COPY_INTERPOLATOR_MEM_TO_HOST();

  // TODO: think about if this is needed? It's done in advance_p
  //Kokkos::deep_copy(sp->k_pc_h, sp->k_pc_d);

  // Because the partial position push when injecting aged particles might
  // place those particles onto the guard list (boundary interaction) and
  // because advance_p requires an empty guard list, particle injection must
  // be done after advance_p and before guard list processing. Note:
  // user_particle_injection should be a stub if species_list is empty.

  if( emitter_list )
    TIC apply_emitter_list( emitter_list ); TOC( emission_model, 1 );
  TIC user_particle_injection(); TOC( user_particle_injection, 1 );

  // This should be after the emission and injection to allow for the
  // possibility of thread parallelizing these operations

  if( species_list )
    TIC reduce_accumulator_array( accumulator_array ); TOC( reduce_accumulators, 1 );

  // At this point, most particle positions are at r_1 and u_{1/2}. Particles
  // that had boundary interactions are now on the guard list. Process the
  // guard lists. Particles that absorbed are added to rhob (using a corrected
  // local accumulation).

  // This should mean the kokkos accum data is up to date
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE();

  TIC
    for( int round=0; round<num_comm_round; round++ )
    {
      //boundary_p( particle_bc_list, species_list, field_array, accumulator_array );
      boundary_p_kokkos( particle_bc_list, species_list, field_array, accumulator_array );
    }
  TOC( boundary_p, num_comm_round );

  // Boundary_p calls move_p, so we need to deal with the current
  // TODO: this will likely break on device?
  //
  //print_fields(accumulator_array->k_a_h);

  //Kokkos::Experimental::contribute(accumulator_array->k_a_h, accumulator_array->k_a_sah);
  //accumulator_array->k_a_sa.reset_except(accumulator_array->k_a_h);

  // Update device so we can pull it all the way back to the host
  Kokkos::deep_copy(accumulator_array->k_a_d, accumulator_array->k_a_h);
  KOKKOS_COPY_ACCUMULATOR_MEM_TO_HOST();
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
  LIST_FOR_EACH( sp, species_list ) {

      const int nm = sp->k_nm_h(0);

      // TODO: this can be hoisted to the end of advance_p if desired
      compress_particle_data(
              sp->k_p_d,
              sp->k_pm_d,
              nm,
              sp->np,
              sp
      );

      // Update np now we removed them...
      sp->np -= nm;

      auto& particles = sp->k_p_d;
      printf("Done compress, now print: \n");
      print_particles_d(particles, sp->np); // should not see any zeros
      printf("Done compress print: \n");

      // Copy data for copies back to device
      //Kokkos::deep_copy(sp->k_pc_d, sp->k_pc_h);

      auto& particle_copy = sp->k_pc_d;

      printf("particle copy size %ld particles size %ld max np %d \n", particle_copy.size() , particles.size(), sp->max_np);


      int num_to_copy = sp->num_to_copy;
      printf("Trying to append %d from particle copy where np = %d max nm %d \n", num_to_copy, sp->np, sp->max_nm);

      int np = sp->np;
      // Append it to the particles
      Kokkos::parallel_for("append moved particles", Kokkos::RangePolicy <
              Kokkos::DefaultExecutionSpace > (0, sp->num_to_copy-1), KOKKOS_LAMBDA
              (int i)
      {
        int npi = np+i; // i goes from 0..n so no need for -1
        printf("append to %d from %d \n", npi, i);
        particles(npi, particle_var::dx) = particle_copy(i, particle_var::dx);
        particles(npi, particle_var::dy) = particle_copy(i, particle_var::dy);
        particles(npi, particle_var::dz) = particle_copy(i, particle_var::dz);
        particles(npi, particle_var::ux) = particle_copy(i, particle_var::ux);
        particles(npi, particle_var::uy) = particle_copy(i, particle_var::uy);
        particles(npi, particle_var::uz) = particle_copy(i, particle_var::uz);
        particles(npi, particle_var::w)  = particle_copy(i, particle_var::w);
        particles(npi, particle_var::pi) = particle_copy(i, particle_var::pi);
      });

      // Reset this to zero now we've done the write back
      //sp->np += num_to_copy;
      sp->num_to_copy = 0;
  }

  // TODO: this can be removed once the below does not rely on host memory
  KOKKOS_COPY_PARTICLE_MEM_TO_HOST();

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
    int nm = sp->nm;
    particle_mover_t * RESTRICT ALIGNED(16)  pm = sp->pm + sp->nm - 1;
    particle_t * RESTRICT ALIGNED(128) p0 = sp->p;
    for (; nm; nm--, pm--) {
      int i = pm->i; // particle index we are removing
      p0[i].i >>= 3; // shift particle voxel down
      // accumulate the particle's charge to the mesh
      accumulate_rhob( field_array->f, p0+i, sp->g, sp->q );
      p0[i] = p0[sp->np-1]; // put the last particle into position i
      sp->np--; // decrement the number of particles
    }
    sp->nm = 0;
  }
  */

  // At this point, all particle positions are at r_1 and u_{1/2}, the
  // guard lists are empty and the accumulators on each processor are current.
  // Convert the accumulators into currents.

  TIC FAK->clear_jf( field_array ); TOC( clear_jf, 1 );
  if( species_list )
    TIC unload_accumulator_array( field_array, accumulator_array ); TOC( unload_accumulator, 1 );
  TIC FAK->synchronize_jf( field_array ); TOC( synchronize_jf, 1 );

  // At this point, the particle currents are known at jf_{1/2}.
  // Let the user add their own current contributions. It is the users
  // responsibility to insure injected currents are consistent across domains.
  // It is also the users responsibility to update rhob according to
  // rhob_1 = rhob_0 + div juser_{1/2} (corrected local accumulation) if
  // the user wants electric field divergence cleaning to work.

  TIC user_current_injection(); TOC( user_current_injection, 1 );

  KOKKOS_FIELD_VARIABLES();
  KOKKOS_COPY_FIELD_MEM_TO_DEVICE();

  // Half advance the magnetic field from B_0 to B_{1/2}
  TIC FAK->advance_b( field_array, 0.5 ); TOC( advance_b, 1 );

  KOKKOS_COPY_FIELD_MEM_TO_HOST();

  // Advance the electric field from E_0 to E_1

  TIC FAK->advance_e( field_array, 1.0 ); TOC( advance_e, 1 );

  // Let the user add their own contributions to the electric field. It is the
  // users responsibility to insure injected electric fields are consistent
  // across domains.

  TIC user_field_injection(); TOC( user_field_injection, 1 );

  // Half advance the magnetic field from B_{1/2} to B_1

  KOKKOS_COPY_FIELD_MEM_TO_DEVICE();

  TIC FAK->advance_b( field_array, 0.5 ); TOC( advance_b, 1 );

  KOKKOS_COPY_FIELD_MEM_TO_HOST();

  // Divergence clean e

  if( (clean_div_e_interval>0) && ((step() % clean_div_e_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Divergence cleaning electric field" ));

    TIC FAK->clear_rhof( field_array ); TOC( clear_rhof,1 );
    if( species_list ) TIC LIST_FOR_EACH( sp, species_list ) accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, species_list->id );
    TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );

    for( int round=0; round<num_div_e_round; round++ ) {
      TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
      if( round==0 || round==num_div_e_round-1 ) {
        TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
        if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
      }
      TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );
    }
  }

  // Divergence clean b

  if( (clean_div_b_interval>0) && ((step() % clean_div_b_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Divergence cleaning magnetic field" ));

    for( int round=0; round<num_div_b_round; round++ ) {
      TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
      if( round==0 || round==num_div_b_round-1 ) {
        TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
        if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
      }
      TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );
    }
  }

  // Synchronize the shared faces

  if( (sync_shared_interval>0) && ((step() % sync_shared_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Synchronizing shared tang e, norm b, rho_b" ));
    TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
    if( rank()==0 ) MESSAGE(( "Domain desynchronization error = %e (arb units)", err ));
  }

  // Fields are updated ... load the interpolator for next time step and
  // particle diagnostics in user_diagnostics if there are any particle
  // species to worry about

  KOKKOS_COPY_FIELD_MEM_TO_DEVICE();

  KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE();

  if( species_list ) TIC load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );

  KOKKOS_COPY_INTERPOLATOR_MEM_TO_HOST();
  KOKKOS_COPY_FIELD_MEM_TO_HOST();

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
  dump_energies("energies.txt", 1);

  return 1;
}

