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

int vpic_simulation::advance(void) {
      //if( rank()==0 ) MESSAGE(( "num_step, step = %i, %f \n", num_step, step() ));

  species_t *sp;
  double err;


  // Determine if we are done ... see note below why this is done here

  if( num_step>0 && step()>=num_step ) return 0;

  
  // Sort the particles for performance if desired.

  LIST_FOR_EACH( sp, species_list )
    if( (sp->sort_interval>0) && ((step() % sp->sort_interval)==0) ) {
      if( rank()==0 ) MESSAGE(( "Performance sorting \"%s\"", sp->name ));
      TIC sort_p( sp ); TOC( sort_p, 1 );
    } 
  
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
    TIC apply_collision_op_list( collision_op_list ); TOC( collision_model, 1 );
    TIC user_particle_collisions(); TOC( user_particle_collisions, 1 );

      LIST_FOR_EACH( sp, species_list )
    TIC hyb_advance_p( sp, accumulator_array, interpolator_array ); TOC( advance_p, 1 );


// Because the partial position push when injecting aged particles might
  // place those particles onto the guard list (boundary interaction) and
  // because advance_p requires an empty guard list, particle injection must
  // be done after advance_p and before guard list processing. Note:
  // user_particle_injection should be a stub if species_list is empty.

  TIC user_particle_injection(); TOC( user_particle_injection, 1 );
    
 
  // At this point, most particle positions are at r_1 and u_{1/2}. Particles
  // that had boundary interactions are now on the guard list. Process the
  // guard lists. Particles that absorbed are added to rhob (using a corrected
  // local accumulation).
   
  TIC
    for( int round=0; round<num_comm_round; round++ )
      hyb_boundary_p( particle_bc_list, species_list,
                  field_array, accumulator_array );  TOC( boundary_p, num_comm_round );
  LIST_FOR_EACH( sp, species_list ) {
    if( sp->nm && verbose )
      ERROR(( "Removing %i particles associated with unprocessed %s movers (increase num_comm_round)",
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
      //accumulate_rhob( field_array->f, p0+i, sp->g, sp->q );
      p0[i] = p0[sp->np-1]; // put the last particle into position i
      sp->np--; // decrement the number of particles
    }
    sp->nm = 0;
  }
  
  // At this point, all particle positions are at r_1 and u_{1/2}, the
  // guard lists are empty and the accumulators on each processor are current.
  // Convert the accumulators into currents.

 if( species_list ) TIC hyb_reduce_accumulator_array( accumulator_array ); TOC( reduce_accumulators, 1 );
  
 TIC FAK->clear_rhof( field_array); TOC( clear_rhof, 1 );
  if( species_list )
    TIC hyb_unload_accumulator_array( field_array, accumulator_array ); TOC(unload_accumulator, 1 );
 
  //  advance the magnetic field from B_0 to B_{1}. also advances E. fixes ghost cells for density/ion flow

  grid->isub=0;
  for(int i=0;i<grid->nsub;i++){
     TIC FAK->advance_b( field_array, 1.0/grid->nsub ); TOC( advance_b, 1 );
     grid->isub++;
  }

 // Let the user add their own contributions to the electric field. It is the
  // users responsibility to insure injected electric fields are consistent
  // across domains.

  TIC user_field_injection(); TOC( user_field_injection, 1 );

  if (grid->nsmb>0) { if (step()%grid->nsmb==0 ) FAK->smooth_b(field_array); }

 // get smooth fields for  interpolating
    int ism = grid->nsm;
	while(ism>0){
	 FAK->smooth_eb( field_array , 1); 
	ism--;
	}  
  // Fields are updated ... load the interpolator for next time step and
  // particle diagnostics in user_diagnostics if there are any particle
  // species to worry about

  if( species_list ) TIC hyb_load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );

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

  return 1;
}

