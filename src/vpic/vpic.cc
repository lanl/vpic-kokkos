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

/* Note that, when a vpic_simulation is created (and thus registered
   with the checkpt service), it is created empty; none of the simulation
   objects on which it depends have been created yet. (These get created
   as the simulation configuration is assembled in the input deck.) As
   such, vpic_simulation pointers do not point to older objects.  We
   thus need to use the checkpt_fptr functions and write a reanimator.

   FIXME: We could avoid this by giving each one of the objects pointed
   to proper resize semantics so we could then create the objects during
   vpic_simulation construction (as opposed to after it). */

void
checkpt_vpic_simulation( const vpic_simulation * vpic ) {
  CHECKPT( vpic, 1 );
  CHECKPT_PTR( vpic->entropy );
  CHECKPT_PTR( vpic->sync_entropy );
  CHECKPT_PTR( vpic->grid );
  CHECKPT_FPTR( vpic->material_list );
  CHECKPT_FPTR( vpic->field_array );
  CHECKPT_FPTR( vpic->interpolator_array );
  CHECKPT_FPTR( vpic->accumulator_array );
  CHECKPT_FPTR( vpic->hydro_array );
  CHECKPT_FPTR( vpic->species_list );
  CHECKPT_FPTR( vpic->particle_bc_list );
  CHECKPT_FPTR( vpic->emitter_list );
  CHECKPT_FPTR( vpic->collision_op_list );
}

vpic_simulation *
restore_vpic_simulation( void ) {
  vpic_simulation * vpic;
  RESTORE( vpic );
  RESTORE_PTR( vpic->entropy );
  RESTORE_PTR( vpic->sync_entropy );
  RESTORE_PTR( vpic->grid );
  RESTORE_FPTR( vpic->material_list );
  RESTORE_FPTR( vpic->field_array );
  RESTORE_FPTR( vpic->interpolator_array );
  RESTORE_FPTR( vpic->accumulator_array );
  RESTORE_FPTR( vpic->hydro_array );
  RESTORE_FPTR( vpic->species_list );
  RESTORE_FPTR( vpic->particle_bc_list );
  RESTORE_FPTR( vpic->emitter_list );
  RESTORE_FPTR( vpic->collision_op_list );
  return vpic;
}

void
reanimate_vpic_simulation( vpic_simulation * vpic ) {
  REANIMATE_FPTR( vpic->material_list );
  REANIMATE_FPTR( vpic->field_array );
  REANIMATE_FPTR( vpic->interpolator_array );
  REANIMATE_FPTR( vpic->accumulator_array );
  REANIMATE_FPTR( vpic->hydro_array );
  REANIMATE_FPTR( vpic->species_list );
  REANIMATE_FPTR( vpic->particle_bc_list );
  REANIMATE_FPTR( vpic->emitter_list );
  REANIMATE_FPTR( vpic->collision_op_list );
}


vpic_simulation::vpic_simulation() {
  CLEAR( this, 1 );

  /* Set non-zero defaults */
  verbose = 1;
  num_comm_round = 3;
  num_div_e_round = 2;
  num_div_b_round = 2;

  int                           n_rng = serial.n_pipeline;
  if( n_rng<thread.n_pipeline ) n_rng = thread.n_pipeline;
# if defined(CELL_PPU_BUILD) && defined(USE_CELL_SPUS)
  if( n_rng<spu.n_pipeline    ) n_rng = spu.n_pipeline;
# endif
  n_rng++;

  entropy      = new_rng_pool( n_rng, 0, 0 );
  sync_entropy = new_rng_pool( n_rng, 0, 1 );
  grid = new_grid();

  REGISTER_OBJECT( this, checkpt_vpic_simulation,
                   restore_vpic_simulation, reanimate_vpic_simulation );
}

vpic_simulation::~vpic_simulation() {
  UNREGISTER_OBJECT( this );
  delete_emitter_list( emitter_list );
  delete_particle_bc_list( particle_bc_list );
  delete_species_list( species_list );
  delete_hydro_array( hydro_array );
  delete_accumulator_array( accumulator_array );
  delete_interpolator_array( interpolator_array );
  delete_field_array( field_array );
  delete_material_list( material_list );
  delete_grid( grid );
  delete_rng_pool( sync_entropy );
  delete_rng_pool( entropy );
  Kokkos::finalize();
}

void vpic_simulation::print_run_details()
{
    if (rank() == 0)
    {
        species_t* sp = nullptr;
        // Read run details and print them out
        // Focus on performance detemring quantities, and allow the deck to print
        // physics focused params:
        // num steps, nx, ny, nz, num particles per species
        std::cout << "######### Run Details ##########" << std::endl;
        std::cout << "## Global:" << std::endl;
        std::cout << "  # Num Step " << num_step << std::endl;
        std::cout << "  # px " << px << " py " << py << " pz " << pz << std::endl;
        std::cout << "  # gnx " << px*grid->nx << " gny " << py*grid->ny << " gnz " << pz*grid->nz << std::endl;
        std::cout << "## Local:" << std::endl;
        std::cout << "  # nx " << grid->nx << " ny " << grid->ny << " nz " << grid->nz << std::endl;
        std::cout << "  # dx " << grid->dx << " dy " << grid->dy << " dz " << grid->dz << std::endl;
        if (species_list )
        {
            std::cout << "## Particle Species: " <<  num_species( species_list ) << std::endl;
            LIST_FOR_EACH( sp, species_list )
            {
                std::cout << "  # " << sp->name << " np " << sp->np << std::endl;
            }
        }
        std::cout << "######### End Run Details ######" << std::endl;
        std::cout << std::endl; // blank line
    }
}


/**
 * @brief After a checkpoint restore, we must move the data back over to the
 * Kokkos objects. This currently must be done for all views
 */
void restore_kokkos(vpic_simulation& simulation)
{
    // I think the way the restore works by copying bytes it messes with the
    // reference counting of the built in Kokkos objects, and thus we have to
    // manual handle a bunch of the data copies

    // TODO: I can probably reach into Kokkos internals and tell it not to
    // decrement the counter for the nonesne object that was reanimated. If I
    // don't it's likely going to do bad stuff at program any way end..

    // TODO: the created objects below need to call the helper creators because
    // they won't get registered otherwise...
    // TODO: changing the objects like this is scary as I can't dealloc them
    // but I need to unregister then from the checkpoint syustem

    // TODO: the grid is hidden in a bunch of objects.. it will be hard to find
    // them all

    // TODO: this should be done on the views insnide
    //simulation.grid = new (simulation.grid) grid_t();

    // This restore methods relies on 'placement new'. Our approach is this:
    // 1) Use placement new to overwrite the garbage from the restart (note: we
    // can throw away the resulting pointer, as it overwrite in place)
    // 2) Overwrite that be doing normal init
    // We may be able to do that one in one step, but this way is clearer

    // Restore Particles
    species_t* sp;
    LIST_FOR_EACH( sp, simulation.species_list )
    {
        // TODO: we can bury this in the class
        new(&sp->k_p_d) k_particles_t();
        new(&sp->k_p_i_d) k_particles_i_t();
        new(&sp->k_pc_d) k_particle_copy_t::HostMirror();
        new(&sp->k_pc_i_d) k_particle_i_copy_t::HostMirror();
        new(&sp->k_pr_h) k_particle_copy_t::HostMirror();
        new(&sp->k_pr_i_h) k_particle_i_copy_t::HostMirror();
        new(&sp->k_pm_d) k_particle_movers_t();
        new(&sp->k_pm_i_d) k_particle_i_movers_t();
        new(&sp->k_nm_d) k_counter_t();
        new(&sp->k_nm_h) k_counter_t::HostMirror();

        new(&sp->k_p_h) k_particles_t::HostMirror();
        new(&sp->k_p_i_h) k_particles_i_t::HostMirror();

        new(&sp->k_pc_h) k_particle_copy_t::HostMirror();
        new(&sp->k_pc_i_h) k_particle_i_copy_t::HostMirror();

        new(&sp->k_pm_h) k_particle_movers_t::HostMirror();
        new(&sp->k_pm_i_h) k_particle_i_movers_t::HostMirror();

        sp->init_kokkos_particles();

        simulation.KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE_SP(sp);
    }

    int nv = simulation.grid->nv;

    // Restore field array
    field_array_t* fa = simulation.field_array;
    new(&fa->k_f_d) k_field_t();
    new(&fa->k_fe_d) k_field_edge_t();
    new(&fa->k_f_h) k_field_t::HostMirror();
    new(&fa->k_fe_h) k_field_edge_t::HostMirror();

    // TODO: restore fb or make it a pointer to avoid it getting filled with
    // garbage on restore

    new(&fa->k_f_rhob_accum_d) k_field_accum_t();
    new(&fa->k_f_rhob_accum_h) k_field_accum_t::HostMirror();

    grid_t* grid = simulation.grid;

    // TODO: this xxx_sz calculation is duplicated in sfa.cc and could be DRYed
    int nx = grid->nx;
    int ny = grid->ny;
    int nz = grid->nz;
    int xyz_sz = 2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz;
    int yzx_sz = 2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx;
    int zxy_sz = 2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny;
    fa->init_kokkos_fields( nv, xyz_sz, yzx_sz, zxy_sz );

    simulation.KOKKOS_COPY_FIELD_MEM_TO_DEVICE(simulation.field_array);

    // Restore Material Data
    sfa_params_t* params = reinterpret_cast<sfa_params_t*>(fa->params);
    new(&params->k_mc_d) k_material_coefficient_t();
    new(&params->k_mc_h) k_material_coefficient_t::HostMirror();

    params->init_kokkos_sfa_params(params->n_materials);
    params->populate_kokkos_data();

    // Restore accumulators
    accumulator_array_t* accum = simulation.accumulator_array;
    //int num_accum = old_accum->na;

    new(&accum->k_a_d) k_accumulators_t();
    new(&accum->k_a_h) k_accumulators_t::HostMirror();
    new(&accum->k_a_sa) k_accumulators_sa_t();

    std::cout << "make accum of size " << accum->na << std::endl;
    accum->init_kokoks_accum(accum->na);

    // Restore interpolators
    interpolator_array_t* interp = simulation.interpolator_array;

    new(&interp->k_i_d) k_interpolator_t();
    new(&interp->k_i_h) k_interpolator_t::HostMirror();

    interp->init_kokkos_interp(nv);
    simulation.KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE(interp);

    // Restore Grid/Neighbors
    new(&grid->k_neighbor_d) k_neighbor_t();
    new(&grid->k_neighbor_h) k_neighbor_t::HostMirror();

    auto nfaces_per_voxel = 6;

    // also restores the neighbors
    grid->init_kokkos_grid(nfaces_per_voxel*nv);


    /* // All this code tries to recreate the objects and then use then
     * populate the Kokkos data, this has limitations and changes the actual
     * underlying pointers
    // Restore Grid (+Neighbors)
    //grid_t* old_grid = simulation->grid;
    //grid_t* new_grid = new_grid();

    //auto nfaces_per_voxel = 6;
    //simulation.grid->init_kokkos_grid(nfaces_per_voxel * nv);

    // Restore Particles
    species_t* sp;

    // As part of this, we have to re-write the species list.... that's lame.
    species_t* prev = nullptr;
    species_t* head = simulation.species_list;

    LIST_FOR_EACH( sp, simulation.species_list )
    {
        // Create a new species object by hand
        species_t* sp2 = new species_t(sp->np, sp->nm);

        // Update the last guy in the list to point to this guy
        if (prev != nullptr)
        {
            prev->next = sp2;
        }

        // Fill in the data it needs
        // TODO: This could basically be a copy constructor
        sp2->name = sp->name;
        sp2->q = sp->q;
        sp2->m = sp->m;
        sp2->np = sp->np;
        sp2->max_np = sp->max_np;
        sp2->p = sp->p;
        sp2->nm = sp->nm;
        sp2->max_nm = sp->max_nm;
        sp2->pm = sp->pm;
        sp2->last_sorted = sp->last_sorted;
        sp2->sort_interval = sp->sort_interval;
        sp2->sort_out_of_place = sp->sort_out_of_place;
        sp2->partition = sp->partition;
        sp2->g = sp->g;
        sp2->id = sp->id;
        sp2->next = sp->next;

        // init it
        sp2->init_kokkos_particles();

        if (sp == simulation.species_list)
        {
            std::cout << "repointing head" << std::endl;
            head = sp2;
        }

        // Swap it in
        // delete(sp); // This techincally leaks the data of size species, but
        // the way we checkpoint the species is currently not safe...
        // This was ~552 bytes per species
        //std::cout << sizeof(*sp) << std::endl;
        std::cout << sp->nm << std::endl;

        simulation.KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE_SP(sp2);

        prev = sp2;

    }
    simulation.species_list = head;

    int nv = simulation.grid->nv;

    // Restore fields
    field_array_t* new_fa = new field_array_t(nv);
    field_array_t* old_fa = simulation.field_array;

    new_fa->f = old_fa->f;
    new_fa->g = old_fa->g;
    new_fa->params = old_fa->params;
    new_fa->kernel[0] = old_fa->kernel[0];

    simulation.field_array = new_fa;
    //simulation.field_array->init_kokkos_fields(simulation.grid->nv);
    simulation.KOKKOS_COPY_FIELD_MEM_TO_DEVICE(simulation.field_array);

    // Restore accumulators
    accumulator_array_t* old_accum = simulation.accumulator_array;
    int num_accum = old_accum->na;
    accumulator_array_t* new_accum = new accumulator_array_t(num_accum);

    new_accum->a = old_accum->a;
    new_accum->n_pipeline = old_accum->n_pipeline;
    new_accum->stride = old_accum->stride;
    new_accum->na = old_accum->na;
    new_accum->g = old_accum->g;

    simulation.accumulator_array = new_accum;

    // restore interpolators?
    interpolator_array_t* new_interp = new interpolator_array_t(nv);
    interpolator_array_t* old_interp = simulation.interpolator_array;

    new_interp->i = old_interp->i;
    new_interp->g = old_interp->g;

    simulation.interpolator_array = new_interp;
    */
}

