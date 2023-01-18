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
  CHECKPT_FPTR( vpic->hydro_array );
  CHECKPT_FPTR( vpic->species_list );
  CHECKPT_FPTR( vpic->particle_bc_list );
  CHECKPT_FPTR( vpic->emitter_list );
  //CHECKPT_FPTR( vpic->collision_op_list );
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
  RESTORE_FPTR( vpic->hydro_array );
  RESTORE_FPTR( vpic->species_list );
  RESTORE_FPTR( vpic->particle_bc_list );
  RESTORE_FPTR( vpic->emitter_list );
  //RESTORE_FPTR( vpic->collision_op_list );
  return vpic;
}

void
reanimate_vpic_simulation( vpic_simulation * vpic ) {
  REANIMATE_FPTR( vpic->material_list );
  REANIMATE_FPTR( vpic->field_array );
  REANIMATE_FPTR( vpic->interpolator_array );
  REANIMATE_FPTR( vpic->hydro_array );
  REANIMATE_FPTR( vpic->species_list );
  REANIMATE_FPTR( vpic->particle_bc_list );
  REANIMATE_FPTR( vpic->emitter_list );
  //REANIMATE_FPTR( vpic->collision_op_list );
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
  delete_interpolator_array( interpolator_array );
  delete_field_array( field_array );
  delete_material_list( material_list );
  delete_grid( grid );
  delete_rng_pool( sync_entropy );
  delete_rng_pool( entropy );
  Kokkos::finalize();
}

/**
 * @brief Helper function to print run details in a formatted way. Useful for
 * both having a clear view of what the run is, but also structured enough to
 * be parsable by tools
 */
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
 * @brief The checkpoint macros will not work on the Kokkos views, so we bypass
 * the checkpointing infrustructure and manually write this data to disk for
 * all views without a legacy array.
 *
 * @param simulation The vpic_simulation that was restored
 * @param fbase The base name for the checkpoint files
 */
void checkpt_kokkos(vpic_simulation& simulation, const char* fbase)
{
# define PBUF_SIZE 32768 // 1MB of particles
#ifndef USE_LEGACY_PARTICLE_ARRAY
    char fname[256];
    FileIO fileIO;
    int buf_start;
    static particle_t * ALIGNED(128) p_buf = NULL;
    if( !p_buf ) MALLOC_ALIGNED( p_buf, PBUF_SIZE, 128 );
    auto& pbuf = p_buf;

    species_t* sp;
    LIST_FOR_EACH( sp, simulation.species_list )
    {
        sprintf( fname, "%s.%s", fbase, sp->name );
        FileIOStatus status = fileIO.open(fname, io_write);
        if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));

        // Copy a PBUF_SIZE hunk of the particle list into the particle buffer,
        // and write it out.  This is simplified from dump_particles since we
        // don't need to call center_p.
        int bufsize = PBUF_SIZE;
        for( buf_start=0; buf_start<sp->np; buf_start += PBUF_SIZE ) {
            if (buf_start + bufsize > sp->np) bufsize = sp->np - buf_start;
            Kokkos::parallel_for("Populate particle dump buffer",
                    host_execution_policy(0, bufsize),
                    KOKKOS_LAMBDA (int i) {

                    pbuf[i].dx = sp->k_p_h(buf_start + i, particle_var::dx);
                    pbuf[i].dy = sp->k_p_h(buf_start + i, particle_var::dy);
                    pbuf[i].dz = sp->k_p_h(buf_start + i, particle_var::dz);
                    pbuf[i].ux = sp->k_p_h(buf_start + i, particle_var::ux);
                    pbuf[i].uy = sp->k_p_h(buf_start + i, particle_var::uy);
                    pbuf[i].uz = sp->k_p_h(buf_start + i, particle_var::uz);
                    pbuf[i].w  = sp->k_p_h(buf_start + i, particle_var::w);
                    pbuf[i].i  = sp->k_p_i_h(buf_start + i);

                    });
            fileIO.write( p_buf, bufsize );
        }
        if( fileIO.close() ) ERROR(("File close failed on checkpt_kokkos particles!!!"));
    }
    pbuf = NULL;
    FREE_ALIGNED(p_buf);

#endif

}

/**
 * @brief After a checkpoint restore, we must move the data back over to the
 * Kokkos objects. This currently must be done for all views
 *
 * @param simulation The vpic_simulation that was restored
 * @param fbase The base name for the checkpoint files
 */
void restore_kokkos(vpic_simulation& simulation, const char * fbase)
{
    // The way the VPIC checkpoint/restore works is by copying raw bytes and
    // pointers.  It messes with the reference counting built into Kokkos, and
    // thus we have to manual handle a bunch of the data copies / fix the
    // Kokkos data. When a view gets restored, its unsafe to delete. This code
    // fixes that (but may leak a few kb per restart...)

    // This restore methods relies on 'placement new'. Our approach is this:
    // 1) Use placement new to overwrite the garbage from the restart (note: we
    // can throw away the resulting pointer, as it overwrite in place)
    // 2) Overwrite that be doing normal init
    // We may be able to do that one in one step, but this way is clearer

#ifndef USE_LEGACY_PARTICLE_ARRAY
    // Restore Particles
    char fname[256];
    FileIO fileIO;
    int buf_start;
    static particle_t * ALIGNED(128) p_buf = NULL;
    if( !p_buf ) MALLOC_ALIGNED( p_buf, PBUF_SIZE, 128 );
    auto& pbuf = p_buf;
#endif

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

        new(&sp->unsafe_index) Kokkos::View<int*>();
        new(&sp->clean_up_to_count) Kokkos::View<int>();
        new(&sp->clean_up_from_count) Kokkos::View<int>();
        new(&sp->clean_up_from_count_h) Kokkos::View<int>::HostMirror();
        new(&sp->clean_up_from) Kokkos::View<int*>();
        new(&sp->clean_up_to) Kokkos::View<int*>();

        sp->init_kokkos_particles();

#ifndef USE_LEGACY_PARTICLE_ARRAY
        sprintf( fname, "%s.%s", fbase, sp->name );
        FileIOStatus status = fileIO.open(fname, io_read);
        if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));

        // Copy a PBUF_SIZE hunk of the particle list into the particle buffer,
        // and write it out.  This is simplified from dump_particles since we
        // don't need to call center_p.
        int bufsize = PBUF_SIZE;
        for( buf_start=0; buf_start<sp->np; buf_start += PBUF_SIZE ) {
            if (buf_start + bufsize > sp->np) bufsize = sp->np - buf_start;
            fileIO.read( p_buf, bufsize );
            Kokkos::parallel_for("Populate particle dump buffer",
                    host_execution_policy(0, bufsize),
                    KOKKOS_LAMBDA (int i) {

                    sp->k_p_h(buf_start + i, particle_var::dx) = pbuf[i].dx;
                    sp->k_p_h(buf_start + i, particle_var::dy) = pbuf[i].dy;
                    sp->k_p_h(buf_start + i, particle_var::dz) = pbuf[i].dz;
                    sp->k_p_h(buf_start + i, particle_var::ux) = pbuf[i].ux;
                    sp->k_p_h(buf_start + i, particle_var::uy) = pbuf[i].uy;
                    sp->k_p_h(buf_start + i, particle_var::uz) = pbuf[i].uz;
                    sp->k_p_h(buf_start + i, particle_var::w)  = pbuf[i].w ;
                    sp->k_p_i_h(buf_start + i) = pbuf[i].i;

                    });
        }
        
        if( fileIO.close() ) ERROR(("File close failed in restore_kokkos!!!"));
#endif

        sp->copy_to_device();
    }
#ifndef USE_LEGACY_PARTICLE_ARRAY
    pbuf = NULL;
    FREE_ALIGNED(p_buf);
#endif
#undef PBUF_SIZE

    int nv = simulation.grid->nv;

    // Restore field array
    field_array_t* fa = simulation.field_array;
    new(&fa->k_f_d) k_field_t();
    new(&fa->k_field_sa_d) k_field_sa_t();
    new(&fa->k_fe_d) k_field_edge_t();
    new(&fa->k_f_h) k_field_t::HostMirror();
    new(&fa->k_fe_h) k_field_edge_t::HostMirror();

    new(&fa->k_f_rhob_accum_d) k_field_accum_t();
    new(&fa->k_f_rhob_accum_h) k_field_accum_t::HostMirror();

    new(&fa->k_jf_accum_d) k_jf_accum_t();
    new(&fa->k_jf_accum_h) k_jf_accum_t::HostMirror();

    grid_t* grid = simulation.grid;

    // TODO: this xxx_sz calculation is duplicated in sfa.cc and could be DRYed
    int nx = grid->nx;
    int ny = grid->ny;
    int nz = grid->nz;
    int xyz_sz = 2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz;
    int yzx_sz = 2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx;
    int zxy_sz = 2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny;
    fa->init_kokkos_fields( nv, xyz_sz, yzx_sz, zxy_sz );

    simulation.field_array->copy_to_device();

    // Restore hydro array
    hydro_array_t* ha = simulation.hydro_array;
    new(&ha->k_h_d) k_hydro_d_t();
    new(&ha->k_h_h) k_hydro_d_t::HostMirror();
    ha->k_h_d = k_hydro_d_t("k_hydro", nv);
    ha->k_h_h = Kokkos::create_mirror_view(ha->k_h_d);
    // No need to populate hydro


    // Restore Material Data
    sfa_params_t* params = reinterpret_cast<sfa_params_t*>(fa->params);
    new(&params->k_mc_d) k_material_coefficient_t();
    new(&params->k_mc_h) k_material_coefficient_t::HostMirror();

    params->init_kokkos_sfa_params(params->n_materials);
    params->populate_kokkos_data();

    // Restore interpolators
    interpolator_array_t* interp = simulation.interpolator_array;

    new(&interp->k_i_d) k_interpolator_t();
    new(&interp->k_i_h) k_interpolator_t::HostMirror();

    interp->init_kokkos_interp(nv);
    interp->copy_to_device();

    // Restore Grid/Neighbors
    new(&grid->k_neighbor_d) k_neighbor_t();
    new(&grid->k_neighbor_h) k_neighbor_t::HostMirror();

    auto nfaces_per_voxel = 6;

    // also restores the neighbors
    grid->init_kokkos_grid(nfaces_per_voxel*nv);

}
