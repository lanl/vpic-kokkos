#ifndef _boundary_h_
#define _boundary_h_

#include "../species_advance/species_advance.h"
#include "../util/io/FileIO.h"

struct particle_bc;
typedef struct particle_bc particle_bc_t;

// may have been moved by Kevin
typedef struct link_boundary {
char fbase[256];	// base of file name to contain link info
double n_out;		// number of writes so far on this node (double to
						// accomodate long long runs)
} link_boundary_t;

/* In boundary.c */

int
num_particle_bc( const particle_bc_t * RESTRICT pbc_list );

void
delete_particle_bc_list( particle_bc_t * RESTRICT pbc_list );

particle_bc_t *
append_particle_bc( particle_bc_t * pbc,
                    particle_bc_t ** pbc_list );

int64_t
get_particle_bc_id( particle_bc_t * pbc );

void
pbd_buff_to_disk( pb_diagnostic_t * diag );

template<typename kpart_floats_t, typename kpart_voxel_t>
void pbd_write_to_buffer(species_t * RESTRICT sp,
                    const kpart_floats_t& kpart,
                    const kpart_voxel_t& kpart_i,
                    const int i){
    pb_diagnostic_t * diag = sp->pb_diag;
    if(!(diag->enable)) return;

    float* buff = diag->buff;
    size_t store = diag->store_counter;

    if(store==diag->bufflen) {
      fprintf(stderr, "Writing lost particles of species %s to disk while other"
              " processors are not.  You may want to increase bufflen or "
              "decrease write_interval.\n", sp->name);
        pbd_buff_to_disk(diag);
        store = diag->store_counter;
    }

    if(store>diag->bufflen) ERROR(( "Well, that shouldn't have happened." ));
    
    if(diag->write_ux) buff[store++] = kpart(i, particle_var::ux);
    if(diag->write_uy) buff[store++] = kpart(i, particle_var::uy);
    if(diag->write_uz) buff[store++] = kpart(i, particle_var::uz);

    if(diag->write_momentum_magnitude){
        buff[store++] = sqrt( pow(kpart(i, particle_var::ux),2)
                + pow(kpart(i, particle_var::uy),2)
                + pow(kpart(i, particle_var::uz),2) );
    }

    if(diag->write_posx | diag->write_posy | diag->write_posz){
        int ii = kpart_i(i);
        grid_t * RESTRICT grid = sp->g;
        float dx0 = kpart(i, particle_var::dx);
        float dy0 = kpart(i, particle_var::dy);
        float dz0 = kpart(i, particle_var::dz);
// These are copied from some tracer macros
#define nxg (grid->nx + 2)
#define nyg (grid->ny + 2)
#define nzg (grid->nz + 2)
#define i0 (ii%nxg)
#define j0 ((ii/nxg)%nyg)
#define k0 (ii/(nxg*nyg))
#define global_pos_x ((i0 + (dx0-1)*0.5) * grid->dx + grid->x0)
#define global_pos_y ((j0 + (dy0-1)*0.5) * grid->dy + grid->y0)
#define global_pos_z ((k0 + (dz0-1)*0.5) * grid->dz + grid->z0)
        if(diag->write_posx) buff[store++] = global_pos_x;
        if(diag->write_posy) buff[store++] = global_pos_y;
        if(diag->write_posz) buff[store++] = global_pos_z;
    }


    if(diag->write_weight) buff[store++] = kpart(i, particle_var::w);

    // TODO: Write the user values
    //if(diag->enable_user) Call the user function

    if(diag->store_counter+diag->num_writes != store)
        ERROR(( "That's pretty bad." ));
    diag->store_counter = store;
}

pb_diagnostic_t *
init_pb_diagnostic(species_t * sp);

void
finalize_pb_diagnostic(pb_diagnostic_t * diag);

/* In boundary_p.cxx */

void
boundary_p( particle_bc_t       * RESTRICT pbc_list,
            species_t           * RESTRICT sp_list,
            field_array_t       * RESTRICT fa,
            accumulator_array_t * RESTRICT aa );

void
boundary_p_kokkos( particle_bc_t       * RESTRICT pbc_list,
            species_t           * RESTRICT sp_list,
            field_array_t       * RESTRICT fa
        );

/* In maxwellian_reflux.c */

particle_bc_t *
maxwellian_reflux( species_t  * RESTRICT sp_list,
                   rng_pool_t * RESTRICT rp );

void
set_reflux_temp( /**/  particle_bc_t * RESTRICT mr,
                 const species_t     * RESTRICT sp,
                 float ut_para,
                 float ut_perp );

/* In absorb_tally.c */

particle_bc_t *
absorb_tally( /**/  species_t      * RESTRICT sp_list,
              const field_array_t  * RESTRICT fa );

int *
get_absorb_tally( particle_bc_t * pbc );

#endif /* _boundary_h_ */

