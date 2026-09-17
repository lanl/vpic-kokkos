#ifndef _boundary_h_
#define _boundary_h_

#include "../species_advance/species_advance.h"
#include "../util/io/FileIO.h"

struct particle_bc;
typedef struct particle_bc particle_bc_t;

/* In boundary.c */

int
num_particle_bc( const particle_bc_t * RESTRICT pbc_list );

void
delete_particle_bc_list( particle_bc_t * RESTRICT pbc_list );

void
checkpt_pbd(pb_diagnostic_t *diag);

pb_diagnostic_t *
restore_pbd(void);

void
delete_pbd(pb_diagnostic_t *diag);

particle_bc_t *
append_particle_bc( particle_bc_t * pbc,
                    particle_bc_t ** pbc_list );

int64_t
get_particle_bc_id( particle_bc_t * pbc );

void
pbd_buff_to_disk( pb_diagnostic_t * diag );

//template<typename kpart_floats_t, typename kpart_voxel_t, typename KokkosBitset>
//void pbd_write_to_buffer(species_t * RESTRICT sp,
//                    const kpart_floats_t& kpart,
//                    const kpart_voxel_t& kpart_i,
//                    const KokkosBitset& apply){
//    pb_diagnostic_t * diag = sp->pb_diag;
//    if(!(diag->enable)) return;
//
//    Kokkos::View<float*> buff_d("pbd buffer", diag->bufflen);
//    Kokkos::View<size_t[1]> store_d("buffer offset");
//    auto store_h = Kokkos::create_mirror_view(store_d);
//    store_h(0) = diag->store_counter;
//    Kokkos::deep_copy(store_d, store_h);
//
//    float* buff = diag->buff;
//    size_t store = diag->store_counter;
//
//    if(store+apply.count() >= diag->bufflen) {
//      fprintf(stderr, "Writing lost particles of species %s to disk while other"
//              " processors are not.  You may want to increase bufflen or "
//              "decrease write_interval.\n", sp->name);
//        pbd_buff_to_disk(diag);
//        store = diag->store_counter;
//    }
//
//    if(store>diag->bufflen) ERROR(( "Well, that shouldn't have happened." ));
//    
//    const bool write_posx = diag->write_posx;
//    const bool write_posy = diag->write_posy;
//    const bool write_posz = diag->write_posz;
//    const bool write_ux = diag->write_ux;
//    const bool write_uy = diag->write_uy;
//    const bool write_uz = diag->write_uz;
//    const bool write_wt = diag->write_weight;
//    const bool write_mom_mag = diag->write_momentum_magnitude;
//    const int nxg = grid->nx, nyg = grid->ny, nzg = grid->nz;
//    const int dx = grid->dx, dy = grid->dy, dz = grid->dz;
//    const int x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
//
//    int floats_per_particle = 0;
//    floats_per_particle += write_posx ? 1 : 0;
//    floats_per_particle += write_posy ? 1 : 0;
//    floats_per_particle += write_posz ? 1 : 0;
//    floats_per_particle += write_ux ? 1 : 0;
//    floats_per_particle += write_uy ? 1 : 0;
//    floats_per_particle += write_uz ? 1 : 0;
//    floats_per_particle += write_wt ? 1 : 0;
//    floats_per_particle += write_mom_mag ? 1 : 0;
//    Kokkos::parallel_for("Write pbd to buffer", Kokkos::RangePolicy(0, kpart.extent(0)), KOKKOS_LAMBDA(const int i) {
//      if(apply.test(i)) {
//        size_t offset = Kokkos::atomic_fetch_add(&store_d(0), floats_per_particle);
//        if(write_ux)
//          buff_d(offset++) = kpart(i, particle_var::ux);
//        if(write_uy)
//          buff_d(offset++) = kpart(i, particle_var::uy);
//        if(write_uz)
//          buff_d(offset++) = kpart(i, particle_var::uz);
//        if(write_mom_mag) {
//          const float ux = kpart(i, particle_var::ux);
//          const float uy = kpart(i, particle_var::uy);
//          const float uz = kpart(i, particle_var::uz);
//          buff_d(offset++) = Kokkos::sqrt( ux*ux + uy*uy + uz*uz );
//        }
//        if(write_posx | write_posy | write_posz) {
//          const int ii = kpart_i(i);
//          const float dx0 = kpart(i, particle_var::dx);
//          const float dy0 = kpart(i, particle_var::dy);
//          const float dz0 = kpart(i, particle_var::dz);
//          const int i0 = (ii%nxg);
//          const int j0 = ((ii/nxg)%nyg);
//          const int k0 = (ii/(nxg*nyg));
//          const int global_pos_x = ((i0 + (dx0-1)*0.5) * dx + x0);
//          const int global_pos_y = ((j0 + (dy0-1)*0.5) * dy + y0);
//          const int global_pos_z = ((k0 + (dz0-1)*0.5) * dz + z0);
//          if(write_posx)
//            buff_d(offset++) = global_pos_x;
//          if(write_posy)
//            buff_d(offset++) = global_pos_y;
//          if(write_posz)
//            buff_d(offset++) = global_pos_z;
//        }
//        if(write_wt)
//          buff_d(offset++) = kpart(i, particle_var::w);
//      }
//    });
//    Kokkos::deep_copy(store_h, store_d);
//    Kokkos::View<float*, Kokkos::MemoryUnmanaged> buff_h(buff+diag->store_counter, diag->bufflen - diag->store_counter);
//
//    // TODO: Write the user values
//    //if(diag->enable_user) Call the user function
//
//    if(diag->store_counter+diag->num_writes != store_h(0))
//        ERROR(( "That's pretty bad." ));
//    diag->store_counter = store(h);
//}

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
init_pb_diagnostic();

void
finalize_pb_diagnostic(species_t * sp);

/* In boundary_p.cxx */

void
boundary_p( particle_bc_t       * RESTRICT pbc_list,
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

