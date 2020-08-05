#define IN_boundary
#include "boundary_private.h"

/* Private interface *********************************************************/

void
checkpt_particle_bc_internal( const particle_bc_t * RESTRICT pbc ) {
  CHECKPT( pbc, 1 );
  CHECKPT_SYM( pbc->interact );
  CHECKPT_SYM( pbc->delete_pbc );
  CHECKPT_PTR( pbc->next );
}

particle_bc_t *
restore_particle_bc_internal( void * params ) {
  particle_bc_t * pbc;
  RESTORE( pbc );
  pbc->params = params;
  RESTORE_SYM( pbc->interact );
  RESTORE_SYM( pbc->delete_pbc );
  RESTORE_PTR( pbc->next );
  return pbc;
}

particle_bc_t *
new_particle_bc_internal( void * params,
                          particle_bc_func_t interact,
                          delete_particle_bc_func_t delete_pbc,
                          checkpt_func_t checkpt,
                          restore_func_t restore,
                          reanimate_func_t reanimate ) {
  particle_bc_t * pbc;
  MALLOC( pbc, 1 );
  CLEAR( pbc, 1 );
  pbc->params     = params;
  pbc->interact   = interact;
  pbc->delete_pbc = delete_pbc;
  /* id, next set by append_particle_bc */
  REGISTER_OBJECT( pbc, checkpt, restore, reanimate );
  return pbc;
}

void
delete_particle_bc_internal( particle_bc_t * pbc ) {
  UNREGISTER_OBJECT( pbc );
  FREE( pbc );
}

/* Public interface **********************************************************/

int
num_particle_bc( const particle_bc_t * RESTRICT pbc_list ) {
  return pbc_list ? (-pbc_list->id-2) : 0;
}

void
delete_particle_bc_list( particle_bc_t * pbc_list ) {
  particle_bc_t * pbc;
  while( pbc_list ) {
    pbc = pbc_list;
    pbc_list = pbc_list->next;
    pbc->delete_pbc( pbc );
  }
}

particle_bc_t *
append_particle_bc( particle_bc_t * pbc,
                    particle_bc_t ** pbc_list ) {
  if( !pbc || !pbc_list ) ERROR(( "Bad args" ));
  if( pbc->next ) ERROR(( "Particle boundary condition already in a list" ));
  // Assumes reflective/absorbing are -1, -2
  pbc->id   = -3-num_particle_bc( *pbc_list );
  pbc->next = *pbc_list;
  *pbc_list = pbc;
  return pbc;
}

int64_t
get_particle_bc_id( particle_bc_t * pbc ) {
  if( !pbc ) return 0;
  if( pbc==(particle_bc_t *) absorb_particles ) return  absorb_particles;
  if( pbc==(particle_bc_t *)reflect_particles ) return reflect_particles;
  return pbc->id;
}

#define BUFLEN (256)
// A file size in sizeof(float) that is easy on your filesystem.
// Will be exceeded by up to bufflen*sizeof(float)
#define FRIENDLY_FILE_SIZE 268435456 // 1 GiB

pb_diagnostic_t *
init_pb_diagnostic(species_t * sp) {
    pb_diagnostic_t * diag;
    MALLOC( diag, 1);

    diag->enable = 0;
    diag->enable_user = 0;
    diag->sp = sp;
    
    MALLOC(diag->fname, BUFLEN);
    CLEAR(diag->fname, BUFLEN);
    sprintf(diag->fname, "pb_diagnostic/%s.%i.", sp->name, world_rank);

    diag->bufflen = pow(2,20); // Somewhat arbitrary
    diag->buff = NULL;

    diag->num_user_writes = 0;
    diag->num_writes = 0;

    diag->write_ux = 0;
    diag->write_uy = 0;
    diag->write_uz = 0;
    diag->write_momentum_magnitude = 0;
    diag->write_posx = 0;
    diag->write_posy = 0;
    diag->write_posz = 0;
    diag->write_weight = 0;

    return diag;
}

void
finalize_pb_diagnostic(pb_diagnostic_t * diag){
    if(diag->write_ux) diag->num_writes += 1;
    if(diag->write_uy) diag->num_writes += 1;
    if(diag->write_uz) diag->num_writes += 1;
    if(diag->write_momentum_magnitude) diag->num_writes += 1;
    if(diag->write_posx) diag->num_writes += 1;
    if(diag->write_posy) diag->num_writes += 1;
    if(diag->write_posz) diag->num_writes += 1;
    if(diag->write_weight) diag->num_writes += 1;

    diag->num_writes += diag->num_user_writes;

    if(diag->num_writes > 0) diag->enable = 1;
    else return;

    // Should be a multiple of num_writes
    diag->bufflen = diag->bufflen/diag->num_writes*diag->num_writes;

    MALLOC(diag->buff, diag->bufflen);

    diag->file_counter = -1;
    diag->store_counter = 0;
    diag->write_counter = 0;

    //fprintf(stderr, "For species %s, there are %d writes per particle.\n", diag->sp->name, diag->num_writes);
}

void
pbd_buff_to_disk( pb_diagnostic_t * diag ){
    if(diag==NULL) return;
    if(diag->store_counter == 0) return;

    size_t store = diag->store_counter;
    size_t write = diag->write_counter;

    FileIO fileIO;
    FileIOStatus status;
    char fname[BUFLEN];

    // Append the buffer to an existing file if small enough
    if(write < FRIENDLY_FILE_SIZE && write != 0){
        sprintf(fname, "%s%d", diag->fname, diag->file_counter);
        status = fileIO.open(fname, io_read_write);
        if ( status==fail ) ERROR(("Could not open file %s.", fname));
        fileIO.seek(0, SEEK_END);
    } else{ // Need to start a new file
        sprintf(fname, "%s%d", diag->fname, ++(diag->file_counter));
        status = fileIO.open(fname, io_write);
        if ( status==fail ) ERROR(("Could not open file %s.", fname));
        write = 0;
    }

    fileIO.write(diag->buff, store);
    fileIO.close();

    write += store;
    store = 0;

    diag->store_counter = store;
    diag->write_counter = write;
}

