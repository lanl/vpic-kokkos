#include "fluid_advance.h"
//#include "../boundary/boundary.h"

/* Private interface *********************************************************/


void
checkpt_fluid_species( const fluid_species_t * fsp ) {
  //  std::cout << "checkptg " << fsp->name << std::endl;
  CHECKPT( fsp, 1 );
  CHECKPT_STR( fsp->name );
  checkpt_data( fsp->fl,
                fsp->g->nv * sizeof(fluid_t),
                fsp->g->nv * sizeof(fluid_t), 1, 1, 128 );
  CHECKPT_PTR( fsp->g );
  CHECKPT_PTR( fsp->next );
}

fluid_species_t *
restore_fluid_species( void ) {
  fluid_species_t * fsp;
  RESTORE( fsp );
  RESTORE_STR( fsp->name );
  fsp->fl = (fluid_t *) restore_data();
  RESTORE_PTR( fsp->g );
  RESTORE_PTR( fsp->next );
  return fsp;
}


void
delete_fluid_species( fluid_species_t * fsp ) {
  UNREGISTER_OBJECT( fsp );
  FREE_ALIGNED( fsp->fl );
  FREE( fsp->name );
  FREE( fsp );
}

/* Public interface **********************************************************/

int
num_fluid_species( const fluid_species_t * fsp_list ) {
  return fsp_list ? fsp_list->id+1 : 0;
}

void
delete_fluid_species_list( fluid_species_t * fsp_list ) {
  fluid_species_t * fsp;
  while( fsp_list ) {
    fsp = fsp_list;
    fsp_list = fsp_list->next;
    delete_fluid_species( fsp );
  }
}

fluid_species_t *
find_fluid_species_id( fluid_species_id id,
		       fluid_species_t * fsp_list ) {
  fluid_species_t * fsp;
  LIST_FIND_FIRST( fsp, fsp_list, fsp->id==id );
  return fsp;
}

fluid_species_t *
find_fluid_species_name( const char * name,
			 fluid_species_t * fsp_list ) {
  fluid_species_t * fsp;
  if( !name ) return NULL;
  LIST_FIND_FIRST( fsp, fsp_list, strcmp( fsp->name, name )==0 );
  return fsp;
}

fluid_species_t *
append_fluid_species( fluid_species_t * fsp,
		      fluid_species_t ** fsp_list ) {
  if( !fsp || !fsp_list ) ERROR(( "Bad args" ));
  //  std::cout << "# Appending fluid species. fsp->next=" << fsp->next << "\n";
  //std::cout << "NULL=" << NULL << "\n";
  if( fsp->next ) WARNING(( "Fluid species \"%s\" already in a list", fsp->name ));
  if( find_fluid_species_name( fsp->name, *fsp_list ) )
    ERROR(( "There is already a fluid species in the list named \"%s\"", fsp->name ));
  if( (*fsp_list) && fsp->g!=(*fsp_list)->g )
    ERROR(( "Fluid species \"%s\" uses a different grid from this list", fsp->name ));
  fsp->id   = num_fluid_species( *fsp_list );
  fsp->next = *fsp_list;
  *fsp_list = fsp;
  return fsp;
}


fluid_species_t *
fluid_species( const char * name,
	       float q,
	       float m,
	       grid_t * g ) {

  fluid_species_t * fsp;
  int len = name ? strlen(name) : 0;

  if( !len ) ERROR(( "Cannot create a nameless fluid species" ));
  if( !g ) ERROR(( "NULL grid" ));
  if( g->nv == 0) ERROR(( "Allocate grid before defining fluid species." ));

  //int nx = g->nx;
  //int ny = g->ny;
  //int nz = g->nz;

  //int xyz_sz = 2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz;
  //int yzx_sz = 2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx;
  //int zxy_sz = 2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny;
  
  fsp = new fluid_species_t(g->nv);//, xyz_sz, yzx_sz, zxy_sz); // To-do: Add back in for halo exchange buffers
  //  MALLOC( fsp, 1 );
  //CLEAR( fsp, 1 );

  MALLOC( fsp->name, len+1 );
  strcpy( fsp->name, name );

  fsp->q = q;
  fsp->m = m;

  if(!world_rank) fprintf(stderr, "Mallocing %.4f GiB for fluid species %s.\n",
          (double (g->nv*sizeof(fluid_t)))/pow(2,30), fsp->name);

  MALLOC_ALIGNED( fsp->fl, g->nv, 128 );
  CLEAR( fsp->fl, g->nv );

  fsp->g = g;

  //  std::cout << "# Creating fluid species. fsp->next=" << fsp->next << "\n";
  /* id, next are set by append fluid species */

  REGISTER_OBJECT( fsp, checkpt_fluid_species, restore_fluid_species, NULL ); // To-do: Implement checkpointing.
  return fsp;
}

/* Class methods **************************************************************/

void
fluid_species_t::copy_to_host()
{

  Kokkos::deep_copy(k_fl_h, k_fl_d);

  // Avoid capturing this
  auto& k_fluid_h = k_fl_h;
  //  auto& host_fluid = fl;
  fluid_t * host_fluid = fl;
  
  Kokkos::parallel_for("copy fluid to host",
    host_execution_policy(0, g->nv - 1) ,
    KOKKOS_LAMBDA (int i) {

      host_fluid[i].den = k_fluid_h(i, fluid_var::den);
      host_fluid[i].tmp = k_fluid_h(i, fluid_var::tmp);
      host_fluid[i].prs = k_fluid_h(i, fluid_var::prs);
      host_fluid[i].ux = k_fluid_h(i, fluid_var::ux);
      host_fluid[i].uy = k_fluid_h(i, fluid_var::uy);
      host_fluid[i].uz = k_fluid_h(i, fluid_var::uz);

    });

  last_copied = g->step;

}

void
fluid_species_t::copy_to_device()
{

  // Avoid capturing this
  auto& k_fluid_h = k_fl_h;
  //  auto& fluid = fl;
  fluid_t * host_fluid = fl;

  Kokkos::parallel_for("copy fluid to device",
    host_execution_policy(0, g->nv - 1) ,
    KOKKOS_LAMBDA (int i) {

      k_fluid_h(i, fluid_var::den) = host_fluid[i].den;
      k_fluid_h(i, fluid_var::tmp) = host_fluid[i].tmp;
      k_fluid_h(i, fluid_var::prs) = host_fluid[i].prs;
      k_fluid_h(i, fluid_var::ux) = host_fluid[i].ux;
      k_fluid_h(i, fluid_var::uy) = host_fluid[i].uy;
      k_fluid_h(i, fluid_var::uz) = host_fluid[i].uz;

    });

  Kokkos::deep_copy(k_fl_d, k_fl_h);

}
