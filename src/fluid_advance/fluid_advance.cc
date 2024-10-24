#include "fluid_advance.h"
//#include "../boundary/boundary.h"

/* Private interface *********************************************************/


void
checkpt_fluid_species( const fluid_species_t * fsp ) {
  std::cout << "checkptg " << fsp->name << std::endl;
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
  std::cout << "Appending fluid species. fsp->next=" << fsp->next << "\n";
  std::cout << "NULL=" << NULL << "\n";
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
  
  //fsp = new fluid_species_t(g->nv, xyz_sz, yzx_sz, zxy_sz); // To-do: Use this for kokkos VPIC!
  MALLOC( fsp, 1 );
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

  std::cout << "Creating fluid species. fsp->next=" << fsp->next << "\n";
  /* id, next are set by append fluid species */

  REGISTER_OBJECT( fsp, checkpt_fluid_species, restore_fluid_species, NULL ); // To-do: Implement checkpointing.
  return fsp;
}

/* Class methods **************************************************************/
/*
void
species_t::copy_to_host()
{

  Kokkos::deep_copy(k_p_h, k_p_d);
  Kokkos::deep_copy(k_p_i_h, k_p_i_d);
  Kokkos::deep_copy(k_pm_h, k_pm_d);
  Kokkos::deep_copy(k_pm_i_h, k_pm_i_d);
  Kokkos::deep_copy(k_nm_h, k_nm_d);

  nm = k_nm_h(0);

  // Avoid capturing this
  auto& k_particle_h = k_p_h;
  auto& k_particle_i_h = k_p_i_h;
  auto& particles = p;

  Kokkos::parallel_for("copy particles to host",
    host_execution_policy(0, np) ,
    KOKKOS_LAMBDA (int i) {

      particles[i].dx = k_particle_h(i, particle_var::dx);
      particles[i].dy = k_particle_h(i, particle_var::dy);
      particles[i].dz = k_particle_h(i, particle_var::dz);
      particles[i].ux = k_particle_h(i, particle_var::ux);
      particles[i].uy = k_particle_h(i, particle_var::uy);
      particles[i].uz = k_particle_h(i, particle_var::uz);
      particles[i].w  = k_particle_h(i, particle_var::w);
      particles[i].i  = k_particle_i_h(i);

    });

  // Avoid capturing this
  auto& k_particle_movers_h = k_pm_h;
  auto& k_particle_i_movers_h = k_pm_i_h;
  auto& movers = pm;

  Kokkos::parallel_for("copy movers to host",
    host_execution_policy(0, max_nm) ,
    KOKKOS_LAMBDA (int i) {

      movers[i].dispx = k_particle_movers_h(i, particle_mover_var::dispx);
      movers[i].dispy = k_particle_movers_h(i, particle_mover_var::dispy);
      movers[i].dispz = k_particle_movers_h(i, particle_mover_var::dispz);
      movers[i].i     = k_particle_i_movers_h(i);

    });

  last_copied = g->step;

}
*/
/*
void
species_t::copy_to_device()
{

  k_nm_h(0) = nm;

  // Avoid capturing this
  auto& k_particle_h = k_p_h;
  auto& k_particle_i_h = k_p_i_h;
  auto& particles = p;

  Kokkos::parallel_for("copy particles to device",
    host_execution_policy(0, np) ,
    KOKKOS_LAMBDA (int i) {

      k_particle_h(i, particle_var::dx) = particles[i].dx;
      k_particle_h(i, particle_var::dy) = particles[i].dy;
      k_particle_h(i, particle_var::dz) = particles[i].dz;
      k_particle_h(i, particle_var::ux) = particles[i].ux;
      k_particle_h(i, particle_var::uy) = particles[i].uy;
      k_particle_h(i, particle_var::uz) = particles[i].uz;
      k_particle_h(i, particle_var::w)  = particles[i].w;
      k_particle_i_h(i) = particles[i].i;

    });

  // Avoid capturing this
  auto& k_particle_movers_h = k_pm_h;
  auto& k_particle_i_movers_h = k_pm_i_h;
  auto& movers = pm;

  Kokkos::parallel_for("copy movers to device",
    host_execution_policy(0, max_nm) ,
    KOKKOS_LAMBDA (int i) {

      k_particle_movers_h(i, particle_mover_var::dispx) = movers[i].dispx;
      k_particle_movers_h(i, particle_mover_var::dispy) = movers[i].dispy;
      k_particle_movers_h(i, particle_mover_var::dispz) = movers[i].dispz;
      k_particle_i_movers_h(i) = movers[i].i;

    });

  Kokkos::deep_copy(k_p_d, k_p_h);
  Kokkos::deep_copy(k_p_i_d, k_p_i_h);
  Kokkos::deep_copy(k_pm_d, k_pm_h);
  Kokkos::deep_copy(k_pm_i_d, k_pm_i_h);
  Kokkos::deep_copy(k_nm_d, k_nm_h);

}
*/
