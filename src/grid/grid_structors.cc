/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

#include "grid.h"

/* Though these functions are not part of grid's public API, they must
   not be declared as static */

void
checkpt_grid( grid_t * g ) {
  CHECKPT( g, 1 );
  if( g->range    ) CHECKPT_ALIGNED( g->range, world_size+1, 16 );

  CHECKPT_PTR( g->mp );
  CHECKPT_PTR( g->mp_k );

  g->copy_to_host();

  CHECKPT_VIEW( g->k_mesh_d );
  CHECKPT_VIEW( g->k_mesh_h );
  CHECKPT_VIEW_DATA( g->k_mesh_h );

  CHECKPT_VIEW( g->k_neighbor_d );
  CHECKPT_VIEW( g->k_neighbor_h );
  CHECKPT_VIEW_DATA( g->k_neighbor_h );

}

grid_t *
restore_grid( void ) {
  grid_t * g;
  RESTORE( g );
  if( g->range    ) RESTORE_ALIGNED( g->range );

  RESTORE_PTR( g->mp );
  RESTORE_PTR( g->mp_k );

  RESTORE_VIEW( &g->k_mesh_d );
  RESTORE_VIEW( &g->k_mesh_h );
  RESTORE_VIEW_DATA( g->k_mesh_h );

  RESTORE_VIEW( &g->k_neighbor_d );
  RESTORE_VIEW( &g->k_neighbor_h );
  RESTORE_VIEW_DATA( g->k_neighbor_h );

  g->copy_to_device();

  return g;
}

grid_t *
new_grid( void ) {
  int i;

  grid_t* g = new grid_t();

  for( i=0; i<27; i++ ) g->bc[i] = anti_symmetric_fields;
  g->bc[BOUNDARY(0,0,0)] = world_rank;
  g->mp = new_mp( 27 );
  g->mp_k = new_mp( 27 );
  g->geometry = Geometry::Cartesian;
  REGISTER_OBJECT( g, checkpt_grid, restore_grid, NULL );
  return g;
}

void
delete_grid( grid_t * g ) {
  if( !g ) return;
  UNREGISTER_OBJECT( g );
  FREE_ALIGNED( g->range );
  delete_mp( g->mp );
  delete_mp( g->mp_k );
  delete(g);
}

void
grid_t::copy_to_host()
{
  Kokkos::deep_copy(k_neighbor_h, k_neighbor_d);
  Kokkos::deep_copy(k_mesh_h, k_mesh_d);
}

void
grid_t::copy_to_device()
{
  Kokkos::deep_copy(k_neighbor_d, k_neighbor_h);
  Kokkos::deep_copy(k_mesh_d, k_mesh_h);
}
