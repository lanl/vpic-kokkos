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
checkpt_grid( const grid_t * g ) {
  CHECKPT( g, 1 );
  if( g->range    ) CHECKPT_ALIGNED( g->range, world_size+1, 16 );
  if( g->neighbor ) CHECKPT_ALIGNED( g->neighbor, 6*g->nv, 128 );
  CHECKPT_PTR( g->mp );
  CHECKPT_PTR( g->mp_k );
}

grid_t *
restore_grid( void ) {
  grid_t * g;
  RESTORE( g );
  if( g->range    ) RESTORE_ALIGNED( g->range );
  if( g->neighbor ) RESTORE_ALIGNED( g->neighbor );
  RESTORE_PTR( g->mp );
  RESTORE_PTR( g->mp_k );
  return g;
}

grid_t *
new_grid( void ) {
  int i;

  grid_t* g = new grid_t();
  //MALLOC( g, 1 );
  //CLEAR( g, 1 );

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
  FREE_ALIGNED( g->neighbor );
  FREE_ALIGNED( g->range );
  delete_mp( g->mp );
  delete_mp( g->mp_k );
//    delete_mp_kokkos(g->mp_k);
  FREE( g );
}

void
grid_t::init_kokkos_grid(int num_neighbor)
{
    k_neighbor_d = k_neighbor_t("k_neighbor_d", num_neighbor);
    k_neighbor_h = Kokkos::create_mirror_view(k_neighbor_d);

    k_mesh_d = k_mesh_t("k_mesh_d", nv);
    k_mesh_h = Kokkos::create_mirror_view(k_mesh_d);

    // TODO: make this a host parlalel for
    for (int i = 0; i < num_neighbor; i++)
    {
        k_neighbor_h(i) = neighbor[i];
    }

    // Construct the dense mesh
    int v = 0;
    for (int k = 0 ; k < nz+2 ; ++k) {
      double fz = (k-0.5) / ((double) nz);
      double z = z0*(1-fz) + z1*fz;

      for (int j = 0 ; j < ny+2 ; ++j) {
        double fy = (j-0.5) / ((double) ny);
        double y = y0*(1-fy) + y1*fy;

        for (int i = 0 ; i < nx+2 ; ++i) {
          double fx = (i-0.5) / ((double) nx);
          double x = x0*(1-fx) + x1*fx;

          k_mesh_h(v, mesh_var::x) = x;
          k_mesh_h(v, mesh_var::y) = y;
          k_mesh_h(v, mesh_var::z) = z;
          v += 1;

        }
      }
    }

    // Copy to device
    Kokkos::deep_copy(k_mesh_d, k_mesh_h);
    Kokkos::deep_copy(k_neighbor_d, k_neighbor_h);

}
