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

#define LOCAL_CELL_ID(x,y,z)  VOXEL(x,y,z, lnx,lny,lnz)
#define REMOTE_CELL_ID(x,y,z) VOXEL(x,y,z, rnx,rny,rnz)

// Everybody must size their local grid in parallel

void
size_grid( grid_t * g,
           int lnx, int lny, int lnz ) {
  int64_t x,y,z;
  int i, j, k;
  int64_t ii, jj, kk;

  if( !g || lnx<1 || lny<1 || lnz<1 ) ERROR(( "Bad args" ));

  // Setup phase 2 data structures
  g->sx =  1;
  g->sy = (lnx+2)*g->sx;
  g->sz = (lny+2)*g->sy;
  g->nv = (lnz+2)*g->sz;
  g->nx = lnx; g->ny = lny; g->nz = lnz;

  for( k=-1; k<=1; k++ )
    for( j=-1; j<=1; j++ )
      for( i=-1; i<=1; i++ )
        g->bc[ BOUNDARY(i,j,k) ] = pec_fields;
  g->bc[ BOUNDARY(0,0,0) ] = world_rank;

  // Setup phase 3 data structures.  This is an ugly kludge to
  // interface phase 2 and phase 3 data structures
  FREE_ALIGNED( g->range );
  MALLOC_ALIGNED( g->range, world_size+1, 16 );
  ii = g->nv; // nv is not 64-bits
  mp_allgather_i64( &ii, g->range, 1 );
  jj = 0;
  g->range[world_size] = 0;
  for( i=0; i<=world_size; i++ ) {
    kk = g->range[i];
    g->range[i] = jj;
    jj += kk;
  }
  g->rangel = g->range[world_rank];
  g->rangeh = g->range[world_rank+1]-1;

  FREE_ALIGNED( g->neighbor );

  int planes_per_axis = g->PLANES_PER_AXIS;
  int num_neighbors = g->NUM_NEIGHBORS;
  MALLOC_ALIGNED( g->neighbor, num_neighbors * g->nv, 128 );
  
    

  // Originally the neighbor mapping is
  // 0 => x-1, y,   z
  // 1 => x,   y-1, z
  // 2 => x,   y,   z-1

  // 3 => x+1, y,   z
  // 4 => x,   y+1, z
  // 5 => x,   y,   z+1
  
  // We change the local neighborhood scheme
  // to include not only the faces, as before, 
  // but the edges and corners of each cell,
  // as well. The 3d -> 1d mapping we employ 
  // is as follows, for x_index, y_index, 
  // z_index each being one of {-1, 0, 1}:
  //
  // neighbor =   ( x_index + 1 ) * 3 * 3
  //            + ( y_index + 1 ) * 3
  //            + ( z_index + 1 ).
  //
  // Note that this is an identity mapping
  // for neighbor = 13. That is, 
  //
  // neighbor(0, 0, 0) = 13,
  //
  // such that we include the original cell 
  // in its own local neighborhood.

  /*

    A schematic of the neighbors is shown below. Each axis (X, Y, Z)
    has 3 planes with values {-1, 0, 1}. The intersection of each of 
    these planes gives rise to 26 neighbors surrounding 1 point at 
    the origin of each voxel (27 points total). The langle/rangle 
    numbers (< 1 >) show which plane corresponds to each axis. The
    other numbers denote the neighbor indices. There are 6 faces, 
    12 edges, and 8 corners to each voxel, and each of these neighbors
    represents a possible location for a particle to be sent through
    in the zigzag algorithm. 

                              Z
                              ^
                              |
                              |            Y
                              |           /
                              |          /
            < 1 >  8 -- -- -- -- 17 -- -- -- -- 26
                  /|          |  /|    /        /|
                 / |          | / |   /        / |
                /  |          |/  |  /        /  |
        < 0 >  5 --|-- -- -- 14 --|-- -- -- 23   |
              /|   |         /|   |/        /|   |
             / |   7 -- -- -/ -- 16 -- -- -/ |- 25
            /  |  /|       /  |  /|       /  |  /|
  < 1/-1 > 2 --|-- -- -- 11 --|-- -- -- 20   | / |
           |   |/  |      |   |/  |      |   |/  |
           |   4 --|-- -- |- 13 --|-- -- |- 22 --|-- -- -- -- --> X
           |  /|   |      |  /|   |      |  /|   |
           | / |   6 -- --|-- -- 15 -- --|-/ |- 24
           |/  |  /       |/  |  /       |/  |  /
    < 0 >  1 --|-- -- -- 10 --|-- -- -- 19   | /
           |   |/         |   |/         |   |/
           |   5 -- -- -- |- 12 -- -- -- |- 21 
           |  /           |  /           |  /
           | /            | /            | /
           |/             |/             |/
   < -1 >  0 -- -- -- --  9 -- -- -- -- 18
        < -1 >          < 0 >          < 1 >

   */  

  int neighbor_index = 0;
  for( z=0; z<=lnz+1; z++ ) {
    for( y=0; y<=lny+1; y++ ) {
      for( x=0; x<=lnx+1; x++ ) {
        i = num_neighbors * LOCAL_CELL_ID(x,y,z);

        int special_cell = 1205;
        // Fill the neighbor array with the appropriate
        // points
        for ( int x_index = -1; x_index < planes_per_axis - 1; ++x_index )
        {
          for ( int y_index = -1; y_index < planes_per_axis - 1; ++y_index )
          {
            for ( int z_index = -1; z_index < planes_per_axis - 1; ++z_index )
            {
              // Determine the local neighbor.
              neighbor_index = get_neighbor_index(x_index, y_index, z_index, planes_per_axis);

              // Write the neighbor index properly. 
              g->neighbor[i + neighbor_index] = g->rangel + LOCAL_CELL_ID( x + x_index, y + y_index, z + z_index ); 
                
              //if ( LOCAL_CELL_ID(x,y,z) == special_cell )
                  //printf("\nCELL ID %d. Neighbor %d == %lld", (int)LOCAL_CELL_ID(x,y,z), (int)neighbor_index, g->neighbor[i + neighbor_index]);
            }
          }
        }
        /*
        if ( LOCAL_CELL_ID(x,y,z) == special_cell ) 
            //printf("\n");
        if ( LOCAL_CELL_ID(x,y,z) == special_cell )
        {
            printf("\n\n*****************************************");
            printf("\nCELL ID %ld corresponds to (%ld, %ld, %ld)", LOCAL_CELL_ID(x,y,z), x,y,z);
            printf("\nNeighbor 23 corresponds to %ld", g->neighbor[num_neighbors * LOCAL_CELL_ID(x,y,z) + 23]);
            printf("\n\n*****************************************");
        }
        */
        
        // Set boundary faces appropriately
        // Here are the English conventions for 
        // the position along each axis:
        //
        // ^ z         z: height ( low to high )
        // |   / y     y: depth  ( shallow to deep )
        // |  /        x: left or right 
        // | /
        // |/
        // ---------> x
        if( x == 1   ) {
            // x-1 plane for the left most cell
            g->neighbor[i +  0] = reflect_particles;
            g->neighbor[i +  1] = reflect_particles;
            g->neighbor[i +  2] = reflect_particles;
            g->neighbor[i +  3] = reflect_particles;
            g->neighbor[i +  4] = reflect_particles;
            g->neighbor[i +  5] = reflect_particles;
            g->neighbor[i +  6] = reflect_particles;
            g->neighbor[i +  7] = reflect_particles;
            g->neighbor[i +  8] = reflect_particles;
        }
        if( y == 1  ) {
            // y-1 plane for the shallowest cell
            g->neighbor[i +  0] = reflect_particles;
            g->neighbor[i +  1] = reflect_particles;
            g->neighbor[i +  2] = reflect_particles;
            g->neighbor[i +  9] = reflect_particles;
            g->neighbor[i + 10] = reflect_particles;
            g->neighbor[i + 11] = reflect_particles;
            g->neighbor[i + 18] = reflect_particles;
            g->neighbor[i + 19] = reflect_particles;
            g->neighbor[i + 20] = reflect_particles;
        }
        if( z==1   ) {
            // z-1 plane for the lowest cell
            g->neighbor[i +  0] = reflect_particles;
            g->neighbor[i +  3] = reflect_particles;
            g->neighbor[i +  6] = reflect_particles;
            g->neighbor[i +  9] = reflect_particles;
            g->neighbor[i + 12] = reflect_particles;
            g->neighbor[i + 15] = reflect_particles;
            g->neighbor[i + 18] = reflect_particles;
            g->neighbor[i + 21] = reflect_particles;
            g->neighbor[i + 24] = reflect_particles;
        }
        if( x==lnx ) {
            // x+1 plane for the right most cell
            //printf("\nCELL ID %ld at (%ld, %ld, %ld) has lnx == %ld", LOCAL_CELL_ID(x,y,z), x, y, z, x);
            //printf("\nBefore:\n\tg->neighbor[i + 22] = %ld", g->neighbor[i + 22]);
            //printf("\n\tg->neighbor[i + 23] = %ld", g->neighbor[i + 23]);
            //printf("\n\tg->neighbor[i + 25] = %ld", g->neighbor[i + 22]);
            
            g->neighbor[i + 18] = reflect_particles;
            g->neighbor[i + 19] = reflect_particles;
            g->neighbor[i + 20] = reflect_particles;
            g->neighbor[i + 21] = reflect_particles;
            g->neighbor[i + 22] = reflect_particles;
            g->neighbor[i + 23] = reflect_particles;
            g->neighbor[i + 24] = reflect_particles;
            g->neighbor[i + 25] = reflect_particles;
            g->neighbor[i + 26] = reflect_particles;
           
            //printf("\nAfter:\n\tg->neighbor[i + 22] = %ld", g->neighbor[i + 22]);
            //printf("\n\tg->neighbor[i + 23] = %ld", g->neighbor[i + 23]);
            //printf("\n\tg->neighbor[i + 25] = %ld", g->neighbor[i + 22]);
            
        }
        if( y==lny ) {
            // y+1 plane for the deepest cell
            //printf("\nCELL ID %ld at (%ld, %ld, %ld) has lny == %ld", LOCAL_CELL_ID(x,y,z), x, y, z, y);
            g->neighbor[i +  6] = reflect_particles;
            g->neighbor[i +  7] = reflect_particles;
            g->neighbor[i +  8] = reflect_particles;
            g->neighbor[i + 15] = reflect_particles;
            g->neighbor[i + 16] = reflect_particles;
            g->neighbor[i + 17] = reflect_particles;
            g->neighbor[i + 24] = reflect_particles;
            g->neighbor[i + 25] = reflect_particles;
            g->neighbor[i + 26] = reflect_particles;
        }
        if( z==lnz ) {
            // y+1 plane for the highest cell
            g->neighbor[i +  2] = reflect_particles;
            g->neighbor[i +  5] = reflect_particles;
            g->neighbor[i +  8] = reflect_particles;
            g->neighbor[i + 11] = reflect_particles;
            g->neighbor[i + 14] = reflect_particles;
            g->neighbor[i + 17] = reflect_particles;
            g->neighbor[i + 20] = reflect_particles;
            g->neighbor[i + 23] = reflect_particles;
            g->neighbor[i + 26] = reflect_particles;
        }

        // Set ghost cells appropriately
        if( x==0 || x==lnx+1 ||
            y==0 || y==lny+1 ||
            z==0 || z==lnz+1 )
        {
            for (int z = 0; z < num_neighbors; z++)
            {
              g->neighbor[i+z] = reflect_particles;
            }
        }
      }
    }
  }


# if 0
  // Setup the space filling curve
  // FIXME: THIS IS A CRUDE HACK UNTIL A GOOD SFC CAN BE WRITTEN
  // CURRENT SFC IS GOOD FOR THREADING WITH POWER-OF-TWO NUMBER OF THREADS.
  // UP TO AND INCLUDING 8 THREADS.

  FREE_ALIGNED( g->sfc );
  MALLOC_ALIGNED( g->sfc, g->nv, 128 );

  do {
    int off;
    int ox, oy, oz;
    int nx, ny, nz;
    int nx1 = (lnx+2)/2,   ny1 = (lny+2)/2,   nz1 = (lnz+2)/2;
    int nx0 = (lnx+2)-nx1, ny0 = (lny+2)-ny1, nz0 = (lnz+2)-nz1;

    for( z=0; z<=lnz+1; z++ )
      for( y=0; y<=lny+1; y++ )
        for( x=0; x<=lnx+1; x++ ) {
          i = LOCAL_CELL_ID(x,y,z);
          off = 0;
          ox=x; nx=nx0; if(ox>=nx) ox-=nx, off+=nx,                 nx=nx1;
          oy=y; ny=ny0; if(oy>=ny) oy-=ny, off+=ny*(lnx+2),         ny=ny1;
          oz=z; nz=nz0; if(oz>=nz) oz-=nz, off+=nz*(lnx+2)*(lny+2), nz=nz1;
          g->sfc[i] = off + ox + nx*( oy + ny*oz );
        }
  } while(0);
# endif
}

// This is literally just for readability
enum class periodic_axis 
{
    x = 0, y = 1, z = 2, num_axes = 3
};

// TODO: This idea may get a lil wonky at the global corners and global edges,
// but I think it should be okay.
void assign_periodic_neighbors( grid_t * g, int rank, periodic_axis axis )
{

  int lnx = g->nx;    // Resolution in x direction
  int lny = g->ny;    // Resolution in y direction
  int lnz = g->nz;    // Resolution in z direction

  int rnc = g->range[rank+1] - g->range[rank];
    
  // Define the remote cell x, y, and z
  // coordinates that the neighbor to 
  // the LOCAL_CELL would be in a periodic 
  // grid. To save Bob some headaches later
  // on, "rnx" stands for "remote neighbor
  // along the x axis".
  int rnx = 0;
  int rny = 0;
  int rnz = 0;

  const int num_neighbors = g->NUM_NEIGHBORS; // Number of cells in the local neighborhood
  const int planes = g->PLANES_PER_AXIS;      // Number of planes {-1, 0, 1} for neighbor indexing
  int neighbor_index = 0;                     // The value of the neighbor index to be computed.
  int plane_value = 0;                        // The value of the plane {-1, 0, 1} for each case axis.

  switch (axis)
  {
    case periodic_axis::x:
    {
      // Move along the x == 1 and x == lnx global
      // planes. Populate all neighbors on these
      // faces.
      for ( int lx = 1; lx < lnx; lx = lx + (lnx - 1) )
      {
        for ( int ly = 1; ly <= lny; ++ly )
        {
          for ( int lz = 1; lz <= lnz; ++lz )
          {
            // Move only along the x == 1 and x == lnx global planes.
            // This fixes rnx to be then either lnx or 1, respectively.
            rnx = (lx == 1 ? lnx : 1);
            plane_value = (lx == 1 ? -1 : 1);
            for ( int yind = -1; yind < planes-1; ++yind  )
            {
              for ( int zind = -1; zind < planes-1; ++zind )
              {
                rny = ly + yind;
                while ( rny > lny ){ rny -= lny; }  // lny + 1 is a ghost cell
                while ( rny < 1 )  { rny += lny; }  // 0 is a ghost cell
                
                rnz = lz + zind;
                while ( rnz > lnz ){ rnz -= lnz; }  // lnz + 1 is a ghost cell
                while ( rnz < 1 )  { rnz += lnz; }  // 0 is a ghost cell
                
                neighbor_index = get_neighbor_index(plane_value, yind, zind, planes);
                g->neighbor[ num_neighbors * LOCAL_CELL_ID(lx, ly, lz) + neighbor_index ] = g->range[rank] + REMOTE_CELL_ID(rnx, rny, rnz);
                
              }
            }
          }
        }
      }
      break;
    }
    
    case periodic_axis::y:
    {
      // Move along the y == 1 and y == lny global
      // planes. Populate all neighbors on these
      // faces.
      for ( int ly = 1; ly < lny; ly = ly + (lny - 1) )
      {
        for ( int lz = 1; lz <= lnz; ++lz )
        {
          for ( int lx = 1; lx <= lnx; ++lx )
          {
            // Move only along the y == 1 and y == lny global planes.
            // This fixes rny to be then either lny or 1, respectively.
            rny = (ly == 1 ? lny : 1);
            plane_value = (ly == 1 ? -1 : 1);
            for ( int zind = -1; zind < planes-1; ++zind )
            {
              for ( int xind = -1; xind < planes-1; ++xind  )
              {
                rnz = lz + zind;
                while ( rnz > lnz ){ rnz -= lnz; }  // lnz + 1 is a ghost cell
                while ( rnz < 1 )  { rnz += lnz; }  // 0 is a ghost cell
                
                rnx = lx + xind;
                while ( rnx > lnx ){ rnx -= lnx; }  // lnx + 1 is a ghost cell
                while ( rnx < 1 )  { rnx += lnx; }  // 0 is a ghost cell
                
                neighbor_index = get_neighbor_index(xind, plane_value, zind, planes);
                g->neighbor[ num_neighbors * LOCAL_CELL_ID(lx, ly, lz) + neighbor_index ] = g->range[rank] + REMOTE_CELL_ID(rnx, rny, rnz);
                
              }
            }
          }
        }
      }
      break;
    }

    case periodic_axis::z:
    {
      // Move along the z == 1 and z == lnz global
      // planes. Populate all neighbors on these
      // faces.
      for ( int lz = 1; lz < lnz; lz = lz + (lnz - 1) )
      {
        for ( int lx = 1; lx <= lnx; ++lx )
        {
          for ( int ly = 1; ly <= lny; ++ly )
          {
            // Move only along the z == 1 and z == lny global planes.
            // This fixes rnz to be then either lny or 1, respectively.
            rnz = (lz == 1 ? lnz : 1);
            plane_value = (lz == 1 ? -1 : 1);
            for ( int xind = -1; xind < planes-1; ++xind  )
            {
              for ( int yind = -1; yind < planes-1; ++yind )
              {
                rnx = lx + xind;
                while ( rnx > lnx ){ rnx -= lnx; }  // lnx + 1 is a ghost cell
                while ( rnx < 1 )  { rnx += lnx; }  // 0 is a ghost cell
                
                rny = ly + yind;
                while ( rny > lny ){ rny -= lny; }  // lny + 1 is a ghost cell
                while ( rny < 1 )  { rny += lny; }  // 0 is a ghost cell
                
                neighbor_index = get_neighbor_index(xind, yind, plane_value, planes);
                g->neighbor[ num_neighbors * LOCAL_CELL_ID(lx, ly, lz) + neighbor_index ] = g->range[rank] + REMOTE_CELL_ID(rnx, rny, rnz);
                
              }
            }
          }
        }
      }
      break;
    }
  }
}

void
join_grid( grid_t * g,
           int boundary,
           int rank ) {
  int lx, ly, lz, lnx, lny, lnz, rx, ry, rz, rnx, rny, rnz, rnc;

  if( !g || boundary<0 || boundary>=27 || boundary==BOUNDARY(0,0,0) ||
      rank<0 || rank>=world_size ) ERROR(( "Bad args" ));

  // Join phase 2 data structures
  g->bc[boundary] = rank;

  // Join phase 3 data structures
  lnx = g->nx;
  lny = g->ny;
  lnz = g->nz;
  rnc = g->range[rank+1] - g->range[rank]; // Note: rnc <~ 2^31 / 6
  // TODO: is the 6 in this comment still true if we change the number of
  // neighbors to be more?

  int planes_per_axis = g->PLANES_PER_AXIS;
  int num_neighbors = g->NUM_NEIGHBORS;
  int neighbor_index = 0;

# define GLUE_FACE(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                                       \
    neighbor_index = get_neighbor_index(i, j, k, planes_per_axis);                      \
    if( boundary==BOUNDARY(i,j,k) ) {                                                   \
      if( rnc%((ln##Y+2)*(ln##Z+2))!=0 )                                                \
        ERROR(("Remote face is incompatible"));                                         \
      rn##X = (rnc/((ln##Y+2)*(ln##Z+2)))-2;                                            \
      rn##Y = ln##Y;                                                                    \
      rn##Z = ln##Z;                                                                    \
      for( l##Z=1; l##Z<=ln##Z; l##Z++ ) {                                              \
        for( l##Y=1; l##Y<=ln##Y; l##Y++ ) {                                            \
          l##X = (i+j+k)<0 ? 1     : ln##X;                                             \
          r##X = (i+j+k)<0 ? rn##X : 1;                                                 \
          r##Y = l##Y;                                                                  \
          r##Z = l##Z;                                                                  \
          g->neighbor[ num_neighbors*LOCAL_CELL_ID(lx,ly,lz) + neighbor_index ] =       \
            g->range[rank] + REMOTE_CELL_ID(rx,ry,rz);                                  \
        }                                                                               \
      }                                                                                 \
      return;                                                                           \
    }                                                                                   \
  } END_PRIMITIVE

  // TODO: Joe, are there more faces to be glued
  // with the increase in the local neighborhood?
  GLUE_FACE(-1, 0, 0,x,y,z);
  GLUE_FACE(0,-1, 0,y,z,x);
  GLUE_FACE(0, 0,-1,z,x,y);
  GLUE_FACE(1, 0, 0,x,y,z);
  GLUE_FACE(0, 1, 0,y,z,x);
  GLUE_FACE(0, 0, 1,z,x,y);

# undef GLUE_FACE
 
  // Assign the periodic boundaries appropriately. 
  assign_periodic_neighbors(g, rank, periodic_axis::x);
  assign_periodic_neighbors(g, rank, periodic_axis::y);
  assign_periodic_neighbors(g, rank, periodic_axis::z);

}

void
set_fbc( grid_t * g,
         int boundary,
         int fbc ) {

  if( !g || boundary<0 || boundary>=27 || boundary==BOUNDARY(0,0,0) ||
      ( fbc!=anti_symmetric_fields && fbc!=symmetric_fields &&
        fbc!=pmc_fields            && fbc!=absorb_fields    ) )
    ERROR(( "Bad args" ));

  g->bc[boundary] = fbc;
}

void
set_pbc( grid_t * g,
         int boundary,
         int pbc ) {
  int lx, ly, lz, lnx, lny, lnz;

  if( !g || boundary<0 || boundary>=27 || boundary==BOUNDARY(0,0,0) || pbc>=0 )
    ERROR(( "Bad args" ));

  lnx = g->nx;
  lny = g->ny;
  lnz = g->nz;

  int planes_per_axis = g->PLANES_PER_AXIS;
  int num_neighbors = g->NUM_NEIGHBORS;
  int neighbor_index = 0;

# define SET_PBC(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                                             \
    neighbor_index = get_neighbor_index(i, j, k, planes_per_axis);                          \
    if( boundary==BOUNDARY(i,j,k) ) {                                                       \
      l##X = (i+j+k)<0 ? 1 : ln##X;                                                         \
      for( l##Z=1; l##Z<=ln##Z; l##Z++ )                                                    \
        for( l##Y=1; l##Y<=ln##Y; l##Y++ )                                                  \
          g->neighbor[ num_neighbors * LOCAL_CELL_ID(lx,ly,lz) + neighbor_index ] = pbc;    \
      return;                                                                               \
    }                                                                                       \
  } END_PRIMITIVE

  // TODO: Joe, are there more faces to be glued
  // with the increase in the local neighborhood?
  SET_PBC(-1, 0, 0,x,y,z);
  SET_PBC(0,-1, 0,y,z,x);
  SET_PBC(0, 0,-1,z,x,y);
  SET_PBC(1, 0, 0,x,y,z);
  SET_PBC(0, 1, 0,y,z,x);
  SET_PBC(0, 0, 1,z,x,y);
 

# undef SET_PBC
}

