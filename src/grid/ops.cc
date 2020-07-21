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

  int cell_planes_per_axis = g->CELL_PLANES_PER_AXIS;
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

  /* TODO: Get rid of the following.
  
  // We extend that to include:
  // remainder of x=0 plane
  // 6 => x, y-1, z-1
  // 7 => x, y-1, z+1
  // 8 => x, y+1, z-1
  // 9 => x, y+1, z+1

  // x-1 plane
  // 10 => x-1, y-1, z
  // 11 => x-1, y+1, z

  // 12 => x-1, y, z-1
  // 13 => x-1, y, z+1

  // diagonals
  // 14 => x-1, y-1, z-1
  // 15 => x-1, y-1, z+1
  // 16 => x-1, y+1, z-1
  // 17 => x-1, y+1, z+1

  // x+1 plane
  // 18 => x+1, y-1, z
  // 19 => x+1, y+1, z

  // 20 => x+1, y, z-1
  // 21 => x+1, y, z+1

  // diagonals
  // 22 => x+1, y-1, z-1
  // 23 => x+1, y-1, z+1
  // 24 => x+1, y+1, z-1
  // 25 => x+1, y+1, z+1

  // Self?
  // 26 => x, y, z
  
  */

  int neighbor_index = 0;
  for( z=0; z<=lnz+1; z++ ) {
    for( y=0; y<=lny+1; y++ ) {
      for( x=0; x<=lnx+1; x++ ) {
        i = num_neighbors * LOCAL_CELL_ID(x,y,z);

        int special_cell = 46008;
        // Fill the neighbor array with the appropriate
        // points
        for ( int x_index = -1; x_index < cell_planes_per_axis - 1; ++x_index )
        {
          for ( int y_index = -1; y_index < cell_planes_per_axis - 1; ++y_index )
          {
            for ( int z_index = -1; z_index < cell_planes_per_axis - 1; ++z_index )
            {
              // Determine the local neighbor.
              neighbor_index = ( x_index + 1 ) * cell_planes_per_axis * cell_planes_per_axis + ( y_index + 1 ) * cell_planes_per_axis + ( z_index + 1 );

              // Write the neighbor index properly. 
              g->neighbor[i + neighbor_index] = g->rangel + LOCAL_CELL_ID( x + x_index, y + y_index, z + z_index ); 
                
              if ( LOCAL_CELL_ID(x,y,z) == special_cell )
                  printf("\nCELL ID %d. Neighbor %d == %d", (int)LOCAL_CELL_ID(x,y,z), (int)neighbor_index, (int)g->neighbor[i + neighbor_index]);
            }
          }
        }
        if ( LOCAL_CELL_ID(x,y,z) == special_cell ) 
            printf("\n");

        /* Not doing this method...

        // TODO: we could replace all this indexing with a constexpr function
        // that knows how to turn each 3d index into a 1d index at compile time
        // for us. This would allow the user to then use a 3d index where the
        // want to, and still have no over / the same over head as the user
        // using 1d
        g->neighbor[i+0]  = g->rangel + LOCAL_CELL_ID(x-1, y,   z  );
        g->neighbor[i+1]  = g->rangel + LOCAL_CELL_ID(x,   y-1, z  );
        g->neighbor[i+2]  = g->rangel + LOCAL_CELL_ID(x,   y,   z-1);
        g->neighbor[i+3]  = g->rangel + LOCAL_CELL_ID(x+1, y,   z  );
        g->neighbor[i+4]  = g->rangel + LOCAL_CELL_ID(x,   y+1, z  );
        g->neighbor[i+5]  = g->rangel + LOCAL_CELL_ID(x,   y,   z+1);

        g->neighbor[i+6]  = g->rangel + LOCAL_CELL_ID(x,   y-1, z-1);
        g->neighbor[i+7]  = g->rangel + LOCAL_CELL_ID(x,   y-1, z+1);
        g->neighbor[i+8]  = g->rangel + LOCAL_CELL_ID(x,   y+1, z+1);
        g->neighbor[i+9]  = g->rangel + LOCAL_CELL_ID(x,   y+1, z+1);

        g->neighbor[i+10] = g->rangel + LOCAL_CELL_ID(x-1, y-1, z  );
        g->neighbor[i+11] = g->rangel + LOCAL_CELL_ID(x-1, y+1, z  );
        g->neighbor[i+12] = g->rangel + LOCAL_CELL_ID(x-1, y,   z-1);
        g->neighbor[i+13] = g->rangel + LOCAL_CELL_ID(x-1, y,   z+1);

        g->neighbor[i+14] = g->rangel + LOCAL_CELL_ID(x-1, y-1, z-1);
        g->neighbor[i+15] = g->rangel + LOCAL_CELL_ID(x-1, y-1, z+1);
        g->neighbor[i+16] = g->rangel + LOCAL_CELL_ID(x-1, y+1, z-1);
        g->neighbor[i+17] = g->rangel + LOCAL_CELL_ID(x-1, y+1, z+1);

        g->neighbor[i+18] = g->rangel + LOCAL_CELL_ID(x+1, y-1, z  );
        g->neighbor[i+19] = g->rangel + LOCAL_CELL_ID(x+1, y+1, z  );
        g->neighbor[i+20] = g->rangel + LOCAL_CELL_ID(x+1, y,   z-1);
        g->neighbor[i+21] = g->rangel + LOCAL_CELL_ID(x+1, y,   z+1);

        g->neighbor[i+22] = g->rangel + LOCAL_CELL_ID(x+1, y-1, z+1);
        g->neighbor[i+23] = g->rangel + LOCAL_CELL_ID(x+1, y-1, z+1);
        g->neighbor[i+24] = g->rangel + LOCAL_CELL_ID(x+1, y+1, z-1);
        g->neighbor[i+25] = g->rangel + LOCAL_CELL_ID(x+1, y+1, z+1);

        // TODO: we likely don't need this?
        g->neighbor[i+26] = g->rangel + LOCAL_CELL_ID(x  , y  , z  );

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
            g->neighbor[i + 18] = reflect_particles;
            g->neighbor[i + 19] = reflect_particles;
            g->neighbor[i + 20] = reflect_particles;
            g->neighbor[i + 21] = reflect_particles;
            g->neighbor[i + 22] = reflect_particles;
            g->neighbor[i + 23] = reflect_particles;
            g->neighbor[i + 24] = reflect_particles;
            g->neighbor[i + 25] = reflect_particles;
            g->neighbor[i + 26] = reflect_particles;
        }
        if( y==lny ) {
            // y+1 plane for the deepest cell
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

# define GLUE_FACE(tag,i,j,k,X,Y,Z) BEGIN_PRIMITIVE {           \
    if( boundary==BOUNDARY(i,j,k) ) {                           \
      if( rnc%((ln##Y+2)*(ln##Z+2))!=0 )                        \
        ERROR(("Remote face is incompatible"));                 \
      rn##X = (rnc/((ln##Y+2)*(ln##Z+2)))-2;                    \
      rn##Y = ln##Y;                                            \
      rn##Z = ln##Z;                                            \
      for( l##Z=1; l##Z<=ln##Z; l##Z++ ) {                      \
        for( l##Y=1; l##Y<=ln##Y; l##Y++ ) {                    \
          l##X = (i+j+k)<0 ? 1     : ln##X;                     \
          r##X = (i+j+k)<0 ? rn##X : 1;                         \
          r##Y = l##Y;                                          \
          r##Z = l##Z;                                          \
          g->neighbor[ num_neighbors*LOCAL_CELL_ID(lx,ly,lz) + tag ] =      \
            g->range[rank] + REMOTE_CELL_ID(rx,ry,rz);          \
        }                                                       \
      }                                                         \
      return;                                                   \
    }                                                           \
  } END_PRIMITIVE

  // TODO: Joe make this 27 cases
  int num_neighbors = g->NUM_NEIGHBORS;
  GLUE_FACE(0,-1, 0, 0,x,y,z);
  GLUE_FACE(1, 0,-1, 0,y,z,x);
  GLUE_FACE(2, 0, 0,-1,z,x,y);
  GLUE_FACE(3, 1, 0, 0,x,y,z);
  GLUE_FACE(4, 0, 1, 0,y,z,x);
  GLUE_FACE(5, 0, 0, 1,z,x,y);

# undef GLUE_FACE
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

// TODO: Fix this for 27 neighbors
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

# define SET_PBC(tag,i,j,k,X,Y,Z) BEGIN_PRIMITIVE {             \
    if( boundary==BOUNDARY(i,j,k) ) {                           \
      l##X = (i+j+k)<0 ? 1 : ln##X;                             \
      for( l##Z=1; l##Z<=ln##Z; l##Z++ )                        \
        for( l##Y=1; l##Y<=ln##Y; l##Y++ )                      \
          g->neighbor[ num_neighbors * LOCAL_CELL_ID(lx,ly,lz) + tag ] = pbc; \
      return;                                                   \
    }                                                           \
  } END_PRIMITIVE

  int num_neighbors = g->NUM_NEIGHBORS;
  SET_PBC(0,-1, 0, 0,x,y,z);
  SET_PBC(1, 0,-1, 0,y,z,x);
  SET_PBC(2, 0, 0,-1,z,x,y);
  SET_PBC(3, 1, 0, 0,x,y,z);
  SET_PBC(4, 0, 1, 0,y,z,x);
  SET_PBC(5, 0, 0, 1,z,x,y);

# undef SET_PBC
}

