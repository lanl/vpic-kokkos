/* 
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

#include "sf_interface.h"

/* Though the checkpt/restore functions are not part of the public
   API, they must not be declared as static. */

void
checkpt_hydro_array( const hydro_array_t * ha ) {
  CHECKPT( ha, 1 );
  CHECKPT_ALIGNED( ha->h, ha->g->nv, 128 );
  CHECKPT_PTR( ha->g );
}

hydro_array_t *
restore_hydro_array( void ) {
  hydro_array_t * ha;
  RESTORE( ha );
  RESTORE_ALIGNED( ha->h );
  RESTORE_PTR( ha->g );
  return ha;
}

hydro_array_t *
new_hydro_array( grid_t * g ) {
  hydro_array_t * ha;
  if( !g ) ERROR(( "NULL grid" ));
//  MALLOC( ha, 1 );
  ha = new hydro_array_t(g->nv);
  MALLOC_ALIGNED( ha->h, g->nv, 128 );
  ha->g = g;
  clear_hydro_array( ha );
  REGISTER_OBJECT( ha, checkpt_hydro_array, restore_hydro_array, NULL );
  return ha;
}

void
delete_hydro_array( hydro_array_t * ha ) {
  if( !ha ) return;
  UNREGISTER_OBJECT( ha );
  FREE_ALIGNED( ha->h );
  FREE( ha );
}

void
clear_hydro_array( hydro_array_t * ha ) {
  if( !ha ) ERROR(( "NULL hydro array" ));
  CLEAR( ha->h, ha->g->nv ); // FIXME: SPU THIS?
}

#define hydro(x,y,z) h0[ VOXEL(x,y,z, nx,ny,nz) ]

// Generic looping
#define XYZ_LOOP(xl,xh,yl,yh,zl,zh) \
  for( z=zl; z<=zh; z++ )	    \
    for( y=yl; y<=yh; y++ )	    \
      for( x=xl; x<=xh; x++ )
	      
// x_NODE_LOOP => Loop over all non-ghost nodes at plane x
#define x_NODE_LOOP(x) XYZ_LOOP(x,x,1,ny+1,1,nz+1)
#define y_NODE_LOOP(y) XYZ_LOOP(1,nx+1,y,y,1,nz+1)
#define z_NODE_LOOP(z) XYZ_LOOP(1,nx+1,1,ny+1,z,z)

void
synchronize_hydro_array( hydro_array_t * ha ) {
  int size, face, bc, x, y, z, nx, ny, nz;
  float *p, lw, rw;
  hydro_t * h0, * h;
  grid_t * g;

  if( !ha ) ERROR(( "NULL hydro array" ));

  h0 = ha->h;
  g  = ha->g;
  nx = g->nx;
  ny = g->ny;
  nz = g->nz;

  // Note: synchronize_hydro assumes that hydro has not been adjusted
  // at the local domain boundary. Because hydro fields are purely
  // diagnostic, correct the hydro along local boundaries to account
  // for accumulations over partial cell volumes

# define ADJUST_HYDRO(i,j,k,X,Y,Z)              \
  do {                                          \
    bc = g->bc[BOUNDARY(i,j,k)];                \
    if( bc<0 || bc>=world_size ) {              \
      face = (i+j+k)<0 ? 1 : n##X+1;            \
      X##_NODE_LOOP(face) {                     \
        h = &hydro(x,y,z);                      \
        h->jx  *= 2;                            \
        h->jy  *= 2;                            \
        h->jz  *= 2;                            \
        h->rho *= 2;                            \
        h->px  *= 2;                            \
        h->py  *= 2;                            \
        h->pz  *= 2;                            \
        h->ke  *= 2;                            \
        h->txx *= 2;                            \
        h->tyy *= 2;                            \
        h->tzz *= 2;                            \
        h->tyz *= 2;                            \
        h->tzx *= 2;                            \
        h->txy *= 2;                            \
      }                                         \
    }                                           \
  } while(0)
  
  ADJUST_HYDRO(-1, 0, 0,x,y,z);
  ADJUST_HYDRO( 0,-1, 0,y,z,x);
  ADJUST_HYDRO( 0, 0,-1,z,x,y);
  ADJUST_HYDRO( 1, 0, 0,x,y,z);
  ADJUST_HYDRO( 0, 1, 0,y,z,x);
  ADJUST_HYDRO( 0, 0, 1,z,x,y);

# undef ADJUST_HYDRO

# define BEGIN_RECV(i,j,k,X,Y,Z) \
  begin_recv_port(i,j,k,( 1 + 14*(n##Y+1)*(n##Z+1) )*sizeof(float),g)

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {      \
    size = ( 1 + 14*(n##Y+1)*(n##Z+1) )*sizeof(float);  \
    p = (float *)size_send_port( i, j, k, size, g );    \
    if( p ) {                                           \
      (*(p++)) = g->d##X;                               \
      face = (i+j+k)<0 ? 1 : n##X+1;                    \
      X##_NODE_LOOP(face) {                             \
        h = &hydro(x,y,z);                              \
        (*(p++)) = h->jx;                               \
        (*(p++)) = h->jy;                               \
        (*(p++)) = h->jz;                               \
        (*(p++)) = h->rho;                              \
        (*(p++)) = h->px;                               \
        (*(p++)) = h->py;                               \
        (*(p++)) = h->pz;                               \
        (*(p++)) = h->ke;                               \
        (*(p++)) = h->txx;                              \
        (*(p++)) = h->tyy;                              \
        (*(p++)) = h->tzz;                              \
        (*(p++)) = h->tyz;                              \
        (*(p++)) = h->tzx;                              \
        (*(p++)) = h->txy;                              \
      }                                                 \
      begin_send_port( i, j, k, size, g );              \
    }                                                   \
  } END_PRIMITIVE

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                \
    p = (float *)end_recv_port(i,j,k,g);                        \
    if( p ) {                                                   \
      rw = (*(p++));                 /* Remote g->d##X */       \
      lw = rw + g->d##X;                                        \
      rw /= lw;                                                 \
      lw = g->d##X/lw;                                          \
      lw += lw;                                                 \
      rw += rw;                                                 \
      face = (i+j+k)<0 ? n##X+1 : 1; /* Twice weighted sum */   \
      X##_NODE_LOOP(face) {                                     \
        h = &hydro(x,y,z);                                      \
        h->jx  = lw*h->jx  + rw*(*(p++));                       \
        h->jy  = lw*h->jy  + rw*(*(p++));                       \
        h->jz  = lw*h->jz  + rw*(*(p++));                       \
        h->rho = lw*h->rho + rw*(*(p++));                       \
        h->px  = lw*h->px  + rw*(*(p++));                       \
        h->py  = lw*h->py  + rw*(*(p++));                       \
        h->pz  = lw*h->pz  + rw*(*(p++));                       \
        h->ke  = lw*h->ke  + rw*(*(p++));                       \
        h->txx = lw*h->txx + rw*(*(p++));                       \
        h->tyy = lw*h->tyy + rw*(*(p++));                       \
        h->tzz = lw*h->tzz + rw*(*(p++));                       \
        h->tyz = lw*h->tyz + rw*(*(p++));                       \
        h->tzx = lw*h->tzx + rw*(*(p++));                       \
        h->txy = lw*h->txy + rw*(*(p++));                       \
      }                                                         \
    }                                                           \
  } END_PRIMITIVE

# define END_SEND(i,j,k,X,Y,Z) end_send_port( i, j, k, g )

  // Exchange x-faces
  BEGIN_SEND(-1, 0, 0,x,y,z);
  BEGIN_SEND( 1, 0, 0,x,y,z);
  BEGIN_RECV(-1, 0, 0,x,y,z);
  BEGIN_RECV( 1, 0, 0,x,y,z);
  END_RECV(-1, 0, 0,x,y,z);
  END_RECV( 1, 0, 0,x,y,z);
  END_SEND(-1, 0, 0,x,y,z);
  END_SEND( 1, 0, 0,x,y,z);

  // Exchange y-faces
  BEGIN_SEND( 0,-1, 0,y,z,x);
  BEGIN_SEND( 0, 1, 0,y,z,x);
  BEGIN_RECV( 0,-1, 0,y,z,x);
  BEGIN_RECV( 0, 1, 0,y,z,x);
  END_RECV( 0,-1, 0,y,z,x);
  END_RECV( 0, 1, 0,y,z,x);
  END_SEND( 0,-1, 0,y,z,x);
  END_SEND( 0, 1, 0,y,z,x);

  // Exchange z-faces
  BEGIN_SEND( 0, 0,-1,z,x,y);
  BEGIN_SEND( 0, 0, 1,z,x,y);
  BEGIN_RECV( 0, 0,-1,z,x,y);
  BEGIN_RECV( 0, 0, 1,z,x,y);
  END_RECV( 0, 0,-1,z,x,y);
  END_RECV( 0, 0, 1,z,x,y);
  END_SEND( 0, 0,-1,z,x,y);
  END_SEND( 0, 0, 1,z,x,y);

# undef BEGIN_RECV
# undef BEGIN_SEND
# undef END_RECV
# undef END_SEND
}

// In my tests it is faster to copy into the legacy hydro arrays and use the
// old synchronize than use this function, so this is unused unless a deck
// specifically call for it, and liable to not be kept up to date.
void
synchronize_hydro_array_kokkos( hydro_array_t * ha ) {
  int size, face, bc, nx, ny, nz;
  float *p, lw, rw;
  grid_t * g;

  if( !ha ) ERROR(( "NULL hydro array" ));

  g  = ha->g;
  nx = g->nx;
  ny = g->ny;
  nz = g->nz;

  auto k_h_d = ha->k_h_d;

#define x_node_policy(x)  Kokkos::MDRangePolicy<Kokkos::Rank<3>>({x,1,1},{x+1, ny+2,nz+2})
#define y_node_policy(y)  Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,y,1},{nx+2,y+1, nz+2})
#define z_node_policy(z)  Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,z},{nx+2,ny+2,z+1})

  // Note: synchronize_hydro assumes that hydro has not been adjusted
  // at the local domain boundary. Because hydro fields are purely
  // diagnostic, correct the hydro along local boundaries to account
  // for accumulations over partial cell volumes

# define ADJUST_HYDRO(i,j,k,X,Y,Z)                              \
  do {                                                          \
    bc = g->bc[BOUNDARY(i,j,k)];                                \
    if( bc<0 || bc>=world_size ) {                              \
      face = (i+j+k)<0 ? 1 : n##X+1;                            \
      Kokkos::parallel_for("Adjust hydro",                      \
      X##_node_policy(face),                                    \
      KOKKOS_LAMBDA(const int x, const int y, const int z) {    \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jx) *= 2;       \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jy) *= 2;       \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jz) *= 2;       \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::rho) *= 2;      \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::px) *= 2;       \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::py) *= 2;       \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::pz) *= 2;       \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::ke) *= 2;       \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::txx) *= 2;      \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tyy) *= 2;      \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tzz) *= 2;      \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tyz) *= 2;      \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tzx) *= 2;      \
        k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::txy) *= 2;      \
      });                                       \
    }                                           \
  } while(0)
  
  ADJUST_HYDRO(-1, 0, 0,x,y,z);
  ADJUST_HYDRO( 0,-1, 0,y,z,x);
  ADJUST_HYDRO( 0, 0,-1,z,x,y);
  ADJUST_HYDRO( 1, 0, 0,x,y,z);
  ADJUST_HYDRO( 0, 1, 0,y,z,x);
  ADJUST_HYDRO( 0, 0, 1,z,x,y);

# undef ADJUST_HYDRO
  
# define BEGIN_RECV(i,j,k,X,Y,Z) \
  begin_recv_port(i,j,k,( 1 + 14*(n##Y+1)*(n##Z+1) )*sizeof(float),g)

#define BEGIN_RECV_KOKKOS(i,j,k,X,Y,Z,recv_buff_h) \
  begin_recv_port_k(i,j,k,(1 + 14*(n##Y+1)*(n##Z+1))*sizeof(float),g,reinterpret_cast<char*>(recv_buff_h.data()));

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {      \
    size = ( 1 + 14*(n##Y+1)*(n##Z+1) )*sizeof(float);  \
    p = (float *)size_send_port( i, j, k, size, g );    \
    if( p ) {                                           \
      (*(p++)) = g->d##X;                               \
      face = (i+j+k)<0 ? 1 : n##X+1;                    \
      X##_NODE_LOOP(face) {                             \
        h = &hydro(x,y,z);                              \
        (*(p++)) = h->jx;                               \
        (*(p++)) = h->jy;                               \
        (*(p++)) = h->jz;                               \
        (*(p++)) = h->rho;                              \
        (*(p++)) = h->px;                               \
        (*(p++)) = h->py;                               \
        (*(p++)) = h->pz;                               \
        (*(p++)) = h->ke;                               \
        (*(p++)) = h->txx;                              \
        (*(p++)) = h->tyy;                              \
        (*(p++)) = h->tzz;                              \
        (*(p++)) = h->tyz;                              \
        (*(p++)) = h->tzx;                              \
        (*(p++)) = h->txy;                              \
      }                                                 \
      begin_send_port( i, j, k, size, g );              \
    }                                                   \
  } END_PRIMITIVE

# define BEGIN_SEND_KOKKOS(i,j,k,X,Y,Z,send_buff_d,send_buff_h) BEGIN_PRIMITIVE {      \
    size = ( 1 + 14*(n##Y+1)*(n##Z+1) )*sizeof(float);  \
    face = (i+j+k)<0 ? 1 : n##X+1;                      \
    Kokkos::parallel_for("load vals", X##_node_policy(face), \
    KOKKOS_LAMBDA(const int x, const int y, const int z) { \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 0) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jx); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 1) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jy); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 2) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jz); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 3) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::rho); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 4) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::px); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 5) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::py); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 6) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::pz); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 7) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::ke); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 8) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::txx); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 9) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tyy); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 10) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tzz); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 11) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tyz); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 12) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tzx); \
      send_buff_d(1+14*((Z-1)*(n##Y)+(Y-1)) + 13) = k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::txy); \
    }); \
    Kokkos::deep_copy(send_buff_h, send_buff_d); \
    send_buff_h(0) = size; \
    begin_send_port_k(i,j,k,size,g, reinterpret_cast<char*>(send_buff_h.data())); \
  } END_PRIMITIVE

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                \
    p = (float *)end_recv_port(i,j,k,g);                        \
    if( p ) {                                                   \
      rw = (*(p++));                 /* Remote g->d##X */       \
      lw = rw + g->d##X;                                        \
      rw /= lw;                                                 \
      lw = g->d##X/lw;                                          \
      lw += lw;                                                 \
      rw += rw;                                                 \
      face = (i+j+k)<0 ? n##X+1 : 1; /* Twice weighted sum */   \
      X##_NODE_LOOP(face) {                                     \
        h = &hydro(x,y,z);                                      \
        h->jx  = lw*h->jx  + rw*(*(p++));                       \
        h->jy  = lw*h->jy  + rw*(*(p++));                       \
        h->jz  = lw*h->jz  + rw*(*(p++));                       \
        h->rho = lw*h->rho + rw*(*(p++));                       \
        h->px  = lw*h->px  + rw*(*(p++));                       \
        h->py  = lw*h->py  + rw*(*(p++));                       \
        h->pz  = lw*h->pz  + rw*(*(p++));                       \
        h->ke  = lw*h->ke  + rw*(*(p++));                       \
        h->txx = lw*h->txx + rw*(*(p++));                       \
        h->tyy = lw*h->tyy + rw*(*(p++));                       \
        h->tzz = lw*h->tzz + rw*(*(p++));                       \
        h->tyz = lw*h->tyz + rw*(*(p++));                       \
        h->tzx = lw*h->tzx + rw*(*(p++));                       \
        h->txy = lw*h->txy + rw*(*(p++));                       \
      }                                                         \
    }                                                           \
  } END_PRIMITIVE

#define END_RECV_KOKKOS(i,j,k,X,Y,Z,recv_buff_d, recv_buff_h)                                           \
BEGIN_PRIMITIVE {                                                                                       \
  p = reinterpret_cast<float*> (end_recv_port_k(i,j,k,g));                                              \
  if(p) {                                                                                               \
    rw = recv_buff_h(0);                                                                                \
    lw = rw + g->d##X;                                                                                  \
    rw /= lw;                                                                                           \
    lw = g->d##X/lw;                                                                                    \
    lw += lw;                                                                                           \
    rw += rw;                                                                                           \
    face = (i+j+k)<0 ? n##X+1 : 1;                                                                      \
    Kokkos::deep_copy(recv_buff_d, recv_buff_h);                                                        \
    Kokkos::parallel_for("load vals", X##_node_policy(face),                                            \
    KOKKOS_LAMBDA(const int x, const int y, const int z) {                                              \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jx) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jx)      \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+0);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jy) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jy)      \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+1);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jz) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::jz)      \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+2);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::rho) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::rho)    \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+3);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::px) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::px)      \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+4);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::py) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::py)      \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+5);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::pz) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::pz)      \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+6);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::ke) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::ke)      \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+7);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::txx) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::txx)    \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+8);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tyy) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tyy)    \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+9);    \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tzz) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tzz)    \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+10);   \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tyz) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tyz)    \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+11);   \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tzx) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::tzx)    \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+12);   \
      k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::txy) = lw*k_h_d(VOXEL(x,y,z,nx,ny,nz), hydro_var::txy)    \
                                                  + rw*recv_buff_d(1+14*((Z-1)*(n##Y)+(Y-1))+13);   \
    });                                                                                                 \
  }                                                                                                     \
} END_PRIMITIVE

# define END_SEND(i,j,k,X,Y,Z) end_send_port( i, j, k, g )

#define END_SEND_KOKKOS(i,j,k,X,Y,Z) end_send_port_k(i,j,k,g);

  Kokkos::View<float*> sbuf_pos_d("Send buffer pos", 1+14*(ny+1)*(nz+1));
  Kokkos::View<float*> sbuf_neg_d("Send buffer neg", 1+14*(ny+1)*(nz+1));
  Kokkos::View<float*> rbuf_pos_d("Recv buffer pos", 1+14*(ny+1)*(nz+1));
  Kokkos::View<float*> rbuf_neg_d("Recv buffer neg", 1+14*(ny+1)*(nz+1));
  Kokkos::View<float*>::HostMirror sbuf_pos_h = Kokkos::create_mirror_view(sbuf_pos_d);
  Kokkos::View<float*>::HostMirror sbuf_neg_h = Kokkos::create_mirror_view(sbuf_neg_d);
  Kokkos::View<float*>::HostMirror rbuf_pos_h = Kokkos::create_mirror_view(rbuf_pos_d);
  Kokkos::View<float*>::HostMirror rbuf_neg_h = Kokkos::create_mirror_view(rbuf_neg_d);
  BEGIN_SEND_KOKKOS(-1, 0, 0,x,y,z,sbuf_neg_d, sbuf_neg_h);
  BEGIN_SEND_KOKKOS( 1, 0, 0,x,y,z,sbuf_pos_d, sbuf_pos_h);
  BEGIN_RECV_KOKKOS(-1, 0, 0,x,y,z,rbuf_neg_h);
  BEGIN_RECV_KOKKOS( 1, 0, 0,x,y,z,rbuf_pos_h);
  END_RECV_KOKKOS(-1, 0, 0,x,y,z,rbuf_neg_d, rbuf_neg_h);
  END_RECV_KOKKOS( 1, 0, 0,x,y,z,rbuf_pos_d, rbuf_pos_h);
  END_SEND_KOKKOS(-1, 0, 0,x,y,z);
  END_SEND_KOKKOS( 1, 0, 0,x,y,z);

  Kokkos::resize(sbuf_pos_d, 1+14*(nz+1)*(nx+1));
  Kokkos::resize(sbuf_neg_d, 1+14*(nz+1)*(nx+1));
  Kokkos::resize(rbuf_pos_d, 1+14*(nz+1)*(nx+1));
  Kokkos::resize(rbuf_neg_d, 1+14*(nz+1)*(nx+1));
  sbuf_pos_h = Kokkos::create_mirror_view(sbuf_pos_d);
  sbuf_neg_h = Kokkos::create_mirror_view(sbuf_neg_d);
  rbuf_pos_h = Kokkos::create_mirror_view(rbuf_pos_d);
  rbuf_neg_h = Kokkos::create_mirror_view(rbuf_neg_d);
  BEGIN_SEND_KOKKOS( 0,-1, 0,y,z,x,sbuf_neg_d, sbuf_neg_h);
  BEGIN_SEND_KOKKOS( 0, 1, 0,y,z,x,sbuf_pos_d, sbuf_pos_h);
  BEGIN_RECV_KOKKOS( 0,-1, 0,y,z,x,rbuf_neg_h);
  BEGIN_RECV_KOKKOS( 0, 1, 0,y,z,x,rbuf_pos_h);
  END_RECV_KOKKOS( 0,-1, 0,y,z,x,rbuf_neg_d, rbuf_neg_h);
  END_RECV_KOKKOS( 0, 1, 0,y,z,x,rbuf_pos_d, rbuf_pos_h);
  END_SEND_KOKKOS( 0,-1, 0,y,z,x);
  END_SEND_KOKKOS( 0, 1, 0,y,z,x);

  Kokkos::resize(sbuf_pos_d, 1+14*(nx+1)*(ny+1));
  Kokkos::resize(sbuf_neg_d, 1+14*(nx+1)*(ny+1));
  Kokkos::resize(rbuf_pos_d, 1+14*(nx+1)*(ny+1));
  Kokkos::resize(rbuf_neg_d, 1+14*(nx+1)*(ny+1));
  sbuf_pos_h = Kokkos::create_mirror_view(sbuf_pos_d);
  sbuf_neg_h = Kokkos::create_mirror_view(sbuf_neg_d);
  rbuf_pos_h = Kokkos::create_mirror_view(rbuf_pos_d);
  rbuf_neg_h = Kokkos::create_mirror_view(rbuf_neg_d);
  BEGIN_SEND_KOKKOS( 0, 0,-1,z,x,y,sbuf_neg_d, sbuf_neg_h);
  BEGIN_SEND_KOKKOS( 0, 0, 1,z,x,y,sbuf_pos_d, sbuf_pos_h);
  BEGIN_RECV_KOKKOS( 0, 0,-1,z,x,y,rbuf_neg_h);
  BEGIN_RECV_KOKKOS( 0, 0, 1,z,x,y,rbuf_pos_h);
  END_RECV_KOKKOS( 0, 0,-1,z,x,y,rbuf_neg_d, rbuf_neg_h);
  END_RECV_KOKKOS( 0, 0, 1,z,x,y,rbuf_pos_d, rbuf_pos_h);
  END_SEND_KOKKOS( 0, 0,-1,z,x,y);
  END_SEND_KOKKOS( 0, 0, 1,z,x,y);

//  // Exchange x-faces
//  BEGIN_SEND(-1, 0, 0,x,y,z);
//  BEGIN_SEND( 1, 0, 0,x,y,z);
//  BEGIN_RECV(-1, 0, 0,x,y,z);
//  BEGIN_RECV( 1, 0, 0,x,y,z);
//  END_RECV(-1, 0, 0,x,y,z);
//  END_RECV( 1, 0, 0,x,y,z);
//  END_SEND(-1, 0, 0,x,y,z);
//  END_SEND( 1, 0, 0,x,y,z);
//
//  // Exchange y-faces
//  BEGIN_SEND( 0,-1, 0,y,z,x);
//  BEGIN_SEND( 0, 1, 0,y,z,x);
//  BEGIN_RECV( 0,-1, 0,y,z,x);
//  BEGIN_RECV( 0, 1, 0,y,z,x);
//  END_RECV( 0,-1, 0,y,z,x);
//  END_RECV( 0, 1, 0,y,z,x);
//  END_SEND( 0,-1, 0,y,z,x);
//  END_SEND( 0, 1, 0,y,z,x);
//
//  // Exchange z-faces
//  BEGIN_SEND( 0, 0,-1,z,x,y);
//  BEGIN_SEND( 0, 0, 1,z,x,y);
//  BEGIN_RECV( 0, 0,-1,z,x,y);
//  BEGIN_RECV( 0, 0, 1,z,x,y);
//  END_RECV( 0, 0,-1,z,x,y);
//  END_RECV( 0, 0, 1,z,x,y);
//  END_SEND( 0, 0,-1,z,x,y);
//  END_SEND( 0, 0, 1,z,x,y);

# undef BEGIN_RECV
# undef BEGIN_SEND
# undef END_RECV
# undef END_SEND
# undef BEGIN_RECV_KOKKOS
# undef BEGIN_SENV_KOKKOS
# undef END_RECV_KOKKOS
# undef END_SEND_KOKKOS
}

void
hydro_array_t::copy_to_host() {
  Kokkos::deep_copy( k_h_h , k_h_d);

  // Avoid capturing this
  auto& k_h = k_h_h;
  hydro_t * h_l = h;

  //for(int i=0; i<hydro_array->k_h_h.extent(0); i++) {
  Kokkos::parallel_for("copy hydro to legacy array",
    host_execution_policy(0, k_h_h.extent(0) - 1) ,
    KOKKOS_LAMBDA (int i) {
    h_l[i].jx = k_h(i, hydro_var::jx);
    h_l[i].jy = k_h(i, hydro_var::jy);
    h_l[i].jz = k_h(i, hydro_var::jz);
    h_l[i].rho = k_h(i, hydro_var::rho);
    h_l[i].px = k_h(i, hydro_var::px);
    h_l[i].py = k_h(i, hydro_var::py);
    h_l[i].pz = k_h(i, hydro_var::pz);
    h_l[i].ke = k_h(i, hydro_var::ke);
    h_l[i].txx = k_h(i, hydro_var::txx);
    h_l[i].tyy = k_h(i, hydro_var::tyy);
    h_l[i].tzz = k_h(i, hydro_var::tzz);
    h_l[i].tyz = k_h(i, hydro_var::tyz);
    h_l[i].tzx = k_h(i, hydro_var::tzx);
    h_l[i].txy = k_h(i, hydro_var::txy);
  #ifdef FIELD_IONIZATION
    h_l[i].max_q = k_h(i, hydro_var::max_q);
    h_l[i].avg_q = k_h(i, hydro_var::avg_q);
  #endif
  });

}
