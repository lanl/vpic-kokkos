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
#define RANK_TO_INDEX(rank,ix,iy,iz,nx,ny,nz) do {        \
    int _ix, _iy, _iz;                                    \
    _ix  = (rank);   /* ix = ix + gpx*( iy + gpy*iz ) */  \
    _iy  = _ix/(nx); /* iy = iy + gpy*iz */               \
    _ix -= _iy*(nx); /* ix = ix */                        \
    _iz  = _iy/(ny); /* iz = iz */                        \
    _iy -= _iz*(ny); /* iy = iy */                        \
    (ix) = _ix;                                           \
    (iy) = _iy;                                           \
    (iz) = _iz;                                           \
  } while(0)

/* Though the checkpt/restore functions are not part of the public
   API, they must not be declared as static. */

void
checkpt_hydro_array( const hydro_array_t * ha ) {
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  CHECKPT( ha, 1 );
  CHECKPT_ALIGNED( ha->h, ha->g->nv, 128 );
  CHECKPT_PTR( ha->g );
#else
  CHECKPT_VIEW( ha->k_h_h );
  CHECKPT_PTR( ha->g );
#endif
}

hydro_array_t *
restore_hydro_array( void ) {
  hydro_array_t * ha;
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  RESTORE( ha );
  RESTORE_ALIGNED( ha->h );
  RESTORE_PTR( ha->g );
#else
  ha = new hydro_array_t(1);
  RESTORE_VIEW( ha->k_h_h );
  RESTORE_PTR( ha->g );
#endif
  return ha;
}

hydro_array_t *
new_hydro_array( grid_t * g ) {
  hydro_array_t * ha;
  if( !g ) ERROR(( "NULL grid" ));
//  MALLOC( ha, 1 );
  ha = new hydro_array_t(g->nv);
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  MALLOC_ALIGNED( ha->h, g->nv, 128 );
#endif
  ha->g = g;
  clear_hydro_array( ha );
  Kokkos::deep_copy(ha->k_h_h, 0);
  REGISTER_OBJECT( ha, checkpt_hydro_array, restore_hydro_array, NULL );
  return ha;
}

void
delete_hydro_array( hydro_array_t * ha ) {
  if( !ha ) return;
  UNREGISTER_OBJECT( ha );
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  FREE_ALIGNED( ha->h );
#endif
  delete ha;
}

void
clear_hydro_array( hydro_array_t * ha ) {
  if( !ha ) ERROR(( "NULL hydro array" ));
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  CLEAR( ha->h, ha->g->nv ); // FIXME: SPU THIS?
#endif
}

#define hydro(x,y,z) h0[ VOXEL(x,y,z, nx,ny,nz) ]

// Generic looping
#define XYZ_LOOP(xl,xh,yl,yh,zl,zh) \
  for( z=zl; z<=zh; z++ )     \
    for( y=yl; y<=yh; y++ )     \
      for( x=xl; x<=xh; x++ )
       
// x_NODE_LOOP => Loop over all non-ghost nodes at plane x
#define x_NODE_LOOP(x) XYZ_LOOP(x,x,1,ny+1,1,nz+1)
#define y_NODE_LOOP(y) XYZ_LOOP(1,nx+1,y,y,1,nz+1)
#define z_NODE_LOOP(z) XYZ_LOOP(1,nx+1,1,ny+1,z,z)

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
void
synchronize_hydro_array( hydro_array_t * ha ) {
  int size, face, bc, x, y, z, nx, ny, nz;
  k_hydro_t::non_const_value_type *p, lw, rw;
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
        h->rho_m  *= 2;                         \
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
  begin_recv_port(i,j,k,( 1 + HYDRO_SYNC_COUNT*(n##Y+1)*(n##Z+1) ) \
                        *sizeof(k_hydro_t::non_const_value_type),g)

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {      \
    size = ( 1 + HYDRO_SYNC_COUNT*(n##Y+1)*(n##Z+1) )   \
         *sizeof(k_hydro_t::non_const_value_type);      \
    p = (k_hydro_t::non_const_value_type *)size_send_port( i, j, k, size, g );    \
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
        (*(p++)) = h->rho_m;                            \
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
    p = (k_hydro_t::non_const_value_type *)end_recv_port(i,j,k,g);                        \
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
        h->jx    = lw*h->jx  + rw*(*(p++));                     \
        h->jy    = lw*h->jy  + rw*(*(p++));                     \
        h->jz    = lw*h->jz  + rw*(*(p++));                     \
        h->rho   = lw*h->rho + rw*(*(p++));                     \
        h->px    = lw*h->px  + rw*(*(p++));                     \
        h->py    = lw*h->py  + rw*(*(p++));                     \
        h->pz    = lw*h->pz  + rw*(*(p++));                     \
        h->rho_m = lw*h->rho_m  + rw*(*(p++));                  \
        h->txx   = lw*h->txx + rw*(*(p++));                     \
        h->tyy   = lw*h->tyy + rw*(*(p++));                     \
        h->tzz   = lw*h->tzz + rw*(*(p++));                     \
        h->tyz   = lw*h->tyz + rw*(*(p++));                     \
        h->tzx   = lw*h->tzx + rw*(*(p++));                     \
        h->txy   = lw*h->txy + rw*(*(p++));                     \
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
#endif

void
synchronize_hydro_array_kokkos( hydro_array_t * ha ) {
  int size, face, bc, x, y, z, nx, ny, nz;
  k_hydro_t::non_const_value_type *p, lw, rw;
  grid_t * g;

  if( !ha ) ERROR(( "NULL hydro array" ));

  g  = ha->g;
  nx = g->nx;
  ny = g->ny;
  nz = g->nz;

  // Note: synchronize_hydro assumes that hydro has not been adjusted
  // at the local domain boundary. Because hydro fields are purely
  // diagnostic, correct the hydro along local boundaries to account
  // for accumulations over partial cell volumes

# define ADJUST_HYDRO(i,j,k,X,Y,Z)                    \
  do {                                                \
    bc = g->bc[BOUNDARY(i,j,k)];                      \
    if( bc<0 || bc>=world_size ) {                    \
      face = (i+j+k)<0 ? 1 : n##X+1;                  \
      X##_NODE_LOOP(face) {                           \
        for (int var=0; var<HYDRO_SYNC_COUNT; var++) { \
          ha->k_h_h(VOXEL(x,y,z,nx,ny,nz), var) *= 2; \
        }                                             \
      }                                               \
    }                                                 \
  } while(0)
  
  ADJUST_HYDRO(-1, 0, 0,x,y,z);
  ADJUST_HYDRO( 0,-1, 0,y,z,x);
  ADJUST_HYDRO( 0, 0,-1,z,x,y);
  ADJUST_HYDRO( 1, 0, 0,x,y,z);
  ADJUST_HYDRO( 0, 1, 0,y,z,x);
  ADJUST_HYDRO( 0, 0, 1,z,x,y);

# undef ADJUST_HYDRO

# define BEGIN_RECV(i,j,k,X,Y,Z) \
  begin_recv_port(i,j,k,( 1 + HYDRO_SYNC_COUNT*(n##Y+1)*(n##Z+1) ) \
                        *sizeof(k_hydro_t::non_const_value_type),g)

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {             \
    size = ( 1 + HYDRO_SYNC_COUNT*(n##Y+1)*(n##Z+1) )          \
         * sizeof(k_hydro_t::non_const_value_type);            \
    p = (k_hydro_t::non_const_value_type *)size_send_port( i, j, k, size, g );           \
    if( p ) {                                                  \
      (*(p++)) = g->d##X;                                      \
      face = (i+j+k)<0 ? 1 : n##X+1;                           \
      X##_NODE_LOOP(face) {                                    \
        for (int var=0; var<HYDRO_SYNC_COUNT; var++) {         \
          (*(p++)) = ha->k_h_h(VOXEL(x,y,z,nx,ny,nz), var);    \
        }                                                      \
      }                                                        \
      begin_send_port( i, j, k, size, g );                     \
    }                                                          \
  } END_PRIMITIVE

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                        \
    p = (k_hydro_t::non_const_value_type *)end_recv_port(i,j,k,g);      \
    if( p ) {                                                           \
      rw = (*(p++));                 /* Remote g->d##X */               \
      lw = rw + g->d##X;                                                \
      rw /= lw;                                                         \
      lw = g->d##X/lw;                                                  \
      lw += lw;                                                         \
      rw += rw;                                                         \
      face = (i+j+k)<0 ? n##X+1 : 1; /* Twice weighted sum */           \
      X##_NODE_LOOP(face) {                                             \
        const int cell = VOXEL(x,y,z,nx,ny,nz);                         \
        for (int var=0; var<HYDRO_SYNC_COUNT; var++) {                  \
          ha->k_h_h(cell, var) = lw*ha->k_h_h(cell, var) + rw*(*(p++)); \
        }                                                               \
      }                                                                 \
    }                                                                   \
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

void
hydro_array_t::copy_to_host(FILE *fp, const int step /*=0*/) {
  Kokkos::deep_copy( k_h_h , k_h_d);

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  // Avoid capturing this
  auto& k_h = k_h_h;
  hydro_t * h_l = h;
  auto wr = world_rank;
  
  Kokkos::parallel_for("copy hydro to legacy array",
    host_execution_policy(0, k_h_h.extent(0)) ,
    KOKKOS_LAMBDA (int i) {
    h_l[i].jx    = k_h(i, hydro_var::jx);
    h_l[i].jy    = k_h(i, hydro_var::jy);
    h_l[i].jz    = k_h(i, hydro_var::jz);
    h_l[i].rho   = k_h(i, hydro_var::rho);
    h_l[i].px    = k_h(i, hydro_var::px);
    h_l[i].py    = k_h(i, hydro_var::py);
    h_l[i].pz    = k_h(i, hydro_var::pz);
    h_l[i].rho_m = k_h(i, hydro_var::rho_m);
    h_l[i].txx   = k_h(i, hydro_var::txx);
    h_l[i].tyy   = k_h(i, hydro_var::tyy);
    h_l[i].tzz   = k_h(i, hydro_var::tzz);
    h_l[i].tyz   = k_h(i, hydro_var::tyz);
    h_l[i].tzx   = k_h(i, hydro_var::tzx);
    h_l[i].txy   = k_h(i, hydro_var::txy);
#ifdef VARIABLE_CHARGE
    h_l[i].qmin = k_h(i, hydro_var::qmin);
    h_l[i].qmax = k_h(i, hydro_var::qmax);
    h_l[i].n_q0 = k_h(i, hydro_var::n_q0);
    h_l[i].n_q1 = k_h(i, hydro_var::n_q1);
    h_l[i].n_q2 = k_h(i, hydro_var::n_q2);
    h_l[i].n_q3 = k_h(i, hydro_var::n_q3);
    h_l[i].n_q4 = k_h(i, hydro_var::n_q4);
    h_l[i].n_q5 = k_h(i, hydro_var::n_q5);
#endif
    
    int ix, iy, iz;
    RANK_TO_INDEX(i, ix, iy, iz, 1, 1, 1);
    
    if(fp && wr==0 && (h_l[i].txx*h_l[i].txx + h_l[i].tyy*h_l[i].tyy + h_l[i].tzz*h_l[i].tzz) > 0) { 
      //get temperature
      auto vx = h_l[i].px;
      auto vy = h_l[i].py;
      auto vz = h_l[i].pz;
      auto Tx = h_l[i].txx;
      auto Ty = h_l[i].tyy;
      auto Tz = h_l[i].tzz;
      
      Tx = ( Tx - vx * vx );
      Ty = ( Ty - vy * vy );
      Tz = ( Tz - vz * vz );
      
      auto T = (Tx+Ty+Tz)/3.0;
      //fprintf(fp,"%d %.15e %.15e %.15e %.15e %.15e %.15e %d",step, h_l[i].txx,h_l[i].tyy,h_l[i].tzz,h_l[i].px,h_l[i].py,h_l[i].pz,i);
      fprintf(fp,"%d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %d",step, 
              0.5*(h_l[i].txx*h_l[i].txx + h_l[i].tyy*h_l[i].tyy + h_l[i].tzz*h_l[i].tzz), 
              h_l[i].txx, h_l[i].tyy, h_l[i].tzz, 
              h_l[i].px,h_l[i].py,h_l[i].pz,T,i);
    } 
  });
  // printf("k_h_h.extent(0)=%d\n",k_h_h.extent(0));
  if(fp && wr==0) fprintf(fp,"\n");
#endif
}
