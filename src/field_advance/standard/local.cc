/* 
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

/******************************************************************************
 * local.c sets local boundary conditions. Functions are divided into two
 * categories:
 *   local_ghosts_xxx where xxx = tang_b, norm_e, div_b
 *   - Sets ghosts values of the fields just interior to a local boundary
 *     condition
 *   local_adjust_xxx where xxx = norm_b, tang_e, rhof, rhob, div_e_err
 *   - Directly enforces local boundary conditions on fields
 *****************************************************************************/
#define IN_sfa
#include <assert.h>
#include <functional>
#include <string>
#include "sfa_private.h"

#define FIELD(voxel, var) k_field(voxel, field_var::var)

#define XYZ_POLICY(xl,xh,yl,yh,zl,zh) Kokkos::MDRangePolicy<Kokkos::Rank<3>>({xl,yl,zl},{xh+1,yh+1,zh+1})

#define yz_EDGE_POLICY(x) XYZ_POLICY(x,x,1,ny,1,nz+1)
#define zx_EDGE_POLICY(y) XYZ_POLICY(1,nx+1,y,y,1,nz)
#define xy_EDGE_POLICY(z) XYZ_POLICY(1,nx,1,ny+1,z,z)

#define zy_EDGE_POLICY(x) XYZ_POLICY(x,x,1,ny+1,1,nz)
#define xz_EDGE_POLICY(y) XYZ_POLICY(1,nx,y,y,1,nz+1)
#define yx_EDGE_POLICY(z) XYZ_POLICY(1,nx+1,1,ny,z,z)

#define x_NODE_POLICY(x) XYZ_POLICY(x,x,1,ny+1,1,nz+1)
#define y_NODE_POLICY(y) XYZ_POLICY(1,nx+1,y,y,1,nz+1)
#define z_NODE_POLICY(z) XYZ_POLICY(1,nx+1,1,ny+1,z,z)

#define x_FACE_POLICY(x) XYZ_POLICY(x,x,1,ny,1,nz)
#define y_FACE_POLICY(y) XYZ_POLICY(1,nx,y,y,1,nz)
#define z_FACE_POLICY(z) XYZ_POLICY(1,nx,1,ny,z,z)

/*****************************************************************************
 * Local ghosts
 *****************************************************************************/

template<int i, int j, int k> 
void 
apply_local_tang_b(const int nx, const int ny, const int nz, 
                   const float cdt_dx, const float cdt_dy, const float cdt_dz,
                   const float higend, field_array_t* RESTRICT f, const grid_t* g) {
# define APPLY_LOCAL_TANG_B(i,j,k,X,Y,Z)                                      \
  do {                                                                        \
    const int bc = g->bc[BOUNDARY(i,j,k)];                                    \
    if( bc<0 || bc>=world_size ) {                                            \
      k_field_t k_field = f->k_f_d;                                           \
      const int ghost = (i+j+k)<0 ? 0 : n##X+1;                               \
      const int face  = (i+j+k)<0 ? 1 : n##X+1;                               \
      switch(bc) {                                                            \
      case anti_symmetric_fields:                                             \
        Kokkos::parallel_for("apply_local_tang_b<" #X #Y #Z "> " #Z #Y        \
          "anti_symmetric_fields edge", Z##Y##_EDGE_POLICY(ghost),            \
          KOKKOS_LAMBDA(const int x, const int y, const int z) {              \
          const size_t g_voxel = VOXEL(x,y,z,nx,ny,nz);                       \
          const size_t f_voxel = VOXEL(x-i,y-j,z-k,nx,ny,nz);                 \
          FIELD(g_voxel, cb##Y) = FIELD(f_voxel, cb##Y);                      \
        });                                                                   \
        Kokkos::parallel_for("apply_local_tang_b<" #X #Y #Z "> " #Y #Z        \
          "anti_symmetric_fields edge",  Y##Z##_EDGE_POLICY(ghost),           \
          KOKKOS_LAMBDA(const int x, const int y, const int z) {              \
          const size_t g_voxel = VOXEL(x,y,z,nx,ny,nz);                       \
          const size_t f_voxel = VOXEL(x-i,y-j,z-k,nx,ny,nz);                 \
          FIELD(g_voxel, cb##Z) = FIELD(f_voxel, cb##Z);                      \
        });                                                                   \
        break;                                                                \
      case symmetric_fields: case pmc_fields:                                 \
        Kokkos::parallel_for("apply_local_tang_b<" #X #Y #Z "> " #Z #Y        \
          "symmetric|pmc fields edge", Z##Y##_EDGE_POLICY(ghost),             \
          KOKKOS_LAMBDA(const int x, const int y, const int z) {              \
          const size_t g_voxel = VOXEL(x,y,z,nx,ny,nz);                       \
          const size_t f_voxel = VOXEL(x-i,y-j,z-k,nx,ny,nz);                 \
          FIELD(g_voxel, cb##Y) = -FIELD(f_voxel, cb##Y);                     \
        });                                                                   \
        Kokkos::parallel_for("apply_local_tang_b<" #X #Y #Z "> " #Y #Z        \
          "symmetric|pmc fields edge", Y##Z##_EDGE_POLICY(ghost),             \
          KOKKOS_LAMBDA(const int x, const int y, const int z) {              \
          const size_t g_voxel = VOXEL(x,y,z,nx,ny,nz);                       \
          const size_t f_voxel = VOXEL(x-i,y-j,z-k,nx,ny,nz);                 \
          FIELD(g_voxel, cb##Z) = -FIELD(f_voxel, cb##Z);                     \
        });                                                                   \
        break;                                                                \
      case absorb_fields:                                                     \
        drive = cdt_d##X*higend;                                              \
        decay = (1-drive)/(1+drive);                                          \
        drive = 2*drive/(1+drive);                                            \
        Kokkos::parallel_for("apply_local_tang_b<" #X #Y #Z "> " #Z #Y "edge", \
          Z##Y##_EDGE_POLICY(ghost), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
          const size_t g0_voxel = VOXEL(x,y,z,nx,ny,nz);                      \
          const size_t f0_voxel = VOXEL(x-i,y-j,z-k,nx,ny,nz);                \
          const int face##X = face, face##Y = Y, face##Z = Z;                 \
          const size_t g1_voxel = VOXEL(facex,facey,facez,nx,ny,nz);          \
          const size_t f1_voxel = VOXEL(facex-i,facey-j,facez-k,nx,ny,nz);    \
          float t1 = cdt_d##X*( FIELD(f1_voxel, e##Z) - FIELD(g1_voxel, e##Z) ); \
          t1 = (i+j+k)<0 ? t1 : -t1;                                          \
          const int ghost##X = ghost, ghost##Y = Y, ghost##Z = Z+1;           \
          const size_t f2_voxel = VOXEL(ghostx-i,ghosty-j,ghostz-k,nx,ny,nz); \
          float t2 = FIELD(f2_voxel, e##X);                                   \
          t2 = cdt_d##Z*( t2 - FIELD(f0_voxel, e##X) );                       \
          FIELD(g0_voxel, cb##Y) = decay*FIELD(g0_voxel, cb##Y)               \
                                 + drive*FIELD(f0_voxel, cb##Y) - t1 + t2;    \
        });                                                                   \
        Kokkos::parallel_for("apply_local_tang_b<" #X #Y #Z "> " #Y #Z "edge", \
          Y##Z##_EDGE_POLICY(ghost), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
          const size_t g0_voxel = VOXEL(x,y,z,nx,ny,nz);                      \
          const size_t f0_voxel = VOXEL(x-i,y-j,z-k,nx,ny,nz);                \
          const int face##X = face, face##Y = Y, face##Z = Z;                 \
          const size_t g1_voxel = VOXEL(facex,facey,facez,nx,ny,nz);          \
          const size_t f1_voxel = VOXEL(facex-i,facey-j,facez-k,nx,ny,nz);    \
          float t1 = cdt_d##X*( FIELD(f1_voxel, e##Y) - FIELD(g1_voxel, e##Y) ); \
          t1 = (i+j+k)<0 ? t1 : -t1;                                          \
          const int ghost##X = ghost, ghost##Y = Y+1, ghost##Z = Z;           \
          const size_t f2_voxel = VOXEL(ghostx-i,ghosty-j,ghostz-k,nx,ny,nz); \
          float t2 = FIELD(f2_voxel, e##X);                                   \
          t2 = cdt_d##Y*( t2 - FIELD(f0_voxel, e##X) );                       \
          FIELD(g0_voxel, cb##Z) = decay*FIELD(g0_voxel, cb##Z)               \
                                 + drive*FIELD(f0_voxel, cb##Z) - t1 + t2;    \
        });                                                                   \
        break;                                                                \
      default:                                                                \
        ERROR(("Bad boundary condition encountered."));                       \
        break;                                                                \
      }                                                                       \
    }                                                                         \
  } while(0)

  float drive, decay;

  if constexpr( i!=0 && j==0 && k==0 ) {
    APPLY_LOCAL_TANG_B(i,j,k,x,y,z);
  } else if constexpr( i==0 && j!=0 && k==0 ) {
    APPLY_LOCAL_TANG_B(i,j,k,y,z,x);
  } else if constexpr( i==0 && j==0 && k!=0 ) {
    APPLY_LOCAL_TANG_B(i,j,k,z,x,y);
  }
#undef APPLY_LOCAL_TANG_B
}

void
local_ghost_tang_b( field_array_t      * RESTRICT f,
                    const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  const float cdt_dx = g->cvac*g->dt*g->rdx;
  const float cdt_dy = g->cvac*g->dt*g->rdy;
  const float cdt_dz = g->cvac*g->dt*g->rdz;

  // Absorbing boundary condition is 2nd order accurate implementation
  // of a 1st order Higend ABC with 15 degree annihilation cone except
  // for 1d simulations where the 2nd order accurate implementation of
  // a 1st order Mur boundary condition is used.
  const float higend = ( nx>1 || ny>1 || nz>1 ) ? 1.03527618 : 1.;
  apply_local_tang_b<-1, 0, 0>(nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
  apply_local_tang_b< 0,-1, 0>(nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
  apply_local_tang_b< 0, 0,-1>(nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
  apply_local_tang_b< 1, 0, 0>(nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
  apply_local_tang_b< 0, 1, 0>(nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
  apply_local_tang_b< 0, 0, 1>(nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
}

// Note: local_adjust_div_e zeros the error on the boundaries for
// absorbing boundary conditions.  Thus, ghost norm e value is
// irrevelant.

template<int i, int j, int k> 
void 
apply_local_norm_e(field_array_t* RESTRICT f, const grid_t* g) {
# define APPLY_LOCAL_NORM_E(i,j,k,X,Y,Z)                                    \
  do {                                                                      \
    const int bc = g->bc[BOUNDARY(i,j,k)];                                  \
    if( bc<0 || bc>=world_size ) {                                          \
      const int nx = g->nx, ny = g->ny, nz = g->nz;                         \
      const int face = (i+j+k)<0 ? 0 : n##X+1;                              \
      k_field_t k_field = f->k_f_d;                                         \
      switch(bc) {                                                          \
      case anti_symmetric_fields:                                           \
        Kokkos::parallel_for("apply_local_norm_e<" #X #Y #Z "> " #X "node", \
          X##_NODE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            const size_t v0 = VOXEL(x,y,z,nx,ny,nz);                        \
            const size_t v1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);                  \
            FIELD(v0, e##X) = FIELD(v1, e##X);                              \
            FIELD(v0, tca##X) = FIELD(v1, tca##X);                          \
          });                                                               \
        break;                                                              \
      case symmetric_fields: case pmc_fields:                               \
        Kokkos::parallel_for("apply_local_norm_e<" #X #Y #Z "> " #X "node", \
          X##_NODE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            const size_t v0 = VOXEL(x,y,z,nx,ny,nz);                        \
            const size_t v1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);                  \
            FIELD(v0, e##X) = -FIELD(v1, e##X);                             \
            FIELD(v0, tca##X) = -FIELD(v1, tca##X);                         \
          });                                                               \
        break;                                                              \
      case absorb_fields:                                                   \
        Kokkos::parallel_for("apply_local_norm_e<" #X #Y #Z "> " #X "node", \
          X##_NODE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            const size_t v0 = VOXEL(x,y,z,nx,ny,nz);                        \
            const size_t v1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);                  \
            const size_t v2 = VOXEL(x-i*2,y-j*2,z-k*2,nx,ny,nz);            \
            FIELD(v0, e##X) = 2*FIELD(v1, e##X)   - FIELD(v2, e##X);        \
            FIELD(v0, e##X) = 2*FIELD(v1, tca##X) - FIELD(v2, tca##X);      \
          });                                                               \
        break;                                                              \
      default:                                                              \
        ERROR(("Bad boundary condition encountered."));                     \
        break;                                                              \
      }                                                                     \
    }                                                                       \
  } while(0)

  if constexpr( i!=0 && j==0 && k==0 ) {
    APPLY_LOCAL_NORM_E(i,j,k,x,y,z);
  } else if constexpr( i==0 && j!=0 && k==0 ) {
    APPLY_LOCAL_NORM_E(i,j,k,y,z,x);
  } else if constexpr( i==0 && j==0 && k!=0 ) {
    APPLY_LOCAL_NORM_E(i,j,k,z,x,y);
  }
#undef APPLY_LOCAL_NORM_E
}

void
local_ghost_norm_e( field_array_t      * ALIGNED(128) f,
                    const grid_t *              g ) {
  apply_local_norm_e<-1,  0,  0>(f, g);
  apply_local_norm_e< 0, -1,  0>(f, g);
  apply_local_norm_e< 0,  0, -1>(f, g);
  apply_local_norm_e< 1,  0,  0>(f, g);
  apply_local_norm_e< 0,  1,  0>(f, g);
  apply_local_norm_e< 0,  0,  1>(f, g);
}

template<int i, int j, int k> 
void 
apply_local_div_b(field_array_t* fa) {
# define APPLY_LOCAL_DIV_B(i,j,k,X,Y,Z)                                       \
  do {                                                                        \
    const grid_t* g = fa->g;                                                  \
    const int bc = g->bc[BOUNDARY(i,j,k)];                                    \
    if( bc<0 || bc>=world_size ) {                                            \
      const int nx = g->nx, ny = g->ny, nz = g->nz;                           \
      k_field_t k_field = fa->k_f_d;                                          \
      const int face = (i+j+k)<0 ? 0 : n##X+1;                                \
      switch(bc) {                                                            \
      case anti_symmetric_fields:                                             \
        Kokkos::parallel_for("apply_local_div_b<" #X #Y #Z "> " #X "face",    \
          X##_FACE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            const size_t ghost_v = VOXEL(x,y,z,nx,ny,nz);                     \
            const size_t face_v  = VOXEL(x-i,y-j,z-k,nx,ny,nz);               \
            FIELD(ghost_v, div_b_err) = FIELD(face_v, div_b_err);             \
          });                                                                 \
        break;                                                                \
      case symmetric_fields: case pmc_fields:                                 \
        Kokkos::parallel_for("apply_local_div_b<" #X #Y #Z "> " #X "face",    \
          X##_FACE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            const size_t ghost_v = VOXEL(x,y,z,nx,ny,nz);                     \
            const size_t face_v  = VOXEL(x-i,y-j,z-k,nx,ny,nz);               \
            FIELD(ghost_v, div_b_err) = -FIELD(face_v, div_b_err);            \
          });                                                                 \
        break;                                                                \
      case absorb_fields:                                                     \
        Kokkos::parallel_for("apply_local_div_b<" #X #Y #Z "> " #X "face",    \
          X##_FACE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), div_b_err) = 0.0f;                   \
          });                                                                 \
        break;                                                                \
      default:                                                                \
        ERROR(("Bad boundary condition encountered."));                       \
        break;                                                                \
      }                                                                       \
    }                                                                         \
  } while(0)

  if constexpr( i!=0 && j==0 && k==0 ) {
    APPLY_LOCAL_DIV_B(i,j,k,x,y,z);
  } else if constexpr( i==0 && j!=0 && k==0 ) {
    APPLY_LOCAL_DIV_B(i,j,k,y,z,x);
  } else if constexpr( i==0 && j==0 && k!=0 ) {
    APPLY_LOCAL_DIV_B(i,j,k,z,x,y);
  }
#undef APPLY_LOCAL_DIV_B
}

void
local_ghost_div_b( field_array_t      * ALIGNED(128) fa,
                   const grid_t *              g ) {
    apply_local_div_b<-1,  0,  0>( fa );
    apply_local_div_b< 0, -1,  0>( fa );
    apply_local_div_b< 0,  0, -1>( fa );
    apply_local_div_b< 1,  0,  0>( fa );
    apply_local_div_b< 0,  1,  0>( fa );
    apply_local_div_b< 0,  0,  1>( fa );
}

/*****************************************************************************
 * Local adjusts
 *****************************************************************************/

// FIXME: Specialty edge loops should be added to zero e_tang on local
// edges exclusively to handle concave domain geometries

template<int i, int j, int k>
void 
adjust_tang_e(k_field_t& k_field, const grid_t* g, int nx, int ny, int nz) {
# define ADJUST_TANG_E(i,j,k,X,Y,Z)                                       \
  do {                                                                    \
    const int bc = g->bc[BOUNDARY(i,j,k)];                                \
    if( bc<0 || bc>=world_size ) {                                        \
      const int face = (i+j+k)<0 ? 1 : n##X+1;                            \
      switch(bc) {                                                        \
      case anti_symmetric_fields:                                         \
        Kokkos::parallel_for("adjust_tang_e<" #X #Y #Z "> " #Y #Z "edge", \
          Y##Z##_EDGE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), e##Y) = 0.0f;                    \
            FIELD(VOXEL(x,y,z,nx,ny,nz), tca##Y) = 0.0f;                  \
          });                                                             \
        Kokkos::parallel_for("adjust_tang_e<" #X #Y #Z "> " #Z #Y "edge", \
          Z##Y##_EDGE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), e##Z) = 0.0f;                    \
            FIELD(VOXEL(x,y,z,nx,ny,nz), tca##Z) = 0.0f;                  \
          });                                                             \
        break;                                                            \
      case symmetric_fields: case pmc_fields: case absorb_fields:         \
        break;                                                            \
      default:                                                            \
        ERROR(("Bad boundary condition encountered."));                   \
        break;                                                            \
      }                                                                   \
    }                                                                     \
  } while(0)

  if constexpr( i!=0 && j==0 && k==0 ) {
    ADJUST_TANG_E(i,j,k,x,y,z);
  } else if constexpr( i==0 && j!=0 && k==0 ) {
    ADJUST_TANG_E(i,j,k,y,z,x);
  } else if constexpr( i==0 && j==0 && k!=0 ) {
    ADJUST_TANG_E(i,j,k,z,x,y);
  }
#undef ADJUST_TANG_E
}

void
local_adjust_tang_e( field_array_t      * RESTRICT f,
                     const grid_t *              g ) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;

    k_field_t& k_field = f->k_f_d;
    adjust_tang_e<-1,  0,  0>(k_field, g, nx, ny, nz);
    adjust_tang_e< 0, -1,  0>(k_field, g, nx, ny, nz);
    adjust_tang_e< 0,  0, -1>(k_field, g, nx, ny, nz);
    adjust_tang_e< 1,  0,  0>(k_field, g, nx, ny, nz);
    adjust_tang_e< 0,  1,  0>(k_field, g, nx, ny, nz);
    adjust_tang_e< 0,  0,  1>(k_field, g, nx, ny, nz);
}

void
local_adjust_norm_b( field_array_t * RESTRICT fa,
                     const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;

  int bc = 0;
  int face = 0;
  int xl = 0;
  int xh = 0;
  int yl = 0;
  int yh = 0;
  int zl = 0;
  int zh = 0;
  int x = 0;
  int y = 0;
  int z = 0;

// TODO: Test the macro unrolling and parallel_for here. This does not
// get touched during a normal harris run
# define K_ADJUST_NORM_B(i,j,k,X,Y,Z)                                         \
  do {                                                                        \
    bc = g->bc[BOUNDARY(i,j,k)];                                              \
    if( bc<0 || bc>=world_size ) {                                            \
      face = (i+j+k)<0 ? 1 : n##X+1;                                          \
      switch(bc) {                                                            \
      case anti_symmetric_fields: case pmc_fields: case absorb_fields:        \
        break;                                                                \
      case symmetric_fields:                                                  \
         switch(X) {                                                          \
           case('x'):                                                         \
             xl=face, xh=face, yl=1, yh=ny, zl=1, zh=nz;                      \
             break;                                                           \
           case('y'):                                                         \
             xl=1, xh=nx, yl=face, yh=face, zl=1, zh=nz;                      \
             break;                                                           \
           case('z'):                                                         \
             xl=1, xh=nx, yl=1, yh=ny, zl=face, zh=face;                      \
             break;                                                           \
           default:                                                           \
             ERROR(("Bad boundary condition encountered."));                  \
             break;                                                           \
         }                                                                    \
         assert(0);                                                           \
         Kokkos::parallel_for(Kokkos::RangePolicy(zl, zh),                    \
           KOKKOS_LAMBDA (const int z) {                                      \
           for(int yi=yl; yi<=yh; yi++ ) {                                    \
             for(int xj=xl; xj<=xh; xj++ ) {                                  \
              (fa->k_f_h)(VOXEL(xj,yi,z, nx,ny,nz), field_var::cb##X) = 0;    \
             }                                                                \
           }                                                                  \
         });                                                                  \
        break;                                                                \
      default:                                                                \
        ERROR(("Bad boundary condition encountered."));                       \
        break;                                                                \
      }                                                                       \
    }                                                                         \
  } while(0)

  K_ADJUST_NORM_B(-1, 0, 0,x,y,z);
  K_ADJUST_NORM_B( 0,-1, 0,y,z,x);
  K_ADJUST_NORM_B( 0, 0,-1,z,x,y);
  K_ADJUST_NORM_B( 1, 0, 0,x,y,z);
  K_ADJUST_NORM_B( 0, 1, 0,y,z,x);
  K_ADJUST_NORM_B( 0, 0, 1,z,x,y);
}

template <int i, int j, int k> 
void 
adjust_div_e_err(field_array_t* fa, const grid_t* g) {
# define ADJUST_DIV_E_ERR(i,j,k,X,Y,Z)                                        \
  do {                                                                        \
    const int bc = g->bc[BOUNDARY(i,j,k)];                                    \
    if( bc<0 || bc>=world_size ) {                                            \
      const int nx = g->nx, ny = g->ny, nz = g->nz;                           \
      k_field_t k_field = fa->k_f_d;                                          \
      const int face = (i+j+k)<0 ? 1 : n##X+1;                                \
      switch(bc) {                                                            \
      case anti_symmetric_fields: case absorb_fields:                         \
        Kokkos::parallel_for("adjust_div_e_err<" #X #Y #Z "> " #X "node",     \
          X##_NODE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), div_e_err) = 0.0f;                   \
          });                                                                 \
        break;                                                                \
      case symmetric_fields: case pmc_fields:                                 \
        break;                                                                \
      default:                                                                \
        ERROR(("Bad boundary condition encountered."));                       \
        break;                                                                \
      }                                                                       \
    }                                                                         \
  } while(0)

  if constexpr( i!=0 && j==0 && k==0 ) {
    ADJUST_DIV_E_ERR(i,j,k,x,y,z);
  } else if constexpr( i==0 && j!=0 && k==0 ) {
    ADJUST_DIV_E_ERR(i,j,k,y,z,x);
  } else if constexpr( i==0 && j==0 && k!=0 ) {
    ADJUST_DIV_E_ERR(i,j,k,z,x,y);
  }
#undef ADJUST_DIV_E_ERR
}

void
local_adjust_div_e( field_array_t      * ALIGNED(128) f,
                    const grid_t *              g ) {
  adjust_div_e_err<-1,  0,  0>(f, g);
  adjust_div_e_err< 0, -1,  0>(f, g);
  adjust_div_e_err< 0,  0, -1>(f, g);
  adjust_div_e_err< 1,  0,  0>(f, g);
  adjust_div_e_err< 0,  1,  0>(f, g);
  adjust_div_e_err< 0,  0,  1>(f, g);
}

// anti_symmetric => Opposite sign image charges (zero jf_tang)
// symmetric      => Same sign image charges (double jf_tang) 
// absorbing      => No image charges, half cell accumulation (double jf_tang)
// (rhob/jf_norm account for particles that hit boundary and reflect/stick)

template<int i, int j, int k> 
void 
adjust_jf(field_array_t* fa, const grid_t* g ) {
# define ADJUST_JF(i,j,k,X,Y,Z)                                         \
  do {                                                                  \
    const int bc = g->bc[BOUNDARY(i,j,k)];                              \
    if( bc<0 || bc>=world_size ) {                                      \
      const int nx = g->nx, ny = g->ny, nz = g->nz;                     \
      k_field_t k_field = fa->k_f_d;                                    \
      const int face = (i+j+k)<0 ? 1 : n##X+1;                          \
      switch(bc) {                                                      \
      case anti_symmetric_fields:                                       \
        Kokkos::parallel_for("adjust_jf<" #X #Y #Z "> " #Y #Z "edge",   \
          Y##Z##_EDGE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), jf##Y) = 0.0f;                 \
          });                                                           \
        Kokkos::parallel_for("adjust_jf<" #X #Y #Z "> " #Z #Y "edge",   \
          Z##Y##_EDGE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), jf##Z) = 0.0f;                 \
          });                                                           \
        break;                                                          \
      case symmetric_fields: case pmc_fields: case absorb_fields:       \
        Kokkos::parallel_for("adjust_jf<" #X #Y #Z "> " #Y #Z "edge",   \
          Y##Z##_EDGE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), jf##Y) *= 2.0f;                \
          });                                                           \
        Kokkos::parallel_for("adjust_jf<" #X #Y #Z "> " #Z #Y "edge",   \
          Z##Y##_EDGE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), jf##Z) *= 2.0f;                \
          });                                                           \
        break;                                                          \
      default:                                                          \
        ERROR(("Bad boundary condition encountered."));                 \
        break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)

  if constexpr( i!=0 && j==0 && k==0 ) {
    ADJUST_JF(i,j,k,x,y,z);
  } else if constexpr( i==0 && j!=0 && k==0 ) {
    ADJUST_JF(i,j,k,y,z,x);
  } else if constexpr( i==0 && j==0 && k!=0 ) {
    ADJUST_JF(i,j,k,z,x,y);
  }
#undef ADJUST_JF
}

void local_adjust_jf(field_array_t* fa, const grid_t* g) {
  adjust_jf<-1,  0,  0>(fa, g);
  adjust_jf< 0, -1,  0>(fa, g);
  adjust_jf< 0,  0, -1>(fa, g);
  adjust_jf< 1,  0,  0>(fa, g);
  adjust_jf< 0,  1,  0>(fa, g);
  adjust_jf< 0,  0,  1>(fa, g);
}

void reduce_jf(field_array_t* RESTRICT fa ) {
  int n_fields = fa->g->nv;
  auto& kad = fa->k_jf_accum_d;
  auto& kah = fa->k_jf_accum_h;
  auto& kfd = fa->k_f_d;
  // Move the current to the accumulator on device
  Kokkos::deep_copy(kad,kah);
  // Sum the accumulator into the field
  // TODO: Is this the right range policy?
  Kokkos::parallel_for("Add jf accumulation to device jf", 
    Kokkos::RangePolicy(0, n_fields), KOKKOS_LAMBDA (const int i) {
    kfd(i, field_var::jfx) += kad(i, accumulator_var::jx);
    kfd(i, field_var::jfy) += kad(i, accumulator_var::jy);
    kfd(i, field_var::jfz) += kad(i, accumulator_var::jz);
  });
  // Clear the accumulators on the host
  Kokkos::deep_copy(kah,0.0f);
}

// anti_symmetric => Opposite sign image charges (zero rhof/rhob)
// symmetric      => Same sign image charges (double rhof)
//                => (double rhof, rhob is already correct)
// absorbing      => No image charges, half cell accumulation (double rhof)
// (rhob/jf_norm account for particles that hit the boundary)

template<int i, int j, int k> 
void 
adjust_rhof(field_array_t* fa, const grid_t* g) {
# define ADJUST_RHOF(i,j,k,X,Y,Z)                                       \
  do {                                                                  \
    const int bc = g->bc[BOUNDARY(i,j,k)];                              \
    if( bc<0 || bc>=world_size ) {                                      \
      const int nx = g->nx, ny = g->ny, nz = g->nz;                     \
      k_field_t k_field = fa->k_f_d;                                    \
      const int face = (i+j+k)<0 ? 1 : n##X+1;                          \
      switch(bc) {                                                      \
      case anti_symmetric_fields:                                       \
        Kokkos::parallel_for("adjust_rhof<" #X #Y #Z "> " #X "node",    \
          X##_NODE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), rhof) = 0.0f;                  \
          });                                                           \
        break;                                                          \
      case symmetric_fields: case pmc_fields: case absorb_fields:       \
        Kokkos::parallel_for("adjust_rhof<" #X #Y #Z "> " #X "node",    \
          X##_NODE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), rhof) *= 2.0f;                 \
          });                                                           \
        break;                                                          \
      default:                                                          \
        ERROR(("Bad boundary condition encountered."));                 \
        break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)

  if constexpr( i!=0 && j==0 && k==0 ) {
    ADJUST_RHOF(i,j,k,x,y,z);
  } else if constexpr( i==0 && j!=0 && k==0 ) {
    ADJUST_RHOF(i,j,k,y,z,x);
  } else if constexpr( i==0 && j==0 && k!=0 ) {
    ADJUST_RHOF(i,j,k,z,x,y);
  }
#undef ADJUST_RHOF
}

void local_adjust_rhof(field_array_t* fa, const grid_t* g) {
  adjust_rhof<-1,  0,  0>(fa, g);
  adjust_rhof< 0, -1,  0>(fa, g);
  adjust_rhof< 0,  0, -1>(fa, g);
  adjust_rhof< 1,  0,  0>(fa, g);
  adjust_rhof< 0,  1,  0>(fa, g);
  adjust_rhof< 0,  0,  1>(fa, g);
}

// anti_symmetric => Opposite sign image charges (zero rhob)
// symmetric      => Same sign image charges (rhob already correct)
// absorbing      => No image charges, half cell accumulation (rhob already
//                   correct)

template<int i, int j, int k> 
void 
adjust_rhob(field_array_t* fa, const grid_t* g) {
# define ADJUST_RHOB(i,j,k,X,Y,Z)                                       \
  do {                                                                  \
    const int bc = g->bc[BOUNDARY(i,j,k)];                              \
    if( bc<0 || bc>=world_size ) {                                      \
      const int nx = g->nx, ny = g->ny, nz = g->nz;                     \
      k_field_t k_field = fa->k_f_d;                                    \
      const int face = (i+j+k)<0 ? 1 : n##X+1;                          \
      switch(bc) {                                                      \
      case anti_symmetric_fields:                                       \
        Kokkos::parallel_for("adjust_rhob<" #X #Y #Z "> " #X "node",    \
          X##_NODE_POLICY(face), KOKKOS_LAMBDA(const int x, const int y, const int z) { \
            FIELD(VOXEL(x,y,z,nx,ny,nz), rhob) = 0.0f;                  \
          });                                                           \
        break;                                                          \
      case symmetric_fields: case pmc_fields: case absorb_fields:       \
        break;                                                          \
      default:                                                          \
        ERROR(("Bad boundary condition encountered."));                 \
        break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)

  if constexpr( i!=0 && j==0 && k==0 ) {
    ADJUST_RHOB(i,j,k,x,y,z);
  } else if constexpr( i==0 && j!=0 && k==0 ) {
    ADJUST_RHOB(i,j,k,y,z,x);
  } else if constexpr( i==0 && j==0 && k!=0 ) {
    ADJUST_RHOB(i,j,k,z,x,y);
  }
#undef ADJUST_RHOB
}

void local_adjust_rhob(field_array_t* fa, const grid_t* g) {
  adjust_rhob<-1,  0,  0>(fa, g);
  adjust_rhob< 0, -1,  0>(fa, g);
  adjust_rhob< 0,  0, -1>(fa, g);
  adjust_rhob< 1,  0,  0>(fa, g);
  adjust_rhob< 0,  1,  0>(fa, g);
  adjust_rhob< 0,  0,  1>(fa, g);
}

#undef FIELD
#undef XYZ_POLICY

#undef yz_EDGE_POLICY
#undef zx_EDGE_POLICY
#undef xy_EDGE_POLICY

#undef zy_EDGE_POLICY
#undef xz_EDGE_POLICY
#undef yx_EDGE_POLICY

#undef x_NODE_POLICY 
#undef y_NODE_POLICY 
#undef z_NODE_POLICY 

#undef x_FACE_POLICY 
#undef y_FACE_POLICY 
#undef z_FACE_POLICY 
