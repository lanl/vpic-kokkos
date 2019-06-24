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

#define f(x,y,z)         f[ VOXEL(x,y,z, nx,ny,nz) ]

#define XYZ_LOOP(xl,xh,yl,yh,zl,zh)		\
  for( z=zl; z<=zh; z++ )			\
    for( y=yl; y<=yh; y++ )			\
      for( x=xl; x<=xh; x++ )

#define yz_EDGE_LOOP(x) XYZ_LOOP(x,x,1,ny,1,nz+1)
#define zx_EDGE_LOOP(y) XYZ_LOOP(1,nx+1,y,y,1,nz)
#define xy_EDGE_LOOP(z) XYZ_LOOP(1,nx,1,ny+1,z,z)

#define zy_EDGE_LOOP(x) XYZ_LOOP(x,x,1,ny+1,1,nz)
#define xz_EDGE_LOOP(y) XYZ_LOOP(1,nx,y,y,1,nz+1)
#define yx_EDGE_LOOP(z) XYZ_LOOP(1,nx+1,1,ny,z,z)

#define x_NODE_LOOP(x) XYZ_LOOP(x,x,1,ny+1,1,nz+1)
#define y_NODE_LOOP(y) XYZ_LOOP(1,nx+1,y,y,1,nz+1)
#define z_NODE_LOOP(z) XYZ_LOOP(1,nx+1,1,ny+1,z,z)

#define x_FACE_LOOP(x) XYZ_LOOP(x,x,1,ny,1,nz)
#define y_FACE_LOOP(y) XYZ_LOOP(1,nx,y,y,1,nz)
#define z_FACE_LOOP(z) XYZ_LOOP(1,nx,1,ny,z,z)

typedef class XYZ {} XYZ;
typedef class YZX {} YZX;
typedef class ZXY {} ZXY;
typedef class XY {} XY;
typedef class YX {} YX;
typedef class XZ {} XZ;
typedef class ZX {} ZX;
typedef class YZ {} YZ;
typedef class ZY {} ZY;

/*****************************************************************************
 * Local ghosts
 *****************************************************************************/

void
local_ghost_tang_b( field_t      * ALIGNED(128) f,
                    const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  const float cdt_dx = g->cvac*g->dt*g->rdx;
  const float cdt_dy = g->cvac*g->dt*g->rdy;
  const float cdt_dz = g->cvac*g->dt*g->rdz;
  int bc, face, ghost, x, y, z;
  float decay, drive, higend, t1, t2;
  field_t *fg, *fh;

  // Absorbing boundary condition is 2nd order accurate implementation
  // of a 1st order Higend ABC with 15 degree annihilation cone except
  // for 1d simulations where the 2nd order accurate implementation of
  // a 1st order Mur boundary condition is used.
  higend = ( nx>1 || ny>1 || nz>1 ) ? 1.03527618 : 1.;

# define APPLY_LOCAL_TANG_B(i,j,k,X,Y,Z)                                 \
  do {                                                                   \
    bc = g->bc[BOUNDARY(i,j,k)];                                         \
    if( bc<0 || bc>=world_size ) {                                       \
      ghost = (i+j+k)<0 ? 0 : n##X+1;                                    \
      face  = (i+j+k)<0 ? 1 : n##X+1;                                    \
      switch(bc) {                                                       \
      case anti_symmetric_fields:                                        \
	Z##Y##_EDGE_LOOP(ghost) f(x,y,z).cb##Y= f(x-i,y-j,z-k).cb##Y;    \
	Y##Z##_EDGE_LOOP(ghost) f(x,y,z).cb##Z= f(x-i,y-j,z-k).cb##Z;    \
	break;                                                           \
      case symmetric_fields: case pmc_fields:                            \
	Z##Y##_EDGE_LOOP(ghost) f(x,y,z).cb##Y=-f(x-i,y-j,z-k).cb##Y;    \
	Y##Z##_EDGE_LOOP(ghost) f(x,y,z).cb##Z=-f(x-i,y-j,z-k).cb##Z;    \
	break;                                                           \
      case absorb_fields:                                                \
        drive = cdt_d##X*higend;                                         \
        decay = (1-drive)/(1+drive);                                     \
        drive = 2*drive/(1+drive);                                       \
	Z##Y##_EDGE_LOOP(ghost) {                                        \
          fg = &f(x,y,z);                                                \
          fh = &f(x-i,y-j,z-k);                                          \
          X = face;                                                      \
          t1 = cdt_d##X*( f(x-i,y-j,z-k).e##Z - f(x,y,z).e##Z );         \
          t1 = (i+j+k)<0 ? t1 : -t1;                                     \
          X = ghost;                                                     \
          Z++; t2 = f(x-i,y-j,z-k).e##X;                                 \
          Z--; t2 = cdt_d##Z*( t2 - fh->e##X );                          \
          fg->cb##Y = decay*fg->cb##Y + drive*fh->cb##Y - t1 + t2;       \
        }                                                                \
	Y##Z##_EDGE_LOOP(ghost) {                                        \
          fg = &f(x,y,z);                                                \
          fh = &f(x-i,y-j,z-k);                                          \
          X = face;                                                      \
          t1 = cdt_d##X*( f(x-i,y-j,z-k).e##Y - f(x,y,z).e##Y );         \
          t1 = (i+j+k)<0 ? t1 : -t1;                                     \
          X = ghost;                                                     \
          Y++; t2 = f(x-i,y-j,z-k).e##X;                                 \
          Y--; t2 = cdt_d##Y*( t2 - fh->e##X );                          \
          fg->cb##Z = decay*fg->cb##Z + drive*fh->cb##Z + t1 - t2;       \
        }                                                                \
	break;                                                           \
      default:                                                           \
	ERROR(("Bad boundary condition encountered."));                  \
	break;                                                           \
      }                                                                  \
    }                                                                    \
  } while(0)

  APPLY_LOCAL_TANG_B(-1, 0, 0,x,y,z);
  APPLY_LOCAL_TANG_B( 0,-1, 0,y,z,x);
  APPLY_LOCAL_TANG_B( 0, 0,-1,z,x,y);
  APPLY_LOCAL_TANG_B( 1, 0, 0,x,y,z);
  APPLY_LOCAL_TANG_B( 0, 1, 0,y,z,x);
  APPLY_LOCAL_TANG_B( 0, 0, 1,z,x,y);
}

template<typename T> void apply_local_tang_b(int i, int j, int k, 
                                        const int nx, const int ny, const int nz, 
                                        const float cdt_dx, const float cdt_dy, const float cdt_dz,
                                        float higend, field_array_t* RESTRICT f, const grid_t* g) {}

template<> void apply_local_tang_b<XYZ>(int i, int j, int k,
                                        const int nx, const int ny, const int nz, 
                                        const float cdt_dx, const float cdt_dy, const float cdt_dz,
                                        float higend, field_array_t* RESTRICT f, const grid_t* g) {
    float drive, decay;
    int bc = g->bc[BOUNDARY(i,j,k)];
    k_field_t k_field = f->k_f_d;
    Kokkos::MDRangePolicy<Kokkos::Rank<2> > zy_edge({1,1},{nz+1,ny+2});
    Kokkos::MDRangePolicy<Kokkos::Rank<2> > yz_edge({1,1},{nz+2,ny+1});
//    Kokkos::View<float**> cby_copy = Kokkos::View<float**>("temporary buffer for XYZ absorb fields: zy edge", nz+2, ny+2);


    if(bc < 0 || bc >= world_size) {
        int ghost = (i+j+k)<0 ? 0 : nx+1;
        int face  = (i+j+k)<0 ? 1 : nx+1;
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: anti_symmetric_fields: ZY Edge loop", zy_edge, KOKKOS_LAMBDA(const int z, const int y) {
                    k_field(VOXEL(ghost,y,z,nx,ny,nz), field_var::cby) = k_field(VOXEL(ghost-i,y-j,z-k,nx,ny,nz), field_var::cby);
                });

                Kokkos::parallel_for("apply_local_tang_b<XYZ>: anti_symmetric_fields: YZ Edge loop", yz_edge, KOKKOS_LAMBDA(const int z, const int y) {
                    k_field(VOXEL(ghost,y,z,nx,ny,nz), field_var::cbz) = k_field(VOXEL(ghost-i,y-j,z-k,nx,ny,nz), field_var::cbz);
                });

                break;
            case symmetric_fields:
            case pmc_fields:
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: pmc_fields: ZY Edge loop", zy_edge, KOKKOS_LAMBDA(const int z, const int y) {
                    k_field(VOXEL(ghost,y,z,nx,ny,nz), field_var::cby) = -k_field(VOXEL(ghost-i,y-j,z-k,nx,ny,nz), field_var::cby);
                });

                Kokkos::parallel_for("apply_local_tang_b<XYZ>: pmc_fields: YZ Edge loop", yz_edge, KOKKOS_LAMBDA(const int z, const int y) {
                    k_field(VOXEL(ghost,y,z,nx,ny,nz), field_var::cbz) = -k_field(VOXEL(ghost-i,y-j,z-k,nx,ny,nz), field_var::cbz);
                });

                break;
            case absorb_fields:
                drive = cdt_dx*higend;
                decay = (1-drive)/(1+drive);
                drive = 2*drive/(1+drive);

                Kokkos::parallel_for("XYZ absorb_fields: zy_edge", zy_edge, KOKKOS_LAMBDA(int z, int y) {
                    int x = ghost;
                    const int fg = VOXEL(x,y,z,nx,ny,nz);
                    const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    const float fg_cby = k_field(fg, field_var::cby);
                    const float fh_cby = k_field(fh, field_var::cby);
                    x = face;
                    float t1 = cdt_dx*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    x = ghost;
                    z++;
                    float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex);
                    z--;
                    t2 = cdt_dz * (t2 - k_field(fh, field_var::ex));
                    Kokkos::memory_fence();
                    k_field(fg, field_var::cby) = decay*fg_cby + drive * fh_cby - t1 + t2;
                });
                Kokkos::parallel_for("XYZ absorb_fields: yz_edge", yz_edge, KOKKOS_LAMBDA(int z, int y) {
                    int x = ghost;
                    const int fg = VOXEL(x,y,z,nx,ny,nz);
                    const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    const float fg_cbz = k_field(fg, field_var::cbz);
                    const float fh_cbz = k_field(fh, field_var::cbz);
                    x = face;
                    float t1 = cdt_dx*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    x = ghost;
                    y++;
                    float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex);
                    y--;
                    t2 = cdt_dy * (t2 - k_field(fh, field_var::ex));
                    Kokkos::memory_fence();
                    k_field(fg, field_var::cbz) = decay*fg_cbz + drive * fh_cbz + t1 - t2;
                });
/*
                Kokkos::parallel_for("XYZ absorb_fields: zy_edge", zy_edge, KOKKOS_LAMBDA(int z, int y) {
                    int x = ghost;
                    const int fg = VOXEL(x,y,z,nx,ny,nz);
                    const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    x = face;
                    float t1 = cdt_dx*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    x = ghost;
                    z++;
                    float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex);
                    z--;
                    t2 = cdt_dz * (t2 - k_field(fh, field_var::ex));
                    cby_copy(y,z) = decay*k_field(fg, field_var::cby) + drive * k_field(fh, field_var::cby) - t1 + t2;
                });
                Kokkos::parallel_for("XYZ absorb_fields set: zy_edge", zy_edge, KOKKOS_LAMBDA(int z, int y) {
                    const int x = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz),field_var::cby) = cby_copy(y,z);
                });

                Kokkos::parallel_for("XYZ absorb_fields: yz_edge", yz_edge, KOKKOS_LAMBDA(int z, int y) {
                    int x = ghost;
                    const int fg = VOXEL(x,y,z,nx,ny,nz);
                    const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    x = face;
                    float t1 = cdt_dx*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    x = ghost;
                    y++;
                    float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex);
                    y--;
                    t2 = cdt_dy * (t2 - k_field(fh, field_var::ex));
                    cby_copy(y,z) = decay*k_field(fg, field_var::cbz) + drive * k_field(fh, field_var::cbz) + t1 - t2;
                });
                Kokkos::parallel_for("XYZ absorb_fields set: yz_edge", yz_edge, KOKKOS_LAMBDA(int z, int y) {
                    const int x = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = cby_copy(y,z);
                });
*/
/*
                Kokkos::parallel_for("XYZ absorb_fields serial", 1, KOKKOS_LAMBDA(const int idx) {
                    for(int z=1; z<=nz; z++) {
                        for(int y=1; y<=ny+1; y++) {
                            for(int x=ghost; x<=ghost; x++) {
                                const int fg = VOXEL(x,y,z,nx,ny,nz);
                                const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                                x = face;
                                float t1 = cdt_dx*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez));
                                t1 = (i+j+k)<0 ? t1 : -t1;
                                x = ghost;
                                z++;
                                float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex);
                                z--;
                                t2 = cdt_dz * (t2 - k_field(fh, field_var::ex));
                                k_field(fg, field_var::cby) = decay*k_field(fg, field_var::cby) + drive * k_field(fh, field_var::cby) - t1 + t2;
                            }
                        }
                    }
                    for(int z=1; z<=nz+1; z++) {
                        for(int y=1; y<=ny; y++) {
                            for(int x=ghost; x<=ghost; x++) {
                                const int fg = VOXEL(x,y,z,nx,ny,nz);
                                const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                                x = face;
                                float t1 = cdt_dx*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey));
                                t1 = (i+j+k)<0 ? t1 : -t1;
                                x = ghost;
                                y++;
                                float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex);
                                y--;
                                t2 = cdt_dy * (t2 - k_field(fh, field_var::ex));
                                k_field(fg, field_var::cbz) = decay*k_field(fg, field_var::cbz) + drive * k_field(fh, field_var::cbz) + t1 - t2;
                            }
                        }
                    }
                });
*/
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template<> void apply_local_tang_b<YZX>(int i, int j, int k, 
                                        const int nx, const int ny, const int nz, 
                                        const float cdt_dx, const float cdt_dy, const float cdt_dz,
                                        float higend, field_array_t* RESTRICT f, const grid_t* g) {
    float drive, decay;
    int bc = g->bc[BOUNDARY(i,j,k)];
    k_field_t k_field = f->k_f_d;
    if(bc < 0 || bc >= world_size) {
        int ghost = (i+j+k)<0 ? 0 : ny+1;
        int face = (i+j+k)<0 ? 1 : ny+1;
        Kokkos::MDRangePolicy<Kokkos::Rank<2> > xz_edge({1,1},{nz+2,nx+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2> > zx_edge({1,1},{nz+1,nx+2});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("apply_local_tang_b<YZX>: anti_symmetric_fields: XZ Edge loop", xz_edge, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbz);
                });
                Kokkos::parallel_for("apply_local_tang_b<YZX>: anti_symmetric_fields: ZX Edge loop", zx_edge, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbx);
                });

                break;
            case symmetric_fields:
            case pmc_fields:
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: pmc_fields: XZ Edge loop", xz_edge, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbz);
                });
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: pmc_fields: ZX Edge loop", zx_edge, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbx);
                });

                break;
            case absorb_fields:
                drive = cdt_dy*higend;
                decay = (1-drive)/(1+drive);
                drive = 2*drive/(1+drive);

                Kokkos::parallel_for("YZX absorb_fields: xz_edge", xz_edge, KOKKOS_LAMBDA(int z, int x) {
                    int y = ghost;
                    const int fg = VOXEL(x,y,z,nx,ny,nz);
                    const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    const float fg_cbz = k_field(fg, field_var::cbz);
                    const float fh_cbz = k_field(fh, field_var::cbz);
                    y = face;
                    float t1 = cdt_dy*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    y = ghost;
                    x++;
                    float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey);
                    x--;
                    t2 = cdt_dx * (t2 - k_field(fh, field_var::ey));
                    Kokkos::memory_fence();
                    k_field(fg, field_var::cbz) = decay*fg_cbz + drive * fh_cbz - t1 + t2;
                });
                Kokkos::parallel_for("YZX absorb_fields: zx_edge", zx_edge, KOKKOS_LAMBDA(int z, int x) {
                    int y = ghost;
                    const int fg = VOXEL(x,y,z,nx,ny,nz);
                    const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    const float fg_cbx = k_field(fg, field_var::cbx);
                    const float fh_cbx = k_field(fh, field_var::cbx);
                    y = face;
                    float t1 = cdt_dy*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    y = ghost;
                    z++;
                    float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey);
                    z--;
                    t2 = cdt_dz * (t2 - k_field(fh, field_var::ey));
                    Kokkos::memory_fence();
                    k_field(fg, field_var::cbx) = decay*fg_cbx + drive * fh_cbx + t1 - t2;
                });
/*
                Kokkos::parallel_for("YZX absorb_fields serial", 1, KOKKOS_LAMBDA(const int idx) {
                    for(int z=1; z<=nz+1; z++) {
                        for(int y=ghost; y<=ghost; y++) {
                            for(int x=1; x<=nx; x++) {
                                const int fg = VOXEL(x,y,z,nx,ny,nz);
                                const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                                y = face;
                                float t1 = cdt_dy*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex));
                                t1 = (i+j+k)<0 ? t1 : -t1;
                                y = ghost;
                                x++;
                                float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey);
                                x--;
                                t2 = cdt_dx * (t2 - k_field(fh, field_var::ey));
                                k_field(fg, field_var::cbz) = decay*k_field(fg, field_var::cbz) + drive * k_field(fh, field_var::cbz) - t1 + t2;
                            }
                        }
                    }
                    for(int z=1; z<=nz; z++) {
                        for(int y=ghost; y<=ghost; y++) {
                            for(int x=1; x<=nx+1; x++) {
                                const int fg = VOXEL(x,y,z,nx,ny,nz);
                                const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                                y = face;
                                float t1 = cdt_dy*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez));
                                t1 = (i+j+k)<0 ? t1 : -t1;
                                y = ghost;
                                z++;
                                float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey);
                                z--;
                                t2 = cdt_dz * (t2 - k_field(fh, field_var::ey));
                                k_field(fg, field_var::cbx) = decay*k_field(fg, field_var::cbx) + drive * k_field(fh, field_var::cbx) + t1 - t2;
                            }
                        }
                    }
                });
*/
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template<> void apply_local_tang_b<ZXY>(int i, int j, int k, 
                                        const int nx, const int ny, const int nz, 
                                        const float cdt_dx, const float cdt_dy, const float cdt_dz,
                                        float higend, field_array_t* RESTRICT f, const grid_t* g) {
    float drive, decay;
    int bc = g->bc[BOUNDARY(i,j,k)];
    k_field_t k_field = f->k_f_d;
    if(bc < 0 || bc >= world_size) {
        int ghost = (i+j+k)<0 ? 0 : nz+1;
        int face = (i+j+k)<0 ? 1 : nz+1;
        Kokkos::MDRangePolicy<Kokkos::Rank<2> > yx_edge({1,1},{ny+1,nx+2});
        Kokkos::MDRangePolicy<Kokkos::Rank<2> > xy_edge({1,1},{ny+2,nx+1});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("apply_local_tang_b<ZXY>: anti_symmetric_fields: YX Edge loop", yx_edge, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbx);
                });
                Kokkos::parallel_for("apply_local_tang_b<ZXY>: anti_symmetric_fields: XY Edge loop", xy_edge, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cby);
                });

                break;
            case symmetric_fields:
            case pmc_fields:
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: pmc_fields: YX Edge loop", yx_edge, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbx);
                });
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: pmc_fields: XY Edge loop", xy_edge, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = ghost;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cby);
                });

                break;
            case absorb_fields:
                drive = cdt_dz*higend;
                decay = (1-drive)/(1+drive);
                drive = 2*drive/(1+drive);

                Kokkos::parallel_for("ZXY absorb_fields: yx_edge", yx_edge, KOKKOS_LAMBDA(int y, int x) {
                    int z = ghost;
                    const int fg = VOXEL(x,y,z,nx,ny,nz);
                    const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    const float fg_cbx = k_field(fg, field_var::cbx);
                    const float fh_cbx = k_field(fh, field_var::cbx);
                    z = face;
                    float t1 = cdt_dz*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    z = ghost;
                    y++;
                    float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez);
                    y--;
                    t2 = cdt_dy * (t2 - k_field(fh, field_var::ez));
                    Kokkos::memory_fence();
                    k_field(fg, field_var::cbx) = decay*fg_cbx + drive * fh_cbx - t1 + t2;
                });
                Kokkos::parallel_for("ZXY absorb_fields: xy_edge", xy_edge, KOKKOS_LAMBDA(int y, int x) {
                    int z = ghost;
                    const int fg = VOXEL(x,y,z,nx,ny,nz);
                    const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    const float fg_cby = k_field(fg, field_var::cby);
                    const float fh_cby = k_field(fh, field_var::cby);
                    z = face;
                    float t1 = cdt_dz*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    z = ghost;
                    x++;
                    float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez);
                    x--;
                    t2 = cdt_dx * (t2 - k_field(fh, field_var::ez));
                    Kokkos::memory_fence();
                    k_field(fg, field_var::cby) = decay*fg_cby + drive * fh_cby + t1 - t2;
                });
/*
                Kokkos::parallel_for("ZXY absorb_fields serial", 1, KOKKOS_LAMBDA(const int idx) {
                    for(int z=ghost; z<=ghost; z++) {
                        for(int y=1; y<=ny; y++) {
                            for(int x=1; x<=nx+1; x++) {
                                const int fg = VOXEL(x,y,z,nx,ny,nz);
                                const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                                z = face;
                                float t1 = cdt_dz*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey));
                                t1 = (i+j+k)<0 ? t1 : -t1;
                                z = ghost;
                                y++;
                                float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez);
                                y--;
                                t2 = cdt_dy * (t2 - k_field(fh, field_var::ez));
                                k_field(fg, field_var::cbx) = decay*k_field(fg, field_var::cbx) + drive * k_field(fh, field_var::cbx) - t1 + t2;
                            }
                        }
                    }
                    for(int z=ghost; z<=ghost; z++) {
                        for(int y=1; y<=ny+1; y++) {
                            for(int x=1; x<=nx; x++) {
                                const int fg = VOXEL(x,y,z,nx,ny,nz);
                                const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                                z = face;
                                float t1 = cdt_dz*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex));
                                t1 = (i+j+k)<0 ? t1 : -t1;
                                z = ghost;
                                x++;
                                float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez);
                                x--;
                                t2 = cdt_dx * (t2 - k_field(fh, field_var::ez));
                                k_field(fg, field_var::cby) = decay*k_field(fg, field_var::cby) + drive * k_field(fh, field_var::cby) + t1 - t2;
                            }
                        }
                    }
                });
*/
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}


void
k_local_ghost_tang_b( field_array_t      * RESTRICT f,
                    const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  const float cdt_dx = g->cvac*g->dt*g->rdx;
  const float cdt_dy = g->cvac*g->dt*g->rdy;
  const float cdt_dz = g->cvac*g->dt*g->rdz;
  float higend;

  // Absorbing boundary condition is 2nd order accurate implementation
  // of a 1st order Higend ABC with 15 degree annihilation cone except
  // for 1d simulations where the 2nd order accurate implementation of
  // a 1st order Mur boundary condition is used.
  higend = ( nx>1 || ny>1 || nz>1 ) ? 1.03527618 : 1.;
    apply_local_tang_b<XYZ>(-1,0,0,nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
    apply_local_tang_b<YZX>(0,-1,0,nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
    apply_local_tang_b<ZXY>(0,0,-1,nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
    apply_local_tang_b<XYZ>(1,0,0,nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
    apply_local_tang_b<YZX>(0,1,0,nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
    apply_local_tang_b<ZXY>(0,0,1,nx,ny,nz,cdt_dx,cdt_dy,cdt_dz,higend,f,g);
}

// Note: local_adjust_div_e zeros the error on the boundaries for
// absorbing boundary conditions.  Thus, ghost norm e value is
// irrevelant.

void
local_ghost_norm_e( field_t      * ALIGNED(128) f,
                    const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int bc, face, x, y, z;
  field_t * ALIGNED(16) f0, * ALIGNED(16) f1, * ALIGNED(16) f2;

# define APPLY_LOCAL_NORM_E(i,j,k,X,Y,Z)                        \
  do {                                                          \
    bc = g->bc[BOUNDARY(i,j,k)];                                \
    if( bc<0 || bc>=world_size ) {                              \
      face = (i+j+k)<0 ? 0 : n##X+1;                            \
      switch(bc) {                                              \
      case anti_symmetric_fields:                               \
	X##_NODE_LOOP(face) {                                   \
          f0 = &f(x,y,z);                                       \
          f1 = &f(x-i,y-j,z-k);                                 \
          f0->e##X   = f1->e##X;                                \
          f0->tca##X = f1->tca##X;                              \
        }                                                       \
	break;                                                  \
      case symmetric_fields: case pmc_fields:                   \
	X##_NODE_LOOP(face) {                                   \
          f0 = &f(x,y,z);                                       \
          f1 = &f(x-i,y-j,z-k);                                 \
          f0->e##X   = -f1->e##X;                               \
          f0->tca##X = -f1->tca##X;                             \
        }                                                       \
	break;                                                  \
      case absorb_fields:                                       \
	X##_NODE_LOOP(face) {                                   \
          f0 = &f(x,y,z);                                       \
          f1 = &f(x-i,y-j,z-k);                                 \
          f2 = &f(x-i*2,y-j*2,z-k*2);                           \
          f0->e##X   = 2*f1->e##X   - f2->e##X;                 \
          f0->tca##X = 2*f1->tca##X - f2->tca##X;               \
        }                                                       \
	break;                                                  \
      default:                                                  \
	ERROR(("Bad boundary condition encountered."));         \
	break;                                                  \
      }                                                         \
    }                                                           \
  } while(0)

  APPLY_LOCAL_NORM_E(-1, 0, 0,x,y,z);
  APPLY_LOCAL_NORM_E( 0,-1, 0,y,z,x);
  APPLY_LOCAL_NORM_E( 0, 0,-1,z,x,y);
  APPLY_LOCAL_NORM_E( 1, 0, 0,x,y,z);
  APPLY_LOCAL_NORM_E( 0, 1, 0,y,z,x);
  APPLY_LOCAL_NORM_E( 0, 0, 1,z,x,y);
}

template <typename T> void apply_local_norm_e(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {}
template <> void apply_local_norm_e<XYZ>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        int face = (i+j+k) < 0 ? 0 : nx+1;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_policy({1, 1}, {nz+2, ny+2});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("apply_local_norm_e<XYZ>", zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    const int f0 = VOXEL(x,y,z,nx,ny,nz);
                    const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    k_field(f0, field_var::ex)   = k_field(f1, field_var::ex);
                    k_field(f0, field_var::tcax) = k_field(f1, field_var::tcax);
                });
/*
                Kokkos::parallel_for("apply_local_norm_e<XYZ>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const int yi) {
                        const int zi = team_member.league_rank();
                        const int x = face;
                        const int y = yi + 1;
                        const int z = zi + 1;
                        const int f0 = VOXEL(x,y,z,nx,ny,nz);
                        const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        k_field(f0, field_var::ex)   = k_field(f1, field_var::ex);
                        k_field(f0, field_var::tcax) = k_field(f1, field_var::tcax);
                    });
                });
*/
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                Kokkos::parallel_for("apply_local_norm_e<XYZ>", zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    const int f0 = VOXEL(x,y,z,nx,ny,nz);
                    const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    k_field(f0, field_var::ex)   = -k_field(f1, field_var::ex);
                    k_field(f0, field_var::tcax) = -k_field(f1, field_var::tcax);
                });
/*
                Kokkos::parallel_for("apply_local_norm_e<XYZ>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const int yi) {
                        const int zi = team_member.league_rank();
                        const int x = face;
                        const int y = yi + 1;
                        const int z = zi + 1;
                        const int f0 = VOXEL(x,y,z,nx,ny,nz);
                        const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        k_field(f0, field_var::ex)   = -k_field(f1, field_var::ex);
                        k_field(f0, field_var::tcax) = -k_field(f1, field_var::tcax);
                    });
                });
*/
                break;
            case absorb_fields:
                Kokkos::parallel_for("apply_local_norm_e<XYZ>", zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    const int f0 = VOXEL(x,y,z,nx,ny,nz);
                    const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    const int f2 = VOXEL(x-i*2,y-j*2,z-k*2,nx,ny,nz);
                    k_field(f0, field_var::ex)   = 2*k_field(f1, field_var::ex)   - k_field(f2, field_var::ex);
                    k_field(f0, field_var::tcax) = 2*k_field(f1, field_var::tcax) - k_field(f2, field_var::tcax);
                });
/*
                Kokkos::parallel_for("apply_local_norm_e<XYZ>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const int yi) {
                        const int zi = team_member.league_rank();
                        const int x = face;
                        const int y = yi + 1;
                        const int z = zi + 1;
                        const int f0 = VOXEL(x,y,z,nx,ny,nz);
                        const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        const int f2 = VOXEL(x-i*2,y-j*2,z-k*2,nx,ny,nz);
                        k_field(f0, field_var::ex)   = 2*k_field(f1, field_var::ex)   - k_field(f2, field_var::ex);
                        k_field(f0, field_var::tcax) = 2*k_field(f1, field_var::tcax) - k_field(f2, field_var::tcax);
                    });
                });
*/
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template <> void apply_local_norm_e<YZX>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        int face = (i+j+k) < 0 ? 0 : ny+1;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_policy({1, 1}, {nz+2, nx+2});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("apply_local_norm_e<YZX>", zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    const int f0 = VOXEL(x,y,z,nx,ny,nz);
                    const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    k_field(f0, field_var::ey)   = k_field(f1, field_var::ey);
                    k_field(f0, field_var::tcay) = k_field(f1, field_var::tcay);
                });
/*
                Kokkos::parallel_for("apply_local_norm_e<YZX>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                        const int zi = team_member.league_rank();
                        const int x = xi + 1;
                        const int y = face;
                        const int z = zi + 1;
                        const int f0 = VOXEL(x,y,z,nx,ny,nz);
                        const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        k_field(f0, field_var::ey)   = k_field(f1, field_var::ey);
                        k_field(f0, field_var::tcay) = k_field(f1, field_var::tcay);
                    });
                });
*/
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                Kokkos::parallel_for("apply_local_norm_e<YZX>", zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    const int f0 = VOXEL(x,y,z,nx,ny,nz);
                    const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    k_field(f0, field_var::ey)   = -k_field(f1, field_var::ey);
                    k_field(f0, field_var::tcay) = -k_field(f1, field_var::tcay);
                });
/*
                Kokkos::parallel_for("apply_local_norm_e<YZX>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                        const int zi = team_member.league_rank();
                        const int x = xi + 1;
                        const int y = face;
                        const int z = zi + 1;
                        const int f0 = VOXEL(x,y,z,nx,ny,nz);
                        const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        k_field(f0, field_var::ey)   = -k_field(f1, field_var::ey);
                        k_field(f0, field_var::tcay) = -k_field(f1, field_var::tcay);
                    });
                });
*/
                break;
            case absorb_fields:
                Kokkos::parallel_for("apply_local_norm_e<YZX>", zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    const int f0 = VOXEL(x,y,z,nx,ny,nz);
                    const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    const int f2 = VOXEL(x-i*2,y-j*2,z-k*2,nx,ny,nz);
                    k_field(f0, field_var::ey)   = 2*k_field(f1, field_var::ey)   - k_field(f2, field_var::ey);
                    k_field(f0, field_var::tcay) = 2*k_field(f1, field_var::tcay) - k_field(f2, field_var::tcay);
                });
/*
                Kokkos::parallel_for("apply_local_norm_e<YZX>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                        const int zi = team_member.league_rank();
                        const int x = xi + 1;
                        const int y = face;
                        const int z = zi + 1;
                        const int f0 = VOXEL(x,y,z,nx,ny,nz);
                        const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        const int f2 = VOXEL(x-i*2,y-j*2,z-k*2,nx,ny,nz);
                        k_field(f0, field_var::ey)   = 2*k_field(f1, field_var::ey)   - k_field(f2, field_var::ey);
                        k_field(f0, field_var::tcay) = 2*k_field(f1, field_var::tcay) - k_field(f2, field_var::tcay);
                    });
                });
*/
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template <> void apply_local_norm_e<ZXY>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        int face = (i+j+k) < 0 ? 0 : nz+1;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_policy({1,1}, {ny+2, nx+2});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("apply_local_norm_e<ZXY>", yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    const int f0 = VOXEL(x,y,z,nx,ny,nz);
                    const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    k_field(f0, field_var::ez)   = k_field(f1, field_var::ez);
                    k_field(f0, field_var::tcaz) = k_field(f1, field_var::tcaz);
                });
/*
                Kokkos::parallel_for("apply_local_norm_e<ZXY>", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                        const int yi = team_member.league_rank();
                        const int x = xi + 1;
                        const int y = yi + 1;
                        const int z = face;
                        const int f0 = VOXEL(x,y,z,nx,ny,nz);
                        const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        k_field(f0, field_var::ez)   = k_field(f1, field_var::ez);
                        k_field(f0, field_var::tcaz) = k_field(f1, field_var::tcaz);
                    });
                });
*/
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                Kokkos::parallel_for("apply_local_norm_e<ZXY>", yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    const int f0 = VOXEL(x,y,z,nx,ny,nz);
                    const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    k_field(f0, field_var::ez)   = -k_field(f1, field_var::ez);
                    k_field(f0, field_var::tcaz) = -k_field(f1, field_var::tcaz);
                });
/*
                Kokkos::parallel_for("apply_local_norm_e<ZXY>", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                        const int yi = team_member.league_rank();
                        const int x = xi + 1;
                        const int y = yi + 1;
                        const int z = face;
                        const int f0 = VOXEL(x,y,z,nx,ny,nz);
                        const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        k_field(f0, field_var::ez)   = -k_field(f1, field_var::ez);
                        k_field(f0, field_var::tcaz) = -k_field(f1, field_var::tcaz);
                    });
                });
*/
                break;
            case absorb_fields:
                Kokkos::parallel_for("apply_local_norm_e<ZXY>", yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    const int f0 = VOXEL(x,y,z,nx,ny,nz);
                    const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                    const int f2 = VOXEL(x-i*2,y-j*2,z-k*2,nx,ny,nz);
                    k_field(f0, field_var::ez)   = 2*k_field(f1, field_var::ez)   - k_field(f2, field_var::ez);
                    k_field(f0, field_var::tcaz) = 2*k_field(f1, field_var::tcaz) - k_field(f2, field_var::tcaz);
                });
/*
                Kokkos::parallel_for("apply_local_norm_e<ZXY>", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                        const int yi = team_member.league_rank();
                        const int x = xi + 1;
                        const int y = yi + 1;
                        const int z = face;
                        const int f0 = VOXEL(x,y,z,nx,ny,nz);
                        const int f1 = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        const int f2 = VOXEL(x-i*2,y-j*2,z-k*2,nx,ny,nz);
                        k_field(f0, field_var::ez)   = 2*k_field(f1, field_var::ez)   - k_field(f2, field_var::ez);
                        k_field(f0, field_var::tcaz) = 2*k_field(f1, field_var::tcaz) - k_field(f2, field_var::tcaz);
                    });
                });
*/
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}

void
k_local_ghost_norm_e( field_array_t      * ALIGNED(128) f,
                    const grid_t *              g ) {
    apply_local_norm_e<XYZ>(f, g, -1,  0,  0);
    apply_local_norm_e<YZX>(f, g,  0, -1,  0);
    apply_local_norm_e<ZXY>(f, g,  0,  0, -1);
    apply_local_norm_e<XYZ>(f, g, 1, 0, 0);
    apply_local_norm_e<YZX>(f, g, 0, 1, 0);
    apply_local_norm_e<ZXY>(f, g, 0, 0, 1);
}

template<typename T> void apply_local_div_b(field_array_t* fa, const int i, const int j, const int k) {}
template<> void apply_local_div_b<XYZ>(field_array_t* fa, const int i, const int j, const int k) {
    const grid_t* g = fa->g;
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        const int face = (i+j+k)<0 ? 0 : nx+1;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_face({1, 1}, {nz+1, ny+1});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("apply_local_div_b<XYZ>: anti_symmetric_fields", x_face, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::div_b_err);
                });
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                Kokkos::parallel_for("apply_local_div_b<XYZ>: pmc_fields", x_face, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::div_b_err);
                });
                break;
            case absorb_fields:
                Kokkos::parallel_for("apply_local_div_b<XYZ>: absorb_fields", x_face, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = 0.f;
                });
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template<> void apply_local_div_b<YZX>(field_array_t* fa, const int i, const int j, const int k) {
    const grid_t* g = fa->g;
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        const int face = (i+j+k)<0 ? 0 : ny+1;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> y_face({1, 1}, {nz+1, nx+1});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("apply_local_div_b<YZX>: anti_symmetric_fields", y_face, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::div_b_err);
                });
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                Kokkos::parallel_for("apply_local_div_b<YZX>: pmc_fields", y_face, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::div_b_err);
                });
                break;
            case absorb_fields:
                Kokkos::parallel_for("apply_local_div_b<YZX>: absorb_fields", y_face, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = 0.f;
                });
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template<> void apply_local_div_b<ZXY>(field_array_t* fa, const int i, const int j, const int k) {
    const grid_t* g = fa->g;
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        const int face = (i+j+k)<0 ? 0 : nz+1;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> z_face({1, 1}, {ny+1, nx+1});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("apply_local_div_b<ZXY>: anti_symmetric_fields", z_face, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::div_b_err);
                });
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                Kokkos::parallel_for("apply_local_div_b<ZXY>: pmc_fields", z_face, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::div_b_err);
                });
                break;
            case absorb_fields:
                Kokkos::parallel_for("apply_local_div_b<ZXY>: absorb_fields", z_face, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = 0.f;
                });
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}

void
k_local_ghost_div_b( field_array_t      * ALIGNED(128) fa,
                   const grid_t *              g ) {
    apply_local_div_b<XYZ>(fa, -1,  0,  0);
    apply_local_div_b<YZX>(fa,  0, -1,  0);
    apply_local_div_b<ZXY>(fa,  0,  0, -1);
    apply_local_div_b<XYZ>(fa,  1,  0,  0);
    apply_local_div_b<YZX>(fa,  0,  1,  0);
    apply_local_div_b<ZXY>(fa,  0,  0,  1);
}

void
local_ghost_div_b( field_t      * ALIGNED(128) f,
                   const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int bc, face, x, y, z;

# define APPLY_LOCAL_DIV_B(i,j,k,X,Y,Z)					    \
  do {									    \
    bc = g->bc[BOUNDARY(i,j,k)];					    \
    if( bc<0 || bc>=world_size ) {                                          \
      face = (i+j+k)<0 ? 0 : n##X+1;					    \
      switch(bc) {							    \
      case anti_symmetric_fields:					    \
	X##_FACE_LOOP(face) f(x,y,z).div_b_err =  f(x-i,y-j,z-k).div_b_err; \
	break;								    \
      case symmetric_fields: case pmc_fields:				    \
	X##_FACE_LOOP(face) f(x,y,z).div_b_err = -f(x-i,y-j,z-k).div_b_err; \
	break;								    \
      case absorb_fields:						    \
	X##_FACE_LOOP(face) f(x,y,z).div_b_err = 0;			    \
	break;								    \
      default:								    \
	ERROR(("Bad boundary condition encountered."));	                    \
	break;								    \
      }									    \
    }									    \
  } while(0)
  
  APPLY_LOCAL_DIV_B(-1, 0, 0,x,y,z);
  APPLY_LOCAL_DIV_B( 0,-1, 0,y,z,x);
  APPLY_LOCAL_DIV_B( 0, 0,-1,z,x,y);
  APPLY_LOCAL_DIV_B( 1, 0, 0,x,y,z);
  APPLY_LOCAL_DIV_B( 0, 1, 0,y,z,x);
  APPLY_LOCAL_DIV_B( 0, 0, 1,z,x,y);
}

/*****************************************************************************
 * Local adjusts
 *****************************************************************************/

// FIXME: Specialty edge loops should be added to zero e_tang on local
// edges exclusively to handle concave domain geometries

void
local_adjust_tang_e( field_t      * ALIGNED(128) f,
                     const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int bc, face, x, y, z;
  field_t *fs;

# define ADJUST_TANG_E(i,j,k,X,Y,Z)                                     \
  do {                                                                  \
    bc = g->bc[BOUNDARY(i,j,k)];                                        \
    if( bc<0 || bc>=world_size ) {                                      \
      face = (i+j+k)<0 ? 1 : n##X+1;                                    \
      switch(bc) {                                                      \
      case anti_symmetric_fields:                                       \
	Y##Z##_EDGE_LOOP(face) {                                        \
          fs = &f(x,y,z);                                               \
          fs->e##Y = 0;                                                 \
          fs->tca##Y = 0;                                               \
        }                                                               \
	Z##Y##_EDGE_LOOP(face) {                                        \
          fs = &f(x,y,z);                                               \
          fs->e##Z = 0;                                                 \
          fs->tca##Z = 0;                                               \
        }                                                               \
	break;                                                          \
      case symmetric_fields: case pmc_fields: case absorb_fields:       \
        break;                                                          \
      default:                                                          \
	ERROR(("Bad boundary condition encountered."));                 \
	break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)

  ADJUST_TANG_E(-1, 0, 0,x,y,z);
  ADJUST_TANG_E( 0,-1, 0,y,z,x);
  ADJUST_TANG_E( 0, 0,-1,z,x,y);
  ADJUST_TANG_E( 1, 0, 0,x,y,z);
  ADJUST_TANG_E( 0, 1, 0,y,z,x);
  ADJUST_TANG_E( 0, 0, 1,z,x,y);
}

template<typename T>
void adjust_tang_e(k_field_t& k_field, const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
/*
    int face;
    int sx1,sy1,sz1,ex1,ey1,ez1;
    int sx2,sy2,sz2,ex2,ey2,ez2;
    field_var::f_v eY, tcaY, eZ, tcaZ;
    if(std::is_same<T, XYZ>::value) {
        face = (i+j+k)<0 ? 1 : nx + 1;
        sx1 = face;
        sy1 = 1;
        sz1 = 1;
        ex1 = face+1;
        ey1 = ny+1;
        ez1 = nz+2;
        sx2 = face;
        sy2 = 1;
        sz2 = 1;
        ex2 = face+1;
        ey2 = ny+2;
        ez2 = nz+1;
        eY = field_var::ey;
        tcaY = field_var::tcay;
        eZ = field_var::ez;
        tcaZ = field_var::tcaz;
    } else if (std::is_same<T, YZX>::value) {
        face = (i+j+k)<0 ? 1 : ny + 1;
        sx1 = 1;
        sy1 = face;
        sz1 = 1;
        ex1 = nx+2;
        ey1 = face+1;
        ez1 = nz+1;
        sx2 = 1;
        sy2 = face;
        sz2 = 1;
        ex2 = nx+1;
        ey2 = face+1;
        ez2 = nz+2;
        eY = field_var::ez;
        tcaY = field_var::tcaz;
        eZ = field_var::ex;
        tcaZ = field_var::tcax;
    } else if (std::is_same<T, ZXY>::value) {
        face = (i+j+k)<0 ? 1 : nz + 1;
        sx1 = 1;
        sy1 = 1;
        sz1 = face;
        ex1 = nx+1;
        ey1 = ny+2;
        ez1 = face+1;
        sx2 = 1;
        sy2 = 1;
        sz2 = face;
        ex2 = nx+2;
        ey2 = ny+1;
        ez2 = face+1;
        eY = field_var::ex;
        tcaY = field_var::tcax;
        eZ = field_var::ey;
        tcaZ = field_var::tcay;
    } else {
        ERROR(("Bad template argument"));
    } 
    int bc = g->bc[BOUNDARY(i,j,k)];
    Kokkos::MDRangePolicy<Kokkos::Rank<3> > policy1({sz1,sy1,sx1}, {ez1,ey1,ex1});
    Kokkos::MDRangePolicy<Kokkos::Rank<3> > policy2({sz2,sy2,sx2}, {ez2,ey2,ex2});
    if(bc < 0 || bc >= world_size) {
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("adjust_tang_e: edge loop", policy1, KOKKOS_LAMBDA(const int z, const int y, const int x) {
                    k_field(VOXEL(x,y,z,nx,ny,nz), eY) = 0;
                    k_field(VOXEL(x,y,z,nx,ny,nz), tcaY) = 0;
                });
                Kokkos::parallel_for("adjust_tang_e: edge loop", policy2, KOKKOS_LAMBDA(const int z, const int y, const int x) {
                    k_field(VOXEL(x,y,z,nx,ny,nz), eZ) = 0;
                    k_field(VOXEL(x,y,z,nx,ny,nz), tcaZ) = 0;
                });
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            case absorb_fields:
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
*/
}

template<> void adjust_tang_e<XYZ>(k_field_t& k_field, const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        int face = (i+j+k) < 0 ? 1 : nx + 1;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yz_policy({1, 1}, {nz+2, ny+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_policy({1, 1}, {nz+1, ny+2});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("adjust_tang_e<XYZ>: YZ Edge Loop", yz_policy, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = 0;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay) = 0;
                });
/*
                Kokkos::parallel_for("adjust_tang_e<XYZ>: YZ Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (const size_t yi) {
                        const size_t x = face;
                        const size_t y = yi + 1;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = 0;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay) = 0;
                    });
                });
*/
                Kokkos::parallel_for("adjust_tang_e<XYZ>: ZY Edge Loop", zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez) = 0;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz) = 0;
                });
/*
                Kokkos::parallel_for("adjust_tang_e<XYZ>: ZY Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const size_t yi) {
                        const size_t x = face;
                        const size_t y = yi + 1;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez) = 0;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz) = 0;
                    });
                });
*/
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            case absorb_fields:
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template<> void adjust_tang_e<YZX>(k_field_t& k_field, const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        int face = (i+j+k) < 0 ? 1 : ny + 1;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_policy({1, 1}, {nz+1, nx+2});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_policy({1, 1}, {nz+2, nx+1});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("adjust_tang_e<YZX>: ZX Edge Loop", zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez) = 0;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz) = 0;
                });
/*
                Kokkos::parallel_for("adjust_tang_e<YZX>: ZX Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = face;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez) = 0;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz) = 0;
                    });
                });
*/
                Kokkos::parallel_for("adjust_tang_e<YZX>: XZ Edge Loop", xz_policy, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = 0;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax) = 0;
                });
/*
                Kokkos::parallel_for("adjust_tang_e<YZX>: XZ Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (const size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = face;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = 0;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax) = 0;
                    });
                });
*/
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            case absorb_fields:
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template<> void adjust_tang_e<ZXY>(k_field_t& k_field, const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        int face = (i+j+k) < 0 ? 1 : nz + 1;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xy_policy({1, 1}, {ny+2, nx+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_policy({1, 1}, {ny+1, nx+2});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("adjust_tang_e<ZXY>: XY Edge Loop", xy_policy, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = 0;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax) = 0;
                });
/*
                Kokkos::parallel_for("adjust_tang_e<ZXY>: XY Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (const size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = yi + 1;
                        const size_t z = face;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = 0;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax) = 0;
                    });
                });
*/
                Kokkos::parallel_for("adjust_tang_e<ZXY>: YX Edge Loop", yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = 0;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay) = 0;
                });
/*
                Kokkos::parallel_for("adjust_tang_e<ZXY>: YX Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(ny, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = yi + 1;
                        const size_t z = face;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = 0;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay) = 0;
                    });
                });
*/
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            case absorb_fields:
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}

/*
# define ADJUST_TANG_E(i,j,k,X,Y,Z)                                     \
  do {                                                                  \
    bc = g->bc[BOUNDARY(i,j,k)];                                        \
    if( bc<0 || bc>=world_size ) {                                      \
      face = (i+j+k)<0 ? 1 : n##X+1;                                    \
      switch(bc) {                                                      \
      case anti_symmetric_fields:                                       \
	Y##Z##_EDGE_LOOP(face) {                                        \
          fs = &f(x,y,z);                                               \
          fs->e##Y = 0;                                                 \
          fs->tca##Y = 0;                                               \
        }                                                               \
	Z##Y##_EDGE_LOOP(face) {                                        \
          fs = &f(x,y,z);                                               \
          fs->e##Z = 0;                                                 \
          fs->tca##Z = 0;                                               \
        }                                                               \
	break;                                                          \
      case symmetric_fields: case pmc_fields: case absorb_fields:       \
        break;                                                          \
      default:                                                          \
	ERROR(("Bad boundary condition encountered."));                 \
	break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)
*/
void
k_local_adjust_tang_e( field_array_t      * RESTRICT f,
                     const grid_t *              g ) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;

    k_field_t& k_field = f->k_f_d;
    adjust_tang_e<XYZ>(k_field, g, -1, 0, 0, nx, ny, nz);
    adjust_tang_e<YZX>(k_field, g, 0, -1, 0, nx, ny, nz);
    adjust_tang_e<ZXY>(k_field, g, 0, 0, -1, nx, ny, nz);
    adjust_tang_e<XYZ>(k_field, g, 1, 0, 0, nx, ny, nz);
    adjust_tang_e<YZX>(k_field, g, 0, 1, 0, nx, ny, nz);
    adjust_tang_e<ZXY>(k_field, g, 0, 0, 1, nx, ny, nz);
/*
  ADJUST_TANG_E(-1, 0, 0,x,y,z);
  ADJUST_TANG_E( 0,-1, 0,y,z,x);
  ADJUST_TANG_E( 0, 0,-1,z,x,y);
  ADJUST_TANG_E( 1, 0, 0,x,y,z);
  ADJUST_TANG_E( 0, 1, 0,y,z,x);
  ADJUST_TANG_E( 0, 0, 1,z,x,y);
*/
}

void
k_local_adjust_norm_b( field_array_t * RESTRICT fa,
                     const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int bc, face, x, y, z, xl, xh, yl, yh, zl, zh;

// TODO: Test the macro unrolling and parallel_for here. This does not
// get touched during a normal harris run
# define K_ADJUST_NORM_B(i,j,k,X,Y,Z)                                   \
  do {                                                                  \
    bc = g->bc[BOUNDARY(i,j,k)];                                        \
    if( bc<0 || bc>=world_size ) {                                      \
      face = (i+j+k)<0 ? 1 : n##X+1;                                    \
      switch(bc) {                                                      \
      case anti_symmetric_fields: case pmc_fields: case absorb_fields:  \
        break;                                                          \
      case symmetric_fields:                                            \
         switch(X) {                                                    \
           case('x'):                                                   \
             xl=face, xh=face, yl=1, yh=ny, zl=1, zh=nz;                \
             break;                                                     \
           case('y'):                                                   \
             xl=1, xh=nx, yl=face, yh=face, zl=1, zh=nz;                \
             break;                                                     \
           case('z'):                                                   \
             xl=1, xh=nx, yl=1, yh=ny, zl=face, zh=face;                \
             break;                                                     \
           default:                                                     \
             ERROR(("Bad boundary condition encountered."));            \
             break;                                                     \
         }                                                              \
         assert(0);                                                     \
         Kokkos::parallel_for(Kokkos::RangePolicy                       \
                 < Kokkos::DefaultExecutionSpace >(zl, zh), KOKKOS_LAMBDA (int z) {      \
           for(int yi=yl; yi<=yh; yi++ ) {			                    \
             for(int xj=xl; xj<=xh; xj++ ) {                            \
	             (fa->k_f_h)(VOXEL(xj,yi,z, nx,ny,nz), field_var::cb##X) = 0;     \
             }                                                          \
           }                                                            \
         });                                                            \
        break;                                                          \
      default:                                                          \
        ERROR(("Bad boundary condition encountered."));                 \
        break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)

  K_ADJUST_NORM_B(-1, 0, 0,x,y,z);
  K_ADJUST_NORM_B( 0,-1, 0,y,z,x);
  K_ADJUST_NORM_B( 0, 0,-1,z,x,y);
  K_ADJUST_NORM_B( 1, 0, 0,x,y,z);
  K_ADJUST_NORM_B( 0, 1, 0,y,z,x);
  K_ADJUST_NORM_B( 0, 0, 1,z,x,y);
}

void
local_adjust_norm_b( field_t      * ALIGNED(128) f,
                     const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int bc, face, x, y, z;

# define ADJUST_NORM_B(i,j,k,X,Y,Z)                                     \
  do {                                                                  \
    bc = g->bc[BOUNDARY(i,j,k)];                                        \
    if( bc<0 || bc>=world_size ) {                                      \
      face = (i+j+k)<0 ? 1 : n##X+1;                                    \
      switch(bc) {                                                      \
      case anti_symmetric_fields: case pmc_fields: case absorb_fields:  \
	break;                                                          \
      case symmetric_fields:                                            \
	X##_FACE_LOOP(face) f(x,y,z).cb##X = 0;                         \
	break;                                                          \
      default:                                                          \
	ERROR(("Bad boundary condition encountered."));                 \
	break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)

  ADJUST_NORM_B(-1, 0, 0,x,y,z);
  ADJUST_NORM_B( 0,-1, 0,y,z,x);
  ADJUST_NORM_B( 0, 0,-1,z,x,y);
  ADJUST_NORM_B( 1, 0, 0,x,y,z);
  ADJUST_NORM_B( 0, 1, 0,y,z,x);
  ADJUST_NORM_B( 0, 0, 1,z,x,y);
}

template <typename T> void adjust_div_e_err(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {}
template <> void adjust_div_e_err<XYZ>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        const int nx = g->nx, ny = g->ny, nz = g->nz;
        int face = (i+j+k)<0 ? 1 : nx+1;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_policy({1,1}, {nz+2, ny+2});
        switch(bc) {
            case anti_symmetric_fields:
//                break;
            case absorb_fields:
                Kokkos::parallel_for("adjust_div_e_err<XYZ>", zy_policy, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_e_err) = 0;
                }); 
/*
                Kokkos::parallel_for("adjust_div_e_err<XYZ>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    const int zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const int yi) {
                        const int x = face;
                        const int y = yi + 1;
                        const int z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_e_err) = 0;
                    });
                });
*/
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template <> void adjust_div_e_err<YZX>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        const int nx = g->nx, ny = g->ny, nz = g->nz;
        int face = (i+j+k)<0 ? 1 : ny+1;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_policy({1,1}, {nz+2, nx+2});
        switch(bc) {
            case anti_symmetric_fields:
//                break;
            case absorb_fields:
                Kokkos::parallel_for("adjust_div_e_err<YZX>", zx_policy, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_e_err) = 0;
                }); 
/*
                Kokkos::parallel_for("adjust_div_e_err<YZX>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    const int zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                        const int x = xi + 1;
                        const int y = face;
                        const int z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_e_err) = 0;
                    });
                });
*/
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template <> void adjust_div_e_err<ZXY>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        const int nx = g->nx, ny = g->ny, nz = g->nz;
        int face = (i+j+k)<0 ? 1 : nz+1;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_policy({1,1}, {ny+2, nx+2});
        switch(bc) {
            case anti_symmetric_fields:
//                break;
            case absorb_fields:
                Kokkos::parallel_for("adjust_div_e_err<ZXY>", yx_policy, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_e_err) = 0;
                }); 
/*
                Kokkos::parallel_for("adjust_div_e_err<ZXY>", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
                    const int yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                        const int x = xi + 1;
                        const int y = yi + 1;
                        const int z = face;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_e_err) = 0;
                    });
                });
*/
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}

void
k_local_adjust_div_e( field_array_t      * ALIGNED(128) f,
                    const grid_t *              g ) {
    adjust_div_e_err<XYZ>(f, g, -1,  0,  0);
    adjust_div_e_err<YZX>(f, g,  0, -1,  0);
    adjust_div_e_err<ZXY>(f, g,  0,  0, -1);
    adjust_div_e_err<XYZ>(f, g, 1, 0, 0);
    adjust_div_e_err<YZX>(f, g, 0, 1, 0);
    adjust_div_e_err<ZXY>(f, g, 0, 0, 1);
}

void
local_adjust_div_e( field_t      * ALIGNED(128) f,
                    const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int bc, face, x, y, z;

# define ADJUST_DIV_E_ERR(i,j,k,X,Y,Z)			         \
  do {							         \
    bc = g->bc[BOUNDARY(i,j,k)];				 \
    if( bc<0 || bc>=world_size ) {				 \
      face = (i+j+k)<0 ? 1 : n##X+1;				 \
      switch(bc) {						 \
      case anti_symmetric_fields: case absorb_fields:		 \
        X##_NODE_LOOP(face) f(x,y,z).div_e_err = 0;              \
        break;                                                   \
      case symmetric_fields: case pmc_fields:			 \
        break;                                                   \
      default:							 \
	ERROR(("Bad boundary condition encountered."));          \
	break;							 \
      }								 \
    }								 \
  } while(0)

  ADJUST_DIV_E_ERR(-1, 0, 0,x,y,z);
  ADJUST_DIV_E_ERR( 0,-1, 0,y,z,x);
  ADJUST_DIV_E_ERR( 0, 0,-1,z,x,y);
  ADJUST_DIV_E_ERR( 1, 0, 0,x,y,z);
  ADJUST_DIV_E_ERR( 0, 1, 0,y,z,x);
  ADJUST_DIV_E_ERR( 0, 0, 1,z,x,y);
}

// anti_symmetric => Opposite sign image charges (zero jf_tang)
// symmetric      => Same sign image charges (double jf_tang) 
// absorbing      => No image charges, half cell accumulation (double jf_tang)
// (rhob/jf_norm account for particles that hit boundary and reflect/stick)

void
local_adjust_jf( field_t      * ALIGNED(128) f,
                 const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int bc, face, x, y, z;

# define ADJUST_JF(i,j,k,X,Y,Z)                                         \
  do {                                                                  \
    bc = g->bc[BOUNDARY(i,j,k)];                                        \
    if( bc<0 || bc>=world_size ) {                                      \
      face = (i+j+k)<0 ? 1 : n##X+1;                                    \
      switch(bc) {                                                      \
      case anti_symmetric_fields:                                       \
	Y##Z##_EDGE_LOOP(face) f(x,y,z).jf##Y = 0;                      \
        Z##Y##_EDGE_LOOP(face) f(x,y,z).jf##Z = 0;                      \
	break;                                                          \
      case symmetric_fields: case pmc_fields: case absorb_fields:       \
	Y##Z##_EDGE_LOOP(face) f(x,y,z).jf##Y *= 2.;                    \
        Z##Y##_EDGE_LOOP(face) f(x,y,z).jf##Z *= 2.;                    \
	break;                                                          \
      default:                                                          \
	ERROR(("Bad boundary condition encountered."));                 \
	break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)
  
  ADJUST_JF(-1, 0, 0,x,y,z);
  ADJUST_JF( 0,-1, 0,y,z,x);
  ADJUST_JF( 0, 0,-1,z,x,y);
  ADJUST_JF( 1, 0, 0,x,y,z);
  ADJUST_JF( 0, 1, 0,y,z,x);
  ADJUST_JF( 0, 0, 1,z,x,y);
}

template<typename T> void adjust_jf(field_array_t* fa, const grid_t* g, int i, int j, int k) {
/*
    int face;
    int sx, sy, sz, ex1, ex2, ey1, ey2, ez1, ez2; // start and end indices
    int nx = g->nx, ny = g->ny, nz = g->nz;
    field_var::f_v jfY, jfZ;
    if(std::is_same<T, XYZ>::value) {
        face = (i+j+k)<0 ? 1 : nx+1;
        jfY = field_var::jfy;
        jfZ = field_var::jfz;
        sx = face, sy = 1, sz = 1;
        ex1 = face+1, ey1 = ny+1, ez1 = nz+2;
        ex2 = face+1, ey2 = ny+2, ez2 = nz+1;
    } else if(std::is_same<T, YZX>::value) {
        face = (i+j+k)<0 ? 1 : ny+1;
        jfY = field_var::jfz;
        jfZ = field_var::jfx;
        sx = 1, sy = face, sz = 1;
        ex1 = nx+2, ey1 = face+1, ez1 = nz+1;
        ex2 = nx+1, ey2 = face+1, ez2 = nz+2;
    } else if(std::is_same<T, ZXY>::value) {
        face = (i+j+k)<0 ? 1 : nz+1;
        jfY = field_var::jfx;
        jfZ = field_var::jfy;
        sx = 1, sy = 1, sz = face;
        ex1 = nx+1, ey1 = ny+2, ez1 = face+1;
        ex2 = nx+2, ey2 = ny+1, ez2 = face+1;
    }
    int bc = g->bc[BOUNDARY(i,j,k)];
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy1({sz, sy, sx}, {ez1, ey1, ex1});
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy2({sz, sy, sx}, {ez2, ey2, ex2});
    k_field_t& k_field = fa->k_f_d;
    
    if(bc < 0 || bc >= world_size) {
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("adjust_jf: edge loop 1", policy1, KOKKOS_LAMBDA(const int kk, const int jj, const int ii) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), jfY) = 0;
                });
                Kokkos::parallel_for("adjust_jf: edge loop 2", policy2, KOKKOS_LAMBDA(const int kk, const int jj, const int ii) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), jfZ) = 0;
                });
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            case absorb_fields:
                Kokkos::parallel_for("adjust_jf: edge loop 1", policy1, KOKKOS_LAMBDA(const int kk, const int jj, const int ii) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), jfY) *= 2;
                });
                Kokkos::parallel_for("adjust_jf: edge loop 2", policy2, KOKKOS_LAMBDA(const int kk, const int jj, const int ii) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), jfZ) *= 2;
                });
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
*/
}
template<> void adjust_jf<XYZ>(field_array_t* fa, const grid_t* g, int i, int j, int k) {
    const int bc = g->bc[BOUNDARY(i,j,k)];
    if( bc < 0 || bc >= world_size ) {
        const k_field_t& k_field = fa->k_f_d;
        const int nx = g->nx, ny = g->ny, nz = g->nz;
        const int face = (i+j+k)<0 ? 1 : nx+1;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yz_edge({1, 1}, {nz+2, ny+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_edge({1, 1}, {nz+1, ny+2});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("adjust_jf<XYZ>: yz_edge_loop", yz_edge, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy) = 0;
                });
                Kokkos::parallel_for("adjust_jf<XYZ>: zy_edge_loop", zy_edge, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz) = 0;
                });
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            case absorb_fields:
                Kokkos::parallel_for("adjust_jf<XYZ>: yz_edge_loop", yz_edge, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy) * 2.;
                });
                Kokkos::parallel_for("adjust_jf<XYZ>: zy_edge_loop", zy_edge, KOKKOS_LAMBDA(const int z, const int y) {
                    const int x = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz) * 2.;
                });
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template<> void adjust_jf<YZX>(field_array_t* fa, const grid_t* g, int i, int j, int k) {
    const int bc = g->bc[BOUNDARY(i,j,k)];
    if( bc < 0 || bc >= world_size ) {
        const k_field_t& k_field = fa->k_f_d;
        const int nx = g->nx, ny = g->ny, nz = g->nz;
        const int face = (i+j+k)<0 ? 1 : ny+1;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_edge({1, 1}, {nz+1, nx+2});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_edge({1, 1}, {nz+2, nx+1});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("adjust_jf<YZX>: zx_edge_loop", zx_edge, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz) = 0;
                });
                Kokkos::parallel_for("adjust_jf<YZX>: xz_edge_loop", xz_edge, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx) = 0;
                });
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            case absorb_fields:
                Kokkos::parallel_for("adjust_jf<YZX>: zx_edge_loop", zx_edge, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz) * 2.;
                });
                Kokkos::parallel_for("adjust_jf<YZX>: xz_edge_loop", xz_edge, KOKKOS_LAMBDA(const int z, const int x) {
                    const int y = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx) * 2.;
                });
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}
template<> void adjust_jf<ZXY>(field_array_t* fa, const grid_t* g, int i, int j, int k) {
    const int bc = g->bc[BOUNDARY(i,j,k)];
    if( bc < 0 || bc >= world_size ) {
        const k_field_t& k_field = fa->k_f_d;
        const int nx = g->nx, ny = g->ny, nz = g->nz;
        const int face = (i+j+k)<0 ? 1 : nz+1;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xy_edge({1, 1}, {ny+2, nx+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_edge({1, 1}, {ny+1, nx+2});
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("adjust_jf<ZXY>: yz_edge_loop", xy_edge, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx) = 0;
                });
                Kokkos::parallel_for("adjust_jf<ZXY>: zy_edge_loop", yx_edge, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy) = 0;
                });
                break;
            case symmetric_fields:
                break;
            case pmc_fields:
                break;
            case absorb_fields:
                Kokkos::parallel_for("adjust_jf<ZXY>: yz_edge_loop", xy_edge, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx) * 2.;
                });
                Kokkos::parallel_for("adjust_jf<ZXY>: zy_edge_loop", yx_edge, KOKKOS_LAMBDA(const int y, const int x) {
                    const int z = face;
                    k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy) * 2.;
                });
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}

void k_local_adjust_jf(field_array_t* fa, const grid_t* g) {
    adjust_jf<XYZ>(fa, g, -1,  0,  0);
    adjust_jf<YZX>(fa, g,  0, -1,  0);
    adjust_jf<ZXY>(fa, g,  0,  0, -1);
    adjust_jf<XYZ>(fa, g, 1, 0, 0);
    adjust_jf<YZX>(fa, g, 0, 1, 0);
    adjust_jf<ZXY>(fa, g, 0, 0, 1);
}

// anti_symmetric => Opposite sign image charges (zero rhof/rhob)
// symmetric      => Same sign image charges (double rhof)
//                => (double rhof, rhob is already correct)
// absorbing      => No image charges, half cell accumulation (double rhof)
// (rhob/jf_norm account for particles that hit the boundary)

void
local_adjust_rhof( field_t      * ALIGNED(128) f,
                   const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int bc, face, x, y, z;

# define ADJUST_RHOF(i,j,k,X,Y,Z)                                       \
  do {                                                                  \
    bc = g->bc[BOUNDARY(i,j,k)];                                        \
    if( bc<0 || bc>=world_size ) {                                      \
      face = (i+j+k)<0 ? 1 : n##X+1;                                    \
      switch(bc) {                                                      \
      case anti_symmetric_fields:                                       \
	X##_NODE_LOOP(face) f(x,y,z).rhof = 0;                          \
	break;                                                          \
      case symmetric_fields: case pmc_fields: case absorb_fields:       \
	X##_NODE_LOOP(face) f(x,y,z).rhof *= 2;                         \
        break;                                                          \
      default:                                                          \
	ERROR(("Bad boundary condition encountered."));                 \
	break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)
  
  ADJUST_RHOF(-1, 0, 0,x,y,z);
  ADJUST_RHOF( 0,-1, 0,y,z,x);
  ADJUST_RHOF( 0, 0,-1,z,x,y);
  ADJUST_RHOF( 1, 0, 0,x,y,z);
  ADJUST_RHOF( 0, 1, 0,y,z,x);
  ADJUST_RHOF( 0, 0, 1,z,x,y);
}

template<typename T> void adjust_rhof(field_array_t* fa, const grid_t* g, int i, int j, int k) {

    int nX, face, bc;
    int startx, starty, startz;
    int endx, endy, endz;
    int nx = g->nx, ny = g->ny, nz = g->nz;
    std::string af_name, as_name;
    if(std::is_same<T, XYZ>::value) {
        nX = nx;
        face = (i+j+k)<0 ? 1 : nX+1;
        startx = face;
        starty = 1;
        startz = 1;
        endx = face+1;
        endy = ny+2;
        endz = nz+2;
        as_name = "adjust_rhof<XYZ>: anti_symmetric_fields: node loop";
        af_name = "adjust_rhof<XYZ>: absorb_fields: node loop";
    } else if (std::is_same<T, YZX>::value) {
        nX = ny;
        face = (i+j+k)<0 ? 1 : nX+1;
        startx = 1;
        starty = face;
        startz = 1;
        endx = nx+2;
        endy = face+1;
        endz = nz+2;
        as_name = "adjust_rhof<YZX>: anti_symmetric_fields: node loop";
        af_name = "adjust_rhof<YZX>: absorb_fields: node loop";
    } else if (std::is_same<T, ZXY>::value) {
        nX = nz;
        face = (i+j+k)<0 ? 1 : nX+1;
        startx = 1;
        starty = 1;
        startz = face;
        endx = nx+2;
        endy = ny+2;
        endz = face+1;
        as_name = "adjust_rhof<ZXY>: anti_symmetric_fields: node loop";
        af_name = "adjust_rhof<ZXY>: absorb_fields: node loop";
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<3> > node_policy({startz, starty, startx}, {endz, endy, endx});
    k_field_t& k_field = fa->k_f_d;

    bc = g->bc[BOUNDARY(i,j,k)];
    if( bc < 0 || bc >= world_size ) {
        face = (i+j+k)<0 ? 1 : nX + 1;
        switch(bc) {
            case anti_symmetric_fields:
                parallel_for(as_name, node_policy, KOKKOS_LAMBDA(const int kk, const int jj, const int ii) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::rhof) = 0;
                });
                break;
            case symmetric_fields:
            case pmc_fields:
            case absorb_fields:
                parallel_for(af_name, node_policy, KOKKOS_LAMBDA(const int kk, const int jj, const int ii) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::rhof) *= 2;
                });
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }

}

void k_local_adjust_rhof(field_array_t* fa, const grid_t* g) {
    adjust_rhof<XYZ>(fa, g, -1, 0, 0);
    adjust_rhof<YZX>(fa, g, 0, -1, 0);
    adjust_rhof<ZXY>(fa, g, 0, 0, -1);
    adjust_rhof<XYZ>(fa, g, 1, 0, 0);
    adjust_rhof<YZX>(fa, g, 0, 1, 0);
    adjust_rhof<ZXY>(fa, g, 0, 0, 1);
}

// anti_symmetric => Opposite sign image charges (zero rhob)
// symmetric      => Same sign image charges (rhob already correct)
// absorbing      => No image charges, half cell accumulation (rhob already
//                   correct)

void
local_adjust_rhob( field_t      * ALIGNED(128) f,
                   const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int bc, face, x, y, z;

# define ADJUST_RHOB(i,j,k,X,Y,Z)                                       \
  do {                                                                  \
    bc = g->bc[BOUNDARY(i,j,k)];                                        \
    if( bc<0 || bc>=world_size ) {                                      \
      face = (i+j+k)<0 ? 1 : n##X+1;                                    \
      switch(bc) {                                                      \
      case anti_symmetric_fields:                                       \
	X##_NODE_LOOP(face) f(x,y,z).rhob = 0;                          \
	break;                                                          \
      case symmetric_fields: case pmc_fields: case absorb_fields:       \
        break;                                                          \
      default:                                                          \
	ERROR(("Bad boundary condition encountered."));                 \
	break;                                                          \
      }                                                                 \
    }                                                                   \
  } while(0)
  
  ADJUST_RHOB(-1, 0, 0,x,y,z);
  ADJUST_RHOB( 0,-1, 0,y,z,x);
  ADJUST_RHOB( 0, 0,-1,z,x,y);
  ADJUST_RHOB( 1, 0, 0,x,y,z);
  ADJUST_RHOB( 0, 1, 0,y,z,x);
  ADJUST_RHOB( 0, 0, 1,z,x,y);
}

template<typename T> void adjust_rhob(field_array_t* fa, const grid_t* g, int i, int j, int k) {
    int nX, face, bc;
    int nx = g->nx, ny = g->ny, nz = g->nz;
    int startx, starty, startz;
    int endx, endy, endz;
    if(std::is_same<T, XYZ>::value) {
        nX = nx;
        face = (i+j+k)<0 ? 1 : nX+1;
        startx = face;
        starty = 1;
        startz = 1;
        endx = face+1;
        endy = ny+2;
        endz = nz+2;
    } else if (std::is_same<T, YZX>::value) {
        nX = ny;
        face = (i+j+k)<0 ? 1 : nX+1;
        startx = 1;
        starty = face;
        startz = 1;
        endx = nx+2;
        endy = face+1;
        endz = nz+2;
    } else if (std::is_same<T, ZXY>::value) {
        nX = nz;
        face = (i+j+k)<0 ? 1 : nX+1;
        startx = 1;
        starty = 1;
        startz = face;
        endx = nx+2;
        endy = ny+2;
        endz = face+1;
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<3> > node_policy({startz, starty, startx}, {endz, endy, endx});
    k_field_t& k_field = fa->k_f_d;

    bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        switch(bc) {
            case anti_symmetric_fields:
                Kokkos::parallel_for("adjust_rhob: anti_symmetric_fields: node loop", node_policy, KOKKOS_LAMBDA(const int kk, const int jj, const int ii) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::rhof) = 0;
                });
                break;
            case symmetric_fields:
            case pmc_fields:
            case absorb_fields:
                break;
            default:
                ERROR(("Bad boundary condition encountered."));
                break;
        }
    }
}

void k_local_adjust_rhob(field_array_t* fa, const grid_t* g) {
    adjust_rhob<XYZ>(fa, g, -1, 0, 0);
    adjust_rhob<YZX>(fa, g, 0, -1, 0);
    adjust_rhob<ZXY>(fa, g, 0, 0, -1);
    adjust_rhob<XYZ>(fa, g, 1, 0, 0);
    adjust_rhob<YZX>(fa, g, 0, 1, 0);
    adjust_rhob<ZXY>(fa, g, 0, 0, 1);
}

