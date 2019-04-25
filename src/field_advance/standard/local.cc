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

/*
template<typename T> void edge_loop(const size_t nx, const size_t ny, const size_t nz, const size_t w, 
                                std::function<void(const size_t, const size_t, const size_t)> lambda) {}

template<> void edge_loop<XY>(const size_t nx, const size_t ny, const size_t nz, const size_t z, 
                                std::function<void(size_t, size_t, size_t)> lambda) {
    Kokkos::parallel_for("XY Edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1,Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
        size_t y = team_member.league_rank() + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
            lambda(xi+1,y,z);
        });
    });
}
template<> void edge_loop<YX>(const size_t nx, const size_t ny, const size_t nz, const size_t z, 
                                std::function<void(size_t, size_t, size_t)> lambda) {
    Kokkos::parallel_for("YX Edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny,Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
        size_t y = team_member.league_rank() + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
            lambda(xi+1,y,z);
        });
    });
}
template<> void edge_loop<YZ>(const size_t nx, const size_t ny, const size_t nz, const size_t x, 
                                std::function<void(size_t, size_t, size_t)> lambda) {
    Kokkos::parallel_for("YZ Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
        size_t z = team_member.league_rank() + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (size_t yi) {
            lambda(x,yi+1,z);
        });
    });
}
template<> void edge_loop<ZY>(const size_t nx, const size_t ny, const size_t nz, const size_t x, 
                                std::function<void(size_t, size_t, size_t)> lambda) {
    const size_t ghost = x;
    Kokkos::parallel_for("ZY Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,ny+1),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
//        size_t z = team_member.league_rank() + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (size_t& yi) {
            size_t z = team_member.league_rank() + 1;
            const size_t y = yi+1;
            lambda(ghost,y,z);
        });
    });
}
template<> void edge_loop<ZX>(const size_t nx, const size_t ny, const size_t nz, const size_t y, 
                                std::function<void(size_t, size_t, size_t)> lambda) {
    Kokkos::parallel_for("ZX Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
        size_t z = team_member.league_rank() + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
            lambda(xi+1,y,z);
        });
    });
}
template<> void edge_loop<XZ>(const size_t nx, const size_t ny, const size_t nz, const size_t y, 
                                std::function<void(size_t, size_t, size_t)> lambda) {
    Kokkos::parallel_for("XZ Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
        size_t z = team_member.league_rank() + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
            lambda(xi+1,y,z);
        });
    });
}
*/

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
//    Kokkos::MDRangePolicy<Kokkos::Rank<2> > zy_policy({1,1},{nz,ny+1});
//    Kokkos::MDRangePolicy<Kokkos::Rank<2> > yz_policy({1,1},{nz+1,ny});

    if(bc < 0 || bc >= world_size) {
        int ghost = (i+j+k)<0 ? 0 : nx+1;
        int face = (i+j+k)<0 ? 1 : nx+1;
        switch(bc) {
            case anti_symmetric_fields:

                Kokkos::parallel_for("apply_local_tang_b<XYZ>: anti_symmetric_fields: ZY edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (size_t yi) {
                        const size_t x = ghost;
                        const size_t y = yi + 1;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cby);
                    });
                });
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: anti_symmetric_fields: YZ edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (size_t yi) {
                        const size_t x = ghost;
                        const size_t y = yi + 1;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbz);
                    });
                });
/*
                Kokkos::parallel_for("ZY Edge loop", zy_policy, KOKKOS_LAMBDA(const int ii, const int jj) {
                    k_field(VOXEL(ghost,jj,ii,nx,ny,nz), field_var::cby) = k_field(VOXEL(ghost-i,jj-j,ii-k,nx,ny,nz), field_var::cby);
                });

                Kokkos::parallel_for("YZ Edge loop", yz_policy, KOKKOS_LAMBDA(const int ii, const int jj) {
                    k_field(VOXEL(ghost,jj,ii,nx,ny,nz),field_var::cbz) = k_field(VOXEL(ghost-i,jj-j,ii-k,nx,ny,nz), field_var::cbz);
                });
*/
                break;
            case symmetric_fields:
            case pmc_fields:

                Kokkos::parallel_for("apply_local_tang_b<XYZ>: pmc_fields: ZY edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (size_t yi) {
                        const size_t x = ghost;
                        const size_t y = yi + 1;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cby);
                    });
                });
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: pmc_fields: YZ edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (size_t yi) {
                        const size_t x = ghost;
                        const size_t y = yi + 1;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbz);
                    });
                });
/*
                Kokkos::parallel_for("ZY Edge loop", zy_policy, KOKKOS_LAMBDA(const int ii, const int jj) {
                    k_field(VOXEL(ghost,jj,ii,nx,ny,nz), field_var::cby) = -k_field(VOXEL(ghost-i,jj-j,ii-k,nx,ny,nz), field_var::cby);
                });

                Kokkos::parallel_for("YZ Edge loop", yz_policy, KOKKOS_LAMBDA(const int ii, const int jj) {
                    k_field(VOXEL(ghost,jj,ii,nx,ny,nz), field_var::cbz) = -k_field(VOXEL(ghost-i,jj-j,ii-k,nx,ny,nz), field_var::cbz);
                });
*/
                break;
            case absorb_fields:
                drive = cdt_dx*higend;
                decay = (1-drive)/(1+drive);
                drive = 2*drive/(1+drive);
Kokkos::parallel_for("single test", KOKKOS_TEAM_POLICY_DEVICE(1,1),
KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
    for(int z=1; z<=nz; z++) {
        for(int y=1; z<=ny+1; y++) {
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
        for(int y=1; z<=ny; y++) {
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
                t2 = cdt_dz * (t2 - k_field(fh, field_var::ex));
                k_field(fg, field_var::cbz) = decay*k_field(fg, field_var::cbz) + drive * k_field(fh, field_var::cbz) - t1 + t2;
            }
        }
    }
});
/*
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: absorb_fields: ZY edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (size_t yi) {
                        const size_t x = ghost;
                        const size_t y = yi + 1;
                        const size_t z = zi + 1;
                        const int fg = VOXEL(x,y,z,nx,ny,nz);
                        const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        float t1 = cdt_dx*(k_field(VOXEL(face-i,y-j,z-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(face,y,z,nx,ny,nz), field_var::ez));
                        t1 = (i+j+k)<0 ? t1 : -t1;
                        float t2 = k_field(VOXEL(x-i,y-j,z+1-k,nx,ny,nz), field_var::ex);
                        t2 = cdt_dz*(t2 - k_field(fh, field_var::ex));
                        k_field(fg, field_var::cby) = decay*k_field(fg, field_var::cby) + drive*k_field(fh, field_var::cby) - t1 + t2;
                    });
                });
                Kokkos::parallel_for("apply_local_tang_b<XYZ>: absorb_fields: YZ edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (size_t yi) {
                        const size_t x = ghost;
                        const size_t y = yi + 1;
                        const size_t z = zi + 1;
                        const int fg = VOXEL(x,y,z,nx,ny,nz);
                        const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        float t1 = cdt_dx*(k_field(VOXEL(face-i,y-j,z-k,nx,ny,nz), field_var::ey) - k_field(VOXEL(face,y,z,nx,ny,nz), field_var::ey));
                        t1 = (i+j+k)<0 ? t1 : -t1;
                        float t2 = k_field(VOXEL(x-i,y+1-j,z-k,nx,ny,nz), field_var::ex);
                        t2 = cdt_dy*(t2 - k_field(fh, field_var::ex));
                        k_field(fg, field_var::cbz) = decay*k_field(fg, field_var::cbz) + drive*k_field(fh, field_var::cbz) - t1 + t2;
                    });
                });
*/
/*
                Kokkos::parallel_for("ZY Edge loop", zy_policy, KOKKOS_LAMBDA(const int kk, const int jj) {
                    int fg = VOXEL(ghost,jj,kk,nx,ny,nz);
                    int fh = VOXEL(ghost-i,jj-j,kk-k,nx,ny,nz);
                    int xx = face;
                    float t1 = cdt_dx*( k_field(VOXEL(xx-i,jj-j,kk-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(xx,jj,kk,nx,ny,nz), field_var::ez));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    int zz = kk+1;
                    float t2 = k_field(VOXEL(ghost-i,jj-j,zz-k,nx,ny,nz), field_var::ex);
                    t2 = cdt_dz*(t2 - k_field(fh, field_var::ex));
                    k_field(fg, field_var::cby) = decay*k_field(fg, field_var::cby) + drive*k_field(fh, field_var::cby) - t1 + t2;
                });
                Kokkos::parallel_for("YZ Edge loop", zy_policy, KOKKOS_LAMBDA(const int kk, const int jj) {
                    int fg = VOXEL(ghost,jj,kk,nx,ny,nz);
                    int fh = VOXEL(ghost-i,jj-j,kk-k,nx,ny,nz);
                    int xx = face;
                    float t1 = cdt_dx*( k_field(VOXEL(xx-i,jj-j,kk-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(xx,jj,kk,nx,ny,nz), field_var::ez));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    int yy = y+1;
                    float t2 = k_field(VOXEL(ghost-i,yy-j,kk-k,nx,ny,nz), field_var::ex);
                    t2 = cdt_dy*(t2 - k_field(fh, field_var::ex));
                    k_field(fg, field_var::cbz) = decay*k_field(fg, field_var::cbz) + drive*k_field(fh, field_var::cbz) - t1 + t2;
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
//        Kokkos::MDRangePolicy<Kokkos::Rank<3> > xz_policy({1,ghost,1},{nx,ghost,nz+1});
//        Kokkos::MDRangePolicy<Kokkos::Rank<3> > zx_policy({1,ghost,1},{nx+1,ghost,nz});
        switch(bc) {
            case anti_symmetric_fields:

                Kokkos::parallel_for("apply_local_tang_b<YZX>: anti_symmetric_fields: XZ edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = ghost;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbz);
                    });
                });
                Kokkos::parallel_for("apply_local_tang_b<YZX>: anti_symmetric_fields: ZX edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz ,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = ghost;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbx);
                    });
                });
/*
                Kokkos::parallel_for("XZ Edge loop", xz_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cbz) = k_field(VOXEL(ii-i,jj-j,kk-k,nx,ny,nz), field_var::cbz);
                });
                Kokkos::parallel_for("ZX Edge loop", zx_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cbx) = k_field(VOXEL(ii-i,jj-j,kk-k,nx,ny,nz), field_var::cbx);
                });
*/
                break;
            case symmetric_fields:
            case pmc_fields:

                Kokkos::parallel_for("apply_local_tang_b<YZX>: pmc_fields: XZ edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = ghost;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbz);
                    });
                });
                Kokkos::parallel_for("apply_local_tang_b<YZX>: pmc_fields: ZX edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = ghost;
                        const size_t z = zi + 1;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbx);
                    });
                });
/*
                Kokkos::parallel_for("XZ Edge loop", xz_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cbz) = -k_field(VOXEL(ii-i,jj-j,kk-k,nx,ny,nz), field_var::cbz);
                });
                Kokkos::parallel_for("ZX Edge loop", zx_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cbx) = -k_field(VOXEL(ii-i,jj-j,kk-k,nx,ny,nz), field_var::cbx);
                });
*/
                break;
            case absorb_fields:
                drive = cdt_dy*higend;
                decay = (1-drive)/(1+drive);
                drive = 2*drive/(1+drive);

Kokkos::parallel_for("single test", KOKKOS_TEAM_POLICY_DEVICE(1,1),
KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
    for(int z=1; z<=nz; z++) {
        for(int y=1; z<=ny+1; y++) {
            for(int x=ghost; x<=ghost; x++) {
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
    for(int z=1; z<=nz+1; z++) {
        for(int y=1; z<=ny; y++) {
            for(int x=ghost; x<=ghost; x++) {
                const int fg = VOXEL(x,y,z,nx,ny,nz);
                const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                y = face;
                float t1 = cdt_dy*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez));
                t1 = (i+j+k)<0 ? t1 : -t1;
                y = ghost;
                z++;
                float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey);
                z--;
                t2 = cdt_dx * (t2 - k_field(fh, field_var::ey));
                k_field(fg, field_var::cbx) = decay*k_field(fg, field_var::cbx) + drive * k_field(fh, field_var::cbx) - t1 + t2;
            }
        }
    }
});
/*
                Kokkos::parallel_for("apply_local_tang_b<YZX>: absorb_fields: XZ edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                        size_t x = xi + 1;
                        size_t y = ghost;
                        size_t z = zi + 1;
                        const int fg = VOXEL(x,y,z,nx,ny,nz);
                        const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        y = face;
                        float t1 = cdt_dy * (k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex));
                        t1 = (i+j+k)<0 ? t1 : -t1;
                        y = ghost;
                        x++;
                        float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey);
                        x--;
                        t2 = cdt_dx * (t2 - k_field(fh, field_var::ey));
                        k_field(fg, field_var::cbz) = decay * k_field(fg, field_var::cbz) + drive * k_field(fh, field_var::cbz) - t1 + t2;
                    });
                });
                Kokkos::parallel_for("apply_local_tang_b<YZX>: absorb_fields: ZX edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t zi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                        size_t x = xi + 1;
                        size_t y = ghost;
                        size_t z = zi + 1;
                        const int fg = VOXEL(x,y,z,nx,ny,nz);
                        const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        y = face;
                        float t1 = cdt_dy * (k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez));
                        t1 = (i+j+k)<0 ? t1 : -t1;
                        y = ghost;
                        z++;
                        float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey);
                        z--;
                        t2 = cdt_dz * (t2 - k_field(fh, field_var::ey));
                        k_field(fg, field_var::cbx) = decay * k_field(fg, field_var::cbx) + drive * k_field(fh, field_var::cbx) - t1 + t2;
                    });
                });
*/
/*
                Kokkos::parallel_for("XZ Edge loop", xz_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    int fg = VOXEL(ii,jj,kk,nx,ny,nz);
                    int fh = VOXEL(ii-i,jj-j,kk-k,nx,ny,nz);
                    int yy = face;
                    float t1 = cdt_dy*( k_field(VOXEL(ii-i,yy-j,kk-k,nx,ny,nz), field_var::ex) - k_field(VOXEL(ii,yy,kk,nx,ny,nz), field_var::ex));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    int xx = x+1;
                    float t2 = k_field(VOXEL(xx-i,jj-j,kk-k,nx,ny,nz), field_var::ey);
                    t2 = cdt_dx*(t2 - k_field(fh, field_var::ey));
                    k_field(fg, field_var::cbz) = decay*k_field(fg, field_var::cbz) + drive*k_field(fh, field_var::cbz) - t1 + t2;
                });
                Kokkos::parallel_for("ZX Edge loop", zx_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    int fg = VOXEL(ii,jj,kk,nx,ny,nz);
                    int fh = VOXEL(ii-i,jj-j,kk-k,nx,ny,nz);
                    int yy = face;
                    float t1 = cdt_dy*( k_field(VOXEL(ii-i,yy-j,kk-k,nx,ny,nz), field_var::ez) - k_field(VOXEL(ii,yy,kk,nx,ny,nz), field_var::ez));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    int zz = z+1;
                    float t2 = k_field(VOXEL(ii-i,jj-j,zz-k,nx,ny,nz), field_var::ey);
                    t2 = cdt_dz*(t2 - k_field(fh, field_var::ey));
                    k_field(fg, field_var::cbx) = decay*k_field(fg, field_var::cbx) + drive*k_field(fh, field_var::cbx) - t1 + t2;
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
//        Kokkos::MDRangePolicy<Kokkos::Rank<3> > yx_policy({1,1,ghost},{nx+1,ny,ghost});
//        Kokkos::MDRangePolicy<Kokkos::Rank<3> > xy_policy({1,1,ghost},{nx,ny+1,ghost});
        switch(bc) {
            case anti_symmetric_fields:

                Kokkos::parallel_for("apply_local_tang_b<ZXY>: anti_symmetric_fields: YX edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = yi + 1;
                        const size_t z = ghost;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbx);
                    });
                });
                Kokkos::parallel_for("apply_local_tang_b<ZXY>: anti_symmetric_fields: XY edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = yi + 1;
                        const size_t z = ghost;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cby);
                    });
                });
/*
                Kokkos::parallel_for("YX Edge loop", yx_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cbx) = k_field(VOXEL(ii-i,jj-j,kk-k,nx,ny,nz), field_var::cbx);
                });
                Kokkos::parallel_for("XY Edge loop", xy_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cby) = k_field(VOXEL(ii-i,jj-j,kk-k,nx,ny,nz), field_var::cby);
                });
*/
                break;
            case symmetric_fields:
            case pmc_fields:

                Kokkos::parallel_for("apply_local_tang_b<ZXY>: pmc_fields: YX edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = yi + 1;
                        const size_t z = ghost;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cbx);
                    });
                });
                Kokkos::parallel_for("apply_local_tang_b<ZXY>: pmc_fields: XY edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = yi + 1;
                        const size_t z = ghost;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = -k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::cby);
                    });
                });
/*
                Kokkos::parallel_for("YX Edge loop", yx_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cbx) = -k_field(VOXEL(ii-i,jj-j,kk-k,nx,ny,nz), field_var::cbx);
                });
                Kokkos::parallel_for("XY Edge loop", xy_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cby) = -k_field(VOXEL(ii-i,jj-j,kk-k,nx,ny,nz), field_var::cby);
                });
*/
                break;
            case absorb_fields:
                drive = cdt_dz*higend;
                decay = (1-drive)/(1+drive);
                drive = 2*drive/(1+drive);

Kokkos::parallel_for("single test", KOKKOS_TEAM_POLICY_DEVICE(1,1),
KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
    for(int z=1; z<=nz; z++) {
        for(int y=1; z<=ny+1; y++) {
            for(int x=ghost; x<=ghost; x++) {
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
    for(int z=1; z<=nz+1; z++) {
        for(int y=1; z<=ny; y++) {
            for(int x=ghost; x<=ghost; x++) {
                const int fg = VOXEL(x,y,z,nx,ny,nz);
                const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                z = face;
                float t1 = cdt_dz*(k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex));
                t1 = (i+j+k)<0 ? t1 : -t1;
                z = ghost;
                x++;
                float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez);
                x--;
                t2 = cdt_dy * (t2 - k_field(fh, field_var::ez));
                k_field(fg, field_var::cby) = decay*k_field(fg, field_var::cby) + drive * k_field(fh, field_var::cby) - t1 + t2;
            }
        }
    }
});
/*
                Kokkos::parallel_for("apply_local_tang_b<ZXY>: absorb_fields: YX edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                        size_t x = xi + 1;
                        size_t y = yi + 1;
                        size_t z = ghost;
                        const int fg = VOXEL(x,y,z,nx,ny,nz);
                        const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        z = face;
                        float t1 = cdt_dz * (k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ey) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey));
                        t1 = (i+j+k)<0 ? t1 : -t1;
                        z = ghost;
                        y++;
                        float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez);
                        y--;
                        t2 = cdt_dy * (t2 - k_field(fh, field_var::ez));
                        k_field(fg, field_var::cbx) = decay * k_field(fg, field_var::cbx) + drive * k_field(fh, field_var::cbx) - t1 + t2;
                    });
                });
                Kokkos::parallel_for("apply_local_tang_b<ZXY>: absorb_fields: XY edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1,Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                        size_t x = xi + 1;
                        size_t y = yi + 1;
                        size_t z = ghost;
                        const int fg = VOXEL(x,y,z,nx,ny,nz);
                        const int fh = VOXEL(x-i,y-j,z-k,nx,ny,nz);
                        z = face;
                        float t1 = cdt_dz * (k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ex) - k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex));
                        t1 = (i+j+k)<0 ? t1 : -t1;
                        z = ghost;
                        x++;
                        float t2 = k_field(VOXEL(x-i,y-j,z-k,nx,ny,nz), field_var::ez);
                        x--;
                        t2 = cdt_dx * (t2 - k_field(fh, field_var::ez));
                        k_field(fg, field_var::cby) = decay * k_field(fg, field_var::cby) + drive * k_field(fh, field_var::cby) - t1 + t2;
                    });
                });
*/
/*

                Kokkos::parallel_for("YX Edge Loop", yx_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    int fg = VOXEL(ii,jj,kk,nx,ny,nz);
                    int fh = VOXEL(ii-i,jj-j,kk-k,nx,ny,nz);
                    int zz = face;
                    float t1 = cdt_dz*( k_field(VOXEL(ii-i,jj-j,zz-k,nx,ny,nz), field_var::ey) - k_field(VOXEL(ii,jj,zz,nx,ny,nz), field_var::ey));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    int yy = y+1;
                    float t2 = k_field(VOXEL(ii-i,yy-j,kk-k,nx,ny,nz), field_var::ez);
                    t2 = cdt_dy*(t2 - k_field(fh, field_var::ez));
                    k_field(fg, field_var::cbx) = decay*k_field(fg, field_var::cbx) + drive*k_field(fh, field_var::cbx) - t1 + t2;
                });
                Kokkos::parallel_for("XY Edge Loop", xy_policy, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
                    int fg = VOXEL(ii,jj,kk,nx,ny,nz);
                    int fh = VOXEL(ii-i,jj-j,kk-k,nx,ny,nz);
                    int zz = face;
                    float t1 = cdt_dz*( k_field(VOXEL(ii-i,jj-j,zz-k,nx,ny,nz), field_var::ex) - k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::ex));
                    t1 = (i+j+k)<0 ? t1 : -t1;
                    int xx = x+1;
                    float t2 = k_field(VOXEL(xx-i,jj-j,kk-k,nx,ny,nz), field_var::ez);
                    t2 = cdt_dx*(t2 - k_field(fh, field_var::ez));
                    k_field(fg, field_var::cby) = decay*k_field(fg, field_var::cby) + drive*k_field(fh, field_var::cby) - t1 + t2;
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
}

template<> void adjust_tang_e<XYZ>(k_field_t& k_field, const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        int face = (i+j+k) < 0 ? 1 : nx + 1;
        Kokkos::MDRangePolicy<Kokkos::Rank<3> > zy_policy({face,1,1}, {face,ny+1,nz});
        Kokkos::MDRangePolicy<Kokkos::Rank<3> > yz_policy({face,1,1}, {face,ny,nz+1});
        switch(bc) {
            case anti_symmetric_fields:
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
                Kokkos::parallel_for("adjust_tang_e<XYZ>: ZY Edge loop", zy_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::ey) = 0;
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::tcay) = 0;
                });
                Kokkos::parallel_for("adjust_tang_e<XYZ>: YZ Edge loop", yz_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::ez) = 0;
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::tcaz) = 0;
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
}
template<> void adjust_tang_e<YZX>(k_field_t& k_field, const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        int face = (i+j+k) < 0 ? 1 : ny + 1;
        Kokkos::MDRangePolicy<Kokkos::Rank<3> > zx_policy({1,face,1},{nx+1,face,nz});
        Kokkos::MDRangePolicy<Kokkos::Rank<3> > xz_policy({1,face,1},{nx,face,nz+1});
        switch(bc) {
            case anti_symmetric_fields:
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

                Kokkos::parallel_for("adjust_tang_e<YZX>: ZX edge loop", zx_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::ez) = 0;
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::tcaz) = 0;
                });
                Kokkos::parallel_for("adjust_tang_e<YZX>: XZ edge loop", xz_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::ex) = 0;
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::tcax) = 0;
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
}
template<> void adjust_tang_e<ZXY>(k_field_t& k_field, const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
    int bc = g->bc[BOUNDARY(i,j,k)];
    if(bc < 0 || bc >= world_size) {
        int face = (i+j+k) < 0 ? 1 : nz + 1;
        Kokkos::MDRangePolicy<Kokkos::Rank<3> > yx_policy({1,1,face}, {nx+1,ny,face});
        Kokkos::MDRangePolicy<Kokkos::Rank<3> > xy_policy({1,1,face}, {nx,ny+1,face});
        switch(bc) {
            case anti_symmetric_fields:
/*
                Kokkos::parallel_for("adjust_tang_e<ZXY>: YX Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(ny, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = yi + 1;
                        const size_t z = face;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = 0;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax) = 0;
                    });
                });
                Kokkos::parallel_for("adjust_tang_e<ZXY>: XY Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
                KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
                    const size_t yi = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (const size_t xi) {
                        const size_t x = xi + 1;
                        const size_t y = yi + 1;
                        const size_t z = face;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = 0;
                        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay) = 0;
                    });
                });
*/
                Kokkos::parallel_for("adjust_tang_e<ZXY>: XY edge loop", xy_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::ex) = 0;
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::tcax) = 0;
                });
                Kokkos::parallel_for("adjust_tang_e<ZXY>: YX edge loop", yx_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::ey) = 0;
                    k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::tcay) = 0;
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

    k_field_t k_field = f->k_f_d;
    adjust_tang_e<XYZ>(k_field, g, -1, 0, 1, nx, ny, nz);
    adjust_tang_e<YZX>(k_field, g, 0, -1, 0, nx, ny, nz);
    adjust_tang_e<ZXY>(k_field, g, 0, 0, -1, nx, ny, nz);
    adjust_tang_e<XYZ>(k_field, g, 1, 0, 1, nx, ny, nz);
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

