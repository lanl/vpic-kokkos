/* 
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

#define IN_sfa
#include "sfa_private.h"

// Indexing macros
#define field(x,y,z) field[ VOXEL(x,y,z, nx,ny,nz) ]

// Generic looping
#define XYZ_LOOP(xl,xh,yl,yh,zl,zh) \
  for( z=zl; z<=zh; z++ )	    \
    for( y=yl; y<=yh; y++ )	    \
      for( x=xl; x<=xh; x++ )
	      
// yz_EDGE_LOOP => Loop over all non-ghost y-oriented edges at plane x
#define yz_EDGE_LOOP(x) XYZ_LOOP(x,x,1,ny,1,nz+1)
#define zx_EDGE_LOOP(y) XYZ_LOOP(1,nx+1,y,y,1,nz)
#define xy_EDGE_LOOP(z) XYZ_LOOP(1,nx,1,ny+1,z,z)

// zy_EDGE_LOOP => Loop over all non-ghost z-oriented edges at plane x
#define zy_EDGE_LOOP(x) XYZ_LOOP(x,x,1,ny+1,1,nz)
#define xz_EDGE_LOOP(y) XYZ_LOOP(1,nx,y,y,1,nz+1)
#define yx_EDGE_LOOP(z) XYZ_LOOP(1,nx+1,1,ny,z,z)

// x_NODE_LOOP => Loop over all non-ghost nodes at plane x
#define x_NODE_LOOP(x) XYZ_LOOP(x,x,1,ny+1,1,nz+1)
#define y_NODE_LOOP(y) XYZ_LOOP(1,nx+1,y,y,1,nz+1)
#define z_NODE_LOOP(z) XYZ_LOOP(1,nx+1,1,ny+1,z,z)

// x_FACE_LOOP => Loop over all x-faces at plane x
#define x_FACE_LOOP(x) XYZ_LOOP(x,x,1,ny,1,nz)
#define y_FACE_LOOP(y) XYZ_LOOP(1,nx,y,y,1,nz)
#define z_FACE_LOOP(z) XYZ_LOOP(1,nx,1,ny,z,z)

/*****************************************************************************
 * Ghost value communications
 *
 * Note: These functions are split into begin / end pairs to facillitate
 * overlapped communications. These functions try to interpolate the ghost
 * values when neighboring domains have a different cell size in the normal
 * direction. Whether or not this is a good idea remains to be seen. Mostly,
 * the issue is whether or not shared fields will maintain synchronicity. This
 * is especially true when materials properties are changing near domain
 * boundaries ... discrepancies over materials in the ghost cell may cause
 * shared fields to desynchronize. It is unclear how ghost material ids should
 * be assigned when different regions have differing cell sizes.
 *
 * Note: Input arguments are not tested for validity as these functions are
 * mean to be called from other field module functions (which presumably do
 * check input arguments).
 *****************************************************************************/
/*
void
begin_remote_ghost_tang_b( field_t      * ALIGNED(128) field,
                           const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int size, face, x, y, z;
  float *p;

# define BEGIN_RECV(i,j,k,X,Y,Z) \
  begin_recv_port(i,j,k,(1+n##Y*(n##Z+1)+n##Z*(n##Y+1))*sizeof(float),g)
  BEGIN_RECV(-1, 0, 0,x,y,z);
  BEGIN_RECV( 0,-1, 0,y,z,x);
  BEGIN_RECV( 0, 0,-1,z,x,y);
  BEGIN_RECV( 1, 0, 0,x,y,z);
  BEGIN_RECV( 0, 1, 0,y,z,x);
  BEGIN_RECV( 0, 0, 1,z,x,y);
# undef BEGIN_RECV

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {          \
    size = (1+n##Y*(n##Z+1)+n##Z*(n##Y+1))*sizeof(float);   \
    p = (float *)size_send_port( i, j, k, size, g );        \
    if( p ) {                                               \
      (*(p++)) = g->d##X;				    \
      face = (i+j+k)<0 ? 1 : n##X;			    \
      Z##Y##_EDGE_LOOP(face) (*(p++)) = field(x,y,z).cb##Y; \
      Y##Z##_EDGE_LOOP(face) (*(p++)) = field(x,y,z).cb##Z; \
      begin_send_port( i, j, k, size, g );                  \
    }                                                       \
  } END_PRIMITIVE
  BEGIN_SEND(-1, 0, 0,x,y,z);
  BEGIN_SEND( 0,-1, 0,y,z,x);
  BEGIN_SEND( 0, 0,-1,z,x,y);
  BEGIN_SEND( 1, 0, 0,x,y,z);
  BEGIN_SEND( 0, 1, 0,y,z,x);
  BEGIN_SEND( 0, 0, 1,z,x,y);
# undef BEGIN_SEND
}
*/

void
end_remote_ghost_tang_b( field_t      * ALIGNED(128) field,
                         const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int face, x, y, z;
  float *p, lw, rw;

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                        \
    p = (float *)end_recv_port(i,j,k,g);                                \
    if( p ) {                                                           \
      lw = (*(p++));                 /* Remote g->d##X */               \
      rw = (2.*g->d##X)/(lw+g->d##X);                                   \
      lw = (lw-g->d##X)/(lw+g->d##X);                                   \
      face = (i+j+k)<0 ? n##X+1 : 0; /* Interpolate */                  \
      Z##Y##_EDGE_LOOP(face)                                            \
        field(x,y,z).cb##Y = rw*(*(p++)) + lw*field(x+i,y+j,z+k).cb##Y; \
      Y##Z##_EDGE_LOOP(face)                                            \
        field(x,y,z).cb##Z = rw*(*(p++)) + lw*field(x+i,y+j,z+k).cb##Z; \
    }                                                                   \
  } END_PRIMITIVE
  END_RECV(-1, 0, 0,x,y,z);
  END_RECV( 0,-1, 0,y,z,x);
  END_RECV( 0, 0,-1,z,x,y);
  END_RECV( 1, 0, 0,x,y,z);
  END_RECV( 0, 1, 0,y,z,x);
  END_RECV( 0, 0, 1,z,x,y);
# undef END_RECV

# define END_SEND(i,j,k,X,Y,Z) end_send_port(i,j,k,g)
  END_SEND(-1, 0, 0,x,y,z);
  END_SEND( 0,-1, 0,y,z,x);
  END_SEND( 0, 0,-1,z,x,y);
  END_SEND( 1, 0, 0,x,y,z);
  END_SEND( 0, 1, 0,y,z,x);
  END_SEND( 0, 0, 1,z,x,y);
# undef END_SEND
}

typedef class XYZ {} XYZ;
typedef class YZX {} YZX;
typedef class ZXY {} ZXY;
typedef class XY {} XY;
typedef class YX {} YX;
typedef class XZ {} XZ;
typedef class ZX {} ZX;
typedef class YZ {} YZ;
typedef class ZY {} ZY;

void begin_recv(int i,int j,int k,int nx,int ny,int nz, const grid_t* g) {
    begin_recv_port(i,j,k,(1+ny*(nz+1)+nz*(ny+1))*sizeof(float),g);
}

template <typename T> void begin_send(int i, int j, int k, int x, int y, int z, int nX, int nY, int nZ, field_array_t*  fa, const grid_t* g) {
}
template <> void begin_send<XYZ>(int i, int j, int k, int x, int y, int z, int nx, int ny, int nz, field_array_t* field, const grid_t* g) {
    k_field_t k_field = field->k_f_d;
    const size_t size = (1+ny*(nz+1)+nz*(ny+1))*sizeof(float);   
    float* p = static_cast<float*>(size_send_port( i, j, k, size, g ));        

    if( p ) {                                               
        Kokkos::View<float*> d_buf("Device buffer", (size/sizeof(float))-1);
        Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);

        p[0] = g->dx;				    
        int face = (i+j+k)<0 ? 1 : nx;			    
        Kokkos::parallel_for(KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (size_t yi) {
                d_buf[zi*(ny+1) + yi] = k_field(VOXEL(face,yi+1,zi+1, nx,ny,nz), field_var::cby);
            });
        });
        Kokkos::parallel_for(KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (size_t yi) {
                d_buf[nz*(ny+1) + zi*ny + yi] = k_field(VOXEL(face,yi+1,zi+1, nx,ny,nz), field_var::cbz);
            });
        });
        Kokkos::deep_copy(h_buf, d_buf);
        for(size_t idx = 0; idx < (size/sizeof(float))-1; idx++) {
            p[idx+1] = h_buf(idx);
        }
        begin_send_port( i, j, k, size, g );                  
    }                                                       
}
template <> void begin_send<YZX>(int i, int j, int k, int x, int y, int z, int nx, int ny, int nz, field_array_t* field, const grid_t* g) {
    k_field_t k_field = field->k_f_d;
    size_t size = (1+nz*(nx+1)+nx*(nz+1))*sizeof(float);   
    int size_items = (1+nz*(nx+1)+nx*(nz+1));
    float* p = static_cast<float *>(size_send_port( i, j, k, size, g ));        
    Kokkos::View<float*> d_buf("device buffer", size_items-1);
    Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);

    if( p ) {                                               
      p[0] = g->dy;				    
      int face = (i+j+k)<0 ? 1 : ny;			    
        Kokkos::parallel_for(KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO), 
            KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t z = team_member.league_rank() + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                size_t x = xi + 1;
                d_buf((z-1)*nx + xi) = k_field(VOXEL(x,face,z,nx,ny,nz), field_var::cbz);
            });
        });

        Kokkos::parallel_for(KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO), 
            KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t z = team_member.league_rank() + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                size_t x = xi + 1;
                d_buf((nz+1)*nx + (nx+1)*(z-1) + xi) = k_field(VOXEL(x,face,z,nx,ny,nz), field_var::cbx);
            });
        });

        Kokkos::deep_copy(h_buf,d_buf);

        for(size_t idx=0; idx<(size/sizeof(float))-1; idx++) {
            p[idx+1] = h_buf(idx);
        }

        begin_send_port( i, j, k, size, g );                  
    }                                                       

}
template <> void begin_send<ZXY>(int i, int j, int k, int x, int y, int z, int nx, int ny, int nz, field_array_t* field, const grid_t* g) {
    size_t size = (1+nx*(ny+1)+ny*(nx+1))*sizeof(float);
    float* p = static_cast<float*>(size_send_port(i,j,k,size,g));
    k_field_t k_field = field->k_f_d;
    Kokkos::View<float*> d_buf("device buffer", 1+nx*(ny+1)+ny*(nx+1));
    Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);

    if(p){
        p[0] = g->dz;
        int face = (i+j+k)<0 ? 1 : nz;
        Kokkos::parallel_for(KOKKOS_TEAM_POLICY_DEVICE(ny+1,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
            size_t yi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                d_buf(nx*yi + xi) = k_field(VOXEL(xi+1,yi+1,face,nx,ny,nz), field_var::cbx);
            });
        });
        Kokkos::parallel_for(KOKKOS_TEAM_POLICY_DEVICE(ny,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
            size_t yi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                d_buf((ny+1)*nx + yi*(nx+1) + xi) = k_field(VOXEL(xi+1,yi+1,face,nx,ny,nz), field_var::cby);
            });
        });

        Kokkos::deep_copy(h_buf, d_buf);

        for(size_t idx=0; idx<(size/sizeof(float))-1; idx++) {
            p[idx+1] = h_buf(idx);
        }

        begin_send_port(i,j,k,size,g);
    }

}


void
k_begin_remote_ghost_tang_b( field_array_t      * RESTRICT fa,
                           const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int x=0, y=0, z=0;

    begin_recv(-1,0,0,nx,ny,nz,g);
    begin_recv(0,-1,0,ny,nz,nx,g);
    begin_recv(0,0,-1,nz,nx,ny,g);
    begin_recv(1,0,0,nx,ny,nz,g);
    begin_recv(0,1,0,ny,nz,nx,g);
    begin_recv(0,0,1,nz,nx,ny,g);
/*
# define BEGIN_RECV(i,j,k,X,Y,Z) \
  begin_recv_port(i,j,k,(1+n##Y*(n##Z+1)+n##Z*(n##Y+1))*sizeof(float),g)
  BEGIN_RECV(-1, 0, 0,x,y,z);
  BEGIN_RECV( 0,-1, 0,y,z,x);
  BEGIN_RECV( 0, 0,-1,z,x,y);
  BEGIN_RECV( 1, 0, 0,x,y,z);
  BEGIN_RECV( 0, 1, 0,y,z,x);
  BEGIN_RECV( 0, 0, 1,z,x,y);
# undef BEGIN_RECV
*/
    begin_send<XYZ>(-1,0,0,x,y,z,nx,ny,nz,fa,g);
    begin_send<YZX>(0,-1,0,x,y,z,nx,ny,nz,fa,g);
    begin_send<ZXY>(0,0,-1,x,y,z,nx,ny,nz,fa,g);
    begin_send<XYZ>(1,0,0,x,y,z,nx,ny,nz,fa,g);
    begin_send<YZX>(0,1,0,x,y,z,nx,ny,nz,fa,g);
    begin_send<ZXY>(0,0,1,x,y,z,nx,ny,nz,fa,g);
/*
# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {          \
    size = (1+n##Y*(n##Z+1)+n##Z*(n##Y+1))*sizeof(float);   \
    p = (float *)size_send_port( i, j, k, size, g );        \
    if( p ) {                                               \
      (*(p++)) = g->d##X;				    \
      face = (i+j+k)<0 ? 1 : n##X;			    \
      Z##Y##_EDGE_LOOP(face) (*(p++)) = field(x,y,z).cb##Y; \
      Y##Z##_EDGE_LOOP(face) (*(p++)) = field(x,y,z).cb##Z; \
      begin_send_port( i, j, k, size, g );                  \
    }                                                       \
  } END_PRIMITIVE
/
  BEGIN_SEND(-1, 0, 0,x,y,z);
  BEGIN_SEND( 0,-1, 0,y,z,x);
  BEGIN_SEND( 0, 0,-1,z,x,y);
  BEGIN_SEND( 1, 0, 0,x,y,z);
  BEGIN_SEND( 0, 1, 0,y,z,x);
  BEGIN_SEND( 0, 0, 1,z,x,y);
# undef BEGIN_SEND
*/
}

void
begin_remote_ghost_tang_b( field_t      * ALIGNED(128) field,
                           const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int size, face, x, y, z;
  float *p;

# define BEGIN_RECV(i,j,k,X,Y,Z) \
  begin_recv_port(i,j,k,(1+n##Y*(n##Z+1)+n##Z*(n##Y+1))*sizeof(float),g)
  BEGIN_RECV(-1, 0, 0,x,y,z);
  BEGIN_RECV( 0,-1, 0,y,z,x);
  BEGIN_RECV( 0, 0,-1,z,x,y);
  BEGIN_RECV( 1, 0, 0,x,y,z);
  BEGIN_RECV( 0, 1, 0,y,z,x);
  BEGIN_RECV( 0, 0, 1,z,x,y);
# undef BEGIN_RECV
# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {          \
    size = (1+n##Y*(n##Z+1)+n##Z*(n##Y+1))*sizeof(float);   \
    p = (float *)size_send_port( i, j, k, size, g );        \
    if( p ) {                                               \
      (*(p++)) = g->d##X;				    \
      face = (i+j+k)<0 ? 1 : n##X;			    \
      Z##Y##_EDGE_LOOP(face) (*(p++)) = field(x,y,z).cb##Y; \
      Y##Z##_EDGE_LOOP(face) (*(p++)) = field(x,y,z).cb##Z; \
      begin_send_port( i, j, k, size, g );                  \
    }                                                       \
  } END_PRIMITIVE

  BEGIN_SEND(-1, 0, 0,x,y,z);
  BEGIN_SEND( 0,-1, 0,y,z,x);
  BEGIN_SEND( 0, 0,-1,z,x,y);
  BEGIN_SEND( 1, 0, 0,x,y,z);
  BEGIN_SEND( 0, 1, 0,y,z,x);
  BEGIN_SEND( 0, 0, 1,z,x,y);
# undef BEGIN_SEND

}

template<typename T> void end_recv(int i, int j, int k, int x, int y, int z, int nx, int ny, int nz, field_array_t* RESTRICT field, const grid_t* g) {
}

template<> void end_recv<XYZ>(int i, int j, int k, int x, int y, int z, int nx, int ny, int nz, field_array_t* RESTRICT field, const grid_t* g) {
    float* p = static_cast<float*>(end_recv_port(i,j,k,g));
    size_t size = 1 + (ny+1)*nz + ny*(nz+1);
    if(p) {
        Kokkos::View<float*> d_buf("Device buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);
        for(size_t idx = 0; idx < size; idx++) {
            h_buf(idx) = p[idx];
        }
        Kokkos::deep_copy(d_buf, h_buf);

        k_field_t k_field = field->k_f_d;

        float lw = h_buf[0];
        float rw = (2.*g->dx) / (lw+g->dx);
        lw = (lw-g->dx)/(lw+g->dx);
        int face = (i+j+k)<0 ? nx+1 : 0;

        Kokkos::MDRangePolicy<Kokkos::Rank<3> > zy_policy({face,1,1}, {face,ny+1,nz});
        Kokkos::MDRangePolicy<Kokkos::Rank<3> > yz_policy({face,1,1}, {face,ny,nz+1});
        Kokkos::parallel_for("end_recv<XYZ>: ZY Edge loop", zy_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
            k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cby) = rw*d_buf[1+kk*(ny+1) + jj] + lw*k_field(VOXEL(ii+i,jj+j,kk+k,nx,ny,nz), field_var::cby);
        });
        Kokkos::parallel_for("end_recv<XYZ>: YZ Edge loop", yz_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = rw*d_buf[1+(ny+1)*nz + kk*ny + jj] + lw*k_field(VOXEL(ii+i,jj+j,kk+k,nx,ny,nz), field_var::cbz);
        });
    }
}
template<> void end_recv<YZX>(int i, int j, int k, int x, int y, int z, int nx, int ny, int nz, field_array_t* RESTRICT field, const grid_t* g) {
    float* p = static_cast<float*>(end_recv_port(i,j,k,g));
    size_t size = 1 + nx*(nz+1) + (nx+1)*nz;
    if(p) {
        Kokkos::View<float*> d_buf("Device buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);
        for(size_t idx = 0; idx < size; idx++) {
            h_buf(idx) = p[idx];
        }
        Kokkos::deep_copy(d_buf, h_buf);
        k_field_t k_field = field->k_f_d;

        float lw = h_buf[0];
        float rw = (2.*g->dy) / (lw+g->dy);
        lw = (lw-g->dy)/(lw+g->dy);
        int face = (i+j+k)<0 ? ny+1 : 0;

        Kokkos::MDRangePolicy<Kokkos::Rank<3> > xz_policy({1,face,1},{nx,face,nz+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<3> > zx_policy({1,face,1},{nx+1,face,nz});
        Kokkos::parallel_for("end_recv<YZX>: XZ Edge loop", xz_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
            k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cbz) = rw*d_buf[1+kk*nx + ii] + lw*k_field(VOXEL(ii+i,jj+j,kk+k,nx,ny,nz), field_var::cbz);
        });
        Kokkos::parallel_for("end_recv<YZX>: ZX Edge loop", zx_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
            k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cbx) = rw*d_buf[1+nx*(nz+1)+kk*(nx+1) + ii] + lw*k_field(VOXEL(ii+i,jj+j,kk+k,nx,ny,nz), field_var::cbx);
        });
    }
}
template<> void end_recv<ZXY>(int i, int j, int k, int x, int y, int z, int nx, int ny, int nz, field_array_t* RESTRICT field, const grid_t* g) {
    float* p = static_cast<float*>(end_recv_port(i,j,k,g));
    if(p) {
        size_t size = 1 + (nx+1)*ny + nx*(ny+1);
        k_field_t k_field = field->k_f_d;
        Kokkos::View<float*> d_buf("Device buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);
        for(size_t idx = 0; idx < size; idx++) {
            h_buf(idx) = p[idx];
        }
        Kokkos::deep_copy(d_buf, h_buf);

        float lw = h_buf[0];
        float rw = (2.*g->dz) / (lw+g->dz);
        lw = (lw-g->dz)/(lw+g->dz);
        int face = (i+j+k)<0 ? nz+1 : 0;

        Kokkos::MDRangePolicy<Kokkos::Rank<3> > yx_policy({1,1,face}, {nx+1,ny,face});
        Kokkos::MDRangePolicy<Kokkos::Rank<3> > xy_policy({1,1,face}, {nx,ny+1,face});
        Kokkos::parallel_for("end_recv<ZXY>: YX Edge loop", yx_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
            k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cbx) = rw*d_buf[1 + (nx+1)*jj + ii] + lw*k_field(VOXEL(ii+i,jj+j,kk+k,nx,ny,nz), field_var::cbx);
        });
        Kokkos::parallel_for("end_recv<ZXY>: XY Edge loop", xy_policy, KOKKOS_LAMBDA(const size_t ii, const size_t jj, const size_t kk) {
            k_field(VOXEL(ii,jj,kk,nx,ny,nz), field_var::cby) = rw*d_buf[1 + (nx+1)*ny + nx*jj + ii] + lw*k_field(VOXEL(ii+i,jj+j,kk+k,nx,ny,nz), field_var::cby);
        });
    }
}

void
k_end_remote_ghost_tang_b( field_array_t      * RESTRICT field,
                         const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int x=0, y=0, z=0;

    end_recv<XYZ>(-1,0,0,x,y,z,nx,ny,nz,field,g);
    end_recv<YZX>(0,-1,0,x,y,z,nx,ny,nz,field,g);
    end_recv<ZXY>(0,0,-1,x,y,z,nx,ny,nz,field,g);
    end_recv<XYZ>(1,0,0,x,y,z,nx,ny,nz,field,g);
    end_recv<YZX>(0,1,0,x,y,z,nx,ny,nz,field,g);
    end_recv<ZXY>(0,0,1,x,y,z,nx,ny,nz,field,g);

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                        \
    p = (float *)end_recv_port(i,j,k,g);                                \
    if( p ) {                                                           \
      lw = (*(p++));                 /* Remote g->d##X */               \
      rw = (2.*g->d##X)/(lw+g->d##X);                                   \
      lw = (lw-g->d##X)/(lw+g->d##X);                                   \
      face = (i+j+k)<0 ? n##X+1 : 0; /* Interpolate */                  \
      Z##Y##_EDGE_LOOP(face)                                            \ field(x,y,z).cb##Y = rw*(*(p++)) + lw*field(x+i,y+j,z+k).cb##Y; \
      Y##Z##_EDGE_LOOP(face)                                            \
        field(x,y,z).cb##Z = rw*(*(p++)) + lw*field(x+i,y+j,z+k).cb##Z; \
    }                                                                   \
  } END_PRIMITIVE
/*
  END_RECV(-1, 0, 0,x,y,z);
  END_RECV( 0,-1, 0,y,z,x);
  END_RECV( 0, 0,-1,z,x,y);
  END_RECV( 1, 0, 0,x,y,z);
  END_RECV( 0, 1, 0,y,z,x);
  END_RECV( 0, 0, 1,z,x,y);
*/
# undef END_RECV

end_send_port(-1,0,0,g);
end_send_port(0,-1,0,g);
end_send_port(0,0,-1,g);
end_send_port(1,0,0,g);
end_send_port(0,1,0,g);
end_send_port(0,0,1,g);

/*
# define END_SEND(i,j,k,X,Y,Z) end_send_port(i,j,k,g)
  END_SEND(-1, 0, 0,x,y,z);
  END_SEND( 0,-1, 0,y,z,x);
  END_SEND( 0, 0,-1,z,x,y);
  END_SEND( 1, 0, 0,x,y,z);
  END_SEND( 0, 1, 0,y,z,x);
  END_SEND( 0, 0, 1,z,x,y);
# undef END_SEND
*/
}

void
begin_remote_ghost_norm_e( field_t      * ALIGNED(128) field,
                           const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int size, face, x, y, z;
  float *p;

# define BEGIN_RECV(i,j,k,X,Y,Z) \
  begin_recv_port(i,j,k,( 1 + (n##Y+1)*(n##Z+1) )*sizeof(float),g)
  BEGIN_RECV(-1, 0, 0,x,y,z);
  BEGIN_RECV( 0,-1, 0,y,z,x);
  BEGIN_RECV( 0, 0,-1,z,x,y);
  BEGIN_RECV( 1, 0, 0,x,y,z);
  BEGIN_RECV( 0, 1, 0,y,z,x);
  BEGIN_RECV( 0, 0, 1,z,x,y);
# undef BEGIN_RECV

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {          \
    size = ( 1+ (n##Y+1)*(n##Z+1) )*sizeof(float);          \
    p = (float *)size_send_port( i, j, k, size, g );        \
    if( p ) {                                               \
      (*(p++)) = g->d##X;				    \
      face = (i+j+k)<0 ? 1 : n##X;			    \
      X##_NODE_LOOP(face) (*(p++)) = field(x,y,z).e##X;     \
      begin_send_port( i, j, k, size, g );                  \
    }                                                       \
  } END_PRIMITIVE
  BEGIN_SEND(-1, 0, 0,x,y,z);
  BEGIN_SEND( 0,-1, 0,y,z,x);
  BEGIN_SEND( 0, 0,-1,z,x,y);
  BEGIN_SEND( 1, 0, 0,x,y,z);
  BEGIN_SEND( 0, 1, 0,y,z,x);
  BEGIN_SEND( 0, 0, 1,z,x,y);
# undef BEGIN_SEND
}

void
end_remote_ghost_norm_e( field_t      * ALIGNED(128) field,
                         const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int face, x, y, z;
  float *p, lw, rw;

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                      \
    p = (float *)end_recv_port(i,j,k,g);                              \
    if( p ) {                                                         \
      lw = (*(p++));                 /* Remote g->d##X */             \
      rw = (2.*g->d##X)/(lw+g->d##X);                                 \
      lw = (lw-g->d##X)/(lw+g->d##X);                                 \
      face = (i+j+k)<0 ? n##X+1 : 0; /* Interpolate */                \
      X##_NODE_LOOP(face)                                             \
        field(x,y,z).e##X = rw*(*(p++)) + lw*field(x+i,y+j,z+k).e##X; \
    }                                                                 \
  } END_PRIMITIVE
  END_RECV(-1, 0, 0,x,y,z);
  END_RECV( 0,-1, 0,y,z,x);
  END_RECV( 0, 0,-1,z,x,y);
  END_RECV( 1, 0, 0,x,y,z);
  END_RECV( 0, 1, 0,y,z,x);
  END_RECV( 0, 0, 1,z,x,y);
# undef END_RECV

# define END_SEND(i,j,k,X,Y,Z) end_send_port(i,j,k,g)
  END_SEND(-1, 0, 0,x,y,z);
  END_SEND( 0,-1, 0,y,z,x);
  END_SEND( 0, 0,-1,z,x,y);
  END_SEND( 1, 0, 0,x,y,z);
  END_SEND( 0, 1, 0,y,z,x);
  END_SEND( 0, 0, 1,z,x,y);
# undef END_SEND
}

void
begin_remote_ghost_div_b( field_t      * ALIGNED(128) field,
                          const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int size, face, x, y, z;
  float *p;

# define BEGIN_RECV(i,j,k,X,Y,Z) \
  begin_recv_port(i,j,k,(1+n##Y*n##Z)*sizeof(float),g)
  BEGIN_RECV(-1, 0, 0,x,y,z);
  BEGIN_RECV( 0,-1, 0,y,z,x);
  BEGIN_RECV( 0, 0,-1,z,x,y);
  BEGIN_RECV( 1, 0, 0,x,y,z);
  BEGIN_RECV( 0, 1, 0,y,z,x);
  BEGIN_RECV( 0, 0, 1,z,x,y);
# undef BEGIN_RECV

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {           \
    size = ( 1 + n##Y*n##Z )*sizeof(float);                  \
    p = (float *)size_send_port( i, j, k, size, g );         \
    if( p ) {                                                \
      (*(p++)) = g->d##X;				     \
      face = (i+j+k)<0 ? 1 : n##X;			     \
      X##_FACE_LOOP(face) (*(p++)) = field(x,y,z).div_b_err; \
      begin_send_port( i, j, k, size, g );                   \
    }                                                        \
  } END_PRIMITIVE
  BEGIN_SEND(-1, 0, 0,x,y,z);
  BEGIN_SEND( 0,-1, 0,y,z,x);
  BEGIN_SEND( 0, 0,-1,z,x,y);
  BEGIN_SEND( 1, 0, 0,x,y,z);
  BEGIN_SEND( 0, 1, 0,y,z,x);
  BEGIN_SEND( 0, 0, 1,z,x,y);
# undef BEGIN_SEND
}

void
end_remote_ghost_div_b( field_t      * ALIGNED(128) field,
                        const grid_t *              g ) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int face, x, y, z;
  float *p, lw, rw;

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                        \
    p = (float *)end_recv_port(i,j,k,g);                                \
    if( p ) {                                                           \
      lw = (*(p++));                 /* Remote g->d##X */               \
      rw = (2.*g->d##X)/(lw+g->d##X);                                   \
      lw = (lw-g->d##X)/(lw+g->d##X);                                   \
      face = (i+j+k)<0 ? n##X+1 : 0; /* Interpolate */                  \
      X##_FACE_LOOP(face)                                               \
        field(x,y,z).div_b_err = rw*(*(p++)) +                          \
                                 lw*field(x+i,y+j,z+k).div_b_err;       \
    }                                                                   \
  } END_PRIMITIVE
  END_RECV(-1, 0, 0,x,y,z);
  END_RECV( 0,-1, 0,y,z,x);
  END_RECV( 0, 0,-1,z,x,y);
  END_RECV( 1, 0, 0,x,y,z);
  END_RECV( 0, 1, 0,y,z,x);
  END_RECV( 0, 0, 1,z,x,y);
# undef END_RECV

# define END_SEND(i,j,k,X,Y,Z) end_send_port(i,j,k,g)
  END_SEND(-1, 0, 0,x,y,z);
  END_SEND( 0,-1, 0,y,z,x);
  END_SEND( 0, 0,-1,z,x,y);
  END_SEND( 1, 0, 0,x,y,z);
  END_SEND( 0, 1, 0,y,z,x);
  END_SEND( 0, 0, 1,z,x,y);
# undef END_SEND
}

/*****************************************************************************
 * Synchronization functions
 *
 * The communication is done in three passes so that small edge and corner
 * communications can be avoided. However, this prevents overlapping
 * synchronizations with other computations. Ideally, synchronize_jf should be
 * overlappable so that a half advance_b can occur while communications are
 * occuring. The other synchronizations are less important to overlap as they
 * only occur in conjunction with infrequent operations.
 * 
 * FIXME: THIS COMMUNICATION PATTERN PROHIBITS CONCAVE LOCAL MESH
 * CONNECTIVITIES.
 *
 * Note: These functions are lightly test the input arguments as these
 * functions are meant to be used externally.
 *****************************************************************************/

double
synchronize_tang_e_norm_b( field_array_t * RESTRICT fa ) {
  field_t * field, * f;
  grid_t * RESTRICT g;
  float * p;
  double w1, w2, err = 0, gerr;
  int size, face, x, y, z, nx, ny, nz;

  if( !fa ) ERROR(( "Bad args" ));
  field = fa->f;
  g     = fa->g;

  local_adjust_tang_e( field, g );
  local_adjust_norm_b( field, g );

  nx = g->nx;
  ny = g->ny;
  nz = g->nz;

# define BEGIN_RECV(i,j,k,X,Y,Z)                                \
  begin_recv_port(i,j,k, ( 2*n##Y*(n##Z+1) + 2*n##Z*(n##Y+1) +  \
                          n##Y*n##Z )*sizeof(float), g )

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {              \
    size = ( 2*n##Y*(n##Z+1) + 2*n##Z*(n##Y+1) +                \
             n##Y*n##Z )*sizeof(float);                         \
    p = (float *)size_send_port( i, j, k, size, g );            \
    if( p ) {                                                   \
      face = (i+j+k)<0 ? 1 : n##X+1;                            \
      X##_FACE_LOOP(face) (*(p++)) = field(x,y,z).cb##X;        \
      Y##Z##_EDGE_LOOP(face) {                                  \
        f = &field(x,y,z);                                      \
        (*(p++)) = f->e##Y;                                     \
        (*(p++)) = f->tca##Y;                                   \
      }                                                         \
      Z##Y##_EDGE_LOOP(face) {                                  \
        f = &field(x,y,z);                                      \
        (*(p++)) = f->e##Z;                                     \
        (*(p++)) = f->tca##Z;                                   \
      }                                                         \
      begin_send_port( i, j, k, size, g );                      \
    }                                                           \
  } END_PRIMITIVE

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {   \
    p = (float *)end_recv_port(i,j,k,g);           \
    if( p ) {                                      \
      face = (i+j+k)<0 ? n##X+1 : 1; /* Average */ \
      X##_FACE_LOOP(face) {                        \
        f = &field(x,y,z);                         \
        w1 = (*(p++));                             \
        w2 = f->cb##X;                             \
        f->cb##X = 0.5*( w1+w2 );		   \
        err += (w1-w2)*(w1-w2);                    \
      }                                            \
      Y##Z##_EDGE_LOOP(face) {                     \
        f = &field(x,y,z);                         \
        w1 = (*(p++));                             \
        w2 = f->e##Y;                              \
        f->e##Y = 0.5*( w1+w2 );		   \
        err += (w1-w2)*(w1-w2);                    \
        w1 = (*(p++));                             \
        w2 = f->tca##Y;                            \
        f->tca##Y = 0.5*( w1+w2 );		   \
      }                                            \
      Z##Y##_EDGE_LOOP(face) {                     \
        f = &field(x,y,z);                         \
        w1 = (*(p++));                             \
        w2 = f->e##Z;                              \
        f->e##Z = 0.5*( w1+w2 );		   \
        err += (w1-w2)*(w1-w2);                    \
        w1 = (*(p++));                             \
        w2 = f->tca##Z;                            \
        f->tca##Z = 0.5*( w1+w2 );		   \
      }                                            \
    }                                              \
  } END_PRIMITIVE

# define END_SEND(i,j,k,X,Y,Z) end_send_port( i, j, k, g )

  // Exchange x-faces
  BEGIN_RECV(-1, 0, 0,x,y,z);
  BEGIN_RECV( 1, 0, 0,x,y,z);
  BEGIN_SEND(-1, 0, 0,x,y,z);
  BEGIN_SEND( 1, 0, 0,x,y,z);
  END_SEND(-1, 0, 0,x,y,z);
  END_SEND( 1, 0, 0,x,y,z);
  END_RECV(-1, 0, 0,x,y,z);
  END_RECV( 1, 0, 0,x,y,z);

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

  mp_allsum_d( &err, &gerr, 1 );
  return gerr;
}

void
synchronize_jf( field_array_t * RESTRICT fa ) {
  field_t * field, * f;
  grid_t * RESTRICT g;
  int size, face, x, y, z, nx, ny, nz;
  float *p, lw, rw;

  if( !fa ) ERROR(( "Bad args" ));
  field = fa->f;
  g     = fa->g;

  local_adjust_jf( field, g );

  nx = g->nx;
  ny = g->ny;
  nz = g->nz;

# define BEGIN_RECV(i,j,k,X,Y,Z)                                        \
  begin_recv_port(i,j,k, ( n##Y*(n##Z+1) +                              \
                           n##Z*(n##Y+1) + 1 )*sizeof(float), g )

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {              \
    size = ( n##Y*(n##Z+1) +                                    \
             n##Z*(n##Y+1) + 1 )*sizeof(float);                 \
    p = (float *)size_send_port( i, j, k, size, g );            \
    if( p ) {                                                   \
      (*(p++)) = g->d##X;                                       \
      face = (i+j+k)<0 ? 1 : n##X+1;                            \
      Y##Z##_EDGE_LOOP(face) (*(p++)) = field(x,y,z).jf##Y;     \
      Z##Y##_EDGE_LOOP(face) (*(p++)) = field(x,y,z).jf##Z;     \
      begin_send_port( i, j, k, size, g );                      \
    }                                                           \
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
      Y##Z##_EDGE_LOOP(face) {                                  \
        f = &field(x,y,z);                                      \
        f->jf##Y = lw*f->jf##Y + rw*(*(p++));                   \
      }                                                         \
      Z##Y##_EDGE_LOOP(face) {                                  \
        f = &field(x,y,z);                                      \
        f->jf##Z = lw*f->jf##Z + rw*(*(p++));                   \
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

// Note: synchronize_rho assumes that rhof has _not_ been adjusted at
// the local domain boundary to account for partial cells but that
// rhob _has_.  Specifically it is very expensive to accumulate rhof
// and doing the adjustment for each particle is adds even more
// expense.  Worse, if we locally corrected it after each species,
// we cannot accumulate the next species in the same unless we use
// (running sum of locally corrected results and thw current species
// rhof being accumulated).  Further, rhof is always accumulated from
// scratch so we don't have to worry about whether or not the previous
// rhof values were in a locally corrected form.  Thus, after all
// particles have accumulated to rhof, we correct it for partial cells
// and remote cells for use with divergence cleaning and this is
// the function that does the correction.
//
// rhob is another story though.  rhob is continuously incrementally
// accumulated over time typically through infrequent surface area
// scaling processes.  Like rho_f, after synchronize_rhob, rhob _must_
// be corrected for partial and remote celle for the benefit of
// divergence cleaning. And like rho_f, since we don't want to have
// to keep around two versions of rhob (rhob contributions since last
// sync and rhob as of last sync), we have no choice but to do the
// charge accumulation per particle to rhob in a locally corrected
// form.

void
synchronize_rho( field_array_t * RESTRICT fa ) {
  field_t * field, * f;
  grid_t * RESTRICT g;
  int size, face, x, y, z, nx, ny, nz;
  float *p, hlw, hrw, lw, rw;

  if( !fa ) ERROR(( "Bad args" ));
  field = fa->f;
  g     = fa->g;

  local_adjust_rhof( field, g );
  local_adjust_rhob( field, g );

  nx = g->nx;
  ny = g->ny;
  nz = g->nz;

# define BEGIN_RECV(i,j,k,X,Y,Z) \
  begin_recv_port(i,j,k, ( 1 + 2*(n##Y+1)*(n##Z+1) )*sizeof(float), g )

# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {      \
    size = ( 1 + 2*(n##Y+1)*(n##Z+1) )*sizeof(float);   \
    p = (float *)size_send_port( i, j, k, size, g );    \
    if( p ) {                                           \
      (*(p++)) = g->d##X;                               \
      face = (i+j+k)<0 ? 1 : n##X+1;                    \
      X##_NODE_LOOP(face) {                             \
        f = &field(x,y,z);                              \
        (*(p++)) = f->rhof;                             \
        (*(p++)) = f->rhob;                             \
      }                                                 \
      begin_send_port( i, j, k, size, g );              \
    }                                                   \
  } END_PRIMITIVE

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                \
    p = (float *)end_recv_port(i,j,k,g);                        \
    if( p ) {                                                   \
      hrw  = (*(p++));               /* Remote g->d##X */       \
      hlw  = hrw + g->d##X;                                     \
      hrw /= hlw;                                               \
      hlw  = g->d##X/hlw;                                       \
      lw   = hlw + hlw;                                         \
      rw   = hrw + hrw;                                         \
      face = (i+j+k)<0 ? n##X+1 : 1;                            \
      X##_NODE_LOOP(face) {					\
        f = &field(x,y,z);					\
        f->rhof =  lw*f->rhof  + rw*(*(p++));                   \
        f->rhob = hlw*f->rhob + hrw*(*(p++));                   \
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


