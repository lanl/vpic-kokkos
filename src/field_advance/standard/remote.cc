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
#include "mpi.h"

// GPU aware MPI macros
#ifdef VPIC_ENABLE_GPU_AWARE_MPI

#define SYNC_MPI_BUFFER(dst, src) Kokkos::fence();
#define BEGIN_SEND_PORT_K(i, j, k, size, g, sendbuf_d, sendbuf_h); \
  begin_send_port_k(i,j,k,size,g, reinterpret_cast<char*>(sendbuf_d.data())); 
#define BEGIN_RECV_PORT_K(i, j, k, size, g, recvbuf_d, recvbuf_h); \
  begin_recv_port_k(i,j,k,size,g, reinterpret_cast<char*>(recvbuf_d.data()));

#else

#define SYNC_MPI_BUFFER(dst, src) Kokkos::deep_copy(dst, src);
#define BEGIN_SEND_PORT_K(i, j, k, size, g, sendbuf_d, sendbuf_h); \
  begin_send_port_k(i,j,k,size,g, reinterpret_cast<char*>(sendbuf_h.data())); 
#define BEGIN_RECV_PORT_K(i, j, k, size, g, recvbuf_d, recvbuf_h); \
  begin_recv_port_k(i,j,k,size,g, reinterpret_cast<char*>(recvbuf_h.data()));

#endif

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

template <int i, int j, int k> 
void 
begin_recv_kokkos(const grid_t* g, field_buffers_t& fb) {
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fb.recv_buffer[BOUNDARY(i,j,k)];;
  if constexpr (i!=0 && j==0 && k==0) {
    size = (1 + ny*(nz+1) + nz*(ny+1))*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = (1 + nx*(nz+1) + nz*(nx+1))*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = (1 + nx*(ny+1) + ny*(nx+1))*sizeof(float);
  }
// Switch between CPU and GPU MPI
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  BEGIN_RECV_PORT_K(i,j,k,size,g,rbuf_d,rbuf_h);
//  begin_recv_port_k(i,j,k,size,g,reinterpret_cast<char*>(rbuf_d.data()));
}

template <typename T> void begin_recv(int i, int j, int k, int nx, int ny, int nz, const grid_t* g) {
    int nY, nZ;
    if (std::is_same<T, XYZ>::value) {
        nY = ny, nZ = nz;
    } else if (std::is_same<T, YZX>::value) {
        nY = nz, nZ = nx;
    } else if (std::is_same<T, ZXY>::value) {
        nY = nx, nZ = ny;
    }
    begin_recv_port(i,j,k,(1 + nY*(nZ+1) + nZ*(nY+1))*sizeof(float),g);
}

//template<> void begin_recv<XYZ>(int i, int j, int k, int nx, int ny, int nz, const grid_t* g) {
//    begin_recv_port(i,j,k,(1+ny*(nz+1)+nz*(ny+1))*sizeof(float),g);
//}
//template<> void begin_recv<YZX>(int i, int j, int k, int nx, int ny, int nz, const grid_t* g) {
//    begin_recv_port(i,j,k,(1+nz*(nx+1)+nx*(nz+1))*sizeof(float),g);
//}
//template<> void begin_recv<ZXY>(int i, int j, int k, int nx, int ny, int nz, const grid_t* g) {
//    begin_recv_port(i,j,k,(1+nx*(ny+1)+ny*(nx+1))*sizeof(float),g);
//}

template <typename Face> 
void 
begin_send_kokkos(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::DualView<float*>& sbuf) {}

//template <int i, int j, int k> 
//void 
//begin_send_kokkos(const grid_t* g, field_array_t* fa, Kokkos::DualView<float*>& sbuf) {
//#define BEGIN_SEND(x_,y_,z_) \
//  const size_t size = (1+n##y_*(n##z_+1)+n##z_*(n##y_+1));                     \
//  const int face = (i+j+k)<0 ? 1 : n##x_;                                      \
//  const float d##x_ = g->d##x_;                                                \
//  Kokkos::MDRangePolicy<Kokkos::Rank<2>> y_##z_##_edge({1,1}, {n##y_+2, n##z_+1); \
//  Kokkos::MDRangePolicy<Kokkos::Rank<2>> z_##y_##_edge({1,1}, {n##y_+1, n##z_+2); \
//  Kokkos::parallel_for("begin_send<" #x_ #y_ #z_ ">", y_##z_##_edge,           \
//    KOKKOS_LAMBDA(const int y_, const int z_) {                                \
//      const int x = face;                                                      \
//      const size_t idx = 1 + (z_-1)*(n##y_+1) + (y_-1);                        \
//      sbuf_d(idx) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::cby);           \
//    });                                                                        \
//  Kokkos::parallel_for("begin_send<" #x_ #y_ #z_ ">", y_##z_##_edge,           \
//    KOKKOS_LAMBDA(const int y_, const int z_) {                                \
//      const int x = face;                                                      \
//      const size_t idx = 1 + n##z_*(n##y_+1) + (z_-1)*n##y_ + (y_-1);          \
//      sbuf_d(idx) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::cby);           \
//    });
//
//  const int nx = g->nx, ny = y->ny, nz = g->nz;
//  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
//  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
//  k_field_t& k_field = fa->k_f_d;
//
//  if constexpr (i!=0 && j==0 && k==0) {
//    const size_t size = (1+ny*(nz+1)+nz*(ny+1));
//    BEGIN_SEND(x,y,z);
//  } else if constexpr (i==0 && j!=0 && k==0) {
//    const size_t size = (1+nz*(nx+1)+nx*(nz+1));
//    BEGIN_SEND(y,z,x);
//  } else if constexpr (i==0 && j==0 && k!=0) {
//    const size_t size = (1+nx*(ny+1)+ny*(nx+1));
//    BEGIN_SEND(z,x,y);
//  }
//
//  sbuf.modify<Kokkos::DefaultExecutionSpace>();
//// CPU
//  sbuf.sync<Kokkos::DefaultHostExecutionSpace>();
////  Kokkos::deep_copy(sbuf_h, sbuf_d);
//  sbuf_h(0) = dx;
//  begin_send_port_k(i,j,k,size*sizeof(float), g, reinterpret_cast<char*>(sbuf_h.data()));
//
//// GPU
////  begin_send_port_k(i,j,k,size*sizeof(float), g, reinterpret_cast<char*>(sbuf_d.data()));
//
//#undef BEGIN_SEND
//}

template <> void begin_send_kokkos<XYZ>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::DualView<float*>& sbuf) {
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    k_field_t& k_field = fa->k_f_d;
    const size_t size = (1+ny*(nz+1)+nz*(ny+1));

        int face = (i+j+k)<0 ? 1 : nx;
        float dx = g->dx;
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_edge({1, 1}, {nz+1, ny+2});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yz_edge({1, 1}, {nz+2, ny+1});

        Kokkos::parallel_for("begin_send<XYZ>: ZY Edge Loop", zy_edge, KOKKOS_LAMBDA(const int z, const int y) {
//            if(z+y == 2) {
//                sbuf_d(0) = dx;
//            }
            const int x = face;
            sbuf_d(1 + (z-1)*(ny+1) + (y-1)) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::cby);
        });
        Kokkos::parallel_for("begin_send<XYZ>: YZ Edge Loop", yz_edge, KOKKOS_LAMBDA(const int z, const int y) {
            const int x = face;
            sbuf_d(1 + nz*(ny+1) + (z-1)*ny + (y-1)) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::cbz);
        });
        sbuf.modify<Kokkos::DefaultExecutionSpace>();
// CPU
        sbuf.sync<Kokkos::DefaultHostExecutionSpace>();
//        Kokkos::deep_copy(sbuf_h, sbuf_d);
        sbuf_h(0) = dx;
        begin_send_port_k(i,j,k,size*sizeof(float), g, reinterpret_cast<char*>(sbuf_h.data()));

// GPU
//        begin_send_port_k(i,j,k,size*sizeof(float), g, reinterpret_cast<char*>(sbuf_d.data()));

}
template <> void begin_send_kokkos<YZX>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::DualView<float*>& sbuf) {
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    k_field_t& k_field = fa->k_f_d;
    size_t size = (1+nz*(nx+1)+nx*(nz+1));
    int face = (i+j+k)<0 ? 1 : ny;
    float dy = g->dy;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_edge({1, 1}, {nz+2, nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_edge({1, 1}, {nz+1, nx+2});

    Kokkos::parallel_for("begin_send<YZX>: XZ Edge Loop", xz_edge, KOKKOS_LAMBDA(const int z, const int x) {
//        if(z+x == 2) {
//            sbuf_d(0) = dy;
//        }
        const int y = face;
        sbuf_d(1 + (z-1)*nx + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz);
    });
    Kokkos::parallel_for("begin_send<YZX>: ZX Edge Loop", zx_edge, KOKKOS_LAMBDA(const int z, const int x) {
        const int y = face;
        sbuf_d(1 + (nz+1)*nx + (nx+1)*(z-1) + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx);
    });
    sbuf.modify<Kokkos::DefaultExecutionSpace>();
// CPU
    sbuf.sync<Kokkos::DefaultHostExecutionSpace>();
//    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = dy;
    begin_send_port_k(i, j, k, size*sizeof(float), g, reinterpret_cast<char*>(sbuf_h.data()));
// GPU
//    begin_send_port_k(i, j, k, size*sizeof(float), g, reinterpret_cast<char*>(sbuf_d.data()));

}
template <> void begin_send_kokkos<ZXY>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::DualView<float*>& sbuf) {
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    size_t size = (1+nx*(ny+1)+ny*(nx+1));
    k_field_t& k_field = fa->k_f_d;
    int face = (i+j+k)<0 ? 1 : nz;
    float dz = g->dz;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_edge({1, 1}, {ny+1, nx+2});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> xy_edge({1, 1}, {ny+2, nx+1});

    Kokkos::parallel_for("begin_send<ZXY>: YX Edge Loop", yx_edge, KOKKOS_LAMBDA(const int y, const int x) {
//        if(y+x == 2) {
//            sbuf_d(0) = dz;
//        }
        const int z = face;
        sbuf_d(1 + (nx+1)*(y-1) + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx);
    });
    Kokkos::parallel_for("begin_send<ZXY>: XY Edge Loop", xy_edge, KOKKOS_LAMBDA(const int y, const int x) {
        const int z = face;
        sbuf_d(1 + (nx+1)*ny + (y-1)*nx + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby);
    });
    sbuf.modify<Kokkos::DefaultExecutionSpace>();
// CPU
    sbuf.sync<Kokkos::DefaultHostExecutionSpace>();
    //Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = dz;
    begin_send_port_k(i,j,k,size*sizeof(float), g, reinterpret_cast<char*>(sbuf_h.data()));
// GPU
//    begin_send_port_k(i,j,k,size*sizeof(float), g, reinterpret_cast<char*>(sbuf_d.data()));
}

template <typename T> void begin_send(int i, int j, int k, int nX, int nY, int nZ, field_array_t*  fa, const grid_t* g) {}
template <> void begin_send<XYZ>(int i, int j, int k, int nx, int ny, int nz, field_array_t* field, const grid_t* g) {
    k_field_t k_field = field->k_f_d;
    const size_t size = (1+ny*(nz+1)+nz*(ny+1));
    float* p = static_cast<float*>(size_send_port( i, j, k, size*sizeof(float), g ));

    if( p ) {
        Kokkos::View<float*> d_buf("Device buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);

        int face = (i+j+k)<0 ? 1 : nx;
        float dx = g->dx;

        Kokkos::parallel_for("begin_send<XYZ>: ZY Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (size_t yi) {
                size_t x = face;
                size_t y = yi + 1;
                size_t z = zi + 1;
                d_buf(1 + zi*(ny+1) + yi) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::cby);
            });
        });
        Kokkos::parallel_for("begin_send<XYZ>: YZ Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (size_t yi) {
                const size_t x = face;
                const size_t y = yi + 1;
                const size_t z = zi + 1;
                d_buf(1 + nz*(ny+1) + zi*ny + yi) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::cbz);
            });
        });

        Kokkos::deep_copy(h_buf, d_buf);
        p[0] = dx;
        Kokkos::parallel_for("Copy host to MPI buffer", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(1, size), KOKKOS_LAMBDA(const int idx) {
            p[idx] = h_buf(idx);
        });
        begin_send_port( i, j, k, size*sizeof(float), g );
    }
}
template <> void begin_send<YZX>(int i, int j, int k, int nx, int ny, int nz, field_array_t* field, const grid_t* g) {
    k_field_t k_field = field->k_f_d;
    size_t size = (1+nz*(nx+1)+nx*(nz+1));
    float* p = static_cast<float *>(size_send_port( i, j, k, size*sizeof(float), g ));
    Kokkos::View<float*> d_buf("device buffer", size);
    Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);

    if( p ) {
      int face = (i+j+k)<0 ? 1 : ny;
        float dy = g->dy;
        Kokkos::parallel_for("begin_send<YZX>: XZ Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
            KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                size_t z = zi + 1;
                size_t y = face;
                size_t x = xi + 1;
                d_buf(1 + zi*nx + xi) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz);
            });
        });

        Kokkos::parallel_for("begin_send<YZX>: ZX Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
            KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                size_t x = xi + 1;
                size_t y = face;
                size_t z = zi + 1;
                d_buf(1 + (nz+1)*nx + (nx+1)*zi + xi) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx);
            });
        });

        Kokkos::deep_copy(h_buf,d_buf);

        h_buf(0) = dy;
        p[0] = dy;
        Kokkos::parallel_for("Copy host to MPI buffer", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(1, size), KOKKOS_LAMBDA(const int idx) {
            p[idx] = h_buf(idx);
        });
        begin_send_port( i, j, k, size*sizeof(float), g );
    }
}
template <> void begin_send<ZXY>(int i, int j, int k, int nx, int ny, int nz, field_array_t* field, const grid_t* g) {
    size_t size = (1+nx*(ny+1)+ny*(nx+1));
    float* p = static_cast<float*>(size_send_port(i,j,k,size*sizeof(float),g));
    k_field_t k_field = field->k_f_d;
    Kokkos::View<float*> d_buf("device buffer", size);
    Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);

    if(p){
        int face = (i+j+k)<0 ? 1 : nz;
        float dz = g->dz;
        Kokkos::parallel_for("begin_send<ZXY>: YX Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(ny,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
            size_t yi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                size_t x = xi + 1;
                size_t y = yi + 1;
                size_t z = face;
                d_buf(1 + (nx+1)*yi + xi) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx);
            });
        });
        Kokkos::parallel_for("begin_send<ZXY>: XY Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
            size_t yi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                size_t x = xi + 1;
                size_t y = yi + 1;
                size_t z = face;
                d_buf(1 + (nx+1)*ny + yi*nx + xi) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby);
            });
        });

        Kokkos::deep_copy(h_buf, d_buf);
        h_buf(0) = dz;
        p[0] = dz;
        Kokkos::parallel_for("Copy host to MPI buffer", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(1, size), KOKKOS_LAMBDA(const int idx) {
            p[idx] = h_buf(idx);
        });
        begin_send_port(i,j,k,size*sizeof(float),g);
    }
}


void
kokkos_begin_remote_ghost_tang_b( field_array_t      * RESTRICT fa,
                                  const grid_t *              g,
                                  field_buffers_t&            f_buffers) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;

    begin_recv_kokkos<-1,0,0>(g, f_buffers);
    begin_recv_kokkos<0,-1,0>(g, f_buffers);
    begin_recv_kokkos<0,0,-1>(g, f_buffers);
    begin_recv_kokkos<1,0,0>(g, f_buffers);
    begin_recv_kokkos<0,1,0>(g, f_buffers);
    begin_recv_kokkos<0,0,1>(g, f_buffers);

    begin_send_kokkos<XYZ>(g,fa,-1,0,0,nx,ny,nz, f_buffers.send_buffer[BOUNDARY(-1,0,0)]);
    begin_send_kokkos<YZX>(g,fa,0,-1,0,nx,ny,nz, f_buffers.send_buffer[BOUNDARY(0,-1,0)]);
    begin_send_kokkos<ZXY>(g,fa,0,0,-1,nx,ny,nz, f_buffers.send_buffer[BOUNDARY(0,0,-1)]);
    begin_send_kokkos<XYZ>(g,fa,1,0,0,nx,ny,nz,  f_buffers.send_buffer[BOUNDARY(1,0,0)]);
    begin_send_kokkos<YZX>(g,fa,0,1,0,nx,ny,nz,  f_buffers.send_buffer[BOUNDARY(0,1,0)]);
    begin_send_kokkos<ZXY>(g,fa,0,0,1,nx,ny,nz,  f_buffers.send_buffer[BOUNDARY(0,0,1)]);

}

void
k_begin_remote_ghost_tang_b( field_array_t      * RESTRICT fa,
                           const grid_t *              g) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;

    begin_recv<XYZ>(-1,0,0,nx,ny,nz,g);
    begin_recv<YZX>(0,-1,0,nx,ny,nz,g);
    begin_recv<ZXY>(0,0,-1,nx,ny,nz,g);
    begin_recv<XYZ>(1,0,0,nx,ny,nz,g);
    begin_recv<YZX>(0,1,0,nx,ny,nz,g);
    begin_recv<ZXY>(0,0,1,nx,ny,nz,g);

    begin_send<XYZ>(-1,0,0,nx,ny,nz,fa,g);
    begin_send<YZX>(0,-1,0,nx,ny,nz,fa,g);
    begin_send<ZXY>(0,0,-1,nx,ny,nz,fa,g);
    begin_send<XYZ>(1,0,0,nx,ny,nz,fa,g);
    begin_send<YZX>(0,1,0,nx,ny,nz,fa,g);
    begin_send<ZXY>(0,0,1,nx,ny,nz,fa,g);
}

template<typename T> void end_recv_kokkos(const grid_t* g, field_array_t* RESTRICT field, int i, int j, int k, int nx, int ny, int nz, Kokkos::DualView<float*>& rbuf) {}

template<> void end_recv_kokkos<XYZ>(const grid_t* g, field_array_t* RESTRICT field, int i, int j, int k, int nx, int ny, int nz, Kokkos::DualView<float*>& rbuf) {
    float* p = static_cast<float*>(end_recv_port_k(i,j,k,g));
//    size_t size = 1 + (ny+1)*nz + ny*(nz+1);
    if(p) {
        rbuf.modify_host();
        rbuf.sync_device();
        auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
        auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
//        Kokkos::deep_copy(rbuf_d, rbuf_h);

        k_field_t k_field = field->k_f_d;

        int face = (i+j+k)<0 ? nx+1 : 0;
        float dx = g->dx;

            float lw = rbuf_h(0);
            const float rw = (2.*dx) / (lw + dx);
            lw = (lw - dx)/(lw + dx);

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_edge({1, 1}, {nz+1, ny+2});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yz_edge({1, 1}, {nz+2, ny+1});
        Kokkos::parallel_for("end_recv<XYZ>: ZY Edge loop", zy_edge, KOKKOS_LAMBDA(const int z, const int y) {
//            float lw = rbuf_d(0);
//            const float rw = (2.*dx) / (lw + dx);
//            lw = (lw - dx)/(lw + dx);
            k_field(VOXEL(face,y,z,nx,ny,nz), field_var::cby) = rw*rbuf_d((z-1)*(ny+1) + (y-1) + 1) + lw*k_field(VOXEL(face+i,y+j,z+k,nx,ny,nz), field_var::cby);
        });
        Kokkos::parallel_for("end_recv<XYZ>: YZ Edge loop", yz_edge, KOKKOS_LAMBDA(const int z, const int y) {
//            float lw = rbuf_d(0);
//            const float rw = (2.*dx) / (lw + dx);
//            lw = (lw - dx)/(lw + dx);
            k_field(VOXEL(face,y,z,nx,ny,nz), field_var::cbz) = rw*rbuf_d((ny+1)*nz + (z-1)*ny + (y-1) + 1) + lw*k_field(VOXEL(face+i,y+j,z+k,nx,ny,nz), field_var::cbz);
        });
    }
}
template<> void end_recv_kokkos<YZX>(const grid_t* g, field_array_t* RESTRICT field, int i, int j, int k, int nx, int ny, int nz, Kokkos::DualView<float*>& rbuf) {
   float* p = static_cast<float*>(end_recv_port_k(i,j,k,g));
//    size_t size = 1 + nx*(nz+1) + (nx+1)*nz;
    if(p) {
// CPU
        rbuf.modify_host();
        rbuf.sync_device();
        auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
        auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
//        Kokkos::deep_copy(rbuf_d, rbuf_h);

        k_field_t k_field = field->k_f_d;

        int face = (i+j+k)<0 ? ny+1 : 0;
        float dy = g->dy;

            float lw = rbuf_h(0);
            const float rw = (2.*dy) / (lw+dy);
            lw = (lw-dy)/(lw+dy);

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_edge({1, 1}, {nz+2, nx+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_edge({1, 1}, {nz+1, nx+2});
        Kokkos::parallel_for("end_recv<YZX>: XZ Edge loop", xz_edge, KOKKOS_LAMBDA(const int z, const int x) {
            const int y = face;
//            float lw = rbuf_d(0);
//            const float rw = (2.*dy) / (lw+dy);
//            lw = (lw-dy)/(lw+dy);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = rw*rbuf_d(1 + (z-1)*nx + (x-1)) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cbz);
        });
        Kokkos::parallel_for("end_recv<YZX>: ZX Edge loop", zx_edge, KOKKOS_LAMBDA(const int z, const int x) {
            const int y = face;
//            float lw = rbuf_d(0);
//            const float rw = (2.*dy) / (lw+dy);
//            lw = (lw-dy)/(lw+dy);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = rw*rbuf_d(1 + nx*(nz+1) + (z-1)*(nx+1) + (x-1)) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cbx);
        });
    }
}
template<> void end_recv_kokkos<ZXY>(const grid_t* g, field_array_t* RESTRICT field, int i, int j, int k, int nx, int ny, int nz, Kokkos::DualView<float*>& rbuf) {
    float* p = static_cast<float*>(end_recv_port_k(i,j,k,g));
    if(p) {
        k_field_t k_field = field->k_f_d;
// CPU
        rbuf.modify_host();
        rbuf.sync_device();
        auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
        auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
//        Kokkos::deep_copy(rbuf_d, rbuf_h);

        int face = (i+j+k)<0 ? nz+1 : 0;
        float dz = g->dz;

            float lw = rbuf_h(0);
            const float rw = (2.*dz) / (lw+dz);
            lw = (lw-dz)/(lw+dz);

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_edge({1, 1}, {ny+1, nx+2});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xy_edge({1, 1}, {ny+2, nx+1});
        Kokkos::parallel_for("end_recv<ZXY>: YX Edge loop", yx_edge, KOKKOS_LAMBDA(const int y, const int x) {
            const int z = face;
//            float lw = rbuf_d(0);
//            const float rw = (2.*dz) / (lw+dz);
//            lw = (lw-dz)/(lw+dz);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = rw*rbuf_d(1 + (y-1)*(nx+1) + (x-1)) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cbx);
        });
        Kokkos::parallel_for("end_recv<ZXY>: XY Edge loop", xy_edge, KOKKOS_LAMBDA(const int y, const int x) {
            const int z = face;
//            float lw = rbuf_d(0);
//            const float rw = (2.*dz) / (lw+dz);
//            lw = (lw-dz)/(lw+dz);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = rw*rbuf_d(1 + ny*(nx+1) + (y-1)*nx + (x-1)) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cby);
        });
    }
}

template<typename T> void end_recv(int i, int j, int k, int nx, int ny, int nz, field_array_t* RESTRICT field, const grid_t* g) {}

template<> void end_recv<XYZ>(int i, int j, int k, int nx, int ny, int nz, field_array_t* RESTRICT field, const grid_t* g) {
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

        float lw = h_buf(0);
        float rw = (2.*g->dx) / (lw+g->dx);
        lw = (lw-g->dx)/(lw+g->dx);
        int face = (i+j+k)<0 ? nx+1 : 0;
        Kokkos::parallel_for("end_recv<XYZ>: ZY Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (size_t yi) {
                k_field(VOXEL(face,yi+1,zi+1,nx,ny,nz), field_var::cby) = rw*d_buf(zi*(ny+1) + yi + 1) + lw*k_field(VOXEL(face+i,yi+1+j,zi+1+k,nx,ny,nz), field_var::cby);
            });
        });
        Kokkos::parallel_for("end_recv<XYZ>: YZ Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (size_t yi) {
                k_field(VOXEL(face,yi+1,zi+1,nx,ny,nz), field_var::cbz) = rw*d_buf((ny+1)*nz + zi*ny + yi + 1) + lw*k_field(VOXEL(face+i,yi+1+j,zi+1+k,nx,ny,nz), field_var::cbz);
            });
        });
    }
}
template<> void end_recv<YZX>(int i, int j, int k, int nx, int ny, int nz, field_array_t* RESTRICT field, const grid_t* g) {
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

        float lw = h_buf(0);
        float rw = (2.*g->dy) / (lw+g->dy);
        lw = (lw-g->dy)/(lw+g->dy);
        int face = (i+j+k)<0 ? ny+1 : 0;
        Kokkos::parallel_for("end_recv<YZX>: XZ Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                const size_t x = xi + 1;
                const size_t y = face;
                const size_t z = zi + 1;
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = rw*d_buf(1 + zi*nx + xi) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cbz);
            });
        });
        Kokkos::parallel_for("end_recv<YZX>: ZX Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                const size_t x = xi + 1;
                const size_t y = face;
                const size_t z = zi + 1;
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = rw*d_buf(1 + nx*(nz+1) + zi*(nx+1) + xi) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cbx);
            });
        });
    }
}
template<> void end_recv<ZXY>(int i, int j, int k, int nx, int ny, int nz, field_array_t* RESTRICT field, const grid_t* g) {
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

        float lw = h_buf(0);
        float rw = (2.*g->dz) / (lw+g->dz);
        lw = (lw-g->dz)/(lw+g->dz);
        int face = (i+j+k)<0 ? nz+1 : 0;
        Kokkos::parallel_for("end_recv<ZXY>: YX Edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t yi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                const size_t x = xi + 1;
                const size_t y = yi + 1;
                const size_t z = face;
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = rw*d_buf(1 + yi*(nx+1) + xi) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cbx);
            });
        });
        Kokkos::parallel_for("end_recv<ZXY>: XY Edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t yi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                const size_t x = xi + 1;
                const size_t y = yi + 1;
                const size_t z = face;
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = rw*d_buf(1 + ny*(nx+1) + yi*nx + xi) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cby);
            });
        });
    }
}

// Completely unnecessary, only for symmetry of function calls
template<typename T> void end_send_kokkos(const grid_t* g, int i, int j, int k) {
    end_send_port_k(i,j,k, g);
}

void
k_end_remote_ghost_tang_b( field_array_t      * RESTRICT field,
                         const grid_t *              g) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;

    end_recv<XYZ>(-1,0,0,nx,ny,nz,field,g);
    end_recv<YZX>(0,-1,0,nx,ny,nz,field,g);
    end_recv<ZXY>(0,0,-1,nx,ny,nz,field,g);
    end_recv<XYZ>(1,0,0,nx,ny,nz,field,g);
    end_recv<YZX>(0,1,0,nx,ny,nz,field,g);
    end_recv<ZXY>(0,0,1,nx,ny,nz,field,g);

    end_send_port(-1,0,0,g);
    end_send_port(0,-1,0,g);
    end_send_port(0,0,-1,g);
    end_send_port(1,0,0,g);
    end_send_port(0,1,0,g);
    end_send_port(0,0,1,g);
}

void
kokkos_end_remote_ghost_tang_b( field_array_t      * RESTRICT field,
                         const grid_t *              g ,
                            field_buffers_t&        f_buffers) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;

    end_recv_kokkos<XYZ>(g, field, -1, 0, 0, nx, ny, nz, f_buffers.recv_buffer[BOUNDARY(-1, 0, 0)]);
    end_recv_kokkos<YZX>(g, field, 0, -1, 0, nx, ny, nz, f_buffers.recv_buffer[BOUNDARY(0, -1, 0)]);
    end_recv_kokkos<ZXY>(g, field, 0, 0, -1, nx, ny, nz, f_buffers.recv_buffer[BOUNDARY(0, 0, -1)]);
    end_recv_kokkos<XYZ>(g, field, 1, 0, 0,  nx, ny, nz, f_buffers.recv_buffer[BOUNDARY(1, 0, 0)]);
    end_recv_kokkos<YZX>(g, field, 0, 1, 0,  nx, ny, nz, f_buffers.recv_buffer[BOUNDARY(0, 1, 0)]);
    end_recv_kokkos<ZXY>(g, field, 0, 0, 1,  nx, ny, nz, f_buffers.recv_buffer[BOUNDARY(0, 0, 1)]);

    end_send_kokkos<XYZ>(g, -1,0,0);
    end_send_kokkos<YZX>(g, 0,-1,0);
    end_send_kokkos<ZXY>(g, 0,0,-1);
    end_send_kokkos<XYZ>(g, 1,0,0);
    end_send_kokkos<YZX>(g, 0,1,0);
    end_send_kokkos<ZXY>(g, 0,0,1);
}

template<typename T> void begin_recv_ghost_norm_e_kokkos(const grid_t* g, int i, int j, int k, Kokkos::DualView<float*>& rbuf) {}
template<> void begin_recv_ghost_norm_e_kokkos<XYZ>(const grid_t* g, int i, int j, int k, Kokkos::DualView<float*>& rbuf) {
    const int ny = g->ny, nz = g->nz;
    int size = ( 1 + (ny+1)*(nz+1) )*sizeof(float);
// CPU
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    begin_recv_port_k(i,j,k,size,g, reinterpret_cast<char*>(rbuf_h.data()));
// GPU
//    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
//    begin_recv_port_k(i,j,k,size,g, reinterpret_cast<char*>(rbuf_d.data()));
}
template<> void begin_recv_ghost_norm_e_kokkos<YZX>(const grid_t* g, int i, int j, int k, Kokkos::DualView<float*>& rbuf) {
    const int nx = g->nx, nz = g->nz;
    int size = ( 1 + (nx+1)*(nz+1) )*sizeof(float);
// CPU
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    begin_recv_port_k(i,j,k,size,g, reinterpret_cast<char*>(rbuf_h.data()));
// GPU
//    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
//    begin_recv_port_k(i,j,k,size,g, reinterpret_cast<char*>(rbuf_d.data()));
}
template<> void begin_recv_ghost_norm_e_kokkos<ZXY>(const grid_t* g, int i, int j, int k, Kokkos::DualView<float*>& rbuf) {
    const int nx = g->nx, ny = g->ny;
    int size = ( 1 + (nx+1)*(ny+1) )*sizeof(float);
// CPU
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    begin_recv_port_k(i,j,k,size,g, reinterpret_cast<char*>(rbuf_h.data()));
// GPU
//    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
//    begin_recv_port_k(i,j,k,size,g, reinterpret_cast<char*>(rbuf_d.data()));
}
template<typename T> void begin_send_ghost_norm_e_kokkos(field_array_t* fa, const grid_t* g, int i, int j, int k, Kokkos::DualView<float*>& sbuf) {}
template<> void begin_send_ghost_norm_e_kokkos<XYZ>(field_array_t* fa, const grid_t* g, int i, int j, int k, Kokkos::DualView<float*>& sbuf) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const int size = ( 1 + (ny+1)*(nz+1) )*sizeof(float);
    k_field_t& k_field = fa->k_f_d;
    const int face = (i+j+k)<0 ? 1 : nx;
    const float dx = g->dx;
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_node({1, 1}, {nz+2, ny+2});
    Kokkos::parallel_for("begin_send_ghost_norm_e_kokkos<XYZ>", zy_node, KOKKOS_LAMBDA(const int z, const int y) {
        if(z+y == 2) {
            sbuf_d(0) = dx;
        }
        const int x = face;
        sbuf_d(1 + (z-1)*(ny+1) + (y-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex);
    });
    sbuf.modify_device();
    sbuf.sync_host();
//    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = g->dx;
    begin_send_port_k(i,j,k,size,g,reinterpret_cast<char*>(sbuf_h.data()));
}
template<> void begin_send_ghost_norm_e_kokkos<YZX>(field_array_t* fa, const grid_t* g, int i, int j, int k, Kokkos::DualView<float*>& sbuf) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const int size = ( 1 + (nx+1)*(nz+1) )*sizeof(float);
    k_field_t& k_field = fa->k_f_d;
    const int face = (i+j+k)<0 ? 1 : ny;
    const float dy = g->dy;
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_node({1, 1}, {nz+2, nx+2});
    Kokkos::parallel_for("begin_send_ghost_norm_e_kokkos<YZX>", zx_node, KOKKOS_LAMBDA(const int z, const int x) {
        if(z+x == 2) {
            sbuf_d(0) = dy;
        }
        const int y = face;
        sbuf_d(1 + (z-1)*(nx+1) + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey);
    });
    sbuf.modify_device();
    sbuf.sync_host();
//    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = g->dy;
    begin_send_port_k(i,j,k,size,g,reinterpret_cast<char*>(sbuf_h.data()));

}
template<> void begin_send_ghost_norm_e_kokkos<ZXY>(field_array_t* fa, const grid_t* g, int i, int j, int k, Kokkos::DualView<float*>& sbuf) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const int size = ( 1 + (nx+1)*(ny+1) )*sizeof(float);
    k_field_t& k_field = fa->k_f_d;
    const int face = (i+j+k)<0 ? 1 : nz;
    const float dz = g->dz;
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_node({1, 1}, {ny+2, nx+2});
    Kokkos::parallel_for("begin_send_ghost_norm_e_kokkos<ZXY>", yx_node, KOKKOS_LAMBDA(const int y, const int x) {
        if(y+x == 2) {
            sbuf_d(0) = dz;
        }
        const int z = face;
        sbuf_d(1 + (y-1)*(nx+1) + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez);
    });
    sbuf.modify_device();
    sbuf.sync_host();
//    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = g->dz;
    begin_send_port_k(i,j,k,size,g,reinterpret_cast<char*>(sbuf_h.data()));
}

template<typename T> void begin_recv_ghost_norm_e(const grid_t* g, int i, int j, int k) {}
template<> void begin_recv_ghost_norm_e<XYZ>(const grid_t* g, int i, int j, int k) {
    const int ny = g->ny, nz = g->nz;
    int size = ( 1 + (ny+1)*(nz+1) )*sizeof(float);
    begin_recv_port(i,j,k,size,g);
}
template<> void begin_recv_ghost_norm_e<YZX>(const grid_t* g, int i, int j, int k) {
    const int nx = g->nx, nz = g->nz;
    int size = ( 1 + (nx+1)*(nz+1) )*sizeof(float);
    begin_recv_port(i,j,k,size,g);
}
template<> void begin_recv_ghost_norm_e<ZXY>(const grid_t* g, int i, int j, int k) {
    const int nx = g->nx, ny = g->ny;
    int size = ( 1 + (nx+1)*(ny+1) )*sizeof(float);
    begin_recv_port(i,j,k,size,g);
}
template<typename T> void begin_send_ghost_norm_e(field_array_t* fa, const grid_t* g, int i, int j, int k) {}
template<> void begin_send_ghost_norm_e<XYZ>(field_array_t* fa, const grid_t* g, int i, int j, int k) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    int size = ( 1 + (ny+1)*(nz+1) )*sizeof(float);
    float* p = reinterpret_cast<float*>(size_send_port(i,j,k,size,g));
    if(p) {
        Kokkos::View<float*> d_buf("Device buffer", size/sizeof(float));
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        k_field_t& k_field = fa->k_f_d;
        int face = (i+j+k)<0 ? 1 : nx;
        Kokkos::parallel_for("begin_send_ghost_norm_e<XYZ>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            const int zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const int yi) {
                const int x = face;
                const int y = yi + 1;
                const int z = zi + 1;
                d_buf(1 + zi*(ny+1) + yi) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex);
            });
        });
        Kokkos::deep_copy(h_buf, d_buf);
        h_buf(0) = g->dx;
        Kokkos::parallel_for("Copy host to mpi buffer", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size/sizeof(float)),
        KOKKOS_LAMBDA(const int idx) {
            p[idx] = h_buf(idx);
        });
        p[0] = g->dx;
        begin_send_port(i,j,k,size,g);
    }
}
template<> void begin_send_ghost_norm_e<YZX>(field_array_t* fa, const grid_t* g, int i, int j, int k) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    int size = ( 1 + (nx+1)*(nz+1) )*sizeof(float);
    float* p = reinterpret_cast<float*>(size_send_port(i,j,k,size,g));
    if(p) {
        Kokkos::View<float*> d_buf("Device buffer", size/sizeof(float));
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        k_field_t& k_field = fa->k_f_d;
        int face = (i+j+k)<0 ? 1 : ny;
        Kokkos::parallel_for("begin_send_ghost_norm_e<XYZ>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            const int zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                const int x = xi + 1;
                const int y = face;
                const int z = zi + 1;
                d_buf(1 + zi*(nx+1) + xi) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey);
            });
        });
        Kokkos::deep_copy(h_buf, d_buf);
        h_buf(0) = g->dy;
        Kokkos::parallel_for("Copy host to mpi buffer", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size/sizeof(float)),
        KOKKOS_LAMBDA(const int idx) {
            p[idx] = h_buf(idx);
        });
        begin_send_port(i,j,k,size,g);
    }
}
template<> void begin_send_ghost_norm_e<ZXY>(field_array_t* fa, const grid_t* g, int i, int j, int k) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    int size = ( 1 + (nx+1)*(ny+1) )*sizeof(float);
    float* p = reinterpret_cast<float*>(size_send_port(i,j,k,size,g));
    if(p) {
        Kokkos::View<float*> d_buf("Device buffer", size/sizeof(float));
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        k_field_t& k_field = fa->k_f_d;
        int face = (i+j+k)<0 ? 1 : nz;
        Kokkos::parallel_for("begin_send_ghost_norm_e<XYZ>", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            const int yi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                const int x = xi + 1;
                const int y = yi + 1;
                const int z = face;
                d_buf(1 + yi*(nx+1) + xi) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez);
            });
        });
        Kokkos::deep_copy(h_buf, d_buf);
        h_buf(0) = g->dz;
        Kokkos::parallel_for("Copy host to mpi buffer", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size/sizeof(float)),
        KOKKOS_LAMBDA(const int idx) {
            p[idx] = h_buf(idx);
        });
        begin_send_port(i,j,k,size,g);
    }
}

void
kokkos_begin_remote_ghost_norm_e( field_array_t      * ALIGNED(128) field,
                                  const grid_t *              g,
                                  field_buffers_t&            f_buffers) {
    begin_recv_ghost_norm_e_kokkos<XYZ>(g, -1,  0,  0, f_buffers.recv_buffer[BOUNDARY(-1,  0,  0)]);
    begin_recv_ghost_norm_e_kokkos<YZX>(g,  0, -1,  0, f_buffers.recv_buffer[BOUNDARY( 0, -1,  0)]);
    begin_recv_ghost_norm_e_kokkos<ZXY>(g,  0,  0, -1, f_buffers.recv_buffer[BOUNDARY( 0,  0, -1)]);
    begin_recv_ghost_norm_e_kokkos<XYZ>(g,  1,  0,  0, f_buffers.recv_buffer[BOUNDARY( 1,  0,  0)]);
    begin_recv_ghost_norm_e_kokkos<YZX>(g,  0,  1,  0, f_buffers.recv_buffer[BOUNDARY( 0,  1,  0)]);
    begin_recv_ghost_norm_e_kokkos<ZXY>(g,  0,  0,  1, f_buffers.recv_buffer[BOUNDARY( 0,  0,  1)]);

    begin_send_ghost_norm_e_kokkos<XYZ>(field, g, -1,  0,  0, f_buffers.send_buffer[BOUNDARY(-1,  0,  0)]);
    begin_send_ghost_norm_e_kokkos<YZX>(field, g,  0, -1,  0, f_buffers.send_buffer[BOUNDARY( 0, -1,  0)]);
    begin_send_ghost_norm_e_kokkos<ZXY>(field, g,  0,  0, -1, f_buffers.send_buffer[BOUNDARY( 0,  0, -1)]);
    begin_send_ghost_norm_e_kokkos<XYZ>(field, g,  1,  0,  0, f_buffers.send_buffer[BOUNDARY( 1,  0,  0)]);
    begin_send_ghost_norm_e_kokkos<YZX>(field, g,  0,  1,  0, f_buffers.send_buffer[BOUNDARY( 0,  1,  0)]);
    begin_send_ghost_norm_e_kokkos<ZXY>(field, g,  0,  0,  1, f_buffers.send_buffer[BOUNDARY( 0,  0,  1)]);
}

void
k_begin_remote_ghost_norm_e( field_array_t      * ALIGNED(128) field,
                           const grid_t *              g ) {
    begin_recv_ghost_norm_e<XYZ>(g, -1,  0,  0);
    begin_recv_ghost_norm_e<YZX>(g,  0, -1,  0);
    begin_recv_ghost_norm_e<ZXY>(g,  0,  0, -1);
    begin_recv_ghost_norm_e<XYZ>(g, 1, 0, 0);
    begin_recv_ghost_norm_e<YZX>(g, 0, 1, 0);
    begin_recv_ghost_norm_e<ZXY>(g, 0, 0, 1);

    begin_send_ghost_norm_e<XYZ>(field, g, -1,  0,  0);
    begin_send_ghost_norm_e<YZX>(field, g,  0, -1,  0);
    begin_send_ghost_norm_e<ZXY>(field, g,  0,  0, -1);
    begin_send_ghost_norm_e<XYZ>(field, g, 1, 0, 0);
    begin_send_ghost_norm_e<YZX>(field, g, 0, 1, 0);
    begin_send_ghost_norm_e<ZXY>(field, g, 0, 0, 1);
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

template<typename T> void end_recv_ghost_norm_e_kokkos(field_array_t* fa, const grid_t* g, const int i, const int j, const int k, Kokkos::DualView<float*>& rbuf) {}
template<> void end_recv_ghost_norm_e_kokkos<XYZ>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k, Kokkos::DualView<float*>& rbuf) {
    float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    if(p) {
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float lw = rbuf_h(0);
        float rw = (2.*g->dx)/(lw+g->dx);
        lw = (lw-g->dx)/(lw+g->dx);
        int face = (i+j+k)<0 ? nx+1 : 0;
        //int size = 1 + (ny+1)*(nz+1);
        k_field_t& k_field = fa->k_f_d;
        rbuf.modify_host();
        rbuf.sync_device();
//        Kokkos::deep_copy(rbuf_d, rbuf_h);

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_node({1, 1}, {nz+2, ny+2});
        Kokkos::parallel_for("begin_send_ghost_norm_e_kokkos<XYZ>", zy_node, KOKKOS_LAMBDA(const int z, const int y) {
            const int x = face;
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = rw*rbuf_d(1 + (z-1)*(ny+1) + (y-1)) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::ex);
        });
    }
}
template<> void end_recv_ghost_norm_e_kokkos<YZX>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k, Kokkos::DualView<float*>& rbuf) {
    float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    if(p) {
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float lw = rbuf_h(0);
        float rw = (2.*g->dy)/(lw+g->dy);
        lw = (lw-g->dy)/(lw+g->dy);
        int face = (i+j+k)<0 ? ny+1 : 0;
        //int size = 1 + (nx+1)*(nz+1);
        k_field_t& k_field = fa->k_f_d;

        rbuf.modify_host();
        rbuf.sync_device();
//        Kokkos::deep_copy(rbuf_d, rbuf_h);

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_node({1, 1}, {nz+2, nx+2});
        Kokkos::parallel_for("begin_send_ghost_norm_e_kokkos<YZX>", zx_node, KOKKOS_LAMBDA(const int z, const int x) {
            const int y = face;
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = rw*rbuf_d(1 + (z-1)*(nx+1) + (x-1)) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::ey);
        });
    }
}
template<> void end_recv_ghost_norm_e_kokkos<ZXY>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k, Kokkos::DualView<float*>& rbuf) {
    float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    if(p) {
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float lw = rbuf_h(0);
        float rw = (2.*g->dz)/(lw+g->dz);
        lw = (lw-g->dz)/(lw+g->dz);
        int face = (i+j+k)<0 ? nz+1 : 0;
        //int size = 1 + (nx+1)*(ny+1);
        k_field_t& k_field = fa->k_f_d;

        rbuf.modify_host();
        rbuf.sync_device();
//        Kokkos::deep_copy(rbuf_d, rbuf_h);

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_node({1, 1}, {ny+2, nx+2});
        Kokkos::parallel_for("begin_send_ghost_norm_e_kokkos<ZXY>", yx_node, KOKKOS_LAMBDA(const int y, const int x) {
            const int z = face;
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez) = rw*rbuf_d(1 + (y-1)*(nx+1) + (x-1)) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::ez);
        });
    }
}
template<typename T> void end_send_ghost_norm_e_kokkos(const grid_t* g, const int i, const int j, const int k) {
    end_send_port_k(i,j,k,g);
}

template<typename T> void end_recv_ghost_norm_e(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {}
template<> void end_recv_ghost_norm_e<XYZ>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {
    float* p = reinterpret_cast<float*>(end_recv_port(i,j,k,g));
    if(p) {
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float lw = p[0];
        float rw = (2.*g->dx)/(lw+g->dx);
        lw = (lw-g->dx)/(lw+g->dx);
        int face = (i+j+k)<0 ? nx+1 : 0;
        int size = 1 + (ny+1)*(nz+1);
        k_field_t& k_field = fa->k_f_d;

        Kokkos::View<float*> d_buf("Device buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        Kokkos::parallel_for("Copy mpi buffer to host", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size),
        KOKKOS_LAMBDA(const int idx) {
            h_buf(idx) = p[idx];
        });
        Kokkos::deep_copy(d_buf, h_buf);

        Kokkos::parallel_for("begin_send_ghost_norm_e<XYZ>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            const int zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const int yi) {
                const int x = face;
                const int y = yi + 1;
                const int z = zi + 1;
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = rw*d_buf(1 + zi*(ny+1) + yi) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::ex);
            });
        });
    }
}
template<> void end_recv_ghost_norm_e<YZX>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {
    float* p = reinterpret_cast<float*>(end_recv_port(i,j,k,g));
    if(p) {
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float lw = p[0];
        float rw = (2.*g->dy)/(lw+g->dy);
        lw = (lw-g->dy)/(lw+g->dy);
        int face = (i+j+k)<0 ? ny+1 : 0;
        int size = 1 + (nx+1)*(nz+1);
        k_field_t& k_field = fa->k_f_d;
        Kokkos::View<float*> d_buf("Device buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        Kokkos::parallel_for("Copy mpi buffer to host", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size),
        KOKKOS_LAMBDA(const int idx) {
            h_buf(idx) = p[idx];
        });
        Kokkos::deep_copy(d_buf, h_buf);

        Kokkos::parallel_for("begin_send_ghost_norm_e<YZX>", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            const int zi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                const int x = xi + 1;
                const int y = face;
                const int z = zi + 1;
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = rw*d_buf(1 + zi*(nx+1) + xi) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::ey);
            });
        });
    }
}
template<> void end_recv_ghost_norm_e<ZXY>(field_array_t* fa, const grid_t* g, const int i, const int j, const int k) {
    float* p = reinterpret_cast<float*>(end_recv_port(i,j,k,g));
    if(p) {
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float lw = p[0];
        float rw = (2.*g->dz)/(lw+g->dz);
        lw = (lw-g->dz)/(lw+g->dz);
        int face = (i+j+k)<0 ? nz+1 : 0;
        int size = 1 + (nx+1)*(ny+1);
        k_field_t& k_field = fa->k_f_d;
        Kokkos::View<float*> d_buf("Device buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        Kokkos::parallel_for("Copy mpi buffer to host", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size),
        KOKKOS_LAMBDA(const int idx) {
            h_buf(idx) = p[idx];
        });
        Kokkos::deep_copy(d_buf, h_buf);

        Kokkos::parallel_for("begin_send_ghost_norm_e<XYZ>", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            const int yi = team_member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                const int x = xi + 1;
                const int y = yi + 1;
                const int z = face;
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez) = rw*d_buf(1 + yi*(nx+1) + xi) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::ez);
            });
        });
    }
}
template<typename T> void end_send_ghost_norm_e(const grid_t* g, const int i, const int j, const int k) {
    end_send_port(i,j,k,g);
}

void
kokkos_end_remote_ghost_norm_e( field_array_t      * ALIGNED(128) field,
                         const grid_t *              g,
                            field_buffers_t&            f_buffers) {

    end_recv_ghost_norm_e_kokkos<XYZ>(field, g, -1,  0,  0, f_buffers.recv_buffer[BOUNDARY(-1, 0, 0)]);
    end_recv_ghost_norm_e_kokkos<YZX>(field, g,  0, -1,  0, f_buffers.recv_buffer[BOUNDARY( 0,-1, 0)]);
    end_recv_ghost_norm_e_kokkos<ZXY>(field, g,  0,  0, -1, f_buffers.recv_buffer[BOUNDARY( 0, 0,-1)]);
    end_recv_ghost_norm_e_kokkos<XYZ>(field, g,  1,  0,  0, f_buffers.recv_buffer[BOUNDARY( 1, 0, 0)]);
    end_recv_ghost_norm_e_kokkos<YZX>(field, g,  0,  1,  0, f_buffers.recv_buffer[BOUNDARY( 0, 1, 0)]);
    end_recv_ghost_norm_e_kokkos<ZXY>(field, g,  0,  0,  1, f_buffers.recv_buffer[BOUNDARY( 0, 0, 1)]);

    end_send_ghost_norm_e_kokkos<XYZ>(g, -1,  0,  0);
    end_send_ghost_norm_e_kokkos<YZX>(g,  0, -1,  0);
    end_send_ghost_norm_e_kokkos<ZXY>(g,  0,  0, -1);
    end_send_ghost_norm_e_kokkos<XYZ>(g, 1, 0, 0);
    end_send_ghost_norm_e_kokkos<YZX>(g, 0, 1, 0);
    end_send_ghost_norm_e_kokkos<ZXY>(g, 0, 0, 1);
}
void
k_end_remote_ghost_norm_e( field_array_t      * ALIGNED(128) field,
                         const grid_t *              g ) {

    end_recv_ghost_norm_e<XYZ>(field, g, -1,  0,  0);
    end_recv_ghost_norm_e<YZX>(field, g,  0, -1,  0);
    end_recv_ghost_norm_e<ZXY>(field, g,  0,  0, -1);
    end_recv_ghost_norm_e<XYZ>(field, g, 1, 0, 0);
    end_recv_ghost_norm_e<YZX>(field, g, 0, 1, 0);
    end_recv_ghost_norm_e<ZXY>(field, g, 0, 0, 1);

    end_send_ghost_norm_e<XYZ>(g, -1,  0,  0);
    end_send_ghost_norm_e<YZX>(g,  0, -1,  0);
    end_send_ghost_norm_e<ZXY>(g,  0,  0, -1);
    end_send_ghost_norm_e<XYZ>(g, 1, 0, 0);
    end_send_ghost_norm_e<YZX>(g, 0, 1, 0);
    end_send_ghost_norm_e<ZXY>(g, 0, 0, 1);
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

template<typename Face> 
void 
begin_recv_ghost_div_b(field_array* fa, const int i, const int j, const int k) {
  field_buffers_t* fb = fa->fb;
  const int nx = fa->g->nx, ny = fa->g->ny, nz=fa->g->nz;
  int size;
  if constexpr (std::is_same<Face,XYZ>::value) {
    size = (1 + ny*nz)*sizeof(float);
  } else if constexpr (std::is_same<Face,YZX>::value) {
    size = (1 + nz*nx)*sizeof(float);
  } else if constexpr (std::is_same<Face,ZXY>::value) {
    size = (1 + nx*ny)*sizeof(float);
  }
  Kokkos::DualView<float*> rbuf = fb->recv_buffer[BOUNDARY(i,j,k)];;
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  begin_recv_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(rbuf_h.data()));
}
//template<> void begin_recv_ghost_div_b<XYZ>(field_array* fa, const int i, const int j, const int k, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const int ny = fa->g->ny, nz=fa->g->nz;
//    const int size = (1 + ny*nz)*sizeof(float);
//    begin_recv_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(rbuf_h.data()));
//}
//template<> void begin_recv_ghost_div_b<YZX>(field_array* fa, const int i, const int j, const int k, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const int nx = fa->g->nx, nz=fa->g->nz;
//    const int size = (1 + nz*nx)*sizeof(float);
//    begin_recv_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(rbuf_h.data()));
//}
//template<> void begin_recv_ghost_div_b<ZXY>(field_array* fa, const int i, const int j, const int k, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const int nx = fa->g->nx, ny = fa->g->ny;
//    const int size = (1 + nx*ny)*sizeof(float);
//    begin_recv_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(rbuf_h.data()));
//}

template<typename T> 
void 
begin_send_ghost_div_b(field_array* fa, const int i, const int j, const int k) {}

template<> 
void 
begin_send_ghost_div_b<XYZ>(field_array* fa, const int i, const int j, const int k) {
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();

    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int size = (1 + ny*nz)*sizeof(float);
    const int face = (i+j+k)<0 ? 1 : nx;
    const k_field_t& k_field = fa->k_f_d;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_face({1, 1}, {nz+1, ny+1});
    Kokkos::parallel_for("begin_send_ghost_div_b<XYZ>", x_face, KOKKOS_LAMBDA(const int z, const int y) {
        const int x = face;
        sbuf_d(1 + (z-1)*ny + (y-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err);
    });
    sbuf.modify_device();
    sbuf.sync_host();
//    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = fa->g->dx;
    begin_send_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(sbuf_h.data()));
}
template<> 
void 
begin_send_ghost_div_b<YZX>(field_array* fa, const int i, const int j, const int k) {
    auto fb = fa->fb;
    Kokkos::DualView<float*> sbuf = fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();

    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int size = (1 + nx*nz)*sizeof(float);
    const int face = (i+j+k) < 0 ? 1 : ny;
    const k_field_t& k_field = fa->k_f_d;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> y_face({1, 1}, {nz+1, nx+1});
    Kokkos::parallel_for("begin_send_ghost_div_b<YZX>", y_face, KOKKOS_LAMBDA(const int z, const int x) {
        const int y = face;
        sbuf_d(1 + (z-1)*nx + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err);
    });
    sbuf.modify_device();
    sbuf.sync_host();
//    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = fa->g->dy;
    begin_send_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(sbuf_h.data()));
}
template<> 
void 
begin_send_ghost_div_b<ZXY>(field_array* fa, const int i, const int j, const int k) {
    auto fb = fa->fb;
    Kokkos::DualView<float*> sbuf = fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();

    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int size = (1 + nx*ny)*sizeof(float);
    const int face = (i+j+k) < 0 ? 1 : nz;
    const k_field_t& k_field = fa->k_f_d;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> z_face({1, 1}, {ny+1, nx+1});
    Kokkos::parallel_for("begin_send_ghost_div_b<ZXY>", z_face, KOKKOS_LAMBDA(const int y, const int x) {
        const int z = face;
        sbuf_d(1 + (y-1)*nx + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err);
    });
    sbuf.modify_device();
    sbuf.sync_host();
//    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = fa->g->dz;
    begin_send_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(sbuf_h.data()));
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

void k_begin_remote_ghost_div_b(field_array_t* ALIGNED(128) fa, const grid_t* g, field_buffers_t& fb) {
// Start receiving
    begin_recv_ghost_div_b<XYZ>(fa, -1,  0,  0);
    begin_recv_ghost_div_b<YZX>(fa,  0, -1,  0);
    begin_recv_ghost_div_b<ZXY>(fa,  0,  0, -1);

    begin_recv_ghost_div_b<XYZ>(fa,  1,  0,  0);
    begin_recv_ghost_div_b<YZX>(fa,  0,  1,  0);
    begin_recv_ghost_div_b<ZXY>(fa,  0,  0,  1);

// Start sending
    begin_send_ghost_div_b<XYZ>(fa, -1,  0,  0);
    begin_send_ghost_div_b<YZX>(fa,  0, -1,  0);
    begin_send_ghost_div_b<ZXY>(fa,  0,  0, -1);

    begin_send_ghost_div_b<XYZ>(fa,  1,  0,  0);
    begin_send_ghost_div_b<YZX>(fa,  0,  1,  0);
    begin_send_ghost_div_b<ZXY>(fa,  0,  0,  1);
}

template<typename Face> 
void 
end_recv_ghost_div_b(field_array_t* fa, const int i, const int j, const int k) {
  int face;
  const grid_t* g = fa->g;
  field_buffers_t* fb = fa->fb;
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    Kokkos::DualView<float*> rbuf = fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    const k_field_t& k_field = fa->k_f_d;
    const int nx = g->nx, ny = g->ny, nz = g->nz;

# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                               \
    face = (i+j+k)<0 ? 1 : n##X;			                                         \
    float lw = rbuf_h(0);                                                      \
    float rw = (2. * g->d##X) / (lw + g->d##X);                                \
    lw = (lw - g->d##X) / (lw + g->d##X);                                      \
    rbuf.modify_host();                                                        \
    rbuf.sync_device();                                                        \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_face({1, 1}, {n##Z+1, n##Y+1}); \
    Kokkos::parallel_for("end_recv_ghost_div_b<X##Y##Z>", X##_face,            \
    KOKKOS_LAMBDA(const int Z, const int Y) {                                  \
        const int X = face;                                                    \
        k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = rw * rbuf_d(1 + (Z-1)*n##Y + (Y-1)) + lw * k_field(VOXEL(x+i, y+j, z+k, nx, ny, nz), field_var::div_b_err); \
    });                                                                        \
  } END_PRIMITIVE

    if constexpr (std::is_same<Face, XYZ>::value) {
      END_RECV(i,j,k,x,y,z);
    } else if constexpr (std::is_same<Face, XYZ>::value) {
      END_RECV(i,j,k,y,z,x);
    } else if constexpr (std::is_same<Face, XYZ>::value) {
      END_RECV(i,j,k,z,x,y);
    }
  }
#undef END_RECV
}

//template<> void end_recv_ghost_div_b<XYZ>(field_array_t* fa, const int i, const int j, const int k, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const grid_t* g = fa->g;
//    float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
//    if(p) {
//        const int nx = g->nx, ny = g->ny, nz = g->nz;
//        const int face = (i+j+k) < 0 ? nx+1 : 0;
//        float lw = rbuf_h(0);
//        float rw = (2. * g->dx) / (lw + g->dx);
//        lw = (lw - g->dx) / (lw + g->dx);
//        const k_field_t& k_field = fa->k_f_d;
//        Kokkos::deep_copy(rbuf_d, rbuf_h);
//        Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_face({1, 1}, {nz+1, ny+1});
//        Kokkos::parallel_for("end_recv_ghost_div_b<XYZ>", x_face, KOKKOS_LAMBDA(const int z, const int y) {
//            const int x = face;
//            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = rw * rbuf_d(1 + (z-1)*ny + (y-1)) + lw * k_field(VOXEL(x+i, y+j, z+k, nx, ny, nz), field_var::div_b_err);
//        });
//    }
//}
//template<> void end_recv_ghost_div_b<YZX>(field_array_t* fa, const int i, const int j, const int k, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const grid_t* g = fa->g;
//    float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
//    if(p) {
//        const int nx = g->nx, ny = g->ny, nz = g->nz;
//        const int face = (i+j+k) < 0 ? ny+1 : 0;
//        float lw = rbuf_h(0);
//        float rw = (2. * g->dy) / (lw + g->dy);
//        lw = (lw - g->dy) / (lw + g->dy);
//        const k_field_t& k_field = fa->k_f_d;
//        Kokkos::deep_copy(rbuf_d, rbuf_h);
//        Kokkos::MDRangePolicy<Kokkos::Rank<2>> y_face({1, 1}, {nz+1, nx+1});
//        Kokkos::parallel_for("end_recv_ghost_div_b<XYZ>", y_face, KOKKOS_LAMBDA(const int z, const int x) {
//            const int y = face;
//            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = rw * rbuf_d(1 + (z-1)*nx + (x-1)) + lw * k_field(VOXEL(x+i, y+j, z+k, nx, ny, nz), field_var::div_b_err);
//        });
//    }
//}
//template<> void end_recv_ghost_div_b<ZXY>(field_array_t* fa, const int i, const int j, const int k, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const grid_t* g = fa->g;
//    float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
//    if(p) {
//        const int nx = g->nx, ny = g->ny, nz = g->nz;
//        const int face = (i+j+k) < 0 ? nz+1 : 0;
//        float lw = rbuf_h(0);
//        float rw = (2. * g->dz) / (lw + g->dz);
//        lw = (lw - g->dz) / (lw + g->dz);
//        const k_field_t& k_field = fa->k_f_d;
//        Kokkos::deep_copy(rbuf_d, rbuf_h);
//        Kokkos::MDRangePolicy<Kokkos::Rank<2>> z_face({1, 1}, {ny+1, nx+1});
//        Kokkos::parallel_for("end_recv_ghost_div_b<XYZ>", z_face, KOKKOS_LAMBDA(const int y, const int x) {
//            const int z = face;
//            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = rw * rbuf_d(1 + (y-1)*nx + (x-1)) + lw * k_field(VOXEL(x+i, y+j, z+k, nx, ny, nz), field_var::div_b_err);
//        });
//    }
//}

template<typename T> 
void 
end_send_ghost_div_b(field_array_t* fa, const int i, const int j, const int k) {
  end_send_port_k(i,j,k,fa->g);
}

void k_end_remote_ghost_div_b(field_array_t* ALIGNED(128) fa, const grid_t* g) {
// End receiving
  end_recv_ghost_div_b<XYZ>(fa, -1,  0,  0);
  end_recv_ghost_div_b<YZX>(fa,  0, -1,  0);
  end_recv_ghost_div_b<ZXY>(fa,  0,  0, -1);

  end_recv_ghost_div_b<XYZ>(fa,  1,  0,  0);
  end_recv_ghost_div_b<YZX>(fa,  0,  1,  0);
  end_recv_ghost_div_b<ZXY>(fa,  0,  0,  1);

// End sending
  end_send_ghost_div_b<XYZ>(fa, -1,  0,  0);
  end_send_ghost_div_b<YZX>(fa,  0, -1,  0);
  end_send_ghost_div_b<ZXY>(fa,  0,  0, -1);

  end_send_ghost_div_b<XYZ>(fa,  1,  0,  0);
  end_send_ghost_div_b<YZX>(fa,  0,  1,  0);
  end_send_ghost_div_b<ZXY>(fa,  0,  0,  1);
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

/*
***** Hybrid 
*/

/**
 * @brief Calculate the size of the local grid face along an axis
 *
 * @tparam i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) ex. Pos x-face: 1,0,0
 * @param[in] nx,ny,nz Local number of cells along each dimension
 * @return Number of cells in the face
 */
template<int i, int j, int k>
int 
get_face_size(const int nx, const int ny, const int nz) {
  if constexpr (i != 0 && j == 0 && k == 0) {
    return ny*nz;
  } else if constexpr (i == 0 && j != 0 && k == 0) {
    return nx*nz;
  } else if constexpr (i == 0 && j == 0 && k != 0) {
    return nx*ny;
  }
}

/**
 * @brief Helper function for getting the receive DualView communication buffer
 * for a specific face
 *
 * @param[in] i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) (Pos. x-dir: 1,0,0)
 * @return Reference to the DualView for buffering MPI communication
 */
inline 
Kokkos::DualView<float*>& 
get_recv_dualview(field_array *fa, const int i, const int j, const int k) {
  return fa->fb->recv_buffer[BOUNDARY(i,j,k)];
}

/**
 * @brief Helper function for getting the sending DualView communication buffer 
 * for a specific face
 *
 * @param[in] i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) (Pos. x-dir: 1,0,0)
 * @return Reference to the DualView for buffering MPI communication
 */
inline 
Kokkos::DualView<float*>& 
get_send_dualview(field_array *fa, const int i, const int j, const int k) {
  return fa->fb->send_buffer[BOUNDARY(i,j,k)];
}

/**
 * @brief Helper function for getting the raw communication buffer
 *
 * Returns either the Host or Device buffer depending on whether GPU aware MPI
 * is enabled or not.
 *
 * @param[inout] dual_view Communication send or recv buffer 
 * @return Raw pointer to buffer
 */
float*
get_comm_buffer(const Kokkos::DualView<float*>& dual_view) {
#ifdef VPIC_ENABLE_GPU_AWARE_MPI
  return dual_view.view<Kokkos::DefaultExecutionSpace>().data();
#else
  return dual_view.view<Kokkos::DefaultHostExecutionSpace>().data();
#endif
}

/**
 * @brief Helper function for synchronizing communication buffers in DualViews
 *
 * Automatically syncs buffers with regular MPI and clears sync flags when 
 * using GPU Aware MPI.
 *
 * @tparam ExecutionSpace Specific execution space that needs to be synced
 * @param[inout] dual_view Communication send or recv buffer 
 */
template<class ExecutionSpace> 
void 
sync_comm_buffer(Kokkos::DualView<float*>& view) {
#ifdef VPIC_ENABLE_GPU_AWARE_MPI
  view.clear_sync_state();
#else
  view.sync<ExecutionSpace>();
#endif
}

/**
 * \def COPY_FACE(x_,y_,z_,fields,i,j,k)
 * Copy a field face to it's opposite ghost face with a parallel_for kernel. 
 * DO NOT USE DIRECTLY! Use the wrapper function for clearer error messages
 * and profiling.
 */
#define COPY_FACE(x_,y_,z_,fields,i,j,k)                                       \
  const int x_##src = (i+j+k) < 0 ? 1 : n##x_;                                 \
  const int x_##dst = (i+j+k) < 0 ? n##x_+1 : 0;                               \
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> face_pol(fa->ghost_comm_space,{1,1,0},\
                                                     {n##y_+1,n##z_+1,nvar});  \
  Kokkos::parallel_for("copy_face_2_ghost<" #x_ #y_ #z_ ">", face_pol,         \
    KOKKOS_LAMBDA(const int y_##src, const int z_##src, const int v) {         \
      const int y_##dst = y_##src;                                             \
      const int z_##dst = z_##src;                                             \
      fields(VOXEL(xdst,ydst,zdst,nx,ny,nz), beg_var+v)                        \
        = fields(VOXEL(xsrc,ysrc,zsrc,nx,ny,nz), beg_var+v);                   \
    });									                            

/**
 * @brief Copy face to ghost cells of opposite face
 *
 * Periodic boundaries may require copying a face from one processor to itself.
 * Going through MPI communication is unnecessary so this helper function 
 * detects when the destination process is itself and performs the copy 
 * directly. Copies are submitted to a separate Stream/ExecutionSpace instance
 * to avoid blocking/slowing the packing kernels.
 *
 * @tparam i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) ex. Pos x-face: 1,0,0
 * @param fa[inout] Pointer to field_array_t structure
 * @param beg_var,end_var[in] Range of field variables to copy [beg_var,end_var)
 */
template<int i, int j, int k>
void 
copy_face(field_array* fa, const int beg_var, const int end_var) {

  int dst = fa->g->bc[BOUNDARY(i,j,k)]; 
  if( 0 <= dst && dst < world_size ) { 
    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int nvar = end_var - beg_var;
    const k_field_t& fields = fa->k_f_d;
    if( dst == world_rank ) {
      if constexpr (i != 0 && j == 0 && k == 0) {
        COPY_FACE(x,y,z,fields,i,j,k);
      } else if constexpr (i == 0 && j != 0 && k == 0) {
        COPY_FACE(y,z,x,fields,i,j,k);
      } else if constexpr (i == 0 && j == 0 && k != 0) {
        COPY_FACE(z,x,y,fields,i,j,k);
      }
    }
  }
}

#undef COPY_FACE

/**
 * \def PACK_FACE(x_,y_,z_,buffer,i,j,k)
 * Copy a field face to the send buffer with a parallel_for kernel. 
 * DO NOT USE DIRECTLY! Use the wrapper function for clearer error messages
 * and profiling.
 */
#define PACK_FACE(x_,y_,z_,sbuf,i,j,k)                                    \
  const int x_ = (i+j+k)<0 ? 1 : n##x_;                                        \
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> face_policy({1,1,0},                  \
                                                     {n##y_+1,n##z_+1,nvar});  \
  Kokkos::parallel_for("begin_send_ghost<" #x_ #y_ #z_ ">", face_policy,       \
    KOKKOS_LAMBDA(const int y_, const int z_, const int v) {                   \
      const int idx = v*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1);                   \
      sbuf(idx) = fields(VOXEL(x,y,z,nx,ny,nz), beg_var+v);                    \
    });									                            

/**
 * @brief Serialize face to specific MPI communication buffer
 *
 * Serializes the specific face into a contiguous buffer for sending to another
 * process. Only packs data when destination process is valid and not 
 * itself. Use copy_face to handle scenarios where process sends to itself.
 *
 * @tparam i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) ex. Pos x-face: 1,0,0
 * @param fa[inout] Pointer to field_array_t structure
 * @param beg_var,end_var[in] Range of field variables to copy [beg_var,end_var)
 */
template<int i, int j, int k>
void 
pack_face(field_array* fa, const int beg_var, const int end_var) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)]; 
  if( 0 <= dst && dst < world_size ) { 
    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int nvar = end_var - beg_var;
    const k_field_t& fields = fa->k_f_d;
    if( dst != world_rank ) {
      Kokkos::DualView<float*> sbuf = get_send_dualview(fa, i, j, k);
      Kokkos::View<float*> sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();

      if constexpr (i != 0 && j == 0 && k == 0) {
        PACK_FACE(x,y,z,sbuf_d,i,j,k);
      } else if constexpr (i == 0 && j != 0 && k == 0) {
        PACK_FACE(y,z,x,sbuf_d,i,j,k);
      } else if constexpr (i == 0 && j == 0 && k != 0) {
        PACK_FACE(z,x,y,sbuf_d,i,j,k);
      }
      sbuf.modify_device();
    }
  }
#undef PACK_FACE
}

/**
 * @brief Start non blocking recv of specific field face
 *
 * @tparam i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) ex. Pos x-face: 1,0,0
 * @param fa[in] Pointer to field_array_t structure
 * @param nvar[in] Number of variables to communicate 
 */
template<int i, int j, int k>
void beg_recv_face(field_array* fa, const int nvar) {
  const int src = fa->g->bc[BOUNDARY(-i, -j, -k)];
  if( 0 <= src && src < world_size && src != world_rank ) {
    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int tag = BOUNDARY(i,j,k);
    const int size = nvar*get_face_size<i,j,k>(nx,ny,nz);
    Kokkos::DualView<float*> rbuf = get_recv_dualview(fa, i, j, k);
    float* buffer = get_comm_buffer(rbuf);
    MPI_Irecv(buffer, size, MPI_FLOAT, src, tag, 
              MPI_COMM_WORLD, &(fa->recv_req[tag]));
    fa->num_recv_wait += 1;
  }
}

/**
 * @brief Start non blocking send of specific field face
 *
 * @tparam i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) ex. Pos x-face: 1,0,0
 * @param fa[in] Pointer to field_array_t structure
 * @param nvar[in] Number of variables to communicate
 */
template<int i, int j, int k>
void beg_send_face(field_array* fa, const int nvar) {
  const int dst = fa->g->bc[BOUNDARY(i, j, k)];
  if( 0 <= dst && dst < world_size && dst != world_rank ) {
    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int tag = BOUNDARY(i,j,k);
    const int size = nvar*get_face_size<i,j,k>(nx,ny,nz);
    Kokkos::DualView<float*> sbuf = get_send_dualview(fa, i, j, k);
    sync_comm_buffer<Kokkos::DefaultHostExecutionSpace>(sbuf);
    float* buffer = get_comm_buffer(sbuf);
    MPI_Isend(buffer, size, MPI_FLOAT, dst, tag, 
               MPI_COMM_WORLD, &(fa->send_req[tag]));
  }
}

/**
 * @brief Start halo exchange of specified field variables [beg_var, end_var)
 * with neighboring processes
 *
 * @tparam i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) ex. Pos x-face: 1,0,0
 * @param fa[in] Pointer to field_array_t structure
 * @param beg_var,end_var[in] Range of field variables to copy [beg_var,end_var)
 */
void 
begin_halo_exchange(field_array* fa, const int beg_var, const int end_var) {
  const int nvar = end_var - beg_var;

  /***************************************************************************
   * Start receiving halos
   ***************************************************************************/
  // Negative faces
  beg_recv_face<-1,0,0>(fa, nvar);
  beg_recv_face<0,-1,0>(fa, nvar);
  beg_recv_face<0,0,-1>(fa, nvar);
  // Positive faces
  beg_recv_face<1,0,0>(fa, nvar);
  beg_recv_face<0,1,0>(fa, nvar);
  beg_recv_face<0,0,1>(fa, nvar);

  /***************************************************************************
   * Serialize halos
   ***************************************************************************/
  // Negative faces
  pack_face<-1,0,0>(fa, beg_var, end_var);
  pack_face<0,-1,0>(fa, beg_var, end_var);
  pack_face<0,0,-1>(fa, beg_var, end_var);

  // Positive faces
  pack_face<1,0,0>(fa, beg_var, end_var);
  pack_face<0,1,0>(fa, beg_var, end_var);
  pack_face<0,0,1>(fa, beg_var, end_var);

  Kokkos::fence();

  /***************************************************************************
   * Start sending halos
   ***************************************************************************/
  // Negative faces
  beg_send_face<-1,0,0>(fa, nvar);
  beg_send_face<0,-1,0>(fa, nvar);
  beg_send_face<0,0,-1>(fa, nvar);
  // Positive faces
  beg_send_face<1,0,0>(fa, nvar);
  beg_send_face<0,1,0>(fa, nvar);
  beg_send_face<0,0,1>(fa, nvar);

  /***************************************************************************
   * Copy halos (Periodic boundaries sending to self)
   ***************************************************************************/
  // Negative faces
  copy_face<-1,0,0>(fa, beg_var, end_var);
  copy_face<0,-1,0>(fa, beg_var, end_var);
  copy_face<0,0,-1>(fa, beg_var, end_var);

  // Positive faces
  copy_face<1,0,0>(fa, beg_var, end_var);
  copy_face<0,1,0>(fa, beg_var, end_var);
  copy_face<0,0,1>(fa, beg_var, end_var);
}

/**
 * \def UNPACK_FACE(x_,y_,z_,buffer,i,j,k)
 * Unpack a buffer into the ghost cells with a parallel_for kernel. 
 * DO NOT USE DIRECTLY! Use the wrapper function for clearer error messages
 * and profiling.
 */
#define UNPACK_FACE(x_,y_,z_,rbuf,i,j,k)                                  \
  const int x_ = (i+j+k) < 0 ? n##x_+1 : 0;				                             \
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> face_policy({1,1,0},                  \
                                                     {n##y_+1,n##z_+1,nvar});  \
  Kokkos::parallel_for("end_recv_ghost<" #x_ #y_ #z_ ">", face_policy,         \
    KOKKOS_LAMBDA(const int y_, const int z_, const int v) {                   \
      const int idx = v*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1);                   \
      fields(VOXEL(x,y,z,nx,ny,nz), beg_var+v) = rbuf(idx);                    \
    });									                            


/**
 * @brief Unpack face from specific MPI communication buffer to the halo 
 * cells
 *
 * @tparam i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) ex. Pos x-face: 1,0,0
 * @param fa[in] Pointer to field_array_t structure
 * @param beg_var,end_var[in] Range of field variables to copy [beg_var,end_var)
 */
template<int i, int j, int k>
void 
unpack_face(field_array* fa, const int beg_var, const int end_var) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  const int nvar = end_var - beg_var;
  const k_field_t& fields = fa->k_f_d;
  Kokkos::DualView<float*> rbuf = get_recv_dualview(fa, i, j, k);
  rbuf.modify_host();
  sync_comm_buffer<Kokkos::DefaultExecutionSpace>(rbuf);
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  if(rbuf_d.data()) { 
    if constexpr (i != 0 && j == 0 && k == 0) {
      UNPACK_FACE(x,y,z,rbuf_d,i,j,k);
    } else if constexpr (i == 0 && j != 0 && k == 0) {
      UNPACK_FACE(y,z,x,rbuf_d,i,j,k);
    } else if constexpr (i == 0 && j == 0 && k != 0) {
      UNPACK_FACE(z,x,y,rbuf_d,i,j,k);
    }
  }
#undef UNPACK_FACE
}

/**
 * @brief Finish face communication and unpack into the ghost cells 
 *
 * Waits for any receiving MPI_Request to finish and unpacks the buffer. The 
 * order in which faces are unpacked is first come first serve. Unpacking 
 * kernels are executed in order of submission but submission is non blocking.
 *
 * @param fa[in] Pointer to field_array_t structure
 * @param beg_var,end_var[in] Range of field variables to copy [beg_var,end_var)
 */
void end_recv_face(field_array* fa, const int beg_var, const int end_var) {
  const int n_total_req = 27;
  int face = 27;
  MPI_Status status;
  while(fa->num_recv_wait > 0) {
    MPI_Waitany(n_total_req, fa->recv_req, &face, &status);
    switch(face) {
      case BOUNDARY(-1, 0, 0): 
        unpack_face<-1,0,0>(fa, beg_var, end_var);
        break;
      case BOUNDARY(0, -1, 0): 
        unpack_face<0,-1,0>(fa, beg_var, end_var);
        break;
      case BOUNDARY(0, 0, -1): 
        unpack_face<0,0,-1>(fa, beg_var, end_var);
        break;
      case BOUNDARY(1, 0, 0):
        unpack_face<1,0,0>(fa, beg_var, end_var);
        break;
      case BOUNDARY(0, 1, 0): 
        unpack_face<0,1,0>(fa, beg_var, end_var);
        break;
      case BOUNDARY(0, 0, 1): 
        unpack_face<0,0,1>(fa, beg_var, end_var);
        break;
      default:
        break;
    }
    fa->num_recv_wait -= 1;
  }
}

/**
 * @brief Finish face communication and acknowledge send operation is complete
 *
 * @tparam i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) ex. Pos x-face: 1,0,0
 * @param fa[in] Pointer to field_array_t structure
 */
template<int i, int j, int k>
void end_send_face(field_array* fa) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size && dst != world_rank ) {
    MPI_Wait( &(fa->send_req[BOUNDARY(i,j,k)]), MPI_STATUS_IGNORE );
  }
}

/**
 * @brief Finish halo exchange of specified field variables [beg_var, end_var)
 * with neighboring processes
 *
 * @tparam i,j,k Face coord. (Pos: 1, Neg: -1, Other: 0) ex. Pos x-face: 1,0,0
 * @param fa[in] Pointer to field_array_t structure
 * @param beg_var,end_var[in] Range of field variables to copy [beg_var,end_var)
 */
void 
end_halo_exchange(field_array* fa, const int beg_var, const int end_var) {

  /***************************************************************************
   * End receiving halos
   ***************************************************************************/
  // Loop through and wait for all requests to finish
  end_recv_face(fa, beg_var, end_var);

  /***************************************************************************
   * End sending halos
   ***************************************************************************/
  // Negative faces
  end_send_face<-1,0,0>(fa);
  end_send_face<0,-1,0>(fa);
  end_send_face<0,0,-1>(fa);
  // Positive faces
  end_send_face<1,0,0>(fa);
  end_send_face<0,1,0>(fa);
  end_send_face<0,0,1>(fa);

  // Ensure all unpacking and copy kernels are complete
  fa->ghost_comm_space.fence();
  Kokkos::fence();
}

//Hybrid JF
#define BRP(x_,y_,z_)							                     \
  const int n##y_ = fa->g->n##y_, n##z_=fa->g->n##z_;	 \
  const int size = (4*n##y_*n##z_)*sizeof(float);			 \
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);
  

/**
 * @brief Begin non blocking receive for sharing jf ghost cells. 
 *
 * Calculates size of receive buffer based on which face and orientation.
 * BRP macro adjusts calculations for different face directions. Template 
 * function wraps the BRP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
begin_recv_ghost_hyb_jf(field_array* fa, const int i, const int j, const int k) {
  field_buffers_t* fb = fa->fb;
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)]; /**< Source rank */
  // Only recv cells if dst is a valid neighbor and not itself
  if( 0 <= src && src < world_size ) {
    Kokkos::DualView<float*> rbuf = fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();

    if constexpr (std::is_same<Face,XYZ>::value) {
      BRP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BRP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BRP(z,x,y);
    }
  }
}

#undef BRP

#define BSP(x_,y_,z_)							                                                  \
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;		                      \
  const int size = (4*n##y_*n##z_)*sizeof(float);				                            \
  const int face = (i+j+k)<0 ? 1 : n##x_;				                                    \
  const k_field_t& k_field = fa->k_f_d;					                                    \
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_##_face({1,1,0}, {n##z_+1, n##y_+1, 4}); \
  Kokkos::parallel_for("begin_send_ghost_hyb_jf<XYZ>", x_##_face,                   \
  KOKKOS_LAMBDA(const int z_, const int y_, const int var) {                        \
      const int x_ = face;						                                              \
      sbuf_d(var*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx+var); \
    });									                                                            \
  sbuf.modify_device();                                                             \
  SYNC_MPI_BUFFER(sbuf_h, sbuf_d);                                                  \
  BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);

/**
 * @brief Begin non blocking send for sharing jf ghost cells. 
 *
 * Serializes jf data in contiguous buffer and sends data to a neighbors 
 * ghost cells. BSP macro adjusts calculations for different face directions. 
 * Template function wraps the BRP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param sbuf_d Send buffer on the device
 * @param sbuf_h Mirror of sbuf_d on the Host
 */
template<typename Face> 
void 
begin_send_ghost_hyb_jf(field_array* fa, const int i, const int j, const int k) {
  field_buffers_t *fb = fa->fb;
  int dst = fa->g->bc[BOUNDARY(i,j,k)]; /**< Destination rank */
  // Only send cells if dst is a valid neighbor and not itself
  if( 0 <= dst && dst < world_size ) {
    Kokkos::DualView<float*> sbuf = fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();

    if constexpr (std::is_same<Face,XYZ>::value) {
      BSP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BSP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BSP(z,x,y);
    }
  }
}

#undef BSP


/**
 * @brief Begin exchanging jf cells between all neighbors.
 *
 * Prepares receive buffers, packs face cells into contiguous buffers
 * and starts non blocking communication with neighbors. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_begin_remote_ghost_hyb_jf(field_array_t* ALIGNED(128) fa, 
                            const grid_t* g, 
                            field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  begin_halo_exchange(fa, field_var::jfx, field_var::rhof+1);
#else
  // Start receiving
  begin_recv_ghost_hyb_jf<XYZ>(fa, -1,  0,  0);
  begin_recv_ghost_hyb_jf<YZX>(fa,  0, -1,  0);
  begin_recv_ghost_hyb_jf<ZXY>(fa,  0,  0, -1);
  begin_recv_ghost_hyb_jf<XYZ>(fa,  1,  0,  0);
  begin_recv_ghost_hyb_jf<YZX>(fa,  0,  1,  0);
  begin_recv_ghost_hyb_jf<ZXY>(fa,  0,  0,  1);

  // Start sending
  begin_send_ghost_hyb_jf<XYZ>(fa, -1,  0,  0);
  begin_send_ghost_hyb_jf<YZX>(fa,  0, -1,  0);
  begin_send_ghost_hyb_jf<ZXY>(fa,  0,  0, -1);
  begin_send_ghost_hyb_jf<XYZ>(fa,  1,  0,  0);
  begin_send_ghost_hyb_jf<YZX>(fa,  0,  1,  0);
  begin_send_ghost_hyb_jf<ZXY>(fa,  0,  0,  1);
#endif
}


#define ERP(x_,y_,z_)							                                                \
  const grid_t* g = fa->g;						                                            \
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));                  \
  if(p) {								                                                          \
    field_buffers *fb = fa->fb;                                                   \
    Kokkos::DualView<float*> rbuf = fb->recv_buffer[BOUNDARY(i,j,k)];             \
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();                     \
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();                 \
                                                                                  \
    const int nx = g->nx, ny = g->ny, nz = g->nz;			                            \
    const int face = (i+j+k) < 0 ? n##x_+1 : 0;				                            \
    const k_field_t& k_field = fa->k_f_d;				                                  \
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h);                                              \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
    Kokkos::parallel_for("end_recv_ghost_hyb_jf<XYZ>", x_##_face,                 \
    KOKKOS_LAMBDA(const int z_, const int y_) {                                   \
      const int x_ = face;						                                            \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx) = rbuf_d(                (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy) = rbuf_d(  n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz) = rbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof)= rbuf_d(3*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      });								                                                          \
  }									
  
/**
 * @brief End non blocking receive for sharing jf ghost cells. 
 *
 * Wait for non blocking communication to complete and unpack the buffer into
 * the local ranks ghost cells. Wait and unpacking the buffer may be made to 
 * overlap between different faces in the future.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
end_recv_ghost_hyb_jf(field_array_t* fa, const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)]; /**< Source rank */
  // Only recv cells if src is a valid neighbor and not itself
  if( 0 <= src && src < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
      ERP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      ERP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      ERP(z,x,y);
    }
  }
}

#undef ERP

/**
 * @brief End non blocking send for sharing jf ghost cells. 
 *
 * Ensures the prior send operation is complete and unsets the send buffer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 */
template<typename Face> 
void 
end_send_ghost_hyb_jf(field_array_t* fa, const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)]; /**< Destination rank */
  // Only send cells if dst is a valid neighbor and not itself
  if( 0 <= dst && dst < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
        end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,YZX>::value) {
        end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
        end_send_port_k(i,j,k,fa->g);
    }
  }
}

/**
 * @brief End exchanging jf cells between all neighbors.
 *
 * Wait until MPI communication is complete then unpack the buffers and
 * fill in the ghost cells with the communicated jf. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_end_remote_ghost_hyb_jf(field_array_t* ALIGNED(128) fa, 
                          const grid_t* g, 
                          field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  end_halo_exchange(fa, field_var::jfx, field_var::rhof+1);
#else
  // End receiving
  end_recv_ghost_hyb_jf<XYZ>(fa, -1,  0,  0);
  end_recv_ghost_hyb_jf<YZX>(fa,  0, -1,  0);
  end_recv_ghost_hyb_jf<ZXY>(fa,  0,  0, -1);
  end_recv_ghost_hyb_jf<XYZ>(fa,  1,  0,  0);
  end_recv_ghost_hyb_jf<YZX>(fa,  0,  1,  0);
  end_recv_ghost_hyb_jf<ZXY>(fa,  0,  0,  1);

  // End sending
  end_send_ghost_hyb_jf<XYZ>(fa, -1,  0,  0);
  end_send_ghost_hyb_jf<YZX>(fa,  0, -1,  0);
  end_send_ghost_hyb_jf<ZXY>(fa,  0,  0, -1);
  end_send_ghost_hyb_jf<XYZ>(fa,  1,  0,  0);
  end_send_ghost_hyb_jf<YZX>(fa,  0,  1,  0);
  end_send_ghost_hyb_jf<ZXY>(fa,  0,  0,  1);
  // Fence to make sure all ghost cells are done unpacking
  Kokkos::fence(); 
#endif
}

//Hybrid E
#define BRP(x_,y_,z_)							                     \
  const int n##y_ = fa->g->n##y_, n##z_=fa->g->n##z_;  \
  const int size = (3*n##y_*n##z_)*sizeof(float);      \
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);

/**
 * @brief Begin non blocking receive for sharing (ex,ey,ez) ghost cells. 
 *
 * Calculates size of receive buffer based on which face and orientation.
 * BRP macro adjusts calculations for different face directions. Template 
 * function wraps the BRP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
begin_recv_ghost_hyb_e(field_array* fa, 
                       const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)]; /**< Source rank */
  // Only recv cells if neighors are valid and not itself
  if( 0 <= src && src < world_size ) {
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();

    if constexpr (std::is_same<Face,XYZ>::value) {
      BRP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BRP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BRP(z,x,y);
    }
  }
}

#undef BRP

#define BSP(x_,y_,z_)							\
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;		\
  const int size = (3*n##y_*n##z_)*sizeof(float);				\
  const int face = (i+j+k)<0 ? 1 : n##x_;				\
  const k_field_t& k_field = fa->k_f_d;					\
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
  Kokkos::parallel_for("begin_send_ghost_hyb_e<XYZ>", \
  x_##_face, KOKKOS_LAMBDA(const int z_, const int y_) { \
      const int x_ = face;						\
      sbuf_d(                (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex); \
      sbuf_d(n##y_*n##z_   + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey); \
      sbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez); \
    });									\
  SYNC_MPI_BUFFER(sbuf_h, sbuf_d); \
  BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);

/**
 * @brief Begin non blocking send for sharing jf ghost cells. 
 *
 * Serializes jf data in contiguous buffer and sends data to a neighbors 
 * ghost cells. BSP macro adjusts calculations for different face directions. 
 * Template function wraps the BSP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param sbuf_d Send buffer on the device
 * @param sbuf_h Mirror of sbuf_d on the Host
 */
template<typename Face> 
void 
begin_send_ghost_hyb_e(field_array* fa, 
                       const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)]; /**< Destination rank */
  // Only send cells if dst is a valid neighbor and not itself
  if( 0 <= dst && dst < world_size ) {
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();

    if constexpr (std::is_same<Face,XYZ>::value) {
      BSP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BSP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BSP(z,x,y);
    }
  }
}

#undef BSP


/**
 * @brief Begin exchanging E fields between all neighbors.
 *
 * Prepares receive buffers, packs face cells into contiguous buffers
 * and starts non blocking communication with neighbors. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_begin_remote_ghost_hyb_e(field_array_t* ALIGNED(128) fa, 
                           const grid_t* g, 
                           field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  begin_halo_exchange(fa, field_var::ex, field_var::ez+1);
#else
  // Start receiving
  begin_recv_ghost_hyb_e<XYZ>(fa, -1,  0,  0);
  begin_recv_ghost_hyb_e<YZX>(fa,  0, -1,  0);
  begin_recv_ghost_hyb_e<ZXY>(fa,  0,  0, -1);
  begin_recv_ghost_hyb_e<XYZ>(fa,  1,  0,  0);
  begin_recv_ghost_hyb_e<YZX>(fa,  0,  1,  0);
  begin_recv_ghost_hyb_e<ZXY>(fa,  0,  0,  1);

  // Start sending
  begin_send_ghost_hyb_e<XYZ>(fa, -1,  0,  0);
  begin_send_ghost_hyb_e<YZX>(fa,  0, -1,  0);
  begin_send_ghost_hyb_e<ZXY>(fa,  0,  0, -1);
  begin_send_ghost_hyb_e<XYZ>(fa,  1,  0,  0);
  begin_send_ghost_hyb_e<YZX>(fa,  0,  1,  0);
  begin_send_ghost_hyb_e<ZXY>(fa,  0,  0,  1);
#endif
}


#define ERP(x_,y_,z_)							                                \
  const grid_t* g = fa->g;						                            \
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));  \
  if(p) {                                                         \
    field_buffers *fb = fa->fb;                                                   \
    Kokkos::DualView<float*> rbuf = fb->recv_buffer[BOUNDARY(i,j,k)];             \
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();                     \
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();                 \
                                                                                  \
    const int nx = g->nx, ny = g->ny, nz = g->nz;			\
    const int face = (i+j+k) < 0 ? n##x_+1 : 0;				\
    const k_field_t& k_field = fa->k_f_d;				\
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
    Kokkos::parallel_for("end_recv_ghost_hyb_e<XYZ>", x_##_face,  \
    KOKKOS_LAMBDA(const int z_, const int y_) { \
      const int x_ = face;						\
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = rbuf_d(                (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = rbuf_d(  n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez) = rbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      });								\
  }									
  
/**
 * @brief End non blocking receive for sharing E field ghost cells. 
 *
 * Wait for non blocking communication to complete and unpack the buffer into
 * the local ranks ghost cells. Wait and unpacking the buffer may be made to 
 * overlap between different faces in the future.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
end_recv_ghost_hyb_e(field_array_t* fa, 
                     const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)]; /**< Source rank */
  // Only recv cells if src is a valid neighbor and not itself
  if( 0 <= src && src < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
      ERP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      ERP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      ERP(z,x,y);
    }
  }
}

#undef ERP

/**
 * @brief End non blocking send for sharing E field ghost cells. 
 *
 * Ensures the prior send operation is complete and unsets the send buffer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 */
template<typename Face> 
void 
end_send_ghost_hyb_e(field_array_t* fa, const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      end_send_port_k(i,j,k,fa->g);
    }
  }
}

/**
 * @brief End exchanging E field cells between all neighbors.
 *
 * Wait until MPI communication is complete then unpack the buffers and
 * fill in the ghost cells with the communicated E field. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_end_remote_ghost_hyb_e(field_array_t* ALIGNED(128) fa, 
                         const grid_t* g, 
                         field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  // End receiving and sending
  end_halo_exchange(fa, field_var::ex, field_var::ez+1);
#else
  // End receiving
  end_recv_ghost_hyb_e<XYZ>(fa, -1,  0,  0);
  end_recv_ghost_hyb_e<YZX>(fa,  0, -1,  0);
  end_recv_ghost_hyb_e<ZXY>(fa,  0,  0, -1);
  end_recv_ghost_hyb_e<XYZ>(fa,  1,  0,  0);
  end_recv_ghost_hyb_e<YZX>(fa,  0,  1,  0);
  end_recv_ghost_hyb_e<ZXY>(fa,  0,  0,  1);

  // End sending
  end_send_ghost_hyb_e<XYZ>(fa, -1,  0,  0);
  end_send_ghost_hyb_e<YZX>(fa,  0, -1,  0);
  end_send_ghost_hyb_e<ZXY>(fa,  0,  0, -1);
  end_send_ghost_hyb_e<XYZ>(fa,  1,  0,  0);
  end_send_ghost_hyb_e<YZX>(fa,  0,  1,  0);
  end_send_ghost_hyb_e<ZXY>(fa,  0,  0,  1);

  Kokkos::fence(); 
#endif
}

//Hybrid Ue
#define BRP(x_,y_,z_)							                     \
  const int n##y_ = fa->g->n##y_, n##z_=fa->g->n##z_;  \
  const int size = (3*n##y_*n##z_)*sizeof(float);      \
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);

/**
 * @brief Begin non blocking receive for sharing (ex,ey,ez) ghost cells. 
 *
 * Calculates size of receive buffer based on which face and orientation.
 * BRP macro adjusts calculations for different face directions. Template 
 * function wraps the BRP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
begin_recv_ghost_hyb_ue(field_array* fa, 
                       const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)]; /**< Source rank */
  // Only recv cells if neighors are valid and not itself
  if( 0 <= src && src < world_size ) {
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();

    if constexpr (std::is_same<Face,XYZ>::value) {
      BRP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BRP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BRP(z,x,y);
    }
  }
}

#undef BRP

#define BSP(x_,y_,z_)							\
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;		\
  const int size = (3*n##y_*n##z_)*sizeof(float);				\
  const int face = (i+j+k)<0 ? 1 : n##x_;				\
  const k_field_t& k_field = fa->k_f_d;					\
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
  Kokkos::parallel_for("begin_send_ghost_hyb_ue<XYZ>", \
  x_##_face, KOKKOS_LAMBDA(const int z_, const int y_) { \
      const int x_ = face;						\
      sbuf_d(                (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ux); \
      sbuf_d(n##y_*n##z_   + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::uy); \
      sbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::uz); \
    });									\
  SYNC_MPI_BUFFER(sbuf_h, sbuf_d); \
  BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);

/**
 * @brief Begin non blocking send for sharing jf ghost cells. 
 *
 * Serializes jf data in contiguous buffer and sends data to a neighbors 
 * ghost cells. BSP macro adjusts calculations for different face directions. 
 * Template function wraps the BSP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param sbuf_d Send buffer on the device
 * @param sbuf_h Mirror of sbuf_d on the Host
 */
template<typename Face> 
void 
begin_send_ghost_hyb_eu(field_array* fa, 
                       const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)]; /**< Destination rank */
  // Only send cells if dst is a valid neighbor and not itself
  if( 0 <= dst && dst < world_size ) {
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();

    if constexpr (std::is_same<Face,XYZ>::value) {
      BSP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BSP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BSP(z,x,y);
    }
  }
}

#undef BSP


/**
 * @brief Begin exchanging E fields between all neighbors.
 *
 * Prepares receive buffers, packs face cells into contiguous buffers
 * and starts non blocking communication with neighbors. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_begin_remote_ghost_hyb_ue(field_array_t* ALIGNED(128) fa, 
                           const grid_t* g, 
                           field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  begin_halo_exchange(fa, field_var::ux, field_var::uz+1);
#else
  // Start receiving
  begin_recv_ghost_hyb_ue<XYZ>(fa, -1,  0,  0);
  begin_recv_ghost_hyb_ue<YZX>(fa,  0, -1,  0);
  begin_recv_ghost_hyb_ue<ZXY>(fa,  0,  0, -1);
  begin_recv_ghost_hyb_ue<XYZ>(fa,  1,  0,  0);
  begin_recv_ghost_hyb_ue<YZX>(fa,  0,  1,  0);
  begin_recv_ghost_hyb_ue<ZXY>(fa,  0,  0,  1);

  // Start sending
  begin_send_ghost_hyb_ue<XYZ>(fa, -1,  0,  0);
  begin_send_ghost_hyb_ue<YZX>(fa,  0, -1,  0);
  begin_send_ghost_hyb_ue<ZXY>(fa,  0,  0, -1);
  begin_send_ghost_hyb_ue<XYZ>(fa,  1,  0,  0);
  begin_send_ghost_hyb_ue<YZX>(fa,  0,  1,  0);
  begin_send_ghost_hyb_ue<ZXY>(fa,  0,  0,  1);
#endif
}


#define ERP(x_,y_,z_)							                                \
  const grid_t* g = fa->g;						                            \
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));  \
  if(p) {                                                         \
    field_buffers *fb = fa->fb;                                                   \
    Kokkos::DualView<float*> rbuf = fb->recv_buffer[BOUNDARY(i,j,k)];             \
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();                     \
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();                 \
                                                                                  \
    const int nx = g->nx, ny = g->ny, nz = g->nz;			\
    const int face = (i+j+k) < 0 ? n##x_+1 : 0;				\
    const k_field_t& k_field = fa->k_f_d;				\
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
    Kokkos::parallel_for("end_recv_ghost_hyb_ue<XYZ>", x_##_face,  \
    KOKKOS_LAMBDA(const int z_, const int y_) { \
      const int x_ = face;						\
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ux) = rbuf_d(                (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::uy) = rbuf_d(  n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::uz) = rbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      });								\
  }									
  
/**
 * @brief End non blocking receive for sharing Ue field ghost cells. 
 *
 * Wait for non blocking communication to complete and unpack the buffer into
 * the local ranks ghost cells. Wait and unpacking the buffer may be made to 
 * overlap between different faces in the future.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
end_recv_ghost_hyb_ue(field_array_t* fa, 
                     const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)]; /**< Source rank */
  // Only recv cells if src is a valid neighbor and not itself
  if( 0 <= src && src < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
      ERP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      ERP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      ERP(z,x,y);
    }
  }
}

#undef ERP

/**
 * @brief End non blocking send for sharing E field ghost cells. 
 *
 * Ensures the prior send operation is complete and unsets the send buffer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 */
template<typename Face> 
void 
end_send_ghost_hyb_ue(field_array_t* fa, const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      end_send_port_k(i,j,k,fa->g);
    }
  }
}

/**
 * @brief End exchanging Ue field cells between all neighbors.
 *
 * Wait until MPI communication is complete then unpack the buffers and
 * fill in the ghost cells with the communicated E field. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_end_remote_ghost_hyb_ue(field_array_t* ALIGNED(128) fa, 
                         const grid_t* g, 
                         field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  // End receiving and sending
  end_halo_exchange(fa, field_var::ue, field_var::uz+1);
#else
  // End receiving
  end_recv_ghost_hyb_ue<XYZ>(fa, -1,  0,  0);
  end_recv_ghost_hyb_ue<YZX>(fa,  0, -1,  0);
  end_recv_ghost_hyb_ue<ZXY>(fa,  0,  0, -1);
  end_recv_ghost_hyb_ue<XYZ>(fa,  1,  0,  0);
  end_recv_ghost_hyb_ue<YZX>(fa,  0,  1,  0);
  end_recv_ghost_hyb_ue<ZXY>(fa,  0,  0,  1);

  // End sending
  end_send_ghost_hyb_ue<XYZ>(fa, -1,  0,  0);
  end_send_ghost_hyb_ue<YZX>(fa,  0, -1,  0);
  end_send_ghost_hyb_ue<ZXY>(fa,  0,  0, -1);
  end_send_ghost_hyb_ue<XYZ>(fa,  1,  0,  0);
  end_send_ghost_hyb_ue<YZX>(fa,  0,  1,  0);
  end_send_ghost_hyb_ue<ZXY>(fa,  0,  0,  1);

  Kokkos::fence(); 
#endif
}



//Hybrid curl_lpl_B
#define BRP(x_,y_,z_)                                  \
  const int n##y_ = fa->g->n##y_, n##z_=fa->g->n##z_;  \
  const int size = (3*n##y_*n##z_)*sizeof(float);      \
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h); 

/**
 * @brief Begin non blocking receive for sharing pressure between ghost cells. 
 *
 * Calculates size of receive buffer based on which face and orientation.
 * BRP macro adjusts calculations for different face directions. Template 
 * function wraps the BRP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
begin_recv_ghost_hyb_curl_lpl_b(field_array* fa, 
                                const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)]; /**< Source rank */
  // Only recv cells if src is a valid neighbor and not itself
  if( 0 <= src && src < world_size ) {
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();

    if constexpr (std::is_same<Face,XYZ>::value) {
      BRP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BRP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BRP(z,x,y);
    }
  }
}

#undef BRP

#define BSP(x_,y_,z_)							\
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;		\
  const int size = (3*n##y_*n##z_)*sizeof(float);				\
  const int face = (i+j+k)<0 ? 1 : n##x_;				\
  const k_field_t& k_field = fa->k_f_d;					\
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
  Kokkos::parallel_for("begin_send_ghost_hyb_curl_lpl_b<XYZ>", x_##_face, \
  KOKKOS_LAMBDA(const int z_, const int y_) { \
      const int x_ = face;						\
      sbuf_d(                (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::pex); \
      sbuf_d(n##y_*n##z_   + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::pey); \
      sbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::pez); \
    });									\
  SYNC_MPI_BUFFER(sbuf_h, sbuf_d); \
  BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);


/**
 * @brief Begin non blocking send for sharing pressure ghost cells. 
 *
 * Serializes pressure data in contiguous buffer and sends data to a neighbors 
 * ghost cells. BSP macro adjusts calculations for different face directions. 
 * Template function wraps the BSP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param sbuf_d Send buffer on the device
 * @param sbuf_h Mirror of sbuf_d on the Host
 */
template<typename Face> 
void 
begin_send_ghost_hyb_curl_lpl_b(field_array* fa, 
                                const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)]; /**< Destination rank */
  // Only recv cells if dst is a valid neighbor and not itself
  if( 0 <= dst && dst < world_size ) {
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();

    if constexpr (std::is_same<Face,XYZ>::value) {
      BSP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BSP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BSP(z,x,y);
    }
  }
}

#undef BSP


/**
 * @brief Begin exchanging Pressure between all neighbors.
 *
 * Prepares receive buffers, packs face cells into contiguous buffers
 * and starts non blocking communication with neighbors. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_begin_remote_ghost_hyb_curl_lpl_b(field_array_t* ALIGNED(128) fa, 
                                    const grid_t* g, 
                                    field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  begin_halo_exchange(fa, field_var::pex, field_var::pez+1);
#else
  // Start receiving
  begin_recv_ghost_hyb_curl_lpl_b<XYZ>(fa, -1,  0,  0);
  begin_recv_ghost_hyb_curl_lpl_b<YZX>(fa,  0, -1,  0);
  begin_recv_ghost_hyb_curl_lpl_b<ZXY>(fa,  0,  0, -1);
  begin_recv_ghost_hyb_curl_lpl_b<XYZ>(fa,  1,  0,  0);
  begin_recv_ghost_hyb_curl_lpl_b<YZX>(fa,  0,  1,  0);
  begin_recv_ghost_hyb_curl_lpl_b<ZXY>(fa,  0,  0,  1);

  // Start sending
  begin_send_ghost_hyb_curl_lpl_b<XYZ>(fa, -1,  0,  0);
  begin_send_ghost_hyb_curl_lpl_b<YZX>(fa,  0, -1,  0);
  begin_send_ghost_hyb_curl_lpl_b<ZXY>(fa,  0,  0, -1);
  begin_send_ghost_hyb_curl_lpl_b<XYZ>(fa,  1,  0,  0);
  begin_send_ghost_hyb_curl_lpl_b<YZX>(fa,  0,  1,  0);
  begin_send_ghost_hyb_curl_lpl_b<ZXY>(fa,  0,  0,  1);
#endif
}


#define ERP(x_,y_,z_)							\
  const grid_t* g = fa->g;						\
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));	\
  if(p) {								\
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];         \
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();                     \
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();                 \
                                                                                  \
    const int nx = g->nx, ny = g->ny, nz = g->nz;			\
    const int face = (i+j+k) < 0 ? n##x_+1 : 0;				\
    const k_field_t& k_field = fa->k_f_d;				\
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
    Kokkos::parallel_for("end_recv_ghost_hyb_curl_lpl_b<XYZ>", x_##_face, KOKKOS_LAMBDA(const int z_, const int y_) { \
      const int x_ = face;						\
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::pex) = rbuf_d(                (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::pey) = rbuf_d(  n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::pez) = rbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      });								\
  }									
  
/**
 * @brief End non blocking receive for sharing pressure between ghost cells. 
 *
 * Wait for non blocking communication to complete and unpack the buffer into
 * the local ranks ghost cells. Wait and unpacking the buffer may be made to 
 * overlap between different faces in the future.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
end_recv_ghost_hyb_curl_lpl_b(field_array_t* fa, 
                              const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)];
  if( 0 <= src && src < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
      ERP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      ERP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      ERP(z,x,y);
    }
  }
}

#undef ERP

/**
 * @brief End non blocking send for sharing pressure between ghost cells. 
 *
 * Ensures the prior send operation is complete and unsets the send buffer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 */
template<typename Face> 
void 
end_send_ghost_hyb_curl_lpl_b(field_array_t* fa, 
                              const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      end_send_port_k(i,j,k,fa->g);
    }
  }
}

/**
 * @brief End exchanging pressure ghost cells between all neighbors.
 *
 * Wait until MPI communication is complete then unpack the buffers and
 * fill in the ghost cells with the communicated pressure. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_end_remote_ghost_hyb_curl_lpl_b(field_array_t* ALIGNED(128) fa, 
                                  const grid_t* g, 
                                  field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  end_halo_exchange(fa, field_var::pex, field_var::pez+1);
#else
  // End receiving
  end_recv_ghost_hyb_curl_lpl_b<XYZ>(fa, -1,  0,  0);
  end_recv_ghost_hyb_curl_lpl_b<YZX>(fa,  0, -1,  0);
  end_recv_ghost_hyb_curl_lpl_b<ZXY>(fa,  0,  0, -1);
  end_recv_ghost_hyb_curl_lpl_b<XYZ>(fa,  1,  0,  0);
  end_recv_ghost_hyb_curl_lpl_b<YZX>(fa,  0,  1,  0);
  end_recv_ghost_hyb_curl_lpl_b<ZXY>(fa,  0,  0,  1);

  // End sending
  end_send_ghost_hyb_curl_lpl_b<XYZ>(fa, -1,  0,  0);
  end_send_ghost_hyb_curl_lpl_b<YZX>(fa,  0, -1,  0);
  end_send_ghost_hyb_curl_lpl_b<ZXY>(fa,  0,  0, -1);
  end_send_ghost_hyb_curl_lpl_b<XYZ>(fa,  1,  0,  0);
  end_send_ghost_hyb_curl_lpl_b<YZX>(fa,  0,  1,  0);
  end_send_ghost_hyb_curl_lpl_b<ZXY>(fa,  0,  0,  1);

  Kokkos::fence(); 
#endif
}


//Hybrid B
#define BRP(x_,y_,z_)							                     \
  const int n##y_ = fa->g->n##y_, n##z_=fa->g->n##z_;  \
  const int size = (4*n##y_*n##z_)*sizeof(float);      \
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);

/**
 * @brief Begin exchanging B fields between all neighbors.
 *
 * Prepares receive buffers, pack face cells into contiguous buffers
 * and starts non blocking communication with neighbors. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
template<typename Face> 
void 
begin_recv_ghost_hyb_b(field_array* fa, 
                       const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)];
  if( 0 <= src && src < world_size ) {
    Kokkos::DualView<float*> sbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      BRP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BRP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BRP(z,x,y);
    }
  }
}

#undef BRP

#define BSP(x_,y_,z_)							\
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;		\
  const int size = (4*n##y_*n##z_)*sizeof(float);				\
  const int face = (i+j+k)<0 ? 1 : n##x_;				\
  const k_field_t& k_field = fa->k_f_d;					\
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_##_face({1, 1, 0}, {n##y_+1, n##z_+1, 4}); \
  Kokkos::parallel_for("begin_send_ghost_hyb_b<" #x_ #y_ #z_ ">", \
    x_##_face, KOKKOS_LAMBDA(const int y_, const int z_, const int var) { \
      const int x_ = face;						\
      sbuf_d(var*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx+var); \
    });									\
  SYNC_MPI_BUFFER(sbuf_h, sbuf_d); \
  BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);

/**
 * @brief Begin non blocking send for sharing B field between ghost cells. 
 *
 * Serializes B field data in contiguous buffer and sends data to a neighbors 
 * ghost cells. BSP macro adjusts calculations for different face directions. 
 * Template function wraps the BSP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param sbuf_d Send buffer on the device
 * @param sbuf_h Mirror of sbuf_d on the Host
 */
template<typename Face> 
void 
begin_send_ghost_hyb_b(field_array* fa, 
                       const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size ) {
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      BSP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BSP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BSP(z,x,y);
    }
  }
}

#undef BSP

/**
 * @brief Begin exchanging B field ghost cells between all neighbors.
 *
 * Prepares receive buffers, packs face cells into contiguous buffers
 * and starts non blocking communication with neighbors. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_begin_remote_ghost_hyb_b(field_array_t* ALIGNED(128) fa, 
                           const grid_t* g, 
                           field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  begin_halo_exchange(fa, field_var::cbx, field_var::pe+1);
#else
  // Start receiving
  begin_recv_ghost_hyb_b<XYZ>(fa, -1,  0,  0);
  begin_recv_ghost_hyb_b<YZX>(fa,  0, -1,  0);
  begin_recv_ghost_hyb_b<ZXY>(fa,  0,  0, -1);
  begin_recv_ghost_hyb_b<XYZ>(fa,  1,  0,  0);
  begin_recv_ghost_hyb_b<YZX>(fa,  0,  1,  0);
  begin_recv_ghost_hyb_b<ZXY>(fa,  0,  0,  1);

  // Start sending
  begin_send_ghost_hyb_b<XYZ>(fa, -1,  0,  0);
  begin_send_ghost_hyb_b<YZX>(fa,  0, -1,  0);
  begin_send_ghost_hyb_b<ZXY>(fa,  0,  0, -1);
  begin_send_ghost_hyb_b<XYZ>(fa,  1,  0,  0);
  begin_send_ghost_hyb_b<YZX>(fa,  0,  1,  0);
  begin_send_ghost_hyb_b<ZXY>(fa,  0,  0,  1);
#endif
}


#define ERP(x_,y_,z_)							\
  const grid_t* g = fa->g;						\
  Kokkos::Profiling::pushRegion("end_recv_ghost_hyb_b<XYZ>::wait"); \
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));	\
  Kokkos::Profiling::popRegion(); \
  if(p) {								\
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];         \
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();                     \
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();                 \
    const int nx = g->nx, ny = g->ny, nz = g->nz;			\
    const int face = (i+j+k) < 0 ? n##x_+1 : 0;				\
    const k_field_t& k_field = fa->k_f_d;				\
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
    Kokkos::parallel_for("end_recv_ghost_hyb_b<XYZ>", x_##_face, KOKKOS_LAMBDA(const int z_, const int y_) { \
      const int x_ = face;						\
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = rbuf_d(                (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = rbuf_d(  n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = rbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::pe)  = rbuf_d(3*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      });								\
  }									
  
/**
 * @brief End exchanging B field ghost cells between all neighbors.
 *
 * Wait until MPI communication is complete then unpack the buffers and
 * fill in the ghost cells with the communicated B fields. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
template<typename Face> 
void 
end_recv_ghost_hyb_b(field_array_t* fa, 
                     const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)];
  if( 0 <= src && src < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
      ERP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      ERP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      ERP(z,x,y);
    }
  }
}

#undef ERP

/**
 * @brief End non blocking send for sharing B field ghost cells. 
 *
 * Ensures the prior send operation is complete and unsets the send buffer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 */
template<typename Face> 
void 
end_send_ghost_hyb_b(field_array_t* fa, const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size ) {
    if constexpr (std::is_same<Face,XYZ>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      end_send_port_k(i,j,k,fa->g);
    }
  }
}

/**
 * @brief End exchanging B field ghost cells between all neighbors.
 *
 * Wait until MPI communication is complete then unpack the buffers and
 * fill in the ghost cells with the communicated B fields. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_end_remote_ghost_hyb_b(field_array_t* ALIGNED(128) fa, 
                         const grid_t* g, 
                         field_buffers_t& fb) {
    
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  // End receiving
  end_halo_exchange(fa, field_var::cbx, field_var::pe+1);
#else
  // End receiving
  end_recv_ghost_hyb_b<XYZ>(fa, -1,  0,  0);
  end_recv_ghost_hyb_b<YZX>(fa,  0, -1,  0);
  end_recv_ghost_hyb_b<ZXY>(fa,  0,  0, -1);
  end_recv_ghost_hyb_b<XYZ>(fa,  1,  0,  0);
  end_recv_ghost_hyb_b<YZX>(fa,  0,  1,  0);
  end_recv_ghost_hyb_b<ZXY>(fa,  0,  0,  1);

  // End sending
  end_send_ghost_hyb_b<XYZ>(fa, -1,  0,  0);
  end_send_ghost_hyb_b<YZX>(fa,  0, -1,  0);
  end_send_ghost_hyb_b<ZXY>(fa,  0,  0, -1);
  end_send_ghost_hyb_b<XYZ>(fa,  1,  0,  0);
  end_send_ghost_hyb_b<YZX>(fa,  0,  1,  0);
  end_send_ghost_hyb_b<ZXY>(fa,  0,  0,  1);
#endif
}

/*****************************************************************************
 * Hybrid temporary variables (tx,ty,tz) used for E-field smoothing
 *****************************************************************************/

#define BRP(x_,y_,z_)							\
  const int n##y_ = fa->g->n##y_, n##z_=fa->g->n##z_;			\
  const int size = (3*n##y_*n##z_)*sizeof(float);			\
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);

/**
 * @brief Begin non blocking receive for sharing temperature between ghost cells. 
 *
 * Calculates size of receive buffer based on which face and orientation.
 * BRP macro adjusts calculations for different face directions. Template 
 * function wraps the BRP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
begin_recv_ghost_hyb_t(field_array* fa, const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)];
  if( 0 <= src && src < world_size ) {
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      BRP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BRP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BRP(z,x,y);
    }
  }
}

#undef BRP

#define BSP(x_,y_,z_)							                                             \
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;                    \
  const int size = (3*n##y_*n##z_)*sizeof(float);                              \
  const int face = (i+j+k)<0 ? 1 : n##x_;	                                     \
  const k_field_t& k_field = fa->k_f_d;                                        \
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1});\
  Kokkos::parallel_for("begin_send_ghost_hyb_t<XYZ>",                          \
      x_##_face, KOKKOS_LAMBDA(const int z_, const int y_) {                   \
      const int x_ = face;                                                     \
      sbuf_d(                (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tx); \
      sbuf_d(n##y_*n##z_   + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ty); \
      sbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tz); \
    });                                                                        \
  SYNC_MPI_BUFFER(sbuf_h, sbuf_d);                                             \
  BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);

/**
 * @brief Begin non blocking send for sharing temperature ghost cells. 
 *
 * Serializes temperature data in contiguous buffer and sends data to a neighbors 
 * ghost cells. BSP macro adjusts calculations for different face directions. 
 * Template function wraps the BRP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param sbuf_d Send buffer on the device
 * @param sbuf_h Mirror of sbuf_d on the Host
 */
template<typename Face> 
void 
begin_send_ghost_hyb_t(field_array* fa, const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size ) {
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      BSP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BSP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BSP(z,x,y);
    }
  }
}

#undef BSP

/**
 * @brief Begin exchanging electron temperature cells between all neighbors.
 *
 * Prepares receive buffers, packs face cells into contiguous buffers
 * and starts non blocking communication with neighbors. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_begin_remote_ghost_hyb_t(field_array_t* ALIGNED(128) fa, 
                           const grid_t* g, 
                           field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  begin_halo_exchange(fa, field_var::tx, field_var::tz+1);
#else
  // Start receiving
  begin_recv_ghost_hyb_t<XYZ>(fa, -1,  0,  0);
  begin_recv_ghost_hyb_t<YZX>(fa,  0, -1,  0);
  begin_recv_ghost_hyb_t<ZXY>(fa,  0,  0, -1);
  begin_recv_ghost_hyb_t<XYZ>(fa,  1,  0,  0);
  begin_recv_ghost_hyb_t<YZX>(fa,  0,  1,  0);
  begin_recv_ghost_hyb_t<ZXY>(fa,  0,  0,  1);

  // Start sending
  begin_send_ghost_hyb_t<XYZ>(fa, -1,  0,  0);
  begin_send_ghost_hyb_t<YZX>(fa,  0, -1,  0);
  begin_send_ghost_hyb_t<ZXY>(fa,  0,  0, -1);
  begin_send_ghost_hyb_t<XYZ>(fa,  1,  0,  0);
  begin_send_ghost_hyb_t<YZX>(fa,  0,  1,  0);
  begin_send_ghost_hyb_t<ZXY>(fa,  0,  0,  1);
#endif
}

#define ERP(x_,y_,z_)                                                          \
  const grid_t* g = fa->g;                                                     \
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));               \
  if(p) {	                                                                     \
    const int nx = g->nx, ny = g->ny, nz = g->nz;                              \
    const int face = (i+j+k) < 0 ? n##x_+1 : 0;                                \
    const k_field_t& k_field = fa->k_f_d;	                                     \
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h);                                           \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
    Kokkos::parallel_for("end_recv_ghost_hyb_t<XYZ>", x_##_face, KOKKOS_LAMBDA(const int z_, const int y_) { \
      const int x_ = face;                                                     \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tx) = rbuf_d(                (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ty) = rbuf_d(  n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tz) = rbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      });                                                                      \
  }

/**
 * @brief End non blocking receive for sharing electron temperature ghost cells. 
 *
 * Wait for non blocking communication to complete and unpack the buffer into
 * the local ranks ghost cells. Wait and unpacking the buffer may be made to 
 * overlap between different faces in the future.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
end_recv_ghost_hyb_t(field_array_t* fa, const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)];
  if( 0 <= src && src < world_size ) {
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      ERP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      ERP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      ERP(z,x,y);
    }
  }
}

#undef ERP

/**
 * @brief End non blocking send for sharing electron temperature ghost cells. 
 *
 * Ensures the prior send operation is complete and unsets the send buffer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 */
template<typename Face> 
void 
end_send_ghost_hyb_t(field_array_t* fa, const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size ) {
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      end_send_port_k(i,j,k,fa->g);
    }
  }
}

/**
 * @brief End exchanging electron temperature cells between all neighbors.
 *
 * Wait until MPI communication is complete then unpack the buffers and
 * fill in the ghost cells with the communicated temperature. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void k_end_remote_ghost_hyb_t(field_array_t* ALIGNED(128) fa, 
                              const grid_t* g, 
                              field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  end_halo_exchange(fa, field_var::tx, field_var::tz+1);
#else
  // End receiving
  end_recv_ghost_hyb_t<XYZ>(fa, -1,  0,  0);
  end_recv_ghost_hyb_t<YZX>(fa,  0, -1,  0);
  end_recv_ghost_hyb_t<ZXY>(fa,  0,  0, -1);
  end_recv_ghost_hyb_t<XYZ>(fa,  1,  0,  0);
  end_recv_ghost_hyb_t<YZX>(fa,  0,  1,  0);
  end_recv_ghost_hyb_t<ZXY>(fa,  0,  0,  1);

  // End sending
  end_send_ghost_hyb_t<XYZ>(fa, -1,  0,  0);
  end_send_ghost_hyb_t<YZX>(fa,  0, -1,  0);
  end_send_ghost_hyb_t<ZXY>(fa,  0,  0, -1);
  end_send_ghost_hyb_t<XYZ>(fa,  1,  0,  0);
  end_send_ghost_hyb_t<YZX>(fa,  0,  1,  0);
  end_send_ghost_hyb_t<ZXY>(fa,  0,  0,  1);

  Kokkos::fence(); 
#endif
}

/*****************************************************************************
 * Hybrid temporary variables (ox,oy,oz) used for B-field smoothing
 *****************************************************************************/

#define BRP(x_,y_,z_)	                                 \
  const int n##y_ = fa->g->n##y_, n##z_=fa->g->n##z_;  \
  const int size = (3*n##y_*n##z_)*sizeof(float);      \
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);

/**
 * @brief Begin non blocking receive for sharing smoothing variables 
 * between ghost cells. 
 *
 * Calculates size of receive buffer based on which face and orientation.
 * BRP macro adjusts calculations for different face directions. Template 
 * function wraps the BRP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
begin_recv_ghost_hyb_o(field_array* fa, const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)];
  if( 0 <= src && src < world_size ) {
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      BRP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BRP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BRP(z,x,y);
    }
  }
}

#undef BRP

#define BSP(x_,y_,z_)	                                                         \
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;                    \
  const int size = (3*n##y_*n##z_)*sizeof(float);                              \
  const int face = (i+j+k)<0 ? 1 : n##x_;                                      \
  const k_field_t& k_field = fa->k_f_d;                                        \
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1});\
  Kokkos::parallel_for("begin_send_ghost_hyb_o<XYZ>", x_##_face, KOKKOS_LAMBDA(const int z_, const int y_) { \
      const int x_ = face;                                                     \
      sbuf_d(                (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ox); \
      sbuf_d(n##y_*n##z_   + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::oy); \
      sbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::oz); \
    });	                                                                       \
  SYNC_MPI_BUFFER(sbuf_h, sbuf_d);                                             \
  BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);

/**
 * @brief Begin non blocking send for sharing B field smoothing ghost cells. 
 *
 * Serializes smoothing data in contiguous buffer and sends data to a neighbors 
 * ghost cells. BSP macro adjusts calculations for different face directions. 
 * Template function wraps the BRP macro to make profiling clearer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param sbuf_d Send buffer on the device
 * @param sbuf_h Mirror of sbuf_d on the Host
 */
template<typename Face> 
void 
begin_send_ghost_hyb_o(field_array* fa, const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size ) {
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      BSP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      BSP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      BSP(z,x,y);
    }
  }
}

#undef BSP

/**
 * @brief Begin exchanging B field smoothing cells between all neighbors.
 *
 * Prepares receive buffers, packs face cells into contiguous buffers
 * and starts non blocking communication with neighbors. Only performs 
 * communication when necessary. Will ignore cases where the process is on a
 * boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_begin_remote_ghost_hyb_o(field_array_t* ALIGNED(128) fa, 
                           const grid_t* g, 
                           field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  begin_halo_exchange(fa, field_var::ox, field_var::oz+1);
#else
  // Start receiving
  begin_recv_ghost_hyb_o<XYZ>(fa, -1,  0,  0);
  begin_recv_ghost_hyb_o<YZX>(fa,  0, -1,  0);
  begin_recv_ghost_hyb_o<ZXY>(fa,  0,  0, -1);
  begin_recv_ghost_hyb_o<XYZ>(fa,  1,  0,  0);
  begin_recv_ghost_hyb_o<YZX>(fa,  0,  1,  0);
  begin_recv_ghost_hyb_o<ZXY>(fa,  0,  0,  1);

  // Start sending
  begin_send_ghost_hyb_o<XYZ>(fa, -1,  0,  0);
  begin_send_ghost_hyb_o<YZX>(fa,  0, -1,  0);
  begin_send_ghost_hyb_o<ZXY>(fa,  0,  0, -1);
  begin_send_ghost_hyb_o<XYZ>(fa,  1,  0,  0);
  begin_send_ghost_hyb_o<YZX>(fa,  0,  1,  0);
  begin_send_ghost_hyb_o<ZXY>(fa,  0,  0,  1);
#endif
}

#define ERP(x_,y_,z_)                                                          \
  const grid_t* g = fa->g;                                                     \
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));               \
  if(p) {                                                                      \
    const int nx = g->nx, ny = g->ny, nz = g->nz;                              \
    const int face = (i+j+k) < 0 ? n##x_+1 : 0;                                \
    const k_field_t& k_field = fa->k_f_d;                                      \
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h);                                           \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_##_face({1, 1}, {n##z_+1, n##y_+1}); \
    Kokkos::parallel_for("end_recv_ghost_hyb_o<XYZ>",                          \
      x_##_face, KOKKOS_LAMBDA(const int z_, const int y_) {                   \
      const int x_ = face;                                                     \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ox) = rbuf_d(                (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::oy) = rbuf_d(  n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::oz) = rbuf_d(2*n##y_*n##z_ + (z_-1)*n##y_ + (y_-1)); \
      });	                                                                     \
  }

/**
 * @brief End non blocking receive for sharing B field smoothing ghost cells. 
 *
 * Wait for non blocking communication to complete and unpack the buffer into
 * the local ranks ghost cells. Wait and unpacking the buffer may be made to 
 * overlap between different faces in the future.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 * @param rbuf_d Receive buffer on the device
 * @param rbuf_h Mirror of rbuf_d on the Host
 */
template<typename Face> 
void 
end_recv_ghost_hyb_o(field_array_t* fa, const int i, const int j, const int k) {
  int src = fa->g->bc[BOUNDARY(-i,-j,-k)];
  if( 0 <= src && src < world_size ) {
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      ERP(x,y,z);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      ERP(y,z,x);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      ERP(z,x,y);
    }
  }
}

#undef ERP

/**
 * @brief End non blocking send for sharing B field smoothing ghost cells. 
 *
 * Ensures the prior send operation is complete and unsets the send buffer.
 *
 * @tparam Face Enum denoting the face and order of dimensions for calculations
 * @param fa Pointer for field array structure containing field and grid data
 * @param i X-dim face coordinate (-1.0: neg. x, 0: origin, 1.0: pos. x)
 * @param i Y-dim face coordinate (-1.0: neg. y, 0: origin, 1.0: pos. y)
 * @param i Z-dim face coordinate (-1.0: neg. z, 0: origin, 1.0: pos. z)
 */
template<typename Face> 
void 
end_send_ghost_hyb_o(field_array_t* fa, const int i, const int j, const int k) {
  int dst = fa->g->bc[BOUNDARY(i,j,k)];
  if( 0 <= dst && dst < world_size ) {
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    if constexpr (std::is_same<Face,XYZ>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,YZX>::value) {
      end_send_port_k(i,j,k,fa->g);
    } else if constexpr (std::is_same<Face,ZXY>::value) {
      end_send_port_k(i,j,k,fa->g);
    }
  }
}

/**
 * @brief End exchanging B field smoothing cells between all neighbors.
 *
 * Wait until MPI communication is complete then unpack the buffers and
 * fill in the ghost cells with the communicated smoothing variables. Only 
 * performs communication when necessary. Will ignore cases where the process 
 * is on a boundary or if the process topology would make the exchange redundant 
 * (ex. 1D and 2D grids).
 *
 * @param fa Pointer for field array structure containing field and grid data
 * @param g Pointer to grid structure 
 * @param fb Reference to field buffers used for MPI communication
 */
void 
k_end_remote_ghost_hyb_o(field_array_t* ALIGNED(128) fa, 
                         const grid_t* g, 
                         field_buffers_t& fb) {
#ifdef VPIC_ENABLE_HALO_EXCHANGE
  end_halo_exchange(fa, field_var::ox, field_var::oz+1);
#else
  // End receiving
  end_recv_ghost_hyb_o<XYZ>(fa, -1,  0,  0);
  end_recv_ghost_hyb_o<YZX>(fa,  0, -1,  0);
  end_recv_ghost_hyb_o<ZXY>(fa,  0,  0, -1);
  end_recv_ghost_hyb_o<XYZ>(fa,  1,  0,  0);
  end_recv_ghost_hyb_o<YZX>(fa,  0,  1,  0);
  end_recv_ghost_hyb_o<ZXY>(fa,  0,  0,  1);

  // End sending
  end_send_ghost_hyb_o<XYZ>(fa, -1,  0,  0);
  end_send_ghost_hyb_o<YZX>(fa,  0, -1,  0);
  end_send_ghost_hyb_o<ZXY>(fa,  0,  0, -1);
  end_send_ghost_hyb_o<XYZ>(fa,  1,  0,  0);
  end_send_ghost_hyb_o<YZX>(fa,  0,  1,  0);
  end_send_ghost_hyb_o<ZXY>(fa,  0,  0,  1);

  Kokkos::fence(); 
#endif
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

template<typename Face> 
void 
begin_recv_tang_e_norm_b(field_array_t* fa, const int i, const int j, const int k) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  if constexpr(std::is_same<Face,XYZ>::value) {
    size = (2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz)*sizeof(float);
  } else if constexpr(std::is_same<Face,YZX>::value) {
    size = (2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx)*sizeof(float);
  } else if constexpr(std::is_same<Face,ZXY>::value) {
    size = (2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny)*sizeof(float);
  }
  begin_recv_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(rbuf_h.data()));
}
//template<> void begin_recv_tang_e_norm_b<XYZ>(field_array_t* fa, const int i, const int j, const int k, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const int ny = fa->g->ny, nz = fa->g->nz;
//    const int size = (2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz)*sizeof(float);
//    begin_recv_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(rbuf_h.data()));
//}
//template<> void begin_recv_tang_e_norm_b<YZX>(field_array_t* fa, const int i, const int j, const int k, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const int nx = fa->g->nx, nz = fa->g->nz;
//    const int size = (2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx)*sizeof(float);
//    begin_recv_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(rbuf_h.data()));
//}
//template<> void begin_recv_tang_e_norm_b<ZXY>(field_array_t* fa, const int i, const int j, const int k, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const int nx = fa->g->nx, ny = fa->g->ny;
//    const int size = (2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny)*sizeof(float);
//    begin_recv_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(rbuf_h.data()));
//}

template<typename Face> 
void 
begin_send_tang_e_norm_b(field_array_t* fa, 
                         const int i, const int j, const int k) {
}

template<> 
void 
begin_send_tang_e_norm_b<XYZ>(field_array_t* fa, const int i, const int j, const int k) {
    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int size = (2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz)*sizeof(float);
    const int face = (i+j+k) < 0 ? 1 : nx + 1;
    const int x = face;
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    k_field_t& k_field = fa->k_f_d;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_face({1, 1}, {nz+1, ny+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> yz_edge({1, 1}, {nz+2, ny+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_edge({1, 1}, {nz+1, ny+2});
    Kokkos::parallel_for("begin_send_tang_e_norm_b<XYZ> x face", x_face, KOKKOS_LAMBDA(const int z, const int y) {
        sbuf_d((z-1)*ny + (y-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx);
    });
    Kokkos::parallel_for("begin_send_tang_e_norm_b<XYZ> yz edge", yz_edge, KOKKOS_LAMBDA(const int z, const int y) {
        sbuf_d(nz*ny + 2*((z-1)*ny + (y-1))) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey);
        sbuf_d(nz*ny + 2*((z-1)*ny + (y-1)) + 1) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay);
    });
    Kokkos::parallel_for("begin_send_tang_e_norm_b<XYZ> zy edge", zy_edge, KOKKOS_LAMBDA(const int z, const int y) {
        sbuf_d(nz*ny + 2*ny*(nz+1) + 2*((z-1)*(ny+1) + (y-1))) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez);
        sbuf_d(nz*ny + 2*ny*(nz+1) + 2*((z-1)*(ny+1) + (y-1)) + 1) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz);
    });
    Kokkos::deep_copy(sbuf_h, sbuf_d);
    begin_send_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(sbuf_h.data()));
}
template<> 
void 
begin_send_tang_e_norm_b<YZX>(field_array_t* fa, const int i, const int j, const int k) {
    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int size = (2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx)*sizeof(float);
    const int face = (i+j+k) < 0 ? 1 : ny + 1;
    const int y = face;
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    k_field_t& k_field = fa->k_f_d;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> y_face({1, 1}, {nz+1, nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_edge({1, 1}, {nz+1, nx+2});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_edge({1, 1}, {nz+2, nx+1});
    Kokkos::parallel_for("begin_send_tang_e_norm_b<YZX> y face", y_face, KOKKOS_LAMBDA(const int z, const int x) {
        sbuf_d((z-1)*nx + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby);
    });
    Kokkos::parallel_for("begin_send_tang_e_norm_b<YZX> zx edge", zx_edge, KOKKOS_LAMBDA(const int z, const int x) {
        sbuf_d(nz*nx + 2*((z-1)*(nx+1) + (x-1))) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez);
        sbuf_d(nz*nx + 2*((z-1)*(nx+1) + (x-1)) + 1) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz);
    });
    Kokkos::parallel_for("begin_send_tang_e_norm_b<YZX> xz edge", xz_edge, KOKKOS_LAMBDA(const int z, const int x) {
        sbuf_d(nx*nz + 2*nz*(nx+1) + 2*((z-1)*nx + (x-1))) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex);
        sbuf_d(nx*nz + 2*nz*(nx+1) + 2*((z-1)*nx + (x-1)) + 1) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax);
    });
    Kokkos::deep_copy(sbuf_h, sbuf_d);
    begin_send_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(sbuf_h.data()));
}
template<> 
void 
begin_send_tang_e_norm_b<ZXY>(field_array_t* fa, const int i, const int j, const int k) {
    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    const int size = (2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny)*sizeof(float);
    const int face = (i+j+k) < 0 ? 1 : nz + 1;
    const int z = face;
    k_field_t& k_field = fa->k_f_d;
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> z_face({1, 1}, {ny+1, nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> xy_edge({1, 1}, {ny+2, nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_edge({1, 1}, {ny+1, nx+2});
    Kokkos::parallel_for("begin_send_tang_e_norm_b<ZXY> z face", z_face, KOKKOS_LAMBDA(const int y, const int x) {
        sbuf_d((y-1)*nx + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz);
    });
    Kokkos::parallel_for("begin_send_tang_e_norm_b<ZXY> xy edge", xy_edge, KOKKOS_LAMBDA(const int y, const int x) {
        sbuf_d(ny*nx + 2*((y-1)*nx + (x-1))) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex);
        sbuf_d(ny*nx + 2*((y-1)*nx + (x-1)) + 1) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax);
    });
    Kokkos::parallel_for("begin_send_tang_e_norm_b<ZXY> yx edge", yx_edge, KOKKOS_LAMBDA(const int y, const int x) {
        sbuf_d(ny*nx + 2*nx*(ny+1) + 2*((y-1)*(nx+1) + (x-1))) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey);
        sbuf_d(ny*nx + 2*nx*(ny+1) + 2*((y-1)*(nx+1) + (x-1)) + 1) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay);
    });
    Kokkos::deep_copy(sbuf_h, sbuf_d);
    begin_send_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(sbuf_h.data()));
}

template<typename Face> 
double 
end_recv_tang_e_norm_b(field_array_t* fa, const int i, const int j, const int k) {return 0.0f;}

template<> 
double 
end_recv_tang_e_norm_b<XYZ>(field_array_t* fa, const int i, const int j, const int k) {
    double err=0.0, err_temp=0.0;
    float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,fa->g));
    if(p) {
        const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
        const int face = (i+j+k)<0 ? nx+1 : 1;
        const int x = face;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
        auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
        auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
        Kokkos::deep_copy(rbuf_d, rbuf_h);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_face({1, 1}, {nz+1, ny+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yz_edge({1, 1}, {nz+2, ny+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_edge({1, 1}, {nz+1, ny+2});
        Kokkos::parallel_reduce("end_recv_tang_e_norm_b<XYZ> x face", x_face, KOKKOS_LAMBDA(const int z, const int y, double& error) {
            const double w1 = static_cast<double>(rbuf_d((z-1)*ny + (y-1)));
            const double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = static_cast<float>(0.5*(w1+w2));
            error += (w1-w2)*(w1-w2);
        }, err_temp);
        err += err_temp;
        err_temp = 0.0f;
        Kokkos::parallel_reduce("end_recv_tang_e_norm_b<XYZ> yz edge", yz_edge, KOKKOS_LAMBDA(const int z, const int y, double& error) {
            double w1 = static_cast<double>(rbuf_d(nz*ny + 2*((z-1)*ny + (y-1))));
            double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = static_cast<float>(0.5*(w1+w2));
            error += (w1-w2)*(w1-w2);
            w1 = static_cast<double>(rbuf_d(nz*ny + 2*((z-1)*ny + (y-1)) + 1));
            w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay) = static_cast<float>(0.5*(w1+w2));
        }, err_temp);
        err += err_temp;
        err_temp = 0.0f;
        Kokkos::parallel_reduce("end_recv_tang_e_norm_b<XYZ> zy edge", zy_edge, KOKKOS_LAMBDA(const int z, const int y, double& error) {
            double w1 = static_cast<double>(rbuf_d(nz*ny + 2*ny*(nz+1) + 2*((z-1)*(ny+1) + (y-1))));
            double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez) = static_cast<float>(0.5*(w1+w2));
            error += (w1-w2)*(w1-w2);
            w1 = static_cast<double>(rbuf_d(nz*ny + 2*ny*(nz+1) + 2*((z-1)*(ny+1) + (y-1)) + 1));
            w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz) = static_cast<float>(0.5*(w1+w2));
        }, err_temp);
        err += err_temp;
    }
    return err;
}
template<> 
double 
end_recv_tang_e_norm_b<YZX>(field_array_t* fa, const int i, const int j, const int k) {
    double err=0.0, err_temp=0.0;
    float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,fa->g));
    if(p) {
        const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
        const int face = (i+j+k)<0 ? ny+1 : 1;
        const int y = face;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
        auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
        auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
        Kokkos::deep_copy(rbuf_d, rbuf_h);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> y_face({1, 1}, {nz+1, nx+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_edge({1, 1}, {nz+1, nx+2});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_edge({1, 1}, {nz+2, nx+1});
        Kokkos::parallel_reduce("end_recv_tang_e_norm_b<YZX> y face", y_face, KOKKOS_LAMBDA(const int z, const int x, double& error) {
            const double w1 = static_cast<double>(rbuf_d((z-1)*nx + (x-1)));
            const double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = static_cast<float>(0.5*(w1+w2));
            error += (w1-w2)*(w1-w2);
        }, err_temp);
        err += err_temp;
        err_temp = 0.0f;
        Kokkos::parallel_reduce("end_recv_tang_e_norm_b<YZX> zx edge", zx_edge, KOKKOS_LAMBDA(const int z, const int x, double& error) {
            double w1 = static_cast<double>(rbuf_d(nz*nx + 2*((z-1)*(nx+1) + (x-1))));
            double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ez) = static_cast<float>(0.5*(w1+w2));
            error += (w1-w2)*(w1-w2);
            w1 = static_cast<double>(rbuf_d(nz*nx + 2*((z-1)*(nx+1) + (x-1)) + 1));
            w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcaz) = static_cast<float>(0.5*(w1+w2));
        }, err_temp);
        err += err_temp;
        err_temp = 0.0f;
        Kokkos::parallel_reduce("end_recv_tang_e_norm_b<ZXY> xz edge", xz_edge, KOKKOS_LAMBDA(const int z, const int x, double& error) {
            double w1 = static_cast<double>(rbuf_d(nz*nx + 2*nz*(nx+1) + 2*((z-1)*nx + (x-1))));
            double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = static_cast<float>(0.5*(w1+w2));
            error += (w1-w2)*(w1-w2);
            w1 = static_cast<double>(rbuf_d(nz*nx + 2*nz*(nx+1) + 2*((z-1)*nx + (x-1)) + 1));
            w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax) = static_cast<float>(0.5*(w1+w2));
        }, err_temp);
        err += err_temp;
    }
    return err;
}
template<> 
double 
end_recv_tang_e_norm_b<ZXY>(field_array_t* fa, const int i, const int j, const int k) {
    double err=0.0, err_temp=0.0;
    float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,fa->g));
    if(p) {
        const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
        const int face = (i+j+k)<0 ? nz+1 : 1;
        const int z = face;
        k_field_t& k_field = fa->k_f_d;
        Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
        auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
        auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
        Kokkos::deep_copy(rbuf_d, rbuf_h);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> z_face({1, 1}, {ny+1, nx+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xy_edge({1, 1}, {ny+2, nx+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_edge({1, 1}, {ny+1, nx+2});
        Kokkos::parallel_reduce("end_recv_tang_e_norm_b<ZXY> z face", z_face, KOKKOS_LAMBDA(const int y, const int x, double& error) {
            const double w1 = static_cast<double>(rbuf_d((y-1)*nx + (x-1)));
            const double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbz) = static_cast<float>(0.5*(w1+w2));
            error += (w1-w2)*(w1-w2);
        }, err_temp);
        err += err_temp;
        err_temp = 0.0f;
        Kokkos::parallel_reduce("end_recv_tang_e_norm_b<ZXY> xy edge", xy_edge, KOKKOS_LAMBDA(const int y, const int x, double& error) {
            double w1 = static_cast<double>(rbuf_d(ny*nx + 2*((y-1)*nx + (x-1))));
            double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ex) = static_cast<float>(0.5*(w1+w2));
            error += (w1-w2)*(w1-w2);
            w1 = static_cast<double>(rbuf_d(ny*nx + 2*((y-1)*nx + (x-1)) + 1));
            w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcax) = static_cast<float>(0.5*(w1+w2));
        }, err_temp);
        err += err_temp;
        err_temp = 0.0f;
        Kokkos::parallel_reduce("end_recv_tang_e_norm_b<ZXY> yx edge", yx_edge, KOKKOS_LAMBDA(const int y, const int x, double& error) {
            double w1 = static_cast<double>(rbuf_d(nx*ny + 2*nx*(ny+1) + 2*((y-1)*(nx+1) + (x-1))));
            double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::ey) = static_cast<float>(0.5*(w1+w2));
            error += (w1-w2)*(w1-w2);
            w1 = static_cast<double>(rbuf_d(nx*ny + 2*nx*(ny+1) + 2*((y-1)*(nx+1) + (x-1)) + 1));
            w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay));
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tcay) = static_cast<float>(0.5*(w1+w2));
        }, err_temp);
        err += err_temp;
    }
    return err;
}

template<typename T> void end_send_tang_e_norm_b(field_array_t* fa, const int i, const int j, const int k) {
    end_send_port_k(i,j,k,fa->g);
}

double
synchronize_tang_e_norm_b_kokkos( field_array_t * RESTRICT fa ) {
    const grid_t * RESTRICT g = fa->g;
    double err = 0, gerr;

    if( !fa ) ERROR(( "Bad args" ));

    k_local_adjust_tang_e( fa, g );
    k_local_adjust_norm_b( fa, g );

    // Exchange x-faces
    begin_recv_tang_e_norm_b<XYZ>(fa, -1,  0,  0 );
    begin_recv_tang_e_norm_b<XYZ>(fa,  1,  0,  0 );
    begin_send_tang_e_norm_b<XYZ>(fa, -1,  0,  0 );
    begin_send_tang_e_norm_b<XYZ>(fa,  1,  0,  0 );
    err += end_recv_tang_e_norm_b<XYZ>(fa,  -1,  0,  0);
    err += end_recv_tang_e_norm_b<XYZ>(fa,   1,  0,  0);
    end_send_tang_e_norm_b<XYZ>(fa,  -1,  0,  0);
    end_send_tang_e_norm_b<XYZ>(fa,   1,  0,  0);

    // Exchange y-faces
    begin_recv_tang_e_norm_b<YZX>(fa,  0, -1,  0);
    begin_recv_tang_e_norm_b<YZX>(fa,  0,  1,  0);
    begin_send_tang_e_norm_b<YZX>(fa,  0, -1,  0);
    begin_send_tang_e_norm_b<YZX>(fa,  0,  1,  0);
    err += end_recv_tang_e_norm_b<YZX>(fa,   0, -1,  0);
    err += end_recv_tang_e_norm_b<YZX>(fa,   0,  1,  0);
    end_send_tang_e_norm_b<YZX>(fa,   0, -1,  0);
    end_send_tang_e_norm_b<YZX>(fa,   0,  1,  0);

    // Exchange z-faces
    begin_recv_tang_e_norm_b<ZXY>(fa,  0,  0, -1);
    begin_recv_tang_e_norm_b<ZXY>(fa,  0,  0,  1);
    begin_send_tang_e_norm_b<ZXY>(fa,  0,  0, -1);
    begin_send_tang_e_norm_b<ZXY>(fa,  0,  0,  1);
    err += end_recv_tang_e_norm_b<ZXY>(fa,   0,  0, -1);
    err += end_recv_tang_e_norm_b<ZXY>(fa,   0,  0,  1);
    end_send_tang_e_norm_b<ZXY>(fa,   0,  0, -1);
    end_send_tang_e_norm_b<ZXY>(fa,   0,  0,  1);

  mp_allsum_d( &err, &gerr, 1 );
  return gerr;
}

template <typename Face> 
void 
begin_recv_jf(const grid_t* g, field_array_t *fa, int i, int j, int k) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  if constexpr(std::is_same<Face,XYZ>::value) {
    size = (ny*(nz+1) + nz*(ny+1) + 1)*sizeof(float);
  } else if constexpr(std::is_same<Face,YZX>::value) {
    size = (nz*(nx+1) + nx*(nz+1) + 1)*sizeof(float);
  } else if constexpr(std::is_same<Face,ZXY>::value) {
    size = (nx*(ny+1) + ny*(nx+1) + 1)*sizeof(float);
  }
// CPU
  begin_recv_port_k(i,j,k,size,g,reinterpret_cast<char*>(rbuf_h.data()));
// GPU
//  begin_recv_port_k(i,j,k,size,g,reinterpret_cast<char*>(rbuf_d.data()));
}

//template <> void begin_recv_jf<XYZ>(const grid_t* g, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const int size = (ny*(nz+1) + nz*(ny+1) + 1)*sizeof(float);
//// Original
////    begin_recv_port(i,j,k,size,g);
//// CPU
//    begin_recv_port_k(i,j,k,size,g,reinterpret_cast<char*>(rbuf_h.data()));
//// GPU
////    begin_recv_port_k(i,j,k,size,g,reinterpret_cast<char*>(rbuf_d.data()));
//}
//template <> void begin_recv_jf<YZX>(const grid_t* g, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const int size = (nz*(nx+1) + nx*(nz+1) + 1)*sizeof(float);
//// Original
////    begin_recv_port(i,j,k,size,g);
//// CPU
//    begin_recv_port_k(i,j,k,size,g,reinterpret_cast<char*>(rbuf_h.data()));
//// GPU
////    begin_recv_port_k(i,j,k,size,g,reinterpret_cast<char*>(rbuf_d.data()));
//}
//template <> void begin_recv_jf<ZXY>(const grid_t* g, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    const int size = (nx*(ny+1) + ny*(nx+1) + 1)*sizeof(float);
//// Original
////    begin_recv_port(i,j,k,size,g);
//// CPU
//    begin_recv_port_k(i,j,k,size,g,reinterpret_cast<char*>(rbuf_h.data()));
//// GPU
////    begin_recv_port_k(i,j,k,size,g,reinterpret_cast<char*>(rbuf_d.data()));
//}

template <typename Face> 
void 
begin_send_jf(const grid_t* g, field_array_t* fa, int i, int j, int k) {}

template<> 
void 
begin_send_jf<XYZ>(const grid_t* g, field_array_t* fa, int i, int j, int k) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const int size = ( 1 + ny*(nz+1) + nz*(ny+1) )*sizeof(float);
    k_field_t& k_field = fa->k_f_d;
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    const int face = (i+j+k)<0 ? 1 : nx+1;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> yz_edge({1, 1}, {nz+2, ny+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_edge({1, 1}, {nz+1, ny+2});
    Kokkos::parallel_for("begin_send_jf<XYZ>: yz_edge_loop", yz_edge, KOKKOS_LAMBDA(const int z, const int y) {
        const int x = face;
        sbuf_d(1 + (z-1)*ny + (y-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy);
    });
    Kokkos::parallel_for("begin_send_jf<XYZ>: zy_edge_loop", zy_edge, KOKKOS_LAMBDA(const int z, const int y) {
        const int x = face;
        sbuf_d(1 + (nz+1)*ny + (z-1)*(ny+1) + (y-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz);
    });
    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = g->dx;
    begin_send_port_k(i,j,k,size,g, reinterpret_cast<char*>(sbuf_h.data()));
}
template<> void begin_send_jf<YZX>(const grid_t* g, field_array_t* fa, int i, int j, int k) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const int size = ( 1 + nz*(nx+1) + nx*(nz+1) )*sizeof(float);
    k_field_t& k_field = fa->k_f_d;
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    const int face = (i+j+k)<0 ? 1 : ny+1;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_edge({1, 1}, {nz+1, nx+2});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_edge({1, 1}, {nz+2, nx+1});
    Kokkos::parallel_for("begin_send_jf<YZX>: zx_edge_loop", zx_edge, KOKKOS_LAMBDA(const int z, const int x) {
        const int y = face;
        sbuf_d(1 + (z-1)*(nx+1) + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz);
    });
    Kokkos::parallel_for("begin_send_jf<YZX>: zx_edge_loop", xz_edge, KOKKOS_LAMBDA(const int z, const int x) {
        const int y = face;
        sbuf_d(1 + (nx+1)*nz + (z-1)*nx + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx);
    });
    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = g->dy;
    begin_send_port_k(i,j,k,size,g, reinterpret_cast<char*>(sbuf_h.data()));
}
template<> 
void 
begin_send_jf<ZXY>(const grid_t* g, field_array_t* fa, int i, int j, int k) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const int size = ( 1 + nx*(ny+1) + ny*(nx+1) )*sizeof(float);
    k_field_t& k_field = fa->k_f_d;
    Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
    auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
    auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
    const int face = (i+j+k)<0 ? 1 : nz+1;
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> xy_edge({1, 1}, {ny+2, nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_edge({1, 1}, {ny+1, nx+2});
    Kokkos::parallel_for("begin_send_jf<ZXY>: xy_edge_loop", xy_edge, KOKKOS_LAMBDA(const int y, const int x) {
        const int z = face;
        sbuf_d(1 + (y-1)*nx + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx);
    });
    Kokkos::parallel_for("begin_send_jf<ZXY>: yx_edge_loop", yx_edge, KOKKOS_LAMBDA(const int y, const int x) {
        const int z = face;
        sbuf_d(1 + (ny+1)*nx + (y-1)*(nx+1) + (x-1)) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy);
    });
    Kokkos::deep_copy(sbuf_h, sbuf_d);
    sbuf_h(0) = g->dz;
    begin_send_port_k(i, j, k, size, g, reinterpret_cast<char*>(sbuf_h.data()));
}

template <typename Face> 
void 
end_recv_jf(const grid_t* g, field_array_t* fa, int i, int j, int k) {}

template<> 
void 
end_recv_jf<XYZ>(const grid_t* g, field_array_t* fa, int i, int j, int k) {
    float* p = reinterpret_cast<float*> (end_recv_port_k(i,j,k,g));
    k_field_t& k_field = fa->k_f_d;
    if(p) {
        const int nx = g->nx, ny = g->ny, nz = g->nz;
        Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
        auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
        auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
        const int face = (i+j+k)<0 ? nx+1 : 1;
        float rw  = rbuf_h(0);
        float lw  = rw + g->dx;
        rw /= lw;
        lw  = g->dx/lw;
        lw += lw;
        rw += rw;
        Kokkos::deep_copy(rbuf_d, rbuf_h);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yz_edge({1, 1}, {nz+2, ny+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_edge({1, 1}, {nz+1, ny+2});
        Kokkos::parallel_for("sync_jf: end_recv_jf<XYZ>: yz_edge_loop", yz_edge, KOKKOS_LAMBDA(const int z, const int y) {
            const int x = face;
            float jfy = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy) = lw*jfy + rw*rbuf_d(1 + (z-1)*ny + (y-1));
        });
        Kokkos::parallel_for("sync_jf: end_recv_jf<XYZ>: zy_edge_loop", zy_edge, KOKKOS_LAMBDA(const int z, const int y) {
            const int x = face;
            float jfz = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz) = lw*jfz + rw*rbuf_d(1 + (nz+1)*ny + (z-1)*(ny+1) + (y-1));
        });

    }
}
template<> 
void 
end_recv_jf<YZX>(const grid_t* g, field_array_t* fa, int i, int j, int k) {
    float* p = reinterpret_cast<float*> (end_recv_port_k(i,j,k,g));
    k_field_t& k_field = fa->k_f_d;
    if(p) {
        const int nx = g->nx, ny = g->ny, nz = g->nz;
        Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
        auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
        auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
        const int face = (i+j+k)<0 ? ny+1 : 1;
        float rw  = rbuf_h(0);
        float lw  = rw + g->dy;
        rw /= lw;
        lw  = g->dy/lw;
        lw += lw;
        rw += rw;
        Kokkos::deep_copy(rbuf_d, rbuf_h);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zx_edge({1, 1}, {nz+1, nx+2});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_edge({1, 1}, {nz+2, nx+1});
        Kokkos::parallel_for("sync_jf: end_recv_jf<YZX>: zx_edge_loop", zx_edge, KOKKOS_LAMBDA(const int z, const int x) {
            const int y = face;
            float jfz = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfz) = lw * jfz + rw * rbuf_d(1 + (z-1)*(nx+1) + (x-1));
        });
        Kokkos::parallel_for("sync_jf: end_recv_jf<YZX>: xz_edge_loop", xz_edge, KOKKOS_LAMBDA(const int z, const int x) {
            const int y = face;
            float jfx = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx)  = lw * jfx + rw * rbuf_d(1 + nz*(nx+1) + (z-1)*nx + (x-1));
        });
    }
}
template<> 
void 
end_recv_jf<ZXY>(const grid_t* g, field_array_t* fa, int i, int j, int k) {

    float* p = reinterpret_cast<float*> (end_recv_port_k(i,j,k,g));
    k_field_t& k_field = fa->k_f_d;
    if(p) {
        const int nx = g->nx, ny = g->ny, nz = g->nz;
        Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
        auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
        auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
        const int face = (i+j+k)<0 ? nz+1 : 1;
        float rw  = rbuf_h(0);
        float lw  = rw + g->dz;
        rw /= lw;
        lw  = g->dz/lw;
        lw += lw;
        rw += rw;
        Kokkos::deep_copy(rbuf_d, rbuf_h);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xy_edge({1, 1}, {ny+2, nx+1});
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_edge({1, 1}, {ny+1, nx+2});
        Kokkos::parallel_for("sync_jf: end_recv_jf<ZXY>: xy_edge_loop", xy_edge, KOKKOS_LAMBDA(const int y, const int x) {
            const int z = face;
            float jfx = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfx) = lw * jfx + rw * rbuf_d(1 + (y-1)*nx + (x-1));
        });
        Kokkos::parallel_for("sync_jf: end_recv_jf<ZXY>: yx_edge_loop", yx_edge, KOKKOS_LAMBDA(const int y, const int x) {
            const int z = face;
            float jfy = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jfy) = lw * jfy + rw * rbuf_d(1 + (ny+1)*nx + (y-1)*(nx+1) + (x-1));
        });
    }
}
template <typename T> void end_send_jf(const grid_t* g, int i, int j, int k) {
    end_send_port_k(i,j,k,g);
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

void k_synchronize_jf(field_array_t* RESTRICT fa) {
    if(!fa) ERROR(( "Bad args" ));
    grid_t* RESTRICT g = fa->g;

    k_local_adjust_jf(fa, g);

    // Exchange x-faces
    begin_recv_jf<XYZ>(g, fa, -1, 0, 0);
    begin_recv_jf<XYZ>(g, fa,  1, 0, 0);
    begin_send_jf<XYZ>(g, fa, -1, 0, 0);
    begin_send_jf<XYZ>(g, fa,  1, 0, 0);
    end_recv_jf<XYZ>(g, fa, -1, 0, 0);
    end_recv_jf<XYZ>(g, fa,  1, 0, 0);
    end_send_jf<XYZ>(g, -1, 0, 0);
    end_send_jf<XYZ>(g,  1, 0, 0);

    // Exchange y-faces
    begin_recv_jf<YZX>(g, fa, 0, -1, 0);
    begin_recv_jf<YZX>(g, fa, 0,  1, 0);
    begin_send_jf<YZX>(g, fa, 0, -1, 0);
    begin_send_jf<YZX>(g, fa, 0,  1, 0);
    end_recv_jf<YZX>(g, fa, 0, -1, 0);
    end_recv_jf<YZX>(g, fa, 0,  1, 0);
    end_send_jf<YZX>(g, 0, -1, 0);
    end_send_jf<YZX>(g, 0,  1, 0);

    // Exchange z-faces
    begin_recv_jf<ZXY>(g, fa, 0, 0, -1);
    begin_recv_jf<ZXY>(g, fa, 0, 0,  1);
    begin_send_jf<ZXY>(g, fa, 0, 0, -1);
    begin_send_jf<ZXY>(g, fa, 0, 0,  1);
    end_recv_jf<ZXY>(g, fa, 0, 0, -1);
    end_recv_jf<ZXY>(g, fa, 0, 0,  1);
    end_send_jf<ZXY>(g, 0, 0, -1);
    end_send_jf<ZXY>(g, 0, 0,  1);

}

template <typename Face> 
void 
begin_recv_rho(field_array* fa, int i, int j, int k) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  if constexpr(std::is_same<Face,XYZ>::value) {
    size = ( 1 + 2*(ny+1)*(nz+1) )*sizeof(float);
  } else if constexpr(std::is_same<Face,YZX>::value) {
    size = ( 1 + 2*(nz+1)*(nx+1) )*sizeof(float);
  } else if constexpr(std::is_same<Face,ZXY>::value) {
    size = ( 1 + 2*(nx+1)*(ny+1) )*sizeof(float);
  }
// CPU
  begin_recv_port_k(i,j,k,size,fa->g,reinterpret_cast<char*>(rbuf_h.data()));
// GPU
//  begin_recv_port_k(i,j,k,size,fa->g,reinterpret_cast<char*>(rbuf_d.data()));
}

//template<> void begin_recv_rho<XYZ>(field_array* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    begin_recv_port_k(i,j,k, ( 1 + 2*(ny+1)*(nz+1) )*sizeof(float), fa->g, reinterpret_cast<char*>(rbuf_h.data()));
//}
//template<> void begin_recv_rho<YZX>(field_array* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    begin_recv_port_k(i,j,k, ( 1 + 2*(nz+1)*(nx+1) )*sizeof(float), fa->g, reinterpret_cast<char*>(rbuf_h.data()));
//}
//template<> void begin_recv_rho<ZXY>(field_array* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf_d, Kokkos::View<float*>::HostMirror& rbuf_h) {
//    begin_recv_port_k(i,j,k, ( 1 + 2*(nx+1)*(ny+1) )*sizeof(float), fa->g, reinterpret_cast<char*>(rbuf_h.data()));
//}

template<typename Face> 
void 
begin_send_rho(field_array_t* fa, int i, int j, int k) {}

template<> 
void 
begin_send_rho<XYZ>(field_array_t* fa, int i, int j, int k) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  const int size = ( 1 + 2*(ny+1)*(nz+1) )*sizeof(float);
  k_field_t& k_field = fa->k_f_d;
  Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  const int face = (i+j+k)<0 ? 1 : nx+1;
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_node({1, 1}, {nz+2, ny+2});
  Kokkos::parallel_for("begin_send_rho<XYZ>: x_node_loop", x_node, KOKKOS_LAMBDA(const int z, const int y) {
      const int x = face;
      const int idx_f = 1 + 2*((z-1)*(ny+1) + (y-1));
      const int idx_b = 1 + 2*((z-1)*(ny+1) + (y-1)) + 1;
      sbuf_d(idx_f) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
      sbuf_d(idx_b) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
  });
  Kokkos::deep_copy(sbuf_h, sbuf_d);
  sbuf_h(0) = fa->g->dx;
  begin_send_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(sbuf_h.data()));
}
template<> 
void 
begin_send_rho<YZX>(field_array_t* fa, int i, int j, int k) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size = ( 1 + 2*(nz+1)*(nx+1) )*sizeof(float);
  k_field_t& k_field = fa->k_f_d;
  int face = (i+j+k)<0 ? 1 : ny+1;
  Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> y_node({1, 1}, {nz+2, nx+2});
  Kokkos::parallel_for("begin_send_rho<YZX>: y_node_loop", y_node, KOKKOS_LAMBDA(const int z, const int x) {
      const int y = face;
      const int idx_f = 1 + 2*((z-1)*(nx+1) + (x-1));
      const int idx_b = 1 + 2*((z-1)*(nx+1) + (x-1)) + 1;
      sbuf_d(idx_f) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
      sbuf_d(idx_b) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
  });
  Kokkos::deep_copy(sbuf_h, sbuf_d);
  sbuf_h(0) = fa->g->dy;
  begin_send_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(sbuf_h.data()));
}
template<> 
void 
begin_send_rho<ZXY>(field_array_t* fa, int i, int j, int k) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size = ( 1 + 2*(nx+1)*(ny+1) )*sizeof(float);
  k_field_t& k_field = fa->k_f_d;
  Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  int face = (i+j+k)<0 ? 1 : nz+1;
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> z_node({1, 1}, {ny+2, nx+2});
  Kokkos::parallel_for("begin_send_rho<ZXY>: z_node_loop", z_node, KOKKOS_LAMBDA(const int y, const int x) {
    const int z = face;
    const int idx_f = 1 + 2*((y-1)*(nx+1) + (x-1));
    const int idx_b = 1 + 2*((y-1)*(nx+1) + (x-1)) + 1;
    sbuf_d(idx_f) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
    sbuf_d(idx_b) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
  });
  Kokkos::deep_copy(sbuf_h, sbuf_d);
  sbuf_h(0) = fa->g->dz;
  begin_send_port_k(i,j,k,size,fa->g, reinterpret_cast<char*>(sbuf_h.data()));
}

template <typename Face> 
void 
end_recv_rho(field_array_t* fa, int i, int j, int k) {}

template<> 
void 
end_recv_rho<XYZ>(field_array_t* fa, int i, int j, int k) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  const int face = (i+j+k)<0 ? nx+1 : 1;
  float hlw, hrw, lw, rw;
  float* p = reinterpret_cast<float *>(end_recv_port_k(i,j,k,fa->g));
  k_field_t& k_field = fa->k_f_d;
  if( p ) {
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    hrw  = rbuf_h(0);
    hlw  = hrw + fa->g->dx;
    hrw /= hlw;
    hlw  = fa->g->dx/hlw;
    lw   = hlw + hlw;
    rw   = hrw + hrw;
    Kokkos::deep_copy(rbuf_d, rbuf_h);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> x_node({1, 1}, {nz+2, ny+2});
    Kokkos::parallel_for("sync_rho: end_recv_rho<XYZ>: x_node_loop", x_node, KOKKOS_LAMBDA(const int z, const int y) {
      const int x = face;
      const int idx_f = 1 + 2*((z-1)*(ny+1) + (y-1));
      const int idx_b = 1 + 2*((z-1)*(ny+1) + (y-1)) + 1;
      const float rhof = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
      const float rhob = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof) = lw*rhof + rw*rbuf_d(idx_f);
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob) = hlw*rhob + hrw*rbuf_d(idx_b);
    });
  }
}
template<> 
void 
end_recv_rho<YZX>(field_array_t* fa, int i, int j, int k) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  const int face = (i+j+k)<0 ? ny+1 : 1;
  float hlw, hrw, lw, rw;
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,fa->g));
  if(p) {
    k_field_t& k_field = fa->k_f_d;
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    hrw  = rbuf_h(0);
    hlw  = hrw + fa->g->dy;
    hrw /= hlw;
    hlw  = fa->g->dy/hlw;
    lw   = hlw + hlw;
    rw   = hrw + hrw;
    Kokkos::deep_copy(rbuf_d, rbuf_h);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> y_node({1, 1}, {nz+2, nx+2});
    Kokkos::parallel_for("sync_rho: end_recv_rho<YZX>: y_node_loop", y_node, KOKKOS_LAMBDA(const int z, const int x) {
      const int y = face;
      const int idx_f = 1 + 2*((z-1)*(nx+1) + (x-1));
      const int idx_b = 1 + 2*((z-1)*(nx+1) + (x-1)) + 1;
      const float rhof = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
      const float rhob = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof) = lw*rhof + rw*rbuf_d(idx_f);
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob) = hlw*rhob + hrw*rbuf_d(idx_b);
    });
  }
}
template<> 
void 
end_recv_rho<ZXY>(field_array_t* fa, int i, int j, int k) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  const int face = (i+j+k)<0 ? nz+1 : 1;
  float hlw, hrw, lw, rw;
  float* p = reinterpret_cast<float *>(end_recv_port_k(i,j,k,fa->g));
  k_field_t& k_field = fa->k_f_d;
  if( p ) {
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    hrw  = rbuf_h(0);
    hlw  = hrw + fa->g->dz;
    hrw /= hlw;
    hlw  = fa->g->dz/hlw;
    lw   = hlw + hlw;
    rw   = hrw + hrw;
    Kokkos::deep_copy(rbuf_d, rbuf_h);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> z_node({1, 1}, {ny+2, nx+2});
    Kokkos::parallel_for("sync_rho: end_recv_rho<ZXY>: z_node_loop", z_node, KOKKOS_LAMBDA(const int y, const int x) {
      const int z = face;
      const int idx_f = 1 + 2*((y-1)*(nx+1) + (x-1));
      const int idx_b = 1 + 2*((y-1)*(nx+1) + (x-1)) + 1;
      const float rhof = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
      const float rhob = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof) = lw*rhof + rw*rbuf_d(idx_f);
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob) = hlw*rhob + hrw*rbuf_d(idx_b);
    });
  }
}

template <typename T> void end_send_rho(field_array* fa, int i, int j, int k) {
    end_send_port_k(i,j,k,fa->g);
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

void k_synchronize_rho(field_array_t* RESTRICT fa) {
    if(!fa) ERROR(( "Bad args" ));
    grid_t* RESTRICT g = fa->g;

    k_local_adjust_rhof(fa, g);
    k_local_adjust_rhob(fa, g);

    // Exchange x-faces
    begin_recv_rho<XYZ>(fa, -1, 0, 0);
    begin_recv_rho<XYZ>(fa,  1, 0, 0);
    begin_send_rho<XYZ>(fa, -1, 0, 0);
    begin_send_rho<XYZ>(fa,  1, 0, 0);
    end_recv_rho<XYZ>(  fa, -1, 0, 0);
    end_recv_rho<XYZ>(  fa,  1, 0, 0);
    end_send_rho<XYZ>(  fa, -1, 0, 0);
    end_send_rho<XYZ>(  fa,  1, 0, 0);

    // Exchange y-faces
    begin_recv_rho<YZX>(fa, 0, -1, 0);
    begin_recv_rho<YZX>(fa, 0,  1, 0);
    begin_send_rho<YZX>(fa, 0, -1, 0);
    begin_send_rho<YZX>(fa, 0,  1, 0);
    end_recv_rho<YZX>(  fa, 0, -1, 0);
    end_recv_rho<YZX>(  fa, 0,  1, 0);
    end_send_rho<YZX>(  fa, 0, -1, 0);
    end_send_rho<YZX>(  fa, 0,  1, 0);

    // Exchange z-faces
    begin_recv_rho<ZXY>(fa, 0, 0, -1);
    begin_recv_rho<ZXY>(fa, 0, 0,  1);
    begin_send_rho<ZXY>(fa, 0, 0, -1);
    begin_send_rho<ZXY>(fa, 0, 0,  1);
    end_recv_rho<ZXY>(  fa, 0, 0, -1);
    end_recv_rho<ZXY>(  fa, 0, 0,  1);
    end_send_rho<ZXY>(  fa, 0, 0, -1);
    end_send_rho<ZXY>(  fa, 0, 0,  1);

}

