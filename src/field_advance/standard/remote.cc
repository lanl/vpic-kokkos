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

//#define TEST_MPI_KOKKOS

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
typedef class XY {} XY;
typedef class YX {} YX;
typedef class XZ {} XZ;
typedef class ZX {} ZX;
typedef class YZ {} YZ;
typedef class ZY {} ZY;

template <typename T> void begin_recv_kokkos(const grid_t* g, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf) {
    int nX, nY, nZ;
    if (std::is_same<T, XYZ>::value) {
        nX = nx, nY = ny, nZ = nz;
    } else if (std::is_same<T, YZX>::value) {
        nX = ny, nY = nz, nZ = nx;
    } else if (std::is_same<T, ZXY>::value) {
        nX = nz, nY = nx, nZ = ny;
    }
    int size = (1 + nY*(nZ+1) + nZ*(nY+1)) * sizeof(float);
    int port = BOUNDARY(-i,-j,-k);
    int tag = BOUNDARY(i,j,k);
//    begin_recv_port_kokkos(g, port, size, tag, reinterpret_cast<char*>(rbuf.data()));
   begin_recv_port_k(i,j,k, size, g, reinterpret_cast<char*>(rbuf.data()));
//    begin_recv_port(i,j,k,size, g);

}

template <typename T> void begin_recv(int i, int j, int k, int nx, int ny, int nz, const grid_t* g) {
    int nX, nY, nZ;
    if (std::is_same<T, XYZ>::value) {
        nX = nx, nY = ny, nZ = nz;
    } else if (std::is_same<T, YZX>::value) {
        nX = ny, nY = nz, nZ = nx;
    } else if (std::is_same<T, ZXY>::value) {
        nX = nz, nY = nx, nZ = ny;
    }
    begin_recv_port(i,j,k,(1 + nY*(nZ+1) + nZ*(nY+1))*sizeof(float),g);
}

template<> void begin_recv<XYZ>(int i, int j, int k, int nx, int ny, int nz, const grid_t* g) {
    begin_recv_port(i,j,k,(1+ny*(nz+1)+nz*(ny+1))*sizeof(float),g);
}
template<> void begin_recv<YZX>(int i, int j, int k, int nx, int ny, int nz, const grid_t* g) {
    begin_recv_port(i,j,k,(1+nz*(nx+1)+nx*(nz+1))*sizeof(float),g);
}
template<> void begin_recv<ZXY>(int i, int j, int k, int nx, int ny, int nz, const grid_t* g) {
    begin_recv_port(i,j,k,(1+nx*(ny+1)+ny*(nx+1))*sizeof(float),g);
}

/*
template <typename T> void begin_send_kokkos(k_field_t& k_field, grid_t* g, float* ALIGNED(128) sbuf, int i, int j, int k) {
    int nX, nY, nZ, face, size;
    float dX;
    Kokkos::MDRangePolicy<Kokkos::Rank<3> > policy_1, policy_2;
    field_var cbY, cbZ;
    k_mpi_t mp = g->k_mp_d;
    std::string ZY_loop_name, YZ_loop_name;

    if(std::is_same<T, XYZ>::value) {
        dX = g->dx;
        nX = g->nx;
        nY = g->ny;
        nZ = g->nz;
        cbY = field_var::cby;
        cbZ = field_var::cbz;
        face = (i+j+k)<0 ? 1 : nX;
        policy_1 = MDRangePolicy({face, 1, 1}, {face+1, g->ny+2, g->nz+1}); //ZY
        policy_2 = MDRangePolicy({face, 1, 1}, {face+1, g->ny+1, g->nz+2}); //YZ
        ZY_loop_name = "begin_send<XYZ>: ZY Edge Loop"
        YZ_loop_name = "begin_send<XYZ>: YZ Edge Loop"
    } else if(std::is_same<T, YZX>::value) {
        dX = g->dy;
        nX = g->ny;
        nY = g->nz;
        nZ = g->nx;
        cbY = field_var::cbz;
        cbZ = field_var::cbx;
        face = (i+j+k)<0 ? 1 : nY;
        policy_1 = MDRangePolicy({1, face, 1}, {g->nx+1, face+1, g->nz+2}); // XZ
        policy_2 = MDRangePolicy({1, face, 1}, {g->nx+2, face+1, g->nz+1}); // ZX
        ZY_loop_name = "begin_send<XYZ>: XZ Edge Loop"
        YZ_loop_name = "begin_send<XYZ>: ZX Edge Loop"
    } else if(std::is_same<T, ZXY>::value) {
        dX = g->dz;
        nX = g->nz;
        nY = g->nx;
        nZ = g->ny;
        cbY = field_var::cbx;
        cbZ = field_var::cby;
        face = (i+j+k)<0 ? 1 : nZ;
        policy_1 = MDRangePolicy({1, 1, face}, {g->nx+2, g->ny+1, face+1}); // YX
        policy_2 = MDRangePolicy({1, 1, face}, {g->nx+1, g->ny+2, face+1}); // XY
        ZY_loop_name = "begin_send<XYZ>: YX Edge Loop"
        YZ_loop_name = "begin_send<XYZ>: XY Edge Loop"
    }

    size = (1+nY*(nZ+1) + nZ*(nY+1))*sizeof(float);
    sbuf[0] = dX;
    face = (i+j+k)<0 ? 1 : nX;
    Kokkos::parallel_for(ZY_loop_name, policy_1, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
        sbuf[1+(kk-1)*(nY+1) + (jj-1)] = k_field(VOXEL(x,y,z, nx,ny,nz), cbY);
    });
    Kokkos::parallel_for(YZ_loop_name, policy_2, KOKKOS_LAMBDA(const int ii, const int jj, const int kk) {
        sbuf[1 + (nY+1)*nZ + (kk-1)*nY + (jj-1)] = k_field(VOXEL(x,y,z, nx,ny,nz), cbY);
    });
    int port = BOUNDARY(i,j,k);
    int dst = g->bc[port];
    begin_send_port_kokkos(mp, g->max_ports, port, size, dst, sbuf);
}
*/
template <typename T> void begin_send_kokkos(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& sbuf) {}

template <> void begin_send_kokkos<XYZ>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& sbuf) {
    k_field_t k_field = fa->k_f_d;
    const size_t size = (1+ny*(nz+1)+nz*(ny+1))*sizeof(float); 
    float* p = static_cast<float*>(size_send_port( i, j, k, size, g ));        

    if( p ) {                                               
        Kokkos::View<float*> d_buf("Device buffer", (size/sizeof(float)));
        Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);

        int face = (i+j+k)<0 ? 1 : nx;			    
        float dx = g->dx;

        Kokkos::parallel_for("begin_send<XYZ>: ZY Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            if(zi == 0) {
                d_buf(0) = dx;
            }
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
        for(size_t idx = 0; idx < (size/sizeof(float)); idx++) {
            p[idx] = h_buf(idx);
        }

        Kokkos::deep_copy(sbuf, d_buf);

//        begin_send_port( i, j, k, size, g );                  
        begin_send_port_k(i,j,k,size,g,reinterpret_cast<char*>(sbuf.data()));
    }                                                       
}
template <> void begin_send_kokkos<YZX>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& sbuf) {
    k_field_t k_field = fa->k_f_d;
    size_t size = (1+nz*(nx+1)+nx*(nz+1))*sizeof(float); 
    float* p = static_cast<float *>(size_send_port( i, j, k, size, g ));        
    Kokkos::View<float*> d_buf("device buffer", (size/sizeof(float)));
    Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);

    if( p ) {                                               
      int face = (i+j+k)<0 ? 1 : ny;			    
        float dy = g->dy;
        Kokkos::parallel_for("begin_send<YZX>: XZ Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO), 
            KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            if(zi == 0) {
                d_buf(0) = dy;
            }
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
        Kokkos::deep_copy(sbuf, d_buf);

        for(size_t idx=0; idx<(size/sizeof(float)); idx++) {
            p[idx] = h_buf(idx);
        }
//        begin_send_port( i, j, k, size, g );                  
        begin_send_port_k(i,j,k,size,g,reinterpret_cast<char*>(sbuf.data()));
    }                                                       
}
template <> void begin_send_kokkos<ZXY>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& sbuf) {
    size_t size = (1+nx*(ny+1)+ny*(nx+1))*sizeof(float); 
    float* p = static_cast<float*>(size_send_port(i,j,k,size,g));
    k_field_t k_field = fa->k_f_d;
    Kokkos::View<float*> d_buf("device buffer", (size/sizeof(float)));
    Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);

    if(p){
        int face = (i+j+k)<0 ? 1 : nz;
        float dz = g->dz;
        Kokkos::parallel_for("begin_send<ZXY>: YX Edge Loop", KOKKOS_TEAM_POLICY_DEVICE(ny,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type & team_member) {
            size_t yi = team_member.league_rank();
            if(yi == 0) {
                d_buf(0) = dz;
            }
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
        Kokkos::deep_copy(sbuf, d_buf);
        for(size_t idx=0; idx<(size/sizeof(float)); idx++) {
            p[idx] = h_buf(idx);
        }
//        begin_send_port(i,j,k,size,g);
        begin_send_port_k(i,j,k,size,g,reinterpret_cast<char*>(sbuf.data()));
    }
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
//        for(size_t idx = 1; idx < size; idx++) {
//            p[idx] = h_buf(idx);
//        }
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
//        for(size_t idx=1; idx<size; idx++) {
//            p[idx] = h_buf(idx);
//        }
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
//        for(size_t idx=1; idx<size; idx++) {
//            p[idx] = h_buf(idx);
//        }
        begin_send_port(i,j,k,size*sizeof(float),g);
    }
}


void
kokkos_begin_remote_ghost_tang_b( field_array_t      * RESTRICT fa,
                           const grid_t *              g,
                            field_buffers_t&            f_buffers) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;

    begin_recv_kokkos<XYZ>(g,-1,0,0,nx,ny,nz, f_buffers.xyz_rbuf_neg);
    begin_recv_kokkos<YZX>(g,0,-1,0,nx,ny,nz, f_buffers.yzx_rbuf_neg);
    begin_recv_kokkos<ZXY>(g,0,0,-1,nx,ny,nz, f_buffers.zxy_rbuf_neg);
    begin_recv_kokkos<XYZ>(g,1,0,0,nx,ny,nz,  f_buffers.xyz_rbuf_pos);
    begin_recv_kokkos<YZX>(g,0,1,0,nx,ny,nz,  f_buffers.yzx_rbuf_pos);
    begin_recv_kokkos<ZXY>(g,0,0,1,nx,ny,nz,  f_buffers.zxy_rbuf_pos);

    begin_send_kokkos<XYZ>(g,fa,-1,0,0,nx,ny,nz, f_buffers.xyz_sbuf_neg);
    begin_send_kokkos<YZX>(g,fa,0,-1,0,nx,ny,nz, f_buffers.yzx_sbuf_neg);
    begin_send_kokkos<ZXY>(g,fa,0,0,-1,nx,ny,nz, f_buffers.zxy_sbuf_neg);
    begin_send_kokkos<XYZ>(g,fa,1,0,0,nx,ny,nz,  f_buffers.xyz_sbuf_pos);
    begin_send_kokkos<YZX>(g,fa,0,1,0,nx,ny,nz,  f_buffers.yzx_sbuf_pos);
    begin_send_kokkos<ZXY>(g,fa,0,0,1,nx,ny,nz,  f_buffers.zxy_sbuf_pos);

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

template<typename T> void end_recv_kokkos(const grid_t* g, field_array_t* RESTRICT field, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf) {}

template<> void end_recv_kokkos<XYZ>(const grid_t* g, field_array_t* RESTRICT field, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf) {
    end_recv_port_k(i,j,k,g);
//    float* p = static_cast<float*>(end_recv_port(i,j,k,g));
    size_t size = 1 + (ny+1)*nz + ny*(nz+1);
//    if(p) {
        Kokkos::View<float*> d_buf("Device buffer", size);
//        Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);
//        for(size_t idx = 0; idx < size; idx++) {
//            h_buf(idx) = p[idx];
//        }
//        Kokkos::deep_copy(d_buf, h_buf);
        Kokkos::deep_copy(d_buf, rbuf);

        k_field_t k_field = field->k_f_d;

        int face = (i+j+k)<0 ? nx+1 : 0;
        float dx = g->dx;
        Kokkos::parallel_for("end_recv<XYZ>: ZY Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            float lw = d_buf(0);
            float rw = (2.*dx) / (lw+dx);
            lw = (lw-dx)/(lw+dx);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (size_t yi) {
                k_field(VOXEL(face,yi+1,zi+1,nx,ny,nz), field_var::cby) = rw*d_buf(zi*(ny+1) + yi + 1) + lw*k_field(VOXEL(face+i,yi+1+j,zi+1+k,nx,ny,nz), field_var::cby);
            });
        });
        Kokkos::parallel_for("end_recv<XYZ>: YZ Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            float lw = d_buf(0);
            float rw = (2.*dx) / (lw+dx);
            lw = (lw-dx)/(lw+dx);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (size_t yi) {
                k_field(VOXEL(face,yi+1,zi+1,nx,ny,nz), field_var::cbz) = rw*d_buf((ny+1)*nz + zi*ny + yi + 1) + lw*k_field(VOXEL(face+i,yi+1+j,zi+1+k,nx,ny,nz), field_var::cbz);
            });
        });
//    }
}
template<> void end_recv_kokkos<YZX>(const grid_t* g, field_array_t* RESTRICT field, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf) {
    end_recv_port_k(i,j,k,g);
//    float* p = static_cast<float*>(end_recv_port(i,j,k,g));
    size_t size = 1 + nx*(nz+1) + (nx+1)*nz;
//    if(p) {
        Kokkos::View<float*> d_buf("Device buffer", size);
//        Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);
//        for(size_t idx = 0; idx < size; idx++) {
//            h_buf(idx) = p[idx];
//        }
//        Kokkos::deep_copy(d_buf, h_buf);
        Kokkos::deep_copy(d_buf, rbuf);

        k_field_t k_field = field->k_f_d;

        float dy = g->dy;
        int face = (i+j+k)<0 ? ny+1 : 0;
        Kokkos::parallel_for("end_recv<YZX>: XZ Edge loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t zi = team_member.league_rank();
            float lw = d_buf(0);
            float rw = (2.*dy) / (lw+dy);
            lw = (lw-dy)/(lw+dy);
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
            float lw = d_buf(0);
            float rw = (2.*dy) / (lw+dy);
            lw = (lw-dy)/(lw+dy);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (size_t xi) {
                const size_t x = xi + 1;
                const size_t y = face;
                const size_t z = zi + 1;
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cbx) = rw*d_buf(1 + nx*(nz+1) + zi*(nx+1) + xi) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cbx);
            });
        });
//    }
}
template<> void end_recv_kokkos<ZXY>(const grid_t* g, field_array_t* RESTRICT field, int i, int j, int k, int nx, int ny, int nz, Kokkos::View<float*>& rbuf) {
    end_recv_port_k(i,j,k,g);
//    float* p = static_cast<float*>(end_recv_port(i,j,k,g));
//    if(p) {
        size_t size = 1 + (nx+1)*ny + nx*(ny+1);
        k_field_t k_field = field->k_f_d;
        Kokkos::View<float*> d_buf("Device buffer", size);
//        Kokkos::View<float*>::HostMirror h_buf = create_mirror_view(d_buf);
//        for(size_t idx = 0; idx < size; idx++) {
//            h_buf(idx) = p[idx];
//        }
//        Kokkos::deep_copy(d_buf, h_buf);
        Kokkos::deep_copy(d_buf, rbuf);

        float dz = g->dz;
        int face = (i+j+k)<0 ? nz+1 : 0;
        Kokkos::parallel_for("end_recv<ZXY>: YX Edge loop", KOKKOS_TEAM_POLICY_DEVICE(ny,Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
            size_t yi = team_member.league_rank();
            float lw = d_buf(0);
            float rw = (2.*dz) / (lw+dz);
            lw = (lw-dz)/(lw+dz);
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
            float lw = d_buf(0);
            float rw = (2.*dz) / (lw+dz);
            lw = (lw-dz)/(lw+dz);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx), [=] (size_t xi) {
                const size_t x = xi + 1;
                const size_t y = yi + 1;
                const size_t z = face;
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cby) = rw*d_buf(1 + ny*(nx+1) + yi*nx + xi) + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cby);
            });
        });
//    }
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
//    end_send_port_kokkos(g, BOUNDARY(i,j,k));
    end_send_port_k(i,j,k, g);
//    end_send_port(i,j,k,g);
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

    end_recv_kokkos<XYZ>(g, field, -1, 0, 0, nx, ny, nz, f_buffers.xyz_rbuf_neg);
    end_recv_kokkos<YZX>(g, field, 0, -1, 0, nx, ny, nz, f_buffers.yzx_rbuf_neg);
    end_recv_kokkos<ZXY>(g, field, 0, 0, -1, nx, ny, nz, f_buffers.zxy_rbuf_neg);
    end_recv_kokkos<XYZ>(g, field, 1, 0, 0, nx, ny, nz, f_buffers.xyz_rbuf_pos);
    end_recv_kokkos<YZX>(g, field, 0, 1, 0, nx, ny, nz, f_buffers.yzx_rbuf_pos);
    end_recv_kokkos<ZXY>(g, field, 0, 0, 1, nx, ny, nz, f_buffers.zxy_rbuf_pos);

    end_send_kokkos<XYZ>(g, -1,0,0);
    end_send_kokkos<YZX>(g, 0,-1,0);
    end_send_kokkos<ZXY>(g, 0,0,-1);
    end_send_kokkos<XYZ>(g, 1,0,0);
    end_send_kokkos<YZX>(g, 0,1,0);
    end_send_kokkos<ZXY>(g, 0,0,1);
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

template <typename T> void begin_recv_rho(const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
/*
    int nX, nY, nZ;
    if (std::is_same<T, XYZ>::value) {
        nX = nx, nY = ny, nZ = nz;
    } else if (std::is_same<T, YZX>::value) {
        nX = ny, nY = nz, nZ = nx;
    } else if (std::is_same<T, ZXY>::value) {
        nX = nz, nY = nx, nZ = ny;
    }
    int size = (1 + 2*(nY+1)*(nZ+1))*sizeof(float);
    begin_recv_port(i,j,k, size, g);
*/
}
template<> void begin_recv_rho<XYZ>(const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
    begin_recv_port(i,j,k, ( 1 + 2*(ny+1)*(nz+1) )*sizeof(float), g);
}
template<> void begin_recv_rho<YZX>(const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
    begin_recv_port(i,j,k, ( 1 + 2*(nz+1)*(nx+1) )*sizeof(float), g);
}
template<> void begin_recv_rho<ZXY>(const grid_t* g, int i, int j, int k, int nx, int ny, int nz) {
    begin_recv_port(i,j,k, ( 1 + 2*(nx+1)*(ny+1) )*sizeof(float), g);
}
template<typename T> void begin_send_rho(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz) {
/*
    int nX, nY, nZ, face;
    float leading_dim;
    k_field_t& k_field = fa->k_f_d;
    std::initializer_list<int> start,end;
    if (std::is_same<T, XYZ>::value) {
        nX = nx, nY = ny, nZ = nz;
        leading_dim = g->dx;
        face = (i+j+k)<0 ? 1 : nx+1;
        start = {1,1,face};
        end = {nz+2,ny+2,face+1};
    } else if (std::is_same<T, YZX>::value) {
        nX = ny, nY = nz, nZ = nx;
        leading_dim = g->dy;
        face = (i+j+k)<0 ? 1 : ny+1;
        start = {1, face, 1};
        end = {nz+2, face+1, nx+2};
    } else if (std::is_same<T, ZXY>::value) {
        nX = nz, nY = nx, nZ = ny;
        leading_dim = g->dz;
        face = (i+j+k)<0 ? 1 : nz+1;
        start = {face, 1, 1};
        end = {face+1, ny+2, nx+2};
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<3> > node_policy(start,end);
    int size = (1 + 2*(nY+1)*(nZ+1));
    Kokkos::View<float*> d_buf("Device buffer", size);
    Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
    float* p = reinterpret_cast<float*>(size_send_port(i,j,k,size*sizeof(float),g));
    if(p) {
        Kokkos::parallel_for("begin_send_rho", node_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
            int idx_f, idx_b;
            if(std::is_same<T, XYZ>::value) {
                idx_f = 1 + 2*((z-1)*(ny+1) + y-1);
                idx_b = 1 + 2*((z-1)*(ny+1) + y-1) + 1;
            } else if (std::is_same<T, YZX>::value) {
                idx_f = 1 + 2*((z-1)*(nx+1) + x-1);
                idx_b = 1 + 2*((z-1)*(nx+1) + x-1) + 1;
            } else if (std::is_same<T, ZXY>::value) {
                idx_f = 1 + 2*((y-1)*(nx+1) + x-1);
                idx_b = 1 + 2*((y-1)*(nx+1) + x-1) + 1;
            }
            d_buf(idx_f) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
            d_buf(idx_b) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
        });
        Kokkos::deep_copy(h_buf, d_buf);
        Kokkos::parallel_for("Host copy to buffer", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(1,size), KOKKOS_LAMBDA(const int idx) {
            p[idx] = h_buf(idx);
        });
        p[0] = leading_dim;
        begin_send_port(i,j,k,size*sizeof(float),g);
    }
*/
}
template<> void begin_send_rho<XYZ>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz) {
    int size = ( 1 + 2*(ny+1)*(nz+1) )*sizeof(float);   
    float* p = (float *)size_send_port( i, j, k, size, g );    
    k_field_t& k_field = fa->k_f_d;
    if( p ) {                                           
      p[0] = g->dx;                               
      int face = (i+j+k)<0 ? 1 : nx+1;                    
        Kokkos::View<float*> d_buf("Device buffer", size/sizeof(float));
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        Kokkos::parallel_for("begin_send_rho<XYZ>: x_node_loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
            const int z = team_member.league_rank() + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const int yi) {
                const int x = face;
                const int y = yi + 1;
                const int idx_f = 1 + 2*((z-1)*(ny+1) + yi);
                const int idx_b = 1 + 2*((z-1)*(ny+1) + yi) + 1;
                d_buf(idx_f) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
                d_buf(idx_b) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
            });
        });
        Kokkos::deep_copy(h_buf, d_buf);
        h_buf(0) = g->dx;
        for(int idx = 0; idx < size/sizeof(float); idx++) {
            p[idx] = h_buf(idx);
        }
        p[0] = g->dx;
        begin_send_port(i,j,k,size,g);
    }
}
template<> void begin_send_rho<YZX>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz) {
    int size = ( 1 + 2*(nz+1)*(nx+1) )*sizeof(float);   
    float* p = (float *)size_send_port( i, j, k, size, g );    
    k_field_t& k_field = fa->k_f_d;
    if( p ) {                                           
      p[0] = g->dy;                               
      int face = (i+j+k)<0 ? 1 : ny+1;                    
        Kokkos::View<float*> d_buf("Device buffer", size/sizeof(float));
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        Kokkos::parallel_for("begin_send_rho<YZX>: y_node_loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
            const int z = team_member.league_rank() + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                const int y = face;
                const int x = xi + 1;
                const int idx_f = 1 + 2*((z-1)*(nx+1) + xi);
                const int idx_b = 1 + 2*((z-1)*(nx+1) + xi) + 1;
                d_buf(idx_f) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
                d_buf(idx_b) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
            });
        });
        Kokkos::deep_copy(h_buf, d_buf);
        h_buf(0) = g->dy;
        for(int idx = 0; idx < size/sizeof(float); idx++) {
            p[idx] = h_buf(idx);
        }
        p[0] = g->dy;
        begin_send_port(i,j,k,size,g);
    }
}
template<> void begin_send_rho<ZXY>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz) {
    int size = ( 1 + 2*(nx+1)*(ny+1) )*sizeof(float);   
    float* p = (float *)size_send_port( i, j, k, size, g );    
    k_field_t& k_field = fa->k_f_d;
    if( p ) {                                           
      p[0] = g->dz;                               
      int face = (i+j+k)<0 ? 1 : nz+1;                    
        Kokkos::View<float*> d_buf("Device buffer", size/sizeof(float));
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        Kokkos::parallel_for("begin_send_rho<ZXY>: z_node_loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
            const int y = team_member.league_rank() + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                const int z = face;
                const int x = xi + 1;
                const int idx_f = 1 + 2*((y-1)*(nx+1) + xi);
                const int idx_b = 1 + 2*((y-1)*(nx+1) + xi) + 1;
                d_buf(idx_f) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
                d_buf(idx_b) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
            });
        });
        Kokkos::deep_copy(h_buf, d_buf);
        h_buf(0) = g->dz;
        for(int idx = 0; idx < size/sizeof(float); idx++) {
            p[idx] = h_buf(idx);
        }
        p[0] = g->dz;
        begin_send_port(i,j,k,size,g);
    }
}
template <typename T> void end_recv_rho(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz) {
/*
    int nX, nY, nZ, face;
    float leading_dim;
    std::initializer_list<int> start, end;
    k_field_t& k_field = fa->k_f_d;
    if(std::is_same<T, XYZ>::value) {
        nX = nx, nY = ny, nZ = nz;
        leading_dim = g->dx;
        face = (i+j+k)<0 ? nx+1 : 1;
        start = {1, 1, face};
        end = {nz+2, ny+2, face+1};
    } else if (std::is_same<T, YZX>::value) {
        nX = ny, nY = nz, nZ = nx;
        leading_dim = g->dy;
        face = (i+j+k)<0 ? ny+1 : 1;
        start = {1, face, 1};
        end = {nz+2, face+1, nx+2};
    } else if (std::is_same<T, ZXY>::value) {
        nX = nz, nY = nx, nZ = ny;
        leading_dim = g->dz;
        face = (i+j+k)<0 ? nz+1 : 1;
        start = {face, 1, 1};
        end = {face+1, ny+2, nx+2};
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<3> > node_policy(start,end);
    int size = (1 + 2*(nY+1)*(nZ+1));
    Kokkos::View<float*> d_buf("Device buffer", size);
    Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
    float* p = reinterpret_cast<float*>(end_recv_port(i,j,k,g));
    if(p) {
        float hrw = p[0];
        float hlw = hrw + leading_dim;
        hrw /= hlw;
        hlw = leading_dim/hlw;
        float lw = hlw + hlw;
        float rw = hrw + hrw;
        Kokkos::parallel_for("Copy buffer to host copy", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size), KOKKOS_LAMBDA(const int idx) {
            h_buf(idx) = p[idx];
        });
        Kokkos::deep_copy(d_buf, h_buf);
        Kokkos::parallel_for("end_recv_rho", node_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
            int idx_f, idx_b;
            if(std::is_same<T, XYZ>::value) {
                idx_f = 1 + 2*((z-1)*(ny+1) + y-1);
                idx_b = 1 + 2*((z-1)*(ny+1) + y-1) + 1;
            } else if (std::is_same<T, YZX>::value) {
                idx_f = 1 + 2*((z-1)*(nx+1) + x-1);
                idx_b = 1 + 2*((z-1)*(nx+1) + x-1) + 1;
            } else if (std::is_same<T, ZXY>::value) {
                idx_f = 1 + 2*((y-1)*(nx+1) + x-1);
                idx_b = 1 + 2*((y-1)*(nx+1) + x-1) + 1;
            }
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof) = lw *  k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof) + rw *  d_buf(idx_f);
            k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob) = hlw * k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob) + hrw * d_buf(idx_b);
        });
    }
*/
}
template<> void end_recv_rho<XYZ>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz) {
    int size, face;
    float hlw, hrw, lw, rw;
    float* p = (float *)end_recv_port(i,j,k,g);                        
    k_field_t& k_field = fa->k_f_d;
    if( p ) {                                                   
      hrw  = p[0];               /* Remote g->d##X */       
      hlw  = hrw + g->dx;                                     
      hrw /= hlw;                                               
      hlw  = g->dx/hlw;                                       
      lw   = hlw + hlw;                                         
      rw   = hrw + hrw;                                         
      face = (i+j+k)<0 ? nx+1 : 1;                            
        size = 1 + 2*(ny+1)*(nz+1);
        Kokkos::View<float*> d_buf("Device_buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        for(int idx = 0; idx < size; idx++) {
            h_buf(idx) = p[idx];
        }
        Kokkos::deep_copy(d_buf, h_buf);
        Kokkos::parallel_for("sync_rho: end_recv_rho<XYZ>: x_node_loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
            const int z = team_member.league_rank() + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny+1), [=] (const int yi) {
                const int y = yi + 1;
                const int x = face;
                int idx_f = 1 + 2*((z-1)*(ny+1) + yi);
                int idx_b = 1 + 2*((z-1)*(ny+1) + yi) + 1;
                float rhof = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
                float rhob = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof) = lw*rhof + rw*d_buf(idx_f);
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob) = hlw*rhob + hrw*d_buf(idx_b);
            });
        });
    }                                                           
}
template<> void end_recv_rho<YZX>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz) {
    int size, face;
    float hlw, hrw, lw, rw;
    float* p = (float *)end_recv_port(i,j,k,g);                        
    k_field_t& k_field = fa->k_f_d;
    if( p ) {                                                   
      hrw  = p[0];               /* Remote g->d##X */       
      hlw  = hrw + g->dy;                                     
      hrw /= hlw;                                               
      hlw  = g->dy/hlw;                                       
      lw   = hlw + hlw;                                         
      rw   = hrw + hrw;                                         
      face = (i+j+k)<0 ? ny+1 : 1;                            
        size = 1 + 2*(nx+1)*(nz+1);
        Kokkos::View<float*> d_buf("Device_buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        for(int idx = 0; idx < size; idx++) {
            h_buf(idx) = p[idx];
        }
        Kokkos::deep_copy(d_buf, h_buf);
        Kokkos::parallel_for("sync_rho: end_recv_rho<YZX>: y_node_loop", KOKKOS_TEAM_POLICY_DEVICE(nz+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
            const int z = team_member.league_rank() + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                const int x = xi + 1;
                const int y = face;
                int idx_f = 1 + 2*((z-1)*(nx+1) + xi);
                int idx_b = 1 + 2*((z-1)*(nx+1) + xi) + 1;
                float rhof = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
                float rhob = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof) = lw*rhof + rw*d_buf(idx_f);
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob) = hlw*rhob + hrw*d_buf(idx_b);
            });
        });
    }                                                           
}
template<> void end_recv_rho<ZXY>(const grid_t* g, field_array_t* fa, int i, int j, int k, int nx, int ny, int nz) {
    int size, face;
    float hlw, hrw, lw, rw;
    float* p = (float *)end_recv_port(i,j,k,g);                        
    k_field_t& k_field = fa->k_f_d;
    if( p ) {                                                   
      hrw  = p[0];               /* Remote g->d##X */       
      hlw  = hrw + g->dz;                                     
      hrw /= hlw;                                               
      hlw  = g->dz/hlw;                                       
      lw   = hlw + hlw;                                         
      rw   = hrw + hrw;                                         
      face = (i+j+k)<0 ? nz+1 : 1;                            
        size = 1 + 2*(nx+1)*(ny+1);
        Kokkos::View<float*> d_buf("Device_buffer", size);
        Kokkos::View<float*>::HostMirror h_buf = Kokkos::create_mirror_view(d_buf);
        for(int idx = 0; idx < size; idx++) {
            h_buf(idx) = p[idx];
        }
        Kokkos::deep_copy(d_buf, h_buf);
        Kokkos::parallel_for("sync_rho: end_recv_rho<ZXY>: z_node_loop", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
            const int y = team_member.league_rank() + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
                const int x = xi + 1;
                const int z = face;
                int idx_f = 1 + 2*((y-1)*(nx+1) + xi);
                int idx_b = 1 + 2*((y-1)*(nx+1) + xi) + 1;
                float rhof = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);
                float rhob = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof) = lw*rhof + rw*d_buf(idx_f);
                k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob) = hlw*rhob + hrw*d_buf(idx_b);
            });
        });
    }                                                           
}
template <typename T> void end_send_rho(const grid_t* g, int i, int j, int k) {
    end_send_port(i,j,k,g);
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
/*
  k_local_adjust_rhof( fa, g );
  k_local_adjust_rhob( fa, g );
  Kokkos::deep_copy(fa->k_f_h, fa->k_f_d); 
  Kokkos::deep_copy(fa->k_fe_h, fa->k_fe_d); 
  auto k_field = fa->k_f_h; 
  Kokkos::parallel_for("copy field to host", host_execution_policy(0, g->nv - 1) , KOKKOS_LAMBDA (int i) { 
          fa->f[i].ex = k_field(i, field_var::ex); 
          fa->f[i].ey = k_field(i, field_var::ey); 
          fa->f[i].ez = k_field(i, field_var::ez); 
          fa->f[i].div_e_err = k_field(i, field_var::div_e_err); 
          
          fa->f[i].cbx = k_field(i, field_var::cbx); 
          fa->f[i].cby = k_field(i, field_var::cby); 
          fa->f[i].cbz = k_field(i, field_var::cbz); 
          fa->f[i].div_b_err = k_field(i, field_var::div_b_err); 
          
          fa->f[i].tcax = k_field(i, field_var::tcax); 
          fa->f[i].tcay = k_field(i, field_var::tcay); 
          fa->f[i].tcaz = k_field(i, field_var::tcaz); 
          fa->f[i].rhob = k_field(i, field_var::rhob); 
          
          fa->f[i].jfx = k_field(i, field_var::jfx); 
          fa->f[i].jfy = k_field(i, field_var::jfy); 
          fa->f[i].jfz = k_field(i, field_var::jfz); 
          fa->f[i].rhof = k_field(i, field_var::rhof); 
  });
*/
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
    int nx = g->nx, ny = g->ny, nz = g->nz;

    k_local_adjust_rhof(fa, g);
    k_local_adjust_rhob(fa, g);
/*    
    // Exchange x-faces
    begin_recv_rho<XYZ>(g, -1, 0, 0, nx, ny, nz);
    begin_recv_rho<XYZ>(g,  1, 0, 0, nx, ny, nz);
    begin_send_rho<XYZ>(g, fa, -1, 0, 0, nx, ny, nz);
    begin_send_rho<XYZ>(g, fa,  1, 0, 0, nx, ny, nz);
    end_recv_rho<XYZ>(g, fa,  1, 0, 0, nx, ny, nz);
    end_send_rho<XYZ>(g, -1, 0, 0);
    end_send_rho<XYZ>(g,  1, 0, 0);

    
    // Exchange y-faces
    begin_recv_rho<YZX>(g, 0, -1, 0, nx, ny, nz);
    begin_recv_rho<YZX>(g, 0,  1, 0, nx, ny, nz);
    begin_send_rho<YZX>(g, fa, 0, -1, 0, nx, ny, nz);
    begin_send_rho<YZX>(g, fa, 0,  1, 0, nx, ny, nz);
    end_recv_rho<YZX>(g, fa, 0, -1, 0, nx, ny, nz);
    end_recv_rho<YZX>(g, fa, 0,  1, 0, nx, ny, nz);
    end_send_rho<YZX>(g, 0, -1, 0);
    end_send_rho<YZX>(g, 0,  1, 0);


    // Exchange z-faces
    begin_recv_rho<ZXY>(g, 0, 0, -1, nx, ny, nz);
    begin_recv_rho<ZXY>(g, 0, 0,  1, nx, ny, nz);
    begin_send_rho<ZXY>(g, fa, 0, 0, -1, nx, ny, nz);
    begin_send_rho<ZXY>(g, fa, 0, 0,  1, nx, ny, nz);
    end_recv_rho<ZXY>(g, fa, 0, 0, -1, nx, ny, nz);
    end_recv_rho<ZXY>(g, fa, 0, 0,  1, nx, ny, nz);
    end_send_rho<ZXY>(g, 0, 0, -1);
    end_send_rho<ZXY>(g, 0, 0,  1);
*/
/*
    begin_recv_rho<XYZ>(g, -1, 0, 0, nx, ny, nz);
    begin_recv_rho<XYZ>(g,  1, 0, 0, nx, ny, nz);
    begin_recv_rho<YZX>(g, 0, -1, 0, nx, ny, nz);
    begin_recv_rho<YZX>(g, 0,  1, 0, nx, ny, nz);
    begin_recv_rho<ZXY>(g, 0, 0, -1, nx, ny, nz);
    begin_recv_rho<ZXY>(g, 0, 0,  1, nx, ny, nz);

    begin_send_rho<XYZ>(g, fa, -1, 0, 0, nx, ny, nz);
    begin_send_rho<XYZ>(g, fa,  1, 0, 0, nx, ny, nz);
    begin_send_rho<YZX>(g, fa, 0, -1, 0, nx, ny, nz);
    begin_send_rho<YZX>(g, fa, 0,  1, 0, nx, ny, nz);
    begin_send_rho<ZXY>(g, fa, 0, 0, -1, nx, ny, nz);
    begin_send_rho<ZXY>(g, fa, 0, 0,  1, nx, ny, nz);

    end_recv_rho<XYZ>(g, fa, -1, 0, 0, nx, ny, nz);
    end_recv_rho<XYZ>(g, fa,  1, 0, 0, nx, ny, nz);
    end_recv_rho<YZX>(g, fa, 0, -1, 0, nx, ny, nz);
    end_recv_rho<YZX>(g, fa, 0,  1, 0, nx, ny, nz);
    end_recv_rho<ZXY>(g, fa, 0, 0, -1, nx, ny, nz);
    end_recv_rho<ZXY>(g, fa, 0, 0,  1, nx, ny, nz);

    end_send_rho<XYZ>(g, -1, 0, 0);
    end_send_rho<XYZ>(g,  1, 0, 0);
    end_send_rho<YZX>(g, 0, -1, 0);
    end_send_rho<YZX>(g, 0,  1, 0);
    end_send_rho<ZXY>(g, 0, 0, -1);
    end_send_rho<ZXY>(g, 0, 0,  1);
*/

    // Exchange x-faces
    begin_send_rho<XYZ>(g, fa, -1, 0, 0, nx, ny, nz);
    begin_send_rho<XYZ>(g, fa,  1, 0, 0, nx, ny, nz);
    begin_recv_rho<XYZ>(g, -1, 0, 0, nx, ny, nz);
    begin_recv_rho<XYZ>(g,  1, 0, 0, nx, ny, nz);
    end_recv_rho<XYZ>(g, fa, -1, 0, 0, nx, ny, nz);
    end_recv_rho<XYZ>(g, fa,  1, 0, 0, nx, ny, nz);
    end_send_rho<XYZ>(g, -1, 0, 0);
    end_send_rho<XYZ>(g,  1, 0, 0);

    // Exchange y-faces
    begin_send_rho<YZX>(g, fa, 0, -1, 0, nx, ny, nz);
    begin_send_rho<YZX>(g, fa, 0,  1, 0, nx, ny, nz);
    begin_recv_rho<YZX>(g, 0, -1, 0, nx, ny, nz);
    begin_recv_rho<YZX>(g, 0,  1, 0, nx, ny, nz);
    end_recv_rho<YZX>(g, fa, 0, -1, 0, nx, ny, nz);
    end_recv_rho<YZX>(g, fa, 0,  1, 0, nx, ny, nz);
    end_send_rho<YZX>(g, 0, -1, 0);
    end_send_rho<YZX>(g, 0,  1, 0);

    // Exchange z-faces
    begin_send_rho<ZXY>(g, fa, 0, 0, -1, nx, ny, nz);
    begin_send_rho<ZXY>(g, fa, 0, 0,  1, nx, ny, nz);
    begin_recv_rho<ZXY>(g, 0, 0, -1, nx, ny, nz);
    begin_recv_rho<ZXY>(g, 0, 0,  1, nx, ny, nz);
    end_recv_rho<ZXY>(g, fa, 0, 0, -1, nx, ny, nz);
    end_recv_rho<ZXY>(g, fa, 0, 0,  1, nx, ny, nz);
    end_send_rho<ZXY>(g, 0, 0, -1);
    end_send_rho<ZXY>(g, 0, 0,  1);

}
