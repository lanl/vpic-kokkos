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

template <int i, int j, int k> 
void 
begin_recv_tang_b(field_array_t* fa) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];;
  if constexpr (i!=0 && j==0 && k==0) {
    size = (1 + ny*(nz+1) + nz*(ny+1))*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = (1 + nz*(nx+1) + nx*(nz+1))*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = (1 + nx*(ny+1) + ny*(nx+1))*sizeof(float);
  }
  // Automatically switch between CPU and GPU MPI
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g,rbuf_d,rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_tang_b(field_array_t* fa) {
#define BEGIN_SEND(X,Y,Z) BEGIN_PRIMITIVE {                                    \
    const size_t size = (1+n##Y*(n##Z+1)+n##Z*(n##Y+1))*sizeof(float);         \
    const int face = (i+j+k)<0 ? 1 : n##X;                                     \
    const float d##X = fa->g->d##X;                                            \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});\
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});\
    Kokkos::parallel_for("beg_send_ghost_tang_b<" #X #Y #Z ">", Z##Y##_edge,   \
      KOKKOS_LAMBDA(const int Y, const int Z) {                                \
        const int X = face;                                                    \
        const size_t idx = 1 + (Z-1)*(n##Y+1) + (Y-1);                         \
        if(idx == 1) {                                                         \
          sbuf_d(0) = d##X;                                                    \
        }                                                                      \
        sbuf_d(idx) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::cb##Y);       \
      });                                                                      \
    Kokkos::parallel_for("beg_send_ghost_tang_b<" #X #Y #Z ">", Y##Z##_edge,   \
      KOKKOS_LAMBDA(const int Y, const int Z) {                                \
        const int X = face;                                                    \
        const size_t idx = 1 + n##Z*(n##Y+1) + (Z-1)*n##Y + (Y-1);             \
        sbuf_d(idx) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::cb##Z);       \
      });                                                                      \
    SYNC_MPI_BUFFER(sbuf_h, sbuf_d);                                           \
    BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);                       \
  } END_PRIMITIVE

  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = fa->k_f_d;

  if constexpr (i!=0 && j==0 && k==0) {
    BEGIN_SEND(x,y,z);
  } else if constexpr (i==0 && j!=0 && k==0) {
    BEGIN_SEND(y,z,x);
  } else if constexpr (i==0 && j==0 && k!=0) {
    BEGIN_SEND(z,x,y);
  }
#undef BEGIN_SEND
}

void
begin_remote_ghost_tang_b( field_array_t * RESTRICT fa,
                           const grid_t * g) {
  begin_recv_tang_b<-1, 0, 0>(fa);
  begin_recv_tang_b< 0,-1, 0>(fa);
  begin_recv_tang_b< 0, 0,-1>(fa);
  begin_recv_tang_b<1,0,0>(fa);
  begin_recv_tang_b<0,1,0>(fa);
  begin_recv_tang_b<0,0,1>(fa);

  begin_send_tang_b<-1, 0, 0>(fa);
  begin_send_tang_b< 0,-1, 0>(fa);
  begin_send_tang_b< 0, 0,-1>(fa);
  begin_send_tang_b<1,0,0>(fa);
  begin_send_tang_b<0,1,0>(fa);
  begin_send_tang_b<0,0,1>(fa);
}

template <int i, int j, int k> 
void 
end_recv_tang_b(field_array_t* fa) {
# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                               \
    float lw, rw;                                                              \
    lw = rbuf_h(0); /* Remote g->d##X */                                       \
    rw = (2.*g->d##X)/(lw+g->d##X);                                            \
    lw = (lw-g->d##X)/(lw+g->d##X);                                            \
    const int face = (i+j+k)<0 ? n##X+1 : 0; /* Interpolate */                 \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});\
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});\
    Kokkos::parallel_for("end_recv_cb<" #X #Y #Z "> " #Z #Y "_edge",           \
      Z##Y##_edge,  KOKKOS_LAMBDA(const int Y, const int Z) {                  \
      const int X = face;                                                      \
      const size_t offset = 1 + (Z-1)*(n##Y+1) + (Y-1);                        \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      const size_t ghost = VOXEL(x+i,y+j,z+k,nx,ny,nz);                        \
      kfield(voxel, field_var::cb##Y) = rw*rbuf_d(offset)                      \
                                      + lw*kfield(ghost, field_var::cb##Y);    \
    });                                                                        \
    Kokkos::parallel_for("end_recv_cb<" #X #Y #Z "> " #Y #Z "_edge",           \
      Y##Z##_edge, KOKKOS_LAMBDA(const int Y, const int Z) {                   \
      const int X = face;                                                      \
      const size_t offset = 1 + n##Z*(n##Y+1) + (Z-1)*n##Y + (Y-1);            \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      const size_t ghost = VOXEL(x+i,y+j,z+k,nx,ny,nz);                        \
      kfield(voxel, field_var::cb##Z) = rw*rbuf_d(offset)                      \
                                      + lw*kfield(ghost, field_var::cb##Z);    \
    }); \
  } END_PRIMITIVE

  const grid_t* g = fa->g;
  const float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const k_field_t& kfield = fa->k_f_d; 
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); 
#ifdef VPIC_ENABLE_GPU_AWARE_MPI
    auto rbuf_sv_d = Kokkos::subview(rbuf_d, Kokkos::make_pair(0,1));
    auto rbuf_sv_h = Kokkos::subview(rbuf_h, Kokkos::make_pair(0,1));
    Kokkos::deep_copy(rbuf_sv_h, rbuf_sv_d);
#endif
    if constexpr(i!=0 && j==0 && k==0) {
      END_RECV(i,j,k, x,y,z);
    } else if constexpr(i==0 && j!=0 && k==0) {
      END_RECV(i,j,k, y,z,x);
    } else if constexpr(i==0 && j==0 && k!=0) {
      END_RECV(i,j,k, z,x,y);
    }
  }
#undef END_RECV
}

template<int i, int j, int k> 
void 
end_send_tang_b(field_array_t* RESTRICT field) {
  end_send_port_k(i,j,k, field->g);
}

void
end_remote_ghost_tang_b( field_array_t * RESTRICT field,
                         const grid_t * g) {
  end_recv_tang_b<-1, 0, 0>(field);
  end_recv_tang_b< 0,-1, 0>(field);
  end_recv_tang_b< 0, 0,-1>(field);
  end_recv_tang_b<1, 0, 0>(field);
  end_recv_tang_b<0, 1, 0>(field);
  end_recv_tang_b<0, 0, 1>(field);

  end_send_tang_b<-1, 0, 0>(field);
  end_send_tang_b< 0,-1, 0>(field);
  end_send_tang_b< 0, 0,-1>(field);
  end_send_tang_b<1,0,0>(field);
  end_send_tang_b<0,1,0>(field);
  end_send_tang_b<0,0,1>(field);
}

template <int i, int j, int k> 
void 
begin_recv_ghost_norm_e(field_array_t* fa) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];;
  if constexpr (i!=0 && j==0 && k==0) {
    size = (1 + (ny+1)*(nz+1))*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = (1 + (nz+1)*(nx+1))*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = (1 + (nx+1)*(ny+1))*sizeof(float);
  }
  // Automatically switch between CPU and GPU MPI
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g,rbuf_d,rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_ghost_norm_e(field_array_t* fa) {
#define BEGIN_SEND(X,Y,Z) BEGIN_PRIMITIVE {                                    \
    const size_t size = (1+(n##Y+1)*(n##Z+1))*sizeof(float);                   \
    const int face = (i+j+k)<0 ? 1 : n##X;                                     \
    const float d##X = fa->g->d##X;                                            \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_node({1,1}, {n##Y+2, n##Z+2});  \
    Kokkos::parallel_for("beg_send_ghost_norm_e<" #X #Y #Z ">", X##_node,      \
      KOKKOS_LAMBDA(const int Y, const int Z) {                                \
        const int X = face;                                                    \
        const size_t idx = 1 + (Z-1)*(n##Y+1) + (Y-1);                         \
        if(idx == 1) {                                                         \
          sbuf_d(0) = d##X;                                                    \
        }                                                                      \
        sbuf_d(idx) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::e##X);        \
      });                                                                      \
    SYNC_MPI_BUFFER(sbuf_h, sbuf_d);                                           \
    BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);                       \
  } END_PRIMITIVE

  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = fa->k_f_d;

  if constexpr (i!=0 && j==0 && k==0) {
    BEGIN_SEND(x,y,z);
  } else if constexpr (i==0 && j!=0 && k==0) {
    BEGIN_SEND(y,z,x);
  } else if constexpr (i==0 && j==0 && k!=0) {
    BEGIN_SEND(z,x,y);
  }
#undef BEGIN_SEND
}

void
begin_remote_ghost_norm_e( field_array_t * ALIGNED(128) field,
                           const grid_t * g) {
  begin_recv_ghost_norm_e<-1,  0,  0>(field);
  begin_recv_ghost_norm_e< 0, -1,  0>(field);
  begin_recv_ghost_norm_e< 0,  0, -1>(field);
  begin_recv_ghost_norm_e< 1,  0,  0>(field);
  begin_recv_ghost_norm_e< 0,  1,  0>(field);
  begin_recv_ghost_norm_e< 0,  0,  1>(field);

  begin_send_ghost_norm_e<-1,  0,  0>(field);
  begin_send_ghost_norm_e< 0, -1,  0>(field);
  begin_send_ghost_norm_e< 0,  0, -1>(field);
  begin_send_ghost_norm_e< 1,  0,  0>(field);
  begin_send_ghost_norm_e< 0,  1,  0>(field);
  begin_send_ghost_norm_e< 0,  0,  1>(field);
}

template <int i, int j, int k> 
void 
end_recv_ghost_norm_e(field_array_t* fa) {
# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                               \
    float lw, rw;                                                              \
    lw = rbuf_h(0); /* Remote g->d##X */                                       \
    rw = (2.*g->d##X)/(lw+g->d##X);                                            \
    lw = (lw-g->d##X)/(lw+g->d##X);                                            \
    const int face = (i+j+k)<0 ? n##X+1 : 0; /* Interpolate */                 \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_node({1,1}, {n##Y+2,n##Z+2});   \
    Kokkos::parallel_for("end_recv_ghost_norm_e<" #X #Y #Z "> " #X "_node",    \
      X##_node, KOKKOS_LAMBDA(const int Y, const int Z) {                      \
      const int X = face;                                                      \
      const size_t offset = 1 + (Z-1)*(n##Y+1) + (Y-1);                        \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      const size_t ghost = VOXEL(x+i,y+j,z+k,nx,ny,nz);                        \
      kfield(voxel, field_var::e##X) = rw*rbuf_d(offset)                       \
                                     + lw*kfield(ghost, field_var::e##X);      \
    });                                                                        \
  } END_PRIMITIVE

  const grid_t* g = fa->g;
  const float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const k_field_t& kfield = fa->k_f_d; 
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); 
#ifdef VPIC_ENABLE_GPU_AWARE_MPI
    auto rbuf_sv_d = Kokkos::subview(rbuf_d, Kokkos::make_pair(0,1));
    auto rbuf_sv_h = Kokkos::subview(rbuf_h, Kokkos::make_pair(0,1));
    Kokkos::deep_copy(rbuf_sv_h, rbuf_sv_d);
#endif
    if constexpr(i!=0 && j==0 && k==0) {
      END_RECV(i,j,k, x,y,z);
    } else if constexpr(i==0 && j!=0 && k==0) {
      END_RECV(i,j,k, y,z,x);
    } else if constexpr(i==0 && j==0 && k!=0) {
      END_RECV(i,j,k, z,x,y);
    }
  }
#undef END_RECV
}

template<int i, int j, int k> 
void 
end_send_ghost_norm_e(field_array_t* RESTRICT field) {
  end_send_port_k(i,j,k, field->g);
}

void
end_remote_ghost_norm_e( field_array_t * ALIGNED(128) field,
                         const grid_t * g) {
  end_recv_ghost_norm_e<-1,  0,  0>(field);
  end_recv_ghost_norm_e< 0, -1,  0>(field);
  end_recv_ghost_norm_e< 0,  0, -1>(field);
  end_recv_ghost_norm_e< 1,  0,  0>(field);
  end_recv_ghost_norm_e< 0,  1,  0>(field);
  end_recv_ghost_norm_e< 0,  0,  1>(field);

  end_send_ghost_norm_e<-1,  0,  0>(field);
  end_send_ghost_norm_e< 0, -1,  0>(field);
  end_send_ghost_norm_e< 0,  0, -1>(field);
  end_send_ghost_norm_e< 1,  0,  0>(field);
  end_send_ghost_norm_e< 0,  1,  0>(field);
  end_send_ghost_norm_e< 0,  0,  1>(field);
}

template <int i, int j, int k> 
void 
begin_recv_ghost_div_b(field_array_t* fa) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];;
  if constexpr (i!=0 && j==0 && k==0) {
    size = (1 + ny*nz)*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = (1 + nz*nx)*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = (1 + nx*ny)*sizeof(float);
  }
  // Automatically switch between CPU and GPU MPI
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g,rbuf_d,rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_ghost_div_b(field_array_t* fa) {
#define BEGIN_SEND(X,Y,Z) BEGIN_PRIMITIVE {                                    \
    const size_t size = (1+(n##Y)*(n##Z))*sizeof(float);                       \
    const int face = (i+j+k)<0 ? 1 : n##X;                                     \
    const float d##X = fa->g->d##X;                                            \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_face({1,1}, {n##Y+1, n##Z+1});  \
    Kokkos::parallel_for("beg_send_ghost_div_b<" #X #Y #Z ">", X##_face,       \
      KOKKOS_LAMBDA(const int Y, const int Z) {                                \
        const int X = face;                                                    \
        const size_t idx = 1 + (Z-1)*(n##Y) + (Y-1);                           \
        if(idx == 1) {                                                         \
          sbuf_d(0) = d##X;                                                    \
        }                                                                      \
        sbuf_d(idx) = k_field(VOXEL(x,y,z, nx,ny,nz), field_var::div_b_err);   \
      });                                                                      \
    SYNC_MPI_BUFFER(sbuf_h, sbuf_d);                                           \
    BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);                       \
  } END_PRIMITIVE

  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = fa->k_f_d;

  if constexpr (i!=0 && j==0 && k==0) {
    BEGIN_SEND(x,y,z);
  } else if constexpr (i==0 && j!=0 && k==0) {
    BEGIN_SEND(y,z,x);
  } else if constexpr (i==0 && j==0 && k!=0) {
    BEGIN_SEND(z,x,y);
  }
#undef BEGIN_SEND
}

void 
begin_remote_ghost_div_b(field_array_t* ALIGNED(128) fa, const grid_t* g) {
  begin_recv_ghost_div_b<-1,  0,  0>(fa);
  begin_recv_ghost_div_b< 0, -1,  0>(fa);
  begin_recv_ghost_div_b< 0,  0, -1>(fa);
                                   
  begin_recv_ghost_div_b< 1,  0,  0>(fa);
  begin_recv_ghost_div_b< 0,  1,  0>(fa);
  begin_recv_ghost_div_b< 0,  0,  1>(fa);
                                   
  begin_send_ghost_div_b<-1,  0,  0>(fa);
  begin_send_ghost_div_b< 0, -1,  0>(fa);
  begin_send_ghost_div_b< 0,  0, -1>(fa);
                                   
  begin_send_ghost_div_b< 1,  0,  0>(fa);
  begin_send_ghost_div_b< 0,  1,  0>(fa);
  begin_send_ghost_div_b< 0,  0,  1>(fa);
}

template <int i, int j, int k> 
void 
end_recv_ghost_div_b(field_array_t* fa) {
# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                               \
    float lw, rw;                                                              \
    lw = rbuf_h(0); /* Remote g->d##X */                                       \
    rw = (2.*g->d##X)/(lw+g->d##X);                                            \
    lw = (lw-g->d##X)/(lw+g->d##X);                                            \
    const int face = (i+j+k)<0 ? n##X+1 : 0; /* Interpolate */                 \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_face({1,1}, {n##Y+1,n##Z+1});   \
    Kokkos::parallel_for("end_recv_ghost_div_b<" #X #Y #Z "> " #X "_face",     \
      X##_face, KOKKOS_LAMBDA(const int Y, const int Z) {                      \
      const int X = face;                                                      \
      const size_t offset = 1 + (Z-1)*(n##Y) + (Y-1);                          \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      const size_t ghost = VOXEL(x+i,y+j,z+k,nx,ny,nz);                        \
      kfield(voxel, field_var::div_b_err) = rw*rbuf_d(offset) +                \
                                            lw*kfield(ghost, field_var::div_b_err); \
    }); \
  } END_PRIMITIVE

  const grid_t* g = fa->g;
  const float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const k_field_t& kfield = fa->k_f_d; 
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); 
#ifdef VPIC_ENABLE_GPU_AWARE_MPI
    auto rbuf_sv_d = Kokkos::subview(rbuf_d, Kokkos::make_pair(0,1));
    auto rbuf_sv_h = Kokkos::subview(rbuf_h, Kokkos::make_pair(0,1));
    Kokkos::deep_copy(rbuf_sv_h, rbuf_sv_d);
#endif
    if constexpr(i!=0 && j==0 && k==0) {
      END_RECV(i,j,k, x,y,z);
    } else if constexpr(i==0 && j!=0 && k==0) {
      END_RECV(i,j,k, y,z,x);
    } else if constexpr(i==0 && j==0 && k!=0) {
      END_RECV(i,j,k, z,x,y);
    }
  }
#undef END_RECV
}

template<int i, int j, int k> 
void 
end_send_ghost_div_b(field_array_t* RESTRICT field) {
  end_send_port_k(i,j,k, field->g);
}

void end_remote_ghost_div_b(field_array_t* ALIGNED(128) fa, const grid_t* g) {
  end_recv_ghost_div_b<-1,  0,  0>(fa);
  end_recv_ghost_div_b< 0, -1,  0>(fa);
  end_recv_ghost_div_b< 0,  0, -1>(fa);
  end_recv_ghost_div_b< 1,  0,  0>(fa);
  end_recv_ghost_div_b< 0,  1,  0>(fa);
  end_recv_ghost_div_b< 0,  0,  1>(fa);
                                 
  end_send_ghost_div_b<-1,  0,  0>(fa);
  end_send_ghost_div_b< 0, -1,  0>(fa);
  end_send_ghost_div_b< 0,  0, -1>(fa);
  end_send_ghost_div_b< 1,  0,  0>(fa);
  end_send_ghost_div_b< 0,  1,  0>(fa);
  end_send_ghost_div_b< 0,  0,  1>(fa);
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

template<int i, int j, int k> 
void 
begin_recv_tang_e_norm_b(field_array_t* fa) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  if constexpr(i!=0 && j==0 && k==0) {
    size = (2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz)*sizeof(float);
  } else if constexpr(i==0 && j!=0 && k==0) {
    size = (2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx)*sizeof(float);
  } else if constexpr(i==0 && j==0 && k!=0) {
    size = (2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny)*sizeof(float);
  }
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);
}

template<int i, int j, int k> 
void 
begin_send_tang_e_norm_b(field_array_t* fa) {
# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                             \
    const int size = ( 2*n##Y*(n##Z+1) + 2*n##Z*(n##Y+1) +                     \
                       n##Y*n##Z )*sizeof(float);                              \
    const int face = (i+j+k)<0 ? 1 : n##X+1;                                   \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_face({1,1}, {n##Y+1,n##Z+1});   \
    Kokkos::parallel_for("beg_send_tang_e_norm_b<" #X #Y #Z "> " #X "_face",   \
      X##_face, KOKKOS_LAMBDA(const int Y, const int Z) {                      \
      const int X = face;                                                      \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      sbuf_d((Z-1)*n##Y + (Y-1)) = kfield(voxel, field_var::cb##X);            \
    });                                                                        \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});\
    Kokkos::parallel_for("beg_send_tang_e_norm_b<" #X #Y #Z "> " #Y #Z "_edge",\
      Y##Z##_edge, KOKKOS_LAMBDA(const int Y, const int Z) {                   \
      const int X = face;                                                      \
      const size_t offset = n##Z*n##Y + 2*((Z-1)*n##Y + (Y-1));                \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      sbuf_d(offset) = kfield(voxel, field_var::e##Y);                         \
      sbuf_d(offset + 1) = kfield(voxel, field_var::tca##Y);                   \
    });                                                                        \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});\
    Kokkos::parallel_for("beg_send_tang_e_norm_b<" #X #Y #Z "> " #Z #Y "_edge",\
      Z##Y##_edge, KOKKOS_LAMBDA(const int Y, const int Z) {                   \
      const int X = face;                                                      \
      const size_t offset = n##Z*n##Y + 2*n##Y*(n##Z+1)                        \
                          + 2*((Z-1)*(n##Y+1) + (Y-1));                        \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      sbuf_d(offset) = kfield(voxel, field_var::e##Z);                         \
      sbuf_d(offset + 1) = kfield(voxel, field_var::tca##Z);                   \
    });                                                                        \
    SYNC_MPI_BUFFER(sbuf_h, sbuf_d);                                           \
    BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);                       \
  } END_PRIMITIVE

  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  const k_field_t& kfield = fa->k_f_d; 
  if constexpr(i!=0 && j==0 && k==0) {
    BEGIN_SEND(i,j,k, x,y,z);
  } else if constexpr(i==0 && j!=0 && k==0) {
    BEGIN_SEND(i,j,k, y,z,x);
  } else if constexpr(i==0 && j==0 && k!=0) {
    BEGIN_SEND(i,j,k, z,x,y);
  }
#undef BEGIN_SEND
}

template<int i, int j, int k> 
double 
end_recv_tang_e_norm_b(field_array_t* fa) {
# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                               \
    const int face = (i+j+k)<0 ? n##X+1 : 1; /* Average */                     \
    double X##_face_err = 0.0, Y##Z##_edge_err=0.0, Z##Y##_edge_err=0.0;       \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_face({1,1}, {n##Y+1,n##Z+1});   \
    Kokkos::parallel_reduce("end_recv_tang_e_norm_b<" #X #Y #Z "> " #X "_face",\
      X##_face, KOKKOS_LAMBDA(const int Y, const int Z, double& error) {       \
      const int X = face;                                                      \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      const double w1 = rbuf_d((Z-1)*n##Y + (Y-1));                            \
      const double w2 = kfield(voxel, field_var::cb##X);                       \
      kfield(voxel, field_var::cb##X) = 0.5*( w1+w2 );                         \
      error += (w1-w2)*(w1-w2);                                                \
    }, X##_face_err);                                                          \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});\
    Kokkos::parallel_reduce("end_recv_tang_e_norm_b<" #X #Y #Z "> " #Y #Z "_edge", \
      Y##Z##_edge, KOKKOS_LAMBDA(const int Y, const int Z, double& error) {    \
      const int X = face;                                                      \
      const size_t offset = n##Z*n##Y + 2*((Z-1)*n##Y + (Y-1));                \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      double w1 = rbuf_d(offset);                                              \
      double w2 = kfield(voxel, field_var::e##Y);                              \
      kfield(voxel, field_var::e##Y) = 0.5*( w1+w2 );                          \
      error += (w1-w2)*(w1-w2);                                                \
      w1 = rbuf_d(offset+1);                                                   \
      w2 = kfield(voxel, field_var::tca##Y);                                   \
      kfield(voxel, field_var::tca##Y) = 0.5*( w1+w2 );                        \
    }, Y##Z##_edge_err);                                                       \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});\
    Kokkos::parallel_reduce("end_recv_tang_e_norm_b<" #X #Y #Z "> " #Z #Y "_edge", \
      Z##Y##_edge, KOKKOS_LAMBDA(const int Y, const int Z, double& error) {    \
      const int X = face;                                                      \
      const size_t offset = n##Z*n##Y + 2*n##Y*(n##Z+1)                        \
                          + 2*((Z-1)*(n##Y+1) + (Y-1));                        \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      double w1 = rbuf_d(offset);                                              \
      double w2 = kfield(voxel, field_var::e##Z);                              \
      kfield(voxel, field_var::e##Z) = 0.5*( w1+w2 );                          \
      error += (w1-w2)*(w1-w2);                                                \
      w1 = rbuf_d(offset+1);                                                   \
      w2 = kfield(voxel, field_var::tca##Z);                                   \
      kfield(voxel, field_var::tca##Z) = 0.5*( w1+w2 );                        \
    }, Z##Y##_edge_err);                                                       \
    err += X##_face_err + Y##Z##_edge_err + Z##Y##_edge_err;                   \
  } END_PRIMITIVE

  double err=0.0;
  const grid_t* g = fa->g;
  const float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const k_field_t& kfield = fa->k_f_d; 
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); 
    if constexpr(i!=0 && j==0 && k==0) {
      END_RECV(i,j,k, x,y,z);
    } else if constexpr(i==0 && j!=0 && k==0) {
      END_RECV(i,j,k, y,z,x);
    } else if constexpr(i==0 && j==0 && k!=0) {
      END_RECV(i,j,k, z,x,y);
    }
  }
#undef END_RECV
  return err;
}

template<int i, int j, int k> 
void 
end_send_tang_e_norm_b(field_array_t* fa) {
  end_send_port_k(i,j,k,fa->g);
}

double
synchronize_tang_e_norm_b( field_array_t * RESTRICT fa ) {
  const grid_t * RESTRICT g = fa->g;
  double err = 0, gerr;

  if( !fa ) ERROR(( "Bad args" ));

  local_adjust_tang_e( fa, g );
  local_adjust_norm_b( fa, g );

  // Exchange x-faces
  begin_recv_tang_e_norm_b<-1, 0, 0>( fa );
  begin_recv_tang_e_norm_b< 1, 0, 0>( fa );
  begin_send_tang_e_norm_b<-1, 0, 0>( fa );
  begin_send_tang_e_norm_b< 1, 0, 0>( fa );
  err += end_recv_tang_e_norm_b<-1, 0, 0>( fa );
  err += end_recv_tang_e_norm_b< 1, 0, 0>( fa );
  end_send_tang_e_norm_b<-1, 0, 0>( fa  );
  end_send_tang_e_norm_b< 1, 0, 0>( fa  );

  // Exchange y-faces
  begin_recv_tang_e_norm_b<0,-1, 0>( fa );
  begin_recv_tang_e_norm_b<0, 1, 0>( fa );
  begin_send_tang_e_norm_b<0,-1, 0>( fa );
  begin_send_tang_e_norm_b<0, 1, 0>( fa );
  err += end_recv_tang_e_norm_b<0,-1, 0>( fa );
  err += end_recv_tang_e_norm_b<0, 1, 0>( fa );
  end_send_tang_e_norm_b<0,-1, 0>( fa );
  end_send_tang_e_norm_b<0, 1, 0>( fa );

  // Exchange z-faces
  begin_recv_tang_e_norm_b<0, 0,-1>( fa );
  begin_recv_tang_e_norm_b<0, 0, 1>( fa );
  begin_send_tang_e_norm_b<0, 0,-1>( fa );
  begin_send_tang_e_norm_b<0, 0, 1>( fa );
  err += end_recv_tang_e_norm_b<0, 0,-1>( fa );
  err += end_recv_tang_e_norm_b<0, 0, 1>( fa );
  end_send_tang_e_norm_b<0, 0,-1>( fa );
  end_send_tang_e_norm_b<0, 0, 1>( fa );

  mp_allsum_d( &err, &gerr, 1 );
  return gerr;
}

template <int i, int j, int k> 
void 
begin_recv_jf(field_array_t *fa) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  if constexpr(i!=0 && j==0 && k==0) {
    size = (ny*(nz+1) + nz*(ny+1) + 1)*sizeof(float);
  } else if constexpr(i==0 && j!=0 && k==0) {
    size = (nz*(nx+1) + nx*(nz+1) + 1)*sizeof(float);
  } else if constexpr(i==0 && j==0 && k!=0) {
    size = (nx*(ny+1) + ny*(nx+1) + 1)*sizeof(float);
  }
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_jf(field_array_t* fa) {
# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                             \
    const int size = ( n##Y*(n##Z+1) +                                         \
                       n##Z*(n##Y+1) + 1 )*sizeof(float);                      \
    const int face = (i+j+k)<0 ? 1 : n##X+1;                                   \
    const float d##X = fa->g->d##X;                                            \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});\
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});\
    Kokkos::parallel_for("beg_send_jf<" #X #Y #Z "> " #Y #Z "_edge",           \
      Y##Z##_edge, KOKKOS_LAMBDA(const int Y, const int Z) {                   \
      const int X = face;                                                      \
      const size_t offset = 1 + (Z-1)*n##Y + (Y-1);                            \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      sbuf_d(offset) = kfield(voxel, field_var::jf##Y);                        \
      if(offset == 1)                                                          \
        sbuf_d(0) = d##X;                                                      \
    });                                                                        \
    Kokkos::parallel_for("beg_send_jf<" #X #Y #Z "> " #Z #Y "_edge",           \
      Z##Y##_edge, KOKKOS_LAMBDA(const int Y, const int Z) {                   \
      const int X = face;                                                      \
      const size_t offset = 1 + n##Y*(n##Z+1) + (Z-1)*(n##Y+1) + (Y-1);        \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      sbuf_d(offset) = kfield(voxel, field_var::jf##Z);                        \
    });                                                                        \
    SYNC_MPI_BUFFER(sbuf_h, sbuf_d);                                           \
    BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);                       \
  } END_PRIMITIVE

  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  const k_field_t& kfield = fa->k_f_d; 
  if constexpr(i!=0 && j==0 && k==0) {
    BEGIN_SEND(i,j,k, x,y,z);
  } else if constexpr(i==0 && j!=0 && k==0) {
    BEGIN_SEND(i,j,k, y,z,x);
  } else if constexpr(i==0 && j==0 && k!=0) {
    BEGIN_SEND(i,j,k, z,x,y);
  }
#undef BEGIN_SEND
}

template <int i, int j, int k> 
void 
end_recv_jf(field_array_t* fa) {
# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                               \
    const int face = (i+j+k)<0 ? n##X+1 : 1; /* Twice weighted sum */          \
    float rw = rbuf_h(0);                                                      \
    float lw  = rw + g->d##X;                                                  \
    rw /= lw;                                                                  \
    lw  = g->d##X/lw;                                                          \
    lw += lw;                                                                  \
    rw += rw;                                                                  \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});\
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});\
    Kokkos::parallel_for("end_recv_jf<" #X #Y #Z "> " #Y #Z "_edge",           \
      Y##Z##_edge, KOKKOS_LAMBDA(const int Y, const int Z) {                   \
      const int X = face;                                                      \
      const size_t offset = 1 + (Z-1)*n##Y + (Y-1);                            \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      const float jf##Y = kfield(voxel, field_var::jf##Y);                     \
      kfield(voxel, field_var::jf##Y) = lw*jf##Y + rw*rbuf_d(offset);          \
    });                                                                        \
    Kokkos::parallel_for("end_recv_jf<" #X #Y #Z "> " #Z #Y "_edge",           \
      Z##Y##_edge, KOKKOS_LAMBDA(const int Y, const int Z) {                   \
      const int X = face;                                                      \
      const size_t offset = 1 + n##Y*(n##Z+1) + (Z-1)*(n##Y+1) + (Y-1);        \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      const float jf##Z = kfield(voxel, field_var::jf##Z);                     \
      kfield(voxel, field_var::jf##Z) = lw*jf##Z + rw*rbuf_d(offset);          \
    });                                                                        \
  } END_PRIMITIVE

  const grid_t* g = fa->g;
  const float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const k_field_t& kfield = fa->k_f_d; 
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); 
#ifdef VPIC_ENABLE_GPU_AWARE_MPI
    auto rbuf_sv_d = Kokkos::subview(rbuf_d, Kokkos::make_pair(0,1));
    auto rbuf_sv_h = Kokkos::subview(rbuf_h, Kokkos::make_pair(0,1));
    Kokkos::deep_copy(rbuf_sv_h, rbuf_sv_d);
#endif
    if constexpr(i!=0 && j==0 && k==0) {
      END_RECV(i,j,k, x,y,z);
    } else if constexpr(i==0 && j!=0 && k==0) {
      END_RECV(i,j,k, y,z,x);
    } else if constexpr(i==0 && j==0 && k!=0) {
      END_RECV(i,j,k, z,x,y);
    }
  }
#undef END_RECV
}

template <int i, int j, int k> 
void 
end_send_jf(field_array_t* fa) {
  end_send_port_k(i,j,k,fa->g);
}

void synchronize_jf(field_array_t* RESTRICT fa) {
  if(!fa) ERROR(( "Bad args" ));
  grid_t* RESTRICT g = fa->g;

  local_adjust_jf(fa, g);

  // Exchange x-faces
  begin_recv_jf<-1, 0, 0>( fa );
  begin_recv_jf< 1, 0, 0>( fa );
  begin_send_jf<-1, 0, 0>( fa );
  begin_send_jf< 1, 0, 0>( fa );
  end_recv_jf<-1, 0, 0>( fa );
  end_recv_jf< 1, 0, 0>( fa );
  end_send_jf<-1, 0, 0>( fa );
  end_send_jf< 1, 0, 0>( fa );

  // Exchange y-faces
  begin_recv_jf<0,-1, 0>( fa );
  begin_recv_jf<0, 1, 0>( fa );
  begin_send_jf<0,-1, 0>( fa );
  begin_send_jf<0, 1, 0>( fa );
  end_recv_jf<0,-1, 0>( fa );
  end_recv_jf<0, 1, 0>( fa );
  end_send_jf<0,-1, 0>( fa );
  end_send_jf<0, 1, 0>( fa );

  // Exchange z-faces
  begin_recv_jf<0, 0,-1>( fa );
  begin_recv_jf<0, 0, 1>( fa );
  begin_send_jf<0, 0,-1>( fa );
  begin_send_jf<0, 0, 1>( fa );
  end_recv_jf<0, 0,-1>( fa );
  end_recv_jf<0, 0, 1>( fa );
  end_send_jf<0, 0,-1>( fa );
  end_send_jf<0, 0, 1>( fa );
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

template <int i, int j, int k> 
void 
begin_recv_rho(field_array* fa) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  if constexpr(i!=0 && j==0 && k==0) {
    size = ( 1 + 2*(ny+1)*(nz+1) )*sizeof(float);
  } else if constexpr(i==0 && j!=0 && k==0) {
    size = ( 1 + 2*(nz+1)*(nx+1) )*sizeof(float);
  } else if constexpr(i==0 && j==0 && k!=0) {
    size = ( 1 + 2*(nx+1)*(ny+1) )*sizeof(float);
  }
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);
}

template<int i, int j, int k> 
void 
begin_send_rho(field_array_t* fa) {
# define BEGIN_SEND(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                             \
    const int size = ( 1 + 2*(n##Y+1)*(n##Z+1) )*sizeof(float);                \
    const int face = (i+j+k)<0 ? 1 : n##X+1;                                   \
    const float d##X = fa->g->d##X;                                            \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_node({1,1}, {n##Y+2,n##Z+2});   \
    Kokkos::parallel_for("beg_send_rho<" #X #Y #Z "> " #X "_node", X##_node,   \
      KOKKOS_LAMBDA(const int Y, const int Z) {                                \
      const int X = face;                                                      \
      const size_t offset = 1 + 2*((Z-1)*(n##Y+1) + (Y-1));                    \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      sbuf_d(offset) = kfield(voxel, field_var::rhof);                         \
      sbuf_d(offset+1) = kfield(voxel, field_var::rhob);                       \
      if(offset == 1)                                                          \
        sbuf_d(0) = d##X;                                                      \
    });                                                                        \
    SYNC_MPI_BUFFER(sbuf_h, sbuf_d);                                           \
    BEGIN_SEND_PORT_K(i,j,k,size,fa->g, sbuf_d, sbuf_h);                       \
  } END_PRIMITIVE

  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  Kokkos::DualView<float*> sbuf = fa->fb->send_buffer[BOUNDARY(i,j,k)];
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  const k_field_t& kfield = fa->k_f_d; 
  if constexpr(i!=0 && j==0 && k==0) {
    BEGIN_SEND(i,j,k, x,y,z);
  } else if constexpr(i==0 && j!=0 && k==0) {
    BEGIN_SEND(i,j,k, y,z,x);
  } else if constexpr(i==0 && j==0 && k!=0) {
    BEGIN_SEND(i,j,k, z,x,y);
  }
#undef BEGIN_SEND
}

template <int i, int j, int k> 
void 
end_recv_rho(field_array_t* fa) {
# define END_RECV(i,j,k,X,Y,Z) BEGIN_PRIMITIVE {                               \
    const int face = (i+j+k)<0 ? n##X+1 : 1; /* Twice weighted sum */          \
    float hrw = rbuf_h(0); /* Remote g->d##X */                                \
    float hlw  = hrw + g->d##X;                                                \
    hrw /= hlw;                                                                \
    hlw  = g->d##X/hlw;                                                        \
    const float lw = hlw + hlw;                                                \
    const float rw = hrw + hrw;                                                \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_node({1,1}, {n##Y+2,n##Z+2});   \
    Kokkos::parallel_for("end_recv_rho<" #X #Y #Z "> " #X "_node", X##_node,   \
      KOKKOS_LAMBDA(const int Y, const int Z) {                                \
      const int X = face;                                                      \
      const size_t offset = 1 + 2*((Z-1)*(n##Y+1) + (Y-1));                    \
      const size_t voxel = VOXEL(x,y,z,nx,ny,nz);                              \
      const float rhof = kfield(voxel, field_var::rhof);                       \
      const float rhob = kfield(voxel, field_var::rhob);                       \
      kfield(voxel, field_var::rhof) =  lw*rhof +  rw*rbuf_d(offset);          \
      kfield(voxel, field_var::rhob) = hlw*rhob + hrw*rbuf_d(offset+1);        \
    });                                                                        \
  } END_PRIMITIVE

  const grid_t* g = fa->g;
  const float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    const k_field_t& kfield = fa->k_f_d; 
    Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    SYNC_MPI_BUFFER(rbuf_d, rbuf_h); 
#ifdef VPIC_ENABLE_GPU_AWARE_MPI
    auto rbuf_sv_d = Kokkos::subview(rbuf_d, Kokkos::make_pair(0,1));
    auto rbuf_sv_h = Kokkos::subview(rbuf_h, Kokkos::make_pair(0,1));
    Kokkos::deep_copy(rbuf_sv_h, rbuf_sv_d);
#endif
    if constexpr(i!=0 && j==0 && k==0) {
      END_RECV(i,j,k, x,y,z);
    } else if constexpr(i==0 && j!=0 && k==0) {
      END_RECV(i,j,k, y,z,x);
    } else if constexpr(i==0 && j==0 && k!=0) {
      END_RECV(i,j,k, z,x,y);
    }
  }
#undef END_RECV
}

template <int i, int j, int k> 
void 
end_send_rho(field_array* fa) {
  end_send_port_k(i,j,k,fa->g);
}

void synchronize_rho(field_array_t* RESTRICT fa) {
  if(!fa) ERROR(( "Bad args" ));
  grid_t* RESTRICT g = fa->g;

  local_adjust_rhof(fa, g);
  local_adjust_rhob(fa, g);

  // Exchange x-faces
  begin_recv_rho<-1, 0, 0>( fa );
  begin_recv_rho< 1, 0, 0>( fa );
  begin_send_rho<-1, 0, 0>( fa );
  begin_send_rho< 1, 0, 0>( fa );
  end_recv_rho<-1, 0, 0>( fa );
  end_recv_rho< 1, 0, 0>( fa );
  end_send_rho<-1, 0, 0>( fa );
  end_send_rho< 1, 0, 0>( fa );

  // Exchange y-faces
  begin_recv_rho<0,-1, 0>( fa );
  begin_recv_rho<0, 1, 0>( fa );
  begin_send_rho<0,-1, 0>( fa );
  begin_send_rho<0, 1, 0>( fa );
  end_recv_rho<0,-1, 0>(  fa );
  end_recv_rho<0, 1, 0>(  fa );
  end_send_rho<0,-1, 0>(  fa );
  end_send_rho<0, 1, 0>(  fa );

  // Exchange z-faces
  begin_recv_rho<0, 0,-1>( fa );
  begin_recv_rho<0, 0, 1>( fa );
  begin_send_rho<0, 0,-1>( fa );
  begin_send_rho<0, 0, 1>( fa );
  end_recv_rho<0, 0,-1>( fa );
  end_recv_rho<0, 0, 1>( fa );
  end_send_rho<0, 0,-1>( fa );
  end_send_rho<0, 0, 1>( fa );
}

