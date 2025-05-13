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
#include <Kokkos_DualView.hpp>
#include "mpi.h"

// GPU aware MPI macros
#define VPIC_ENABLE_GPU_AWARE_MPI
#ifdef VPIC_ENABLE_GPU_AWARE_MPI

#define BEGIN_SEND_PORT_K(i, j, k, size, g, sendbuf_d, sendbuf_h); \
  begin_send_port_k(i,j,k,size,g, reinterpret_cast<char*>(sendbuf_d.data())); 
#define BEGIN_RECV_PORT_K(i, j, k, size, g, recvbuf_d, recvbuf_h); \
  begin_recv_port_k(i,j,k,size,g, reinterpret_cast<char*>(recvbuf_d.data()));

#else

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

typedef class XYZ {} XYZ;
typedef class YZX {} YZX;
typedef class ZXY {} ZXY;
using EdgeMDRangePolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>; 
using NodeMDRangePolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>; 
using FaceMDRangePolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>; 

/*****************************************************************************
 * Halo exchange functions
 *
 * Note: These functions are split into begin / end pairs to facillitate
 * overlapped communications. These functions try to interpolate the ghost
 * values when neighboring domains have a different cell size in the normal
 * direction. The halo exchange functions are generic and support communication 
 * of a range of field variables with neighbors. The variable range must be 
 * contiguous for now (i.e., ex,ey,ez). GPU aware MPI support is enabled by the 
 * VPIC_ENABLE_GPU_AWARE_MPI compile option. Unnecessary communication 
 * (i.e., processes without neighbors, send to self) is automatically removed 
 * or replaced by an in memory copy kernel.
 *
 * Note: Input arguments are not tested for validity as these functions are
 * mean to be called from other field module functions (which presumably do
 * check input arguments).
 *****************************************************************************/

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

template <int i, int j, int k> 
void 
begin_recv_tang_b_kokkos(field_array_t* fa) {
  const grid_t* g = fa->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = get_recv_dualview(fa, i,j,k);
  if constexpr (i!=0 && j==0 && k==0) {
    size = (1 + ny*(nz+1) + nz*(ny+1))*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = (1 + nz*(nx+1) + nx*(nz+1))*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = (1 + nx*(ny+1) + ny*(nx+1))*sizeof(float);
  }
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  BEGIN_RECV_PORT_K(i,j,k,size,g,rbuf_d,rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_tang_b_kokkos(field_array_t* fa) {
  const grid_t* g = fa->g;
  Kokkos::DualView<float*> sbuf = get_send_dualview(fa, i,j,k);
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = fa->k_f_d;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  size_t size;

# define BEGIN_SEND_TANG_B(X,Y,Z) BEGIN_PRIMITIVE {                             \
    size = (1+n##Y*(n##Z+1)+n##Z*(n##Y+1))*sizeof(float);                    \
    const int face = (i+j+k) < 0 ? 1 : n##X;                                    \
    const int X = face;                                                       \
    const float d##X = g->d##X;                                                \
    EdgeMDRangePolicy Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});    \
    EdgeMDRangePolicy Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});    \
    Kokkos::parallel_for("begin_send_tang_b<" #X #Y #Z ">", Z##Y##_edge,          \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = 1 + (Z-1)*(n##Y+1) + (Y-1);                             \
      if(idx == 1)                                                               \
        sbuf_d(0) = d##X;                                                       \
      sbuf_d(idx) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cb##Y);           \
    });                                                                          \
    Kokkos::parallel_for("begin_send_tang_b<" #X #Y #Z ">", Y##Z##_edge,          \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = 1 + n##Z*(n##Y+1) + (Z-1)*(n##Y) + (Y-1);           \
      sbuf_d(idx) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cb##Z);           \
    });                                                                          \
  } END_PRIMITIVE

  if constexpr (i!=0 && j==0 && k==0) {
    BEGIN_SEND_TANG_B(x,y,z);
  } else if constexpr (i==0 && j!=0 && k==0) {
    BEGIN_SEND_TANG_B(y,z,x);
  } else if constexpr (i==0 && j==0 && k!=0) {
    BEGIN_SEND_TANG_B(z,x,y);
  }
#undef BEGIN_SEND_TANG_B

  sbuf.modify<Kokkos::DefaultExecutionSpace>();
  sync_comm_buffer<Kokkos::DefaultHostExecutionSpace>(sbuf);
  Kokkos::fence();
  BEGIN_SEND_PORT_K(i,j,k,size,g,sbuf_d,sbuf_h);
}

void
kokkos_begin_remote_ghost_tang_b( field_array_t      * RESTRICT fa,
                                  const grid_t *              g,
                                  field_buffers_t&            f_buffers) {
  begin_recv_tang_b_kokkos<-1,0,0>(fa);
  begin_recv_tang_b_kokkos<0,-1,0>(fa);
  begin_recv_tang_b_kokkos<0,0,-1>(fa);
  begin_recv_tang_b_kokkos<1,0,0>(fa);
  begin_recv_tang_b_kokkos<0,1,0>(fa);
  begin_recv_tang_b_kokkos<0,0,1>(fa);

  begin_send_tang_b_kokkos<-1,0,0>(fa);
  begin_send_tang_b_kokkos<0,-1,0>(fa);
  begin_send_tang_b_kokkos<0,0,-1>(fa);
  begin_send_tang_b_kokkos<1,0,0>(fa);
  begin_send_tang_b_kokkos<0,1,0>(fa);
  begin_send_tang_b_kokkos<0,0,1>(fa);

}

template<int i, int j, int k> 
void 
end_recv_tang_b_kokkos(field_array_t* RESTRICT field) {
  const grid_t* g = field->g;
  Kokkos::DualView<float*> rbuf = get_recv_dualview(field, i,j,k);
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = field->k_f_d;
  const int nx = g->nx, ny = g->ny, nz = g->nz;

# define END_RECV_TANG_B(X,Y,Z) BEGIN_PRIMITIVE {                              \
    const int face = (i+j+k) < 0 ? n##X+1 : 0;                                 \
    const int X = face;                                                        \
    const float d##X = g->d##X;                                                \
    EdgeMDRangePolicy Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});                     \
    EdgeMDRangePolicy Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});                     \
    Kokkos::parallel_for("end_recv_tang_b<" #X #Y #Z ">", Z##Y##_edge,         \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      float lw = rbuf_d(0);                                                    \
      const float rw = (2.*d##X) / (lw + d##X);                                \
      lw = (lw - d##X) / (lw + d##X);                                          \
      const int idx = 1 + (Z-1)*(n##Y+1) + (Y-1);                              \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cb##Y) = rw*rbuf_d(idx)        \
        + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cb##Y);           \
    });                                                                        \
    Kokkos::parallel_for("end_recv_tang_b<" #X #Y #Z ">", Y##Z##_edge,         \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      float lw = rbuf_d(0);                                                    \
      const float rw = (2.*d##X) / (lw + d##X);                                \
      lw = (lw - d##X) / (lw + d##X);                                          \
      const int idx = 1 + n##Z*(n##Y+1) + (Z-1)*(n##Y) + (Y-1);                \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cb##Z) = rw*rbuf_d(idx)        \
        + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::cb##Z);           \
    });                                                                        \
  } END_PRIMITIVE

  float* p = static_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    rbuf.modify_host();
    sync_comm_buffer<Kokkos::DefaultExecutionSpace>(rbuf);
    Kokkos::fence();
    if constexpr (i!=0 && j==0 && k==0) {
      END_RECV_TANG_B(x,y,z);
    } else if constexpr (i==0 && j!=0 && k==0) {
      END_RECV_TANG_B(y,z,x);
    } else if constexpr (i==0 && j==0 && k!=0) {
      END_RECV_TANG_B(z,x,y);
    }
  }
#undef END_RECV_TANG_B
}

// Completely unnecessary, only for symmetry of function calls
template<int i, int j, int k> 
void 
end_send_tang_b_kokkos(field_array_t* RESTRICT field) {
  end_send_port_k(i,j,k, field->g);
}

void
kokkos_end_remote_ghost_tang_b( field_array_t      * RESTRICT field,
                         const grid_t *              g ,
                            field_buffers_t&        f_buffers) {
  end_recv_tang_b_kokkos<-1, 0, 0>(field);
  end_recv_tang_b_kokkos<0, -1, 0>(field);
  end_recv_tang_b_kokkos<0, 0, -1>(field);
  end_recv_tang_b_kokkos<1, 0, 0>(field);
  end_recv_tang_b_kokkos<0, 1, 0>(field);
  end_recv_tang_b_kokkos<0, 0, 1>(field);

  end_send_tang_b_kokkos<-1,0,0>(field);
  end_send_tang_b_kokkos<0,-1,0>(field);
  end_send_tang_b_kokkos<0,0,-1>(field);
  end_send_tang_b_kokkos<1,0,0>(field);
  end_send_tang_b_kokkos<0,1,0>(field);
  end_send_tang_b_kokkos<0,0,1>(field);
}

template <int i, int j, int k> 
void 
begin_recv_ghost_norm_e_kokkos(field_array_t* fa) {
  const grid_t* g = fa->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = get_recv_dualview(fa, i, j, k);;
  if constexpr (i!=0 && j==0 && k==0) {
    size = (1 + (ny+1)*(nz+1))*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = (1 + (nz+1)*(nx+1))*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = (1 + (nx+1)*(ny+1))*sizeof(float);
  }
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  BEGIN_RECV_PORT_K(i,j,k,size,g,rbuf_d,rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_ghost_norm_e_kokkos(field_array_t* fa) {
  const grid_t* g = fa->g;
  Kokkos::DualView<float*> sbuf = get_send_dualview(fa, i,j,k);
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = fa->k_f_d;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  size_t size;

# define BEGIN_SEND_NORM_E(X,Y,Z) BEGIN_PRIMITIVE {                            \
    size = (1+(n##Y+1)*(n##Z+1))*sizeof(float);                                \
    const int face = (i+j+k) < 0 ? 1 : n##X;                                   \
    const int X = face;                                                        \
    const float d##X = g->d##X;                                                \
    NodeMDRangePolicy X##_node({1,1}, {n##Y+2,n##Z+2});                        \
    Kokkos::parallel_for("begin_send_norm_e<" #X #Y #Z ">", X##_node,          \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = 1 + (Z-1)*(n##Y+1) + (Y-1);                              \
      if(idx == 1)                                                             \
        sbuf_d(0) = d##X;                                                      \
      sbuf_d(idx) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::e##Y);           \
    });                                                                        \
  } END_PRIMITIVE
  
  if constexpr (i!=0 && j==0 && k==0) {
    BEGIN_SEND_NORM_E(x,y,z);
  } else if constexpr (i==0 && j!=0 && k==0) {
    BEGIN_SEND_NORM_E(y,z,x);
  } else if constexpr (i==0 && j==0 && k!=0) {
    BEGIN_SEND_NORM_E(z,x,y);
  }
#undef BEGIN_SEND_NORM_E

  Kokkos::fence();
  sbuf.modify<Kokkos::DefaultExecutionSpace>();
  sync_comm_buffer<Kokkos::DefaultHostExecutionSpace>(sbuf);
  BEGIN_SEND_PORT_K(i,j,k,size,g,sbuf_d,sbuf_h);
}

void
kokkos_begin_remote_ghost_norm_e( field_array_t      * ALIGNED(128) field,
                                  const grid_t *              g,
                                  field_buffers_t&            f_buffers) {
  begin_recv_ghost_norm_e_kokkos<-1,  0,  0>(field);
  begin_recv_ghost_norm_e_kokkos< 0, -1,  0>(field);
  begin_recv_ghost_norm_e_kokkos< 0,  0, -1>(field);
  begin_recv_ghost_norm_e_kokkos< 1,  0,  0>(field);
  begin_recv_ghost_norm_e_kokkos< 0,  1,  0>(field);
  begin_recv_ghost_norm_e_kokkos< 0,  0,  1>(field);

  begin_send_ghost_norm_e_kokkos<-1,  0,  0>(field);
  begin_send_ghost_norm_e_kokkos< 0, -1,  0>(field);
  begin_send_ghost_norm_e_kokkos< 0,  0, -1>(field);
  begin_send_ghost_norm_e_kokkos< 1,  0,  0>(field);
  begin_send_ghost_norm_e_kokkos< 0,  1,  0>(field);
  begin_send_ghost_norm_e_kokkos< 0,  0,  1>(field);
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

template<int i, int j, int k> 
void 
end_recv_ghost_norm_e_kokkos(field_array_t* RESTRICT field) {
  const grid_t* g = field->g;
  Kokkos::DualView<float*> rbuf = get_recv_dualview(field, i,j,k);
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = field->k_f_d;
  const int nx = g->nx, ny = g->ny, nz = g->nz;

# define END_RECV_NORM_E(X,Y,Z) BEGIN_PRIMITIVE {                              \
    const int face = (i+j+k) < 0 ? n##X+1 : 0;                                 \
    const int X = face;                                                        \
    const float d##X = g->d##X;                                                \
    NodeMDRangePolicy X##_node({1,1}, {n##Y+2,n##Z+2});                        \
    Kokkos::parallel_for("end_recv_norm_e<" #X #Y #Z ">", X##_node,            \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      float lw = rbuf_d(0);                                                    \
      const float rw = (2.*d##X) / (lw + d##X);                                \
      lw = (lw - d##X) / (lw + d##X);                                          \
      const int idx = 1 + (Z-1)*(n##Y+1) + (Y-1);                              \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::e##X) = rw*rbuf_d(idx)         \
        + lw*k_field(VOXEL(x+i,y+j,z+k,nx,ny,nz), field_var::e##X);            \
    });                                                                        \
  } END_PRIMITIVE

  float* p = static_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    rbuf.modify_host();
    //rbuf.sync_device();
    sync_comm_buffer<Kokkos::DefaultExecutionSpace>(rbuf);
    Kokkos::fence();
    if constexpr (i!=0 && j==0 && k==0) {
      END_RECV_NORM_E(x,y,z);
    } else if constexpr (i==0 && j!=0 && k==0) {
      END_RECV_NORM_E(y,z,x);
    } else if constexpr (i==0 && j==0 && k!=0) {
      END_RECV_NORM_E(z,x,y);
    }
  }
#undef END_RECV_NORM_E
}

template<int i, int j, int k> 
void 
end_send_ghost_norm_e_kokkos(field_array_t* RESTRICT field) {
  end_send_port_k(i,j,k, field->g);
}

void
kokkos_end_remote_ghost_norm_e( field_array_t      * ALIGNED(128) field,
                         const grid_t *              g,
                            field_buffers_t&            f_buffers) {
  end_recv_ghost_norm_e_kokkos<-1,  0,  0>(field);
  end_recv_ghost_norm_e_kokkos< 0, -1,  0>(field);
  end_recv_ghost_norm_e_kokkos< 0,  0, -1>(field);
  end_recv_ghost_norm_e_kokkos< 1,  0,  0>(field);
  end_recv_ghost_norm_e_kokkos< 0,  1,  0>(field);
  end_recv_ghost_norm_e_kokkos< 0,  0,  1>(field);

  end_send_ghost_norm_e_kokkos<-1,  0,  0>(field);
  end_send_ghost_norm_e_kokkos< 0, -1,  0>(field);
  end_send_ghost_norm_e_kokkos< 0,  0, -1>(field);
  end_send_ghost_norm_e_kokkos< 1,  0,  0>(field);
  end_send_ghost_norm_e_kokkos< 0,  1,  0>(field);
  end_send_ghost_norm_e_kokkos< 0,  0,  1>(field);
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

template <int i, int j, int k> 
void 
begin_recv_ghost_div_b_kokkos(field_array_t* fa) {
  const grid_t* g = fa->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = get_recv_dualview(fa, i, j, k);;
  if constexpr (i!=0 && j==0 && k==0) {
    size = (1 + ny*nz)*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = (1 + nz*nx)*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = (1 + nx*ny)*sizeof(float);
  }
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  BEGIN_RECV_PORT_K(i,j,k,size,g,rbuf_d,rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_ghost_div_b_kokkos(field_array_t* fa) {
  const grid_t* g = fa->g;
  Kokkos::DualView<float*> sbuf = get_send_dualview(fa, i,j,k);
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = fa->k_f_d;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  size_t size;

# define BEGIN_SEND_DIV_B(X,Y,Z) BEGIN_PRIMITIVE {                             \
    size = (1+n##Y*n##Z)*sizeof(float);                                        \
    const int face = (i+j+k) < 0 ? 1 : n##X;                                   \
    const float d##X = g->d##X;                                                \
    NodeMDRangePolicy X##_face({1,1}, {n##Y+2,n##Z+2});                        \
    Kokkos::parallel_for("begin_send_div_b<" #X #Y #Z ">", X##_face,           \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int X = face;                                                      \
      const int idx = 1 + (Z-1)*n##Y + (Y-1);                                  \
      if(idx == 1)                                                             \
        sbuf_d(0) = d##X;                                                      \
      sbuf_d(idx) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err);      \
    });                                                                        \
  } END_PRIMITIVE
  
  if constexpr (i!=0 && j==0 && k==0) {
    BEGIN_SEND_DIV_B(x,y,z);
  } else if constexpr (i==0 && j!=0 && k==0) {
    BEGIN_SEND_DIV_B(y,z,x);
  } else if constexpr (i==0 && j==0 && k!=0) {
    BEGIN_SEND_DIV_B(z,x,y);
  }
#undef BEGIN_SEND_DIV_B

  Kokkos::fence();
  sbuf.modify<Kokkos::DefaultExecutionSpace>();
  sync_comm_buffer<Kokkos::DefaultHostExecutionSpace>(sbuf);
  BEGIN_SEND_PORT_K(i,j,k,size,g,sbuf_d,sbuf_h);
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
k_begin_remote_ghost_div_b(field_array_t* ALIGNED(128) fa, const grid_t* g, field_buffers_t& fb) {
  // Start receiving
  begin_recv_ghost_div_b_kokkos<-1,  0,  0>(fa);
  begin_recv_ghost_div_b_kokkos< 0, -1,  0>(fa);
  begin_recv_ghost_div_b_kokkos< 0,  0, -1>(fa);
                                          
  begin_recv_ghost_div_b_kokkos< 1,  0,  0>(fa);
  begin_recv_ghost_div_b_kokkos< 0,  1,  0>(fa);
  begin_recv_ghost_div_b_kokkos< 0,  0,  1>(fa);

  // Start sending
  begin_send_ghost_div_b_kokkos<-1,  0,  0>(fa);
  begin_send_ghost_div_b_kokkos< 0, -1,  0>(fa);
  begin_send_ghost_div_b_kokkos< 0,  0, -1>(fa);
                                          
  begin_send_ghost_div_b_kokkos< 1,  0,  0>(fa);
  begin_send_ghost_div_b_kokkos< 0,  1,  0>(fa);
  begin_send_ghost_div_b_kokkos< 0,  0,  1>(fa);
}

template<int i, int j, int k> 
void 
end_recv_ghost_div_b_kokkos(field_array_t* fa) {
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
    rbuf.modify_host(); 
    //rbuf.sync_device(); 
    sync_comm_buffer<Kokkos::DefaultExecutionSpace>(rbuf);
    Kokkos::fence();

# define END_RECV_DIV_B(X,Y,Z) BEGIN_PRIMITIVE {                               \
    face = (i+j+k)<0 ? 1 : n##X;			                                         \
    const int X = face;                                                        \
    const float d##X = g->d##X;                                                \
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> X##_face({1, 1}, {n##Z+1, n##Y+1}); \
    Kokkos::parallel_for("end_recv_ghost_div_b<X##Y##Z>", X##_face,            \
    KOKKOS_LAMBDA(const int Z, const int Y) {                                  \
      float lw = rbuf_d(0);                                                    \
      float rw = (2. * d##X) / (lw + d##X);                                    \
      lw = (lw - d##X) / (lw + d##X);                                          \
      const int idx = 1 + (Z-1)*n##Y + (Y-1);                                  \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_b_err) = rw * rbuf_d(idx)  \
        + lw*k_field(VOXEL(x+i, y+j, z+k, nx, ny, nz), field_var::div_b_err);  \
    });                                                                        \
  } END_PRIMITIVE

    if constexpr (i!=0 && j==0 && k==0) {
      END_RECV_DIV_B(x,y,z);
    } else if constexpr (i==0 && j!=0 && k==0) {
      END_RECV_DIV_B(y,z,x);
    } else if constexpr (i==0 && j==0 && k!=0) {
      END_RECV_DIV_B(z,x,y);
    }
  }
#undef END_RECV_DIV_B
}

template<int i, int j, int k> 
void 
end_send_ghost_div_b_kokkos(field_array_t* RESTRICT field) {
  end_send_port_k(i,j,k, field->g);
}

void k_end_remote_ghost_div_b(field_array_t* ALIGNED(128) fa, const grid_t* g, field_buffers_t& fb) {
  // End receiving
  end_recv_ghost_div_b_kokkos<-1,  0,  0>(fa);
  end_recv_ghost_div_b_kokkos< 0, -1,  0>(fa);
  end_recv_ghost_div_b_kokkos< 0,  0, -1>(fa);
                                        
  end_recv_ghost_div_b_kokkos< 1,  0,  0>(fa);
  end_recv_ghost_div_b_kokkos< 0,  1,  0>(fa);
  end_recv_ghost_div_b_kokkos< 0,  0,  1>(fa);
                                        
  // End sending                          
  end_send_ghost_div_b_kokkos<-1,  0,  0>(fa);
  end_send_ghost_div_b_kokkos< 0, -1,  0>(fa);
  end_send_ghost_div_b_kokkos< 0,  0, -1>(fa);
                                        
  end_send_ghost_div_b_kokkos< 1,  0,  0>(fa);
  end_send_ghost_div_b_kokkos< 0,  1,  0>(fa);
  end_send_ghost_div_b_kokkos< 0,  0,  1>(fa);
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

template<int i, int j, int k> 
void 
begin_recv_tang_e_norm_b(field_array_t* fa) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  if constexpr (i!=0 && j==0 && k==0) {
    size = (2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz)*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = (2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx)*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = (2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny)*sizeof(float);
  }
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_tang_e_norm_b(field_array_t* fa) {
  const grid_t* g = fa->g;
  Kokkos::DualView<float*> sbuf = get_send_dualview(fa, i,j,k);
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = fa->k_f_d;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  size_t size;

# define BEGIN_SEND_TANG_E_NORM_B(X,Y,Z) BEGIN_PRIMITIVE {                     \
    size = (2*n##Y*(n##Z+1) + 2*n##Z*(n##Y+1) + n##Y*n##Z)*sizeof(float);      \
    const int face = (i+j+k) < 0 ? 1 : n##X + 1;                               \
    const int X = face;                                                        \
    FaceMDRangePolicy X##_face({1,1}, {n##Y+1,n##Z+1});                        \
    EdgeMDRangePolicy Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});                     \
    EdgeMDRangePolicy Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});                     \
    Kokkos::parallel_for("begin_send_tang_e_norm_b<" #X #Y #Z "> " #X "face", X##_face,          \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = (Z-1)*n##Y + (Y-1);                                      \
      sbuf_d(idx) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cb##X);          \
    });                                                                        \
    Kokkos::parallel_for("begin_send_tang_e_norm_b<" #X #Y #Z "> " #Y #Z "edge", Y##Z##_edge, \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = n##Z*n##Y + 2*((Z-1)*n##Y + (Y-1));;                     \
      sbuf_d(idx)   = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::e##Y);         \
      sbuf_d(idx+1) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tca##Y);       \
    });                                                                        \
    Kokkos::parallel_for("begin_send_tang_e_norm_b<" #X #Y #Z "> " #Y #Z "edge", Z##Y##_edge, \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = n##Z*n##Y + 2*n##Y*(n##Z+1) + 2*((Z-1)*(n##Y+1) + (Y-1));            \
      sbuf_d(idx)   = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::e##Z);         \
      sbuf_d(idx+1) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tca##Z);       \
    });                                                                        \
  } END_PRIMITIVE
  
  if constexpr (i!=0 && j==0 && k==0) {
    BEGIN_SEND_TANG_E_NORM_B(x,y,z);
  } else if constexpr (i==0 && j!=0 && k==0) {
    BEGIN_SEND_TANG_E_NORM_B(y,z,x);
  } else if constexpr (i==0 && j==0 && k!=0) {
    BEGIN_SEND_TANG_E_NORM_B(z,x,y);
  }
#undef BEGIN_SEND_TANG_E_NORM_B

  Kokkos::fence();
  sbuf.modify<Kokkos::DefaultExecutionSpace>();
  //sbuf.sync<Kokkos::DefaultHostExecutionSpace>();
  sync_comm_buffer<Kokkos::DefaultHostExecutionSpace>(sbuf);
  BEGIN_SEND_PORT_K(i,j,k,size,g,sbuf_d,sbuf_h);
}

template<int i, int j, int k> 
double 
end_recv_tang_e_norm_b(field_array_t* fa) {
  int face;
  const grid_t* g = fa->g;
  field_buffers_t* fb = fa->fb;
  double err=0.0, err_temp=0.0;
  float* p = reinterpret_cast<float*>(end_recv_port_k(i,j,k,g));
  if(p) {
    Kokkos::DualView<float*> rbuf = fb->recv_buffer[BOUNDARY(i,j,k)];
    auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
    auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
    const k_field_t& k_field = fa->k_f_d;
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    rbuf.modify_host(); 
    //rbuf.sync_device(); 
    sync_comm_buffer<Kokkos::DefaultExecutionSpace>(rbuf);
    Kokkos::fence();

# define END_RECV_TANG_E_NORM_B(X, Y, Z) BEGIN_PRIMITIVE {                     \
    face = (i+j+k)<0 ? 1 : n##X;			                                         \
    const int X = face;                                                        \
    FaceMDRangePolicy X##_face({1,1}, {n##Y+1,n##Z+1});                        \
    EdgeMDRangePolicy Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});                     \
    EdgeMDRangePolicy Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});                     \
    Kokkos::parallel_reduce("end_recv_tang_e_norm_b<" #X #Y #Z "> " #X "face", X##_face,          \
    KOKKOS_LAMBDA(const int Y, const int Z, double& error) {                   \
      const int idx = (Z-1)*n##Y + (Y-1);                                      \
      const double w1 = static_cast<double>(rbuf_d(idx));                      \
      const double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cb##X)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::cb##X) = static_cast<float>(0.5*(w1+w2)); \
      error += (w1-w2)*(w1-w2);                                                \
    }, err_temp);                                                              \
    err += err_temp;                                                           \
    err_temp = 0.0f;                                                           \
    Kokkos::parallel_reduce("end_recv_tang_e_norm_b<" #X #Y #Z "> " #Y #Z "edge", Y##Z##_edge, \
    KOKKOS_LAMBDA(const int Y, const int Z, double& error) {                   \
      const int idx = n##Z*n##Y + 2*((Z-1)*n##Y + (Y-1));;                     \
      double w1 = static_cast<double>(rbuf_d(idx));                            \
      double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::e##Y)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::e##Y) = static_cast<float>(0.5*(w1+w2)); \
      error += (w1-w2)*(w1-w2);                                                \
      w1 = static_cast<double>(rbuf_d(idx + 1));                               \
      w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tca##Y)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tca##Y) = static_cast<float>(0.5*(w1+w2)); \
    }, err_temp);                                                              \
    err += err_temp;                                                           \
    err_temp = 0.0f;                                                           \
    Kokkos::parallel_reduce("end_recv_tang_e_norm_b<" #X #Y #Z "> " #Y #Z "edge", Z##Y##_edge, \
    KOKKOS_LAMBDA(const int Y, const int Z, double& error) {                   \
      const int idx = n##Z*n##Y + 2*n##Y*(n##Z+1) + 2*((Z-1)*(n##Y+1) + (Y-1));\
      double w1 = static_cast<double>(rbuf_d(idx));                            \
      double w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::e##Z)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::e##Z) = static_cast<float>(0.5*(w1+w2)); \
      error += (w1-w2)*(w1-w2);                                                \
      w1 = static_cast<double>(rbuf_d(idx + 1));                               \
      w2 = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tca##Z)); \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::tca##Z) = static_cast<float>(0.5*(w1+w2)); \
    }, err_temp);                                                              \
    err += err_temp;                                                           \
  } END_PRIMITIVE

    if constexpr (i!=0 && j==0 && k==0) {
      END_RECV_TANG_E_NORM_B(x,y,z);
    } else if constexpr (i==0 && j!=0 && k==0) {
      END_RECV_TANG_E_NORM_B(y,z,x);
    } else if constexpr (i==0 && j==0 && k!=0) {
      END_RECV_TANG_E_NORM_B(z,x,y);
    }
  }
#undef END_RECV_DIV_B
  return err;
}

template<int i, int j, int k> 
void 
end_send_tang_e_norm_b(field_array_t* RESTRICT field) {
  end_send_port_k(i,j,k, field->g);
}

double
synchronize_tang_e_norm_b_kokkos( field_array_t * RESTRICT fa ) {
  const grid_t * RESTRICT g = fa->g;
  double err = 0, gerr;

  if( !fa ) ERROR(( "Bad args" ));

  k_local_adjust_tang_e( fa, g );
  k_local_adjust_norm_b( fa, g );

  // Exchange x-faces
  begin_recv_tang_e_norm_b<-1,  0,  0>(fa);
  begin_recv_tang_e_norm_b< 1,  0,  0>(fa);
  begin_send_tang_e_norm_b<-1,  0,  0>(fa);
  begin_send_tang_e_norm_b< 1,  0,  0>(fa);
  err += end_recv_tang_e_norm_b<-1,  0,  0>(fa);
  err += end_recv_tang_e_norm_b< 1,  0,  0>(fa);
  end_send_tang_e_norm_b<-1,  0,  0>(fa);
  end_send_tang_e_norm_b< 1,  0,  0>(fa);

  // Exchange y-faces
  begin_recv_tang_e_norm_b<0, -1,  0>(fa);
  begin_recv_tang_e_norm_b<0,  1,  0>(fa);
  begin_send_tang_e_norm_b<0, -1,  0>(fa);
  begin_send_tang_e_norm_b<0,  1,  0>(fa);
  err += end_recv_tang_e_norm_b<0, -1,  0>(fa);
  err += end_recv_tang_e_norm_b<0,  1,  0>(fa);
  end_send_tang_e_norm_b<0, -1,  0>(fa);
  end_send_tang_e_norm_b<0,  1,  0>(fa);

  // Exchange z-faces
  begin_recv_tang_e_norm_b<0,  0, -1>(fa);
  begin_recv_tang_e_norm_b<0,  0,  1>(fa);
  begin_send_tang_e_norm_b<0,  0, -1>(fa);
  begin_send_tang_e_norm_b<0,  0,  1>(fa);
  err += end_recv_tang_e_norm_b<0,  0, -1>(fa);
  err += end_recv_tang_e_norm_b<0,  0,  1>(fa);
  end_send_tang_e_norm_b<0,  0, -1>(fa);
  end_send_tang_e_norm_b<0,  0,  1>(fa);

  mp_allsum_d( &err, &gerr, 1 );
  return gerr;
}

template<int i, int j, int k> 
void 
begin_recv_jf(field_array_t* fa) {
  const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = fa->fb->recv_buffer[BOUNDARY(i,j,k)];
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  if constexpr (i!=0 && j==0 && k==0) {
    size = (ny*(nz+1) + nz*(ny+1) + 1)*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = (nz*(nx+1) + nx*(nz+1) + 1)*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = (nx*(ny+1) + ny*(nx+1) + 1)*sizeof(float);
  }
  BEGIN_RECV_PORT_K(i,j,k,size,fa->g, rbuf_d, rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_jf(field_array_t* fa) {
  const grid_t* g = fa->g;
  Kokkos::DualView<float*> sbuf = get_send_dualview(fa, i,j,k);
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = fa->k_f_d;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  size_t size = 0;

# define BEGIN_SEND_JF(X,Y,Z) BEGIN_PRIMITIVE {                                \
    size = (1 + n##Y*(n##Z+1) + n##Z*(n##Y+1))*sizeof(float);                  \
    const int face = (i+j+k) < 0 ? 1 : n##X + 1;                               \
    const int X = face;                                                        \
    const float d##X = g->d##X;                                                \
    EdgeMDRangePolicy Y##Z##_edge({1,1}, {n##Y+1,n##Z+2});                     \
    EdgeMDRangePolicy Z##Y##_edge({1,1}, {n##Y+2,n##Z+1});                     \
    Kokkos::parallel_for("begin_send_jf<" #X #Y #Z "> " #Y #Z "edge", Y##Z##_edge, \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = 1 + (Z-1)*n##Y + (Y-1);                                  \
      if(idx == 1)                                                             \
        sbuf_d(0) = d##X;                                                      \
      sbuf_d(idx) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jf##Y);          \
    });                                                                        \
    Kokkos::parallel_for("begin_send_jf<" #X #Y #Z "> " #Z #Y "edge", Z##Y##_edge, \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = 1 + (n##Z+1)*n##Y + (Z-1)*(n##Y+1) + (Y-1);              \
      sbuf_d(idx) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jf##Z);          \
    });                                                                        \
  } END_PRIMITIVE
  
  if constexpr (i!=0 && j==0 && k==0) {
    BEGIN_SEND_JF(x,y,z);
  } else if constexpr (i==0 && j!=0 && k==0) {
    BEGIN_SEND_JF(y,z,x);
  } else if constexpr (i==0 && j==0 && k!=0) {
    BEGIN_SEND_JF(z,x,y);
  }
#undef BEGIN_SEND_JF

  Kokkos::fence();
  sbuf.modify<Kokkos::DefaultExecutionSpace>();
  sync_comm_buffer<Kokkos::DefaultHostExecutionSpace>(sbuf);
  BEGIN_SEND_PORT_K(i,j,k,size,g,sbuf_d,sbuf_h);
}

template<int i, int j, int k> 
void 
end_recv_jf(field_array_t* fa) {
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
    rbuf.modify_host();
    sync_comm_buffer<Kokkos::DefaultExecutionSpace>(rbuf);
    Kokkos::fence();

# define END_RECV_JF(X,Y,Z) BEGIN_PRIMITIVE {                                  \
    face = (i+j+k)<0 ? n##X+1 : 1;			                                       \
    const int X = face;                                                        \
    const float d##X = g->d##X;                                                \
    EdgeMDRangePolicy Y##Z##_edge({1, 1}, {n##Y+1, n##Z+2});                   \
    EdgeMDRangePolicy Z##Y##_edge({1, 1}, {n##Y+2, n##Z+1});                   \
    Kokkos::parallel_for("end_recv_jf<"#X#Y#Z">" #Y #Z "edge", Y##Z##_edge,    \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = 1 + (Z-1)*n##Y + (Y-1);                                  \
      const float jf##Y = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jf##Y);    \
      float rw = rbuf_d(0);                                                    \
      float lw = rw + d##X;                                                    \
      rw /= lw;                                                                \
      lw  = d##X/lw;                                                           \
      lw += lw;                                                                \
      rw += rw;                                                                \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jf##Y) = lw*jf##Y + rw*rbuf_d(idx); \
    });                                                                        \
    Kokkos::parallel_for("end_recv_jf<"#X#Y#Z">" #Z #Y "edge", Z##Y##_edge,    \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx = 1 + (n##Z+1)*n##Y + (Z-1)*(n##Y+1) + (Y-1);              \
      const float jf##Z = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jf##Z);    \
      float rw = rbuf_d(0);                                                    \
      float lw = rw + d##X;                                                    \
      rw /= lw;                                                                \
      lw  = d##X/lw;                                                           \
      lw += lw;                                                                \
      rw += rw;                                                                \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::jf##Z) = lw*jf##Z + rw*rbuf_d(idx); \
    });                                                                        \
  } END_PRIMITIVE

    if constexpr (i!=0 && j==0 && k==0) {
      END_RECV_JF(x,y,z);
    } else if constexpr (i==0 && j!=0 && k==0) {
      END_RECV_JF(y,z,x);
    } else if constexpr (i==0 && j==0 && k!=0) {
      END_RECV_JF(z,x,y);
    }
  }
#undef END_RECV_JF
}

template<int i, int j, int k> 
void 
end_send_jf(field_array_t* RESTRICT field) {
  end_send_port_k(i,j,k, field->g);
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

void 
k_synchronize_jf(field_array_t* RESTRICT fa) {
  if(!fa) ERROR(( "Bad args" ));
  grid_t* RESTRICT g = fa->g;

  k_local_adjust_jf(fa, g);

  // Exchange x-faces
  begin_recv_jf<-1, 0, 0>(fa);
  begin_recv_jf< 1, 0, 0>(fa);
  begin_send_jf<-1, 0, 0>(fa);
  begin_send_jf< 1, 0, 0>(fa);
  end_recv_jf<-1, 0, 0>(fa);
  end_recv_jf< 1, 0, 0>(fa);
  end_send_jf<-1, 0, 0>(fa);
  end_send_jf< 1, 0, 0>(fa);

  // Exchange y-faces
  begin_recv_jf<0, -1, 0>(fa);
  begin_recv_jf<0,  1, 0>(fa);
  begin_send_jf<0, -1, 0>(fa);
  begin_send_jf<0,  1, 0>(fa);
  end_recv_jf<0, -1, 0>(fa);
  end_recv_jf<0,  1, 0>(fa);
  end_send_jf<0, -1, 0>(fa);
  end_send_jf<0,  1, 0>(fa);

  // Exchange z-faces
  begin_recv_jf<0, 0, -1>(fa);
  begin_recv_jf<0, 0,  1>(fa);
  begin_send_jf<0, 0, -1>(fa);
  begin_send_jf<0, 0,  1>(fa);
  end_recv_jf<0, 0, -1>(fa);
  end_recv_jf<0, 0,  1>(fa);
  end_send_jf<0, 0, -1>(fa);
  end_send_jf<0, 0,  1>(fa);
}

template <int i, int j, int k> 
void 
begin_recv_rho(field_array_t* fa) {
  const grid_t* g = fa->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  int size;
  Kokkos::DualView<float*> rbuf = get_recv_dualview(fa, i, j, k);;
  if constexpr (i!=0 && j==0 && k==0) {
    size = ( 1 + 2*(ny+1)*(nz+1) )*sizeof(float);
  } else if constexpr (i==0 && j!=0 && k==0) {
    size = ( 1 + 2*(nz+1)*(nx+1) )*sizeof(float);
  } else if constexpr (i==0 && j==0 && k!=0) {
    size = ( 1 + 2*(nx+1)*(ny+1) )*sizeof(float);
  }
  auto rbuf_h = rbuf.view<Kokkos::DefaultHostExecutionSpace>();
  auto rbuf_d = rbuf.view<Kokkos::DefaultExecutionSpace>();
  BEGIN_RECV_PORT_K(i,j,k,size,g,rbuf_d,rbuf_h);
}

template <int i, int j, int k> 
void 
begin_send_rho(field_array_t* fa) {
  const grid_t* g = fa->g;
  Kokkos::DualView<float*> sbuf = get_send_dualview(fa, i,j,k);
  auto sbuf_d = sbuf.view<Kokkos::DefaultExecutionSpace>();
  auto sbuf_h = sbuf.view<Kokkos::DefaultHostExecutionSpace>();
  k_field_t& k_field = fa->k_f_d;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  size_t size;

# define BEGIN_SEND_RHO(X,Y,Z) BEGIN_PRIMITIVE {                               \
    size = ( 1 + 2*(n##Y+1)*(n##Z+1) )*sizeof(float);                          \
    const int face = (i+j+k) < 0 ? 1 : n##X + 1;                               \
    const int X = face;                                                        \
    const float d##X = g->d##X;                                                \
    NodeMDRangePolicy X##_node({1,1}, {n##Y+2,n##Z+2});                        \
    Kokkos::parallel_for("begin_send_rho<" #X #Y #Z ">", X##_node,             \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      const int idx_f = 1 + 2*((Z-1)*(n##Y+1) + (Y-1));                        \
      const int idx_b = 1 + 2*((Z-1)*(n##Y+1) + (Y-1)) + 1;                    \
      if(Y+Z == 2)                                                             \
        sbuf_d(0) = d##X;                                                      \
      sbuf_d(idx_f) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);         \
      sbuf_d(idx_b) = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);         \
    });                                                                        \
  } END_PRIMITIVE
  
  if constexpr (i!=0 && j==0 && k==0) {
    BEGIN_SEND_RHO(x,y,z);
  } else if constexpr (i==0 && j!=0 && k==0) {
    BEGIN_SEND_RHO(y,z,x);
  } else if constexpr (i==0 && j==0 && k!=0) {
    BEGIN_SEND_RHO(z,x,y);
  }
#undef BEGIN_SEND_RHO

  Kokkos::fence();
  sbuf.modify<Kokkos::DefaultExecutionSpace>();
  sync_comm_buffer<Kokkos::DefaultHostExecutionSpace>(sbuf);
  BEGIN_SEND_PORT_K(i,j,k,size,g,sbuf_d,sbuf_h);
}

template<int i, int j, int k> 
void 
end_recv_rho(field_array_t* fa) {
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
    rbuf.modify_host(); 
    //rbuf.sync_device(); 
    sync_comm_buffer<Kokkos::DefaultExecutionSpace>(rbuf);
    Kokkos::fence();

# define END_RECV_RHO(X,Y,Z) BEGIN_PRIMITIVE {                                 \
    face = (i+j+k)<0 ? n##X+1 : 1;			                                       \
    const int X = face;                                                        \
    const float d##X = g->d##X;                                                \
    NodeMDRangePolicy X##_node({1, 1}, {n##Y+2, n##Z+2});                      \
    Kokkos::parallel_for("end_recv_rho<"#X#Y#Z">" #X "node", X##_node,         \
    KOKKOS_LAMBDA(const int Y, const int Z) {                                  \
      float hrw = rbuf_d(0);                                                   \
      float hlw = hrw + d##X;                                                  \
      hrw /= hlw;                                                              \
      hlw  = d##X/hlw;                                                         \
      const float lw  = 2*hlw;                                                 \
      const float rw  = 2*hrw;                                                 \
      const int idx_f = 1 + 2*((Z-1)*(n##Y+1) + (Y-1));                        \
      const int idx_b = 1 + 2*((Z-1)*(n##Y+1) + (Y-1)) + 1;                    \
      const float rhof = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof);      \
      const float rhob = k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob);      \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhof) = lw*rhof + rw*rbuf_d(idx_f);           \
      k_field(VOXEL(x,y,z,nx,ny,nz), field_var::rhob) = hlw*rhob + hrw*rbuf_d(idx_b);           \
    });                                                                        \
  } END_PRIMITIVE

    if constexpr (i!=0 && j==0 && k==0) {
      END_RECV_RHO(x,y,z);
    } else if constexpr (i==0 && j!=0 && k==0) {
      END_RECV_RHO(y,z,x);
    } else if constexpr (i==0 && j==0 && k!=0) {
      END_RECV_RHO(z,x,y);
    }
  }
#undef END_RECV_JF
}

template<int i, int j, int k> 
void 
end_send_rho(field_array_t* RESTRICT field) {
  end_send_port_k(i,j,k, field->g);
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

void 
k_synchronize_rho(field_array_t* RESTRICT fa) {
  if(!fa) ERROR(( "Bad args" ));
  grid_t* RESTRICT g = fa->g;

  k_local_adjust_rhof(fa, g);
  k_local_adjust_rhob(fa, g);

  // Exchange x-faces
  begin_recv_rho<-1, 0, 0>(fa);
  begin_recv_rho< 1, 0, 0>(fa);
  begin_send_rho<-1, 0, 0>(fa);
  begin_send_rho< 1, 0, 0>(fa);
  end_recv_rho<-1, 0, 0>(fa);
  end_recv_rho< 1, 0, 0>(fa);
  end_send_rho<-1, 0, 0>(fa);
  end_send_rho< 1, 0, 0>(fa);

  // Exchange y-faces
  begin_recv_rho<0, -1, 0>(fa);
  begin_recv_rho<0,  1, 0>(fa);
  begin_send_rho<0, -1, 0>(fa);
  begin_send_rho<0,  1, 0>(fa);
  end_recv_rho<0, -1, 0>(fa);
  end_recv_rho<0,  1, 0>(fa);
  end_send_rho<0, -1, 0>(fa);
  end_send_rho<0,  1, 0>(fa);

  // Exchange z-faces
  begin_recv_rho<0, 0, -1>(fa);
  begin_recv_rho<0, 0,  1>(fa);
  begin_send_rho<0, 0, -1>(fa);
  begin_send_rho<0, 0,  1>(fa);
  end_recv_rho<0, 0, -1>(fa);
  end_recv_rho<0, 0,  1>(fa);
  end_send_rho<0, 0, -1>(fa);
  end_send_rho<0, 0,  1>(fa);
}

