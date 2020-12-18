#ifndef _cartesian_geometry_h_
#define _cartesian_geometry_h_

template<class mesh_view_t>
class CartesianGeometry {
private:

  const float px,  py,  pz;
  const float dx,  dy,  dz,  dV;
  const float rdx, rdy, rdz, rdV;
  const mesh_view_t mesh;

public:

  // TODO: Logical flaw. Setting p{x,y,z} as done here is consistent
  // with previous versions of VPIC, but it assumes that n > 1 locally
  // is the same as N > 1 globally. This is not true in the edge case
  // where the domain decomposition allows a local domain to have
  // exactly one cell.

  CartesianGeometry(
      const int nx,
      const int ny,
      const int nz,
      const float dx,
      const float dy,
      const float dz,
      const mesh_view_t mesh
  )
  : mesh(mesh),
    dx(dx), dy(dy), dz(dz), dV(dx*dy*dz),
    rdx(1./dx), rdy(1./dy), rdz(1./dz), rdV(1./(dx*dy*dz)),
    px(nx > 1 ? 1./dx : 0), // Used for calculating derivatives
    py(ny > 1 ? 1./dy : 0), // Used for calculating derivatives
    pz(nz > 1 ? 1./dz : 0)  // Used for calculating derivatives

  {

  }

  /**
   * @brief Curl of a vector A defined on the edges. Output on the faces.
   * @param voxel Voxel to compute in
   * @param Az0 Az(ix, iy, iz)
   * @param Azy Az(ix, iy+1, iz)
   * @param Ay0 Ay(ix, iy, iz)
   * @param Ayz Az(ix, iy, iz+1)
   * @returns curlx(A)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  edge_curl_x(int voxel, float Az0, float Azy, float Ay0, float Ayz) const {
    return py*(Azy - Az0) - pz*(Ayz - Ay0);
  }

  /**
   * @brief Curl of a vector A defined on the edges. Output on the faces.
   * @param voxel Voxel to compute in
   * @param Ax0 Ax(ix, iy, iz)
   * @param Axz Ax(ix, iy, iz+1)
   * @param Az0 Az(ix, iy, iz)
   * @param Azx Az(ix+1, iy, iz)
   * @returns curly(A)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  edge_curl_y(int voxel, float Ax0, float Axz, float Az0, float Azx) const {
    return pz*(Axz - Ax0) - px*(Azx - Az0);
  }

  /**
   * @brief Curl of a vector A defined on the edges. Output on the faces.
   * @param voxel Voxel to compute in
   * @param Ay0 Az(ix, iy, iz)
   * @param Ayx Az(ix+1, iy, iz)
   * @param Ax0 Ax(ix, iy, iz)
   * @param Axy Ax(ix, iy+1, iz)
   * @returns curlz(A)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  edge_curl_z(int voxel, float Ay0, float Ayx, float Ax0, float Axy) const {
    return px*(Ayx - Ay0) - py*(Axy - Ax0);
  }

  /**
   * @brief Curl of a vector A defined on the faces. Output on the edges.
   * @param voxel Voxel to compute in
   * @param Az0 Az(ix, iy, iz)
   * @param Azy Az(ix, iy-1, iz)
   * @param Ay0 Ay(ix, iy, iz)
   * @param Ayz Az(ix, iy, iz-1)
   * @returns curlx(A)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  face_curl_x(int voxel, float Az0, float Azy, float Ay0, float Ayz) const {
    return py*(Az0 - Azy) - pz*(Ay0 - Ayz);
  }

  /**
   * @brief Curl of a vector A defined on the faces. Output on the edges.
   * @param voxel Voxel to compute in
   * @param Ax0 Ax(ix, iy, iz)
   * @param Axz Ax(ix, iy, iz-1)
   * @param Az0 Az(ix, iy, iz)
   * @param Azx Az(ix-1, iy, iz)
   * @returns curly(A)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  face_curl_y(int voxel, float Ax0, float Axz, float Az0, float Azx) const {
    return pz*(Ax0 - Axz) - px*(Az0 - Azx);
  }

  /**
   * @brief Curl of a vector A defined on the faces. Output on the edges.
   * @param voxel Voxel to compute in
   * @param Ay0 Ay(ix, iy, iz)
   * @param Ayx Ay(ix-1, iy, iz)
   * @param Ax0 Ax(ix, iy, iz)
   * @param Axy Ax(ix, iy-1, iz)
   * @returns curlz(A)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  face_curl_z(int voxel, float Ay0, float Ayx, float Ax0, float Axy) const {
    return px*(Ay0 - Ayx) - py*(Ax0 - Axy);
  }

  /**
   * @brief Gradient of a scalar defined on cell-centers. Output on the faces.
   * @param voxel Voxel to compute in
   * @param f0 f(ix, iy, iz)
   * @param fx f(ix-1, iy, iz)
   * @returns gradx(f)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  cell_gradient_x(int voxel, float f0, float fx) const {
    return px*(f0 - fx);
  }

  /**
   * @brief Gradient of a scalar defined on cell-centers. Output on the faces.
   * @param voxel Voxel to compute in
   * @param f0 f(ix, iy, iz)
   * @param fy f(ix, iy-1, iz)
   * @returns grady(f)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  cell_gradient_y(int voxel, float f0, float fy) const {
    return py*(f0 - fy);
  }

    /**
   * @brief Gradient of a scalar defined on cell-centers. Output on the faces.
   * @param voxel Voxel to compute in
   * @param f0 f(ix, iy, iz)
   * @param fz f(ix, iy, iz-1)
   * @returns gradz(f)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  cell_gradient_z(int voxel, float f0, float fz) const {
    return pz*(f0 - fz);
  }

  /**
   * @brief Gradient of a scalar defined on nodes. Output on the edges.
   * @param voxel Voxel to compute in
   * @param f0 f(ix, iy, iz)
   * @param fx f(ix+1, iy, iz)
   * @returns gradx(f)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  node_gradient_x(int voxel, float f0, float fx) const {
    return px*(fx - f0);
  }

  /**
   * @brief Gradient of a scalar defined on nodes. Output on the edges.
   * @param voxel Voxel to compute in
   * @param f0 f(ix, iy, iz)
   * @param fy f(ix, iy+1, iz)
   * @returns grady(f)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  node_gradient_y(int voxel, float f0, float fy) const {
    return py*(fy - f0);
  }

    /**
   * @brief Gradient of a scalar defined on nodes. Output on the edges.
   * @param voxel Voxel to compute in
   * @param f0 f(ix, iy, iz)
   * @param fz f(ix, iy, iz+1)
   * @returns gradz(f)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  node_gradient_z(int voxel, float f0, float fz) const {
    return pz*(fz - f0);
  }

   /**
   * @brief Divergence of a vector A defined on the faces. Output on the cell center.
   * @param voxel Voxel to compute in
   * @param Ax0 Ax(ix, iy, iz)
   * @param Ax1 Ax(ix+1, iy, iz)
   * @param Ay0 Ay(ix, iy, iz)
   * @param Ay1 Ay(ix, iy+1, iz)
   * @param Az0 Az(ix, iy, iz)
   * @param Az1 Az(ix, iy, iz+1)
   * @returns div(A)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  face_divergence(int voxel, float Ax0, float Ax1, float Ay0, float Ay1, float Az0, float Az1) const {
    return px*(Ax1 - Ax0) + py*(Ay1 - Ay0) + pz*(Az1 - Az0);
  }

   /**
   * @brief Divergence of a vector A defined on the edges. Output on the nodes.
   * @param voxel Voxel to compute in
   * @param Ax0 Ax(ix, iy, iz)
   * @param Ax1 Ax(ix-1, iy, iz)
   * @param Ay0 Ay(ix, iy, iz)
   * @param Ay1 Ay(ix, iy-1, iz)
   * @param Az0 Az(ix, iy, iz)
   * @param Az1 Az(ix, iy, iz-1)
   * @returns div(A)(ix, iy, iz)
   */
  const float KOKKOS_INLINE_FUNCTION
  edge_divergence(int voxel, float Ax0, float Ax1, float Ay0, float Ay1, float Az0, float Az1) const {
    return px*(Ax0 - Ax1) + py*(Ay0 - Ay1) + pz*(Az0 - Az1);
  }

  /**
   * @brief Computes the inverse volume of a voxel
   * @param voxel Voxel index
   */
  const float KOKKOS_INLINE_FUNCTION
  inverse_voxel_volume (int voxel) const {

    return rdV;

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param ex Ex(i, j, k)
   * @param ex_y Ex(i, j+1, k)
   * @param ex_z Ex(i, j, k+1)
   * @param ex_yz Ex(i, j+1, k+1)
   */
  const void KOKKOS_INLINE_FUNCTION
  prescale_interpolated_ex (
    const int voxel,
    float& ex, float& ex_y, float& ex_z, float& ex_yz
  ) const {

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param ey Ey(i, j, k)
   * @param ey_x Ey(i+1, j, k)
   * @param ey_z Ey(i, j, k+1)
   * @param ey_xz Ey(i+1, j, k+1)
   */
  const void KOKKOS_INLINE_FUNCTION
  prescale_interpolated_ey (
    const int voxel,
    float& ey, float& ey_x, float& ey_z, float& ey_xz
  ) const {

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param ez Ez(i, j, k)
   * @param ez_x Ez(i+1, j, k)
   * @param ez_y Ez(i, j+1, k)
   * @param ez_xy Ez(i+1, j+1, k)
   */
  const void KOKKOS_INLINE_FUNCTION
  prescale_interpolated_ez (
    const int voxel,
    float& ez, float& ez_x, float& ez_y, float& ez_xy
  ) const {

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param cbx cBx(i, j, k)
   * @param cbx_x cBx(i+1, j, k)
   */
  const void KOKKOS_INLINE_FUNCTION
  prescale_interpolated_cbx (
    const int voxel,
    float& cbx, float& cbx_x
  ) const {

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param cby cBy(i, j, k)
   * @param cby_y cBy(i, j+1, k)
   */
  const void KOKKOS_INLINE_FUNCTION
  prescale_interpolated_cby (
    const int voxel,
    float& cby, float& cby_y
  ) const {

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param cbz cBz(i, j, k)
   * @param cbz_z cBz(i, j, k+1)
   */
  const void KOKKOS_INLINE_FUNCTION
  prescale_interpolated_cbz (
    const int voxel,
    float& cbz, float& cbz_z
  ) const {

  }

  /**
   * @brief Apply a postscaling to the interpolated fields.
   * @param fields A set of field vectors
   * @param voxel Voxel index
   * @param dx0 x-coordinate, logical
   * @param dy0 y-coordinate, logical
   * @param dz0 z-coordinate, logical
   */
  const void KOKKOS_INLINE_FUNCTION
  postscale_interpolated_fields (
    field_vectors_t& fields, int voxel, float dx0, float dy0, float dz0
  ) const {

  }

  /**
   * @brief Transforms a vector between locally Cartesian frames
   * @param voxel Voxel index
   * @param dx0 Starting x-coordinate, logical
   * @param dy0 Starting y-coordinate, logical
   * @param dz0 Starting z-coordinate, logical
   * @param dispx x-displacement, locally Cartesian in inital frame
   * @param dispy y-displacement, locally Cartesian in inital frame
   * @param dispz z-displacement, locally Cartesian in inital frame
   * @param Ax x-component, locally Cartesian in initial frame on input and in displaced frame on output
   * @param Ay y-component, locally Cartesian in initial frame on input and in displaced frame on output
   * @param Az z-component, locally Cartesian in initial frame on input and in displaced frame on output
   */
  template<class T>
  void KOKKOS_INLINE_FUNCTION
  realign_cartesian_vector (
    const int voxel,
    const float  dx0,   const float  dy0,   const float  dz0,
    const float  dispx, const float  dispy, const float  dispz,
    /**/  float& Ax,    /**/  float& Ay,    /**/  float& Az
  ) const {

  }

  /**
   * @brief Transforms a particle displacement from a local Cartesian to logical frames.
   * @param voxel Voxel index
   * @param dx0 Starting x-coordinate, logical
   * @param dy0 Starting y-coordinate, logical
   * @param dz0 Starting z-coordinate, logical
   * @param dispx x-displacement, Cartesian on input, logical on output
   * @param dispy y-displacement, Cartesian on input, logical on output
   * @param dispz z-displacement, Cartesian on input, logical on output
   * @param ux x-momentum, Cartesian on input, logical on output
   * @param uy y-momentum, Cartesian on input, logical on output
   * @param uz z-momentum, Cartesian on input, logical on output
   */
  void KOKKOS_INLINE_FUNCTION
  displacement_to_half_logical (
    const int voxel,
    const float  dx0,   const float  dy0,   const float  dz0,
    /**/  float& dispx, /**/  float& dispy, /**/  float& dispz
  ) const {

    dispx *= rdx;
    dispy *= rdy;
    dispz *= rdz;

  }

  /**
   * @brief Calculates the age and direction required to hit a boundary
   * @param voxel Voxel index
   * @param dx0 Starting x-coordinate, logical
   * @param dy0 Starting y-coordinate, logical
   * @param dz0 Starting z-coordinate, logical
   * @param dispx x-displacement, Cartesian on input, logical on output
   * @param dispy y-displacement, Cartesian on input, logical on output
   * @param dispz z-displacement, Cartesian on input, logical on output
   * @param axis direction towards boundary
   * @param dir direction of the boundary
   * @param age age until the particle hits the boundary
   */
  void KOKKOS_INLINE_FUNCTION
  age_to_boundary(
    const int voxel,
    const float dx0,   const float dy0,   const float dz0,
    const float dispx, const float dispy, const float dispz,
    /**/  int&  axis,  /**/  int&  dir,   /**/  float& age
  ) const {

    const int xdir = (dispx>0) ? 1 : -1;
    const int ydir = (dispy>0) ? 1 : -1;
    const int zdir = (dispz>0) ? 1 : -1;

    const float xage = ( dispx == 0 ) ? 3.4e38 : dx*(xdir - dx0)/dispx;
    const float yage = ( dispy == 0 ) ? 3.4e38 : dy*(ydir - dy0)/dispy;
    const float zage = ( dispz == 0 ) ? 3.4e38 : dz*(zdir - dz0)/dispz;

    age = 2; axis = 3;
    if( zage < age ) { age = zage; axis = 2; dir = zdir; }
    if( yage < age ) { age = yage; axis = 1; dir = ydir; }
    if( xage < age ) { age = xage; axis = 0; dir = xdir; }
    age *= 0.5;

  }

};

#endif
