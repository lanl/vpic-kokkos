#ifndef _cylindrical_geometry_h_
#define _cylindrical_geometry_h_

template<class mesh_view_t>
class CylindricalGeometry {
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

  CylindricalGeometry(
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

private:

  // Coordinates ===============================================================

  // const float& KOKKOS_INLINE_FUNCTION
  // r_edge_x(const int voxel) const {
  //   return mesh(voxel, mesh_var::x);
  // }

  // const float& KOKKOS_INLINE_FUNCTION
  // r_edge_y(const int voxel) const {
  //   return mesh(voxel, mesh_var::x) - 0.5*dx;
  // }

  // const float& KOKKOS_INLINE_FUNCTION
  // r_edge_z(const int voxel) const {
  //   return mesh(voxel, mesh_var::x) - 0.5*dx;
  // }

  // const float& KOKKOS_INLINE_FUNCTION
  // r_face_x(const int voxel) const {
  //   return mesh(voxel, mesh_var::x) - 0.5*dx;
  // }

  // const float& KOKKOS_INLINE_FUNCTION
  // r_face_y(const int voxel) const {
  //   return mesh(voxel, mesh_var::x);
  // }

  // const float& KOKKOS_INLINE_FUNCTION
  // r_face_z(const int voxel) const {
  //   return mesh(voxel, mesh_var::x);
  // }

  // const float& KOKKOS_INLINE_FUNCTION
  // r_node(const int voxel) const {
  //   return mesh(voxel, mesh_var::x) - 0.5*dx;
  // }

  // const float& KOKKOS_INLINE_FUNCTION
  // r_voxel(const int voxel) const {
  //   return mesh(voxel, mesh_var::x);
  // }

public:

  // Differential operators ====================================================

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
    float r = mesh(voxel, mesh_var::x) - 0.5*dx;
    return py*(Azy - Az0)/r - pz*(Ayz - Ay0);
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
    float r_face = mesh(voxel, mesh_var::x);
    float r_edge = r_face - 0.5*dx;
    return (px*((r_edge+dx)*Ayx - r_edge*Ay0) - py*(Axy - Ax0)) / r_face;
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
    float r = mesh(voxel, mesh_var::x);
    return py*(Az0 - Azy)/r - pz*(Ay0 - Ayz);
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
    float r_face = mesh(voxel, mesh_var::x);
    float r_edge = r_face - 0.5*dx;
    return (px*(r_face*Ay0 - (r_face-dx)*Ayx) - py*(Ax0 - Axy)) / r_edge;
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
    return py*(f0 - fy) / mesh(voxel, mesh_var::x);
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
    float r = mesh(voxel, mesh_var::x) - 0.5*dx;
    return py*(fy - f0) / r;
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
    float r_voxel = mesh(voxel, mesh_var::x);
    float r_face = r_voxel - 0.5*dx;
    return (px*((r_face+dx)*Ax1 - r_face*Ax0) + py*(Ay1 - Ay0))/r_voxel+ pz*(Az1 - Az0);
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
    float r_edge = mesh(voxel, mesh_var::x);
    float r_node = r_edge - 0.5*dx;
    return (px*(r_edge*Ax0 - (r_edge-dx)*Ax1) + py*(Ay0 - Ay1))/r_node + pz*(Az0 - Az1);
  }

  // Field - particle operations ===============================================

  /**
   * @brief Computes the inverse volume of a voxel
   * @param voxel Voxel index
   */
  const float KOKKOS_INLINE_FUNCTION
  inverse_voxel_volume(
    int voxel
  ) const {

    return rdV/mesh(voxel, mesh_var::x);

  }

  /**
   * @brief Converts a set of field vectors from logical to Cartesian representation.
   * @param fields A set of field vectors, logical on input, Cartesian on output
   * @param voxel Voxel index
   * @param dx0 x-coordinate, logical
   * @param dy0 y-coordinate, logical
   * @param dz0 z-coordinate, logical
   */
  const void KOKKOS_INLINE_FUNCTION
  transform_interpolated_fields (
    field_vectors_t& fields, int voxel, float dx0, float dy0, float dz0
  ) const {

    fields.ey /= mesh(voxel, mesh_var::x) + 0.5f*dx*dx0;

  }

  /**
   * @brief Transforms a particle displacement and momentum from Cartesian to logical coordinates.
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
  convert_to_logical (
      int voxel,
      float  dx0,   float  dy0,   float  dz0,
      float& dispx, float& dispy, float& dispz,
      float& ux,    float& uy,    float& uz
  ) const {

    const float r0 = mesh(voxel, mesh_var::x) + 0.5f*dx*dx0;
    const float x1 = r0 + dispx;
    const float r1 = sqrtf(x1*x1 + dispy*dispy);
    const float dtheta = atan2f(dispy, x1);

    // Work in double for better energy conservation.
    const double uxd = ux;
    const double uyd = uy;
    const double x1d = x1;
    const double y1d = dispy;
    const double scale = 1.0/sqrt(x1d*x1d + y1d*y1d);

    ux = scale*( uxd*x1d + uyd*y1d);
    uy = scale*(-uxd*y1d + uyd*x1d);

    dispx  = 2*rdx*(2*r0*dispx + dispx*dispx + dispy*dispy)/(r0 + r1);
    dispy  = 2*rdy*dtheta;
    dispz *= 2*rdz;

  }

};

#endif
