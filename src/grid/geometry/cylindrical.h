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

  const float domain_volume;

  // TODO: Logical flaw. Setting p{x,y,z} as done here is consistent
  // with previous versions of VPIC, but it assumes that n > 1 locally
  // is the same as N > 1 globally. This is not true in the edge case
  // where the domain decomposition allows a local domain to have
  // exactly one cell.

  CylindricalGeometry(
      const float x0, const float y0, const float z0,
      const float x1, const float y1, const float z1,
      const float dx, const float dy, const float dz,
      const int   nx, const int   ny, const int   nz,
      const mesh_view_t mesh
  )
  : mesh(mesh),
    domain_volume( 0.5*(y1-y0)*(x1*x1 - x0*x0)*(z1-z0) ),
    dx(dx), dy(dy), dz(dz), dV(dx*dy*dz),
    rdx(1./dx), rdy(1./dy), rdz(1./dz), rdV(1./(dx*dy*dz)),
    px(nx > 1 ? 1./dx : 0), // Used for calculating derivatives
    py(ny > 1 ? 1./dy : 0), // Used for calculating derivatives
    pz(nz > 1 ? 1./dz : 0)  // Used for calculating derivatives
  {

  }

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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
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
  KOKKOS_INLINE_FUNCTION float
  inverse_voxel_volume(
    int voxel
  ) const {

    return rdV/mesh(voxel, mesh_var::x);

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param ex Ex(i, j, k)
   * @param ex_y Ex(i, j+1, k)
   * @param ex_z Ex(i, j, k+1)
   * @param ex_yz Ex(i, j+1, k+1)
   */
  KOKKOS_INLINE_FUNCTION void
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
  KOKKOS_INLINE_FUNCTION void
  prescale_interpolated_ey (
    const int voxel,
    float& ey, float& ey_x, float& ey_z, float& ey_xz
  ) const {

    // Multiply by R
    const float r0 = mesh(voxel, mesh_var::x) - 0.5f*dx;
    const float r1 = r0 + dx;

    ey    *= r0;
    ey_z  *= r0;
    ey_x  *= r1;
    ey_xz *= r1;

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param ez Ez(i, j, k)
   * @param ez_x Ez(i+1, j, k)
   * @param ez_y Ez(i, j+1, k)
   * @param ez_xy Ez(i+1, j+1, k)
   */
  KOKKOS_INLINE_FUNCTION void
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
  KOKKOS_INLINE_FUNCTION void
  prescale_interpolated_cbx (
    const int voxel,
    float& cbx, float& cbx_x
  ) const {

    // Set dbxdx = 0;
    cbx_x = cbx = 0.5f*(cbx + cbx_x);

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param cby cBy(i, j, k)
   * @param cby_y cBy(i, j+1, k)
   */
  KOKKOS_INLINE_FUNCTION void
  prescale_interpolated_cby (
    const int voxel,
    float& cby, float& cby_y
  ) const {

    // Set dbydy = 0;
    cby_y = cby = 0.5f*(cby + cby_y);

  }

  /**
   * @brief Apply a prescaling to the interpolating fields.
   * @param voxel Voxel index
   * @param cbz cBz(i, j, k)
   * @param cbz_z cBz(i, j, k+1)
   */
  KOKKOS_INLINE_FUNCTION void
  prescale_interpolated_cbz (
    const int voxel,
    float& cbz, float& cbz_z
  ) const {

    // Set dbzdz = 0;
    cbz_z = cbz = 0.5f*(cbz + cbz_z);

  }

  /**
   * @brief Apply a postscaling to the interpolated fields.
   * @param fields A set of field vectors
   * @param voxel Voxel index
   * @param dx0 x-coordinate, logical
   * @param dy0 y-coordinate, logical
   * @param dz0 z-coordinate, logical
   */
  KOKKOS_INLINE_FUNCTION void
  postscale_interpolated_fields (
    field_vectors_t& fields, int voxel, float dx0, float dy0, float dz0
  ) const {

    fields.ey /= mesh(voxel, mesh_var::x) + 0.5f*dx*dx0;

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
  KOKKOS_INLINE_FUNCTION void
  realign_cartesian_vector (
    const int voxel,
    const float  dx0,   const float  dy0,   const float  dz0,
    const float  dispx, const float  dispy, const float  dispz,
    /**/  float& Ax,    /**/  float& Ay,    /**/  float& Az
  ) const {

    const T Ax_T = Ax;
    const T Ay_T = Ay;
    const T x = mesh(voxel, mesh_var::x) + T(0.5)*dx*dx0 + dispx;
    const T y = dispy;

    // Using std::sqrt will resovle types correctly.
    const T scale = T(1.0) / std::sqrt(x*x + y*y);

    Ax = scale*( Ax_T*x + Ay_T*y);
    Ay = scale*(-Ax_T*y + Ay_T*x);

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
  KOKKOS_INLINE_FUNCTION void
  displacement_to_half_logical (
    const int voxel,
    const float  dx0,   const float  dy0,   const float  dz0,
    /**/  float& dispx, /**/  float& dispy, /**/  float& dispz
  ) const {

    const float r0 = mesh(voxel, mesh_var::x) + 0.5f*dx*dx0;
    const float x1 = r0 + dispx;
    const float r1 = sqrtf(x1*x1 + dispy*dispy);
    const float dtheta = atan2f(dispy, x1);

    dispx  = rdx*(2*r0*dispx + dispx*dispx + dispy*dispy)/(r0 + r1);
    dispy  = rdy*dtheta;
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
  KOKKOS_INLINE_FUNCTION void
  age_to_boundary(
    const int voxel,
    const float dx0,   const float dy0,   const float dz0,
    const float dispx, const float dispy, const float dispz,
    /**/  int&  axis,  /**/  int&  dir,   /**/  float& age
  ) const {

    float delta, deter;

    const float r0  = mesh(voxel, mesh_var::x) + 0.5f*dx0*dx;
    const float rdr = fabs(dispx*r0);
    const float dr  = dispx*dispx + dispy*dispy;

    const int ydir = (dispy>0) ? 1 : -1;
    const int zdir = (dispz>0) ? 1 : -1;

    float xage1 = 3.4e38;
    float xage2 = 3.4e38;
    float yage  = 3.4e38;
    float zage  = 3.4e38;

    // Calculate z age.
    if( dispz != 0 )
      zage = dz*(zdir - dz0)/dispz;

    // Calculate y age.
    delta = 0.5f*(ydir - dy0)*dy;
    delta = fabs(tanf(delta));

    if( ydir == -1 )
      delta = -delta;

    if( dispy != 0 && dispy - dispx*delta != 0 )
      yage = 2.0f*r0*delta/(dispy - dispx*delta);

    if( yage < 0 )
      yage = 3.4e38;


    // Calculate x age.

    // Check R+
    if( dr != 0 )
    {

      delta = 0.5f*(1.0f-dx0)*dx;
      delta = (2.0f*r0 + delta)*delta;
      deter = rdr*rdr + dr*delta;

      if( deter >= 0 )
      {
        deter = sqrtf(deter);
        deter = rdr + deter;
        xage1 = 2.0f*(dispx > 0. ? delta/deter : deter/dr);
      }

    }

    // Check R-
    if( dr != 0 && dispx < 0 )
    {

      delta = -0.5f*(1.0f+dx0)*dx;
      delta = fminf((2.0f*r0 + delta)*delta, 0);
      deter = rdr*rdr + dr*delta;

      if ( deter >= 0 )
      {
        deter = sqrtf(deter);
        deter = rdr - deter;
        xage2 = 2.0f*delta/deter;
      }

    }

    age = 2; axis = 3;
    if( zage  < age ) { age = zage;  axis = 2; dir = zdir; }
    if( yage  < age ) { age = yage;  axis = 1; dir = ydir; }
    if( xage1 < age ) { age = xage1; axis = 0; dir =  1;   }
    if( xage2 < age ) { age = xage2; axis = 0; dir = -1;   }
    age *= 0.5;

  }

};

#endif
