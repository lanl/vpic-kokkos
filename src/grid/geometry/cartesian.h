#ifndef _cartesian_geometry_h_
#define _cartesian_geometry_h_

template<class mesh_view_t>
class CartesianGeometry {
public:

  CartesianGeometry(
      const float dx,
      const float dy,
      const float dz,
      const mesh_view_t mesh
  )
  : mesh(mesh),
    dx(dx), dy(dy), dz(dz), dV(dx*dy*dz),
    rdx(1./dx), rdy(1./dy), rdz(1./dz), rdV(1./(dx*dy*dz))
  {

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

    dispx *= 2*rdx;
    dispy *= 2*rdy;
    dispz *= 2*rdz;

  }

private:

  const float dx,  dy,  dz,  dV;
  const float rdx, rdy, rdz, rdV;
  const mesh_view_t mesh;

};

#endif
