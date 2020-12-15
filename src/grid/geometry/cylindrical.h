#ifndef _cylindrical_geometry_h_
#define _cylindrical_geometry_h_

// Cylindrical implementation ====================================================

template<> const float KOKKOS_INLINE_FUNCTION
GeometricOperator<Geometry::Cylindrical, class mesh_view_t>::
inverse_voxel_volume (
  int voxel
) const {

  return rdV/mesh(voxel, mesh_var::x);

}

template<> const float KOKKOS_INLINE_FUNCTION
GeometricOperator<Geometry::Cylindrical, class mesh_view_t>::
scale_interpolated_fields (
  field_vector_t& field,
  int voxel, float  dx0, float  dy0, float  dz0
) const {

  field.ey /= mesh(voxel, mesh_var::x) + 0.5f*dx*dx0;

}

template<> const float KOKKOS_INLINE_FUNCTION
GeometricOperator<Geometry::Cylindrical, class mesh_view_t>::
transform_to_logical (
    int voxel,
    float  dx0,   float  dy0,   float  dz0,
    float& dispx, float& dispy, float& dispz,
    float& ux,    float& uy,    float& uz
) const {

  const float r0 = mesh(voxel, mesh_var::x) + 0.5f*dx*dx0
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

template<> const float KOKKOS_INLINE_FUNCTION
GeometricOperator<Geometry::Cylindrical, class mesh_view_t>::
scale_particle_momentum (
    int voxel,
    float  dx0, float  dy0, float  dz0,
    float& ux,  float& uy,  float& uz
) const {

  field.ey /= mesh(voxel, mesh_var::x) + 0.5f*dx*dx0;

}

#endif
