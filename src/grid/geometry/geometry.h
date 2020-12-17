#ifndef _geometry_h_
#define _geometry_h_

#include <Kokkos_Core.hpp>

// Available geometries
enum class Geometry {
  Cartesian,               // Cartesian coordinates,   (x, y, z)
  Cylindrical,             // Cylindrical coordinates, (r, theta, z)
};

// TODO: Clean up placement, this doesn't really belong here.
typedef struct field_vectors {
    float ex,  ey,  ez;
    float cbx, cby, cbz;
} field_vectors_t;


#include "cartesian.h"
#include "cylindrical.h"

// TODO: Can we lift this out of an ugly macro and template mess and still get
// compile-time optimization? I think we need to keep geometry and mesh_type
// as template parameters to do so, but then to specialize geometry we also
// have to specialize host/device which leads to lots of code duplication. This
// version avoids this issue.
//
// Alternatively, C++14 would help things a lot since we could decltype(auto).

template<Geometry geo> struct GeometryClass{ };

template<> struct GeometryClass<Geometry::Cartesian> {
  typedef CartesianGeometry<k_mesh_t> device;
  typedef CartesianGeometry<k_mesh_t::HostMirror> host;
};

template<> struct GeometryClass<Geometry::Cylindrical> {
  typedef CylindricalGeometry<k_mesh_t> device;
  typedef CylindricalGeometry<k_mesh_t::HostMirror> host;
};


// Unsafe macro to help conditionally select geometries.
#define SELECT_GEOMETRY(VAR, GEONAME, BLOCK)            \
switch(VAR) {                                           \
  case Geometry::Cartesian : {                          \
    constexpr Geometry GEONAME = Geometry::Cartesian;   \
    BLOCK;                                              \
    break;                                              \
  }                                                     \
  case Geometry::Cylindrical : {                        \
    constexpr Geometry GEONAME = Geometry::Cylindrical; \
    BLOCK;                                              \
    break;                                              \
  }                                                     \
  default:                                              \
    ERROR(("Unknown geometry"));                        \
    break;                                              \
}

#endif
