#ifndef _interpolator_h_
#define _interpolator_h_

#include "../../grid/grid.h"
#include "../../field_advance/field_advance.h"

template<typename geo_type>
class Interpolator {
public:

  /**
   * @brief Functor to handle field interpolation.
   */
  Interpolator(
    const grid_t * g,
    geo_type geometry,
    k_interpolator_t interp
  )
  : geometry(geometry), interp(interp)
  {

  }

  /**
   * @brief Interpolate the field at a point.
   */
  const field_vectors_t
  KOKKOS_INLINE_FUNCTION
  operator() (
    int voxel,
    float dx, float dy, float dz
  ) const {

    using namespace interpolator_var;

    field_vectors_t result;

    result.ex  =    ( interp(voxel, ex)    + dy*interp(voxel, dexdy)    ) +
                 dz*( interp(voxel, dexdz) + dy*interp(voxel, d2exdydz) );
    result.ey =     ( interp(voxel, ey)    + dz*interp(voxel, deydz)    ) +
                 dx*( interp(voxel, deydx) + dz*interp(voxel, d2eydzdx) );
    result.ez =     ( interp(voxel, ez)    + dx*interp(voxel, dezdx)    ) +
                 dy*( interp(voxel, dezdy) + dx*interp(voxel, d2ezdxdy) );

    result.cbx = interp(voxel, cbx) + dx*interp(voxel, dcbxdx);
    result.cby = interp(voxel, cby) + dy*interp(voxel, dcbydy);
    result.cbz = interp(voxel, cbz) + dz*interp(voxel, dcbzdz);

    geometry.postscale_interpolated_fields(result, voxel, dx, dy, dz);

    return result;

  }


private:
  const geo_type geometry;
  k_interpolator_t interp;

};

/*****************************************************************************/

// Interpolator arrays shall be a (nx+2) x (ny+2) x (nz+2) allocation
// indexed FORTRAN style from (0:nx+1,0:ny+1,0:nz+1). Interpolators
// for voxels on the surface of the local domain (for example
// fi(0,:,:) or fi(nx+1,:,:)) are not used.

typedef struct interpolator {
  float ex, dexdy, dexdz, d2exdydz;
  float ey, deydz, deydx, d2eydzdx;
  float ez, dezdx, dezdy, d2ezdxdy;
  float cbx, dcbxdx;
  float cby, dcbydy;
  float cbz, dcbzdz;
  float _pad[2];  // 16-byte align
} interpolator_t;

class interpolator_array_t {
public:

  interpolator_t * ALIGNED(128) i;
  grid_t * g;
  k_interpolator_t k_i_d;
  k_interpolator_t::HostMirror k_i_h;

  interpolator_array_t(int nv)
  {
      init_kokkos_interp(nv);
  }

  void init_kokkos_interp(int nv)
  {
    k_i_d = k_interpolator_t("k_interpolators", nv);
    k_i_h = Kokkos::create_mirror_view(k_i_d);
  }

  /**
   * @brief Get an interpolator object to use on the device.
   */
  template<Geometry geo>
  const Interpolator<typename GeometryClass<geo>::device>
  get_device_interpolator() const {
    return Interpolator<typename GeometryClass<geo>::device>(
      g,
      g->get_device_geometry<geo>(),
      k_i_d
    );
  }

  /**
   * @brief Load the fields into the interpolator.
   * Going into load_interpolator, the field array f contains the
   * current information such that the fields can be interpolated to
   * particles within the local domain.  Load interpolate computes the
   * field array into a set of interpolation coefficients for each voxel
   * inside the local domain suitable for use by the particle update
   * functions.
   */
  void
  load( const field_array_t * RESTRICT fa );

  /**
   * @brief Copies the interpolator data to the host.
   */
  void copy_to_host();

  /**
   * @brief Copies the interpolator data to the device.
   */
  void copy_to_device();

};

// In interpolator_array.cxx

interpolator_array_t *
new_interpolator_array( grid_t * g );

void
delete_interpolator_array( interpolator_array_t * ALIGNED(128) ia );



#endif
