#ifndef _accumulator_h_
#define _accumulator_h_

#include "../../grid/grid.h"
#include "../../field_advance/field_advance.h"

template<typename view_type>
class Accumulator {
public:

  /**
   * @brief Functor to handle particle current accumlation.
   */
  Accumulator(view_type current) : current(current) { }

  /**
   * @brief Accumulate the current from a particle.
   * Accumulator values are four times the total physical charge that passed
   * through the appropriate current quadrant in a time-step.
   */
  void KOKKOS_INLINE_FUNCTION
  operator() (
    const int voxel, const float q,
    const float dx,  const float dy, const float dz,
    const float ux,  const float uy, const float uz
  ) const {
    accumulate(current, voxel, q, dx, dy, dz, ux, uy, uz);
  }


private:

  /**
   * @brief Accumulate the current from a particle.
   */
  template<typename current_type>
  void KOKKOS_INLINE_FUNCTION
  accumulate(
    current_type& current_access,
    const int voxel, const float q,
    const float dx,  const float dy, const float dz,
    const float ux,  const float uy, const float uz
  ) const {

    float v0, v1, v2, v3, v4, v5;

    // Compute correction
    v5 = q*ux*uy*uz*(1.f/3.f);

    #define ACCUMULATE_J(X,Y,Z)                                         \
      v4  = q*u##X;   /* v2 = q ux                            */        \
      v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
      v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
      v1 += v4;       /* v1 = q ux (1+dy)                     */        \
      v4  = 1.f+d##Z; /* v4 = 1+dz                            */        \
      v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
      v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
      v4  = 1.f-d##Z; /* v4 = 1-dz                            */        \
      v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
      v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
      v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
      v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
      v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
      v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

    ACCUMULATE_J( x,y,z );
    current_access(voxel, accumulator_var::jx, 0) += v0;
    current_access(voxel, accumulator_var::jx, 1) += v1;
    current_access(voxel, accumulator_var::jx, 2) += v2;
    current_access(voxel, accumulator_var::jx, 3) += v3;

    ACCUMULATE_J( y,z,x );
    current_access(voxel, accumulator_var::jy, 0) += v0;
    current_access(voxel, accumulator_var::jy, 1) += v1;
    current_access(voxel, accumulator_var::jy, 2) += v2;
    current_access(voxel, accumulator_var::jy, 3) += v3;

    ACCUMULATE_J( z,x,y );
    current_access(voxel, accumulator_var::jz, 0) += v0;
    current_access(voxel, accumulator_var::jz, 1) += v1;
    current_access(voxel, accumulator_var::jz, 2) += v2;
    current_access(voxel, accumulator_var::jz, 3) += v3;

    #undef ACCUMULATE_J

  }

  view_type current;

};

// Specialization for device scatter access
template<>
void KOKKOS_INLINE_FUNCTION
Accumulator<k_accumulators_sa_t>::operator() (
  const int voxel, const float q,
  const float dx,  const float dy, const float dz,
  const float ux,  const float uy, const float uz
) const {
  auto access = current.access();
  accumulate(access, voxel, q, dx, dy, dz, ux, uy, uz);
}


/*****************************************************************************/

// Accumulator arrays shall be a
//   POW2_CEIL((nx+2)x(ny+2)x(nz+2),2)x(1+n_pipeline)
// allocation indexed FORTRAN style.  That is, the accumulator array
// is a 4d array.  a(:,:,:,0) is the accumulator used by the host
// processor.  a(:,:,:,1:n_pipeline) are the accumulators used by
// pipelines during operations.  Like the interpolator, accumualtors
// on the surface of the local domain are not used.

typedef struct accumulator {
  float jx[4];   // jx0@(0,-1,-1),jx1@(0,1,-1),jx2@(0,-1,1),jx3@(0,1,1)
  float jy[4];   // jy0@(-1,0,-1),jy1@(-1,0,1),jy2@(1,0,-1),jy3@(1,0,1)
  float jz[4];   // jz0@(-1,-1,0),jz1@(1,-1,0),jz2@(-1,1,0),jz3@(1,1,0)
} accumulator_t;

class accumulator_array_t {
public:

  // TODO: Deprecate accumulator_t in favor of k_a_h. Currently it is untouched.
  accumulator_t * ALIGNED(128) a;
  int n_pipeline; // Number of pipelines supported by this accumulator
  int stride;     // Stride be each pipeline's accumulator array
  int na;         // Number of accumulators in a
  grid_t * g;

  k_accumulators_t k_a_d;
  k_accumulators_t::HostMirror k_a_h;
  k_accumulators_sa_t k_a_sa;
  //k_accumulators_sah_t k_a_sah;
  k_accumulators_t k_a_d_copy;

  accumulator_array_t(int _na)
  {
      init_kokoks_accum(_na);
  }

  void init_kokoks_accum(int _na)
  {
      na = _na;

      k_a_d = k_accumulators_t("k_accumulators", _na);
      k_a_d_copy = k_accumulators_t("k_accumulators_copy", _na);
      k_a_sa = Kokkos::Experimental::create_scatter_view(k_a_d);
      k_a_h  = Kokkos::create_mirror_view(k_a_d);
  }


  /**
   * @brief Get an Accumulator object to use on the device.
   */
  Accumulator<k_accumulators_sa_t>
  get_device_accumulator() {
    return Accumulator<k_accumulators_sa_t>(k_a_sa);
  }

  /**
   * @brief Get an Accumulator object to use on the host. No scatter access.
   */
  Accumulator<k_accumulators_t::HostMirror>
  get_host_accumulator() {
    return Accumulator<k_accumulators_t::HostMirror>(k_a_h);
  }

  /**
   * @brief Combine scatter access contributions on the device.
   */
  void contribute();

  /**
   * @brief Send host contributions to the device.
   */
  void combine();

  /**
   * @brief Unload currents into the field array.
   */
  void unload( field_array_t * RESTRICT fa );

  /**
   * @brief Clears and resets the current accumulators.
   */
  void clear();

};

// In sf_structors.c

accumulator_array_t *
new_accumulator_array( grid_t * g );

void
delete_accumulator_array( accumulator_array_t * a );

#endif
