#ifndef _hydro_array_h_
#define _hydro_array_h_

#include "../../vpic/kokkos_helpers.h"
#include "../../util/weighting/trilinear.h"


template<typename geo_type>
class HydroAccumulator {
public:

  /**
   * @brief Functor to handle hydro moment accumlation.
   */
  HydroAccumulator(
    const grid_t * g,
    geo_type geo,
    k_hydro_sa_t hydro
  )
  : nx(g->nx), ny(g->ny), nz(g->nz),
    sx(g->sx), sy(g->sy), sz(g->sz),
    cvac(g->cvac), geo(geo), hydro(hydro) {

  }

  /**
   * @brief Accumulate the hydro moments from a particle.
   */
  void KOKKOS_INLINE_FUNCTION
  operator() (
    const int voxel,
    const float w,   const float q,  const float m,
    const float dx,  const float dy, const float dz,
    const float ux,  const float uy, const float uz
  ) const {
    auto hydro_access = hydro.access();
    float ke = ux*ux + uy*uy + uz*uz;
    float c_gamma = sqrtf(1 + ke);

    ke *= cvac/(1+c_gamma);
    c_gamma = cvac/c_gamma;

    float rdV  = geo.inverse_voxel_volume(voxel);
    float rhoq = q*w*rdV;
    float rhom = m*w*rdV;

    float vx   = ux*c_gamma;
    float vy   = uy*c_gamma;
    float vz   = uz*c_gamma;

    float px   = rhom*ux;
    float py   = rhom*uy;
    float pz   = rhom*uz;

    TrilinearWeighting weighter(nx, ny, nz, sx, sy, sz);
    weighter.set_position(dx, dy, dz);

    weighter.deposit(hydro_access, voxel, hydro_var::jx,  rhoq*vx);
    weighter.deposit(hydro_access, voxel, hydro_var::jy,  rhoq*vy);
    weighter.deposit(hydro_access, voxel, hydro_var::jz,  rhoq*vz);
    weighter.deposit(hydro_access, voxel, hydro_var::rho, rhoq);

    weighter.deposit(hydro_access, voxel, hydro_var::px,  px);
    weighter.deposit(hydro_access, voxel, hydro_var::py,  py);
    weighter.deposit(hydro_access, voxel, hydro_var::pz,  pz);
    weighter.deposit(hydro_access, voxel, hydro_var::ke,  ke);

    weighter.deposit(hydro_access, voxel, hydro_var::txx, px*vx);
    weighter.deposit(hydro_access, voxel, hydro_var::tyy, py*vy);
    weighter.deposit(hydro_access, voxel, hydro_var::tzz, pz*vz);
    weighter.deposit(hydro_access, voxel, hydro_var::tyz, py*vz);
    weighter.deposit(hydro_access, voxel, hydro_var::tzx, pz*vx);
    weighter.deposit(hydro_access, voxel, hydro_var::txy, px*vy);

  }

  const int nx, ny, nz;
  const int sx, sy, sz;
  const float cvac;
  k_hydro_sa_t hydro;
  geo_type geo;

};

/*****************************************************************************/

// Hydro arrays shall be a (nx+2) x (ny+2) x (nz+2) allocation indexed
// FORTRAN style from (0:nx+1,0:ny+1,0:nz+1).  Hydros for voxels on
// the surface of the local domain (for example h(0,:,:) or
// h(nx+1,:,:)) are not used.

typedef struct hydro {
  float jx, jy, jz, rho; // Current and charge density => <q v_i f>, <q f>
  float px, py, pz, ke;  // Momentum and K.E. density  => <p_i f>, <m c^2 (gamma-1) f>
  float txx, tyy, tzz;   // Stress diagonal            => <p_i v_j f>, i==j
  float tyz, tzx, txy;   // Stress off-diagonal        => <p_i v_j f>, i!=j
  float _pad[2];         // 16-byte align
} hydro_t;

class hydro_array_t {
public:

  hydro_t * ALIGNED(128) h;
  grid_t * g;

  k_hydro_t k_h_d;
  k_hydro_t::HostMirror k_h_h;
  k_hydro_sa_t k_h_sa;

  hydro_array_t(int nv)
  {
    init_kokkos_hydro(nv);
  }

  void init_kokkos_hydro(int nv)
  {
    k_h_d = k_hydro_t("k_hydro", nv);
    k_h_sa = Kokkos::Experimental::create_scatter_view(k_h_d);
    k_h_h  = Kokkos::create_mirror_view(k_h_d);
  }

  /**
   * @brief Get a hydro accumulator object to use on the device.
   */
  template<Geometry geo>
  const HydroAccumulator<typename GeometryClass<geo>::device>
   get_device_accumulator() const {

    // TODO: This is so ugly.
    return HydroAccumulator<typename GeometryClass<geo>::device>
      (g, g->get_device_geometry<geo>(), k_h_sa);

  }

  /**
   * @brief Synchronize the hydro array with local boundary conditions and
   * copies it to the host. Device-side is left unsynchronized.
   */
  void synchronize();

  /**
   * @brief Clears the hydro array.
   */
  void clear();

  /**
   * @brief Copies the hydro data to the host.
   */
  void copy_to_host();

  /**
   * @brief Copies the hydro data to the device.
   */
  void copy_to_device();

};

// In hydro_array.c

// Construct a hydro array suitable for the grid

hydro_array_t *
new_hydro_array( grid_t * g );

// Destruct a hydro array

void
delete_hydro_array( hydro_array_t * ha );

// Zero out the hydro array.  Use before accumulating species to
// a hydro array.

void
clear_hydro_array( hydro_array_t * ha );


#endif
