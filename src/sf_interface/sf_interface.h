#ifndef _sf_interface_h_
#define _sf_interface_h_

// FIXME: THE HOST PROCESSED FIELD KERNELS SHOULD BE UPDATED TO USE
// SCALAR FMA INSTRUCTIONS WITH COMMENSURATE ROUND-OFF PROPERTIES TO
// THE FMA INSTRUCTIONS USED ON THE PIPELINE PROCESSED FIELDS!

// FIXME: (nx>1) ? (1/dx) : 0 TYPE LOGIC SHOULD BE FIXED SO THAT NX
// REFERS TO THE GLOBAL NUMBER OF CELLS IN THE X-DIRECTION (NOT THE
// _LOCAL_ NUMBER OF CELLS).  THIS LATENT BUG IS NOT EXPECTED TO
// AFFECT ANY PRACTICAL SIMULATIONS.

#include "../field_advance/field_advance.h"
// FIXME: SHOULD INCLUDE SPECIES_ADVANCE TOO ONCE READY

/*****************************************************************************/

// Interpolator arrays shall be a (nx+2) x (ny+2) x (nz+2) allocation
// indexed FORTRAN style from (0:nx+1,0:ny+1,0:nz+1). Interpolators
// for voxels on the surface of the local domain (for example
// fi(0,:,:) or fi(nx+1,:,:)) are not used.

typedef struct interpolator {
#ifdef SHAPE_NGP
  //float ex, dexdy, dexdz, d2exdydz;
  //float ey, deydz, deydx, d2eydzdx;
  //float ez, dezdx, dezdy, d2ezdxdy;
  //float cbx, dcbxdx;
  //float cby, dcbydy;
  //float cbz, dcbzdz;
  float ex, ey, ez;
  float cbx, cby, cbz;
  #ifdef EXTERNAL_FORCE
  float Ex0, Ey0, Ez0;
  float Gx0, Gy0, Gz0;
  #else
  float _pad[2];  // 16-byte align
  #endif
#else
#ifdef SHAPE_QS
  // TODO(low-priority) TEST LAYOUT - is it better to interleave padding so ex,ey,ez;bx,by,bz
  // are cleanly spaced on 32-byte boundaries, or only pad end of struct????
  // --ATr,2024nov08
  float ex,   dexdx,  dexdy,  dexdz,  d2exdx,  d2exdy,  d2exdz;
  float ey,   deydx,  deydy,  deydz,  d2eydx,  d2eydy,  d2eydz;
  float ez,   dezdx,  dezdy,  dezdz,  d2ezdx,  d2ezdy,  d2ezdz;
  float cbx, dcbxdx, dcbxdy, dcbxdz, d2cbxdx, d2cbxdy, d2cbxdz;
  float cby, dcbydx, dcbydy, dcbydz, d2cbydx, d2cbydy, d2cbydz;
  float cbz, dcbzdx, dcbzdy, dcbzdz, d2cbzdx, d2cbzdy, d2cbzdz;
  #ifdef EXTERNAL_FORCE
  float Ex0, dEx0dx, dEx0dy, dEx0dz, d2Ex0dx, d2Ex0dy, d2Ex0dz;
  float Ey0, dEy0dx, dEy0dy, dEy0dz, d2Ey0dx, d2Ey0dy, d2Ey0dz;
  float Ez0, dEz0dx, dEz0dy, dEz0dz, d2Ez0dx, d2Ez0dy, d2Ez0dz;
  float Gx0, dGx0dx, dGx0dy, dGx0dz, d2Gx0dx, d2Gx0dy, d2Gx0dz;
  float Gy0, dGy0dx, dGy0dy, dGy0dz, d2Gy0dx, d2Gy0dy, d2Gy0dz;
  float Gz0, dGz0dx, dGz0dy, dGz0dz, d2Gz0dx, d2Gz0dy, d2Gz0dz;
  #else
  float _pad[2]; // 16-byte align
  #endif
#endif
#endif
} interpolator_t;

typedef struct interpolator_array {
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  interpolator_t * ALIGNED(128) i;
#endif
  grid_t * g;
  k_interpolator_t k_i_d;
  k_interpolator_t::HostMirror k_i_h;

  interpolator_array(int nv)
  {
      init_kokkos_interp(nv);
  }

  void init_kokkos_interp(int nv)
  {
    k_i_d = k_interpolator_t("k_interpolators", nv);
    k_i_h = Kokkos::create_mirror_view(k_i_d);
  }

  /**
   * @brief Copies the interpolator data to the host.
   */
  void copy_to_host();

  /**
   * @brief Copies the interpolator data to the device.
   */
  void copy_to_device();

} interpolator_array_t;

// In interpolator_array.cxx

interpolator_array_t *
new_interpolator_array( grid_t * g );

void
delete_interpolator_array( interpolator_array_t * ALIGNED(128) ia );

// Going into load_interpolator, the field array f contains the
// current information such that the fields can be interpolated to
// particles within the local domain.  Load interpolate computes the
// field array into a set of interpolation coefficients for each voxel
// inside the local domain suitable for use by the particle update
// functions.

void
load_interpolator_array( /**/  interpolator_array_t * RESTRICT ia,
                         const field_array_t        * RESTRICT fa );

template<typename InterpView>
KOKKOS_INLINE_FUNCTION
interpolator_t 
read_interpolator(InterpView& interp, const size_t idx) {
  return interpolator_t {
#ifdef SHAPE_NGP
    interp(idx, interpolator_var::ex),
    interp(idx, interpolator_var::ey),
    interp(idx, interpolator_var::ez),
    interp(idx, interpolator_var::cbx),
    interp(idx, interpolator_var::cby),
    interp(idx, interpolator_var::cbz),
  #ifdef EXTERNAL_FORCE
    interp(idx, interpolator_var::Ex0),
    interp(idx, interpolator_var::Ey0),
    interp(idx, interpolator_var::Ez0),
    interp(idx, interpolator_var::Gx0),
    interp(idx, interpolator_var::Gy0),
    interp(idx, interpolator_var::Gz0),
  #endif
#elif defined( SHAPE_QS )
    interp(idx, interpolator_var::ex),
    interp(idx, interpolator_var::dexdx),
    interp(idx, interpolator_var::dexdy),
    interp(idx, interpolator_var::dexdz),
    interp(idx, interpolator_var::d2exdx),
    interp(idx, interpolator_var::d2exdy),
    interp(idx, interpolator_var::d2exdz),
    interp(idx, interpolator_var::ey),
    interp(idx, interpolator_var::deydx),
    interp(idx, interpolator_var::deydy),
    interp(idx, interpolator_var::deydz),
    interp(idx, interpolator_var::d2eydx),
    interp(idx, interpolator_var::d2eydy),
    interp(idx, interpolator_var::d2eydz),
    interp(idx, interpolator_var::ez),
    interp(idx, interpolator_var::dezdx),
    interp(idx, interpolator_var::dezdy),
    interp(idx, interpolator_var::dezdz),
    interp(idx, interpolator_var::d2ezdx),
    interp(idx, interpolator_var::d2ezdy),
    interp(idx, interpolator_var::d2ezdz),
    interp(idx, interpolator_var::cbx),
    interp(idx, interpolator_var::dcbxdx),
    interp(idx, interpolator_var::dcbxdy),
    interp(idx, interpolator_var::dcbxdz),
    interp(idx, interpolator_var::d2cbxdx),
    interp(idx, interpolator_var::d2cbxdy),
    interp(idx, interpolator_var::d2cbxdz),
    interp(idx, interpolator_var::cby),
    interp(idx, interpolator_var::dcbydx),
    interp(idx, interpolator_var::dcbydy),
    interp(idx, interpolator_var::dcbydz),
    interp(idx, interpolator_var::d2cbydx),
    interp(idx, interpolator_var::d2cbydy),
    interp(idx, interpolator_var::d2cbydz),
    interp(idx, interpolator_var::cbz),
    interp(idx, interpolator_var::dcbzdx),
    interp(idx, interpolator_var::dcbzdy),
    interp(idx, interpolator_var::dcbzdz),
    interp(idx, interpolator_var::d2cbzdx),
    interp(idx, interpolator_var::d2cbzdy),
    interp(idx, interpolator_var::d2cbzdz),
  #ifdef EXTERNAL_FORCE
    interp(idx, interpolator_var::Ex0),
    interp(idx, interpolator_var::dEx0dx),
    interp(idx, interpolator_var::dEx0dy),
    interp(idx, interpolator_var::dEx0dz),
    interp(idx, interpolator_var::d2Ex0dx),
    interp(idx, interpolator_var::d2Ex0dy),
    interp(idx, interpolator_var::d2Ex0dz),
    interp(idx, interpolator_var::Ey0),
    interp(idx, interpolator_var::dEy0dx),
    interp(idx, interpolator_var::dEy0dy),
    interp(idx, interpolator_var::dEy0dz),
    interp(idx, interpolator_var::d2Ey0dx),
    interp(idx, interpolator_var::d2Ey0dy),
    interp(idx, interpolator_var::d2Ey0dz),
    interp(idx, interpolator_var::Ez0),
    interp(idx, interpolator_var::dEz0dx),
    interp(idx, interpolator_var::dEz0dy),
    interp(idx, interpolator_var::dEz0dz),
    interp(idx, interpolator_var::d2Ez0dx),
    interp(idx, interpolator_var::d2Ez0dy),
    interp(idx, interpolator_var::d2Ez0dz),
    interp(idx, interpolator_var::Gx0),
    interp(idx, interpolator_var::dGx0dx),
    interp(idx, interpolator_var::dGx0dy),
    interp(idx, interpolator_var::dGx0dz),
    interp(idx, interpolator_var::d2Gx0dx),
    interp(idx, interpolator_var::d2Gx0dy),
    interp(idx, interpolator_var::d2Gx0dz),
    interp(idx, interpolator_var::Gy0),
    interp(idx, interpolator_var::dGy0dx),
    interp(idx, interpolator_var::dGy0dy),
    interp(idx, interpolator_var::dGy0dz),
    interp(idx, interpolator_var::d2Gy0dx),
    interp(idx, interpolator_var::d2Gy0dy),
    interp(idx, interpolator_var::d2Gy0dz),
    interp(idx, interpolator_var::Gz0),
    interp(idx, interpolator_var::dGz0dx),
    interp(idx, interpolator_var::dGz0dy),
    interp(idx, interpolator_var::dGz0dz),
    interp(idx, interpolator_var::d2Gz0dx),
    interp(idx, interpolator_var::d2Gz0dy),
    interp(idx, interpolator_var::d2Gz0dz),
  #endif
#endif
  };
}

KOKKOS_INLINE_FUNCTION
void 
interpolate_e( const interpolator_t& f,
               const float p_dx, const float p_dy, const float p_dz,
               float& ex, float& ey, float& ez,
               const float qdt_2mc, const float dt_2c ) {
#ifdef SHAPE_NGP
  #ifdef EXTERNAL_FORCE
    ex = qdt_2mc * (f.ex + f.Ex0) + dt_2c * f.Gx0;
    ey = qdt_2mc * (f.ey + f.Ey0) + dt_2c * f.Gy0;
    ez = qdt_2mc * (f.ez + f.Ez0) + dt_2c * f.Gz0;
  #else
    ex = qdt_2mc * f.ex;
    ey = qdt_2mc * f.ey;
    ez = qdt_2mc * f.ez;
  #endif
#elif defined( SHAPE_QS )
  #ifdef EXTERNAL_FORCE
    ex = qdt_2mc*( f.ex + p_dx*( f.dexdx  + p_dx*f.d2exdx )
                        + p_dy*( f.dexdy  + p_dy*f.d2exdy )
                        + p_dz*( f.dexdz  + p_dz*f.d2exdz )
                + f.Ex0 + p_dx*( f.dEx0dx + p_dx*f.d2Ex0dx )
                        + p_dy*( f.dEx0dy + p_dy*f.d2Ex0dy )
                        + p_dz*( f.dEx0dz + p_dz*f.d2Ex0dz ) );
    ey = qdt_2mc*( f.ey + p_dx*( f.deydx  + p_dx*f.d2eydx )
                        + p_dy*( f.deydy  + p_dy*f.d2eydy )
                        + p_dz*( f.deydz  + p_dz*f.d2eydz )
                + f.Ey0 + p_dx*( f.dEy0dx + p_dx*f.d2Ey0dx )
                        + p_dy*( f.dEy0dy + p_dy*f.d2Ey0dy )
                        + p_dz*( f.dEy0dz + p_dz*f.d2Ey0dz ) );
    ez = qdt_2mc*( f.ez + p_dx*( f.dezdx  + p_dx*f.d2ezdx )
                        + p_dy*( f.dezdy  + p_dy*f.d2ezdy )
                        + p_dz*( f.dezdz  + p_dz*f.d2ezdz )
                + f.Ez0 + p_dx*( f.dEz0dx + p_dx*f.d2Ez0dx )
                        + p_dy*( f.dEz0dy + p_dy*f.d2Ez0dy )
                        + p_dz*( f.dEz0dz + p_dz*f.d2Ez0dz ) );
    ex += dt_2c *( f.Gx0 + p_dx*( f.dGx0dx + p_dx*f.d2Gx0dx )
                         + p_dy*( f.dGx0dy + p_dy*f.d2Gx0dy )
                         + p_dz*( f.dGx0dz + p_dz*f.d2Gx0dz ) );
    ey += dt_2c *( f.Gy0 + p_dx*( f.dGy0dx + p_dx*f.d2Gy0dx )
                         + p_dy*( f.dGy0dy + p_dy*f.d2Gy0dy )
                         + p_dz*( f.dGy0dz + p_dz*f.d2Gy0dz ) );
    ez += dt_2c *( f.Gz0 + p_dx*( f.dGz0dx + p_dx*f.d2Gz0dx )
                         + p_dy*( f.dGz0dy + p_dy*f.d2Gz0dy )
                         + p_dz*( f.dGz0dz + p_dz*f.d2Gz0dz ) );
  #else
    ex = qdt_2mc*( f.ex + p_dx*( f.dexdx + p_dx*f.d2exdx )
                        + p_dy*( f.dexdy + p_dy*f.d2exdy )
                        + p_dz*( f.dexdz + p_dz*f.d2exdz ) );
    ey = qdt_2mc*( f.ey + p_dx*( f.deydx + p_dx*f.d2eydx )
                        + p_dy*( f.deydy + p_dy*f.d2eydy )
                        + p_dz*( f.deydz + p_dz*f.d2eydz ) );
    ez = qdt_2mc*( f.ez + p_dx*( f.dezdx + p_dx*f.d2ezdx )
                        + p_dy*( f.dezdy + p_dy*f.d2ezdy )
                        + p_dz*( f.dezdz + p_dz*f.d2ezdz ) );
  #endif
#endif
}

KOKKOS_INLINE_FUNCTION
void 
interpolate_b( const interpolator_t& f,  
               const float p_dx, const float p_dy, const float p_dz,
               float& bx, float& by, float& bz ) {
#ifdef SHAPE_NGP
  bx = f.cbx;
  by = f.cby;
  bz = f.cbz;
#elif defined( SHAPE_QS )
  bx = f.cbx + p_dx*( f.dcbxdx + p_dx*f.d2cbxdx )
             + p_dy*( f.dcbxdy + p_dy*f.d2cbxdy )
             + p_dz*( f.dcbxdz + p_dz*f.d2cbxdz );
  by = f.cby + p_dx*( f.dcbydx + p_dx*f.d2cbydx )
             + p_dy*( f.dcbydy + p_dy*f.d2cbydy )
             + p_dz*( f.dcbydz + p_dz*f.d2cbydz );
  bz = f.cbz + p_dx*( f.dcbzdx + p_dx*f.d2cbzdx )
             + p_dy*( f.dcbzdy + p_dy*f.d2cbzdy )
             + p_dz*( f.dcbzdz + p_dz*f.d2cbzdz );
#endif
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
  float rho[4];   // jz0@(-1,-1,0),jz1@(1,-1,0),jz2@(-1,1,0),jz3@(1,1,0)

} accumulator_t;

typedef struct accumulator_array {
  accumulator_t * ALIGNED(128) a;
  int n_pipeline; // Number of pipelines supported by this accumulator
  int stride;     // Stride be each pipeline's accumulator array
  int na;         // Number of accumulators in a
  grid_t * g;

  k_accumulators_t k_a_d;
  k_accumulators_t::HostMirror k_a_h;
  k_accumulators_sv_t k_a_sv;
  //k_accumulators_sah_t k_a_sah;
  k_accumulators_t k_a_d_copy;

  accumulator_array(int _na)
  {
    init_kokoks_accum(_na);
  }

  void init_kokoks_accum(int _na)
  {
    na = _na;

    k_a_d = k_accumulators_t("k_accumulators", _na);
    k_a_d_copy = k_accumulators_t("k_accumulators_copy", _na);
    k_a_sv = Kokkos::Experimental::create_scatter_view(k_a_d);
    k_a_h  = Kokkos::create_mirror_view(k_a_d);
  }

  /**
   * @brief Copies the accumulator data to the host.
   */
  void copy_to_host();

  /**
   * @brief Copies the accumulator data to the device.
   */
  void copy_to_device();

} accumulator_array_t;

// In sf_structors.c

accumulator_array_t *
new_accumulator_array( grid_t * g );

void
delete_accumulator_array( accumulator_array_t * a );

// In clear_accumulators.c

// This zeros out all the accumulator arrays in a pipelined fashion.

void
clear_accumulator_array( accumulator_array_t * RESTRICT a );

void
clear_accumulator_array_kokkos( accumulator_array_t * RESTRICT a );

// In reduce_accumulators.c

// Going into reduce_accumulators, the host cores and the pipeline
// cores have each accumulated values to their personal
// accumulators.  This reduces the pipeline accumulators into the host
// accumulator with a pipelined horizontal reduction (a deterministic
// reduction).

void
reduce_accumulator_array( accumulator_array_t * RESTRICT a );

void
reduce_accumulator_array_kokkos( accumulator_array_t * RESTRICT a );

// In unload_accumulator.c

// Going into unload_accumulator, the accumulator contains 4 times the
// net amount of charge that crossed the quarter face associated with
// each accumulator component (has units of physical charge, i.e. C)
// computed by the advance_p functions.  unload_accumulator computes
// the physical current density (A/m^2 in MKS units) associated with
// all local quarter faces and accumulates the local quarter faces to
// local field array jf.  unload_accumulator assumes all the pipeline
// accumulators have been reduced into the host accumulator.

void
unload_accumulator_array( /**/  field_array_t       * RESTRICT fa,
                          const accumulator_array_t * RESTRICT aa );
void
unload_accumulator_array_kokkos( /**/  field_array_t       * RESTRICT fa,
                          const accumulator_array_t * RESTRICT aa );

void
combine_accumulators( accumulator_array_t * RESTRICT aa );

/*****************************************************************************/

// Hydro arrays shall be a (nx+2) x (ny+2) x (nz+2) allocation indexed
// FORTRAN style from (0:nx+1,0:ny+1,0:nz+1).  Hydros for voxels on
// the surface of the local domain (for example h(0,:,:) or
// h(nx+1,:,:)) are not used.
// The kokkos hydro array is easily accessed as hydro_array->k_h_d(index,
// hydro_var::var), with var being any member of a hydro_t.

struct hydro_t {
  float jx, jy, jz, rho; // Current and charge density => <q v_i f>, <q f>
  float px, py, pz, rho_m; // Momentum and mass density (changed from ke_density)
  float txx, tyy, tzz;   // Stress diagonal            => <p_i v_j f>, i==j
  float tyz, tzx, txy;   // Stress off-diagonal        => <p_i v_j f>, i!=j
#if VARIABLE_CHARGE
  float qmin, qmax;      // Minimum and maximum charge within a cell
  float n_q0, n_q1, n_q2, n_q3, n_q4, n_q5;
  float _pad[2];
#else
  float _pad[2];         // 16-byte align
#endif
};

struct hydro_array_t {
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  hydro_t * ALIGNED(128) h;
#endif
  k_hydro_t k_h_d;
  k_hydro_t::HostMirror k_h_h;
  grid_t * g;
  
  hydro_array_t(int nv)
  {
    k_h_d = k_hydro_t("k_hydro", nv);
    k_h_h = Kokkos::create_mirror_view(k_h_d);
  }

  /**
    * @brief Copies the hydro data to host legacy array
    */
  void copy_to_host(FILE *fp=nullptr,  const int step = 0);

  /**
    * @brief Copies the hydro data to device from the host legacy array 
    */
  void copy_to_device(FILE *fp=nullptr,  const int step = 0);

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

// Synchronize the hydro array with local boundary conditions and
// neighboring processes.  Use after all species have been
// accumulated to the hydro array.

void
synchronize_hydro_array( hydro_array_t * ha );

void
synchronize_hydro_array_kokkos( hydro_array_t * ha );

#endif // _sf_interface_h_
