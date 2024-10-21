#ifndef _fluid_advance_h_
#define _fluid_advance_h_

#include "../grid/grid.h"
#include "../vpic/kokkos_helpers.h"

typedef struct fluid {
  float den,   tmp,   prs;     // density, temperature, pressure
  float vx,  vy,  vz;          // velocity
} field_t;

// fluid_advance_kernels holds all the function pointers to all the
// kernels used by a specific fluid_advance instance.


struct fluid_array;

typedef struct fluid_advance_kernels {

  void (*delete_fl)( struct fluid_array * RESTRICT fl );

  // Time stepping interface
  void (*advance_fl)( struct fluid_array * RESTRICT fl );
  
  // Diagnostic interface
  void (*energy_fl)( /**/  double        * RESTRICT en, // 6 elem
                    const struct field_array * RESTRICT fa );

} fluid_advance_kernels_t;

typedef struct field_buffers
{
    Kokkos::View<float*>   xyz_sbuf_pos;
    Kokkos::View<float*>   yzx_sbuf_pos;
    Kokkos::View<float*>   zxy_sbuf_pos;
    Kokkos::View<float*>   xyz_rbuf_pos;
    Kokkos::View<float*>   yzx_rbuf_pos;
    Kokkos::View<float*>   zxy_rbuf_pos;
    Kokkos::View<float*>   xyz_sbuf_neg;
    Kokkos::View<float*>   yzx_sbuf_neg;
    Kokkos::View<float*>   zxy_sbuf_neg;
    Kokkos::View<float*>   xyz_rbuf_neg;
    Kokkos::View<float*>   yzx_rbuf_neg;
    Kokkos::View<float*>   zxy_rbuf_neg;

    Kokkos::View<float*>::HostMirror   xyz_sbuf_pos_h;
    Kokkos::View<float*>::HostMirror   yzx_sbuf_pos_h;
    Kokkos::View<float*>::HostMirror   zxy_sbuf_pos_h;
    Kokkos::View<float*>::HostMirror   xyz_rbuf_pos_h;
    Kokkos::View<float*>::HostMirror   yzx_rbuf_pos_h;
    Kokkos::View<float*>::HostMirror   zxy_rbuf_pos_h;
    Kokkos::View<float*>::HostMirror   xyz_sbuf_neg_h;
    Kokkos::View<float*>::HostMirror   yzx_sbuf_neg_h;
    Kokkos::View<float*>::HostMirror   zxy_sbuf_neg_h;
    Kokkos::View<float*>::HostMirror   xyz_rbuf_neg_h;
    Kokkos::View<float*>::HostMirror   yzx_rbuf_neg_h;
    Kokkos::View<float*>::HostMirror   zxy_rbuf_neg_h;

    field_buffers() {
        // User should try avoid calling this
    }

    field_buffers(int xyz_size, int yzx_size, int zxy_size) {
        xyz_sbuf_pos = Kokkos::View<float*>("Send buffer for XYZ positive face", xyz_size);
        xyz_rbuf_pos = Kokkos::View<float*>("Receive buffer for XYZ positive face", xyz_size);
        yzx_sbuf_pos = Kokkos::View<float*>("Send buffer for YZX positive face", yzx_size);
        yzx_rbuf_pos = Kokkos::View<float*>("Receive buffer for YZX positive face", yzx_size);
        zxy_sbuf_pos = Kokkos::View<float*>("Send buffer for ZXY positive face", zxy_size);
        zxy_rbuf_pos = Kokkos::View<float*>("Receive buffer for ZXY positive face", zxy_size);

        xyz_sbuf_neg = Kokkos::View<float*>("Send buffer for XYZ negative face", xyz_size);
        xyz_rbuf_neg = Kokkos::View<float*>("Receive buffer for XYZ negative face", xyz_size);
        yzx_sbuf_neg = Kokkos::View<float*>("Send buffer for YZX negative face", yzx_size);
        yzx_rbuf_neg = Kokkos::View<float*>("Receive buffer for YZX negative face", yzx_size);
        zxy_sbuf_neg = Kokkos::View<float*>("Send buffer for ZXY negative face", zxy_size);
        zxy_rbuf_neg = Kokkos::View<float*>("Receive buffer for ZXY negative face", zxy_size);

        xyz_sbuf_pos_h = Kokkos::create_mirror_view(xyz_sbuf_pos);
        yzx_sbuf_pos_h = Kokkos::create_mirror_view(yzx_sbuf_pos);
        zxy_sbuf_pos_h = Kokkos::create_mirror_view(zxy_sbuf_pos);
        xyz_rbuf_pos_h = Kokkos::create_mirror_view(xyz_rbuf_pos);
        yzx_rbuf_pos_h = Kokkos::create_mirror_view(yzx_rbuf_pos);
        zxy_rbuf_pos_h = Kokkos::create_mirror_view(zxy_rbuf_pos);

        xyz_sbuf_neg_h = Kokkos::create_mirror_view(xyz_sbuf_neg);
        yzx_sbuf_neg_h = Kokkos::create_mirror_view(yzx_sbuf_neg);
        zxy_sbuf_neg_h = Kokkos::create_mirror_view(zxy_sbuf_neg);
        xyz_rbuf_neg_h = Kokkos::create_mirror_view(xyz_rbuf_neg);
        yzx_rbuf_neg_h = Kokkos::create_mirror_view(yzx_rbuf_neg);
        zxy_rbuf_neg_h = Kokkos::create_mirror_view(zxy_rbuf_neg);
    }
} field_buffers_t;
// A field_array holds all the field quanties and pointers to
// kernels used to advance them.

typedef struct field_array {
  field_t * ALIGNED(128) f;           // Local field data
  grid_t  * g;                        // Underlying grid
  void    * params;                   // Field advance specific parameters
  field_advance_kernels_t kernel[1];  // Field advance kernels

  // I don't want this to be a pointer, but given it only holds Kokkos data
  // this avoids a fiasco when checkpointing...
  field_buffers_t* fb;

  k_field_t k_f_d;                   // Kokkos field data on device
  k_field_t::HostMirror k_f_h;       // Kokkos field data on host
  k_field_sa_t k_field_sa_d;
  k_field_edge_t k_fe_d;             // Kokkos field_edge data (part of field_t) on device
  k_field_edge_t::HostMirror k_fe_h; // Kokkos field_edge data on host

  k_field_accum_t k_f_rhob_accum_d;//TODO: Remove when absorbing pbc on device
  k_field_accum_t::HostMirror k_f_rhob_accum_h;

  k_jf_accum_t k_jf_accum_d;
  k_jf_accum_t::HostMirror k_jf_accum_h;

  // Step when the field was last copied to to the host.  The copy can
  // take place at any time during the step, so checking
  // last_copied==step() does not mean that the host and device
  // data are the same.  Typically, copy is called immediately after the
  // step is incremented and before or during user_diagnostics.  Checking
  // last_copied==step() in these circumstances does mean the host
  // is up to date, unless you do unusual stuff in user_diagnostics.
  //
  // This number is tracked on the host only, and may be inaccurate on
  // the device.
  int64_t last_copied = -1;

  // Constructors don't get called on restart..
  // Initialize Kokkos Field Array
  field_array(int n_fields, int xyz_sz, int yzx_sz, int zxy_sz)
  {
      init_kokkos_fields(n_fields, xyz_sz, yzx_sz, zxy_sz);
  }

  void init_kokkos_fields(int n_fields, int xyz_sz, int yzx_sz, int zxy_sz)
  {
      k_f_d = k_field_t("k_fields", n_fields);
      k_field_sa_d = Kokkos::Experimental::create_scatter_view(k_f_d);
      k_fe_d = k_field_edge_t("k_field_edges", n_fields);
      k_f_h = Kokkos::create_mirror_view(k_f_d);
      k_fe_h = Kokkos::create_mirror_view(k_fe_d);

      k_f_rhob_accum_d = k_field_accum_t("k_rhob_accum", n_fields);
      k_f_rhob_accum_h = Kokkos::create_mirror_view(k_f_rhob_accum_d);

      k_jf_accum_d = k_jf_accum_t("k_jf_accum", n_fields);
      k_jf_accum_h = Kokkos::create_mirror_view(k_jf_accum_d);

      fb = new field_buffers_t(xyz_sz, yzx_sz, zxy_sz);
  }

  ~field_array()
  {
      delete fb;
  }

  /**
   * @brief Copies the field data to the host.
   */
  void copy_to_host();

  /**
   * @brief Copies the field data to the device.
   */
  void copy_to_device();


} field_array_t;



field_array_t *
new_standard_field_array( grid_t           * RESTRICT g,
                          const material_t * RESTRICT m_list,
                          float                       damp );

void
delete_field_array( field_array_t * fa );


#endif // _field_advance_h_
