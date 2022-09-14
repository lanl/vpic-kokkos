#ifndef _field_advance_h_
#define _field_advance_h_

// FIXME: (nx>1) ? (1/dx) : 0 TYPE LOGIC SHOULD BE FIXED SO THAT NX
// REFERS TO THE GLOBAL NUMBER OF CELLS IN THE X-DIRECTION (NOT THE
// _LOCAL_ NUMBER OF CELLS).  THIS LATENT BUG IS NOT EXPECTED TO
// AFFECT ANY PRACTICAL SIMULATIONS.

#include "../grid/grid.h"
#include "../material/material.h"
#include "../vpic/kokkos_helpers.h"

// FIXME: UPDATE THIS COMMENT BLOCK AND MOVE IT INTO APPROPRIATE LOCATIONS
//
// This module implements the following the difference equations on a
// superhexahedral domain decomposed Yee-mesh:
//
// advance_b -> Finite Differenced Faraday
//   cB_new = cB_old - frac c dt curl E
//
// advance_e -> Exponentially Differenced Ampere
//   TCA_new = -damp TCA_old + (1+damp) c dt curl (cB/mu_r)
//   E_new = decay E_old + drive [ TCA_new - (dt/eps0) J  ]
//     where the coefficients decay and drive are defined by the
//     diagonal materials properties tensor. That is:
//       decay_xx = exp(-alpha_xx)
//       drive_xx = [ 1 - exp(-alpha_xx) ] / (alpha_xx epsr_xx)
//       alpha_xx = sigma_xx dt / (eps0 epsr_xx)
//
// div_clean_b -> Marder pass on magnetic fields
//   cB_new = cB_old + D dt grad div cB_old
//     where the diffusion coefficient D is automatically selected to
//     rapidly reduce RMS divergence error assuming divergences errors
//     are due to accumulation of numerical roundoff when integrating
//     Faraday. See clean_div.c for details.
//
// div_clean_e -> Modified Marder pass on electric fields
//   E_new = E_old + drive D dt grad err_mul div ( epsr E_old - rho/eps0 )
//     Since the total rho may not be known everywhere (for example in
//     conductive regions), err_mul sets the divergence error to zero
//     in regions where total rho is unknown. Same considerations as
//     div_clean_b for the diffusion coefficient determination. See
//     clean_div.c for details.
//
// Domain fields are stored in a field_t array which is FORTRAN
// indexed 0:nx+1,0:ny+1,0:nz+1.  The fields are located on a
// staggered Yee-mesh
//
//   f(i,j,k).ex  @ i+0.5,j,k (all 1:nx,1:ny+1,1:nz+1; int 1:nx,2:ny,2:nz)
//   f(i,j,k).ey  @ i,j+0.5,k (all 1:nx+1,1:ny,1:nz+1; int 2:nx,1:ny,2:nz)
//   f(i,j,k).ez  @ i,j,k+0.5 (all 1:nx+1,1:ny+1,1:nz; int 2:nx,2:ny,1:nz)
//   f(i,j,k).cbx @ i,j+0.5,k+0.5 (all 1:nx+1,1:ny,1:nz; int 2:nx,1:ny,1:nz)
//   f(i,j,k).cby @ i+0.5,j,k+0.5 (all 1:nx,1:ny+1,1:nz; int 1:nx,2:ny,1:nz)
//   f(i,j,k).cbz @ i+0.5,j+0.5,k (all 1:nx,1:ny,1:nz+1; int 1:nx,1:ny,2:nz)
//   f(i,j,k).rhof @ i,j,k (all 1:nx+1,1:ny+1,1:nz+1; int 2:nx,2:ny,2:nz)
//   f(i,j,k).div_b_err(i,j,k) @ i+0.5,j+0.5,k+0.5 (all+int 1:nx,1:ny,1:nz)
//   f(i,j,k).rhob, div_e_err, nmat is on same mesh as rhof
//   f(i,j,k).jfx, jfy, jfz is on same mesh as ex,ey,ez respectively
//   f(i,j,k).tcax ,tcay, tcaz is on same mesh ex,ey,ez respectively
//   f(i,j,k).ematx, ematy, ematz are on same mesh as ex,ey,ez respectively
//   f(i,j,k).fmatx, fmaty, fmatz are on same mesh as cbx,cby,cbz respectively
//   f(i,j,k).cmat is on same mesh as div_b_err
//
// Alternatively, ex,ey,ez / jfx,jfy,jfz / tcax,tcay,tcaz /
// ematx,ematy,ematz are all on the "edge mesh". cbx,cby,cbz /
// fmatx,fmaty,fmatz are all on the "face
// mesh". rhof,rhob,div_e_err,nmat are on the "nodes mesh".
// div_b_err,cmat are on the "cell mesh".
//
// Above, for "edge mesh" quantities, interior means that the
// component is not a tangential field directly on the surface of the
// domain. For "face mesh" quantities, interior means that the
// component is not a normal field directly on the surface of the
// computational domain. For "node mesh" quantities, interior means
// that the component is not on the surface of the computational
// domain. All "cell mesh" quantities are interior to the computation
// domains.
//
// Boundary conditions and field communications are implemented in
// part by ghost values:
//   advance_e uses tangential b ghosts
//     cbx => 1:nx+1,{0|ny+1},1:nz and 1:nx+1,1:ny,{0|nz+1}
//     cby => 1:nx,1:ny+1,{0|nz+1} and {0|nx+1},1:ny+1,1:nz
//     cbz => {0|nx+1},1:ny,1:nz+1 and 1:nx,{0|ny+1},1:nz+1
//   advance_b does not use any ghosts
//   clean_div_e uses normal e ghosts.
//   clean_div_b uses derr ghosts.
//
// To setup the field advance, the following sequence is suggested:
//   grid_t grid[1];
//   material_t *material_list = NULL;
//   fields_t * ALIGNED(16) fields = NULL;
//   material_coefficient_t * ALIGNED(16) material_coefficients = NULL;
//
//   grid->nx = 32; grid->ny = 32; grid->nz = 32;
//   ...
//   new_material("Vacuum",1,1,0,&material_list);
//   ...
//   material_coefficients = new_material_coefficients(grid,material_list);
//   fields = new_fields(grid);
//
//   ... Set the initial field values and place materials ...
//
//   synchronize_fields(fields,grid);
//
// Note: synchronize_fields makes sure shared faces on each domain
// have the same fields (in case the user messed up setting the
// initial fields or errors in the source terms or different floating
// point properties on different nodes cause the shared faces to have
// different fields).
//
// To advance the fields in a PIC simulation with TCA radation damping
// and periodic divergence cleaning, the following sequence is
// suggested:
//   ... push/accumulate particles using E_0, B_0 => Jf_0.5
//   advance_b( fields, grid, 0.5  );                  => B_0 to B_0.5
//   advance_e( fields, material_coefficients, grid ); => E_0 to E_1
//   advance_b( fields, grid, 0.5  );                  => B_0.5 to B_1
//   if( should_clean_div_e ) {
//     ... adjust rho_f, rho_b and/or rho_c as necessary
//     do {
//       rms_err = clean_div_e( fields, material_coefficients, grid );
//     } while( rms_err_too_high );
//   }
//   if( should_clean_div_b ) {
//     do {
//       rms_err = clean_div_b( fields, grid );
//     } while( rms_err_too_high );
//   }
//   if( should_sync_fields ) synchronize_fields(fields,grid);
//
// To clean up the field advance, the following sequence is suggested:
//   delete_materials(&materials_list);
//   delete_materials_coefficients(&materials_coefficients);
//   delete_fields(&fields);

// Field arrays shall be a (nx+2) x (ny+2) x (nz+2) allocation indexed
// FORTRAN style from (0:nx+1,0:ny+1,0:nz+1).  Fields for voxels on
// the surface of the local domain (for example h(0,:,:) or
// h(nx+1,:,:)) are used to store values of the field array on
// neighbor processors or used to enforce bonundary conditions.

// Note: When setting the material IDs on the mesh, the material IDs
// should be set in the ghost cells too. Further, these IDs should be
// consistent with the neighboring domains (if any)!
//

// FIXME: MATERIAL-LESS FIELD_T SHOULD EVENTUALLY USE ITS OWN FIELD_T
// WITH MORE COMPACT LAYOUT.

// FIXME: SHOULD HAVE DIFFERENT FIELD_T FOR CELL BUILDS AND USE NEW
// INFRASTRUCTURE

typedef struct field {
  float ex,   ey,   ez,   div_e_err;     // Electric field and div E error
  float cbx,  cby,  cbz,  div_b_err;     // Magnetic field and div B error
  float tcax, tcay, tcaz, rhob;          // TCA fields and bound charge density
  float jfx,  jfy,  jfz,  rhof;          // Free current and charge density
  material_id ematx, ematy, ematz, nmat; // Material at edge centers and nodes
  material_id fmatx, fmaty, fmatz, cmat; // Material at face and cell centers
} field_t;

// field_advance_kernels holds all the function pointers to all the
// kernels used by a specific field_advance instance.

// FIXME: DOCUMENT THESE INTERFACES HERE AND NOT IN STANDARD FIELD
// ADVANCE PRIVATE

struct field_array;

typedef struct field_advance_kernels {

  // FIXME: DUMP.CXX SHOULD BE DECENTRALIZED AND DIAGNOSTIC DUMP
  // FOR FIELDS SHOULD BE ADDED TO THIS
  // FIXME: FOR SYSTEMS WITH MAGNETIC CURRENTS (E.G. PML LAYERS)
  // WOULD INTERFACES FOR xif,kf BE USEFUL?

  void (*delete_fa)( struct field_array * RESTRICT fa );

  // Time stepping interface

  void (*advance_b)( struct field_array * RESTRICT fa, float frac );
  void (*advance_e)( struct field_array * RESTRICT fa, float frac );

  // Diagnostic interface
  // FIXME: MAY NEED MORE CAREFUL THOUGHT FOR CURVILINEAR SYSTEMS

  void (*energy_f)( /**/  double        * RESTRICT en, // 6 elem
                    const struct field_array * RESTRICT fa );

  // Accumulator interface

  void (*clear_jf       )( struct field_array * RESTRICT fa );
  void (*synchronize_jf )( struct field_array * RESTRICT fa );
  void (*clear_rhof     )( struct field_array * RESTRICT fa );
  void (*synchronize_rho)( struct field_array * RESTRICT fa );

  // Initialization interface

  void (*compute_rhob  )( struct field_array * RESTRICT fa );
  void (*compute_curl_b)( struct field_array * RESTRICT fa );

  // Local/remote shared face cleaning

  double (*synchronize_tang_e_norm_b)( struct field_array * RESTRICT fa );

  // Electric field divergence cleaning interface

  void   (*compute_div_e_err    )( /**/  struct field_array * RESTRICT fa );
  double (*compute_rms_div_e_err)( const struct field_array * RESTRICT fa );
  void   (*clean_div_e          )( /**/  struct field_array * RESTRICT fa );

  // Magnetic field divergence cleaning interface

  void   (*compute_div_b_err    )( /**/  struct field_array * RESTRICT fa );
  double (*compute_rms_div_b_err)( const struct field_array * RESTRICT fa );
  void   (*clean_div_b          )( /**/  struct field_array * RESTRICT fa );

  void (*advance_e_kokkos)( struct field_array * RESTRICT fa, float frac );
  void (*energy_f_kokkos)( /**/  double        * RESTRICT en, // 6 elem
                    const struct field_array * RESTRICT fa );
  void (*clear_jf_kokkos)( struct field_array * RESTRICT fa );
  void (*clear_rhof_kokkos     )( struct field_array * RESTRICT fa );
  void (*k_synchronize_jf )( struct field_array * RESTRICT fa );
  void (*k_reduce_jf )( struct field_array * RESTRICT fa );
  void (*k_synchronize_rho)( struct field_array * RESTRICT fa );
  double (*synchronize_tang_e_norm_b_kokkos)( struct field_array * RESTRICT fa );

  void   (*compute_div_e_err_kokkos  )( /**/  struct field_array * RESTRICT fa );
  double (*compute_rms_div_e_err_kokkos)(const struct field_array * RESTRICT fa);
  void   (*clean_div_e_kokkos   )( /**/  struct field_array * RESTRICT fa );

  void   (*compute_div_b_err_kokkos)( struct field_array* RESTRICT fa );
  double (*compute_rms_div_b_err_kokkos)( const struct field_array * RESTRICT fa );
  void   (*clean_div_b_kokkos   )( /**/  struct field_array * RESTRICT fa );


} field_advance_kernels_t;

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


// Move in from SFA private
typedef struct material_coefficient {
  float decayx, drivex;         // Decay of ex and drive of (curl H)x and Jx
  float decayy, drivey;         // Decay of ey and drive of (curl H)y and Jy
  float decayz, drivez;         // Decay of ez and drive of (curl H)z and Jz
  float rmux, rmuy, rmuz;       // Reciprocle of relative permeability
  float nonconductive;          // Divergence cleaning related coefficients
  float epsx, epsy, epsz;
  float pad[3];                 // For 64-byte alignment and future expansion
} material_coefficient_t;

typedef struct sfa_params {
    material_coefficient_t * mc;
    int n_mc;
    float damp;

    k_material_coefficient_t k_mc_d;
    k_material_coefficient_t::HostMirror k_mc_h;

    int n_materials;

    sfa_params(int _n_materials) {
        n_materials = _n_materials;
        init_kokkos_sfa_params(_n_materials);
    }
    void init_kokkos_sfa_params(int n_materials)
    {
        k_mc_d = k_material_coefficient_t("k_material_coefficents", n_materials);
        k_mc_h = Kokkos::create_mirror_view(k_mc_d);
    }

    void populate_kokkos_data()
    {
        Kokkos::parallel_for("Copy materials to device", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_mc), KOKKOS_CLASS_LAMBDA (const int i)
        {
                k_mc_h(i, material_coeff_var::decayx) = mc[i].decayx;
                k_mc_h(i, material_coeff_var::drivex) = mc[i].drivex;
                k_mc_h(i, material_coeff_var::decayy) = mc[i].decayy;
                k_mc_h(i, material_coeff_var::drivey) = mc[i].drivey;
                k_mc_h(i, material_coeff_var::decayz) = mc[i].decayz;
                k_mc_h(i, material_coeff_var::drivez) = mc[i].drivez;
                k_mc_h(i, material_coeff_var::rmux) = mc[i].rmux;
                k_mc_h(i, material_coeff_var::rmuy) = mc[i].rmuy;
                k_mc_h(i, material_coeff_var::rmuz) = mc[i].rmuz;
                k_mc_h(i, material_coeff_var::nonconductive) = mc[i].nonconductive;
                k_mc_h(i, material_coeff_var::epsx) = mc[i].epsx;
                k_mc_h(i, material_coeff_var::epsy) = mc[i].epsy;
                k_mc_h(i, material_coeff_var::epsz) = mc[i].epsz;
        });
        Kokkos::deep_copy(k_mc_d, k_mc_h);
    }

} sfa_params_t;

#endif // _field_advance_h_
