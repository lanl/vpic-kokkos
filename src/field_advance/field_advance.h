#ifndef _field_advance_h_
#define _field_advance_h_

// FIXME: (nx>1) ? (1/dx) : 0 TYPE LOGIC SHOULD BE FIXED SO THAT NX
// REFERS TO THE GLOBAL NUMBER OF CELLS IN THE X-DIRECTION (NOT THE
// _LOCAL_ NUMBER OF CELLS).  THIS LATENT BUG IS NOT EXPECTED TO
// AFFECT ANY PRACTICAL SIMULATIONS.

#include "../grid/grid.h"
#include "../material/material.h"
#include "../vpic/kokkos_helpers.h"

#include "field_buffers.h"
#include "field_kernels.h"

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
  k_field_edge_t k_fe_d;             // Kokkos field_edge data (part of field_t) on device
  k_field_edge_t::HostMirror k_fe_h; // Kokkos field_edge data on host

  k_field_accum_t k_f_rhob_accum_d;//TODO: Remove when absorbing pbc on device
  k_field_accum_t::HostMirror k_f_rhob_accum_h;


  field_array(int n_fields, int xyz_sz, int yzx_sz, int zxy_sz);

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
