#define IN_sf_interface
#define HAS_V4_PIPELINE
#include "sf_interface_private.h"


void
checkpt_interpolator_array( const interpolator_array_t * ia ) {
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  CHECKPT( ia, 1 );
  CHECKPT_ALIGNED( ia->i, ia->g->nv, 128 );
  CHECKPT_PTR( ia->g );
#else
  CHECKPT_VIEW( ia->k_i_h );
  CHECKPT_PTR( ia->g );
#endif
}

interpolator_array_t *
restore_interpolator_array( void ) {
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  interpolator_array_t * ia;
  RESTORE( ia );
  RESTORE_ALIGNED( ia->i );
  RESTORE_PTR( ia->g );
#else
  interpolator_array_t * ia = new interpolator_array_t(1);
  RESTORE_VIEW( ia->k_i_h );
  RESTORE_PTR( ia->g );
#endif
  return ia;
}

interpolator_array_t *
new_interpolator_array( grid_t * g ) {
  interpolator_array_t * ia;
  if( !g ) ERROR(( "NULL grid" ));
  ia = new interpolator_array_t(g->nv);
  //MALLOC( ia, 1 );
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  MALLOC_ALIGNED( ia->i, g->nv, 128 );
  CLEAR( ia->i, g->nv );
#endif
  ia->g = g;
  REGISTER_OBJECT( ia, checkpt_interpolator_array, restore_interpolator_array,
                   NULL );
  return ia;
}

void
delete_interpolator_array( interpolator_array_t * ia ) {
  if( !ia ) return;
  UNREGISTER_OBJECT( ia );
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  FREE_ALIGNED( ia->i );
#endif
  delete(ia);
  //FREE( ia );
}

KOKKOS_INLINE_FUNCTION
void transform_E_to_cartesian(
    const k_curvilinear_mesh_t& k_cmesh,
    int mesh_index,
    float E_xi, float E_eta, float E_zeta,
    float& Ex, float& Ey, float& Ez)
{
    // Equation 62: E^α = E_ν (∇ξ^ν)^α
    // For orthogonal coordinates: ∇ξⁱ = (1/hⁱ²) eⁱ
    // where eⁱ = ∂x/∂ξⁱ are the tangent basis vectors
    
    // Load scale factors
    float h1 = k_cmesh(mesh_index, curv_mesh_var::h_1);
    float h2 = k_cmesh(mesh_index, curv_mesh_var::h_2);
    float h3 = k_cmesh(mesh_index, curv_mesh_var::h_3);
    
    // Load tangent basis vectors (∂x/∂ξⁱ)
    float e1_u = k_cmesh(mesh_index, curv_mesh_var::e_1_u);  // e₁ˣ
    float e1_v = k_cmesh(mesh_index, curv_mesh_var::e_1_v);  // e₁ʸ
    float e1_w = k_cmesh(mesh_index, curv_mesh_var::e_1_w);  // e₁ᶻ
    
    float e2_u = k_cmesh(mesh_index, curv_mesh_var::e_2_u);  // e₂ˣ
    float e2_v = k_cmesh(mesh_index, curv_mesh_var::e_2_v);  // e₂ʸ
    float e2_w = k_cmesh(mesh_index, curv_mesh_var::e_2_w);  // e₂ᶻ
    
    float e3_u = k_cmesh(mesh_index, curv_mesh_var::e_3_u);  // e₃ˣ
    float e3_v = k_cmesh(mesh_index, curv_mesh_var::e_3_v);  // e₃ʸ
    float e3_w = k_cmesh(mesh_index, curv_mesh_var::e_3_w);  // e₃ᶻ
    
    // Compute reciprocal basis: ∇ξⁱ = (1/hⁱ²) eⁱ
    float inv_h1_sq = 1.0f / (h1 * h1);
    float inv_h2_sq = 1.0f / (h2 * h2);
    float inv_h3_sq = 1.0f / (h3 * h3);
    
    float grad_xi_x   = inv_h1_sq * e1_u;
    float grad_xi_y   = inv_h1_sq * e1_v;
    float grad_xi_z   = inv_h1_sq * e1_w;
    
    float grad_eta_x  = inv_h2_sq * e2_u;
    float grad_eta_y  = inv_h2_sq * e2_v;
    float grad_eta_z  = inv_h2_sq * e2_w;
    
    float grad_zeta_x = inv_h3_sq * e3_u;
    float grad_zeta_y = inv_h3_sq * e3_v;
    float grad_zeta_z = inv_h3_sq * e3_w;
    
    // Transform: E^α = E_ν (∇ξ^ν)^α
    Ex = E_xi * grad_xi_x + E_eta * grad_eta_x + E_zeta * grad_zeta_x;
    Ey = E_xi * grad_xi_y + E_eta * grad_eta_y + E_zeta * grad_zeta_y;
    Ez = E_xi * grad_xi_z + E_eta * grad_eta_z + E_zeta * grad_zeta_z;
}

KOKKOS_INLINE_FUNCTION
void transform_B_to_cartesian(
    const k_curvilinear_mesh_t& k_cmesh,
    int mesh_index,
    float B_xi, float B_eta, float B_zeta,
    float& Bx, float& By, float& Bz)
{
    // Equation 63: B^α = B^ν (∂x/∂ξ^ν)^α
    // We have the tangent basis vectors directly stored!
    
    // Load tangent basis vectors (∂x/∂ξⁱ)
    float e1_u = k_cmesh(mesh_index, curv_mesh_var::e_1_u);  // (∂x/∂ξ)ˣ
    float e1_v = k_cmesh(mesh_index, curv_mesh_var::e_1_v);  // (∂x/∂ξ)ʸ
    float e1_w = k_cmesh(mesh_index, curv_mesh_var::e_1_w);  // (∂x/∂ξ)ᶻ
    
    float e2_u = k_cmesh(mesh_index, curv_mesh_var::e_2_u);  // (∂x/∂η)ˣ
    float e2_v = k_cmesh(mesh_index, curv_mesh_var::e_2_v);  // (∂x/∂η)ʸ
    float e2_w = k_cmesh(mesh_index, curv_mesh_var::e_2_w);  // (∂x/∂η)ᶻ
    
    float e3_u = k_cmesh(mesh_index, curv_mesh_var::e_3_u);  // (∂x/∂ζ)ˣ
    float e3_v = k_cmesh(mesh_index, curv_mesh_var::e_3_v);  // (∂x/∂ζ)ʸ
    float e3_w = k_cmesh(mesh_index, curv_mesh_var::e_3_w);  // (∂x/∂ζ)ᶻ
    
    // Transform: B^α = B^ν (∂x/∂ξ^ν)^α
    Bx = B_xi * e1_u + B_eta * e2_u + B_zeta * e3_u;
    By = B_xi * e1_v + B_eta * e2_v + B_zeta * e3_v;
    Bz = B_xi * e1_w + B_eta * e2_w + B_zeta * e3_w;
}

void 
load_interpolator_array_kokkos(k_interpolator_t k_interp, 
                               k_field_t k_field,
                               k_curvilinear_mesh_t k_cmesh,  // ← NEW: Pass curvilinear mesh
                               int nx, int ny, int nz) 
{
  #define pi_ex       k_interp(pi_index, interpolator_var::ex)
  #define pi_dexdx    k_interp(pi_index, interpolator_var::dexdx)
  #define pi_dexdy    k_interp(pi_index, interpolator_var::dexdy)
  #define pi_dexdz    k_interp(pi_index, interpolator_var::dexdz)
  #define pi_d2exdx   k_interp(pi_index, interpolator_var::d2exdx)
  #define pi_d2exdy   k_interp(pi_index, interpolator_var::d2exdy)
  #define pi_d2exdz   k_interp(pi_index, interpolator_var::d2exdz)
  #define pi_ey       k_interp(pi_index, interpolator_var::ey)
  #define pi_deydx    k_interp(pi_index, interpolator_var::deydx)
  #define pi_deydy    k_interp(pi_index, interpolator_var::deydy)
  #define pi_deydz    k_interp(pi_index, interpolator_var::deydz)
  #define pi_d2eydx   k_interp(pi_index, interpolator_var::d2eydx)
  #define pi_d2eydy   k_interp(pi_index, interpolator_var::d2eydy)
  #define pi_d2eydz   k_interp(pi_index, interpolator_var::d2eydz)
  #define pi_ez       k_interp(pi_index, interpolator_var::ez)
  #define pi_dezdx    k_interp(pi_index, interpolator_var::dezdx)
  #define pi_dezdy    k_interp(pi_index, interpolator_var::dezdy)
  #define pi_dezdz    k_interp(pi_index, interpolator_var::dezdz)
  #define pi_d2ezdx   k_interp(pi_index, interpolator_var::d2ezdx)
  #define pi_d2ezdy   k_interp(pi_index, interpolator_var::d2ezdy)
  #define pi_d2ezdz   k_interp(pi_index, interpolator_var::d2ezdz)
  #define pi_cbx      k_interp(pi_index, interpolator_var::cbx)
  #define pi_dcbxdx   k_interp(pi_index, interpolator_var::dcbxdx)
  #define pi_dcbxdy   k_interp(pi_index, interpolator_var::dcbxdy)
  #define pi_dcbxdz   k_interp(pi_index, interpolator_var::dcbxdz)
  #define pi_d2cbxdx  k_interp(pi_index, interpolator_var::d2cbxdx)
  #define pi_d2cbxdy  k_interp(pi_index, interpolator_var::d2cbxdy)
  #define pi_d2cbxdz  k_interp(pi_index, interpolator_var::d2cbxdz)
  #define pi_cby      k_interp(pi_index, interpolator_var::cby)
  #define pi_dcbydx   k_interp(pi_index, interpolator_var::dcbydx)
  #define pi_dcbydy   k_interp(pi_index, interpolator_var::dcbydy)
  #define pi_dcbydz   k_interp(pi_index, interpolator_var::dcbydz)
  #define pi_d2cbydx  k_interp(pi_index, interpolator_var::d2cbydx)
  #define pi_d2cbydy  k_interp(pi_index, interpolator_var::d2cbydy)
  #define pi_d2cbydz  k_interp(pi_index, interpolator_var::d2cbydz)
  #define pi_cbz      k_interp(pi_index, interpolator_var::cbz)
  #define pi_dcbzdx   k_interp(pi_index, interpolator_var::dcbzdx)
  #define pi_dcbzdy   k_interp(pi_index, interpolator_var::dcbzdy)
  #define pi_dcbzdz   k_interp(pi_index, interpolator_var::dcbzdz)
  #define pi_d2cbzdx  k_interp(pi_index, interpolator_var::d2cbzdx)
  #define pi_d2cbzdy  k_interp(pi_index, interpolator_var::d2cbzdy)
  #define pi_d2cbzdz  k_interp(pi_index, interpolator_var::d2cbzdz)

  #define pi_Ex0       k_interp(pi_index, interpolator_var::Ex0)
  #define pi_dEx0dx    k_interp(pi_index, interpolator_var::dEx0dx)
  #define pi_dEx0dy    k_interp(pi_index, interpolator_var::dEx0dy)
  #define pi_dEx0dz    k_interp(pi_index, interpolator_var::dEx0dz)
  #define pi_d2Ex0dx   k_interp(pi_index, interpolator_var::d2Ex0dx)
  #define pi_d2Ex0dy   k_interp(pi_index, interpolator_var::d2Ex0dy)
  #define pi_d2Ex0dz   k_interp(pi_index, interpolator_var::d2Ex0dz)
  #define pi_Ey0       k_interp(pi_index, interpolator_var::Ey0)
  #define pi_dEy0dx    k_interp(pi_index, interpolator_var::dEy0dx)
  #define pi_dEy0dy    k_interp(pi_index, interpolator_var::dEy0dy)
  #define pi_dEy0dz    k_interp(pi_index, interpolator_var::dEy0dz)
  #define pi_d2Ey0dx   k_interp(pi_index, interpolator_var::d2Ey0dx)
  #define pi_d2Ey0dy   k_interp(pi_index, interpolator_var::d2Ey0dy)
  #define pi_d2Ey0dz   k_interp(pi_index, interpolator_var::d2Ey0dz)
  #define pi_Ez0       k_interp(pi_index, interpolator_var::Ez0)
  #define pi_dEz0dx    k_interp(pi_index, interpolator_var::dEz0dx)
  #define pi_dEz0dy    k_interp(pi_index, interpolator_var::dEz0dy)
  #define pi_dEz0dz    k_interp(pi_index, interpolator_var::dEz0dz)
  #define pi_d2Ez0dx   k_interp(pi_index, interpolator_var::d2Ez0dx)
  #define pi_d2Ez0dy   k_interp(pi_index, interpolator_var::d2Ez0dy)
  #define pi_d2Ez0dz   k_interp(pi_index, interpolator_var::d2Ez0dz)

  #define pi_Gx0       k_interp(pi_index, interpolator_var::Gx0)
  #define pi_dGx0dx    k_interp(pi_index, interpolator_var::dGx0dx)
  #define pi_dGx0dy    k_interp(pi_index, interpolator_var::dGx0dy)
  #define pi_dGx0dz    k_interp(pi_index, interpolator_var::dGx0dz)
  #define pi_d2Gx0dx   k_interp(pi_index, interpolator_var::d2Gx0dx)
  #define pi_d2Gx0dy   k_interp(pi_index, interpolator_var::d2Gx0dy)
  #define pi_d2Gx0dz   k_interp(pi_index, interpolator_var::d2Gx0dz)
  #define pi_Gy0       k_interp(pi_index, interpolator_var::Gy0)
  #define pi_dGy0dx    k_interp(pi_index, interpolator_var::dGy0dx)
  #define pi_dGy0dy    k_interp(pi_index, interpolator_var::dGy0dy)
  #define pi_dGy0dz    k_interp(pi_index, interpolator_var::dGy0dz)
  #define pi_d2Gy0dx   k_interp(pi_index, interpolator_var::d2Gy0dx)
  #define pi_d2Gy0dy   k_interp(pi_index, interpolator_var::d2Gy0dy)
  #define pi_d2Gy0dz   k_interp(pi_index, interpolator_var::d2Gy0dz)
  #define pi_Gz0       k_interp(pi_index, interpolator_var::Gz0)
  #define pi_dGz0dx    k_interp(pi_index, interpolator_var::dGz0dx)
  #define pi_dGz0dy    k_interp(pi_index, interpolator_var::dGz0dy)
  #define pi_dGz0dz    k_interp(pi_index, interpolator_var::dGz0dz)
  #define pi_d2Gz0dx   k_interp(pi_index, interpolator_var::d2Gz0dx)
  #define pi_d2Gz0dy   k_interp(pi_index, interpolator_var::d2Gz0dy)
  #define pi_d2Gz0dz   k_interp(pi_index, interpolator_var::d2Gz0dz)

  constexpr float twelfth = 1./12.;
  constexpr float sixth   = 1./6.;

  Kokkos::MDRangePolicy<Kokkos::Rank<3>> load_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
  Kokkos::parallel_for("load interpolator", load_policy, 
      KOKKOS_LAMBDA(const int x, const int y, const int z) 
  {
      const int pi_index   = VOXEL(x,   y,   z,   nx,ny,nz); 
      const int pf0_index  = VOXEL(x,   y,   z,   nx,ny,nz); 
      const int pfx_index  = VOXEL(x+1, y,   z,   nx,ny,nz); 
      const int pfy_index  = VOXEL(x,   y+1, z,   nx,ny,nz); 
      const int pfz_index  = VOXEL(x,   y,   z+1, nx,ny,nz); 
      const int pfmx_index = VOXEL(x-1, y,   z,   nx,ny,nz);
      const int pfmy_index = VOXEL(x,   y-1, z,   nx,ny,nz);
      const int pfmz_index = VOXEL(x,   y,   z-1, nx,ny,nz);

      const int pm0_index  = VOXEL_TO_MESH(pf0_index, nx, ny, nz);

#ifdef SHAPE_NGP
      // ====================================================================
      // NGP: Transform curvilinear → Cartesian at this grid point
      // ====================================================================
      
      // Read curvilinear E field (covariant components)
      float E_xi   = k_field(pf0_index, field_var::tx);
      float E_eta  = k_field(pf0_index, field_var::ty);
      float E_zeta = k_field(pf0_index, field_var::tz);
      
      // Transform to Cartesian using equation 62
      float Ex, Ey, Ez;
      transform_E_to_cartesian(k_cmesh, pm0_index, 
                               E_xi, E_eta, E_zeta,
                               Ex, Ey, Ez);
      
      pi_ex = Ex;
      pi_ey = Ey;
      pi_ez = Ez;
      
      // Read B field (contravariant components in curvilinear basis)
      float B_xi   = k_field(pf0_index, field_var::ox) + k_field(pf0_index, field_var::cbx0);
      float B_eta  = k_field(pf0_index, field_var::oy) + k_field(pf0_index, field_var::cby0);
      float B_zeta = k_field(pf0_index, field_var::oz) + k_field(pf0_index, field_var::cbz0);
      
      // Transform to Cartesian using equation 63
      float Bx, By, Bz;
      transform_B_to_cartesian(k_cmesh, pm0_index,
                               B_xi, B_eta, B_zeta,
                               Bx, By, Bz);
      
      pi_cbx = Bx;
      pi_cby = By;
      pi_cbz = Bz;
      
#ifdef EXTERNAL_FORCE
      // External forces (assuming these are already in Cartesian coordinates)
      // If they're in curvilinear coordinates, they need similar transformation
      pi_Ex0 = k_field(pf0_index, field_var::Ex0);
      pi_Ey0 = k_field(pf0_index, field_var::Ey0);
      pi_Ez0 = k_field(pf0_index, field_var::Ez0);
      
      pi_Gx0 = k_field(pf0_index, field_var::Gx0);
      pi_Gy0 = k_field(pf0_index, field_var::Gy0);
      pi_Gz0 = k_field(pf0_index, field_var::Gz0);
#endif

#elif defined( SHAPE_QS )
      // ====================================================================
      // QS: Compute derivatives in curvilinear, then transform to Cartesian
      // ====================================================================
      
      // -------------------- E FIELD (all 3 components) --------------------
      
      // E_xi component (covariant)
      auto E_xi_0  = k_field(pf0_index,  field_var::tx);
      auto E_xi_x  = k_field(pfx_index,  field_var::tx);
      auto E_xi_y  = k_field(pfy_index,  field_var::tx);
      auto E_xi_z  = k_field(pfz_index,  field_var::tx);
      auto E_xi_mx = k_field(pfmx_index, field_var::tx);
      auto E_xi_my = k_field(pfmy_index, field_var::tx);
      auto E_xi_mz = k_field(pfmz_index, field_var::tx);
      
      float E_xi_interp     = twelfth*(6.f*E_xi_0 + E_xi_x + E_xi_y + E_xi_z + E_xi_mx + E_xi_my + E_xi_mz);
      float dE_xi_dx        = sixth*(E_xi_x - E_xi_mx);
      float dE_xi_dy        = sixth*(E_xi_y - E_xi_my);
      float dE_xi_dz        = sixth*(E_xi_z - E_xi_mz);
      float d2E_xi_dx       = twelfth*(E_xi_x + E_xi_mx - 2.f*E_xi_0);
      float d2E_xi_dy       = twelfth*(E_xi_y + E_xi_my - 2.f*E_xi_0);
      float d2E_xi_dz       = twelfth*(E_xi_z + E_xi_mz - 2.f*E_xi_0);
      
      // E_eta component (covariant)
      auto E_eta_0  = k_field(pf0_index,  field_var::ty);
      auto E_eta_x  = k_field(pfx_index,  field_var::ty);
      auto E_eta_y  = k_field(pfy_index,  field_var::ty);
      auto E_eta_z  = k_field(pfz_index,  field_var::ty);
      auto E_eta_mx = k_field(pfmx_index, field_var::ty);
      auto E_eta_my = k_field(pfmy_index, field_var::ty);
      auto E_eta_mz = k_field(pfmz_index, field_var::ty);
      
      float E_eta_interp    = twelfth*(6.f*E_eta_0 + E_eta_x + E_eta_y + E_eta_z + E_eta_mx + E_eta_my + E_eta_mz);
      float dE_eta_dx       = sixth*(E_eta_x - E_eta_mx);
      float dE_eta_dy       = sixth*(E_eta_y - E_eta_my);
      float dE_eta_dz       = sixth*(E_eta_z - E_eta_mz);
      float d2E_eta_dx      = twelfth*(E_eta_x + E_eta_mx - 2.f*E_eta_0);
      float d2E_eta_dy      = twelfth*(E_eta_y + E_eta_my - 2.f*E_eta_0);
      float d2E_eta_dz      = twelfth*(E_eta_z + E_eta_mz - 2.f*E_eta_0);
      
      // E_zeta component (covariant)
      auto E_zeta_0  = k_field(pf0_index,  field_var::tz);
      auto E_zeta_x  = k_field(pfx_index,  field_var::tz);
      auto E_zeta_y  = k_field(pfy_index,  field_var::tz);
      auto E_zeta_z  = k_field(pfz_index,  field_var::tz);
      auto E_zeta_mx = k_field(pfmx_index, field_var::tz);
      auto E_zeta_my = k_field(pfmy_index, field_var::tz);
      auto E_zeta_mz = k_field(pfmz_index, field_var::tz);
      
      float E_zeta_interp   = twelfth*(6.f*E_zeta_0 + E_zeta_x + E_zeta_y + E_zeta_z + E_zeta_mx + E_zeta_my + E_zeta_mz);
      float dE_zeta_dx      = sixth*(E_zeta_x - E_zeta_mx);
      float dE_zeta_dy      = sixth*(E_zeta_y - E_zeta_my);
      float dE_zeta_dz      = sixth*(E_zeta_z - E_zeta_mz);
      float d2E_zeta_dx     = twelfth*(E_zeta_x + E_zeta_mx - 2.f*E_zeta_0);
      float d2E_zeta_dy     = twelfth*(E_zeta_y + E_zeta_my - 2.f*E_zeta_0);
      float d2E_zeta_dz     = twelfth*(E_zeta_z + E_zeta_mz - 2.f*E_zeta_0);
      
      // Transform E field to Cartesian (equation 62)
      float Ex_cart, Ey_cart, Ez_cart;
      transform_E_to_cartesian(k_cmesh, pf0_index,
                               E_xi_interp, E_eta_interp, E_zeta_interp,
                               Ex_cart, Ey_cart, Ez_cart);
      
      // Store Cartesian E field and derivatives
      // NOTE: Derivatives are still in curvilinear coordinates (would need tensor transform for full correctness)
      pi_ex     = Ex_cart;
      pi_dexdx  = dE_xi_dx;   // Approximate - should be transformed
      pi_dexdy  = dE_xi_dy;
      pi_dexdz  = dE_xi_dz;
      pi_d2exdx = d2E_xi_dx;
      pi_d2exdy = d2E_xi_dy;
      pi_d2exdz = d2E_xi_dz;
      
      pi_ey     = Ey_cart;
      pi_deydx  = dE_eta_dx;
      pi_deydy  = dE_eta_dy;
      pi_deydz  = dE_eta_dz;
      pi_d2eydx = d2E_eta_dx;
      pi_d2eydy = d2E_eta_dy;
      pi_d2eydz = d2E_eta_dz;
      
      pi_ez     = Ez_cart;
      pi_dezdx  = dE_zeta_dx;
      pi_dezdy  = dE_zeta_dy;
      pi_dezdz  = dE_zeta_dz;
      pi_d2ezdx = d2E_zeta_dx;
      pi_d2ezdy = d2E_zeta_dy;
      pi_d2ezdz = d2E_zeta_dz;

      // -------------------- B FIELD (all 3 components) --------------------
      
      // B^xi component (contravariant)
      auto B_xi_0  = k_field(pf0_index,  field_var::ox) + k_field(pf0_index,  field_var::cbx0);
      auto B_xi_x  = k_field(pfx_index,  field_var::ox) + k_field(pfx_index,  field_var::cbx0);
      auto B_xi_y  = k_field(pfy_index,  field_var::ox) + k_field(pfy_index,  field_var::cbx0);
      auto B_xi_z  = k_field(pfz_index,  field_var::ox) + k_field(pfz_index,  field_var::cbx0);
      auto B_xi_mx = k_field(pfmx_index, field_var::ox) + k_field(pfmx_index, field_var::cbx0);
      auto B_xi_my = k_field(pfmy_index, field_var::ox) + k_field(pfmy_index, field_var::cbx0);
      auto B_xi_mz = k_field(pfmz_index, field_var::ox) + k_field(pfmz_index, field_var::cbx0);
      
      float B_xi_interp     = twelfth*(6.f*B_xi_0 + B_xi_x + B_xi_y + B_xi_z + B_xi_mx + B_xi_my + B_xi_mz);
      float dB_xi_dx        = sixth*(B_xi_x - B_xi_mx);
      float dB_xi_dy        = sixth*(B_xi_y - B_xi_my);
      float dB_xi_dz        = sixth*(B_xi_z - B_xi_mz);
      float d2B_xi_dx       = twelfth*(B_xi_x + B_xi_mx - 2.f*B_xi_0);
      float d2B_xi_dy       = twelfth*(B_xi_y + B_xi_my - 2.f*B_xi_0);
      float d2B_xi_dz       = twelfth*(B_xi_z + B_xi_mz - 2.f*B_xi_0);
      
      // B^eta component (contravariant)
      auto B_eta_0  = k_field(pf0_index,  field_var::oy) + k_field(pf0_index,  field_var::cby0);
      auto B_eta_x  = k_field(pfx_index,  field_var::oy) + k_field(pfx_index,  field_var::cby0);
      auto B_eta_y  = k_field(pfy_index,  field_var::oy) + k_field(pfy_index,  field_var::cby0);
      auto B_eta_z  = k_field(pfz_index,  field_var::oy) + k_field(pfz_index,  field_var::cby0);
      auto B_eta_mx = k_field(pfmx_index, field_var::oy) + k_field(pfmx_index, field_var::cby0);
      auto B_eta_my = k_field(pfmy_index, field_var::oy) + k_field(pfmy_index, field_var::cby0);
      auto B_eta_mz = k_field(pfmz_index, field_var::oy) + k_field(pfmz_index, field_var::cby0);
      
      float B_eta_interp    = twelfth*(6.f*B_eta_0 + B_eta_x + B_eta_y + B_eta_z + B_eta_mx + B_eta_my + B_eta_mz);
      float dB_eta_dx       = sixth*(B_eta_x - B_eta_mx);
      float dB_eta_dy       = sixth*(B_eta_y - B_eta_my);
      float dB_eta_dz       = sixth*(B_eta_z - B_eta_mz);
      float d2B_eta_dx      = twelfth*(B_eta_x + B_eta_mx - 2.f*B_eta_0);
      float d2B_eta_dy      = twelfth*(B_eta_y + B_eta_my - 2.f*B_eta_0);
      float d2B_eta_dz      = twelfth*(B_eta_z + B_eta_mz - 2.f*B_eta_0);
      
      // B^zeta component (contravariant)
      auto B_zeta_0  = k_field(pf0_index,  field_var::oz) + k_field(pf0_index,  field_var::cbz0);
      auto B_zeta_x  = k_field(pfx_index,  field_var::oz) + k_field(pfx_index,  field_var::cbz0);
      auto B_zeta_y  = k_field(pfy_index,  field_var::oz) + k_field(pfy_index,  field_var::cbz0);
      auto B_zeta_z  = k_field(pfz_index,  field_var::oz) + k_field(pfz_index,  field_var::cbz0);
      auto B_zeta_mx = k_field(pfmx_index, field_var::oz) + k_field(pfmx_index, field_var::cbz0);
      auto B_zeta_my = k_field(pfmy_index, field_var::oz) + k_field(pfmy_index, field_var::cbz0);
      auto B_zeta_mz = k_field(pfmz_index, field_var::oz) + k_field(pfmz_index, field_var::cbz0);
      
      float B_zeta_interp   = twelfth*(6.f*B_zeta_0 + B_zeta_x + B_zeta_y + B_zeta_z + B_zeta_mx + B_zeta_my + B_zeta_mz);
      float dB_zeta_dx      = sixth*(B_zeta_x - B_zeta_mx);
      float dB_zeta_dy      = sixth*(B_zeta_y - B_zeta_my);
      float dB_zeta_dz      = sixth*(B_zeta_z - B_zeta_mz);
      float d2B_zeta_dx     = twelfth*(B_zeta_x + B_zeta_mx - 2.f*B_zeta_0);
      float d2B_zeta_dy     = twelfth*(B_zeta_y + B_zeta_my - 2.f*B_zeta_0);
      float d2B_zeta_dz     = twelfth*(B_zeta_z + B_zeta_mz - 2.f*B_zeta_0);
      
      // Transform B field to Cartesian (equation 63)
      float Bx_cart, By_cart, Bz_cart;
      transform_B_to_cartesian(k_cmesh, pf0_index,
                               B_xi_interp, B_eta_interp, B_zeta_interp,
                               Bx_cart, By_cart, Bz_cart);
      
      // Store Cartesian B field and derivatives
      pi_cbx     = Bx_cart;
      pi_dcbxdx  = dB_xi_dx;   // Approximate - should be transformed
      pi_dcbxdy  = dB_xi_dy;
      pi_dcbxdz  = dB_xi_dz;
      pi_d2cbxdx = d2B_xi_dx;
      pi_d2cbxdy = d2B_xi_dy;
      pi_d2cbxdz = d2B_xi_dz;
      
      pi_cby     = By_cart;
      pi_dcbydx  = dB_eta_dx;
      pi_dcbydy  = dB_eta_dy;
      pi_dcbydz  = dB_eta_dz;
      pi_d2cbydx = d2B_eta_dx;
      pi_d2cbydy = d2B_eta_dy;
      pi_d2cbydz = d2B_eta_dz;
      
      pi_cbz     = Bz_cart;
      pi_dcbzdx  = dB_zeta_dx;
      pi_dcbzdy  = dB_zeta_dy;
      pi_dcbzdz  = dB_zeta_dz;
      pi_d2cbzdx = d2B_zeta_dx;
      pi_d2cbzdy = d2B_zeta_dy;
      pi_d2cbzdz = d2B_zeta_dz;

#ifdef EXTERNAL_FORCE
      // -------------------- EXTERNAL FORCES --------------------
      // Same pattern for Ex0, Ey0, Ez0, Gx0, Gy0, Gz0
      // (Assuming these are already in Cartesian or need similar transform)
      
      // Ex0 component
      auto Ex0_0  = k_field(pf0_index,  field_var::Ex0);
      auto Ex0_x  = k_field(pfx_index,  field_var::Ex0);
      auto Ex0_y  = k_field(pfy_index,  field_var::Ex0);
      auto Ex0_z  = k_field(pfz_index,  field_var::Ex0);
      auto Ex0_mx = k_field(pfmx_index, field_var::Ex0);
      auto Ex0_my = k_field(pfmy_index, field_var::Ex0);
      auto Ex0_mz = k_field(pfmz_index, field_var::Ex0);
      pi_Ex0     = twelfth*(6.f*Ex0_0 + Ex0_x + Ex0_y + Ex0_z + Ex0_mx + Ex0_my + Ex0_mz);
      pi_dEx0dx  = sixth*(Ex0_x - Ex0_mx);
      pi_dEx0dy  = sixth*(Ex0_y - Ex0_my);
      pi_dEx0dz  = sixth*(Ex0_z - Ex0_mz);
      pi_d2Ex0dx = twelfth*(Ex0_x + Ex0_mx - 2.f*Ex0_0);
      pi_d2Ex0dy = twelfth*(Ex0_y + Ex0_my - 2.f*Ex0_0);
      pi_d2Ex0dz = twelfth*(Ex0_z + Ex0_mz - 2.f*Ex0_0);
      
      // Ey0 component
      auto Ey0_0  = k_field(pf0_index,  field_var::Ey0);
      auto Ey0_x  = k_field(pfx_index,  field_var::Ey0);
      auto Ey0_y  = k_field(pfy_index,  field_var::Ey0);
      auto Ey0_z  = k_field(pfz_index,  field_var::Ey0);
      auto Ey0_mx = k_field(pfmx_index, field_var::Ey0);
      auto Ey0_my = k_field(pfmy_index, field_var::Ey0);
      auto Ey0_mz = k_field(pfmz_index, field_var::Ey0);
      pi_Ey0     = twelfth*(6.f*Ey0_0 + Ey0_x + Ey0_y + Ey0_z + Ey0_mx + Ey0_my + Ey0_mz);
      pi_dEy0dx  = sixth*(Ey0_x - Ey0_mx);
      pi_dEy0dy  = sixth*(Ey0_y - Ey0_my);
      pi_dEy0dz  = sixth*(Ey0_z - Ey0_mz);
      pi_d2Ey0dx = twelfth*(Ey0_x + Ey0_mx - 2.f*Ey0_0);
      pi_d2Ey0dy = twelfth*(Ey0_y + Ey0_my - 2.f*Ey0_0);
      pi_d2Ey0dz = twelfth*(Ey0_z + Ey0_mz - 2.f*Ey0_0);
      
      // Ez0 component
      auto Ez0_0  = k_field(pf0_index,  field_var::Ez0);
      auto Ez0_x  = k_field(pfx_index,  field_var::Ez0);
      auto Ez0_y  = k_field(pfy_index,  field_var::Ez0);
      auto Ez0_z  = k_field(pfz_index,  field_var::Ez0);
      auto Ez0_mx = k_field(pfmx_index, field_var::Ez0);
      auto Ez0_my = k_field(pfmy_index, field_var::Ez0);
      auto Ez0_mz = k_field(pfmz_index, field_var::Ez0);
      pi_Ez0     = twelfth*(6.f*Ez0_0 + Ez0_x + Ez0_y + Ez0_z + Ez0_mx + Ez0_my + Ez0_mz);
      pi_dEz0dx  = sixth*(Ez0_x - Ez0_mx);
      pi_dEz0dy  = sixth*(Ez0_y - Ez0_my);
      pi_dEz0dz  = sixth*(Ez0_z - Ez0_mz);
      pi_d2Ez0dx = twelfth*(Ez0_x + Ez0_mx - 2.f*Ez0_0);
      pi_d2Ez0dy = twelfth*(Ez0_y + Ez0_my - 2.f*Ez0_0);
      pi_d2Ez0dz = twelfth*(Ez0_z + Ez0_mz - 2.f*Ez0_0);
            // Gx0 component
      auto Gx0_0  = k_field(pf0_index,  field_var::Gx0);
      auto Gx0_x  = k_field(pfx_index,  field_var::Gx0);
      auto Gx0_y  = k_field(pfy_index,  field_var::Gx0);
      auto Gx0_z  = k_field(pfz_index,  field_var::Gx0);
      auto Gx0_mx = k_field(pfmx_index, field_var::Gx0);
      auto Gx0_my = k_field(pfmy_index, field_var::Gx0);
      auto Gx0_mz = k_field(pfmz_index, field_var::Gx0);
      pi_Gx0     = twelfth*(6.f*Gx0_0 + Gx0_x + Gx0_y + Gx0_z + Gx0_mx + Gx0_my + Gx0_mz);
      pi_dGx0dx  = sixth*(Gx0_x - Gx0_mx);
      pi_dGx0dy  = sixth*(Gx0_y - Gx0_my);
      pi_dGx0dz  = sixth*(Gx0_z - Gx0_mz);
      pi_d2Gx0dx = twelfth*(Gx0_x + Gx0_mx - 2.f*Gx0_0);
      pi_d2Gx0dy = twelfth*(Gx0_y + Gx0_my - 2.f*Gx0_0);
      pi_d2Gx0dz = twelfth*(Gx0_z + Gx0_mz - 2.f*Gx0_0);
      
      // Gy0 component
      auto Gy0_0  = k_field(pf0_index,  field_var::Gy0);
      auto Gy0_x  = k_field(pfx_index,  field_var::Gy0);
      auto Gy0_y  = k_field(pfy_index,  field_var::Gy0);
      auto Gy0_z  = k_field(pfz_index,  field_var::Gy0);
      auto Gy0_mx = k_field(pfmx_index, field_var::Gy0);
      auto Gy0_my = k_field(pfmy_index, field_var::Gy0);
      auto Gy0_mz = k_field(pfmz_index, field_var::Gy0);
      pi_Gy0     = twelfth*(6.f*Gy0_0 + Gy0_x + Gy0_y + Gy0_z + Gy0_mx + Gy0_my + Gy0_mz);
      pi_dGy0dx  = sixth*(Gy0_x - Gy0_mx);
      pi_dGy0dy  = sixth*(Gy0_y - Gy0_my);
      pi_dGy0dz  = sixth*(Gy0_z - Gy0_mz);
      pi_d2Gy0dx = twelfth*(Gy0_x + Gy0_mx - 2.f*Gy0_0);
      pi_d2Gy0dy = twelfth*(Gy0_y + Gy0_my - 2.f*Gy0_0);
      pi_d2Gy0dz = twelfth*(Gy0_z + Gy0_mz - 2.f*Gy0_0);
      
      // Gz0 component
      auto Gz0_0  = k_field(pf0_index,  field_var::Gz0);
      auto Gz0_x  = k_field(pfx_index,  field_var::Gz0);
      auto Gz0_y  = k_field(pfy_index,  field_var::Gz0);
      auto Gz0_z  = k_field(pfz_index,  field_var::Gz0);
      auto Gz0_mx = k_field(pfmx_index, field_var::Gz0);
      auto Gz0_my = k_field(pfmy_index, field_var::Gz0);
      auto Gz0_mz = k_field(pfmz_index, field_var::Gz0);
      pi_Gz0     = twelfth*(6.f*Gz0_0 + Gz0_x + Gz0_y + Gz0_z + Gz0_mx + Gz0_my + Gz0_mz);
      pi_dGz0dx  = sixth*(Gz0_x - Gz0_mx);
      pi_dGz0dy  = sixth*(Gz0_y - Gz0_my);
      pi_dGz0dz  = sixth*(Gz0_z - Gz0_mz);
      pi_d2Gz0dx = twelfth*(Gz0_x + Gz0_mx - 2.f*Gz0_0);
      pi_d2Gz0dy = twelfth*(Gz0_y + Gz0_my - 2.f*Gz0_0);
      pi_d2Gz0dz = twelfth*(Gz0_z + Gz0_mz - 2.f*Gz0_0);
#endif  // EXTERNAL_FORCE

#endif  // SHAPE_QS

  }); // end Kokkos::parallel_for("load interpolator")

  // ============================================================================
  // CLEANUP MACROS
  // ============================================================================
  
  #undef pi_ex
  #undef pi_dexdx
  #undef pi_dexdy
  #undef pi_dexdz
  #undef pi_d2exdx
  #undef pi_d2exdy
  #undef pi_d2exdz
  #undef pi_ey
  #undef pi_deydx
  #undef pi_deydy
  #undef pi_deydz
  #undef pi_d2eydx
  #undef pi_d2eydy
  #undef pi_d2eydz
  #undef pi_ez
  #undef pi_dezdx
  #undef pi_dezdy
  #undef pi_dezdz
  #undef pi_d2ezdx
  #undef pi_d2ezdy
  #undef pi_d2ezdz
  #undef pi_cbx
  #undef pi_dcbxdx
  #undef pi_dcbxdy
  #undef pi_dcbxdz
  #undef pi_d2cbxdx
  #undef pi_d2cbxdy
  #undef pi_d2cbxdz
  #undef pi_cby
  #undef pi_dcbydx
  #undef pi_dcbydy
  #undef pi_dcbydz
  #undef pi_d2cbydx
  #undef pi_d2cbydy
  #undef pi_d2cbydz
  #undef pi_cbz
  #undef pi_dcbzdx
  #undef pi_dcbzdy
  #undef pi_dcbzdz
  #undef pi_d2cbzdx
  #undef pi_d2cbzdy
  #undef pi_d2cbzdz

  #undef pi_Ex0
  #undef pi_dEx0dx
  #undef pi_dEx0dy
  #undef pi_dEx0dz
  #undef pi_d2Ex0dx
  #undef pi_d2Ex0dy
  #undef pi_d2Ex0dz
  #undef pi_Ey0
  #undef pi_dEy0dx
  #undef pi_dEy0dy
  #undef pi_dEy0dz
  #undef pi_d2Ey0dx
  #undef pi_d2Ey0dy
  #undef pi_d2Ey0dz
  #undef pi_Ez0
  #undef pi_dEz0dx
  #undef pi_dEz0dy
  #undef pi_dEz0dz
  #undef pi_d2Ez0dx
  #undef pi_d2Ez0dy
  #undef pi_d2Ez0dz

  #undef pi_Gx0
  #undef pi_dGx0dx
  #undef pi_dGx0dy
  #undef pi_dGx0dz
  #undef pi_d2Gx0dx
  #undef pi_d2Gx0dy
  #undef pi_d2Gx0dz
  #undef pi_Gy0
  #undef pi_dGy0dx
  #undef pi_dGy0dy
  #undef pi_dGy0dz
  #undef pi_d2Gy0dx
  #undef pi_d2Gy0dy
  #undef pi_d2Gy0dz
  #undef pi_Gz0
  #undef pi_dGz0dx
  #undef pi_dGz0dy
  #undef pi_dGz0dz
  #undef pi_d2Gz0dx
  #undef pi_d2Gz0dy
  #undef pi_d2Gz0dz
  
} // end load_interpolator_array_kokkos

void
load_interpolator_array( /**/  interpolator_array_t * RESTRICT ia,
                         const field_array_t        * RESTRICT fa ) {

  if( !ia || !fa || ia->g!=fa->g ) ERROR(( "Bad args" ));

  k_interpolator_t k_interp = ia->k_i_d;
  k_field_t         k_field  = fa->k_f_d;
  grid_t *g = fa->g;
  int nx = g->nx;
  int ny = g->ny;
  int nz = g->nz;

  load_interpolator_array_kokkos(k_interp, k_field, g->k_curvilinear_mesh_d, nx,ny,nz);
}

void
interpolator_array_t::copy_to_host() {

  if(k_i_h.span() < k_i_d.span())
    Kokkos::resize(k_i_h, k_i_d.extent(0));
  Kokkos::deep_copy(k_i_h, k_i_d);

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  // Avoid capturing this
  auto& host_interp = this->i;
  auto& k_interpolator_h = k_i_h;

  Kokkos::parallel_for("Copy interpolators to host",
    host_execution_policy(0, g->nv) ,
    KOKKOS_LAMBDA (int i) {
#ifdef SHAPE_NGP
      host_interp[i].ex       = k_interpolator_h(i, interpolator_var::ex);
      host_interp[i].ey       = k_interpolator_h(i, interpolator_var::ey);
      host_interp[i].ez       = k_interpolator_h(i, interpolator_var::ez);
      host_interp[i].cbx      = k_interpolator_h(i, interpolator_var::cbx);
      host_interp[i].cby      = k_interpolator_h(i, interpolator_var::cby);
      host_interp[i].cbz      = k_interpolator_h(i, interpolator_var::cbz);
  #ifdef EXTERNAL_FORCE
      host_interp[i].Ex0      = k_interpolator_h(i, interpolator_var::Ex0);
      host_interp[i].Ey0      = k_interpolator_h(i, interpolator_var::Ey0);
      host_interp[i].Ez0      = k_interpolator_h(i, interpolator_var::Ez0);
      host_interp[i].Gx0      = k_interpolator_h(i, interpolator_var::Gx0);
      host_interp[i].Gy0      = k_interpolator_h(i, interpolator_var::Gy0);
      host_interp[i].Gz0      = k_interpolator_h(i, interpolator_var::Gz0);
  #endif
#elif defined( SHAPE_QS )
      host_interp[i].ex      = k_interpolator_h(i, interpolator_var::ex     );
      host_interp[i].dexdx   = k_interpolator_h(i, interpolator_var::dexdx  );
      host_interp[i].dexdy   = k_interpolator_h(i, interpolator_var::dexdy  );
      host_interp[i].dexdz   = k_interpolator_h(i, interpolator_var::dexdz  );
      host_interp[i].d2exdx  = k_interpolator_h(i, interpolator_var::d2exdx );
      host_interp[i].d2exdy  = k_interpolator_h(i, interpolator_var::d2exdy );
      host_interp[i].d2exdz  = k_interpolator_h(i, interpolator_var::d2exdz );
      host_interp[i].ey      = k_interpolator_h(i, interpolator_var::ey     );
      host_interp[i].deydx   = k_interpolator_h(i, interpolator_var::deydx  );
      host_interp[i].deydy   = k_interpolator_h(i, interpolator_var::deydy  );
      host_interp[i].deydz   = k_interpolator_h(i, interpolator_var::deydz  );
      host_interp[i].d2eydx  = k_interpolator_h(i, interpolator_var::d2eydx );
      host_interp[i].d2eydy  = k_interpolator_h(i, interpolator_var::d2eydy );
      host_interp[i].d2eydz  = k_interpolator_h(i, interpolator_var::d2eydz );
      host_interp[i].ez      = k_interpolator_h(i, interpolator_var::ez     );
      host_interp[i].dezdx   = k_interpolator_h(i, interpolator_var::dezdx  );
      host_interp[i].dezdy   = k_interpolator_h(i, interpolator_var::dezdy  );
      host_interp[i].dezdz   = k_interpolator_h(i, interpolator_var::dezdz  );
      host_interp[i].d2ezdx  = k_interpolator_h(i, interpolator_var::d2ezdx );
      host_interp[i].d2ezdy  = k_interpolator_h(i, interpolator_var::d2ezdy );
      host_interp[i].d2ezdz  = k_interpolator_h(i, interpolator_var::d2ezdz );
      host_interp[i].cbx     = k_interpolator_h(i, interpolator_var::cbx    );
      host_interp[i].dcbxdx  = k_interpolator_h(i, interpolator_var::dcbxdx );
      host_interp[i].dcbxdy  = k_interpolator_h(i, interpolator_var::dcbxdy );
      host_interp[i].dcbxdz  = k_interpolator_h(i, interpolator_var::dcbxdz );
      host_interp[i].d2cbxdx = k_interpolator_h(i, interpolator_var::d2cbxdx);
      host_interp[i].d2cbxdy = k_interpolator_h(i, interpolator_var::d2cbxdy);
      host_interp[i].d2cbxdz = k_interpolator_h(i, interpolator_var::d2cbxdz);
      host_interp[i].cby     = k_interpolator_h(i, interpolator_var::cby    );
      host_interp[i].dcbydx  = k_interpolator_h(i, interpolator_var::dcbydx );
      host_interp[i].dcbydy  = k_interpolator_h(i, interpolator_var::dcbydy );
      host_interp[i].dcbydz  = k_interpolator_h(i, interpolator_var::dcbydz );
      host_interp[i].d2cbydx = k_interpolator_h(i, interpolator_var::d2cbydx);
      host_interp[i].d2cbydy = k_interpolator_h(i, interpolator_var::d2cbydy);
      host_interp[i].d2cbydz = k_interpolator_h(i, interpolator_var::d2cbydz);
      host_interp[i].cbz     = k_interpolator_h(i, interpolator_var::cbz    );
      host_interp[i].dcbzdx  = k_interpolator_h(i, interpolator_var::dcbzdx );
      host_interp[i].dcbzdy  = k_interpolator_h(i, interpolator_var::dcbzdy );
      host_interp[i].dcbzdz  = k_interpolator_h(i, interpolator_var::dcbzdz );
      host_interp[i].d2cbzdx = k_interpolator_h(i, interpolator_var::d2cbzdx);
      host_interp[i].d2cbzdy = k_interpolator_h(i, interpolator_var::d2cbzdy);
      host_interp[i].d2cbzdz = k_interpolator_h(i, interpolator_var::d2cbzdz);

  #ifdef EXTERNAL_FORCE
      host_interp[i].Ex0      = k_interpolator_h(i, interpolator_var::Ex0     );
      host_interp[i].dEx0dx   = k_interpolator_h(i, interpolator_var::dEx0dx  );
      host_interp[i].dEx0dy   = k_interpolator_h(i, interpolator_var::dEx0dy  );
      host_interp[i].dEx0dz   = k_interpolator_h(i, interpolator_var::dEx0dz  );
      host_interp[i].d2Ex0dx  = k_interpolator_h(i, interpolator_var::d2Ex0dx );
      host_interp[i].d2Ex0dy  = k_interpolator_h(i, interpolator_var::d2Ex0dy );
      host_interp[i].d2Ex0dz  = k_interpolator_h(i, interpolator_var::d2Ex0dz );
      host_interp[i].Ey0      = k_interpolator_h(i, interpolator_var::Ey0     );
      host_interp[i].dEy0dx   = k_interpolator_h(i, interpolator_var::dEy0dx  );
      host_interp[i].dEy0dy   = k_interpolator_h(i, interpolator_var::dEy0dy  );
      host_interp[i].dEy0dz   = k_interpolator_h(i, interpolator_var::dEy0dz  );
      host_interp[i].d2Ey0dx  = k_interpolator_h(i, interpolator_var::d2Ey0dx );
      host_interp[i].d2Ey0dy  = k_interpolator_h(i, interpolator_var::d2Ey0dy );
      host_interp[i].d2Ey0dz  = k_interpolator_h(i, interpolator_var::d2Ey0dz );
      host_interp[i].Ez0      = k_interpolator_h(i, interpolator_var::Ez0     );
      host_interp[i].dEz0dx   = k_interpolator_h(i, interpolator_var::dEz0dx  );
      host_interp[i].dEz0dy   = k_interpolator_h(i, interpolator_var::dEz0dy  );
      host_interp[i].dEz0dz   = k_interpolator_h(i, interpolator_var::dEz0dz  );
      host_interp[i].d2Ez0dx  = k_interpolator_h(i, interpolator_var::d2Ez0dx );
      host_interp[i].d2Ez0dy  = k_interpolator_h(i, interpolator_var::d2Ez0dy );
      host_interp[i].d2Ez0dz  = k_interpolator_h(i, interpolator_var::d2Ez0dz );

      host_interp[i].Gx0      = k_interpolator_h(i, interpolator_var::Gx0     );
      host_interp[i].dGx0dx   = k_interpolator_h(i, interpolator_var::dGx0dx  );
      host_interp[i].dGx0dy   = k_interpolator_h(i, interpolator_var::dGx0dy  );
      host_interp[i].dGx0dz   = k_interpolator_h(i, interpolator_var::dGx0dz  );
      host_interp[i].d2Gx0dx  = k_interpolator_h(i, interpolator_var::d2Gx0dx );
      host_interp[i].d2Gx0dy  = k_interpolator_h(i, interpolator_var::d2Gx0dy );
      host_interp[i].d2Gx0dz  = k_interpolator_h(i, interpolator_var::d2Gx0dz );
      host_interp[i].Gy0      = k_interpolator_h(i, interpolator_var::Gy0     );
      host_interp[i].dGy0dx   = k_interpolator_h(i, interpolator_var::dGy0dx  );
      host_interp[i].dGy0dy   = k_interpolator_h(i, interpolator_var::dGy0dy  );
      host_interp[i].dGy0dz   = k_interpolator_h(i, interpolator_var::dGy0dz  );
      host_interp[i].d2Gy0dx  = k_interpolator_h(i, interpolator_var::d2Gy0dx );
      host_interp[i].d2Gy0dy  = k_interpolator_h(i, interpolator_var::d2Gy0dy );
      host_interp[i].d2Gy0dz  = k_interpolator_h(i, interpolator_var::d2Gy0dz );
      host_interp[i].Gz0      = k_interpolator_h(i, interpolator_var::Gz0     );
      host_interp[i].dGz0dx   = k_interpolator_h(i, interpolator_var::dGz0dx  );
      host_interp[i].dGz0dy   = k_interpolator_h(i, interpolator_var::dGz0dy  );
      host_interp[i].dGz0dz   = k_interpolator_h(i, interpolator_var::dGz0dz  );
      host_interp[i].d2Gz0dx  = k_interpolator_h(i, interpolator_var::d2Gz0dx );
      host_interp[i].d2Gz0dy  = k_interpolator_h(i, interpolator_var::d2Gz0dy );
      host_interp[i].d2Gz0dz  = k_interpolator_h(i, interpolator_var::d2Gz0dz );
  #endif
#endif
    });
#endif
}

void
interpolator_array_t::copy_to_device() {

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  // Avoid capturing this
  auto& host_interp = this->i;
  auto& k_interpolator_h = k_i_h;

  Kokkos::parallel_for("Copy interpolators to device",
    host_execution_policy(0, g->nv) ,
    KOKKOS_LAMBDA (int i) {
#ifdef SHAPE_NGP
      k_interpolator_h(i, interpolator_var::ex)       = host_interp[i].ex;
      k_interpolator_h(i, interpolator_var::ey)       = host_interp[i].ey;
      k_interpolator_h(i, interpolator_var::ez)       = host_interp[i].ez;
      k_interpolator_h(i, interpolator_var::cbx)      = host_interp[i].cbx;
      k_interpolator_h(i, interpolator_var::cby)      = host_interp[i].cby;
      k_interpolator_h(i, interpolator_var::cbz)      = host_interp[i].cbz;
  #ifdef EXTERNAL_FORCE
      k_interpolator_h(i, interpolator_var::Ex0)       = host_interp[i].Ex0;
      k_interpolator_h(i, interpolator_var::Ey0)       = host_interp[i].Ey0;
      k_interpolator_h(i, interpolator_var::Ez0)       = host_interp[i].Ez0;
      k_interpolator_h(i, interpolator_var::Gx0)       = host_interp[i].Gx0;
      k_interpolator_h(i, interpolator_var::Gy0)       = host_interp[i].Gy0;
      k_interpolator_h(i, interpolator_var::Gz0)       = host_interp[i].Gz0;
  #endif
#elif defined( SHAPE_QS )
      k_interpolator_h(i, interpolator_var::ex      ) = host_interp[i].ex     ;
      k_interpolator_h(i, interpolator_var::dexdx   ) = host_interp[i].dexdx  ;
      k_interpolator_h(i, interpolator_var::dexdy   ) = host_interp[i].dexdy  ;
      k_interpolator_h(i, interpolator_var::dexdz   ) = host_interp[i].dexdz  ;
      k_interpolator_h(i, interpolator_var::d2exdx  ) = host_interp[i].d2exdx ;
      k_interpolator_h(i, interpolator_var::d2exdy  ) = host_interp[i].d2exdy ;
      k_interpolator_h(i, interpolator_var::d2exdz  ) = host_interp[i].d2exdz ;
      k_interpolator_h(i, interpolator_var::ey      ) = host_interp[i].ey     ;
      k_interpolator_h(i, interpolator_var::deydx   ) = host_interp[i].deydx  ;
      k_interpolator_h(i, interpolator_var::deydy   ) = host_interp[i].deydy  ;
      k_interpolator_h(i, interpolator_var::deydz   ) = host_interp[i].deydz  ;
      k_interpolator_h(i, interpolator_var::d2eydx  ) = host_interp[i].d2eydx ;
      k_interpolator_h(i, interpolator_var::d2eydy  ) = host_interp[i].d2eydy ;
      k_interpolator_h(i, interpolator_var::d2eydz  ) = host_interp[i].d2eydz ;
      k_interpolator_h(i, interpolator_var::ez      ) = host_interp[i].ez     ;
      k_interpolator_h(i, interpolator_var::dezdx   ) = host_interp[i].dezdx  ;
      k_interpolator_h(i, interpolator_var::dezdy   ) = host_interp[i].dezdy  ;
      k_interpolator_h(i, interpolator_var::dezdz   ) = host_interp[i].dezdz  ;
      k_interpolator_h(i, interpolator_var::d2ezdx  ) = host_interp[i].d2ezdx ;
      k_interpolator_h(i, interpolator_var::d2ezdy  ) = host_interp[i].d2ezdy ;
      k_interpolator_h(i, interpolator_var::d2ezdz  ) = host_interp[i].d2ezdz ;
      k_interpolator_h(i, interpolator_var::cbx     ) = host_interp[i].cbx    ;
      k_interpolator_h(i, interpolator_var::dcbxdx  ) = host_interp[i].dcbxdx ;
      k_interpolator_h(i, interpolator_var::dcbxdy  ) = host_interp[i].dcbxdy ;
      k_interpolator_h(i, interpolator_var::dcbxdz  ) = host_interp[i].dcbxdz ;
      k_interpolator_h(i, interpolator_var::d2cbxdx ) = host_interp[i].d2cbxdx;
      k_interpolator_h(i, interpolator_var::d2cbxdy ) = host_interp[i].d2cbxdy;
      k_interpolator_h(i, interpolator_var::d2cbxdz ) = host_interp[i].d2cbxdz;
      k_interpolator_h(i, interpolator_var::cby     ) = host_interp[i].cby    ;
      k_interpolator_h(i, interpolator_var::dcbydx  ) = host_interp[i].dcbydx ;
      k_interpolator_h(i, interpolator_var::dcbydy  ) = host_interp[i].dcbydy ;
      k_interpolator_h(i, interpolator_var::dcbydz  ) = host_interp[i].dcbydz ;
      k_interpolator_h(i, interpolator_var::d2cbydx ) = host_interp[i].d2cbydx;
      k_interpolator_h(i, interpolator_var::d2cbydy ) = host_interp[i].d2cbydy;
      k_interpolator_h(i, interpolator_var::d2cbydz ) = host_interp[i].d2cbydz;
      k_interpolator_h(i, interpolator_var::cbz     ) = host_interp[i].cbz    ;
      k_interpolator_h(i, interpolator_var::dcbzdx  ) = host_interp[i].dcbzdx ;
      k_interpolator_h(i, interpolator_var::dcbzdy  ) = host_interp[i].dcbzdy ;
      k_interpolator_h(i, interpolator_var::dcbzdz  ) = host_interp[i].dcbzdz ;
      k_interpolator_h(i, interpolator_var::d2cbzdx ) = host_interp[i].d2cbzdx;
      k_interpolator_h(i, interpolator_var::d2cbzdy ) = host_interp[i].d2cbzdy;
      k_interpolator_h(i, interpolator_var::d2cbzdz ) = host_interp[i].d2cbzdz;

  #ifdef EXTERNAL_FORCE
      k_interpolator_h(i, interpolator_var::Ex0     ) = host_interp[i].Ex0    ;
      k_interpolator_h(i, interpolator_var::dEx0dx  ) = host_interp[i].dEx0dx ;
      k_interpolator_h(i, interpolator_var::dEx0dy  ) = host_interp[i].dEx0dy ;
      k_interpolator_h(i, interpolator_var::dEx0dz  ) = host_interp[i].dEx0dz ;
      k_interpolator_h(i, interpolator_var::d2Ex0dx ) = host_interp[i].d2Ex0dx;
      k_interpolator_h(i, interpolator_var::d2Ex0dy ) = host_interp[i].d2Ex0dy;
      k_interpolator_h(i, interpolator_var::d2Ex0dz ) = host_interp[i].d2Ex0dz;
      k_interpolator_h(i, interpolator_var::Ey0     ) = host_interp[i].Ey0    ;
      k_interpolator_h(i, interpolator_var::dEy0dx  ) = host_interp[i].dEy0dx ;
      k_interpolator_h(i, interpolator_var::dEy0dy  ) = host_interp[i].dEy0dy ;
      k_interpolator_h(i, interpolator_var::dEy0dz  ) = host_interp[i].dEy0dz ;
      k_interpolator_h(i, interpolator_var::d2Ey0dx ) = host_interp[i].d2Ey0dx;
      k_interpolator_h(i, interpolator_var::d2Ey0dy ) = host_interp[i].d2Ey0dy;
      k_interpolator_h(i, interpolator_var::d2Ey0dz ) = host_interp[i].d2Ey0dz;
      k_interpolator_h(i, interpolator_var::Ez0     ) = host_interp[i].Ez0    ;
      k_interpolator_h(i, interpolator_var::dEz0dx  ) = host_interp[i].dEz0dx ;
      k_interpolator_h(i, interpolator_var::dEz0dy  ) = host_interp[i].dEz0dy ;
      k_interpolator_h(i, interpolator_var::dEz0dz  ) = host_interp[i].dEz0dz ;
      k_interpolator_h(i, interpolator_var::d2Ez0dx ) = host_interp[i].d2Ez0dx;
      k_interpolator_h(i, interpolator_var::d2Ez0dy ) = host_interp[i].d2Ez0dy;
      k_interpolator_h(i, interpolator_var::d2Ez0dz ) = host_interp[i].d2Ez0dz;

      k_interpolator_h(i, interpolator_var::Gx0     ) = host_interp[i].Gx0    ;
      k_interpolator_h(i, interpolator_var::dGx0dx  ) = host_interp[i].dGx0dx ;
      k_interpolator_h(i, interpolator_var::dGx0dy  ) = host_interp[i].dGx0dy ;
      k_interpolator_h(i, interpolator_var::dGx0dz  ) = host_interp[i].dGx0dz ;
      k_interpolator_h(i, interpolator_var::d2Gx0dx ) = host_interp[i].d2Gx0dx;
      k_interpolator_h(i, interpolator_var::d2Gx0dy ) = host_interp[i].d2Gx0dy;
      k_interpolator_h(i, interpolator_var::d2Gx0dz ) = host_interp[i].d2Gx0dz;
      k_interpolator_h(i, interpolator_var::Gy0     ) = host_interp[i].Gy0    ;
      k_interpolator_h(i, interpolator_var::dGy0dx  ) = host_interp[i].dGy0dx ;
      k_interpolator_h(i, interpolator_var::dGy0dy  ) = host_interp[i].dGy0dy ;
      k_interpolator_h(i, interpolator_var::dGy0dz  ) = host_interp[i].dGy0dz ;
      k_interpolator_h(i, interpolator_var::d2Gy0dx ) = host_interp[i].d2Gy0dx;
      k_interpolator_h(i, interpolator_var::d2Gy0dy ) = host_interp[i].d2Gy0dy;
      k_interpolator_h(i, interpolator_var::d2Gy0dz ) = host_interp[i].d2Gy0dz;
      k_interpolator_h(i, interpolator_var::Gz0     ) = host_interp[i].Gz0    ;
      k_interpolator_h(i, interpolator_var::dGz0dx  ) = host_interp[i].dGz0dx ;
      k_interpolator_h(i, interpolator_var::dGz0dy  ) = host_interp[i].dGz0dy ;
      k_interpolator_h(i, interpolator_var::dGz0dz  ) = host_interp[i].dGz0dz ;
      k_interpolator_h(i, interpolator_var::d2Gz0dx ) = host_interp[i].d2Gz0dx;
      k_interpolator_h(i, interpolator_var::d2Gz0dy ) = host_interp[i].d2Gz0dy;
      k_interpolator_h(i, interpolator_var::d2Gz0dz ) = host_interp[i].d2Gz0dz;
  #endif
#endif
    });
#endif

  if(k_i_d.span() < k_i_h.span())
    Kokkos::resize(k_i_d, k_i_h.extent(0));
  Kokkos::deep_copy(k_i_d, k_i_h);

}
