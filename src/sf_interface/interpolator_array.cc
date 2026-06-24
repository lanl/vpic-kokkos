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

void 
load_interpolator_array_kokkos(k_interpolator_t& k_interp, k_field_t& k_field, 
                               const int nx, const int ny, const int nz) {

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
  Kokkos::parallel_for("load interpolator", load_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
      const int pi_index   = VOXEL(x,   y,   z,   nx,ny,nz); 
      const int pf0_index  = VOXEL(x,   y,   z,   nx,ny,nz); 
      const int pfx_index  = VOXEL(x+1, y,   z,   nx,ny,nz); 
      const int pfy_index  = VOXEL(x,   y+1, z,   nx,ny,nz); 
      const int pfz_index  = VOXEL(x,   y,   z+1, nx,ny,nz); 
      const int pfmx_index = VOXEL(x-1, y,   z,   nx,ny,nz);
      const int pfmy_index = VOXEL(x,   y-1, z,   nx,ny,nz);
      const int pfmz_index = VOXEL(x,   y,   z-1, nx,ny,nz);

#ifdef SHAPE_NGP
      // ex interpolation coefficients
      // ex->tx from hyb_smooth_eb(...)
      pi_ex = k_field(pf0_index, field_var::tx);
      // ey interpolation coefficients
      // ey->ty from hyb_smooth_eb(...)
      pi_ey = k_field(pf0_index, field_var::ty);
      // ez interpolation coefficients
      // ez->tz from hyb_smooth_eb(...)
      pi_ez = k_field(pf0_index, field_var::tz);

      // bx interpolation coefficients
      // cbx->ox from hyb_smooth_eb(...)
      pi_cbx = k_field(pf0_index, field_var::ox) + k_field(pf0_index, field_var::cbx0);
      // by interpolation coefficients
      // cby->oy from hyb_smooth_eb(...)
      pi_cby = k_field(pf0_index, field_var::oy) + k_field(pf0_index, field_var::cby0);
      // bz interpolation coefficients
      // cbz->oz from hyb_smooth_eb(...)
      pi_cbz = k_field(pf0_index, field_var::oz) + k_field(pf0_index, field_var::cbz0);
    
#ifdef EXTERNAL_FORCE
      // Ex0 interpolation coefficients
      pi_Ex0 = k_field(pf0_index,  field_var::Ex0);
      // Ey0 interpolation coefficients
      pi_Ey0 = k_field(pf0_index,  field_var::Ey0);
      // Ez0 interpolation coefficients
      pi_Ez0 = k_field(pf0_index,  field_var::Ez0);
      
      // Gx0 interpolation coefficients
      pi_Gx0 = k_field(pf0_index,  field_var::Gx0);
      // Gy0 interpolation coefficients
      pi_Gy0 = k_field(pf0_index,  field_var::Gy0);
      // Gz0 interpolation coefficients
      pi_Gz0 = k_field(pf0_index,  field_var::Gz0);
#endif
#elif defined( SHAPE_QS )
      // ex interpolation coefficients
      // ex->tx from hyb_smooth_eb(...)
      auto w0  = k_field(pf0_index,  field_var::tx);
      auto wx  = k_field(pfx_index,  field_var::tx);
      auto wy  = k_field(pfy_index,  field_var::tx);
      auto wz  = k_field(pfz_index,  field_var::tx);
      auto wmx = k_field(pfmx_index, field_var::tx);
      auto wmy = k_field(pfmy_index, field_var::tx);
      auto wmz = k_field(pfmz_index, field_var::tx);
      pi_ex     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dexdx  = sixth*(wx - wmx);
      pi_dexdy  = sixth*(wy - wmy);
      pi_dexdz  = sixth*(wz - wmz);
      pi_d2exdx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2exdy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2exdz = twelfth*(wz + wmz - 2.f*w0);
      // ey interpolation coefficients
      // ey->ty from hyb_smooth_eb(...)
      w0  = k_field(pf0_index,  field_var::ty);
      wx  = k_field(pfx_index,  field_var::ty);
      wy  = k_field(pfy_index,  field_var::ty);
      wz  = k_field(pfz_index,  field_var::ty);
      wmx = k_field(pfmx_index, field_var::ty);
      wmy = k_field(pfmy_index, field_var::ty);
      wmz = k_field(pfmz_index, field_var::ty);
      pi_ey     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_deydx  = sixth*(wx - wmx);
      pi_deydy  = sixth*(wy - wmy);
      pi_deydz  = sixth*(wz - wmz);
      pi_d2eydx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2eydy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2eydz = twelfth*(wz + wmz - 2.f*w0);
      // ez interpolation coefficients
      // ez->tz from hyb_smooth_eb(...)
      w0  = k_field(pf0_index,  field_var::tz);
      wx  = k_field(pfx_index,  field_var::tz);
      wy  = k_field(pfy_index,  field_var::tz);
      wz  = k_field(pfz_index,  field_var::tz);
      wmx = k_field(pfmx_index, field_var::tz);
      wmy = k_field(pfmy_index, field_var::tz);
      wmz = k_field(pfmz_index, field_var::tz);
      pi_ez     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dezdx  = sixth*(wx - wmx);
      pi_dezdy  = sixth*(wy - wmy);
      pi_dezdz  = sixth*(wz - wmz);
      pi_d2ezdx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2ezdy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2ezdz = twelfth*(wz + wmz - 2.f*w0);

      // bx interpolation coefficients
      // cbx->ox from hyb_smooth_eb(...)
      w0  = k_field(pf0_index,  field_var::ox) + k_field(pf0_index,  field_var::cbx0);
      wx  = k_field(pfx_index,  field_var::ox) + k_field(pfx_index,  field_var::cbx0);
      wy  = k_field(pfy_index,  field_var::ox) + k_field(pfy_index,  field_var::cbx0);
      wz  = k_field(pfz_index,  field_var::ox) + k_field(pfz_index,  field_var::cbx0);
      wmx = k_field(pfmx_index, field_var::ox) + k_field(pfmx_index, field_var::cbx0);
      wmy = k_field(pfmy_index, field_var::ox) + k_field(pfmy_index, field_var::cbx0);
      wmz = k_field(pfmz_index, field_var::ox) + k_field(pfmz_index, field_var::cbx0);
      pi_cbx     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dcbxdx  = sixth*(wx - wmx);
      pi_dcbxdy  = sixth*(wy - wmy);
      pi_dcbxdz  = sixth*(wz - wmz);
      pi_d2cbxdx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2cbxdy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2cbxdz = twelfth*(wz + wmz - 2.f*w0);
      // by interpolation coefficients
      // cby->oy from hyb_smooth_eb(...)
      w0  = k_field(pf0_index,  field_var::oy) + k_field(pf0_index,  field_var::cby0);
      wx  = k_field(pfx_index,  field_var::oy) + k_field(pfx_index,  field_var::cby0);
      wy  = k_field(pfy_index,  field_var::oy) + k_field(pfy_index,  field_var::cby0);
      wz  = k_field(pfz_index,  field_var::oy) + k_field(pfz_index,  field_var::cby0);
      wmx = k_field(pfmx_index, field_var::oy) + k_field(pfmx_index, field_var::cby0);
      wmy = k_field(pfmy_index, field_var::oy) + k_field(pfmy_index, field_var::cby0);
      wmz = k_field(pfmz_index, field_var::oy) + k_field(pfmz_index, field_var::cby0);
      pi_cby     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dcbydx  = sixth*(wx - wmx);
      pi_dcbydy  = sixth*(wy - wmy);
      pi_dcbydz  = sixth*(wz - wmz);
      pi_d2cbydx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2cbydy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2cbydz = twelfth*(wz + wmz - 2.f*w0);
      // bz interpolation coefficients
      // cbz->oz from hyb_smooth_eb(...)
      w0  = k_field(pf0_index,  field_var::oz) + k_field(pf0_index,  field_var::cbz0);
      wx  = k_field(pfx_index,  field_var::oz) + k_field(pfx_index,  field_var::cbz0);
      wy  = k_field(pfy_index,  field_var::oz) + k_field(pfy_index,  field_var::cbz0);
      wz  = k_field(pfz_index,  field_var::oz) + k_field(pfz_index,  field_var::cbz0);
      wmx = k_field(pfmx_index, field_var::oz) + k_field(pfmx_index, field_var::cbz0);
      wmy = k_field(pfmy_index, field_var::oz) + k_field(pfmy_index, field_var::cbz0);
      wmz = k_field(pfmz_index, field_var::oz) + k_field(pfmz_index, field_var::cbz0);
      pi_cbz     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dcbzdx  = sixth*(wx - wmx);
      pi_dcbzdy  = sixth*(wy - wmy);
      pi_dcbzdz  = sixth*(wz - wmz);
      pi_d2cbzdx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2cbzdy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2cbzdz = twelfth*(wz + wmz - 2.f*w0);

#ifdef EXTERNAL_FORCE
      // Ex0 interpolation coefficients
      w0  = k_field(pf0_index,  field_var::Ex0);
      wx  = k_field(pfx_index,  field_var::Ex0);
      wy  = k_field(pfy_index,  field_var::Ex0);
      wz  = k_field(pfz_index,  field_var::Ex0);
      wmx = k_field(pfmx_index, field_var::Ex0);
      wmy = k_field(pfmy_index, field_var::Ex0);
      wmz = k_field(pfmz_index, field_var::Ex0);
      pi_Ex0     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dEx0dx  = sixth*(wx - wmx);
      pi_dEx0dy  = sixth*(wy - wmy);
      pi_dEx0dz  = sixth*(wz - wmz);
      pi_d2Ex0dx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2Ex0dy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2Ex0dz = twelfth*(wz + wmz - 2.f*w0);
      // Ey0 interpolation coefficients
      w0  = k_field(pf0_index,  field_var::Ey0);
      wx  = k_field(pfx_index,  field_var::Ey0);
      wy  = k_field(pfy_index,  field_var::Ey0);
      wz  = k_field(pfz_index,  field_var::Ey0);
      wmx = k_field(pfmx_index, field_var::Ey0);
      wmy = k_field(pfmy_index, field_var::Ey0);
      wmz = k_field(pfmz_index, field_var::Ey0);
      pi_Ey0     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dEy0dx  = sixth*(wx - wmx);
      pi_dEy0dy  = sixth*(wy - wmy);
      pi_dEy0dz  = sixth*(wz - wmz);
      pi_d2Ey0dx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2Ey0dy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2Ey0dz = twelfth*(wz + wmz - 2.f*w0);
      // Ez0 interpolation coefficients
      w0  = k_field(pf0_index,  field_var::Ez0);
      wx  = k_field(pfx_index,  field_var::Ez0);
      wy  = k_field(pfy_index,  field_var::Ez0);
      wz  = k_field(pfz_index,  field_var::Ez0);
      wmx = k_field(pfmx_index, field_var::Ez0);
      wmy = k_field(pfmy_index, field_var::Ez0);
      wmz = k_field(pfmz_index, field_var::Ez0);
      pi_Ez0     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dEz0dx  = sixth*(wx - wmx);
      pi_dEz0dy  = sixth*(wy - wmy);
      pi_dEz0dz  = sixth*(wz - wmz);
      pi_d2Ez0dx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2Ez0dy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2Ez0dz = twelfth*(wz + wmz - 2.f*w0);
    
      // Gx0 interpolation coefficients
      w0  = k_field(pf0_index,  field_var::Gx0);
      wx  = k_field(pfx_index,  field_var::Gx0);
      wy  = k_field(pfy_index,  field_var::Gx0);
      wz  = k_field(pfz_index,  field_var::Gx0);
      wmx = k_field(pfmx_index, field_var::Gx0);
      wmy = k_field(pfmy_index, field_var::Gx0);
      wmz = k_field(pfmz_index, field_var::Gx0);
      pi_Gx0     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dGx0dx  = sixth*(wx - wmx);
      pi_dGx0dy  = sixth*(wy - wmy);
      pi_dGx0dz  = sixth*(wz - wmz);
      pi_d2Gx0dx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2Gx0dy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2Gx0dz = twelfth*(wz + wmz - 2.f*w0);
      // Gy0 interpolation coefficients
      w0  = k_field(pf0_index,  field_var::Gy0);
      wx  = k_field(pfx_index,  field_var::Gy0);
      wy  = k_field(pfy_index,  field_var::Gy0);
      wz  = k_field(pfz_index,  field_var::Gy0);
      wmx = k_field(pfmx_index, field_var::Gy0);
      wmy = k_field(pfmy_index, field_var::Gy0);
      wmz = k_field(pfmz_index, field_var::Gy0);
      pi_Gy0     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dGy0dx  = sixth*(wx - wmx);
      pi_dGy0dy  = sixth*(wy - wmy);
      pi_dGy0dz  = sixth*(wz - wmz);
      pi_d2Gy0dx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2Gy0dy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2Gy0dz = twelfth*(wz + wmz - 2.f*w0);
      // Gz0 interpolation coefficients
      w0  = k_field(pf0_index,  field_var::Gz0);
      wx  = k_field(pfx_index,  field_var::Gz0);
      wy  = k_field(pfy_index,  field_var::Gz0);
      wz  = k_field(pfz_index,  field_var::Gz0);
      wmx = k_field(pfmx_index, field_var::Gz0);
      wmy = k_field(pfmy_index, field_var::Gz0);
      wmz = k_field(pfmz_index, field_var::Gz0);
      pi_Gz0     = twelfth*(6.f*w0 + wx + wy + wz + wmx + wmy + wmz);
      pi_dGz0dx  = sixth*(wx - wmx);
      pi_dGz0dy  = sixth*(wy - wmy);
      pi_dGz0dz  = sixth*(wz - wmz);
      pi_d2Gz0dx = twelfth*(wx + wmx - 2.f*w0);
      pi_d2Gz0dy = twelfth*(wy + wmy - 2.f*w0);
      pi_d2Gz0dz = twelfth*(wz + wmz - 2.f*w0);
#endif
#endif
  }); // end Kokkos::parallel_for("load interpolator")

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
}

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

  load_interpolator_array_kokkos(k_interp, k_field, nx, ny, nz);
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
