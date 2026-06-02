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

void load_interpolator_array_kokkos(k_interpolator_t k_interp, k_field_t k_field, int nx, int ny, int nz) {

  #define pi_ex       k_interp(pi_index, interpolator_var::ex)
  #define pi_dexdy    k_interp(pi_index, interpolator_var::dexdy)
  #define pi_dexdz    k_interp(pi_index, interpolator_var::dexdz)
  #define pi_d2exdydz k_interp(pi_index, interpolator_var::d2exdydz)

  #define pi_ey       k_interp(pi_index, interpolator_var::ey)
  #define pi_deydz    k_interp(pi_index, interpolator_var::deydz)
  #define pi_deydx    k_interp(pi_index, interpolator_var::deydx)
  #define pi_d2eydzdx k_interp(pi_index, interpolator_var::d2eydzdx)

  #define pi_ez       k_interp(pi_index, interpolator_var::ez)
  #define pi_dezdx    k_interp(pi_index, interpolator_var::dezdx)
  #define pi_dezdy    k_interp(pi_index, interpolator_var::dezdy)
  #define pi_d2ezdxdy k_interp(pi_index, interpolator_var::d2ezdxdy)

  #define pi_cbx      k_interp(pi_index, interpolator_var::cbx)
  #define pi_dcbxdx   k_interp(pi_index, interpolator_var::dcbxdx)

  #define pi_cby      k_interp(pi_index, interpolator_var::cby)
  #define pi_dcbydy   k_interp(pi_index, interpolator_var::dcbydy)

  #define pi_cbz      k_interp(pi_index, interpolator_var::cbz)
  #define pi_dcbzdz   k_interp(pi_index, interpolator_var::dcbzdz)

  constexpr float fourth = 0.25;
  constexpr float half   = 0.5;

  Kokkos::MDRangePolicy<Kokkos::Rank<3>> load_policy({1, 1, 1}, {nz+1, ny+1, nx+1});
  Kokkos::parallel_for("load interpolator", load_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
    //pi   = &fi(1,y,z);
    //pf0  = &f(1,y,z);
    //pfx  = &f(2,y,z);
    //pfy  = &f(1,y+1,z);
    //pfz  = &f(1,y,z+1);
    //pfyz = &f(1,y+1,z+1);
    //pfzx = &f(2,y,z+1);
    //pfxy = &f(2,y+1,z);
    int pi_index   = VOXEL(1, y,   z,   nx,ny,nz) + x-1;
    int pf0_index  = VOXEL(1, y,   z,   nx,ny,nz) + x-1;
    int pfx_index  = VOXEL(2, y,   z,   nx,ny,nz) + x-1;
    int pfy_index  = VOXEL(1, y+1, z,   nx,ny,nz) + x-1;
    int pfz_index  = VOXEL(1, y,   z+1, nx,ny,nz) + x-1;
    int pfyz_index = VOXEL(1, y+1, z+1, nx,ny,nz) + x-1;
    int pfzx_index = VOXEL(2, y,   z+1, nx,ny,nz) + x-1;
    int pfxy_index = VOXEL(2, y+1, z,   nx,ny,nz) + x-1;

    // ex interpolation coefficients
    // ex->tx from hyb_smooth_eb(...)
    //w0 = pf0->ex;
    #define w0 k_field(pf0_index, field_var::tx)
    //w1 = pfy->ex;
    #define w1 k_field(pfy_index, field_var::tx)
    //w2 = pfz->ex;
    #define w2 k_field(pfz_index, field_var::tx)
    //w3 = pfyz->ex;
    #define w3 k_field(pfyz_index, field_var::tx)

    pi_ex       = w0;//fourth*( (w3 + w0) + (w1 + w2) );
    pi_dexdy    = fourth*( (w3 - w0) + (w1 - w2) );
    pi_dexdz    = fourth*( (w3 - w0) - (w1 - w2) );
    pi_d2exdydz = fourth*( (w3 + w0) - (w1 + w2) );

    #undef w0
    #undef w1
    #undef w2
    #undef w3

    // ey interpolation coefficients
    // ey->ty from hyb_smooth_eb(...)

    //w0 = pf0->ey;
    #define w0 k_field(pf0_index, field_var::ty)
    //w1 = pfz->ey;
    #define w1 k_field(pfz_index, field_var::ty)
    //w2 = pfx->ey;
    #define w2 k_field(pfx_index, field_var::ty)
    //w3 = pfzx->ey;
    #define w3 k_field(pfzx_index, field_var::ty)

    pi_ey       = w0;//fourth*( (w3 + w0) + (w1 + w2) );
    pi_deydz    = fourth*( (w3 - w0) + (w1 - w2) );
    pi_deydx    = fourth*( (w3 - w0) - (w1 - w2) );
    pi_d2eydzdx = fourth*( (w3 + w0) - (w1 + w2) );

    #undef w0
    #undef w1
    #undef w2
    #undef w3

    // ez interpolation coefficients
    // ez->tz from hyb_smooth_eb(...)

    // w0 = pf0->ez;
    #define w0 k_field(pf0_index, field_var::tz)
    // w1 = pfx->ez;
    #define w1 k_field(pfx_index, field_var::tz)
    // w2 = pfy->ez;
    #define w2 k_field(pfy_index, field_var::tz)
    // w3 = pfxy->ez;
    #define w3 k_field(pfxy_index, field_var::tz)
    pi_ez       = w0;//fourth*( (w3 + w0) + (w1 + w2) );
    pi_dezdx    = fourth*( (w3 - w0) + (w1 - w2) );
    pi_dezdy    = fourth*( (w3 - w0) - (w1 - w2) );
    pi_d2ezdxdy = fourth*( (w3 + w0) - (w1 + w2) );

    #undef w0
    #undef w1
    #undef w2
    #undef w3

    // bx interpolation coefficients
    // cbx->ox from hyb_smooth_eb(...)

    //w0 = pf0->cbx;
    #define w0 k_field(pf0_index, field_var::ox)
    #define w0b k_field(pf0_index, field_var::cbx0)
    //w1 = pfx->cbx;
    #define w1 k_field(pfx_index, field_var::ox)
    //#define w1b k_field(pfx_index, field_var::cbx0)
    pi_cbx    = w0 + w0b;//half*( w1 + w0 );
    pi_dcbxdx = half*( w1 - w0 ); // To-do: Fix for QS

    #undef w0
    #undef w0b
    #undef w1

    // by interpolation coefficients
    // cby->oy from hyb_smooth_eb(...)

    // w0 = pf0->cby;
    #define w0 k_field(pf0_index, field_var::oy)
    #define w0b k_field(pf0_index, field_var::cby0)
    // w1 = pfy->cby;
    #define w1 k_field(pfy_index, field_var::oy)

    pi_cby    = w0 + w0b;//half*( w1 + w0 );
    pi_dcbydy = half*( w1 - w0 ); // To-do: Fix for QS

    #undef w0
    #undef w0b
    #undef w1

    // bz interpolation coefficients
    // cbz->oz from hyb_smooth_eb(...)

    // w0 = pf0->cbz;
    #define w0 k_field(pf0_index, field_var::oz)
    #define w0b k_field(pf0_index, field_var::cbz0)
    // w1 = pfz->cbz;
    #define w1 k_field(pfz_index, field_var::oz)
    pi_cbz    = w0 + w0b;//half*( w1 + w0 );
    pi_dcbzdz = half*( w1 - w0 ); // To-do: Fix for QS

    #undef w0
    #undef w0b
    #undef w1

    //pi++; pf0++; pfx++; pfy++; pfz++; pfyz++; pfzx++; pfxy++;
  });
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
      host_interp[i].ex       = k_interpolator_h(i, interpolator_var::ex);
      host_interp[i].ey       = k_interpolator_h(i, interpolator_var::ey);
      host_interp[i].ez       = k_interpolator_h(i, interpolator_var::ez);
      host_interp[i].dexdy    = k_interpolator_h(i, interpolator_var::dexdy);
      host_interp[i].dexdz    = k_interpolator_h(i, interpolator_var::dexdz);
      host_interp[i].d2exdydz = k_interpolator_h(i, interpolator_var::d2exdydz);
      host_interp[i].deydz    = k_interpolator_h(i, interpolator_var::deydz);
      host_interp[i].deydx    = k_interpolator_h(i, interpolator_var::deydx);
      host_interp[i].d2eydzdx = k_interpolator_h(i, interpolator_var::d2eydzdx);
      host_interp[i].dezdx    = k_interpolator_h(i, interpolator_var::dezdx);
      host_interp[i].dezdy    = k_interpolator_h(i, interpolator_var::dezdy);
      host_interp[i].d2ezdxdy = k_interpolator_h(i, interpolator_var::d2ezdxdy);
      host_interp[i].cbx      = k_interpolator_h(i, interpolator_var::cbx);
      host_interp[i].cby      = k_interpolator_h(i, interpolator_var::cby);
      host_interp[i].cbz      = k_interpolator_h(i, interpolator_var::cbz);
      host_interp[i].dcbxdx   = k_interpolator_h(i, interpolator_var::dcbxdx);
      host_interp[i].dcbydy   = k_interpolator_h(i, interpolator_var::dcbydy);
      host_interp[i].dcbzdz   = k_interpolator_h(i, interpolator_var::dcbzdz);
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
      k_interpolator_h(i, interpolator_var::ex)       = host_interp[i].ex;
      k_interpolator_h(i, interpolator_var::ey)       = host_interp[i].ey;
      k_interpolator_h(i, interpolator_var::ez)       = host_interp[i].ez;
      k_interpolator_h(i, interpolator_var::dexdy)    = host_interp[i].dexdy;
      k_interpolator_h(i, interpolator_var::dexdz)    = host_interp[i].dexdz;
      k_interpolator_h(i, interpolator_var::d2exdydz) = host_interp[i].d2exdydz;
      k_interpolator_h(i, interpolator_var::deydz)    = host_interp[i].deydz;
      k_interpolator_h(i, interpolator_var::deydx)    = host_interp[i].deydx;
      k_interpolator_h(i, interpolator_var::d2eydzdx) = host_interp[i].d2eydzdx;
      k_interpolator_h(i, interpolator_var::dezdx)    = host_interp[i].dezdx;
      k_interpolator_h(i, interpolator_var::dezdy)    = host_interp[i].dezdy;
      k_interpolator_h(i, interpolator_var::d2ezdxdy) = host_interp[i].d2ezdxdy;
      k_interpolator_h(i, interpolator_var::cbx)      = host_interp[i].cbx;
      k_interpolator_h(i, interpolator_var::cby)      = host_interp[i].cby;
      k_interpolator_h(i, interpolator_var::cbz)      = host_interp[i].cbz;
      k_interpolator_h(i, interpolator_var::dcbxdx)   = host_interp[i].dcbxdx;
      k_interpolator_h(i, interpolator_var::dcbydy)   = host_interp[i].dcbydy;
      k_interpolator_h(i, interpolator_var::dcbzdz)   = host_interp[i].dcbzdz;
    });
#endif

  if(k_i_d.span() < k_i_h.span())
    Kokkos::resize(k_i_d, k_i_h.extent(0));
  Kokkos::deep_copy(k_i_d, k_i_h);

}
