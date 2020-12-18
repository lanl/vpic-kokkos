#include "interpolator_array.h"

void
checkpt_interpolator_array( const interpolator_array_t * ia ) {
  CHECKPT( ia, 1 );
  CHECKPT_ALIGNED( ia->i, ia->g->nv, 128 );
  CHECKPT_PTR( ia->g );
}

interpolator_array_t *
restore_interpolator_array( void ) {
  interpolator_array_t * ia;
  RESTORE( ia );
  RESTORE_ALIGNED( ia->i );
  RESTORE_PTR( ia->g );
  return ia;
}

interpolator_array_t *
new_interpolator_array( grid_t * g ) {
  interpolator_array_t * ia;
  if( !g ) ERROR(( "NULL grid" ));
  ia = new interpolator_array_t(g->nv);
  //MALLOC( ia, 1 );
  MALLOC_ALIGNED( ia->i, g->nv, 128 );
  CLEAR( ia->i, g->nv );
  ia->g = g;
  REGISTER_OBJECT( ia, checkpt_interpolator_array, restore_interpolator_array,
                   NULL );
  return ia;
}

void
delete_interpolator_array( interpolator_array_t * ia ) {
  if( !ia ) return;
  UNREGISTER_OBJECT( ia );
  FREE_ALIGNED( ia->i );
  delete(ia);
  //FREE( ia );
}

template<class geo_t> void
load_interpolator_array_kokkos(
  geo_t& geometry,
  k_interpolator_t k_interp,
  k_field_t k_field,
  int nx,
  int ny,
  int nz
)
{

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

  const float fourth = 0.25;
  const float half   = 0.5;

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> load_policy({1, 1, 1}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_for("load interpolator", load_policy,
      KOKKOS_LAMBDA(const int z, const int y, const int x) {

        float w0, w1, w2, w3;

        int pi_index = VOXEL(1,   y,   z, nx,ny,nz) + x-1;
        int pf0_index = VOXEL(1,  y,   z, nx,ny,nz) + x-1;
        int pfx_index = VOXEL(2,  y,   z, nx,ny,nz) + x-1;
        int pfy_index = VOXEL(1,  y+1, z, nx,ny,nz) + x-1;
        int pfz_index = VOXEL(1,  y,   z+1, nx,ny,nz) + x-1;
        int pfyz_index = VOXEL(1, y+1, z+1, nx,ny,nz) + x-1;
        int pfzx_index = VOXEL(2, y,   z+1, nx,ny,nz) + x-1;
        int pfxy_index = VOXEL(2, y+1, z, nx,ny,nz) + x-1;

        // ex interpolation coefficients
        w0 = k_field(pf0_index, field_var::ex);
        w1 = k_field(pfy_index, field_var::ex);
        w2 = k_field(pfz_index, field_var::ex);
        w3 = k_field(pfyz_index, field_var::ex);

        geometry.prescale_interpolated_ex(pi_index, w0, w1, w2, w3);

        pi_ex       = fourth*( (w3 + w0) + (w1 + w2) );
        pi_dexdy    = fourth*( (w3 - w0) + (w1 - w2) );
        pi_dexdz    = fourth*( (w3 - w0) - (w1 - w2) );
        pi_d2exdydz = fourth*( (w3 + w0) - (w1 + w2) );


        // ey interpolation coefficients

        w0 = k_field(pf0_index, field_var::ey);
        w1 = k_field(pfz_index, field_var::ey);
        w2 = k_field(pfx_index, field_var::ey);
        w3 = k_field(pfzx_index, field_var::ey);

        geometry.prescale_interpolated_ey(pi_index, w0, w2, w1, w3);

        pi_ey       = fourth*( (w3 + w0) + (w1 + w2) );
        pi_deydz    = fourth*( (w3 - w0) + (w1 - w2) );
        pi_deydx    = fourth*( (w3 - w0) - (w1 - w2) );
        pi_d2eydzdx = fourth*( (w3 + w0) - (w1 + w2) );


        // ez interpolation coefficients

        w0 = k_field(pf0_index, field_var::ez);
        w1 = k_field(pfx_index, field_var::ez);
        w2 = k_field(pfy_index, field_var::ez);
        w3 = k_field(pfxy_index, field_var::ez);

        geometry.prescale_interpolated_ez(pi_index, w0, w1, w2, w3);

        pi_ez       = fourth*( (w3 + w0) + (w1 + w2) );
        pi_dezdx    = fourth*( (w3 - w0) + (w1 - w2) );
        pi_dezdy    = fourth*( (w3 - w0) - (w1 - w2) );
        pi_d2ezdxdy = fourth*( (w3 + w0) - (w1 + w2) );


        // bx interpolation coefficients

        w0 = k_field(pf0_index, field_var::cbx);
        w1 = k_field(pfx_index, field_var::cbx);

        geometry.prescale_interpolated_cbx(pi_index, w0, w1);

        pi_cbx    = half*( w1 + w0 );
        pi_dcbxdx = half*( w1 - w0 );


        // by interpolation coefficients

        w0 = k_field(pf0_index, field_var::cby);
        w1 = k_field(pfy_index, field_var::cby);

        geometry.prescale_interpolated_cby(pi_index, w0, w1);

        pi_cby    = half*( w1 + w0 );
        pi_dcbydy = half*( w1 - w0 );


        // bz interpolation coefficients

        w0 = k_field(pf0_index, field_var::cbz);
        w1 = k_field(pfz_index, field_var::cbz);

        geometry.prescale_interpolated_cbz(pi_index, w0, w1);

        pi_cbz    = half*( w1 + w0 );
        pi_dcbzdz = half*( w1 - w0 );

    });

}

void
interpolator_array_t::load( const field_array_t * RESTRICT fa ) {

  if( !fa || g!=fa->g ) ERROR(( "Bad args" ));

  grid_t *g = fa->g;
  int nx = g->nx;
  int ny = g->ny;
  int nz = g->nz;

  SELECT_GEOMETRY(g->geometry, geo, ({

    auto geometry = g->get_device_geometry<geo>();

    load_interpolator_array_kokkos(
      geometry,
      k_i_d,
      fa->k_f_d,
      nx,
      ny,
      nz
    );

  }));



}

void
interpolator_array_t::copy_to_host() {

  Kokkos::deep_copy(k_i_h, k_i_d);

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

}

void
interpolator_array_t::copy_to_device() {

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

  Kokkos::deep_copy(k_i_d, k_i_h);

}
