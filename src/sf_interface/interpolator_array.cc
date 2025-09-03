#define IN_sf_interface
#define HAS_V4_PIPELINE
#include "sf_interface_private.h"


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

void load_interpolator_array_kokkos(k_interpolator_t k_interp, k_field_t k_field, int nx, int ny, int nz) {

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

  const float twelfth = 1./12.;
  const float sixth   = 1./6.;
  const float half    = 0.5;
  const float two     = 2.0;
  const float six     = 6.;

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> load_policy({1, 1, 1}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_for("load interpolator", load_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
        int pi_index = VOXEL(1,   y,   z, nx,ny,nz) + x-1; //pi = &fi(1,y,z);
        int pf0_index = VOXEL(1,  y,   z, nx,ny,nz) + x-1; //pf0 = &f(1,y,z);
        int pfx_index = VOXEL(2,  y,   z, nx,ny,nz) + x-1; //pfx = &f(2,y,z);
        int pfy_index = VOXEL(1,  y+1, z, nx,ny,nz) + x-1; //pfy = &f(1,y+1,z);
        int pfz_index = VOXEL(1,  y,   z+1, nx,ny,nz) + x-1; //pfz = &f(1,y,z+1);
        int pfmx_index = VOXEL(x-1, y,   z,   nx,ny,nz);
        int pfmy_index = VOXEL(x,   y-1, z,   nx,ny,nz);
        int pfmz_index = VOXEL(x,   y,   z-1, nx,ny,nz);

        // ex interpolation coefficients
        // ex->tx from hyb_smooth_eb(...)
        auto w0  = k_field(pf0_index,  field_var::tx);
        auto wx  = k_field(pfx_index,  field_var::tx);
        auto wy  = k_field(pfy_index,  field_var::tx);
        auto wz  = k_field(pfz_index,  field_var::tx);
        auto wmx = k_field(pfmx_index, field_var::tx);
        auto wmy = k_field(pfmy_index, field_var::tx);
        auto wmz = k_field(pfmz_index, field_var::tx);

#ifdef SHAPE_NGP
        pi_ex     = w0;
#else
#ifdef SHAPE_QS
        pi_ex     = twelfth*(six*w0 + wx + wy + wz + wmx + wmy + wmz);
        pi_dexdx  = sixth*(wx - wmx);
        pi_dexdy  = sixth*(wy - wmy);
        pi_dexdz  = sixth*(wz - wmz);
        pi_d2exdx = twelfth*(wx + wmx - two*w0);
        pi_d2exdy = twelfth*(wy + wmy - two*w0);
        pi_d2exdz = twelfth*(wz + wmz - two*w0);
#endif
#endif

        // ey interpolation coefficients
        // ey->ty from hyb_smooth_eb(...)
        w0  = k_field(pf0_index,  field_var::ty);
        wx  = k_field(pfx_index,  field_var::ty);
        wy  = k_field(pfy_index,  field_var::ty);
        wz  = k_field(pfz_index,  field_var::ty);
        wmx = k_field(pfmx_index, field_var::ty);
        wmy = k_field(pfmy_index, field_var::ty);
        wmz = k_field(pfmz_index, field_var::ty);

#ifdef SHAPE_NGP
        pi_ey     = w0;
#else
#ifdef SHAPE_QS
        pi_ey     = twelfth*(six*w0 + wx + wy + wz + wmx + wmy + wmz);
        pi_deydx  = sixth*(wx - wmx);
        pi_deydy  = sixth*(wy - wmy);
        pi_deydz  = sixth*(wz - wmz);
        pi_d2eydx = twelfth*(wx + wmx - two*w0);
        pi_d2eydy = twelfth*(wy + wmy - two*w0);
        pi_d2eydz = twelfth*(wz + wmz - two*w0);
#endif
#endif

        // ez interpolation coefficients
        // ez->tz from hyb_smooth_eb(...)
        w0  = k_field(pf0_index,  field_var::tz);
        wx  = k_field(pfx_index,  field_var::tz);
        wy  = k_field(pfy_index,  field_var::tz);
        wz  = k_field(pfz_index,  field_var::tz);
        wmx = k_field(pfmx_index, field_var::tz);
        wmy = k_field(pfmy_index, field_var::tz);
        wmz = k_field(pfmz_index, field_var::tz);

#ifdef SHAPE_NGP
        pi_ez     = w0;
#else
#ifdef SHAPE_QS
        pi_ez     = twelfth*(six*w0 + wx + wy + wz + wmx + wmy + wmz);
        pi_dezdx  = sixth*(wx - wmx);
        pi_dezdy  = sixth*(wy - wmy);
        pi_dezdz  = sixth*(wz - wmz);
        pi_d2ezdx = twelfth*(wx + wmx - two*w0);
        pi_d2ezdy = twelfth*(wy + wmy - two*w0);
        pi_d2ezdz = twelfth*(wz + wmz - two*w0);
#endif
#endif

        // bx interpolation coefficients
        // cbx->ox from hyb_smooth_eb(...)
        w0  = k_field(pf0_index,  field_var::ox) + k_field(pf0_index,  field_var::cbx0);
        wx  = k_field(pfx_index,  field_var::ox) + k_field(pfx_index,  field_var::cbx0);
        wy  = k_field(pfy_index,  field_var::ox) + k_field(pfy_index,  field_var::cbx0);
        wz  = k_field(pfz_index,  field_var::ox) + k_field(pfz_index,  field_var::cbx0);
        wmx = k_field(pfmx_index, field_var::ox) + k_field(pfmx_index, field_var::cbx0);
        wmy = k_field(pfmy_index, field_var::ox) + k_field(pfmy_index, field_var::cbx0);
        wmz = k_field(pfmz_index, field_var::ox) + k_field(pfmz_index, field_var::cbx0);

#ifdef SHAPE_NGP
        pi_cbx     = w0;
#else
#ifdef SHAPE_QS
        pi_cbx     = twelfth*(six*w0 + wx + wy + wz + wmx + wmy + wmz);
        pi_dcbxdx  = sixth*(wx - wmx);
        pi_dcbxdy  = sixth*(wy - wmy);
        pi_dcbxdz  = sixth*(wz - wmz);
        pi_d2cbxdx = twelfth*(wx + wmx - two*w0);
        pi_d2cbxdy = twelfth*(wy + wmy - two*w0);
        pi_d2cbxdz = twelfth*(wz + wmz - two*w0);
#endif
#endif

        // by interpolation coefficients
        // cby->oy from hyb_smooth_eb(...)
        w0  = k_field(pf0_index,  field_var::oy) + k_field(pf0_index,  field_var::cby0);
        wx  = k_field(pfx_index,  field_var::oy) + k_field(pfx_index,  field_var::cby0);
        wy  = k_field(pfy_index,  field_var::oy) + k_field(pfy_index,  field_var::cby0);
        wz  = k_field(pfz_index,  field_var::oy) + k_field(pfz_index,  field_var::cby0);
        wmx = k_field(pfmx_index, field_var::oy) + k_field(pfmx_index, field_var::cby0);
        wmy = k_field(pfmy_index, field_var::oy) + k_field(pfmy_index, field_var::cby0);
        wmz = k_field(pfmz_index, field_var::oy) + k_field(pfmz_index, field_var::cby0);

#ifdef SHAPE_NGP
        pi_cby     = w0;
#else
#ifdef SHAPE_QS
        pi_cby     = twelfth*(six*w0 + wx + wy + wz + wmx + wmy + wmz);
        pi_dcbydx  = sixth*(wx - wmx);
        pi_dcbydy  = sixth*(wy - wmy);
        pi_dcbydz  = sixth*(wz - wmz);
        pi_d2cbydx = twelfth*(wx + wmx - two*w0);
        pi_d2cbydy = twelfth*(wy + wmy - two*w0);
        pi_d2cbydz = twelfth*(wz + wmz - two*w0);
#endif
#endif

        // bz interpolation coefficients
        // cbz->oz from hyb_smooth_eb(...)
        w0  = k_field(pf0_index,  field_var::oz) + k_field(pf0_index,  field_var::cbz0);
        wx  = k_field(pfx_index,  field_var::oz) + k_field(pfx_index,  field_var::cbz0);
        wy  = k_field(pfy_index,  field_var::oz) + k_field(pfy_index,  field_var::cbz0);
        wz  = k_field(pfz_index,  field_var::oz) + k_field(pfz_index,  field_var::cbz0);
        wmx = k_field(pfmx_index, field_var::oz) + k_field(pfmx_index, field_var::cbz0);
        wmy = k_field(pfmy_index, field_var::oz) + k_field(pfmy_index, field_var::cbz0);
        wmz = k_field(pfmz_index, field_var::oz) + k_field(pfmz_index, field_var::cbz0);

#ifdef SHAPE_NGP
        pi_cbz     = w0;
#else
#ifdef SHAPE_QS
        pi_cbz     = twelfth*(six*w0 + wx + wy + wz + wmx + wmy + wmz);
        pi_dcbzdx  = sixth*(wx - wmx);
        pi_dcbzdy  = sixth*(wy - wmy);
        pi_dcbzdz  = sixth*(wz - wmz);
        pi_d2cbzdx = twelfth*(wx + wmx - two*w0);
        pi_d2cbzdy = twelfth*(wy + wmy - two*w0);
        pi_d2cbzdz = twelfth*(wz + wmz - two*w0);
#endif
#endif

        //pi++; pf0++; pfx++; pfy++; pfz++; pfyz++; pfzx++; pfxy++;

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

/*
    Kokkos::parallel_for("load interpolator", KOKKOS_TEAM_POLICY_DEVICE
      (nz, Kokkos::AUTO),
      KOKKOS_LAMBDA
      (const KOKKOS_TEAM_POLICY_DEVICE::member_type &team_member) {
    const unsigned int z = team_member.league_rank() + 1;

    //for( z=1; z<=nz; z++ ) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny), [=] (int yi) {
      const unsigned int y = yi + 1;

      //for( x=1; x<=nx; x++ ) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, nx), [=] (int x) {

        //pi = &fi(1,y,z);
        int pi_index = VOXEL(1,   y,   z, nx,ny,nz) + x;

        //pf0 = &f(1,y,z);
        int pf0_index = VOXEL(1,  y,   z, nx,ny,nz) + x;

        //pfx = &f(2,y,z);
        int pfx_index = VOXEL(2,  y,   z, nx,ny,nz) + x;

        //pfy = &f(1,y+1,z);
        int pfy_index = VOXEL(1,  y+1, z, nx,ny,nz) + x;

        //pfz = &f(1,y,z+1);
        int pfz_index = VOXEL(1,  y,   z+1, nx,ny,nz) + x;

        //pfyz = &f(1,y+1,z+1);
        int pfyz_index = VOXEL(1, y+1, z+1, nx,ny,nz) + x;

        //pfzx = &f(2,y,z+1);
        int pfzx_index = VOXEL(2, y,   z+1, nx,ny,nz) + x;

        //pfxy = &f(2,y+1,z);
        int pfxy_index = VOXEL(2, y+1, z, nx,ny,nz) + x;

        // ex interpolation coefficients
        //w0 = pf0->ex;
        #define w0 k_field(pf0_index, field_var::ex)
        //w1 = pfy->ex;
        #define w1 k_field(pfy_index, field_var::ex)
        //w2 = pfz->ex;
        #define w2 k_field(pfz_index, field_var::ex)
        //w3 = pfyz->ex;
        #define w3 k_field(pfyz_index, field_var::ex)

        pi_ex       = fourth*( (w3 + w0) + (w1 + w2) );
        pi_dexdy    = fourth*( (w3 - w0) + (w1 - w2) );
        pi_dexdz    = fourth*( (w3 - w0) - (w1 - w2) );
        pi_d2exdydz = fourth*( (w3 + w0) - (w1 + w2) );

        #undef w0
        #undef w1
        #undef w2
        #undef w3

        // ey interpolation coefficients

        //w0 = pf0->ey;
        #define w0 k_field(pf0_index, field_var::ey)
        //w1 = pfz->ey;
        #define w1 k_field(pfz_index, field_var::ey)
        //w2 = pfx->ey;
        #define w2 k_field(pfx_index, field_var::ey)
        //w3 = pfzx->ey;
        #define w3 k_field(pfzx_index, field_var::ey)

        pi_ey       = fourth*( (w3 + w0) + (w1 + w2) );
        pi_deydz    = fourth*( (w3 - w0) + (w1 - w2) );
        pi_deydx    = fourth*( (w3 - w0) - (w1 - w2) );
        pi_d2eydzdx = fourth*( (w3 + w0) - (w1 + w2) );

        #undef w0
        #undef w1
        #undef w2
        #undef w3

        // ez interpolation coefficients

        // w0 = pf0->ez;
        #define w0 k_field(pf0_index, field_var::ez)
        // w1 = pfx->ez;
        #define w1 k_field(pfx_index, field_var::ez)
        // w2 = pfy->ez;
        #define w2 k_field(pfy_index, field_var::ez)
        // w3 = pfxy->ez;
        #define w3 k_field(pfxy_index, field_var::ez)
        pi_ez       = fourth*( (w3 + w0) + (w1 + w2) );
        pi_dezdx    = fourth*( (w3 - w0) + (w1 - w2) );
        pi_dezdy    = fourth*( (w3 - w0) - (w1 - w2) );
        pi_d2ezdxdy = fourth*( (w3 + w0) - (w1 + w2) );

        #undef w0
        #undef w1
        #undef w2
        #undef w3

        // bx interpolation coefficients

        //w0 = pf0->cbx;
        #define w0 k_field(pf0_index, field_var::cbx)
        //w1 = pfx->cbx;
        #define w1 k_field(pfx_index, field_var::cbx)
        pi_cbx    = half*( w1 + w0 );
        pi_dcbxdx = half*( w1 - w0 );

        #undef w0
        #undef w1

        // by interpolation coefficients

        // w0 = pf0->cby;
        #define w0 k_field(pf0_index, field_var::cby)
        // w1 = pfy->cby;
        #define w1 k_field(pfy_index, field_var::cby)

        pi_cby    = half*( w1 + w0 );
        pi_dcbydy = half*( w1 - w0 );

        #undef w0
        #undef w1

        // bz interpolation coefficients

        // w0 = pf0->cbz;
        #define w0 k_field(pf0_index, field_var::cbz)
        // w1 = pfz->cbz;
        #define w1 k_field(pfz_index, field_var::cbz)
        pi_cbz    = half*( w1 + w0 );
        pi_dcbzdz = half*( w1 - w0 );

        #undef w0
        #undef w1

        //pi++; pf0++; pfx++; pfy++; pfz++; pfyz++; pfzx++; pfxy++;
      });
    }
    );
  });
*/
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

  Kokkos::deep_copy(k_i_h, k_i_d);

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
#else
#ifdef SHAPE_QS
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
#endif
#endif
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
#ifdef SHAPE_NGP
      k_interpolator_h(i, interpolator_var::ex)       = host_interp[i].ex;
      k_interpolator_h(i, interpolator_var::ey)       = host_interp[i].ey;
      k_interpolator_h(i, interpolator_var::ez)       = host_interp[i].ez;
      k_interpolator_h(i, interpolator_var::cbx)      = host_interp[i].cbx;
      k_interpolator_h(i, interpolator_var::cby)      = host_interp[i].cby;
      k_interpolator_h(i, interpolator_var::cbz)      = host_interp[i].cbz;
#else
#ifdef SHAPE_QS
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
#endif
#endif
    });

  Kokkos::deep_copy(k_i_d, k_i_h);

}
