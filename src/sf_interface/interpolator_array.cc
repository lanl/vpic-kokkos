#define IN_sf_interface
#define HAS_V4_PIPELINE
#include "sf_interface_private.h"
#include "../vpic/kokkos_helpers.h"


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
  //MALLOC_ALIGNED( ia->i, g->nv, 128 );
  //CLEAR( ia->i, g->nv );
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


#define pi_ex       k_interp(pi_index, ex)
#define pi_dexdy    k_interp(pi_index, dexdy)
#define pi_dexdz    k_interp(pi_index, dexdz)
#define pi_d2exdydz k_interp(pi_index, d2exdydz)

#define pi_ey       k_interp(pi_index, ey)
#define pi_deydz    k_interp(pi_index, deydz)
#define pi_deydx    k_interp(pi_index, deydx)
#define pi_d2eydzdx k_interp(pi_index, d2eydzdx)

#define pi_ez       k_interp(pi_index, ez) 
#define pi_dezdx    k_interp(pi_index, dezdx)
#define pi_dezdy    k_interp(pi_index, dezdy)
#define pi_d2ezdxdy k_interp(pi_index, d2ezdxdy)


#define pi_cbx      k_interp(pi_index, cbx)
#define pi_dcbxdx   k_interp(pi_index, dcbxdx)

#define pi_cby      k_interp(pi_index, cby)
#define pi_dcbydy   k_interp(pi_index, dcbydy)

#define pi_cbz      k_interp(pi_index, cbz)
#define pi_dcbzdz   k_interp(pi_index, dcbzdz)

void load_interpolator_array_kokkos(k_interpolator_t k_interp, k_field_t k_field, int nx, int ny, int nz) {
 
  //for( y=1; y<=ny; y++ ) {
  Kokkos::parallel_for(Kokkos::TeamPolicy< Kokkos::DefaultExecutionSpace>
      (ny, Kokkos::AUTO), KOKKOS_LAMBDA (const k_member_t &team_member) {
    const unsigned int y = team_member.league_rank() + 1;

    // Switched the y and z loops
    //for( z=1; z<=nz; z++ ) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nz), [=] (int j) {
      const unsigned int z = team_member.team_rank() + 1;
       
      //for( x=1; x<=nx; x++ ) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, nx - 1), [=] (int x) {
        KOKKOS_ENUMS()
        //pi = &fi(1,y,z);
        int pi_index = VOXEL(1,y,z, nx,ny,nz) + x;

        //pf0 = &f(1,y,z);
        int pf0_index = VOXEL(1,y,z, nx,ny,nz) + x;

        //pfx = &f(2,y,z);
        int pfx_index = VOXEL(2,y,z, nx,ny,nz) + x;

        //pfy = &f(1,y+1,z);
        int pfy_index = VOXEL(1,y+1,z, nx,ny,nz) + x;

        //pfz = &f(1,y,z+1);
        int pfz_index = VOXEL(1,y,z+1, nx,ny,nz) + x;

        //pfyz = &f(1,y+1,z+1);
        int pfyz_index = VOXEL(1,y+1,z+1, nx,ny,nz) + x;

        //pfzx = &f(2,y,z+1);
        int pfzx_index = VOXEL(2,y,z+1, nx,ny,nz) + x;

        //pfxy = &f(2,y+1,z);
        int pfxy_index = VOXEL(2,y+1,z, nx,ny,nz) + x;

        // ex interpolation coefficients
        //w0 = pf0->ex;
        #define w0 k_field(pf0_index, ex)
        //w1 = pfy->ex;
        #define w1 k_field(pfy_index, ex)
        //w2 = pfz->ex;
        #define w2 k_field(pfz_index, ex)
        //w3 = pfyz->ex;
        #define w3 k_field(pfyz_index, ex)

        pi_ex       = 0.25*(  w0 + w1 + w2 + w3 );
        pi_dexdy    = 0.25*( -w0 + w1 - w2 + w3 );
        pi_dexdz    = 0.25*( -w0 - w1 + w2 + w3 );
        pi_d2exdydz = 0.25*(  w0 - w1 - w2 + w3 );
        
        #undef w0 
        #undef w1 
        #undef w2 
        #undef w3

        // ey interpolation coefficients

        //w0 = pf0->ey;
        #define w0 k_field(pf0_index, ey)
        //w1 = pfz->ey;
        #define w1 k_field(pfz_index, ey)
        //w2 = pfx->ey;
        #define w2 k_field(pfx_index, ey)
        //w3 = pfzx->ey;
        #define w3 k_field(pfzx_index, ey)
        pi_ey       = 0.25*(  w0 + w1 + w2 + w3 );
        pi_deydz    = 0.25*( -w0 + w1 - w2 + w3 );
        pi_deydx    = 0.25*( -w0 - w1 + w2 + w3 );
        pi_d2eydzdx = 0.25*(  w0 - w1 - w2 + w3 );
        
        #undef w0 
        #undef w1 
        #undef w2 
        #undef w3

        // ez interpolation coefficients

        // w0 = pf0->ez;
        #define w0 k_field(pf0_index, ez)
        // w1 = pfx->ez;
        #define w1 k_field(pfx_index, ez)
        // w2 = pfy->ez;
        #define w2 k_field(pfy_index, ez)
        // w3 = pfxy->ez;
        #define w3 k_field(pfxy_index, ez)
        pi_ez       = 0.25*(  w0 + w1 + w2 + w3 );
        pi_dezdx    = 0.25*( -w0 + w1 - w2 + w3 );
        pi_dezdy    = 0.25*( -w0 - w1 + w2 + w3 );
        pi_d2ezdxdy = 0.25*(  w0 - w1 - w2 + w3 );
            
        #undef w0 
        #undef w1 
        #undef w2 
        #undef w3

        // bx interpolation coefficients

        //w0 = pf0->cbx;
        #define w0 k_field(pf0_index, cbx)
        //w1 = pfx->cbx;
        #define w1 k_field(pfx_index, cbx)
        pi_cbx    = 0.5*(  w0 + w1 );
        pi_dcbxdx = 0.5*( -w0 + w1 );
        
        #undef w0 
        #undef w1 

        // by interpolation coefficients

        // w0 = pf0->cby;
        #define w0 k_field(pf0_index, cby)
        // w1 = pfy->cby;
        #define w1 k_field(pfx_index, cby)
        pi_cby    = 0.5*(  w0 + w1 );
        pi_dcbydy = 0.5*( -w0 + w1 );

        #undef w0 
        #undef w1 
        
        // bz interpolation coefficients

        // w0 = pf0->cbz;
        #define w0 k_field(pf0_index, cbz)
        // w1 = pfz->cbz;
        #define w1 k_field(pfx_index, cbz)
        pi_cbz    = 0.5*(  w0 + w1 );
        pi_dcbzdz = 0.5*( -w0 + w1 );

        #undef w0 
        #undef w1 

        //pi++; pf0++; pfx++; pfy++; pfz++; pfyz++; pfzx++; pfxy++;
      });
    });
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

