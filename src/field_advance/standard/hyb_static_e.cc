// Note: This is similar to vacuum_compute_curl_b

#define IN_sfa
#include "sfa_private.h"

typedef struct pipeline_args {
  /**/  field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
} pipeline_args_t;


KOKKOS_INLINE_FUNCTION void update_ex(const k_field_t& k_field, const size_t f0,
				      const size_t f, const size_t fm, const float p) {

    const float f_te   = k_field(f, field_var::te);
    const float f_rho  = k_field(f, field_var::rhof);
    const float fm_te  = k_field(fm,field_var::te);
    const float fm_rho = k_field(fm,field_var::rhof);


    k_field(f0, field_var::ex) = -p * (f_rho - fm_rho);
}
KOKKOS_INLINE_FUNCTION void update_ey(const k_field_t& k_field, const size_t f0,
				      const size_t f, const size_t fm, const float p) {
  
    const float f_te   = k_field(f,  field_var::te);
    const float f_rho  = k_field(f,  field_var::rhof);
    const float fm_te  = k_field(fm, field_var::te);
    const float fm_rho = k_field(fm, field_var::rhof);


    k_field(f0, field_var::ey) = -p * (f_te*f_rho - fm_te*fm_rho);
}
KOKKOS_INLINE_FUNCTION void update_ez(const k_field_t& k_field, const size_t f0,
				      const size_t f, const size_t fm, const float p) {
  
    const float f_te   = k_field(f,  field_var::te);
    const float f_rho  = k_field(f,  field_var::rhof);
    const float fm_te  = k_field(fm, field_var::te);
    const float fm_rho = k_field(fm, field_var::rhof);


    k_field(f0, field_var::ez) = -p * (f_te*f_rho - fm_te*fm_rho);
}



void hyb_static_e_interior_kokkos(k_field_t& k_field,
                                const size_t nx, const size_t ny, const size_t nz,
                                const float px,  const float py, const float pz) {

    // EXEC_PIPELINE
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1, 1, 1}, {nz+1, ny+1, nx+1});
    Kokkos::parallel_for("hyb_static_e: Majority of interior", zyx_policy, KOKKOS_LAMBDA(const int z, const int y, const int x) {
        const int f0 =  VOXEL(x,   y,   z,   nx, ny, nz);
        const int fx =  VOXEL(x+1, y,   z,   nx, ny, nz);
        const int fy =  VOXEL(x,   y+1, z,   nx, ny, nz);
        const int fz =  VOXEL(x,   y,   z+1, nx, ny, nz);
        const int fmx = VOXEL(x-1, y,   z,   nx, ny, nz);
        const int fmy = VOXEL(x,   y-1, z,   nx, ny, nz);
        const int fmz = VOXEL(x,   y,   z-1, nx, ny, nz);
	
        update_ex(k_field, f0, fx, fmx, px);
        update_ey(k_field, f0, fy, fmy, py);
        update_ez(k_field, f0, fz, fmz, pz);
    });

}



void
hyb_static_e_kokkos( field_array_t * RESTRICT fa,
                  float frac ) {
  if( !fa     ) ERROR(( "Bad args" ));
  if( frac!=1 ) ERROR(( "standard advance_e does not support frac!=1 yet" ));

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  k_field_t k_field = fa->k_f_d;
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;
  const grid_t                 *              g = args->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;

  const float px     = (nx>1) ? 0.5*g->rdx : 0;
  const float py     = (ny>1) ? 0.5*g->rdy : 0;
  const float pz     = (nz>1) ? 0.5*g->rdz : 0;

  /***************************************************************************
   * Begin tangential B ghost setup
   ***************************************************************************/

  //k_begin_remote_ghost_hyb_jf( fa, fa->g );
    k_begin_remote_ghost_hyb_jf(fa, fa->g, fa->fb );

//    k_local_ghost_tang_b( fa, fa->g );

//k_end_remote_ghost_hyb_jf( fa, fa->g );
    k_end_remote_ghost_hyb_jf(fa, fa->g, fa->fb );

    hyb_static_e_interior_kokkos(k_field, nx, ny, nz, px, py, pz);

    
    //   k_local_adjust_tang_e( fa, fa->g );
}
