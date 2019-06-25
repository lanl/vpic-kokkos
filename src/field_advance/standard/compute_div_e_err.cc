// Note: This is virtually identical to compute_rhob
#define IN_sfa
#include "sfa_private.h"

typedef struct pipeline_args {
  /**/  field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
} pipeline_args_t;

#define DECLARE_STENCIL()                                       \
  /**/  field_t                * ALIGNED(128) f = args->f;      \
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;  \
  const grid_t                 *              g = args->g;      \
  const int nx = g->nx, ny = g->ny, nz = g->nz;                 \
                                                                \
  const float px = (nx>1) ? g->rdx : 0;                         \
  const float py = (ny>1) ? g->rdy : 0;                         \
  const float pz = (nz>1) ? g->rdz : 0;                         \
  const float cj = 1./g->eps0;                                  \
                                                                \
  field_t * ALIGNED(16) f0;                                     \
  field_t * ALIGNED(16) fx, * ALIGNED(16) fy, * ALIGNED(16) fz; \
  int x, y, z

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

#define INIT_STENCIL()  \
  f0 = &f(x,  y,  z  ); \
  fx = &f(x-1,y,  z  ); \
  fy = &f(x,  y-1,z  ); \
  fz = &f(x,  y,  z-1)

#define NEXT_STENCIL()                \
  f0++; fx++; fy++; fz++; x++;        \
  if( x>nx ) {                        \
    /**/       y++;            x = 2; \
    if( y>ny ) z++; if( y>ny ) y = 2; \
    INIT_STENCIL();                   \
  }

#define UPDATE_DERR_E() f0->div_e_err = m[f0->nmat].nonconductive * \
  ( px*( m[f0->ematx].epsx*f0->ex - m[fx->ematx].epsx*fx->ex ) +    \
    py*( m[f0->ematy].epsy*f0->ey - m[fy->ematy].epsy*fy->ey ) +    \
    pz*( m[f0->ematz].epsz*f0->ez - m[fz->ematz].epsz*fz->ez ) -    \
    cj*( f0->rhof + f0->rhob ) )

void compute_div_e_err_interior_kokkos(field_array_t* fa, const grid_t* g);
void compute_div_e_err_exterior_kokkos(field_array_t* fa, const grid_t* g);

KOKKOS_INLINE_FUNCTION void update_derr_e(const k_field_t& k_field, const k_field_edge_t& k_field_edge, const k_material_coefficient_t& k_mat_coeff, int f0, int fx, int fy, int fz, float px, float py, float pz, float cj) {
    const material_id f0_nmat = k_field_edge(f0, field_edge_var::nmat);
    const material_id f0_ematx = k_field_edge(f0, field_edge_var::ematx);
    const material_id f0_ematy = k_field_edge(f0, field_edge_var::ematy);
    const material_id f0_ematz = k_field_edge(f0, field_edge_var::ematz);
    const material_id fx_ematx = k_field_edge(fx, field_edge_var::ematx);
    const material_id fy_ematy = k_field_edge(fy, field_edge_var::ematy);
    const material_id fz_ematz = k_field_edge(fz, field_edge_var::ematz);
    const float f0_ex = k_field(f0, field_var::ex);
    const float f0_ey = k_field(f0, field_var::ey);
    const float f0_ez = k_field(f0, field_var::ez);
    const float fx_ex = k_field(fx, field_var::ex);
    const float fy_ey = k_field(fy, field_var::ey);
    const float fz_ez = k_field(fz, field_var::ez);
    const float f0_rhof = k_field(f0, field_var::rhof);
    const float f0_rhob = k_field(f0, field_var::rhob);

    k_field(f0, field_var::div_e_err) = k_mat_coeff(f0_nmat, material_coeff_var::nonconductive) *
        ( px*( k_mat_coeff(f0_ematx, material_coeff_var::epsx)*f0_ex - k_mat_coeff(fx_ematx, material_coeff_var::epsx)*fx_ex ) +
          py*( k_mat_coeff(f0_ematy, material_coeff_var::epsy)*f0_ey - k_mat_coeff(fy_ematy, material_coeff_var::epsy)*fy_ey ) +
          pz*( k_mat_coeff(f0_ematz, material_coeff_var::epsz)*f0_ez - k_mat_coeff(fz_ematz, material_coeff_var::epsz)*fz_ez ) -
          cj*( f0_rhof + f0_rhob ) );
}

void
compute_div_e_err_pipeline( pipeline_args_t * args,
                            int pipeline_rank,
                            int n_pipeline ) {
  DECLARE_STENCIL();

  int n_voxel;
  DISTRIBUTE_VOXELS( 2,nx, 2,ny, 2,nz, 16,
                     pipeline_rank, n_pipeline,
                     x, y, z, n_voxel );

  INIT_STENCIL();
  for( ; n_voxel; n_voxel-- ) {
    UPDATE_DERR_E();
    NEXT_STENCIL();
  }
}

#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

#error "Not implemented"

#endif

void
compute_div_e_err( field_array_t * RESTRICT fa ) {
  if( !fa ) ERROR(( "Bad args" ));

  // Have pipelines compute the interior of local domain (the host
  // handles stragglers in the interior)

  // Begin setting normal e ghosts

  begin_remote_ghost_norm_e( fa->f, fa->g );

  local_ghost_norm_e( fa->f, fa->g );

  // Have pipelines compute interior of local domain
  pipeline_args_t args[1];  
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  EXEC_PIPELINES( compute_div_e_err, args, 0 );

  // While pipelines are busy, have host compute the exterior
  // of the local domain

  DECLARE_STENCIL();

  // Finish setting normal e ghosts

  end_remote_ghost_norm_e( fa->f, fa->g );

  // z faces, x edges, y edges and all corners
  for( y=1; y<=ny+1; y++ ) {
    f0 = &f(1,y,  1);
    fx = &f(0,y,  1);
    fy = &f(1,y-1,1);
    fz = &f(1,y,  0);
    for( x=1; x<=nx+1; x++ ) {
      UPDATE_DERR_E();
      f0++;
      fx++;
      fy++;
      fz++;
    }
  }
  for( y=1; y<=ny+1; y++ ) {
    f0 = &f(1,y,  nz+1);
    fx = &f(0,y,  nz+1);
    fy = &f(1,y-1,nz+1);
    fz = &f(1,y,  nz);
    for( x=1; x<=nx+1; x++ ) {
      UPDATE_DERR_E();
      f0++;
      fx++;
      fy++;
      fz++;
    }
  }

  // y faces, z edges
  for( z=2; z<=nz; z++ ) {
    f0 = &f(1,1,z);
    fx = &f(0,1,z);
    fy = &f(1,0,z);
    fz = &f(1,1,z-1);
    for( x=1; x<=nx+1; x++ ) {
      UPDATE_DERR_E();
      f0++;
      fx++;
      fy++;
      fz++;
    }
  }
  for( z=2; z<=nz; z++ ) {
    f0 = &f(1,ny+1,z);
    fx = &f(0,ny+1,z);
    fy = &f(1,ny,  z);
    fz = &f(1,ny+1,z-1);
    for( x=1; x<=nx+1; x++ ) {
      UPDATE_DERR_E();
      f0++;
      fx++;
      fy++;
      fz++;
    }
  }

  // x faces
  for( z=2; z<=nz; z++ ) {
    for( y=2; y<=ny; y++ ) {
      f0 = &f(1,y,  z);
      fx = &f(0,y,  z);
      fy = &f(1,y-1,z);
      fz = &f(1,y,  z-1);
      UPDATE_DERR_E();
      f0 = &f(nx+1,y,  z);
      fx = &f(nx,  y,  z);
      fy = &f(nx+1,y-1,z);
      fz = &f(nx+1,y,  z-1);
      UPDATE_DERR_E();
    }
  }


  // Finish up setting interior

  WAIT_PIPELINES();

  local_adjust_div_e( fa->f, fa->g );
}

void compute_div_e_err_interior_kokkos(field_array_t* fa, const grid_t* g) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    sfa_params_t* sfa_p = reinterpret_cast<sfa_params_t*>(fa->params);

    const float px = (nx>1) ? g->rdx : 0;
    const float py = (ny>1) ? g->rdy : 0;
    const float pz = (nz>1) ? g->rdz : 0;
    const float cj = 1./g->eps0;

    k_field_t& k_field = fa->k_f_d;
    k_field_edge_t& k_field_edge = fa->k_fe_d;
    k_material_coefficient_t& k_matcoeff = sfa_p->k_mc_d;

    Kokkos::parallel_for("compute_div_e interior", KOKKOS_TEAM_POLICY_DEVICE(nz-1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
        const int z = team_member.league_rank() + 2;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny-1), [=] (const int yi) {
            const int y = yi + 2;
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, nx-1), [=] (const int xi) {
                const int x = xi + 2;

                const int f0 = VOXEL(x,   y,    z, nx, ny, nz);
                const int fx = VOXEL(x-1, y,    z, nx, ny, nz);
                const int fy = VOXEL(x,   y-1,  z, nx, ny, nz);
                const int fz = VOXEL(x,   y,    z-1, nx, ny, nz);
                update_derr_e(k_field, k_field_edge, k_matcoeff, f0, fx, fy, fz, px, py, pz, cj);
            });
        });
    });
}

void compute_div_e_err_exterior_kokkos(field_array_t* fa, const grid_t* g) {
    const int nx = g->nx, ny = g->ny, nz = g->nz;
    sfa_params_t* sfa_p = reinterpret_cast<sfa_params_t*>(fa->params);

    const float px = (nx>1) ? g->rdx : 0;
    const float py = (ny>1) ? g->rdy : 0;
    const float pz = (nz>1) ? g->rdz : 0;
    const float cj = 1./g->eps0;

    k_field_t&                  k_field = fa->k_f_d;
    k_field_edge_t&             k_field_edge = fa->k_fe_d;
    k_material_coefficient_t&   k_matcoef = sfa_p->k_mc_d;
    
    // z faces, x edges, y edges and all corners
    Kokkos::parallel_for("z faces, x edges, y edges and all corners", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
            const int y = team_member.league_rank() + 1;
            const int f0 = VOXEL(1, y,   1, nx, ny, nz) + xi;
            const int fx = VOXEL(0, y,   1, nx, ny, nz) + xi;
            const int fy = VOXEL(1, y-1, 1, nx, ny, nz) + xi;
            const int fz = VOXEL(1, y,   0, nx, ny, nz) + xi;
            update_derr_e(k_field, k_field_edge, k_matcoef, f0, fx, fy, fz, px, py, pz, cj);
        });
    });

    Kokkos::parallel_for("z faces, x edges, y edges and all corners: end of z", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
            const int y = team_member.league_rank() + 1;
            const int f0 = VOXEL(1, y,   nz+1, nx, ny, nz) + xi;
            const int fx = VOXEL(0, y,   nz+1, nx, ny, nz) + xi;
            const int fy = VOXEL(1, y-1, nz+1, nx, ny, nz) + xi;
            const int fz = VOXEL(1, y,   nz, nx, ny, nz) + xi;
            update_derr_e(k_field, k_field_edge, k_matcoef, f0, fx, fy, fz, px, py, pz, cj);
        });
    });

    // y faces, z edges
    Kokkos::parallel_for("y faces, z edges start", KOKKOS_TEAM_POLICY_DEVICE(nz-1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
            const int z = team_member.league_rank() + 2;
            const int f0 = VOXEL(1, 1, z, nx, ny, nz) + xi;
            const int fx = VOXEL(0, 1, z, nx, ny, nz) + xi;
            const int fy = VOXEL(1, 0, z, nx, ny, nz) + xi;
            const int fz = VOXEL(1, 1, z-1, nx, ny, nz) + xi;
            update_derr_e(k_field, k_field_edge, k_matcoef, f0, fx, fy, fz, px, py, pz, cj);
        });
    });

    Kokkos::parallel_for("y faces, z edges end", KOKKOS_TEAM_POLICY_DEVICE(nz-1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
            const int z = team_member.league_rank() + 2;
            const int f0 = VOXEL(1, ny+1, z, nx, ny, nz) + xi;
            const int fx = VOXEL(0, ny+1, z, nx, ny, nz) + xi;
            const int fy = VOXEL(1, ny,   z, nx, ny, nz) + xi;
            const int fz = VOXEL(1, ny+1, z-1, nx, ny, nz) + xi;
            update_derr_e(k_field, k_field_edge, k_matcoef, f0, fx, fy, fz, px, py, pz, cj);
        });
    });

    // x faces
    Kokkos::parallel_for("x faces", KOKKOS_TEAM_POLICY_DEVICE(nz-1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ny-1), [=] (const int yi) {
            const int z = team_member.league_rank() + 2;
            const int y = yi + 2;
            int f0 = VOXEL(1, y,   z, nx, ny, nz);
            int fx = VOXEL(0, y,   z, nx, ny, nz);
            int fy = VOXEL(1, y-1, z, nx, ny, nz);
            int fz = VOXEL(1, y,   z-1, nx, ny, nz);
            update_derr_e(k_field, k_field_edge, k_matcoef, f0, fx, fy, fz, px, py, pz, cj);

            f0 = VOXEL(nx+1, y,   z, nx, ny, nz);
            fx = VOXEL(nx  , y,   z, nx, ny, nz);
            fy = VOXEL(nx+1, y-1, z, nx, ny, nz);
            fz = VOXEL(nx+1, y,   z-1, nx, ny, nz);
            update_derr_e(k_field, k_field_edge, k_matcoef, f0, fx, fy, fz, px, py, pz, cj);
        });
    });
}

void
compute_div_e_err_kokkos( field_array_t * RESTRICT fa ) {
  if( !fa ) ERROR(( "Bad args" ));

  // Have pipelines compute the interior of local domain (the host
  // handles stragglers in the interior)

  // Begin setting normal e ghosts

    k_begin_remote_ghost_norm_e( fa, fa->g );

    k_local_ghost_norm_e( fa, fa->g );

  // Have pipelines compute interior of local domain
/*
  pipeline_args_t args[1];  
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
*/    
    compute_div_e_err_interior_kokkos(fa, fa->g);

  // While pipelines are busy, have host compute the exterior
  // of the local domain

//  DECLARE_STENCIL();

  // Finish setting normal e ghosts

    k_end_remote_ghost_norm_e( fa, fa->g );

    compute_div_e_err_exterior_kokkos(fa, fa->g);

  // Finish up setting interior

    k_local_adjust_div_e( fa, fa->g );
}
