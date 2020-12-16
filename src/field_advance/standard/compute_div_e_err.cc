#define IN_sfa
#include "sfa_private.h"

template<class geo_t, class edge_t>
class div_e_kernel {
public:

  div_e_kernel(
    const geo_t& geometry,
    const k_field_t& k_field,
    const edge_t& k_field_edge,
    const k_material_coefficient_t& k_mat_coeff,
    const float eps0
  )
  : geometry(geometry),
    k_field(k_field),
    k_field_edge(k_field_edge),
    k_mat_coeff(k_mat_coeff),
    eps0_inv(1.0/eps0)
  {

  }

  KOKKOS_INLINE_FUNCTION void
  operator() (int f0, int fx, int fy, int fz) const {

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
      (
        geometry.edge_divergence(
          f0,
          k_mat_coeff(f0_ematx, material_coeff_var::epsx)*f0_ex,
          k_mat_coeff(fx_ematx, material_coeff_var::epsx)*fx_ex,
          k_mat_coeff(f0_ematy, material_coeff_var::epsy)*f0_ey,
          k_mat_coeff(fy_ematy, material_coeff_var::epsy)*fy_ey,
          k_mat_coeff(f0_ematz, material_coeff_var::epsz)*f0_ez,
          k_mat_coeff(fz_ematz, material_coeff_var::epsz)*fz_ez
        )
        - eps0_inv*( f0_rhof + f0_rhob )
      );

  }

  const geo_t geometry;
  const k_field_t k_field;
  const edge_t k_field_edge;
  const k_material_coefficient_t k_mat_coeff;
  const float eps0_inv;

};

template<class geo_t, class edge_t>
class rhob_kernel {
public:

  rhob_kernel(
    const geo_t& geometry,
    const k_field_t& k_field,
    const edge_t& k_field_edge,
    const k_material_coefficient_t& k_mat_coeff,
    const float eps0
  )
  : geometry(geometry),
    k_field(k_field),
    k_field_edge(k_field_edge),
    k_mat_coeff(k_mat_coeff),
    eps0(eps0)
  {

  }

  KOKKOS_INLINE_FUNCTION void
  operator() (int f0, int fx, int fy, int fz) const {

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

    k_field(f0, field_var::rhob) = k_mat_coeff(f0_nmat, material_coeff_var::nonconductive) *
      (
        eps0*geometry.edge_divergence(
          f0,
          k_mat_coeff(f0_ematx, material_coeff_var::epsx)*f0_ex,
          k_mat_coeff(fx_ematx, material_coeff_var::epsx)*fx_ex,
          k_mat_coeff(f0_ematy, material_coeff_var::epsy)*f0_ey,
          k_mat_coeff(fy_ematy, material_coeff_var::epsy)*fy_ey,
          k_mat_coeff(f0_ematz, material_coeff_var::epsz)*f0_ez,
          k_mat_coeff(fz_ematz, material_coeff_var::epsz)*fz_ez
        )
        - f0_rhof
      );

  }

  const geo_t geometry;
  const k_field_t k_field;
  const edge_t k_field_edge;
  const k_material_coefficient_t k_mat_coeff;
  const float eps0;

};


template<class kernel_t> void
interior_div_e_loop_kokkos(
  const kernel_t& kernel,
  const grid_t* g
)
{
    const int nx = g->nx, ny = g->ny, nz = g->nz;

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
                kernel(f0, fx, fy, fz);
            });
        });
    });
}

template <class kernel_t> void
exterior_div_e_loop_kokkos(
  const kernel_t& kernel,
  const grid_t* g
)
{
    const int nx = g->nx, ny = g->ny, nz = g->nz;

    // z faces, x edges, y edges and all corners
    Kokkos::parallel_for("z faces, x edges, y edges and all corners", KOKKOS_TEAM_POLICY_DEVICE(ny+1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type& team_member) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nx+1), [=] (const int xi) {
            const int y = team_member.league_rank() + 1;
            const int f0 = VOXEL(1, y,   1, nx, ny, nz) + xi;
            const int fx = VOXEL(0, y,   1, nx, ny, nz) + xi;
            const int fy = VOXEL(1, y-1, 1, nx, ny, nz) + xi;
            const int fz = VOXEL(1, y,   0, nx, ny, nz) + xi;
            kernel(f0, fx, fy, fz);
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
            kernel(f0, fx, fy, fz);
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
            kernel(f0, fx, fy, fz);
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
            kernel(f0, fx, fy, fz);
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
            kernel(f0, fx, fy, fz);

            f0 = VOXEL(nx+1, y,   z, nx, ny, nz);
            fx = VOXEL(nx  , y,   z, nx, ny, nz);
            fy = VOXEL(nx+1, y-1, z, nx, ny, nz);
            fz = VOXEL(nx+1, y,   z-1, nx, ny, nz);
            kernel(f0, fx, fy, fz);
        });
    });
}

template<class kernel_t> void
div_e_loop_kokkos(
  const kernel_t& kernel,
  field_array_t * fa,
  const grid_t* g
)
{

  // Begin setting normal e ghosts

  k_begin_remote_ghost_norm_e( fa, g );

  k_local_ghost_norm_e( fa, g );

  // Have pipelines compute interior of local domain

  interior_div_e_loop_kokkos(kernel, g);

  // While pipelines are busy, have host compute the exterior
  // of the local domain

  // Finish setting normal e ghosts

  k_end_remote_ghost_norm_e( fa, g );

  exterior_div_e_loop_kokkos(kernel, g);

  // Finish up setting interior

  k_local_adjust_div_e( fa, g );

}

void
compute_div_e_err(
  field_array_t * RESTRICT fa
)
{

  if( !fa )
  {
    ERROR(( "Bad args" ));
  }

  // Have pipelines compute the interior of local domain (the host
  // handles stragglers in the interior)

  sfa_params_t* sfa_p = reinterpret_cast<sfa_params_t*>(fa->params);

  const grid_t * g = fa->g;

  k_field_t& k_field = fa->k_f_d;
  k_field_edge_t& k_field_edge = fa->k_fe_d;
  k_material_coefficient_t& k_matcoeff = sfa_p->k_mc_d;

  SELECT_GEOMETRY(g->geometry, geo, ({

    using geo_t = GeometryClass<geo>::device;
    geo_t geometry = g->get_device_geometry<geo>();

    if( k_matcoeff.extent(0) == 1 ) {

      // Optimize for vacuum.
      VacuumMaterialId vac;
      auto kernel = div_e_kernel<geo_t, VacuumMaterialId>(
        geometry,
        k_field,
        vac,
        k_matcoeff,
        g->eps0
      );

      div_e_loop_kokkos(kernel, fa, g);


    } else {

      auto kernel = div_e_kernel<geo_t, k_field_edge_t>(
        geometry,
        k_field,
        k_field_edge,
        k_matcoeff,
        g->eps0
      );

      div_e_loop_kokkos(kernel, fa, g);

    }

  }));

}

void
compute_rhob(
  field_array_t * RESTRICT fa
)
{

  if( !fa )
  {
    ERROR(( "Bad args" ));
  }

  // Have pipelines compute the interior of local domain (the host
  // handles stragglers in the interior)

  sfa_params_t* sfa_p = reinterpret_cast<sfa_params_t*>(fa->params);

  const grid_t * g = fa->g;

  k_field_t& k_field = fa->k_f_d;
  k_field_edge_t& k_field_edge = fa->k_fe_d;
  k_material_coefficient_t& k_matcoeff = sfa_p->k_mc_d;

  SELECT_GEOMETRY(g->geometry, geo, ({

    using geo_t = GeometryClass<geo>::device;
    geo_t geometry = g->get_device_geometry<geo>();

    if( k_matcoeff.extent(0) == 1 ) {

      // Optimize for vacuum.
      VacuumMaterialId vac;
      auto kernel = rhob_kernel<geo_t, VacuumMaterialId>(
        geometry,
        k_field,
        vac,
        k_matcoeff,
        g->eps0
      );

      div_e_loop_kokkos(kernel, fa, g);


    } else {

      auto kernel = rhob_kernel<geo_t, k_field_edge_t>(
        geometry,
        k_field,
        k_field_edge,
        k_matcoeff,
        g->eps0
      );

      div_e_loop_kokkos(kernel, fa, g);

    }

  }));

}
