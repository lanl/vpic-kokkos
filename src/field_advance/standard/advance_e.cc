// Note: This is similar to compute_curl_b

#define IN_sfa
#include "sfa_private.h"
#include <Kokkos_Core.hpp>

template<class geo_t, class edge_t>
class advance_e_kerenls {
public:

  advance_e_kerenls (
    geo_t& geometry,
    k_field_t& k_field,
    edge_t& k_field_edge,
    const k_material_coefficient_t& k_material,
    const float dt,
    const float damp,
    const float cj
  )
  : geometry(geometry),
    k_field(k_field),
    k_field_edge(k_field_edge),
    k_material(k_material),
    dt(dt),
    damp(damp),
    cj(cj)
  {

  }

  KOKKOS_INLINE_FUNCTION void
  update_x(int f0, int fx, int fy, int fz) const {

    using namespace field_var;
    using namespace field_edge_var;
    using namespace material_coeff_var;

    float Hz0 = k_field(f0, cbz) * k_material( k_field_edge(f0, fmatz), rmuz );
    float Hzy = k_field(fy, cbz) * k_material( k_field_edge(fy, fmatz), rmuz );

    float Hy0 = k_field(f0, cby) * k_material( k_field_edge(f0, fmaty), rmuy );
    float Hyz = k_field(fz, cby) * k_material( k_field_edge(fz, fmaty), rmuy );

    float curlHx = geometry.face_curl_x(f0, Hz0, Hzy, Hy0, Hyz);

    float decay = k_material(k_field_edge(f0, ematx), decayx);
    float drive = k_material(k_field_edge(f0, ematx), drivex);

    k_field(f0, tcax) = dt*curlHx - damp*k_field(f0, tcax);
    k_field(f0, ex) = decay*k_field(f0, ex) + drive*(k_field(f0, tcax) - cj*k_field(f0, jfx));

  }

  KOKKOS_INLINE_FUNCTION void
  update_y(int f0, int fx, int fy, int fz) const {

    using namespace field_var;
    using namespace field_edge_var;
    using namespace material_coeff_var;

    float Hx0 = k_field(f0, cbx) * k_material( k_field_edge(f0, fmatx), rmux );
    float Hxz = k_field(fz, cbx) * k_material( k_field_edge(fz, fmatx), rmux );

    float Hz0 = k_field(f0, cbz) * k_material( k_field_edge(f0, fmatz), rmuz );
    float Hzx = k_field(fx, cbz) * k_material( k_field_edge(fx, fmatz), rmuz );

    float curlHy = geometry.face_curl_y(f0, Hx0, Hxz, Hz0, Hzx);

    float decay = k_material(k_field_edge(f0, ematy), decayy);
    float drive = k_material(k_field_edge(f0, ematy), drivey);

    k_field(f0, tcay) = dt*curlHy - damp*k_field(f0, tcay);
    k_field(f0, ey) = decay*k_field(f0, ey) + drive*(k_field(f0, tcay) - cj*k_field(f0, jfy));

  }

  KOKKOS_INLINE_FUNCTION void
  update_z(int f0, int fx, int fy, int fz) const {

    using namespace field_var;
    using namespace field_edge_var;
    using namespace material_coeff_var;

    float Hy0 = k_field(f0, cby) * k_material( k_field_edge(f0, fmaty), rmuy );
    float Hyx = k_field(fx, cby) * k_material( k_field_edge(fx, fmaty), rmuy );

    float Hx0 = k_field(f0, cbx) * k_material( k_field_edge(f0, fmatx), rmux );
    float Hxy = k_field(fy, cbx) * k_material( k_field_edge(fy, fmatx), rmux );

    float curlHz = geometry.face_curl_z(f0, Hy0, Hyx, Hx0, Hxy);

    float decay = k_material(k_field_edge(f0, ematz), decayz);
    float drive = k_material(k_field_edge(f0, ematz), drivez);

    k_field(f0, tcaz) = dt*curlHz - damp*k_field(f0, tcaz);
    k_field(f0, ez) = decay*k_field(f0, ez) + drive*(k_field(f0, tcaz) - cj*k_field(f0, jfz));

  }

  geo_t geometry;
  k_field_t k_field;
  edge_t k_field_edge;
  const k_material_coefficient_t k_material;
  const float dt, damp, cj;

};

template<class geo_t, class edge_t>
class compute_curl_b_kernels {
public:

  compute_curl_b_kernels (
    geo_t& geometry,
    k_field_t& k_field,
    edge_t& k_field_edge,
    const k_material_coefficient_t& k_material
  )
  : geometry(geometry),
    k_field(k_field),
    k_field_edge(k_field_edge),
    k_material(k_material)
  {

  }

  KOKKOS_INLINE_FUNCTION void
  update_x(int f0, int fx, int fy, int fz) const {

    using namespace field_var;
    using namespace field_edge_var;
    using namespace material_coeff_var;

    float Hz0 = k_field(f0, cbz) * k_material( k_field_edge(f0, fmatz), rmuz );
    float Hzy = k_field(fy, cbz) * k_material( k_field_edge(fy, fmatz), rmuz );

    float Hy0 = k_field(f0, cby) * k_material( k_field_edge(f0, fmaty), rmuy );
    float Hyz = k_field(fz, cby) * k_material( k_field_edge(fz, fmaty), rmuy );

    k_field(f0, tcax) = geometry.face_curl_x(f0, Hz0, Hzy, Hy0, Hyz);

  }

  KOKKOS_INLINE_FUNCTION void
  update_y(int f0, int fx, int fy, int fz) const {

    using namespace field_var;
    using namespace field_edge_var;
    using namespace material_coeff_var;

    float Hx0 = k_field(f0, cbx) * k_material( k_field_edge(f0, fmatx), rmux );
    float Hxz = k_field(fz, cbx) * k_material( k_field_edge(fz, fmatx), rmux );

    float Hz0 = k_field(f0, cbz) * k_material( k_field_edge(f0, fmatz), rmuz );
    float Hzx = k_field(fx, cbz) * k_material( k_field_edge(fx, fmatz), rmuz );

    k_field(f0, tcay) = geometry.face_curl_y(f0, Hx0, Hxz, Hz0, Hzx);

  }

  KOKKOS_INLINE_FUNCTION void
  update_z(int f0, int fx, int fy, int fz) const {

    using namespace field_var;
    using namespace field_edge_var;
    using namespace material_coeff_var;

    float Hy0 = k_field(f0, cby) * k_material( k_field_edge(f0, fmaty), rmuy );
    float Hyx = k_field(fx, cby) * k_material( k_field_edge(fx, fmaty), rmuy );

    float Hx0 = k_field(f0, cbx) * k_material( k_field_edge(f0, fmatx), rmux );
    float Hxy = k_field(fy, cbx) * k_material( k_field_edge(fy, fmatx), rmux );

    k_field(f0, tcaz) = geometry.face_curl_z(f0, Hy0, Hyx, Hx0, Hxy);

  }

  geo_t geometry;
  k_field_t k_field;
  edge_t k_field_edge;
  const k_material_coefficient_t k_material;

};


template<class kernels_t> void
dispatch_advance_e(
  std::string name,
  kernels_t& kernels,
  field_array_t* RESTRICT fa
)
{

  const int nx = fa->g->nx;
  const int ny = fa->g->ny;
  const int nz = fa->g->nz;

  /***************************************************************************
  * Begin tangential B ghost setup
  ***************************************************************************/

  kokkos_begin_remote_ghost_tang_b( fa, fa->g, *(fa->fb) );

  k_local_ghost_tang_b( fa, fa->g );

  /***************************************************************************
  * Update interior
  ***************************************************************************/

  // Do the majority of the interior
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({2, 2, 2}, {nz+1, ny+1, nx+1});
  Kokkos::parallel_for(name + ": Majority of interior", zyx_policy,
    KOKKOS_LAMBDA(const int z, const int y, const int x) {
      const size_t f0_idx = VOXEL(x,   y,   z,   nx, ny, nz);
      const size_t fx_idx = VOXEL(x-1, y,   z,   nx, ny, nz);
      const size_t fy_idx = VOXEL(x,   y-1, z,   nx, ny, nz);
      const size_t fz_idx = VOXEL(x,   y,   z-1, nx, ny, nz);
      kernels.update_x(f0_idx, fx_idx, fy_idx, fz_idx);
      kernels.update_y(f0_idx, fx_idx, fy_idx, fz_idx);
      kernels.update_z(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  // Do left over interior ex
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_policy({2, 2}, {nz+1, ny+1});
  Kokkos::parallel_for(name + ": left over interior ex", ex_policy,
    KOKKOS_LAMBDA(const int z, const int y) {
      const size_t f0_idx = VOXEL(1, y,   z, nx, ny, nz);
      const size_t fx_idx = 0;
      const size_t fy_idx = VOXEL(1, y-1, z, nx, ny, nz);
      const size_t fz_idx = VOXEL(1, y,   z-1, nx, ny ,nz);
      kernels.update_x(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  // Do left over interior ey
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_policy({2, 2}, {nz+1, nx+1});
  Kokkos::parallel_for(name + ": left over interior ey", ey_policy,
    KOKKOS_LAMBDA(const int z, const int x) {
      const size_t f0_idx = VOXEL(2, 1, z, nx, ny, nz) + (x-2);
      const size_t fx_idx = VOXEL(1, 1, z, nx, ny, nz) + (x-2);
      const size_t fy_idx = 0;
      const size_t fz_idx = VOXEL(2, 1, z-1, nx, ny ,nz) + (x-2);
      kernels.update_y(f0_idx, fx_idx, fy_idx, fz_idx);
  });

  // Do left over interior ez
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_policy({2, 2}, {ny+1, nx+1});
  Kokkos::parallel_for(name + ": left over interior ez", ez_policy,
    KOKKOS_LAMBDA(const int y, const int x) {
      const size_t f0_idx = VOXEL(2, y,   1, nx, ny, nz) + (x-2);
      const size_t fx_idx = VOXEL(1, y,   1, nx, ny, nz) + (x-2);
      const size_t fy_idx = VOXEL(2, y-1, 1, nx, ny, nz) + (x-2);
      const size_t fz_idx = 0;
      kernels.update_z(f0_idx, fx_idx, fy_idx, fz_idx);
  });

  /***************************************************************************
  * Finish tangential B ghost setup
  ***************************************************************************/

  kokkos_end_remote_ghost_tang_b( fa, fa->g, *(fa->fb) );

  /***************************************************************************
  * Update exterior fields
  ***************************************************************************/

  // Do exterior ex
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_yx_policy({1, 1}, {ny+2, nx+1});
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ex_zx_policy({2, 1}, {nz+1, nx+1});
  Kokkos::parallel_for(name + ": exterior ex loop 1", ex_yx_policy,
    KOKKOS_LAMBDA(const int y, const int x) {
      const size_t f0_idx = VOXEL(1, y,   1,nx,ny,nz) + (x-1);
      const size_t fx_idx = 0;
      const size_t fy_idx = VOXEL(1, y-1, 1,nx,ny,nz) + (x-1);
      const size_t fz_idx = VOXEL(1, y,   0,nx,ny,nz) + (x-1);
      kernels.update_x(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  Kokkos::parallel_for(name + ": exterior ex loop 2", ex_yx_policy,
    KOKKOS_LAMBDA(const int y, const int x) {
      const size_t f0_idx = VOXEL(1,y,  nz+1, nx,ny,nz) + (x-1);
      const size_t fx_idx = 0;
      const size_t fy_idx = VOXEL(1,y-1,nz+1, nx,ny,nz) + (x-1);
      const size_t fz_idx = VOXEL(1,y,  nz,   nx,ny,nz) + (x-1);
      kernels.update_x(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  Kokkos::parallel_for(name + ": exterior ex loop 3", ex_zx_policy,
    KOKKOS_LAMBDA(const int z, const int x) {
      const size_t f0_idx = VOXEL(1,1,z,nx,ny,nz) + (x-1);
      const size_t fx_idx = 0; // Don't care about x index, not used in update_ex anyway.
      const size_t fy_idx = VOXEL(1,0,z,nx,ny,nz) + (x-1);
      const size_t fz_idx = VOXEL(1,1,z-1,nx,ny,nz) + (x-1);
      kernels.update_x(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  Kokkos::parallel_for(name + ": exterior ex loop 4", ex_zx_policy,
    KOKKOS_LAMBDA(const int z, const int x) {
      const size_t f0_idx = VOXEL(1,ny+1,z,nx,ny,nz) + (x-1);
      const size_t fx_idx = 0; // Don't care about x index, not used in update_ex anyway.
      const size_t fy_idx = VOXEL(1,ny,z,nx,ny,nz) + (x-1);
      const size_t fz_idx = VOXEL(1,ny+1,z-1,nx,ny,nz) + (x-1);
      kernels.update_x(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  // Do exterior ey
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_zy_policy({1, 1}, {nz+2, ny+1});
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ey_yx_policy({1, 2}, {ny+1, nx+1});
  Kokkos::parallel_for(name + ": exterior ey loop 1", ey_zy_policy,
    KOKKOS_LAMBDA(const int z, const int y) {
      const size_t f0_idx = VOXEL(1,y,z,nx,ny,nz);
      const size_t fx_idx = VOXEL(0,y,z,nx,ny,nz);
      const size_t fy_idx = 0;
      const size_t fz_idx = VOXEL(1,y,z-1,nx,ny,nz);
      kernels.update_y(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  Kokkos::parallel_for(name + ": exterior ey loop 2", ey_zy_policy,
    KOKKOS_LAMBDA(const int z, const int y) {
      const size_t f0_idx = VOXEL(nx+1,y,z,nx,ny,nz);
      const size_t fx_idx = VOXEL(nx,y,z,nx,ny,nz);
      const size_t fy_idx = 0;
      const size_t fz_idx = VOXEL(nx+1,y,z-1,nx,ny,nz);
      kernels.update_y(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  Kokkos::parallel_for(name + ": exterior ey loop 3", ey_yx_policy,
    KOKKOS_LAMBDA(const int y, const int x) {
      const size_t f0_idx = VOXEL(2,y,1,nx,ny,nz) + (x-2);
      const size_t fx_idx = VOXEL(1,y,1,nx,ny,nz) + (x-2);
      const size_t fy_idx = 0;
      const size_t fz_idx = VOXEL(2,y,0,nx,ny,nz) + (x-2);
      kernels.update_y(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  Kokkos::parallel_for(name + ": exterior ey loop 4", ey_yx_policy,
    KOKKOS_LAMBDA(const int y, const int x) {
      const size_t f0_idx = VOXEL(2,y,nz+1,nx,ny,nz) + (x-2);
      const size_t fx_idx = VOXEL(1,y,nz+1,nx,ny,nz) + (x-2);
      const size_t fy_idx = 0;
      const size_t fz_idx = VOXEL(2,y,nz,nx,ny,nz) + (x-2);
      kernels.update_y(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  // Do exterior ez
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zx_policy({1, 1}, {nz+1, nx+2});
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ez_zy_policy({1, 2}, {nz+1, ny+1});
  Kokkos::parallel_for(name + ": exterior ez loop 1", ez_zx_policy,
    KOKKOS_LAMBDA(const int z, const int x) {
      const size_t f0_idx = VOXEL(1,1,z,nx,ny,nz) + (x-1);
      const size_t fx_idx = VOXEL(0,1,z,nx,ny,nz) + (x-1);
      const size_t fy_idx = VOXEL(1,0,z,nx,ny,nz) + (x-1);
      const size_t fz_idx = 0;
      kernels.update_z(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  Kokkos::parallel_for(name + ": exterior ez loop 2", ez_zx_policy,
    KOKKOS_LAMBDA(const int z, const int x) {
      const size_t f0_idx = VOXEL(1,ny+1,z,nx,ny,nz) + (x-1);
      const size_t fx_idx = VOXEL(0,ny+1,z,nx,ny,nz) + (x-1);
      const size_t fy_idx = VOXEL(1,ny  ,z,nx,ny,nz) + (x-1);
      const size_t fz_idx = 0;
      kernels.update_z(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  Kokkos::parallel_for(name + ": exterior ez loop 3", ez_zy_policy,
    KOKKOS_LAMBDA(const int z, const int y) {
      const size_t f0_idx = VOXEL(1,y,z,nx,ny,nz);
      const size_t fx_idx = VOXEL(0,y,z,nx,ny,nz);
      const size_t fy_idx = VOXEL(1,y-1,z,nx,ny,nz);
      const size_t fz_idx = 0;
      kernels.update_z(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  Kokkos::parallel_for(name + ": exterior ez loop 4", ez_zy_policy,
    KOKKOS_LAMBDA(const int z, const int y) {
      const size_t f0_idx = VOXEL(nx+1, y,   z,nx,ny,nz);
      const size_t fx_idx = VOXEL(nx,   y,   z,nx,ny,nz);
      const size_t fy_idx = VOXEL(nx+1, y-1, z,nx,ny,nz);
      const size_t fz_idx = 0;
      kernels.update_z(f0_idx, fx_idx, fy_idx, fz_idx);
    });

  /***************************************************************************
  * Adjust tangential e
  ***************************************************************************/

  k_local_adjust_tang_e( fa, fa->g );

}


void advance_e(
  field_array_t* RESTRICT fa,
  float frac
)
{

  if( !fa     ) ERROR(( "Bad args" ));
  if( frac!=1 ) ERROR(( "standard advance_e does not support frac!=1 yet" ));

  sfa_params_t* sfa = reinterpret_cast<sfa_params_t *>(fa->params);
  k_field_t k_field = fa->k_f_d;
  k_field_edge_t k_field_edge = fa->k_fe_d;

  const grid_t *g = fa->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;

  const float damp = sfa->damp;
  const float dt   = (1+damp)*g->cvac*g->dt;
  const float cj   = g->dt/g->eps0;

  const k_material_coefficient_t& k_material = sfa->k_mc_d;

  SELECT_GEOMETRY(g->geometry, geo, ({

    using geo_t = GeometryClass<geo>::device;
    geo_t geometry = g->get_device_geometry<geo>();

    if( k_material.extent(0) == 1 ) {

      VacuumMaterialId vac;
      auto kernels = advance_e_kerenls<geo_t, VacuumMaterialId>(
        geometry,
        k_field,
        vac,   // Optimize for vacuum
        k_material,
        dt,
        damp,
        cj
      );

      dispatch_advance_e("advance_e", kernels, fa);

    } else {

      auto kernels = advance_e_kerenls<geo_t, k_field_edge_t>(
        geometry,
        k_field,
        k_field_edge,
        k_material,
        dt,
        damp,
        cj
      );

      dispatch_advance_e("advance_e", kernels, fa);

    }

  }));

}


void compute_curl_b_kokkos(
  field_array_t* RESTRICT fa
)
{

  if( !fa )
  {
    ERROR(( "Bad args" ));
  }

  k_field_t k_field = fa->k_f_d;
  k_field_edge_t k_field_edge = fa->k_fe_d;
  sfa_params_t* sfa = reinterpret_cast<sfa_params_t *>(fa->params);

  const grid_t *g = fa->g;
  const k_material_coefficient_t& k_material = sfa->k_mc_d;

  SELECT_GEOMETRY(g->geometry, geo, ({

    using geo_t = GeometryClass<geo>::device;
    geo_t geometry = g->get_device_geometry<geo>();

    if( k_material.extent(0) == 1 ) {

      VacuumMaterialId vac;
      auto kernels = compute_curl_b_kernels<geo_t, VacuumMaterialId>(
        geometry,
        k_field,
        vac,   // Optimize for vacuum
        k_material
      );

      dispatch_advance_e("compute_curl_b", kernels, fa);

    } else {

      auto kernels = compute_curl_b_kernels<geo_t, k_field_edge_t>(
        geometry,
        k_field,
        k_field_edge,
        k_material
      );

      dispatch_advance_e("compute_curl_b", kernels, fa);

    }

  }));

}

