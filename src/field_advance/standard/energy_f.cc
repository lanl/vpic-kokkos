// FIXME: USE THE DISCRETIZED VARIATIONAL PRINCIPLE DEFINITION OF ENERGY

#define IN_sfa
#include "sfa_private.h"

template<class geo_type>
struct field_reduce {
    typedef double value_type[];
    typedef k_field_t::size_type size_type;

    const geo_type geometry;
    k_field_t k_field;
    k_field_edge_t k_field_edge;
    k_material_coefficient_t k_mat;
    const float eps_half;
    int nx, ny, nz;
    size_type value_count;

    field_reduce(
        const geo_type geometry_,
        const k_field_t k_field_,
        const k_field_edge_t k_field_edge_,
        const k_material_coefficient_t k_mat_,
        const float eps_half_,
        const int nx_, const int ny_, const int nz_
    )
    : geometry(geometry_),
      k_field(k_field_),
      k_field_edge(k_field_edge_),
      k_mat(k_mat_),
      eps_half(eps_half_),
      nx(nx_),
      ny(ny_),
      nz(nz_)
    {
        value_count = 6;
    }

    KOKKOS_INLINE_FUNCTION void
    operator() (const size_type z, const size_type y, const size_type x, value_type en) const {
        const int f0 =  VOXEL(x,   y,   z,   nx,ny,nz);
        const int fx =  VOXEL(x+1, y,   z,   nx,ny,nz);
        const int fy =  VOXEL(x,   y+1, z,   nx,ny,nz);
        const int fz =  VOXEL(x,   y,   z+1, nx,ny,nz);
        const int fyz = VOXEL(x,   y+1, z+1, nx,ny,nz);
        const int fzx = VOXEL(x+1, y,   z+1, nx,ny,nz);
        const int fxy = VOXEL(x+1, y+1, z,   nx,ny,nz);

        float w0, w1, w2, w3;
        const float epsdV_half = eps_half / geometry.inverse_voxel_volume(f0);

        field_vectors_t f;

        w0 = k_mat(k_field_edge(f0,  field_edge_var::ematx), material_coeff_var::epsx) * k_field(f0,  field_var::ex) * k_field(f0,  field_var::ex) ;
        w1 = k_mat(k_field_edge(fy,  field_edge_var::ematx), material_coeff_var::epsx) * k_field(fy,  field_var::ex) * k_field(fy,  field_var::ex) ;
        w2 = k_mat(k_field_edge(fz,  field_edge_var::ematx), material_coeff_var::epsx) * k_field(fz,  field_var::ex) * k_field(fz,  field_var::ex) ;
        w3 = k_mat(k_field_edge(fyz, field_edge_var::ematx), material_coeff_var::epsx) * k_field(fyz, field_var::ex) * k_field(fyz, field_var::ex) ;

        geometry.prescale_interpolated_ex(f0, w0, w1, w2, w3);
        f.ex = 0.25 * (w0 + w1 + w2 + w3);

        w0 = k_mat(k_field_edge(f0,  field_edge_var::ematy), material_coeff_var::epsy) * k_field(f0,  field_var::ey) * k_field(f0,  field_var::ey) ;
        w1 = k_mat(k_field_edge(fz,  field_edge_var::ematy), material_coeff_var::epsy) * k_field(fz,  field_var::ey) * k_field(fz,  field_var::ey) ;
        w2 = k_mat(k_field_edge(fx,  field_edge_var::ematy), material_coeff_var::epsy) * k_field(fx,  field_var::ey) * k_field(fx,  field_var::ey) ;
        w3 = k_mat(k_field_edge(fzx, field_edge_var::ematy), material_coeff_var::epsy) * k_field(fzx, field_var::ey) * k_field(fzx, field_var::ey) ;

        geometry.prescale_interpolated_ey(f0, w0, w2, w1, w3);
        f.ey = 0.25 * (w0 + w1 + w2 + w3);

        w0 = k_mat(k_field_edge(f0,  field_edge_var::ematz), material_coeff_var::epsz) * k_field(f0,  field_var::ez) * k_field(f0,  field_var::ez) ;
        w1 = k_mat(k_field_edge(fx,  field_edge_var::ematz), material_coeff_var::epsz) * k_field(fx,  field_var::ez) * k_field(fx,  field_var::ez) ;
        w2 = k_mat(k_field_edge(fy,  field_edge_var::ematz), material_coeff_var::epsz) * k_field(fy,  field_var::ez) * k_field(fy,  field_var::ez) ;
        w3 = k_mat(k_field_edge(fxy, field_edge_var::ematz), material_coeff_var::epsz) * k_field(fxy, field_var::ez) * k_field(fxy, field_var::ez) ;

        geometry.prescale_interpolated_ez(f0, w0, w2, w1, w3);
        f.ez = 0.25 * (w0 + w1 + w2 + w3);

        w0 = k_mat(k_field_edge(f0, field_edge_var::fmatx), material_coeff_var::rmux) * k_field(f0, field_var::cbx) * k_field(f0, field_var::cbx) ;
        w1 = k_mat(k_field_edge(fx, field_edge_var::fmatx), material_coeff_var::rmux) * k_field(fx, field_var::cbx) * k_field(fx, field_var::cbx) ;

        geometry.prescale_interpolated_cbx(f0, w0, w1);
        f.cbx = 0.5 * (w0 + w1);

        w0 = k_mat(k_field_edge(f0, field_edge_var::fmaty), material_coeff_var::rmuy) * k_field(f0, field_var::cby) * k_field(f0, field_var::cby) ;
        w1 = k_mat(k_field_edge(fy, field_edge_var::fmaty), material_coeff_var::rmuy) * k_field(fy, field_var::cby) * k_field(fy, field_var::cby) ;

        geometry.prescale_interpolated_cby(f0, w0, w1);
        f.cby = 0.5 * (w0 + w1);

        w0 = k_mat(k_field_edge(f0, field_edge_var::fmatz), material_coeff_var::rmuz) * k_field(f0, field_var::cbz) * k_field(f0, field_var::cbz) ;
        w1 = k_mat(k_field_edge(fz, field_edge_var::fmatz), material_coeff_var::rmuz) * k_field(fz, field_var::cbz) * k_field(fz, field_var::cbz) ;

        geometry.prescale_interpolated_cbz(f0, w0, w1);
        f.cbz = 0.5 * (w0 + w1);

        geometry.postscale_interpolated_fields(f, f0, 0.0, 0.0, 0.0);

        en[0] += epsdV_half * f.ex;
        en[1] += epsdV_half * f.ey;
        en[2] += epsdV_half * f.ez;
        en[3] += epsdV_half * f.cbx;
        en[4] += epsdV_half * f.cby;
        en[5] += epsdV_half * f.cbz;

    }

    KOKKOS_INLINE_FUNCTION void
    join(volatile value_type dst, const volatile value_type src) const {
        for(size_type i = 0; i < 6; i++) {
            dst[i] += src[i];
        }
    }

    KOKKOS_INLINE_FUNCTION void
    init(value_type sums) const {
        for(size_type i=0; i<6; i++) {
            sums[i] = 0.0f;
        }
    }
};

void energy_f(
  double* global,
  const field_array_t* RESTRICT fa
)
{
    if( !fa ) ERROR(( "Bad args" ));

    double en[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({1,1,1}, {nz+1,ny+1,nx+1});
    sfa_params_t* sfa = reinterpret_cast<sfa_params_t*>(fa->params);

    const float eps_half = 0.5*fa->g->eps0;

    SELECT_GEOMETRY(fa->g->geometry, geo, ({

        auto geometry = fa->g->get_device_geometry<geo>();

        field_reduce<decltype(geometry)> field_reducer(
            geometry,
            fa->k_f_d,
            fa->k_fe_d,
            sfa->k_mc_d,
            eps_half,
            nx, ny, nz
        );

        Kokkos::parallel_reduce(
            "field energy reduction",
            policy,
            field_reducer,
            en
        );

    }));

    mp_allsum_d( en, global, 6 );

}

