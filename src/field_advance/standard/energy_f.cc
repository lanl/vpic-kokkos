// FIXME: USE THE DISCRETIZED VARIATIONAL PRINCIPLE DEFINITION OF ENERGY

#define IN_sfa
#include "sfa_private.h"

struct field_reduce {
    typedef double value_type[];
    typedef k_field_t::size_type size_type;

    k_field_t k_field;
    k_field_edge_t k_field_edge;
    k_material_coefficient_t k_mat;
    int nx, ny, nz;
    size_type value_count;

    field_reduce(const k_field_t k_field_, const k_field_edge_t k_field_edge_, const k_material_coefficient_t k_mat_, const int nx_, const int ny_, const int nz_) : k_field(k_field_), k_field_edge(k_field_edge_), k_mat(k_mat_), nx(nx_), ny(ny_), nz(nz_) {value_count = 6;}

    KOKKOS_INLINE_FUNCTION void
    operator() (const size_type z, const size_type y, const size_type x, value_type en) const {
        const int f0 =  VOXEL(x,   y,   z,   nx,ny,nz);
        const int fx =  VOXEL(x+1, y,   z,   nx,ny,nz);
        const int fy =  VOXEL(x,   y+1, z,   nx,ny,nz);
        const int fz =  VOXEL(x,   y,   z+1, nx,ny,nz);
        const int fyz = VOXEL(x,   y+1, z+1, nx,ny,nz);
        const int fzx = VOXEL(x+1, y,   z+1, nx,ny,nz);
        const int fxy = VOXEL(x+1, y+1, z,   nx,ny,nz);
        en[0] += 0.25*( k_mat(k_field_edge(f0,  field_edge_var::ematx), material_coeff_var::epsx) * k_field(f0,  field_var::ex) * k_field(f0,  field_var::ex) +
                        k_mat(k_field_edge(fy,  field_edge_var::ematx), material_coeff_var::epsx) * k_field(fy,  field_var::ex) * k_field(fy,  field_var::ex) +
                        k_mat(k_field_edge(fz,  field_edge_var::ematx), material_coeff_var::epsx) * k_field(fz,  field_var::ex) * k_field(fz,  field_var::ex) +
                        k_mat(k_field_edge(fyz, field_edge_var::ematx), material_coeff_var::epsx) * k_field(fyz, field_var::ex) * k_field(fyz, field_var::ex) );

        en[1] += 0.25*( k_mat(k_field_edge(f0,  field_edge_var::ematy), material_coeff_var::epsy) * k_field(f0,  field_var::ey) * k_field(f0,  field_var::ey) +
                        k_mat(k_field_edge(fz,  field_edge_var::ematy), material_coeff_var::epsy) * k_field(fz,  field_var::ey) * k_field(fz,  field_var::ey) +
                        k_mat(k_field_edge(fx,  field_edge_var::ematy), material_coeff_var::epsy) * k_field(fx,  field_var::ey) * k_field(fx,  field_var::ey) +
                        k_mat(k_field_edge(fzx, field_edge_var::ematy), material_coeff_var::epsy) * k_field(fzx, field_var::ey) * k_field(fzx, field_var::ey) );

        en[2] += 0.25*( k_mat(k_field_edge(f0,  field_edge_var::ematz), material_coeff_var::epsz) * k_field(f0,  field_var::ez) * k_field(f0,  field_var::ez) +
                        k_mat(k_field_edge(fx,  field_edge_var::ematz), material_coeff_var::epsz) * k_field(fx,  field_var::ez) * k_field(fx,  field_var::ez) +
                        k_mat(k_field_edge(fy,  field_edge_var::ematz), material_coeff_var::epsz) * k_field(fy,  field_var::ez) * k_field(fy,  field_var::ez) +
                        k_mat(k_field_edge(fxy, field_edge_var::ematz), material_coeff_var::epsz) * k_field(fxy, field_var::ez) * k_field(fxy, field_var::ez) );

        en[3] += 0.5*(  k_mat(k_field_edge(f0, field_edge_var::fmatx), material_coeff_var::rmux) * k_field(f0, field_var::cbx) * k_field(f0, field_var::cbx) +
                        k_mat(k_field_edge(fx, field_edge_var::fmatx), material_coeff_var::rmux) * k_field(fx, field_var::cbx) * k_field(fx, field_var::cbx) );

        en[4] += 0.5*(  k_mat(k_field_edge(f0, field_edge_var::fmaty), material_coeff_var::rmuy) * k_field(f0, field_var::cby) * k_field(f0, field_var::cby) +
                        k_mat(k_field_edge(fy, field_edge_var::fmaty), material_coeff_var::rmuy) * k_field(fy, field_var::cby) * k_field(fy, field_var::cby) );

        en[5] += 0.5*(  k_mat(k_field_edge(f0, field_edge_var::fmatz), material_coeff_var::rmuz) * k_field(f0, field_var::cbz) * k_field(f0, field_var::cbz) +
                        k_mat(k_field_edge(fz, field_edge_var::fmatz), material_coeff_var::rmuz) * k_field(fz, field_var::cbz) * k_field(fz, field_var::cbz) );
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

    field_reduce field_reducer(fa->k_f_d, fa->k_fe_d, sfa->k_mc_d, nx, ny, nz);
    Kokkos::parallel_reduce("field energy reduction", policy, field_reducer, en);

    double v0 = 0.5*fa->g->eps0*fa->g->dV;
    for(int i=0; i<6; i++) {
        en[i] *= v0;
    }
    mp_allsum_d( en, global, 6 );
}

