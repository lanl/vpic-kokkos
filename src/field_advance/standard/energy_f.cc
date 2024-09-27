// FIXME: USE THE DISCRETIZED VARIATIONAL PRINCIPLE DEFINITION OF ENERGY

#define IN_sfa
#include "sfa_private.h"

typedef struct pipeline_args {
  const field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
  double en[MAX_PIPELINE+1][6];
} pipeline_args_t;

#define DECLARE_STENCIL()                                                  \
  const field_t                * ALIGNED(128) f = args->f;                 \
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;             \
  const grid_t                 *              g = args->g;                 \
  const int nx = g->nx, ny = g->ny, nz = g->nz;                            \
                                                                           \
  const field_t * ALIGNED(16) f0;                                          \
  const field_t * ALIGNED(16) fx,  * ALIGNED(16) fy,  * ALIGNED(16) fz;    \
  const field_t * ALIGNED(16) fyz, * ALIGNED(16) fzx, * ALIGNED(16) fxy;   \
  double en_ex = 0, en_ey = 0, en_ez = 0, en_bx = 0, en_by = 0, en_bz = 0; \
  int x, y, z

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

#define INIT_STENCIL()   \
  f0  = &f(x,  y,  z  ); \
  fx  = &f(x+1,y,  z  ); \
  fy  = &f(x,  y+1,z  ); \
  fz  = &f(x,  y,  z+1); \
  fyz = &f(x,  y+1,z+1); \
  fzx = &f(x+1,y,  z+1); \
  fxy = &f(x+1,y+1,z  )

#define NEXT_STENCIL()                              \
  f0++; fx++; fy++; fz++; fyz++; fzx++; fxy++; x++; \
  if( x>nx ) {                                      \
    /**/       y++;            x = 1;               \
    if( y>ny ) z++; if( y>ny ) y = 1;               \
    INIT_STENCIL();                                 \
  }

#define REDUCE_EN()                                       \
  en_ex += f0->ex * f0->ex ;  \
  en_ey += f0->ey * f0->ey ;  \
  en_ez += f0->ez * f0->ez ;  \
  en_bx += (f0->cbx + f0->cbx0) * (f0->cbx + f0->cbx0); \
  en_by += (f0->cby + f0->cby0) * (f0->cby + f0->cby0); \
  en_bz += (f0->cbz + f0->cbz0) * (f0->cbz + f0->cbz0);

void
energy_f_pipeline( pipeline_args_t * args,
                   int pipeline_rank,
                   int n_pipeline ) {
  DECLARE_STENCIL();

  int n_voxel;
  DISTRIBUTE_VOXELS( 1,nx, 1,ny, 1,nz, 16,
                     pipeline_rank, n_pipeline,
                     x, y, z, n_voxel );

  INIT_STENCIL();
  for( ; n_voxel; n_voxel-- ) {
    REDUCE_EN();
    NEXT_STENCIL();
  }

  args->en[pipeline_rank][0] = 0.5*en_ex;
  args->en[pipeline_rank][1] = 0.5*en_ey;
  args->en[pipeline_rank][2] = 0.5*en_ez;
  args->en[pipeline_rank][3] = 0.5*en_bx;
  args->en[pipeline_rank][4] = 0.5*en_by;
  args->en[pipeline_rank][5] = 0.5*en_bz;
}

#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

#error "Not implemented"

#endif

void
energy_f( double              *          global,
          const field_array_t * RESTRICT fa ) {
  if( !global || !fa ) ERROR(( "Bad args" ));

  // Have each pipeline and the host handle a portion of the
  // local voxels

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  EXEC_PIPELINES( energy_f, args, 0 );
  WAIT_PIPELINES();

  // Reduce results from each pipelines

  int p;
  for( p=1; p<=N_PIPELINE; p++ ) {
    args->en[0][0] += args->en[p][0]; args->en[0][1] += args->en[p][1];
    args->en[0][2] += args->en[p][2]; args->en[0][3] += args->en[p][3];
    args->en[0][4] += args->en[p][4]; args->en[0][5] += args->en[p][5];
  }

  // Convert to physical units and reduce results between nodes

  double v0 = 0.5*fa->g->eps0*fa->g->dV;
  args->en[0][0] *= v0; args->en[0][1] *= v0;
  args->en[0][2] *= v0; args->en[0][3] *= v0;
  args->en[0][4] *= v0; args->en[0][5] *= v0;

  // Reduce results between nodes

  mp_allsum_d( args->en[0], global, 6 );
}

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
      
        en[0] += k_field(f0,  field_var::ex) * k_field(f0,  field_var::ex);
        en[1] += k_field(f0,  field_var::ey) * k_field(f0,  field_var::ey);
        en[2] += k_field(f0,  field_var::ez) * k_field(f0,  field_var::ez);
        en[3] += k_field(f0,  field_var::cbx) * k_field(f0,  field_var::cbx);
        en[4] += k_field(f0,  field_var::cby) * k_field(f0,  field_var::cby);
        en[5] += k_field(f0,  field_var::cbz) * k_field(f0,  field_var::cbz);
        }

   KOKKOS_INLINE_FUNCTION void
    join(value_type dst, const value_type src) const {
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

void energy_f_kokkos(double* global, const field_array_t* RESTRICT fa) {
    if( !fa ) ERROR(( "Bad args" ));

    double en[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int nx = fa->g->nx, ny = fa->g->ny, nz = fa->g->nz;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({1,1,1}, {nz+1,ny+1,nx+1});
    sfa_params_t* sfa = reinterpret_cast<sfa_params_t*>(fa->params);

    field_reduce field_reducer(fa->k_f_d, fa->k_fe_d, sfa->k_mc_d, nx, ny, nz);
    Kokkos::parallel_reduce("field energy reduction", policy, field_reducer, en);

    double v0 = 0.5*fa->g->dV;
    for(int i=0; i<6; i++) {
        en[i] *= v0;
    }
    mp_allsum_d( en, global, 6 );
}

