// FIXME: USE THE DISCRETIZED VARIATIONAL PRINCIPLE DEFINITION OF ENERGY

#define IN_sfa
#include "sfa_private.h"

typedef struct pipeline_args {
  const field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
  double en[MAX_PIPELINE+1][7];
} pipeline_args_t;

#define DECLARE_STENCIL()						\
  const field_t                * ALIGNED(128) f = args->f;		\
  const material_coefficient_t * ALIGNED(128) m = args->p->mc;		\
  const grid_t                 *              g = args->g;		\
  const int nx = g->nx, ny = g->ny, nz = g->nz;				\
  									\
  const field_t * ALIGNED(16) f0;					\
  double en_ex = 0, en_ey = 0, en_ez = 0, en_bx = 0, en_by = 0, en_bz = 0, en_te = 0; \
  int x, y, z

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

#define INIT_STENCIL()   \
  f0  = &f(x,  y,  z  );
  
#define NEXT_STENCIL()                              \
  f0++; x++;					    \
  if( x>nx ) {                                      \
    /**/       y++;            x = 1;               \
    if( y>ny ) z++; if( y>ny ) y = 1;               \
    INIT_STENCIL();                                 \
  }

#define REDUCE_EN()                                       \
  en_ex += f0->ex  * f0->ex ;				  \
  en_ey += f0->ey  * f0->ey ;				  \
  en_ez += f0->ez  * f0->ez ;				  \
  en_bx += f0->cbx * f0->cbx ;				  \
  en_by += f0->cby * f0->cby ;				  \
  en_bz += f0->cbz * f0->cbz ;				  \
  en_te += g->te   * pow(f0->rhof/g->den,g->gamma);
  
 
void
hyb_energy_f_pipeline( pipeline_args_t * args,
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

  args->en[pipeline_rank][0] = en_ex;
  args->en[pipeline_rank][1] = en_ey;
  args->en[pipeline_rank][2] = en_ez;
  args->en[pipeline_rank][3] = en_bx;
  args->en[pipeline_rank][4] = en_by;
  args->en[pipeline_rank][5] = en_bz;
  args->en[pipeline_rank][6] = en_te;

}

#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

#error "Not implemented"

#endif

void
hyb_energy_f( double              *          global,
          const field_array_t * RESTRICT fa ) {
  if( !global || !fa ) ERROR(( "Bad args" ));

  // Have each pipeline and the host handle a portion of the
  // local voxels
  
  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  EXEC_PIPELINES( hyb_energy_f, args, 0 );
  WAIT_PIPELINES();

  // Reduce results from each pipelines
  
  int p;
  for( p=1; p<=N_PIPELINE; p++ ) {
    args->en[0][0] += args->en[p][0]; args->en[0][1] += args->en[p][1];
    args->en[0][2] += args->en[p][2]; args->en[0][3] += args->en[p][3];
    args->en[0][4] += args->en[p][4]; args->en[0][5] += args->en[p][5];
    args->en[0][6] += args->en[p][6];
  }
    
  // Convert to physical units and reduce results between nodes
  
  double v0 = 0.5*fa->g->dV;
  args->en[0][0] *= v0; args->en[0][1] *= v0;
  args->en[0][2] *= v0; args->en[0][3] *= v0;
  args->en[0][4] *= v0; args->en[0][5] *= v0;
  args->en[0][6] *= v0*3.0 /* 3/2 P_e */;

  // Reduce results between nodes

  mp_allsum_d( args->en[0], global, 7 );
}

