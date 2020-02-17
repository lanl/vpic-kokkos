#define IN_sfa
#include "sfa_private.h"

#define f(x,y,z) f[ VOXEL(x,y,z, nx,ny,nz) ]

typedef struct pipeline_args {
  const field_t * ALIGNED(128) f;
  const grid_t  *              g;
  double err[MAX_PIPELINE+1];
} pipeline_args_t;

static void
compute_rms_div_e_err_pipeline( pipeline_args_t * args,
                                int pipeline_rank,
                                int n_pipeline ) {
  const field_t * ALIGNED(128) f = args->f;
  const grid_t  *              g = args->g;
  
  const field_t * ALIGNED(16) f0;
  int x, y, z, n_voxel;

  const int nx = g->nx;
  const int ny = g->ny;
  const int nz = g->nz;

  double err;

  // Process voxels assigned to this pipeline

  DISTRIBUTE_VOXELS( 2,nx, 2,ny, 2,nz, 16,
                     pipeline_rank, n_pipeline,
                     x, y, z, n_voxel );

  f0 = &f(x,y,z);

  err = 0;
  for( ; n_voxel; n_voxel-- ) {
    err += f0->div_e_err*f0->div_e_err;
    f0++;

    x++;
    if( x>nx ) {
      x=2, y++;
      if( y>ny ) y=2, z++;
      f0 = &f(x,y,z);
    }
  }

  args->err[pipeline_rank] = err;
}

double
compute_rms_div_e_err_kokkos( const field_array_t * RESTRICT fa ) {
//  pipeline_args_t args[1];
//  const field_t * f, * f0;
  const grid_t * RESTRICT g;
  double err = 0, temp_err = 0, local[2], global[2];
  int nx, ny, nz;
//  int x, y, z, nx, ny, nz, p;

  if( !fa ) ERROR(( "Bad args" ));
//  f = fa->f;
  g = fa->g; 

  nx = g->nx;
  ny = g->ny;
  nz = g->nz;

    const k_field_t& k_field = fa->k_f_d;


#if 0 // Original non-pipelined version
  for( z=2; z<=nz; z++ ) {
    for( y=2; y<=ny; y++ ) {
      for( x=2; x<=nx; x++ ) {
        err += f0->div_e_err*f0->div_e_err;
        f0++;
      }
    }
  }
# endif
  
  // Have the pipelines accumulate the interior of the local domain
  // (the host handled stragglers in the interior).

//  args->f = f;
//  args->g = g;
//  EXEC_PIPELINES( compute_rms_div_e_err, args, 0 );
    
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy_zyx({2,2,2}, {nz+1,ny+1,nx+1});
    Kokkos::parallel_reduce("compute_rms_div_e_err", policy_zyx, KOKKOS_LAMBDA(const int z, const int y, const int x, double& error) {
        double div_e_err = static_cast<double>(k_field(VOXEL(x,y,z,nx,ny,nz), field_var::div_e_err));
        error += div_e_err*div_e_err;
    },err);

  // Have the host accumulate the exterior of the local domain

  // Do exterior faces
/*
  for( y=2; y<=ny; y++ ) {
    for( z=2; z<=nz; z++ ) {
      f0 = &f(   1, y, z); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
      f0 = &f(nx+1, y, z); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
    }
  }

  for( z=2; z<=nz; z++ ) {
    for( x=2; x<=nx; x++ ) {
      f0 = &f( x,   1, z); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
      f0 = &f( x,ny+1, z); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
    }
  }

  for( x=2; x<=nx; x++ ) {
    for( y=2; y<=ny; y++ ) {
      f0 = &f(   x,   y,   1); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
      f0 = &f(   x,   y,nz+1); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
    }
  }
*/
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_yz({2,2}, {ny+1,nz+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_zx({2,2}, {nz+1,nx+1});
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_xy({2,2}, {nx+1,ny+1});

    Kokkos::parallel_reduce("compute_rms_div_e_err", policy_yz, KOKKOS_LAMBDA(const int y, const int z, double& error) {
        double div_e_err_1 = static_cast<double>(k_field(VOXEL(1,y,z,nx,ny,nz), field_var::div_e_err));
        double div_e_err_2 = static_cast<double>(k_field(VOXEL(nx+1,y,z,nx,ny,nz), field_var::div_e_err));
        error += 0.5*static_cast<double>(div_e_err_1)*static_cast<double>(div_e_err_1) + 0.5*static_cast<double>(div_e_err_2)*static_cast<double>(div_e_err_2);
    },temp_err);

    err += temp_err;
    temp_err = 0.0;

    Kokkos::parallel_reduce("compute_rms_div_e_err", policy_zx, KOKKOS_LAMBDA(const int z, const int x, double& error) {
        double div_e_err_1 = static_cast<double>(k_field(VOXEL(x,1,z,nx,ny,nz), field_var::div_e_err));
        double div_e_err_2 = static_cast<double>(k_field(VOXEL(x,ny+1,z,nx,ny,nz), field_var::div_e_err));
        error += 0.5*static_cast<double>(div_e_err_1)*static_cast<double>(div_e_err_1) + 0.5*static_cast<double>(div_e_err_2)*static_cast<double>(div_e_err_2);
    },temp_err);

    err += temp_err;
    temp_err = 0.0;

    Kokkos::parallel_reduce("compute_rms_div_e_err", policy_xy, KOKKOS_LAMBDA(const int x, const int y, double& error) {
        double div_e_err_1 = static_cast<double>(k_field(VOXEL(x,y,1,nx,ny,nz), field_var::div_e_err));
        double div_e_err_2 = static_cast<double>(k_field(VOXEL(x,y,nz+1,nx,ny,nz), field_var::div_e_err));
        error += 0.5*static_cast<double>(div_e_err_1)*static_cast<double>(div_e_err_1) + 0.5*static_cast<double>(div_e_err_2)*static_cast<double>(div_e_err_2);
    },temp_err);

    err += temp_err;
    temp_err = 0.0;

            

  // Do exterior edges
/*
  for( x=2; x<=nx; x++ ) {
    f0 = &f(   x,   1,   1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   x,ny+1,   1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   x,   1,nz+1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   x,ny+1,nz+1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
  }

  for( y=2; y<=ny; y++ ) {
    f0 = &f(   1,   y,   1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   1,   y,nz+1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(nx+1,   y,   1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(nx+1,   y,nz+1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
  }

  for( z=2; z<=nz; z++ ) {
    f0 = &f(   1,   1,   z); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(nx+1,   1,   z); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   1,ny+1,   z); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(nx+1,ny+1,   z); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
  }
*/
    Kokkos::parallel_reduce("compute_rms_div_e_err", Kokkos::RangePolicy<>(2,nx+1), KOKKOS_LAMBDA(const int x, double& error) {
        double edge1 = static_cast<double>(k_field(VOXEL(x, 1,    1,    nx,ny,nz), field_var::div_e_err));
        double edge2 = static_cast<double>(k_field(VOXEL(x, ny+1, 1,    nx,ny,nz), field_var::div_e_err));
        double edge3 = static_cast<double>(k_field(VOXEL(x, 1,    nz+1, nx,ny,nz), field_var::div_e_err));
        double edge4 = static_cast<double>(k_field(VOXEL(x, ny+1, nz+1, nx,ny,nz), field_var::div_e_err));
        error += 0.25*edge1*edge1 + 0.25*edge2*edge2 + 0.25*edge3*edge3 + 0.25*edge4*edge4;
    },temp_err);

    err += temp_err;
    temp_err = 0.0;

    Kokkos::parallel_reduce("compute_rms_div_e_err", Kokkos::RangePolicy<>(2,ny+1), KOKKOS_LAMBDA(const int y, double& error) {
        double edge1 = static_cast<double>(k_field(VOXEL(   1, y,    1, nx,ny,nz), field_var::div_e_err));
        double edge2 = static_cast<double>(k_field(VOXEL(   1, y, nz+1, nx,ny,nz), field_var::div_e_err));
        double edge3 = static_cast<double>(k_field(VOXEL(nx+1, y,    1, nx,ny,nz), field_var::div_e_err));
        double edge4 = static_cast<double>(k_field(VOXEL(nx+1, y, nz+1, nx,ny,nz), field_var::div_e_err));
        error += 0.25*edge1*edge1 + 0.25*edge2*edge2 + 0.25*edge3*edge3 + 0.25*edge4*edge4;
    },temp_err);

    err += temp_err;
    temp_err = 0.0;

    Kokkos::parallel_reduce("compute_rms_div_e_err", Kokkos::RangePolicy<>(2,nz+1), KOKKOS_LAMBDA(const int z, double& error) {
        double edge1 = static_cast<double>(k_field(VOXEL(   1, 1,    z, nx,ny,nz), field_var::div_e_err));
        double edge2 = static_cast<double>(k_field(VOXEL(nx+1, 1,    z, nx,ny,nz), field_var::div_e_err));
        double edge3 = static_cast<double>(k_field(VOXEL(   1, ny+1, z, nx,ny,nz), field_var::div_e_err));
        double edge4 = static_cast<double>(k_field(VOXEL(nx+1, ny+1, z, nx,ny,nz), field_var::div_e_err));
        error += 0.25*edge1*edge1 + 0.25*edge2*edge2 + 0.25*edge3*edge3 + 0.25*edge4*edge4;
    },temp_err);

    err += temp_err;
    temp_err = 0.0;


  // Do exterior corners
/*
  f0 = &f(   1,   1,   1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(nx+1,   1,   1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(   1,ny+1,   1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(nx+1,ny+1,   1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(   1,   1,nz+1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(nx+1,   1,nz+1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(   1,ny+1,nz+1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(nx+1,ny+1,nz+1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
*/
    Kokkos::parallel_reduce("exterior corners", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int i, double& error) {
        double corner0 = static_cast<double>(k_field(VOXEL(   1,    1,    1,nx,ny,nz), field_var::div_e_err));
        double corner1 = static_cast<double>(k_field(VOXEL(nx+1,    1,    1,nx,ny,nz), field_var::div_e_err));
        double corner2 = static_cast<double>(k_field(VOXEL(   1, ny+1,    1,nx,ny,nz), field_var::div_e_err));
        double corner3 = static_cast<double>(k_field(VOXEL(nx+1, ny+1,    1,nx,ny,nz), field_var::div_e_err));
        double corner4 = static_cast<double>(k_field(VOXEL(   1,    1, nz+1,nx,ny,nz), field_var::div_e_err));
        double corner5 = static_cast<double>(k_field(VOXEL(nx+1,    1, nz+1,nx,ny,nz), field_var::div_e_err));
        double corner6 = static_cast<double>(k_field(VOXEL(   1, ny+1, nz+1,nx,ny,nz), field_var::div_e_err));
        double corner7 = static_cast<double>(k_field(VOXEL(nx+1, ny+1, nz+1,nx,ny,nz), field_var::div_e_err));
        error += 0.125*corner0*corner0 + 0.125*corner1*corner1 + 0.125*corner2*corner2 + 0.125*corner3*corner3 +
                0.125*corner4*corner4 + 0.125*corner5*corner5 + 0.125*corner6*corner6 + 0.125*corner7*corner7; 
    },temp_err);

    err += temp_err;
    temp_err = 0.0;

  
  // Reduce the results from the host and pipelines

//  WAIT_PIPELINES();

//  for( p=0; p<=N_PIPELINE; p++ ) err += args->err[p];
//    deep_copy(err_h, err_d);

  // Reduce the results from all nodes

//  local[0] = err*g->dV;
//  local[1] = (g->nx*g->ny*g->nz)*g->dV;
//  mp_allsum_d( local, global, 2 );
//  return g->eps0*sqrt(global[0]/global[1]);
    local[0] = err*g->dV;
    local[1] = (g->nx*g->ny*g->nz)*g->dV;
    mp_allsum_d(local, global, 2);
    return g->eps0*sqrt(global[0]/global[1]);
}

double
compute_rms_div_e_err( const field_array_t * RESTRICT fa ) {
  pipeline_args_t args[1];
  const field_t * f, * f0;
  const grid_t * RESTRICT g;
  double err = 0, local[2], global[2];
  int x, y, z, nx, ny, nz, p;

  if( !fa ) ERROR(( "Bad args" ));
  f = fa->f;
  g = fa->g; 

#if 0 // Original non-pipelined version
  for( z=2; z<=nz; z++ ) {
    for( y=2; y<=ny; y++ ) {
      for( x=2; x<=nx; x++ ) {
        err += f0->div_e_err*f0->div_e_err;
        f0++;
      }
    }
  }
# endif
  
  // Have the pipelines accumulate the interior of the local domain
  // (the host handled stragglers in the interior).

  args->f = f;
  args->g = g;
  EXEC_PIPELINES( compute_rms_div_e_err, args, 0 );

  // Have the host accumulate the exterior of the local domain

  nx = g->nx;
  ny = g->ny;
  nz = g->nz;

  // Do exterior faces

  for( y=2; y<=ny; y++ ) {
    for( z=2; z<=nz; z++ ) {
      f0 = &f(   1, y, z); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
      f0 = &f(nx+1, y, z); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
    }
  }

  for( z=2; z<=nz; z++ ) {
    for( x=2; x<=nx; x++ ) {
      f0 = &f( x,   1, z); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
      f0 = &f( x,ny+1, z); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
    }
  }

  for( x=2; x<=nx; x++ ) {
    for( y=2; y<=ny; y++ ) {
      f0 = &f(   x,   y,   1); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
      f0 = &f(   x,   y,nz+1); err += 0.5*(double)f0->div_e_err*(double)f0->div_e_err;
    }
  }

  // Do exterior edges

  for( x=2; x<=nx; x++ ) {
    f0 = &f(   x,   1,   1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   x,ny+1,   1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   x,   1,nz+1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   x,ny+1,nz+1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
  }

  for( y=2; y<=ny; y++ ) {
    f0 = &f(   1,   y,   1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   1,   y,nz+1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(nx+1,   y,   1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(nx+1,   y,nz+1); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
  }

  for( z=2; z<=nz; z++ ) {
    f0 = &f(   1,   1,   z); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(nx+1,   1,   z); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(   1,ny+1,   z); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
    f0 = &f(nx+1,ny+1,   z); err += 0.25*(double)f0->div_e_err*(double)f0->div_e_err;
  }

  // Do exterior corners

  f0 = &f(   1,   1,   1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(nx+1,   1,   1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(   1,ny+1,   1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(nx+1,ny+1,   1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(   1,   1,nz+1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(nx+1,   1,nz+1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(   1,ny+1,nz+1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  f0 = &f(nx+1,ny+1,nz+1); err += 0.125*(double)f0->div_e_err*(double)f0->div_e_err;
  
  // Reduce the results from the host and pipelines

  WAIT_PIPELINES();

  for( p=0; p<=N_PIPELINE; p++ ) err += args->err[p];

  // Reduce the results from all nodes

  local[0] = err*g->dV;
  local[1] = (g->nx*g->ny*g->nz)*g->dV;
  mp_allsum_d( local, global, 2 );
  return g->eps0*sqrt(global[0]/global[1]);
}
