#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"
#include "mpi.h"
#include <string>
#include <cstdlib>
#define IN_sfa
#include "src/field_advance/standard/sfa_private.h"
#include "src/vpic/vpic.h"

#define RANK_TO_INDEX(rank,ix,iy,iz) do {               \
    int _ix, _iy, _iz;                                  \
    _ix  = (rank);  /* ix = ix + gpx*( iy + gpy*iz ) */ \
    _iy  = _ix/tx; /* iy = iy + gpy*iz */              \
    _ix -= _iy*tx; /* ix = ix */                       \
    _iz  = _iy/ty; /* iz = iz */                       \
    _iy -= _iz*ty; /* iy = iy */                       \
    (ix) = _ix;                                         \
    (iy) = _iy;                                         \
    (iz) = _iz;                                         \
  } while(0)

int tx, ty, tz;

void verify_fields_match(const field_array_t* fa_a, const field_array_t* fa_b, field_var::f_v var) {
  const int nx = fa_a->g->nx;
  const int ny = fa_a->g->ny;
  const int nz = fa_a->g->nz;

  for(int i=0; i<nx+2; i++) {
    for(int j=0; j<ny+2; j++) {
      for(int k=0; k<nz+2; k++) {
        const int voxel = VOXEL(i,j,k,nx,ny,nz);
        REQUIRE(fa_a->k_f_h(voxel, var) == fa_b->k_f_h(voxel, var));
      }
    }
  }
}

void verify_fields_match_device(const field_array_t* fa_a, const field_array_t* fa_b, field_var::f_v var) {
  const int nx = fa_a->g->nx;
  const int ny = fa_a->g->ny;
  const int nz = fa_a->g->nz;

  auto fields_a = fa_a->k_f_d;
  auto fields_b = fa_b->k_f_d;
  size_t n_match = 0;
  Kokkos::parallel_reduce("Compare fields", 
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx+2,ny+2,nz+2}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k, size_t& num_match) {
    const int voxel = VOXEL(i,j,k,nx,ny,nz);
    if(fields_a(voxel, var) == fields_b(voxel, var)) {
      num_match += 1;
    }
  }, n_match);
  REQUIRE(n_match == (nx+2)*(ny+2)*(nz+2));
}

void test_local_boundaries(field_array_t* fa_legacy, field_array_t* fa_updated, grid_t* grid, const std::string& grid_name) {
  fa_legacy->g = grid;
  fa_updated->g = grid;

  SECTION( grid_name + ": Local Ghost Tang B" ) {
    fa_legacy->copy_to_host();
    legacy_local_ghost_tang_b(fa_legacy->f, fa_legacy->g);
    fa_legacy->copy_to_device();

    local_ghost_tang_b(fa_updated, fa_updated->g);
    fa_updated->copy_to_host();
    
    verify_fields_match_device(fa_legacy, fa_updated, field_var::cbx);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::cby);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::cbz);

    verify_fields_match(fa_legacy, fa_updated, field_var::cbx);
    verify_fields_match(fa_legacy, fa_updated, field_var::cby);
    verify_fields_match(fa_legacy, fa_updated, field_var::cbz);
  }

  SECTION( grid_name + ": Local Ghost Norm E" ) {
    fa_legacy->copy_to_host();
    legacy_local_ghost_norm_e(fa_legacy->f, fa_legacy->g);
    fa_legacy->copy_to_device();

    local_ghost_norm_e(fa_updated, fa_updated->g);
    fa_updated->copy_to_host();

    verify_fields_match_device(fa_legacy, fa_updated, field_var::ex);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::ey);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::ez);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::tcax);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::tcay);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::tcaz);
    
    verify_fields_match(fa_legacy, fa_updated, field_var::ex);
    verify_fields_match(fa_legacy, fa_updated, field_var::ey);
    verify_fields_match(fa_legacy, fa_updated, field_var::ez);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcax);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcay);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcaz);
  }

  SECTION( grid_name + ": Local Ghost Div B" ) {
    fa_legacy->copy_to_host();
    legacy_local_ghost_div_b( fa_legacy->f, fa_legacy->g );
    fa_legacy->copy_to_device();

    local_ghost_div_b( fa_updated, fa_updated->g );
    fa_updated->copy_to_host();
    
    verify_fields_match_device(fa_legacy, fa_updated, field_var::div_b_err);

    verify_fields_match(fa_legacy, fa_updated, field_var::div_b_err);
  }

  SECTION( grid_name + ": Local Adjust Tang E" ) {
    fa_legacy->copy_to_host();
    legacy_local_adjust_tang_e( fa_legacy->f, fa_legacy->g );
    fa_legacy->copy_to_device();

    local_adjust_tang_e( fa_updated, fa_updated->g );
    fa_updated->copy_to_host();
    
    verify_fields_match_device(fa_legacy, fa_updated, field_var::ex);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::ey);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::ez);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::tcax);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::tcay);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::tcaz);
    
    verify_fields_match(fa_legacy, fa_updated, field_var::ex);
    verify_fields_match(fa_legacy, fa_updated, field_var::ey);
    verify_fields_match(fa_legacy, fa_updated, field_var::ez);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcax);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcay);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcaz);
  }

  SECTION( grid_name + ": Local Adjust Norm B" ) {
    fa_legacy->copy_to_host();
    legacy_local_adjust_norm_b( fa_legacy->f, fa_legacy->g );
    fa_legacy->copy_to_device();

    local_adjust_norm_b( fa_updated, fa_updated->g );
    fa_updated->copy_to_host();
    
    verify_fields_match_device(fa_legacy, fa_updated, field_var::cbx);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::cby);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::cbz);
    
    verify_fields_match(fa_legacy, fa_updated, field_var::cbx);
    verify_fields_match(fa_legacy, fa_updated, field_var::cby);
    verify_fields_match(fa_legacy, fa_updated, field_var::cbz);
  }

  SECTION( grid_name + ": Local Adjust Div E" ) {
    fa_legacy->copy_to_host();
    legacy_local_adjust_div_e( fa_legacy->f, fa_legacy->g );
    fa_legacy->copy_to_device();

    local_adjust_div_e( fa_updated, fa_updated->g );
    fa_updated->copy_to_host();
    
    verify_fields_match_device(fa_legacy, fa_updated, field_var::div_e_err);
    
    verify_fields_match(fa_legacy, fa_updated, field_var::div_e_err);
  }

  SECTION( grid_name + ": Local Adjust JF" ) {
    fa_legacy->copy_to_host();
    legacy_local_adjust_jf( fa_legacy->f, fa_legacy->g );
    fa_legacy->copy_to_device();

    local_adjust_jf( fa_updated, fa_updated->g );
    fa_updated->copy_to_host();
    
    verify_fields_match_device(fa_legacy, fa_updated, field_var::jfx);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::jfy);
    verify_fields_match_device(fa_legacy, fa_updated, field_var::jfz);
    
    verify_fields_match(fa_legacy, fa_updated, field_var::jfx);
    verify_fields_match(fa_legacy, fa_updated, field_var::jfy);
    verify_fields_match(fa_legacy, fa_updated, field_var::jfz);
  }

  SECTION( grid_name + ": Local Adjust Rhof" ) {
    fa_legacy->copy_to_host();
    legacy_local_adjust_rhof( fa_legacy->f, fa_legacy->g );
    fa_legacy->copy_to_device();

    local_adjust_rhof( fa_updated, fa_updated->g );
    fa_updated->copy_to_host();
    
    verify_fields_match_device(fa_legacy, fa_updated, field_var::rhof);
    
    verify_fields_match(fa_legacy, fa_updated, field_var::rhof);
  }

  SECTION( grid_name + ": Local Adjust Rhob" ) {
    fa_legacy->copy_to_host();
    legacy_local_adjust_rhob( fa_legacy->f, fa_legacy->g );
    fa_legacy->copy_to_device();

    local_adjust_rhob( fa_updated, fa_updated->g );
    fa_updated->copy_to_host();
    
    verify_fields_match_device(fa_legacy, fa_updated, field_var::rhob);

    verify_fields_match(fa_legacy, fa_updated, field_var::rhob);
  }
}

TEST_CASE( "Verify functions for setting local boundaries operate correctly", "[Local Boundaries]" )
{
  double xmin = -0.5;
  double ymin = -0.5;
  double zmin = -0.5;
  double xmax = 0.5;
  double ymax = 0.5;
  double zmax = 0.5;

  double Nx = 24; //12;
  double Ny = 24; //12;
  double Nz = 24; //12;

  int size = -1;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  _world_rank = rank;
  _world_size = size;
  int px, py, pz;
  RANK_TO_INDEX( world_rank, px,py,pz );

  // Setup and partition grid amoung processes
  grid_t* g_symmetric = new_grid();
  partition_periodic_box( g_symmetric, xmin, ymin, zmin, xmax, ymax, zmax,
          (int)Nx, (int)Ny, (int)Nz,
          (int)tx, (int)ty, (int)tz );
  if( px==0 && Nx>1 ) {
    set_fbc(g_symmetric,BOUNDARY(-1,0,0),symmetric_fields);
    set_pbc(g_symmetric,BOUNDARY(-1,0,0),reflect_particles);
  }

  if( px==tx-1 && Nx>1 ) {
    set_fbc(g_symmetric,BOUNDARY(1,0,0),symmetric_fields);
    set_pbc(g_symmetric,BOUNDARY(1,0,0),reflect_particles);
  }

  if( py==0 && Ny>1 ) {
    set_fbc(g_symmetric,BOUNDARY(0,-1,0),symmetric_fields);
    set_pbc(g_symmetric,BOUNDARY(0,-1,0),reflect_particles);
  }

  if( py==ty-1 && Ny>1 ) {
    set_fbc(g_symmetric,BOUNDARY(0,1,0),symmetric_fields);
    set_pbc(g_symmetric,BOUNDARY(0,1,0),reflect_particles);
  }

  if( pz==0 && Nz>1 ) {
    set_fbc(g_symmetric,BOUNDARY(0,0,-1),symmetric_fields);
    set_pbc(g_symmetric,BOUNDARY(0,0,-1),reflect_particles);
  }

  if( pz==tz-1 && Nz>1 ) {
    set_fbc(g_symmetric,BOUNDARY(0,0,1),symmetric_fields);
    set_pbc(g_symmetric,BOUNDARY(0,0,1),reflect_particles);
  }

  grid_t* g_anti_symmetric = new_grid();
  partition_periodic_box( g_anti_symmetric, xmin, ymin, zmin, xmax, ymax, zmax,
          (int)Nx, (int)Ny, (int)Nz,
          (int)tx, (int)ty, (int)tz );
  if( px==0 && Nx>1 ) {
    set_fbc(g_anti_symmetric,BOUNDARY(-1,0,0),anti_symmetric_fields);
    set_pbc(g_anti_symmetric,BOUNDARY(-1,0,0),reflect_particles);
  }

  if( px==tx-1 && Nx>1 ) {
    set_fbc(g_anti_symmetric,BOUNDARY(1,0,0),anti_symmetric_fields);
    set_pbc(g_anti_symmetric,BOUNDARY(1,0,0),reflect_particles);
  }

  if( py==0 && Ny>1 ) {
    set_fbc(g_anti_symmetric,BOUNDARY(0,-1,0),anti_symmetric_fields);
    set_pbc(g_anti_symmetric,BOUNDARY(0,-1,0),reflect_particles);
  }

  if( py==ty-1 && Ny>1 ) {
    set_fbc(g_anti_symmetric,BOUNDARY(0,1,0),anti_symmetric_fields);
    set_pbc(g_anti_symmetric,BOUNDARY(0,1,0),reflect_particles);
  }

  if( pz==0 && Nz>1 ) {
    set_fbc(g_anti_symmetric,BOUNDARY(0,0,-1),anti_symmetric_fields);
    set_pbc(g_anti_symmetric,BOUNDARY(0,0,-1),reflect_particles);
  }

  if( pz==tz-1 && Nz>1 ) {
    set_fbc(g_anti_symmetric,BOUNDARY(0,0,1),anti_symmetric_fields);
    set_pbc(g_anti_symmetric,BOUNDARY(0,0,1),reflect_particles);
  }

  grid_t* g_absorb = new_grid();
  partition_absorbing_box( g_absorb, xmin, ymin, zmin, xmax, ymax, zmax,
          (int)Nx, (int)Ny, (int)Nz,
          (int)tx, (int)ty, (int)tz,
          reflect_particles );

  int nx = g_absorb->nx;
  int ny = g_absorb->ny;
  int nz = g_absorb->nz;
  int xyz_sz = FIELD_VAR_COUNT*(ny+2)*(nz+2); //2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz;
  int yzx_sz = FIELD_VAR_COUNT*(nz+2)*(nx+2); //2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx;
  int zxy_sz = FIELD_VAR_COUNT*(nx+2)*(ny+2); //2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny;

  // Setup field arrays
  field_array_t* fa_legacy = new field_array_t(g_absorb->nv, xyz_sz, yzx_sz, zxy_sz);
  MALLOC_ALIGNED( fa_legacy->f, g_absorb->nv, 128 );
  CLEAR( fa_legacy->f, g_absorb->nv );

  field_array_t* fa_updated = new field_array_t(g_absorb->nv, xyz_sz, yzx_sz, zxy_sz);
  MALLOC_ALIGNED( fa_updated->f, g_absorb->nv, 128 );
  CLEAR( fa_updated->f, g_absorb->nv );

  // Fill fields view. Each process starts with the same data and multiplies
  // it by its rank. This allows easy verification since each process knows
  // its neighbors rank along each face.
  auto updated_fields = fa_updated->k_f_d;
  auto legacy_fields = fa_legacy->k_f_d;
  Kokkos::parallel_for("Initialize fields on device", 
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx+2,ny+2,nz+2}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
    const int cell = VOXEL(i,j,k,nx,ny,nz);
    legacy_fields(cell, field_var::ex)        = static_cast<float>((rank+1)*(cell+0));
    legacy_fields(cell, field_var::ey)        = static_cast<float>((rank+1)*(cell+1));
    legacy_fields(cell, field_var::ez)        = static_cast<float>((rank+1)*(cell+2));
    legacy_fields(cell, field_var::div_e_err) = static_cast<float>((rank+1)*(cell+3));
    legacy_fields(cell, field_var::cbx)       = static_cast<float>((rank+1)*(cell+4));
    legacy_fields(cell, field_var::cby)       = static_cast<float>((rank+1)*(cell+5));
    legacy_fields(cell, field_var::cbz)       = static_cast<float>((rank+1)*(cell+6));
    legacy_fields(cell, field_var::jfx)       = static_cast<float>((rank+1)*(cell+7));
    legacy_fields(cell, field_var::jfy)       = static_cast<float>((rank+1)*(cell+8));
    legacy_fields(cell, field_var::jfz)       = static_cast<float>((rank+1)*(cell+9));
    legacy_fields(cell, field_var::tcax)      = static_cast<float>((rank+1)*(cell+10));
    legacy_fields(cell, field_var::tcay)      = static_cast<float>((rank+1)*(cell+11));
    legacy_fields(cell, field_var::tcaz)      = static_cast<float>((rank+1)*(cell+12));
    legacy_fields(cell, field_var::div_b_err) = static_cast<float>((rank+1)*(cell+13));
    legacy_fields(cell, field_var::rhof)      = static_cast<float>((rank+1)*(cell+14));
    legacy_fields(cell, field_var::rhob)      = static_cast<float>((rank+1)*(cell+15));

    updated_fields(cell, field_var::ex)        = static_cast<float>((rank+1)*(cell+0));
    updated_fields(cell, field_var::ey)        = static_cast<float>((rank+1)*(cell+1));
    updated_fields(cell, field_var::ez)        = static_cast<float>((rank+1)*(cell+2));
    updated_fields(cell, field_var::div_e_err) = static_cast<float>((rank+1)*(cell+3));
    updated_fields(cell, field_var::cbx)       = static_cast<float>((rank+1)*(cell+4));
    updated_fields(cell, field_var::cby)       = static_cast<float>((rank+1)*(cell+5));
    updated_fields(cell, field_var::cbz)       = static_cast<float>((rank+1)*(cell+6));
    updated_fields(cell, field_var::jfx)       = static_cast<float>((rank+1)*(cell+7));
    updated_fields(cell, field_var::jfy)       = static_cast<float>((rank+1)*(cell+8));
    updated_fields(cell, field_var::jfz)       = static_cast<float>((rank+1)*(cell+9));
    updated_fields(cell, field_var::tcax)      = static_cast<float>((rank+1)*(cell+10));
    updated_fields(cell, field_var::tcay)      = static_cast<float>((rank+1)*(cell+11));
    updated_fields(cell, field_var::tcaz)      = static_cast<float>((rank+1)*(cell+12));
    updated_fields(cell, field_var::div_b_err) = static_cast<float>((rank+1)*(cell+13));
    updated_fields(cell, field_var::rhof)      = static_cast<float>((rank+1)*(cell+14));
    updated_fields(cell, field_var::rhob)      = static_cast<float>((rank+1)*(cell+15));
  });
  Kokkos::fence();

  test_local_boundaries(fa_legacy, fa_updated, g_absorb, "Absorbing bc grid");

  test_local_boundaries(fa_legacy, fa_updated, g_symmetric, "Symmetric bc grid");

  test_local_boundaries(fa_legacy, fa_updated, g_anti_symmetric, "Anti-symmetric bc grid");

  Kokkos::fence();
  delete_grid(g_absorb);
  delete_grid(g_symmetric);
  delete_grid(g_anti_symmetric);
}

int main(int argc, char** argv) {
  boot_services( &argc, &argv );

  // Put process topology in global space
  tx = std::atoi(argv[1]);
  ty = std::atoi(argv[2]);
  tz = std::atoi(argv[3]);

  // Run tests
  Catch::Session session;
  int ret = session.run();

  halt_services();
  return ret;
}

