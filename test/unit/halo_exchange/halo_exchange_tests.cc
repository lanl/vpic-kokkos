//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"

#include "mpi.h"
#include <algorithm>
#include <iterator>
#include <random>
#include <vector>
#include <string>

#include "src/vpic/vpic.h"
#include "src/field_advance/field_advance.h"
#define IN_sfa
#include "src/field_advance/standard/sfa_private.h"

void verify_faces(const field_array_t* fa, int field_var) {
  int i, j, k;
  const int nx = fa->g->nx;
  const int ny = fa->g->ny;
  const int nz = fa->g->nz;
  const int rank = fa->g->bc[BOUNDARY(0,0,0)];
  const int neg_x_rank = fa->g->bc[BOUNDARY(-1, 0, 0)];
  const int neg_y_rank = fa->g->bc[BOUNDARY(0, -1, 0)];
  const int neg_z_rank = fa->g->bc[BOUNDARY(0, 0, -1)];
  const int pos_x_rank = fa->g->bc[BOUNDARY(1, 0, 0)];
  const int pos_y_rank = fa->g->bc[BOUNDARY(0, 1, 0)];
  const int pos_z_rank = fa->g->bc[BOUNDARY(0, 0, 1)];

  // X Face
  for(j=1; j<ny+1; j++) {
    for(k=1; k<nz+1; k++) {
      // Negative x face
      int ghost_cell = VOXEL(0,j,k,nx,ny,nz);
      int oppos_cell = VOXEL(nx,j,k,nx,ny,nz);
      float ghost_val = fa->k_f_h(ghost_cell, field_var) / (neg_x_rank+1);
      float expec_val = fa->k_f_h(oppos_cell, field_var) / (rank+1);
      REQUIRE(ghost_val == expec_val);

      // Positive x face
      ghost_cell = VOXEL(nx+1,j,k,nx,ny,nz);
      oppos_cell = VOXEL(1,j,k,nx,ny,nz);
      ghost_val = fa->k_f_h(ghost_cell, field_var) / (pos_x_rank+1);
      expec_val = fa->k_f_h(oppos_cell, field_var) / (rank+1);
      REQUIRE(ghost_val == expec_val);
    }
  }

  // Y Face
  for(i=1; i<nx+1; i++) {
    for(k=1; k<nz+1; k++) {
      // Negative x face
      int ghost_cell = VOXEL(i,0,k,nx,ny,nz);
      int oppos_cell = VOXEL(i,ny,k,nx,ny,nz);
      float ghost_val = fa->k_f_h(ghost_cell, field_var) / (neg_y_rank+1);
      float expec_val = fa->k_f_h(oppos_cell, field_var) / (rank+1);
      REQUIRE(ghost_val == expec_val);

      // Positive x face
      ghost_cell = VOXEL(i,ny+1,k,nx,ny,nz);
      oppos_cell = VOXEL(i,1,k,nx,ny,nz);
      ghost_val = fa->k_f_h(ghost_cell, field_var) / (pos_y_rank+1);
      expec_val = fa->k_f_h(oppos_cell, field_var) / (rank+1);
      REQUIRE(ghost_val == expec_val);
    }
  }

  // Z Face
  for(i=1; i<nx+1; i++) {
    for(j=1; j<ny+1; j++) {
      // Negative x face
      int ghost_cell = VOXEL(i,j,0,nx,ny,nz);
      int oppos_cell = VOXEL(i,j,nz,nx,ny,nz);
      float ghost_val = fa->k_f_h(ghost_cell, field_var) / (neg_z_rank+1);
      float expec_val = fa->k_f_h(oppos_cell, field_var) / (rank+1);
      REQUIRE(ghost_val == expec_val);

      // Positive x face
      ghost_cell = VOXEL(i,j,nz+1,nx,ny,nz);
      oppos_cell = VOXEL(i,j,1,nx,ny,nz);
      ghost_val = fa->k_f_h(ghost_cell, field_var) / (pos_z_rank+1);
      expec_val = fa->k_f_h(oppos_cell, field_var) / (rank+1);
      REQUIRE(ghost_val == expec_val);
    }
  }
}

TEST_CASE( "Verify halo exchange functions operate correctly", "[HaloExchange]" )
{
  double xmin = -0.5;
  double ymin = -0.5;
  double zmin = -0.5;
  double xmax = 0.5;
  double ymax = 0.5;
  double zmax = 0.5;

  double Nx = 12;
  double Ny = 12;
  double Nz = 12;

  double tx = 3;
  double ty = 3;
  double tz = 3;

  int size = -1;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  _world_rank = rank;
  _world_size = size;

  // Setup and partition grid amoung processes
  grid_t* g = new grid_t();
  for(int i=0; i<27; i++) g->bc[i] = anti_symmetric_fields;
  partition_periodic_box( g, xmin, ymin, zmin, xmax, ymax, zmax,
          (int)Nx, (int)Ny, (int)Nz,
          (int)tx, (int)ty, (int)tz );

  int nx = g->nx;
  int ny = g->ny;
  int nz = g->nz;
  int xyz_sz = 2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz;
  int yzx_sz = 2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx;
  int zxy_sz = 2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny;

  // Setup field array
  field_array_t* fa = new field_array_t(g->nv, xyz_sz, yzx_sz, zxy_sz);
  fa->g = g;

  auto& fields_h = fa->k_f_h;
  Kokkos::deep_copy(fields_h, 0);

  // Fill fields view. Each process starts with the same data and multiplies
  // it by its rank. This allows easy verification since each process knows
  // its neighbors rank along each face.
  using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
  Kokkos::MDRangePolicy<HostExecSpace, Kokkos::Rank<4>> setup_policy({1,1,1,0}, {nx+1,ny+1,nz+1,FIELD_VAR_COUNT});
  Kokkos::parallel_for("Setup fields", setup_policy,
  KOKKOS_LAMBDA(const int i, const int j, const int k, const int var) {
    int cell = VOXEL(i,j,k,nx,ny,nz);
    fields_h(cell, var) = static_cast<float>((rank+1)*(cell+var));
  });
  Kokkos::fence();
  Kokkos::deep_copy(fa->k_f_d, fa->k_f_h);

  // Test different halo exchange functions
  SECTION( "Exchange E fields" ) {
    k_begin_remote_ghost_hyb_e(fa, g, fa->fb);
    k_end_remote_ghost_hyb_e(fa, g, fa->fb);

    MPI_Barrier(MPI_COMM_WORLD);
    Kokkos::deep_copy(fa->k_f_h, fa->k_f_d);
    
    verify_faces(fa, field_var::ex);
    verify_faces(fa, field_var::ey);
    verify_faces(fa, field_var::ez);
  }

  SECTION( "Exchange B fields" ) {
    k_begin_remote_ghost_hyb_b(fa, g, fa->fb);
    k_end_remote_ghost_hyb_b(fa, g, fa->fb);

    MPI_Barrier(MPI_COMM_WORLD);
    Kokkos::deep_copy(fa->k_f_h, fa->k_f_d);
    
    verify_faces(fa, field_var::cbx);
    verify_faces(fa, field_var::cby);
    verify_faces(fa, field_var::cbz);
  }

  SECTION( "Exchange JF + RhoF" ) {
    k_begin_remote_ghost_hyb_jf(fa, g, fa->fb);
    k_end_remote_ghost_hyb_jf(fa, g, fa->fb);

    MPI_Barrier(MPI_COMM_WORLD);
    Kokkos::deep_copy(fa->k_f_h, fa->k_f_d);
    
    verify_faces(fa, field_var::jfx);
    verify_faces(fa, field_var::jfy);
    verify_faces(fa, field_var::jfz);
    verify_faces(fa, field_var::rhof);
  }

  SECTION( "Exchange Pressure" ) {
    k_begin_remote_ghost_hyb_curl_lpl_b(fa, g, fa->fb);
    k_end_remote_ghost_hyb_curl_lpl_b(fa, g, fa->fb);

    MPI_Barrier(MPI_COMM_WORLD);
    Kokkos::deep_copy(fa->k_f_h, fa->k_f_d);
    
    verify_faces(fa, field_var::pex);
    verify_faces(fa, field_var::pey);
    verify_faces(fa, field_var::pez);
  }

  SECTION( "Exchange Electron Temperature" ) {
    k_begin_remote_ghost_hyb_t(fa, g, fa->fb);
    k_end_remote_ghost_hyb_t(fa, g, fa->fb);

    MPI_Barrier(MPI_COMM_WORLD);
    Kokkos::deep_copy(fa->k_f_h, fa->k_f_d);
    
    verify_faces(fa, field_var::tx);
    verify_faces(fa, field_var::ty);
    verify_faces(fa, field_var::tz);
  }

  SECTION( "Exchange B field smoothing" ) {
    k_begin_remote_ghost_hyb_o(fa, g, fa->fb);
    k_end_remote_ghost_hyb_o(fa, g, fa->fb);

    MPI_Barrier(MPI_COMM_WORLD);
    Kokkos::deep_copy(fa->k_f_h, fa->k_f_d);
    
    verify_faces(fa, field_var::ox);
    verify_faces(fa, field_var::oy);
    verify_faces(fa, field_var::oz);
  }

  Kokkos::fence();
}

int main(int argc, char** argv) {
  int ret = 0;
  boot_services( &argc, &argv );

  Catch::Session session;
  ret = session.run();

  halt_services();
  return ret;
}
