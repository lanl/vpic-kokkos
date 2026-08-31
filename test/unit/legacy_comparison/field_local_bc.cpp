#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"
#include "mpi.h"
#include <string>
#include <cstdlib>
#define IN_sfa
#include "src/field_advance/standard/sfa_private.h"
#include "src/vpic/vpic.h"

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

TEST_CASE( "Verify field communication functions operate correctly", "[Field Communication]" )
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

  int size = -1;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  _world_rank = rank;
  _world_size = size;

  // Setup and partition grid amoung processes
  grid_t* g = new_grid();
  for(int i=0; i<27; i++) g->bc[i] = anti_symmetric_fields;
  partition_periodic_box( g, xmin, ymin, zmin, xmax, ymax, zmax,
          (int)Nx, (int)Ny, (int)Nz,
          (int)tx, (int)ty, (int)tz );

  int nx = g->nx;
  int ny = g->ny;
  int nz = g->nz;
  int xyz_sz = FIELD_VAR_COUNT*(ny+2)*(nz+2); //2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz;
  int yzx_sz = FIELD_VAR_COUNT*(nz+2)*(nx+2); //2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx;
  int zxy_sz = FIELD_VAR_COUNT*(nx+2)*(ny+2); //2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny;

  // Setup field arrays
  field_array_t* fa_legacy = new field_array_t(g->nv, xyz_sz, yzx_sz, zxy_sz);
  MALLOC_ALIGNED( fa_legacy->f, g->nv, 128 );
  CLEAR( fa_legacy->f, g->nv );
  fa_legacy->g = g;
  field_array_t* fa_updated = new field_array_t(g->nv, xyz_sz, yzx_sz, zxy_sz);
  MALLOC_ALIGNED( fa_updated->f, g->nv, 128 );
  CLEAR( fa_updated->f, g->nv );
  fa_updated->g = g;

  // Fill fields view. Each process starts with the same data and multiplies
  // it by its rank. This allows easy verification since each process knows
  // its neighbors rank along each face.
  for(size_t i=0; i<nx+2; i++) {
    for(size_t j=0; j<ny+2; j++) {
      for(size_t k=0; k<nz+2; k++) {
        int cell = VOXEL(i,j,k,nx,ny,nz);
        fa_legacy->f[cell].ex        = static_cast<float>((rank+1)*(cell+0));
        fa_legacy->f[cell].ey        = static_cast<float>((rank+1)*(cell+1));
        fa_legacy->f[cell].ez        = static_cast<float>((rank+1)*(cell+2));
        fa_legacy->f[cell].div_e_err = static_cast<float>((rank+1)*(cell+3));
        fa_legacy->f[cell].cbx       = static_cast<float>((rank+1)*(cell+4));
        fa_legacy->f[cell].cby       = static_cast<float>((rank+1)*(cell+5));
        fa_legacy->f[cell].cbz       = static_cast<float>((rank+1)*(cell+6));
        fa_legacy->f[cell].jfx       = static_cast<float>((rank+1)*(cell+7));
        fa_legacy->f[cell].jfy       = static_cast<float>((rank+1)*(cell+8));
        fa_legacy->f[cell].jfz       = static_cast<float>((rank+1)*(cell+9));
        fa_legacy->f[cell].tcax      = static_cast<float>((rank+1)*(cell+10));
        fa_legacy->f[cell].tcay      = static_cast<float>((rank+1)*(cell+11));
        fa_legacy->f[cell].tcaz      = static_cast<float>((rank+1)*(cell+12));
        fa_legacy->f[cell].div_b_err = static_cast<float>((rank+1)*(cell+13));
        fa_legacy->f[cell].rhof      = static_cast<float>((rank+1)*(cell+14));
        fa_legacy->f[cell].rhob      = static_cast<float>((rank+1)*(cell+15));

        fa_updated->f[cell].ex        = static_cast<float>((rank+1)*(cell+0));
        fa_updated->f[cell].ey        = static_cast<float>((rank+1)*(cell+1));
        fa_updated->f[cell].ez        = static_cast<float>((rank+1)*(cell+2));
        fa_updated->f[cell].div_e_err = static_cast<float>((rank+1)*(cell+3));
        fa_updated->f[cell].cbx       = static_cast<float>((rank+1)*(cell+4));
        fa_updated->f[cell].cby       = static_cast<float>((rank+1)*(cell+5));
        fa_updated->f[cell].cbz       = static_cast<float>((rank+1)*(cell+6));
        fa_updated->f[cell].jfx       = static_cast<float>((rank+1)*(cell+7));
        fa_updated->f[cell].jfy       = static_cast<float>((rank+1)*(cell+8));
        fa_updated->f[cell].jfz       = static_cast<float>((rank+1)*(cell+9));
        fa_updated->f[cell].tcax      = static_cast<float>((rank+1)*(cell+10));
        fa_updated->f[cell].tcay      = static_cast<float>((rank+1)*(cell+11));
        fa_updated->f[cell].tcaz      = static_cast<float>((rank+1)*(cell+12));
        fa_updated->f[cell].div_b_err = static_cast<float>((rank+1)*(cell+13));
        fa_updated->f[cell].rhof      = static_cast<float>((rank+1)*(cell+14));
        fa_updated->f[cell].rhob      = static_cast<float>((rank+1)*(cell+15));
      }
    }
  }
  fa_legacy->copy_to_device();
  fa_updated->copy_to_device();

  SECTION( "Local Ghost Tang B" ) {
    legacy_local_ghost_tang_b(fa_legacy->f, g);
    fa_legacy->copy_to_device();

    local_ghost_tang_b(fa_updated, g);
    fa_updated->copy_to_host();
    
    verify_fields_match(fa_legacy, fa_updated, field_var::cbx);
    verify_fields_match(fa_legacy, fa_updated, field_var::cby);
    verify_fields_match(fa_legacy, fa_updated, field_var::cbz);
  }

  SECTION( "Local Ghost Norm E" ) {
    legacy_local_ghost_norm_e(fa_legacy->f, g);
    fa_legacy->copy_to_device();

    local_ghost_norm_e(fa_updated, g);
    fa_updated->copy_to_host();
    
    verify_fields_match(fa_legacy, fa_updated, field_var::ex);
    verify_fields_match(fa_legacy, fa_updated, field_var::ey);
    verify_fields_match(fa_legacy, fa_updated, field_var::ez);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcax);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcay);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcaz);
  }

  SECTION( "Local Ghost Div B" ) {
    legacy_local_ghost_div_b( fa_legacy->f, g );
    fa_legacy->copy_to_device();

    local_ghost_div_b( fa_updated, g );
    fa_updated->copy_to_host();
    
    verify_fields_match(fa_legacy, fa_updated, field_var::div_b_err);
  }

  SECTION( "Local Adjust Tang E" ) {
    legacy_local_adjust_tang_e( fa_legacy->f, g );
    fa_legacy->copy_to_device();

    local_adjust_tang_e( fa_updated, g );
    fa_updated->copy_to_host();
    
    verify_fields_match(fa_legacy, fa_updated, field_var::ex);
    verify_fields_match(fa_legacy, fa_updated, field_var::ey);
    verify_fields_match(fa_legacy, fa_updated, field_var::ez);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcax);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcay);
    verify_fields_match(fa_legacy, fa_updated, field_var::tcaz);
  }

  SECTION( "Local Adjust Norm B" ) {
    legacy_local_adjust_norm_b( fa_legacy->f, g );
    fa_legacy->copy_to_device();

    local_adjust_norm_b( fa_updated, g );
    fa_updated->copy_to_host();
    
    verify_fields_match(fa_legacy, fa_updated, field_var::cbx);
    verify_fields_match(fa_legacy, fa_updated, field_var::cby);
    verify_fields_match(fa_legacy, fa_updated, field_var::cbz);
  }

  SECTION( "Local Adjust Div E" ) {
    legacy_local_adjust_div_e( fa_legacy->f, g );
    fa_legacy->copy_to_device();

    local_adjust_div_e( fa_updated, g );
    fa_updated->copy_to_host();
    
    verify_fields_match(fa_legacy, fa_updated, field_var::div_e_err);
  }

  SECTION( "Local Adjust JF" ) {
    legacy_local_adjust_jf( fa_legacy->f, g );
    fa_legacy->copy_to_device();

    local_adjust_jf( fa_updated, g );
    fa_updated->copy_to_host();
    
    verify_fields_match(fa_legacy, fa_updated, field_var::jfx);
    verify_fields_match(fa_legacy, fa_updated, field_var::jfy);
    verify_fields_match(fa_legacy, fa_updated, field_var::jfz);
  }

  SECTION( "Local Adjust Rhof" ) {
    legacy_local_adjust_rhof( fa_legacy->f, g );
    fa_legacy->copy_to_device();

    local_adjust_rhof( fa_updated, g );
    fa_updated->copy_to_host();
    
    verify_fields_match(fa_legacy, fa_updated, field_var::rhof);
  }

  SECTION( "Local Adjust Rhob" ) {
    legacy_local_adjust_rhob( fa_legacy->f, g );
    fa_legacy->copy_to_device();

    local_adjust_rhob( fa_updated, g );
    fa_updated->copy_to_host();
    
    verify_fields_match(fa_legacy, fa_updated, field_var::rhob);
  }

  Kokkos::fence();
  delete_grid(g);
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

