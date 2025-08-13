#define CATCH_CONFIG_RUNNER // Use our own main
#include "catch.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <iterator>
#include <random>
#include <vector>
#include "../src/scatter_gather_sort.hpp"

TEST_CASE( "Verify ParticleSorter reorders particles correctly", "[ParticleSorter]" )
{
  const int tilesize = 7;
  const int nx = 3;
  const int ny = 5;
  const int nz = 13;
  const int nppc = 11;
  const int num_part = nppc*nx*ny*nz;

  Kokkos::View<uint64_t*> part_i("Particles", num_part);
  Kokkos::View<uint64_t*>::HostMirror part_i_h = Kokkos::create_mirror_view(part_i);

  // Initialize cell indices using stdlib for correctness
  std::vector<int> cell_indices;
  for(int i=0; i<nx; i++) {
    for(int j=0; j<ny; j++) {
      for(int k=0; k<nz; k++) {
        for(int n=0; n<nppc; n++) {
          const int cell = i*ny*nz + j*nz + k;
          cell_indices.push_back(cell);
        }
      }
    }
  }
  // Randomly shuffle cell indices
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(cell_indices.begin(), cell_indices.end(), g);

  // Copy to Views
  using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
  Kokkos::RangePolicy<HostExecSpace> part_policy(0,num_part);
  Kokkos::parallel_for("Load indices", part_policy, KOKKOS_LAMBDA(const int i) {
    part_i_h(i) = cell_indices[i];
    printf("%d ", part_i_h(i));
  });
  printf("\n");
  
  Kokkos::deep_copy(part_i, part_i_h);
  Kokkos::fence();

  // Test different sort functions
  SECTION( "Standard sort test" ) {
    standard_sort(part_i, num_part, nx*ny*nz, tilesize);

    Kokkos::deep_copy(part_i_h, part_i);

    // Verify particles are in standard order
    for(int i=0; i<num_part; i++) {
      REQUIRE(part_i_h(i) == i/nppc);
    }
  }

  SECTION( "Strided sort test" ) {
    strided_sort(part_i, num_part, nx*ny*nz, tilesize);

    Kokkos::deep_copy(part_i_h, part_i);

    // Verify particles are in strided order
    for(int i=0; i<num_part; i++) {
      REQUIRE(part_i_h(i) == i%(nx*ny*nz));
    }
  }

  SECTION( "Tiled sort test" ) {
    tiled_sort(part_i, num_part, nx*ny*nz, tilesize);

    Kokkos::deep_copy(part_i_h, part_i);

    // Verify particles are in tiled order
    int nseq = num_part/(tilesize*nx*ny*nz);
    for(int s=0; s<nseq; s++) {
      for(int c=0; c<nx*ny*nz; c++) {
        for(int i=0; i<tilesize; i++) {
          const int idx = s*nx*ny*nz*tilesize + c*tilesize + i;
          REQUIRE(part_i_h(idx) == c);
        }
      }
    }
  }

  SECTION( "Tiled strided sort test" ) {
    tiled_strided_sort(part_i, num_part, nx*ny*nz, tilesize);

    Kokkos::deep_copy(part_i_h, part_i);

    // Verify particles are in tiled strided order
    int nseq = num_part/(nppc*tilesize);
    int tile_start = 0;
    for(int s=0; s<nseq; s++) {
      for(int i=0; i<nppc; i++) {
        for(int c=0; c<tilesize; c++) {
          const int idx = s*nppc*tilesize + i*tilesize + c;
          REQUIRE(part_i_h(idx) == tile_start+c);
        }
      }
      tile_start += tilesize;
    }
  }

  Kokkos::fence();
}

int main(int argc, char** argv) {
  int ret = 0;
  Kokkos::initialize(argc, argv);
  {
    ret = Catch::Session().run();
  }
  Kokkos::finalize();
  return ret;
}

