#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"
#include "mpi.h"
#include <string>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <map>
#define IN_sfa
#include "src/field_advance/standard/sfa_private.h"
#include "src/vpic/vpic.h"

int tx, ty, tz;

void verify_shuffle(const species_t* sp, std::map<int,std::pair<uint64_t,uint64_t>>& partition_range) {
  Kokkos::deep_copy(sp->k_sortindex_h, sp->k_sortindex_d);
  Kokkos::deep_copy(sp->k_partition_h, sp->k_partition_d);
  int nx = sp->g->nx;
  int ny = sp->g->ny;
  int nz = sp->g->nz;

  std::cout << "\tSort indices: \n";
  for(int p=0; p<nx*ny*nz; p++) {
    int ix, iy, iz;
    _RANK_TO_INDEX(p, ix, iy, iz, nx, ny, nz);
    const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
    const auto i0 = sp->k_partition_h(v);
    const auto i1 = sp->k_partition_h(v+1);
    if(i1-i0 > 0) {
      std::cout << "\t\tPartition " << v << "[" << i0 << "," << i1 << "): ";
      for(size_t i=i0; i<i1; i++) {
        auto val = sp->k_sortindex_h(i);
        std::cout << val << " ";
      }
      std::cout << std::endl;
    }
  }
  for(int p=0; p<nx*ny*nz; p++) {
    int ix, iy, iz;
    _RANK_TO_INDEX(p, ix, iy, iz, nx, ny, nz);
    const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
    const auto i0 = sp->k_partition_h(v);
    const auto i1 = sp->k_partition_h(v+1);
    if(i1-i0 > 0) {
      auto range = partition_range[v];
      std::vector<bool> seen(i1-i0, false);
      for(size_t i=i0; i<i1; i++) {
        auto val = sp->k_sortindex_h(i);
        bool in_range = (range.first <= val) && (val <= range.second);
        REQUIRE( in_range ); // Verify index is in range
if(seen[val-range.first])
  std::cout << "Found duplicate index " << val << " at " << i << std::endl;
        REQUIRE( !seen[val-range.first] ); // Verify no duplicates
        seen[val-range.first] = true;
      }
      
      bool pass = true; // Verify all indices covered
      for(size_t i=0; i<i1-i0; i++) {
        pass = pass && seen[i];
      }
      REQUIRE(pass);
    }
  }
}

TEST_CASE( "Verify shuffler produces valid permutations", "[RandomShuffle]" )
{
  double xmin = -0.5;
  double ymin = -0.5;
  double zmin = -0.5;
  double xmax = 0.5;
  double ymax = 0.5;
  double zmax = 0.5;

  double Nx = tx; //12;
  double Ny = ty; //12;
  double Nz = tz; //12;

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
                          1, 1, 1 );
  g->init_cartesian_grid();

printf("Setup grid\n");

  int nx = g->nx;
  int ny = g->ny;
  int nz = g->nz;

  float q = 1.0;
  float m = 1.0;
  int nppc = 140;
  int max_local_np = int(nx*ny*nz*nppc*1.00);
  int max_local_nm = int(nx*ny*nz*nppc*1.00);
  int sort_interval = 10000;
  int sort_out_of_place = 1;
  double vthi = 1.0;

  // Create species
  const char* name = "test";
  species_t * sp;
  int len = name ? strlen(name) : 0;

  if( !len ) ERROR(( "Cannot create a nameless species" ));
  if( !g ) ERROR(( "NULL grid" ));
  if( g->nv == 0) ERROR(( "Allocate grid before defining species." ));
  if( max_local_np<1 ) max_local_np = 1;
  if( max_local_nm<1 ) max_local_nm = 1;

  sp = new species_t(max_local_np, max_local_nm);

  sp->q = q;
  sp->m = m;

  sp->max_np = max_local_np;

  sp->max_nm = max_local_nm;

  sp->last_sorted       = INT64_MIN;
  sp->sort_interval     = sort_interval;
  sp->sort_out_of_place = sort_out_of_place;

  sp->g = g;
  sp->np = nx*ny*nz*nppc;

printf("Created species of %zu particles\n", sp->np);

  // Create rng
  kokkos_rng_pool_t rng_pool(12345);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> xdis(xmin, xmax);
  std::uniform_real_distribution<> ydis(ymin, ymax);
  std::uniform_real_distribution<> zdis(zmin, zmax);
  std::normal_distribution ndis{0.0, vthi};

printf("Created rng\n");

  // Fill particles
  Kokkos::deep_copy(sp->k_p_h, 1.0f);
  Kokkos::deep_copy(sp->k_p_i_h, 0);
  Kokkos::deep_copy(sp->k_partition_h, 0);
  Kokkos::deep_copy(sp->k_sortindex_h, 0);
  Kokkos::deep_copy(sp->k_p_d, 1.0f);
  Kokkos::deep_copy(sp->k_p_i_d, 0);
  Kokkos::deep_copy(sp->k_partition_d, 0);
  Kokkos::deep_copy(sp->k_sortindex_d, 0);

  size_t count = 0;
  for(int i=1; i<nx+1; i++) {
    for(int j=1; j<ny+1; j++) {
      for(int k=1; k<nz+1; k++) {
        for(int n=0; n<nppc; n++) {
          sp->k_p_i_h(count) = VOXEL(i,j,k,nx,ny,nz); //i*ny*nz + j*nz + k;
          count++;
        }
      }
    }
  }
  
  //std::cout << "Cell ids (" << sp->k_p_i_h.extent(0) << "): ";
  //for(int i=0; i<sp->k_p_i_h.extent(0); i++) {
  //  std::cout << sp->k_p_i_h(i) << "  ";
  //}
  //std::cout << std::endl;

printf("Filled particles\n");

  // Sync device particles
printf("k_p_ device host extents: %zu, %zu\n", sp->k_p_d.extent(0), sp->k_p_h.extent(0));
printf("k_p_i_ device host extents: %zu, %zu\n", sp->k_p_i_d.extent(0), sp->k_p_i_h.extent(0));
  Kokkos::deep_copy(sp->k_p_d, sp->k_p_h);
  Kokkos::deep_copy(sp->k_p_i_d, sp->k_p_i_h);

printf("Synced particles to device\n");

  // Sort particles
  k_ParticleSorter<BinSort> sorter;
  sorter.sort( sp, false );

printf("Sorted particles to host\n");

  // Sync host sort indices and partitioning
  Kokkos::fence();
  Kokkos::deep_copy(sp->k_sortindex_h, sp->k_sortindex_d);
  Kokkos::deep_copy(sp->k_partition_h, sp->k_partition_d);

printf("Synced particles\n");
  Kokkos::fence();

  std::map<int,std::pair<uint64_t,uint64_t>> partition_map;

  printf("\tSort index len: %zu\n", sp->k_sortindex_d.extent(0));
  printf("\tPartition len: %zu: [",  sp->k_partition_d.extent(0));
  for(int p=0; p<nx*ny*nz; p++) {
    int ix, iy, iz;
    _RANK_TO_INDEX(p, ix, iy, iz, nx, ny, nz);
    const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
    std::cout << sp->k_partition_h(v) << "  ";
  }
  std::cout << "]\n";
  std::cout << "\tSort indices: \n";
  for(int p=0; p<nx*ny*nz; p++) {
    int ix, iy, iz;
    _RANK_TO_INDEX(p, ix, iy, iz, nx, ny, nz);
    const int v = VOXEL(ix+1, iy+1, iz+1, nx, ny, nz);
    const auto i0 = sp->k_partition_h(v);
    const auto i1 = sp->k_partition_h(v+1);
    if(i1-i0 > 0) {
      std::cout << "\t\tPartition " << v << "[" << i0 << "," << i1 << "): [";
      uint64_t min_idx=ULONG_MAX, max_idx=0;
      for(size_t i=i0; i<i1; i++) {
        std::cout << sp->k_sortindex_h(i) << " ";
        if(sp->k_sortindex_h(i) < min_idx)
          min_idx = sp->k_sortindex_h(i);
        if(sp->k_sortindex_h(i) > max_idx)
          max_idx = sp->k_sortindex_h(i);
      }
      std::cout << "]: [" << min_idx << "," << max_idx << "]";
      std::cout << std::endl;
      partition_map[v] = std::pair<int,int>(min_idx, max_idx);
    }
  }

  // Test different shuffle functions
  SECTION( "Fisher-Yates" ) {
    std::cout << "Fisher Yates shuffle\n";

    ParticleShuffler<FisherYatesShuffle> shuffler;
    shuffler.shuffle(sp, rng_pool, false);
    verify_shuffle(sp, partition_map);
  }

  SECTION( "MergeShuffle" ) {
    std::cout << "Merge shuffle\n";

    ParticleShuffler<MergeShuffle> shuffler;
    shuffler.shuffle(sp, rng_pool, false);
    verify_shuffle(sp, partition_map);
  }

  SECTION( "SortShuffle" ) {
    std::cout << "Sort shuffle\n";

    ParticleShuffler<SortShuffle> shuffler;
    shuffler.shuffle(sp, rng_pool, false);
    verify_shuffle(sp, partition_map);
  }

  SECTION( "Bijection+Variable Philox Shuffle" ) {
    std::cout << "Bijection shuffle\n";

    ParticleShuffler<BijectiveShuffle<PhiloxBijectiveFunction>> shuffler;
    shuffler.shuffle(sp, rng_pool, false);
    verify_shuffle(sp, partition_map);
  }

  Kokkos::fence();
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
