#include <string>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_ScatterView.hpp>
#include <argparse/argparse.hpp>
#include "scatter_gather_sort.hpp"
#include "kernels.hpp"

//#define DEBUG
#define SORT_SKIP

enum BenchMode {
  Stream,
  Scatter,
  Gather,
  GatherScatter,
  ScatterStencil,
  GatherStencil,
  GatherScatterStencil
};

enum KeyPattern {
  Random,
  Repeat,
  Contig,
  Stride
};

void 
write_log(const std::string& logname, 
          const std::string& kernel_name,
          argparse::ArgumentParser& program, 
          const size_t bytes_touched,
          std::vector<double>& sort_times, 
          std::vector<double>& kernel_times) {
  std::fstream log(logname+std::string(".csv"), std::ios::app);
  if(log.tellp() == 0) {
    log << "Kernel,Sort mode,Pattern,Num keys,Num unique keys,Data length,NX,NY,NZ,Block size,Tile size,Stencil radius,Num runs,Stride,Sort time,Kernel time,Bytes touched" << std::endl;
  }
  std::vector<int> grid = program.get<std::vector<int>>("--grid-dims");

  // Configuration
  std::string config_str = "";
  config_str +=  kernel_name + ",";
  config_str +=  program.get<std::string>("--sort") + ",";
  config_str +=  program.get<std::string>("--pattern") + ",";
  config_str +=  std::to_string(program.get<uint64_t>("--sample-size")) + ",";
  config_str +=  std::to_string(program.get<uint64_t>("--unique-keys")) + ",";
  config_str +=  std::to_string(program.get<uint64_t>("--data-length")) + ",";
  config_str +=  std::to_string(grid[0]) + "," + std::to_string(grid[1]) + "," + std::to_string(grid[2]) + ",";
  config_str +=  std::to_string(program.get<int>("--block-size")) + ",";
  config_str +=  std::to_string(program.get<int>("--tile-size")) + ",";
  config_str +=  std::to_string(program.get<int>("--stencil-radius")) + ",";
  config_str +=  std::to_string(program.get<int>("--num-iter")) + ",";
  config_str +=  std::to_string(program.get<int>("--stride")) + ",";

  // Times
  for(uint64_t i=0; i<kernel_times.size(); i++) {
    log << config_str;
    if(sort_times.size() == 0) {
      log << 0 << ",";
    } else {
      log << std::to_string(sort_times[i]) << ","; 
    }
    log << std::to_string(kernel_times[i]) << "," 
        << std::to_string(bytes_touched) << std::endl;
  }
}

// Template struct for getting the correct sorting function signature
template<typename KeyType>
struct FunctionType {
  typedef std::function<void(KeyType&,uint64_t,uint64_t,int32_t)> type;
};

// No-op sort
template<typename KeyView>
void no_sort(
        KeyView& key_view,
        const uint64_t n,
        const uint64_t num_bins,
        const int32_t tile_size) {}

// Get correct sorting function
template<typename KeyType>
typename FunctionType<KeyType>::type
get_sort_func(std::string& mode_str) {
  if(mode_str.compare("standard") == 0) {
    return standard_sort<KeyType>;
  } else if (mode_str.compare("strided") == 0) {
    return strided_sort<KeyType>;
  } else if (mode_str.compare("tiled") == 0) {
    return tiled_sort<KeyType>;
  } else if (mode_str.compare("tiled-strided") == 0) {
    return tiled_strided_sort<KeyType>;
  } else if (mode_str.compare("random") == 0) {
    return no_sort<KeyType>;
  } else if (mode_str.compare("default") == 0) {
    return no_sort<KeyType>;
  }
  return no_sort<KeyType>;
}

// Print keys for debugging
template<typename KeyView>
void print_keys(const KeyView keys, int nkeys) {
  Kokkos::parallel_for("Print keys", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int i) {
    for(int i=0; i<nkeys; i++) 
      printf("%i ", keys(i));
  });
  Kokkos::fence();
  printf("\n");
}

// Profiling region wrapper
#define BEG_REGION(label) \
  Kokkos::fence(); \
  Kokkos::Profiling::pushRegion( #label ); \
  auto label##_beg = std::chrono::high_resolution_clock::now(); 

#define END_REGION(label, timer_vec) \
  Kokkos::fence(); \
  auto label##_end = std::chrono::high_resolution_clock::now(); \
  Kokkos::Profiling::popRegion(); \
  timer_vec.push_back(std::chrono::duration<double>(label##_end - label##_beg).count());

int 
main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    argparse::ArgumentParser program("s-g-bench");
    program.add_description("Benchmark for different scatter/gather/stencil patterns and sorting orders");
    program.add_argument("bench-mode")
      .help("Benchmark mode: [scatter, gather, gather-scatter, scatter-stencil, gather-stencil, gather-scatter-stencil]")
      .choices("stream", "scatter", "gather", "gather-scatter", "scatter-stencil", "gather-stencil", "gather-scatter-stencil")
      .default_value("scatter");
    program.add_argument("--sort")
      .help("Sort function: [default, random, standard, strided, tiled, tiled-strided]")
      .choices("default", "random", "standard", "strided", "tiled", "tiled-strided")
      .default_value("default");
    program.add_argument("-p", "--pattern")
      .help("Pattern for keys: [random, unique, repeat, contig, stride]")
      .choices("random", "unique", "repeat", "contig", "stride")
      .default_value("random");
    program.add_argument("-d", "--data-length")
      .help("Length of data to scatter/gather from")
      .default_value(static_cast<uint64_t>(268435456))
      .scan<'u', uint64_t>();
    program.add_argument("-s", "--sample-size")
      .help("Number of samples to gather/scatter")
      .default_value(static_cast<uint64_t>(1000000))
      .scan<'u', uint64_t>();
    program.add_argument("-u", "--unique-keys")
      .help("Number of unique keys to generate.")
      .default_value(static_cast<uint64_t>(1000000))
      .scan<'u', uint64_t>();
    program.add_argument("-b", "--block-size")
      .help("Number of elements to read per sample")
      .default_value(1)
      .scan<'d', int>();
    program.add_argument("-t", "--tile-size")
      .help("Number of elements in each tile for tiled and tiled-strided sort")
      .default_value(1)
      .scan<'d', int>();
    program.add_argument("--stencil-radius")
      .help("Radius of stencil")
      .default_value(1)
      .scan<'d', int>();
    program.add_argument("-n", "--num-iter")
      .help("Number of iterations to run")
      .default_value(5)
      .scan<'d', int>();
    program.add_argument("-g", "--grid-dims")
      .help("Dimensions of grid for stencil benchmark")
      .nargs(1,3)
      .default_value(std::vector<int>{100,100,100})
      .scan<'d', int>();
    program.add_argument("--stride")
      .help("Stride for key generation")
      .default_value(1)
      .scan<'d', int>();
    program.add_argument("--log")
      .help("Filename for logging results")
      .default_value("result_log");
    program.parse_args(argc, argv);

    // Which benchmark mode to use
    std::string bench_mode = program.get<std::string>("bench-mode");
    BenchMode b_mode = Scatter;
    if(bench_mode.compare("stream") == 0) {
      b_mode = Stream;
    } else if (bench_mode.compare("scatter") == 0) {
      b_mode = Scatter;
    } else if (bench_mode.compare("gather") == 0) {
      b_mode = Gather;
    } else if (bench_mode.compare("gather-scatter") == 0) {
      b_mode = GatherScatter;
    } else if (bench_mode.compare("scatter-stencil") == 0) {
      b_mode = ScatterStencil;
    } else if (bench_mode.compare("gather-stencil") == 0) {
      b_mode = GatherStencil;
    } else if (bench_mode.compare("gather-scatter-stencil") == 0) {
      b_mode = GatherScatterStencil;
    }
    // Which sort function to use
    using KeyScalar = uint64_t;
    using KeyType = Kokkos::View<KeyScalar*>;
    using UnmanagedHostKeyType = Kokkos::View<KeyScalar*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using DataType = double;
    std::string mode_str = program.get<std::string>("--sort");
    FunctionType<KeyType>::type sort_func = get_sort_func<KeyType>(mode_str);

    // Pattern for key generation
    KeyPattern key_pattern = Random;
    std::string pattern_str = program.get<std::string>("--pattern");
    if (pattern_str.compare("repeat") == 0) {
      key_pattern = Repeat;
    } else if (pattern_str.compare("contig") == 0) {
      key_pattern = Contig;
    } else if (pattern_str.compare("stride") == 0) {
      key_pattern = Stride;
    }
    // Stride for key generation
    int stride = program.get<int>("--stride");
    // Length of data 
    uint64_t data_len = program.get<uint64_t>("--data-length");
    // Number of data movement operations to run per iteration
    uint64_t sample_sz = program.get<uint64_t>("--sample-size");
    // Number of elements to move per key or the stencil size 
    int block_sz = program.get<int>("--block-size");
    // Number of unique keys to generate
    uint64_t unique_sz = program.get<uint64_t>("--unique-keys");
    // Tile size
    int tile_sz = program.get<int>("--tile-size");
    // Stencil radius
    int stencil_rad = program.get<int>("--stencil-radius");
    // Number of iterations
    int niters = program.get<int>("--num-iter");
    std::vector<double> times, sort_times, copy_times, scale_times, add_times, triad_times;
    // Grid dims (stencil only)
    int nx=0, ny=0, nz=0;
    // Log name
    std::string logname = program.get<std::string>("--log");

    size_t bytes_touched = 0;
    size_t tot_sample_sz = sample_sz;

    if(b_mode == Stream) {
      // Normal kernels
      data_len = program.get<uint64_t>("--data-length");
      Kokkos::View<DataType*> a("View a", data_len);
      Kokkos::View<DataType*> b("View b", data_len);
      Kokkos::View<DataType*> c("View c", data_len);
      Kokkos::deep_copy(a, 1.);
      Kokkos::deep_copy(b, 2.);
      Kokkos::deep_copy(c, 0.);
      DataType scalar = 3.0;
      Kokkos::fence();
      printf("Setup stream views a, b, and c\n");
      Kokkos::fence();

      // Run microbenchmark
      for(int idx=0; idx<niters; idx++) {
        BEG_REGION(Copy);
        Kokkos::parallel_for("Copy kernel", Kokkos::RangePolicy<uint64_t>(0, data_len),
        KOKKOS_LAMBDA(const uint64_t idx) {
          c(idx) = a(idx);
        });
        END_REGION(Copy, copy_times);

        BEG_REGION(Scale);
        Kokkos::parallel_for("Scale kernel", Kokkos::RangePolicy<uint64_t>(0, data_len),
        KOKKOS_LAMBDA(const uint64_t idx) {
          b(idx) = scalar*c(idx);
        });
        END_REGION(Scale, scale_times);

        BEG_REGION(Add);
        Kokkos::parallel_for("Add kernel", Kokkos::RangePolicy<uint64_t>(0, data_len),
        KOKKOS_LAMBDA(const uint64_t idx) {
          c(idx) = a(idx)+b(idx);
        });
        END_REGION(Add, add_times);

        BEG_REGION(Triad);
        Kokkos::parallel_for("Triad kernel", Kokkos::RangePolicy<uint64_t>(0, data_len),
        KOKKOS_LAMBDA(const uint64_t idx) {
          a(idx) = b(idx)+scalar*c(idx);
        });
        END_REGION(Triad, triad_times);
      }
    } else {
printf("Allocating %lu GB for Key view\n", sizeof(KeyScalar)*sample_sz/1000000000LLU);
      KeyType keys("Key view", sample_sz);
      auto host_keys = Kokkos::create_mirror_view(keys);
      
      if(b_mode == ScatterStencil || b_mode == GatherStencil || b_mode == GatherScatterStencil) {
        // Stencil kernels

        std::vector<int> dims = program.get<std::vector<int>>("--grid-dims");
        if(dims.size() == 1) {
          nx = dims[0], ny = 1, nz = 1;
        } else if (dims.size() == 2) {
          nx = dims[0], ny = dims[1], nz = 1;
        } else if (dims.size() == 3) {
          nx = dims[0], ny = dims[1], nz = dims[2];
        }
        printf("Grid dims: %d,%d,%d\n", nx,ny,nz);
        data_len = (nx+(2*stencil_rad))*(ny+(2*stencil_rad))*(nz+(2*stencil_rad));
        const int max_idx = map3Dto1D(nx+stencil_rad-1,ny+stencil_rad-1,nz+stencil_rad-1, nx,ny,nz, stencil_rad);
        const int min_idx = map3Dto1D(stencil_rad, stencil_rad, stencil_rad,  nx,ny,nz, stencil_rad);
        printf("Min idx: %d\tMax idx: %d\n", min_idx, max_idx);
        
        // Initialize cell indices using stdlib for correctness
//        std::vector<KeyScalar> key_vec;
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<KeyScalar> uniform_dist(min_idx, max_idx);
        size_t gather_bytes_touched = 0, scatter_bytes_touched = 0;
        switch(key_pattern) {
          case Random:
            for(uint64_t i=0; i<sample_sz; i++) {
//              key_vec.push_back(uniform_dist(g));
              host_keys(i) = uniform_dist(g);
            }
            break;
          case Repeat: // Key values will repeat 
            {
              bool done = false;
              uint64_t counter = 0;
              uint64_t repeat_sz = sample_sz/(nx*ny*nz); // Number of times a key repeats
              if(repeat_sz*(nx*ny*nz) < sample_sz)
                repeat_sz += 1;
              for(int i=stencil_rad; i<nx+stencil_rad && !done; i++) {
                for(int j=stencil_rad; j<ny+stencil_rad && !done; j++) {
                  for(int k=stencil_rad; k<nz+stencil_rad && !done; k++) {
                    gather_bytes_touched  += sizeof(DataType) * (1 + block_sz); 
                    scatter_bytes_touched += sizeof(DataType) * (1 + 2*block_sz); 
                    if( i==stencil_rad || i==(nx+stencil_rad-1) ||
                        j==stencil_rad || j==(ny+stencil_rad-1) ||  
                        k==stencil_rad || k==(nz+stencil_rad-1) ) {
                      gather_bytes_touched += sizeof(DataType)*stencil_rad*block_sz;
                      scatter_bytes_touched += sizeof(DataType)*stencil_rad*2*block_sz;
                    }
                    for(uint64_t v=0; v<repeat_sz && !done; v++) {
                      host_keys(counter) = map3Dto1D(i,j,k,nx,ny,nz,stencil_rad);
//                      key_vec.push_back(map3Dto1D(i,j,k,nx,ny,nz,stencil_rad));
                      if(++counter == sample_sz) 
                        done = true;
                    }
                  }
                }
              }
            }
            break;
          case Contig: // Keys increase monotonically up to max possible then repeats
            {
              bool done = false;
              uint64_t counter = 0;
              while(!done) {
                for(int i=stencil_rad; i<nx+stencil_rad && !done; i++) {
                  for(int j=stencil_rad; j<ny+stencil_rad && !done; j++) {
                    for(int k=stencil_rad; k<nz+stencil_rad && !done; k++) {
                      if(counter < nx*ny*nz) {
                        gather_bytes_touched  += sizeof(DataType) * (1 + block_sz); 
                        scatter_bytes_touched += sizeof(DataType) * (1 + 2*block_sz); 
                        if( i==stencil_rad || i==(nx+stencil_rad-1) ||
                            j==stencil_rad || j==(ny+stencil_rad-1) ||  
                            k==stencil_rad || k==(nz+stencil_rad-1) ) {
                          gather_bytes_touched += sizeof(DataType)*stencil_rad*block_sz;
                          scatter_bytes_touched += sizeof(DataType)*stencil_rad*2*block_sz;
                        }
                      }
                      host_keys(counter) = map3Dto1D(i,j,k,nx,ny,nz,stencil_rad);
                      //key_vec.push_back(map3Dto1D(i,j,k,nx,ny,nz,stencil_rad));
                      if(++counter == sample_sz) 
                        done = true;
                    }
                  }
                }
              }
            }
            break;
          case Stride: // Same as contiguous but keys skip based on stride
            {
              int counter = 0;
              uint64_t x,y,z, idx_counter=0;
              std::vector<bool> avail((nx+(2*stencil_rad))*(ny+(2*stencil_rad))*(nz+(2*stencil_rad)), true);
              while(host_keys.size() < sample_sz) {
              //while(key_vec.size() < sample_sz) {
                auto idx = min_idx + ((counter*stride) % (max_idx-min_idx));
                map1Dto3D(idx, x,y,z, nx,ny,nz, stencil_rad);
                if(  stencil_rad <= x && x <= nx+stencil_rad 
                  && stencil_rad <= y && y <= ny+stencil_rad
                  && stencil_rad <= z && z <= nz+stencil_rad) {
                  if(avail[idx]) {
                    gather_bytes_touched  += sizeof(DataType) * (1 + block_sz); 
                    scatter_bytes_touched += sizeof(DataType) * (1 + 2*block_sz); 
                    if( x==stencil_rad || x==(nx+stencil_rad-1) ||
                        y==stencil_rad || y==(ny+stencil_rad-1) ||  
                        z==stencil_rad || z==(nz+stencil_rad-1) ) {
                      gather_bytes_touched += sizeof(DataType)*stencil_rad*block_sz;
                      scatter_bytes_touched += sizeof(DataType)*stencil_rad*2*block_sz;
                    }
                    avail[idx] = false;
                  } 
                  host_keys(idx_counter) = idx;
                  idx_counter++;
                  //key_vec.push_back(idx);
                }
                counter++;
              }
            }
            break;
          default:
            printf("Error: Invalid Key pattern %d\n", key_pattern);
            break;
        }
        //std::vector<KeyScalar> key_vec_copy(key_vec);
        //std::sort(key_vec_copy.begin(), key_vec_copy.end());
        //auto unique_count = std::unique(key_vec_copy.begin(), key_vec_copy.end()) - key_vec_copy.begin();
        Kokkos::sort(host_keys);
        uint64_t unique_count = 0;
        Kokkos::parallel_reduce("Count unique keys", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(1, host_keys.size()), 
        KOKKOS_LAMBDA(const uint64_t i, uint64_t& usum) {
          if(host_keys[i] != host_keys[i-1])
            usum += 1;
        }, unique_count);
        unique_count += 1;
        printf("Created %lu unique keys out of %lu samples\n", unique_count, sample_sz);
        if(mode_str.compare("random") == 0) {
          // Randomly shuffle keys
          std::shuffle(Kokkos::Experimental::begin(host_keys), Kokkos::Experimental::end(host_keys), g);
          //std::shuffle(key_vec.begin(), key_vec.end(), g);
        }

        // Copy keys to View
        //UnmanagedHostKeyType host_keys(key_vec.data(), key_vec.size());
        Kokkos::deep_copy(keys, host_keys);

        printf("Setup stencil\n");

        // Run microbenchmark
        for(int idx=0; idx<niters; idx++) {
#ifdef DEBUG
          printf("Pre: First 100 keys: ");
          print_keys(keys, 100);
#endif

//#ifdef SORT_SKIP
//if(unique_count != sample_sz) {
          // Sort keys each iteration to get perf stats
          BEG_REGION(Sort);
          sort_func(keys, sample_sz, max_idx+1, tile_sz);
          END_REGION(Sort, sort_times);
          verify_sort(keys, mode_str, unique_count, tile_sz);
//}
//#endif

#ifdef DEBUG
          printf("Post: First 100 keys: ");
          print_keys(keys, 100);
#endif

          switch(b_mode) {
            case GatherStencil:
              {
printf("Allocating %f GB for src\n", static_cast<double>(sizeof(DataType)*(data_len*block_sz))/1000000000.0);
printf("Allocating %f GB for dst\n", static_cast<double>(sizeof(DataType)*sample_sz)/1000000000.0);
                Kokkos::View<DataType**> src("Src view", data_len, block_sz);
                Kokkos::View<DataType*> dst("Dst view", sample_sz);
                Kokkos::deep_copy(src, 1.);
                Kokkos::deep_copy(dst, 0.);

                BEG_REGION(Gather_stencil);
                gather_stencil_kernel(src, dst, keys, nx,ny,nz, block_sz, stencil_rad);
                END_REGION(Gather_stencil, times);
                bytes_touched = gather_bytes_touched;
              }
              break;
            case ScatterStencil:
              {
printf("Allocating %f GB for src\n", static_cast<double>(sizeof(DataType)*sample_sz)/1000000000.0);
printf("Allocating %f GB for dst\n", static_cast<double>(sizeof(DataType)*(data_len*block_sz))/1000000000.0);
                Kokkos::View<DataType*> src("Src view", sample_sz);
                Kokkos::View<DataType**> dst("Dst view", data_len, block_sz);
                Kokkos::deep_copy(src, 1.);
                Kokkos::deep_copy(dst, 0.);
                BEG_REGION(Scatter_stencil);
                scatter_stencil_kernel(src, dst, keys, (uint64_t)nx,(uint64_t)ny,(uint64_t)nz, (uint64_t)block_sz, (uint64_t)stencil_rad);
                END_REGION(Scatter_stencil, times);
                bytes_touched = scatter_bytes_touched;
              }
              break;
            case GatherScatterStencil:
              {
printf("Allocating %f GB for src\n", static_cast<double>(sizeof(DataType)*(data_len*block_sz))/1000000000.0);
printf("Allocating %f GB for dst\n", static_cast<double>(sizeof(DataType)*sample_sz)/1000000000.0);
                Kokkos::View<DataType**> src("Src view", data_len, block_sz);
                Kokkos::View<DataType*> dst("Dst view", sample_sz);
                Kokkos::deep_copy(src, 1.);
                Kokkos::deep_copy(dst, 0.);
                BEG_REGION(GatherScatterStencil);
                gather_stencil_kernel(src, dst, keys, nx,ny,nz, block_sz, stencil_rad);
                scatter_stencil_kernel(dst, src, keys, nx,ny,nz, block_sz, stencil_rad);
                END_REGION(GatherScatterStencil, times);
                bytes_touched = gather_bytes_touched + scatter_bytes_touched;
                //tot_sample_sz *= 2;
              }
              break;
            default:
              printf("Error: Invalid benchmark operation\n");
          }
          // Reset key view to original random order
          Kokkos::deep_copy(keys, host_keys);
        }
      } else if(b_mode == Gather || b_mode == Scatter || b_mode == GatherScatter) {
        // Normal kernels
        data_len = program.get<uint64_t>("--data-length");

        // Initialize cell indices using stdlib for correctness
        //std::vector<KeyScalar> key_vec;
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<KeyScalar> uniform_dist(0, data_len);
        uint64_t max_key = data_len;
        uint64_t max_range = std::min(unique_sz, max_key);
        printf("Number of samples: %lu\n", sample_sz);
        printf("Data length: %lu\n", data_len);
        printf("Max key value: %lu\n", max_key);
        switch(key_pattern) {
          case Random: // Randomly generate keys
            for(uint64_t i=0; i<sample_sz; i++) {
              host_keys(i) = uniform_dist(g);
              //key_vec.push_back(uniform_dist(g));
            }
            break;
          case Repeat: // Repeat keys based on tile size
            for(uint64_t i=0; i<sample_sz; i++) {
              host_keys(i) = i % max_range;
              //key_vec.push_back( i % max_range );
            }
            break;
          case Contig: // Keys are contiguous
            for(uint64_t i=0; i<sample_sz; i++) {
              host_keys(i) = i % max_key;
              //key_vec.push_back( i % max_key );
            }
            break;
          case Stride: // Keys increase in value based on stride
            for(uint64_t i=0; i<sample_sz; i++) {
              host_keys(i) = (i*stride) % max_key;
              //key_vec.push_back( (i*stride) % max_key );
            }
            break;
          default:
            printf("Error: Invalid Key pattern %d\n", key_pattern);
            break;
        }
        Kokkos::sort(host_keys);
        uint64_t unique_count = 0;
        Kokkos::parallel_reduce("Count unique keys", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(1, host_keys.size()), 
        KOKKOS_LAMBDA(const uint64_t i, uint64_t& usum) {
          if(host_keys[i] != host_keys[i-1])
            usum += 1;
        }, unique_count);
        unique_count += 1;
        //std::vector<KeyScalar> key_vec_copy(key_vec);
        //std::sort(key_vec_copy.begin(), key_vec_copy.end());
        //auto unique_count = std::unique(key_vec_copy.begin(), key_vec_copy.end()) - key_vec_copy.begin();
        printf("Created %td unique keys out of %lu samples\n", unique_count, sample_sz);
        if(b_mode != Gather && unique_count < keys.size()) {
          printf("Running atomic scatter kernel\n");
        } else {
          printf("Running scatter kernel\n");
        }
        if(mode_str.compare("random") == 0) {
          // Randomly shuffle cell indices
          //std::shuffle(key_vec.begin(), key_vec.end(), g);
          std::shuffle(Kokkos::Experimental::begin(host_keys), Kokkos::Experimental::end(host_keys), g);
        }

        // Copy to Views
        //UnmanagedHostKeyType host_keys(key_vec.data(), key_vec.size());
        Kokkos::deep_copy(keys, host_keys);

        // Run microbenchmark
        for(int idx=0; idx<niters; idx++) {
#ifdef DEBUG
          printf("Pre: First 100 keys: ");
          print_keys(keys, 100);
#endif

          BEG_REGION(Sort);
          sort_func(keys, sample_sz, max_key, tile_sz);
          END_REGION(Sort, sort_times);
          verify_sort(keys, mode_str, unique_count, tile_sz);

#ifdef DEBUG
          printf("Post: First 100 keys: ");
          print_keys(keys, 100);
#endif

          switch(b_mode) {
            case Scatter:
              {
                // Set src size to number of samples to avoid false sharing
printf("Allocating %lu GB for src\n", sizeof(DataType)*(sample_sz*block_sz)/1000000000LLU);
printf("Allocating %lu GB for dst\n", sizeof(DataType)*(data_len*block_sz)/1000000000LLU);
                Kokkos::View<DataType**> src("Src view", sample_sz, block_sz);
                Kokkos::View<DataType**> dst("Dst view", data_len, block_sz);
                Kokkos::deep_copy(src, 1.);
                Kokkos::deep_copy(dst, 0.);
      
                // Switch to atomic scatter if there are repeated keys
                if(unique_count < keys.size()) {
                  BEG_REGION(Scatter);
                  scatter_atomic_kernel(src, dst, keys, block_sz);
                  END_REGION(Scatter, times);
                  bytes_touched = sizeof(DataType)*unique_count*( 1 + 3*block_sz );
                } else {
                  BEG_REGION(Scatter);
                  scatter_kernel(src, dst, keys, block_sz);
                  END_REGION(Scatter, times);
                  bytes_touched = sizeof(DataType)*unique_count*( 1 + 2*block_sz );
                }
              }
              break;
            case Gather:
              {
                // Set dst size to number of samples to avoid false sharing
printf("Allocating %lu GB for src\n", sizeof(DataType)*(data_len*block_sz)/1000000000LLU);
printf("Allocating %lu GB for dst\n", sizeof(DataType)*(sample_sz*block_sz)/1000000000LLU);
                Kokkos::View<DataType**> src("Src view", data_len, block_sz);
                Kokkos::View<DataType**> dst("Dst view", sample_sz, block_sz);
                Kokkos::deep_copy(src, 1.);
                Kokkos::deep_copy(dst, 0.);

                BEG_REGION(Gather);
                gather_kernel(src, dst, keys, block_sz);
                END_REGION(Gather, times);
                bytes_touched = sizeof(DataType)*unique_count*( 1 + 2*block_sz );
              }
              break;
            case GatherScatter:
              {
                // Set dst size to number of samples to avoid false sharing
printf("Allocating %lu GB for src\n", sizeof(DataType)*(data_len*block_sz)/1000000000LLU);
printf("Allocating %lu GB for dst\n", sizeof(DataType)*(sample_sz*block_sz)/1000000000LLU);
                Kokkos::View<DataType**> src("Src view", data_len, block_sz);
                Kokkos::View<DataType**> dst("Dst view", sample_sz, block_sz);
                Kokkos::deep_copy(src, 1.);
                Kokkos::deep_copy(dst, 0.);
                // Switch to atomic scatter if there are repeated keys
                if(unique_count < keys.size()) {
                  BEG_REGION(GatherScatter);
                  gather_kernel(src, dst, keys, block_sz);
                  scatter_atomic_kernel(dst, src, keys, block_sz);
                  END_REGION(GatherScatter, times);
                  bytes_touched = sizeof(DataType)*unique_count*(2*( 1 + 2*block_sz ) + block_sz);
                } else {
                  BEG_REGION(GatherScatter);
                  gather_kernel(src, dst, keys, block_sz);
                  scatter_kernel(dst, src, keys, block_sz);
                  END_REGION(GatherScatter, times);
                  bytes_touched = 2*sizeof(DataType)*unique_count*( 1 + 2*block_sz );
                }
                //tot_sample_sz *= 2;
              }
              break;
            default:
              printf("Error: Invalid benchmark operation\n");
          }
          // Reset random key view 
          Kokkos::deep_copy(keys, host_keys);
        }
      }
    }

    // Print configuration and stats
    std::cout << "===============================================" << std::endl;
    std::cout << " Configuration " << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "Benchmark mode:     " << bench_mode << std::endl;
    std::cout << "Sort function:      " << mode_str << std::endl;
    std::cout << "Key gen pattern:    " << pattern_str << std::endl;
    std::cout << "Data length:        " << data_len << std::endl;
    std::cout << "Stencil grid:       " << nx << "," << ny << "," << nz << std::endl;
    std::cout << "Sample size:        " << sample_sz << std::endl;
    std::cout << "Block size:         " << block_sz << std::endl;
    std::cout << "Sort tile size:     " << tile_sz << std::endl;
    std::cout << "Stencil radius:     " << stencil_rad << std::endl;
    std::cout << "Stride length:      " << stride << std::endl;
    std::cout << "Num iterations:     " << niters << std::endl;
    if(b_mode == Stream) {
      std::sort(copy_times.begin(), copy_times.end());
      double copy_median = copy_times[(copy_times.size()+1)/2];
      double copy_sum = std::accumulate(copy_times.begin(), copy_times.end(), 0.0);
      double copy_mean = copy_sum / copy_times.size();
      double copy_min = *std::min_element(copy_times.begin(), copy_times.end());
      double copy_max = *std::max_element(copy_times.begin(), copy_times.end());
      std::sort(scale_times.begin(), scale_times.end());
      double scale_median = scale_times[(scale_times.size()+1)/2];
      double scale_sum = std::accumulate(scale_times.begin(), scale_times.end(), 0.0);
      double scale_mean = scale_sum / scale_times.size();
      double scale_min = *std::min_element(scale_times.begin(), scale_times.end());
      double scale_max = *std::max_element(scale_times.begin(), scale_times.end());
      std::sort(add_times.begin(), add_times.end());
      double add_median = add_times[(add_times.size()+1)/2];
      double add_sum = std::accumulate(add_times.begin(), add_times.end(), 0.0);
      double add_mean = add_sum / add_times.size();
      double add_min = *std::min_element(add_times.begin(), add_times.end());
      double add_max = *std::max_element(add_times.begin(), add_times.end());
      std::sort(triad_times.begin(), triad_times.end());
      double triad_median = triad_times[(triad_times.size()+1)/2];
      double triad_sum = std::accumulate(triad_times.begin(), triad_times.end(), 0.0);
      double triad_mean = triad_sum / triad_times.size();
      double triad_min = *std::min_element(triad_times.begin(), triad_times.end());
      double triad_max = *std::max_element(triad_times.begin(), triad_times.end());
      std::cout << "===============================================" << std::endl;
      std::cout << " Copy time " << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << "Total time:  " << copy_sum    << "s" << std::endl;
      std::cout << "Median time: " << copy_median << "s" << std::endl;
      std::cout << "Mean time:   " << copy_mean   << "s" << std::endl;
      std::cout << "Min time:    " << copy_min    << "s" << std::endl;
      std::cout << "Max time:    " << copy_max    << "s" << std::endl;
      std::cout << "Peak GB/s:   " << 2*sizeof(DataType)*data_len / (1.0E9*copy_min) << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << " Scale time " << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << "Total time:  " << scale_sum    << "s" << std::endl;
      std::cout << "Median time: " << scale_median << "s" << std::endl;
      std::cout << "Mean time:   " << scale_mean   << "s" << std::endl;
      std::cout << "Min time:    " << scale_min    << "s" << std::endl;
      std::cout << "Max time:    " << scale_max    << "s" << std::endl;
      std::cout << "Peak GB/s:   " << 2*sizeof(DataType)*data_len / (1.0E9*scale_min) << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << " Add time " << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << "Total time:  " << add_sum    << "s" << std::endl;
      std::cout << "Median time: " << add_median << "s" << std::endl;
      std::cout << "Mean time:   " << add_mean   << "s" << std::endl;
      std::cout << "Min time:    " << add_min    << "s" << std::endl;
      std::cout << "Max time:    " << add_max    << "s" << std::endl;
      std::cout << "Peak GB/s:   " << 3*sizeof(DataType)*data_len / (1.0E9*add_min) << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << " Triad time " << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << "Total time:  " << triad_sum    << "s" << std::endl;
      std::cout << "Median time: " << triad_median << "s" << std::endl;
      std::cout << "Mean time:   " << triad_mean   << "s" << std::endl;
      std::cout << "Min time:    " << triad_min    << "s" << std::endl;
      std::cout << "Max time:    " << triad_max    << "s" << std::endl;
      std::cout << "Peak GB/s:   " << 3*sizeof(DataType)*data_len / (1.0E9*triad_min) << std::endl;
      write_log(logname, "copy",  program, 2*sizeof(DataType)*data_len, sort_times, copy_times);
      write_log(logname, "scale", program, 2*sizeof(DataType)*data_len, sort_times, scale_times);
      write_log(logname, "add",   program, 3*sizeof(DataType)*data_len, sort_times, add_times);
      write_log(logname, "triad", program, 3*sizeof(DataType)*data_len, sort_times, triad_times);
    } else {
      std::sort(sort_times.begin(), sort_times.end());
      double sort_median = sort_times[(sort_times.size()+1)/2];
      double sort_sum = std::accumulate(sort_times.begin(), sort_times.end(), 0.0);
      double sort_mean = sort_sum / sort_times.size();
      double sort_min = *std::min_element(sort_times.begin(), sort_times.end());
      double sort_max = *std::max_element(sort_times.begin(), sort_times.end());
      std::sort(times.begin(), times.end());
      double median = times[(times.size()+1)/2];
      double sum = std::accumulate(times.begin(), times.end(), 0.0);
      double mean = sum / times.size();
      double min = *std::min_element(times.begin(), times.end());
      double max = *std::max_element(times.begin(), times.end());
      std::cout << "===============================================" << std::endl;
      std::cout << " Sort time " << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << "Sort Total time:  " << sort_sum << "s" << std::endl;
      std::cout << "Sort Median time: " << sort_median << "s" << std::endl;
      std::cout << "Sort Mean time:   " << sort_mean << "s" << std::endl;
      std::cout << "Sort Min time:    " << sort_min << "s" << std::endl;
      std::cout << "Sort Max time:    " << sort_max << "s" << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << " Kernel time " << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << "Total time:     " << sum << "s" << std::endl;
      std::cout << "Median time:    " << median << "s" << std::endl;
      std::cout << "Mean time:      " << mean << "s" << std::endl;
      std::cout << "Min time:       " << min << "s" << std::endl;
      std::cout << "Max time:       " << max << "s" << std::endl;
      std::cout << "Peak Samples/s: " << tot_sample_sz / min << std::endl;
      std::cout << "Peak GB/s:      " << bytes_touched / (1.0E9*min) << std::endl;
      std::cout << "===============================================" << std::endl;
      write_log(logname, bench_mode, program, bytes_touched, sort_times, times);
    }
  }
  Kokkos::finalize();
  return 0;
}

