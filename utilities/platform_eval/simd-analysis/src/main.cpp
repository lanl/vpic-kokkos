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
#include "axpy_kernel.hpp"
#include "planckian_kernel.hpp"
#include "fir_kernel.hpp"
#include "pressure_kernel.hpp"
#include "if_quad_kernel.hpp"
#include "pi_reduce_kernel.hpp"
//#include "matmul_kernel.hpp"

enum VecMode {
  Auto,
  Guided,
  Manual,
  AdHoc
};

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

void print_results(const std::string kernel_name, const std::string mode, std::vector<double> times) {
  if(times.size() > 0) {
    std::sort(times.begin(), times.end());
    double median = times[(times.size()+1)/2];
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    double min = *std::min_element(times.begin(), times.end());
    double max = *std::max_element(times.begin(), times.end());
    std::cout << "===============================================" << std::endl;
    std::cout << " " << kernel_name << " " << mode << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "Total time:     " << sum << "s" << std::endl;
    std::cout << "Median time:    " << median << "s" << std::endl;
    std::cout << "Mean time:      " << mean << "s" << std::endl;
    std::cout << "Min time:       " << min << "s" << std::endl;
    std::cout << "Max time:       " << max << "s" << std::endl;
    std::cout << "===============================================" << std::endl;
  }
}

void 
write_log(const std::string& logname, 
          const std::string& kernel_name,
          const std::string& mode_name,
          const uint64_t len,
          argparse::ArgumentParser& program, 
          std::vector<double>& kernel_times) {
  const int nruns = program.get<int>("--num-runs");
//  const uint64_t len = program.get<uint64_t>("--len");
  std::vector<uint64_t> dims = program.get<std::vector<uint64_t>>("--dims");
  std::fstream log(logname+std::string(".csv"), std::ios::app);
  if(log.tellp() == 0) {
    log << "Kernel,Mode,Data length,Num runs,Kernel time" << std::endl;
  }
  // Configuration
  std::string config_str = "";
  config_str += kernel_name + ",";
  config_str += mode_name + ",";
  config_str += std::to_string(len) + ",";
  config_str += std::to_string(nruns) + ",";

  // Times
  for(uint64_t i=0; i<kernel_times.size(); i++) {
    log << config_str;
    log << std::to_string(kernel_times[i]) << std::endl;
  }
}

void init_matrix(Kokkos::View<double**>& mat) {
  Kokkos::Random_XorShift64_Pool<> pool(12345);
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> fill_policy({0,0},{mat.extent(0),mat.extent(1)});
  Kokkos::parallel_for("Fill matrix with random", fill_policy, 
    KOKKOS_LAMBDA(const uint64_t i, const uint64_t j) {
    auto gen = pool.get_state();
    mat(i,j) = gen.drand();
    pool.free_state(gen);
  });
}

std::map<std::string,std::vector<double>>
test_axpy(argparse::ArgumentParser& program, const std::string& mode) {
  const int nruns = program.get<int>("--num-runs");
//  const uint64_t len = program.get<uint64_t>("--len");
  const uint64_t len = 24000000000/(sizeof(double)*2);
  Kokkos::View<double*> X("Vector X", len), Y("Vector Y", len);
  const double a = 0.5;
  std::map<std::string, std::vector<double> > times;
  std::vector<double> auto_times, guided_times, manual_times, ad_hoc_times;

  for(int i=0; i<nruns; i++) {
    Kokkos::deep_copy(X, 1.0);
    Kokkos::deep_copy(Y, 0.0);
    
    if(mode.compare("auto") == 0) {
      BEG_REGION(Auto);
      axpy_auto(a, X, Y );
      END_REGION(Auto, auto_times);
    } else if (mode.compare("guided") == 0) {
      BEG_REGION(Guided);
      axpy_guided(a, X, Y );
      END_REGION(Guided, guided_times);
    } else if (mode.compare("manual") == 0) {
      BEG_REGION(Manual);
      axpy_manual(a, X, Y );
      END_REGION(Manual, manual_times);
    } else {
      BEG_REGION(Auto);
      axpy_auto(a, X, Y );
      END_REGION(Auto, auto_times);
      BEG_REGION(Guided);
      axpy_guided(a, X, Y );
      END_REGION(Guided, guided_times);
      BEG_REGION(Manual);
      axpy_manual(a, X, Y );
      END_REGION(Manual, manual_times);
    }
  }
  times[std::string("auto")] = auto_times;
  times[std::string("guided")] = guided_times;
  times[std::string("manual")] = manual_times;
  return times;
}

std::map<std::string,std::vector<double>>
test_planckian(argparse::ArgumentParser& program, const std::string& mode) {
  const int nruns = program.get<int>("--num-runs");
//  const uint64_t len = program.get<uint64_t>("--len");
  const uint64_t len = 24000000000/(sizeof(double)*5);
  Kokkos::View<double*> X("Vector X", len), Y("Vector Y", len), U("Vector U", len), V("Vector V", len), W("Vector W", len);
  std::map<std::string, std::vector<double> > times;
  std::vector<double> auto_times, guided_times, manual_times, ad_hoc_times;

  for(int i=0; i<nruns; i++) {
    Kokkos::deep_copy(U, 2.0);
    Kokkos::deep_copy(V, 3.0);
    Kokkos::deep_copy(W, 0.0);
    Kokkos::deep_copy(X, 1.0);
    Kokkos::deep_copy(Y, 0.0);
    
    if(mode.compare("auto") == 0) {
      BEG_REGION(Auto);
      planckian_auto(U, V, W, X, Y );
      END_REGION(Auto, auto_times);
    } else if (mode.compare("guided") == 0) {
      BEG_REGION(Guided);
      planckian_guided(U, V, W, X, Y );
      END_REGION(Guided, guided_times);
    } else if (mode.compare("manual") == 0) {
      BEG_REGION(Manual);
      planckian_manual(U, V, W, X, Y );
      END_REGION(Manual, manual_times);
    } else {
      BEG_REGION(Auto);
      planckian_auto(U, V, W, X, Y );
      END_REGION(Auto, auto_times);
      BEG_REGION(Guided);
      planckian_guided(U, V, W, X, Y );
      END_REGION(Guided, guided_times);
      BEG_REGION(Manual);
      planckian_manual(U, V, W, X, Y );
      END_REGION(Manual, manual_times);
    }
  }
  times[std::string("auto")] = auto_times;
  times[std::string("guided")] = guided_times;
  times[std::string("manual")] = manual_times;
  return times;
}

std::map<std::string,std::vector<double>>
test_fir(argparse::ArgumentParser& program, const std::string& mode) {
  const int nruns = program.get<int>("--num-runs");
  //const uint64_t len = program.get<uint64_t>("--len");
  const uint64_t len = 24000000000/(sizeof(double)*2);
  Kokkos::View<double*> out("Vector out", len), in("Vector in", len+16);
  std::map<std::string, std::vector<double> > times;
  std::vector<double> auto_times, guided_times, manual_times, ad_hoc_times;

  for(int i=0; i<nruns; i++) {
    Kokkos::deep_copy(in, 1.0);
    Kokkos::deep_copy(out, 0.0);
    
    if(mode.compare("auto") == 0) {
      BEG_REGION(Auto);
      fir_auto(in, out );
      END_REGION(Auto, auto_times);
    } else if (mode.compare("guided") == 0) {
      BEG_REGION(Guided);
      fir_guided(in, out );
      END_REGION(Guided, guided_times);
    } else if (mode.compare("manual") == 0) {
      BEG_REGION(Manual);
      fir_manual(in, out );
      END_REGION(Manual, manual_times);
    } else {
      BEG_REGION(Auto);
      fir_auto(in, out );
      END_REGION(Auto, auto_times);
      BEG_REGION(Guided);
      fir_guided(in, out );
      END_REGION(Guided, guided_times);
      BEG_REGION(Manual);
      fir_manual(in, out );
      END_REGION(Manual, manual_times);
    }
  }
  times[std::string("auto")] = auto_times;
  times[std::string("guided")] = guided_times;
  times[std::string("manual")] = manual_times;
  return times;
}

std::map<std::string,std::vector<double>>
test_pressure(argparse::ArgumentParser& program, const std::string& mode) {
  const int nruns = program.get<int>("--num-runs");
  const uint64_t len = 24000000000/(sizeof(double)*5);

  const double cls = 2.0, p_cut = 2.0, pmin = 2.0, eosvmax = 2.0;
  Kokkos::View<double*> compression("Compression View", len), bvc("bvc View", len);
  Kokkos::View<double*> p_new("p_new View", len), e_old("e_old View", len);
  Kokkos::View<double*> vnewc("vnewc View", len);
  std::map<std::string, std::vector<double> > times;
  std::vector<double> auto_times, guided_times, manual_times, ad_hoc_times;

  for(int i=0; i<nruns; i++) {
    Kokkos::deep_copy(compression, 2.0);
    Kokkos::deep_copy(e_old, 2.0);
    Kokkos::deep_copy(bvc, 2.0);
    Kokkos::deep_copy(vnewc, 2.0);
    Kokkos::deep_copy(p_new, 0.0);
    
    if(mode.compare("auto") == 0) {
      BEG_REGION(Auto);
      pressure_auto( cls, p_cut, pmin, eosvmax, compression, bvc, p_new, e_old, vnewc );
      END_REGION(Auto, auto_times);
    } else if (mode.compare("guided") == 0) {
      BEG_REGION(Guided);
      pressure_guided( cls, p_cut, pmin, eosvmax, compression, bvc, p_new, e_old, vnewc );
      END_REGION(Guided, guided_times);
    } else if (mode.compare("manual") == 0) {
      BEG_REGION(Manual);
      pressure_manual( cls, p_cut, pmin, eosvmax, compression, bvc, p_new, e_old, vnewc );
      END_REGION(Manual, manual_times);
    } else {
      BEG_REGION(Auto);
      pressure_auto( cls, p_cut, pmin, eosvmax, compression, bvc, p_new, e_old, vnewc );
      END_REGION(Auto, auto_times);
      BEG_REGION(Guided);
      pressure_guided( cls, p_cut, pmin, eosvmax, compression, bvc, p_new, e_old, vnewc );
      END_REGION(Guided, guided_times);
      BEG_REGION(Manual);
      pressure_manual( cls, p_cut, pmin, eosvmax, compression, bvc, p_new, e_old, vnewc );
      END_REGION(Manual, manual_times);
    }
  }
  times[std::string("auto")] = auto_times;
  times[std::string("guided")] = guided_times;
  times[std::string("manual")] = manual_times;
  return times;
}

std::map<std::string,std::vector<double>>
test_if_quad(argparse::ArgumentParser& program, const std::string& mode) {
  const int nruns = program.get<int>("--num-runs");
  const uint64_t len = 24000000000/(sizeof(double)*5);

  Kokkos::View<double*> a("a View", len), b("b View", len), c("c View", len);
  Kokkos::View<double*> x1("x1 View", len), x2("x2 View", len);
  std::map<std::string, std::vector<double> > times;
  std::vector<double> auto_times, guided_times, manual_times, ad_hoc_times;

  for(int i=0; i<nruns; i++) {
    Kokkos::deep_copy(a, 1.0);
    Kokkos::deep_copy(b, 2.0);
    Kokkos::deep_copy(c, 3.0);
    
    if(mode.compare("auto") == 0) {
      BEG_REGION(Auto);
      if_quad_auto( a, b, c, x1, x2 );
      END_REGION(Auto, auto_times);
    } else if (mode.compare("guided") == 0) {
      BEG_REGION(Guided);
      if_quad_guided( a, b, c, x1, x2 );
      END_REGION(Guided, guided_times);
    } else if (mode.compare("manual") == 0) {
      BEG_REGION(Manual);
      if_quad_manual( a, b, c, x1, x2 );
      END_REGION(Manual, manual_times);
    } else {
      BEG_REGION(Auto);
      if_quad_auto( a, b, c, x1, x2 );
      END_REGION(Auto, auto_times);
      BEG_REGION(Guided);
      if_quad_guided( a, b, c, x1, x2 );
      END_REGION(Guided, guided_times);
      BEG_REGION(Manual);
      if_quad_manual( a, b, c, x1, x2 );
      END_REGION(Manual, manual_times);
    }
  }
  times[std::string("auto")] = auto_times;
  times[std::string("guided")] = guided_times;
  times[std::string("manual")] = manual_times;
  return times;
}

std::map<std::string,std::vector<double>>
test_pi_reduce(argparse::ArgumentParser& program, const std::string& mode) {
  const int nruns = program.get<int>("--num-runs");
  const uint64_t len = 24000000000;

  double pi = 0.0;
  double dx = 1.0 / len;
  std::map<std::string, std::vector<double> > times;
  std::vector<double> auto_times, guided_times, manual_times, ad_hoc_times;

  for(int i=0; i<nruns; i++) {
    if(mode.compare("auto") == 0) {
      BEG_REGION(Auto);
      pi_reduce_auto( pi, dx, len );
      END_REGION(Auto, auto_times);
    } else if (mode.compare("guided") == 0) {
      BEG_REGION(Guided);
      pi_reduce_guided( pi, dx, len );
      END_REGION(Guided, guided_times);
    } else if (mode.compare("manual") == 0) {
      BEG_REGION(Manual);
      pi_reduce_manual( pi, dx, len );
      END_REGION(Manual, manual_times);
    } else {
      BEG_REGION(Auto);
      pi_reduce_auto( pi, dx, len );
      END_REGION(Auto, auto_times);
      BEG_REGION(Guided);
      pi_reduce_guided( pi, dx, len );
      END_REGION(Guided, guided_times);
      BEG_REGION(Manual);
      pi_reduce_manual( pi, dx, len );
      END_REGION(Manual, manual_times);
    }
  }
  times[std::string("auto")] = auto_times;
  times[std::string("guided")] = guided_times;
  times[std::string("manual")] = manual_times;
  return times;
}

int 
main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    argparse::ArgumentParser program("vec-bench");
    program.add_description("Benchmark for different vectorization methods");
    program.add_argument("bench-mode")
      .help("Benchmark mode")
      .choices("all", "auto", "guided", "manual", "ad-hoc")
      .default_value("all");
    program.add_argument("-d", "--dims")
      .help("Dimensions of matrix")
      .nargs(1,2)
      .default_value(std::vector<uint64_t>{10000LLU,10000LLU})
      .scan<'u', uint64_t>();
    program.add_argument("-l", "--limit")
      .help("Memory limit in bytes")
      .nargs(1)
      .default_value(static_cast<uint64_t>(24'000'000'000LLU))
      .scan<'u', uint64_t>();
    program.add_argument("-n", "--num-runs")
      .help("Number of iterations to run")
      .default_value(5)
      .scan<'d', int>();
    program.add_argument("--log")
      .help("Filename for logging results")
      .default_value("result_log");
    program.parse_args(argc, argv);
    
    std::vector<uint64_t> dims = program.get<std::vector<uint64_t>>("--dims");
    uint64_t nrows=dims[0], ncols=dims[1];
    const int nruns = program.get<int>("--num-runs");
    const uint64_t mem_lim = program.get<uint64_t>("--limit");
    std::vector<double> ref_times, guided_times, manual_times, ad_hoc_times;
    std::string mode_str = program.get<std::string>("bench-mode");
    std::string logname = program.get<std::string>("--log");

//    using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
//    using matrix_t = Kokkos::View<double**>;
//    matrix_t A("Matrix A", nrows,ncols), B("Matrix B", nrows,ncols), C("Matrix C", nrows,ncols);
//    init_matrix(A);
//    init_matrix(B);
//    init_matrix(C);
//    Kokkos::fence();
//    double alpha = 1.0;
//    double beta = 1.0;

//    for(int i=0; i<nruns; i++) {
//      BEG_REGION(Reference);
//      KokkosBlas::gemm("N", "N", alpha, A, B, beta, C);
//      END_REGION(Reference, ref_times);
//    }
//    print_results(std::string("GEMM"), mode_str, ref_times);
//    write_log(logname, std::string("GEMM"), mode_str, (A.size()+B.size()+C.size())*sizeof(double), program, ref_times);

    std::map<std::string, std::vector<double> > axpy_times = test_axpy(program, mode_str);
    for(auto& [mode, times] : axpy_times) {
      print_results(std::string("AXPY"), mode, times);
      write_log(logname, std::string("AXPY"), mode, mem_lim/(sizeof(double)*2), program, times);
    }

    std::map<std::string, std::vector<double> > planckian_times = test_planckian(program, mode_str);
    for(auto& [mode, times] : planckian_times) {
      print_results(std::string("PLANCKIAN"), mode, times);
      write_log(logname, std::string("PLANCKIAN"), mode, mem_lim/(sizeof(double)*5), program, times);
    }

    std::map<std::string, std::vector<double> > fir_times = test_fir(program, mode_str);
    for(auto& [mode, times] : fir_times) {
      print_results(std::string("FIR"), mode, times);
      write_log(logname, std::string("FIR"), mode, mem_lim/(sizeof(double)*2), program, times);
    }

    std::map<std::string, std::vector<double> > pressure_times = test_pressure(program, mode_str);
    for(auto& [mode, times] : pressure_times) {
      print_results(std::string("PRESSURE"), mode, times);
      write_log(logname, std::string("PRESSURE"), mode, mem_lim/(sizeof(double)*2), program, times);
    }

    std::map<std::string, std::vector<double> > if_quad_times = test_if_quad(program, mode_str);
    for(auto& [mode, times] : if_quad_times) {
      print_results(std::string("IF_QUAD"), mode, times);
      write_log(logname, std::string("IF_QUAD"), mode, mem_lim/(sizeof(double)*2), program, times);
    }

    std::map<std::string, std::vector<double> > pi_reduce_times = test_pi_reduce(program, mode_str);
    for(auto& [mode, times] : pi_reduce_times) {
      print_results(std::string("PI_REDUCE"), mode, times);
      write_log(logname, std::string("PI_REDUCE"), mode, mem_lim, program, times);
    }

    return 0;
  }
}
