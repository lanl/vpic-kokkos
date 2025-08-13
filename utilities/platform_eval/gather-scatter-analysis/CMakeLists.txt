cmake_minimum_required(VERSION 3.9)
cmake_policy(SET CMP0074 NEW)

#~----------------------------------------------------------------------------~#
# VPIC project configuration
#~----------------------------------------------------------------------------~#

project(s-g-bench CXX)

#------------------------------------------------------------------------------#
# If a C++14 compiler is available, then set the appropriate flags
# Note: As of February 2021, VPIC requires only C++11, but requiring C++14
# makes building Kokkos easier.  If you are adventurous and using your own
# Kokkos build, feel free to try a lower standard.
# Fixing the this capture requires C++17, but if you don't mind the warnings,
# feel free to experiment with lower standards.
#------------------------------------------------------------------------------#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#------------------------------------------------------------------------------#
# MPI detection
#------------------------------------------------------------------------------#

# TODO: it seems like Cmake version 3.12 (and likely others) have a bug/change
# where it has a hard time detecting MPI. Cmake version. 3.9.0 works great
# though..
#find_package(MPI REQUIRED)
#list(APPEND MPI_CXX_COMPILE_FLAGS -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX )
#include_directories(${MPI_C_INCLUDE_PATH})
#include_directories(${MPI_CXX_INCLUDE_PATH})

#------------------------------------------------------------------------------#
# Add build options
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Create include and link aggregates
#
# NOTE: These must be set before creating the compile scripts below.
#------------------------------------------------------------------------------#

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

#------------------------------------------------------------------------------#
# OpenMP
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Kokkos
#------------------------------------------------------------------------------#
set(Kokkos_ENABLE_DEBUG OFF)

option(VPIC_KOKKOS_DEBUG "Enable Debug In Kokkos" OFF)

if (VPIC_KOKKOS_DEBUG)
    set(Kokkos_ENABLE_DEBUG ON)
endif(VPIC_KOKKOS_DEBUG)

set(Kokkos_ENABLE_EXPLICIT_INSTANTIATION OFF)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Default allow serial execution")

if(ENABLE_KOKKOS_OPENMP)
  if (OPENMP_FOUND)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Use OpenMP if found")
  endif()
else()
    set(Kokkos_ENABLE_OPENMP OFF)
endif()

if(ENABLE_KOKKOS_CUDA)
  find_package(CUDA)
  if (CUDA_FOUND)
    add_definitions(-DENABLE_KOKKOS_CUDA)
    # If not on a Cray system, this can cause CMake to stall
    if(DEFINED ENV{CRAYPE_VERSION})
      set(ENV{NVCC_WRAPPER_DEFAULT_COMPILER} ${CMAKE_CXX_COMPILER})
    endif()
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Use Cuda with Kokkos")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable cuda lambda support")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")

    # turn off -std=gnu++ to avoid error: identifier "__float128" is undefined issue
    # see https://github.com/UoB-HPC/BabelStream/issues/16
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Don't use intel intrinsics if we are using CUDA
    option(USE_V4_ALTIVEC "Enable V4 Altivec" OFF)
    option(USE_V4_PORTABLE "Enable V4 Portable" OFF)
    option(USE_V4_SSE "Enable V4 SSE" OFF)

  else()
    message("CUDA not found and options not changed")
  endif()
endif()

# Disable platform specific options
if (APPLE)
  set(Kokkos_ENABLE_LIBRT OFF CACHE BOOL "librt is not available on macOS")
endif()

# TODO: This is only need for non-linux platforms
set(Kokkos_ENABLE_LIBDL OFF)
set(Kokkos_ENABLE_PROFILING OFF) # NO libDL => no profiling

if (BUILD_INTERNAL_KOKKOS)
  message("Building our own kokkos..")
  add_subdirectory(kokkos)
  set(Kokkos_ENABLE_LIBDL OFF)
  set(Kokkos_ENABLE_PROFILING OFF) # NO libDL => no profiling
else()
  find_package(Kokkos REQUIRED)
endif(BUILD_INTERNAL_KOKKOS)
#------------------------------------------------------------------------------#
# Argparse
#------------------------------------------------------------------------------#
# fetch latest argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

#------------------------------------------------------------------------------#
# OpenSSL
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Act on build options set in project.cmake
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Output log variables
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Handle vpic compile script last
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Add library target
#------------------------------------------------------------------------------#

add_executable(s-g-bench src/main.cpp)
target_include_directories(s-g-bench INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(s-g-bench  Kokkos::kokkos argparse)
#target_compile_options(s-g-sort)

#------------------------------------------------------------------------------#
# Add VPIC integrated test mechanism
#------------------------------------------------------------------------------#

enable_testing()

# NOTE: This can also be done by a subdirectory (useful when we have multiple tests)
# Add includes for Catch2
set(TEST_DIR "./test/include")
include_directories(${TEST_DIR})

add_subdirectory(test)

#~---------------------------------------------------------------------------~-#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#~---------------------------------------------------------------------------~-#
