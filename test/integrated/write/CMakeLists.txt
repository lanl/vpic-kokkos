# ==============================================================================
# VPIC I/O Correctness Tests
# ==============================================================================
# Tests the new binary and HDF5 writers against legacy reference data
# 
# Test Modes:
#   - Binary: Always runs (no external dependencies)
#   - HDF5:   Only runs if VPIC_ENABLE_HDF5=ON
#
# Each mode has:
#   - Single-rank test (basic correctness)
#   - 4-rank test (domain decomposition + MPI oversubscription support)
#
# Cleanup Behavior:
#   - Success: Output files removed automatically
#   - Failure: Output preserved for debugging
# ==============================================================================

cmake_minimum_required(VERSION 3.18)
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_EXECUTABLE)
  find_program(Python3_EXECUTABLE python3 REQUIRED)
endif()

# ==============================================================================
# MPI LAUNCHER CONFIGURATION
# ==============================================================================
# Universal approach: Use mpirun for tests (works inside/outside allocations).
# Advanced users can override with: ctest -E . && srun -n 4 ./test_binary
# ==============================================================================

# ==============================================================================
# MPI OVERSUBSCRIPTION DETECTION
# ==============================================================================

# Try multiple methods to find the MPI compiler wrapper
set(MPI_COMPILER_TO_CHECK "")

if(MPI_CXX_COMPILER)
  set(MPI_COMPILER_TO_CHECK ${MPI_CXX_COMPILER})
elseif(MPI_C_COMPILER)
  set(MPI_COMPILER_TO_CHECK ${MPI_C_COMPILER})
elseif(CMAKE_CXX_COMPILER MATCHES "mpi")
  # CMake might have used the MPI wrapper directly
  set(MPI_COMPILER_TO_CHECK ${CMAKE_CXX_COMPILER})
endif()

if(MPI_COMPILER_TO_CHECK)
  execute_process(
    COMMAND ${MPI_COMPILER_TO_CHECK} --version
    OUTPUT_VARIABLE MPI_VERSION_STRING
    ERROR_VARIABLE MPI_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  
  # Detect MPI implementation
  if(MPI_VERSION_STRING MATCHES "Open MPI")
    set(MPI_OVERSUBSCRIBE_FLAG "--oversubscribe")
    message(STATUS "Detected Open MPI - using '${MPI_OVERSUBSCRIBE_FLAG}' flag")
  else()
    set(MPI_OVERSUBSCRIBE_FLAG "")
    message(STATUS "Detected non-OpenMPI - no oversubscription flag needed")
  endif()
else()
  # Fallback: Check if OpenMPI is in the path
  execute_process(
    COMMAND ompi_info --version
    OUTPUT_VARIABLE OMPI_CHECK
    ERROR_QUIET
  )
  
  if(OMPI_CHECK MATCHES "Open MPI")
    set(MPI_OVERSUBSCRIBE_FLAG "--oversubscribe")
    message(STATUS "Detected Open MPI via ompi_info")
  else()
    set(MPI_OVERSUBSCRIBE_FLAG "")
    message(STATUS "MPI compiler not found - assuming no oversubscription flag needed")
  endif()
endif()

# Force mpirun for tests (avoids Slurm PMI configuration hell)
set(MPIEXEC "mpirun" CACHE STRING "MPI launcher for tests" FORCE)
set(MPIEXEC_NUMPROC_FLAG "-np" CACHE STRING "Flag for process count" FORCE)
set(MPIEXEC_PREFLAGS "${MPI_OVERSUBSCRIBE_FLAG}" CACHE STRING "MPI pre-flags" FORCE)

message(STATUS "===============================================")
message(STATUS "TEST LAUNCHER CONFIGURATION")
message(STATUS "  Launcher: ${MPIEXEC}")
message(STATUS "  Proc Flag: ${MPIEXEC_NUMPROC_FLAG}")
message(STATUS "  Extra Flags: ${MPIEXEC_PREFLAGS}")
message(STATUS "  Note: Tests use mpirun for cross-environment compatibility")
message(STATUS "        To use srun, run manually: srun -n 4 --mpi=pmix ./my_test")
message(STATUS "===============================================")

# ==============================================================================
# TEST SUBDIRECTORIES
# ==============================================================================

add_subdirectory(binary)

if(VPIC_ENABLE_HDF5)
  add_subdirectory(hdf5)
  message(STATUS "HDF5 I/O tests enabled")
else()
  message(STATUS "HDF5 I/O tests disabled (VPIC_ENABLE_HDF5=OFF)")
endif()

message(STATUS "I/O Test Configuration:")
message(STATUS "  Binary tests:    Enabled")
message(STATUS "  HDF5 tests:      ${VPIC_ENABLE_HDF5}")
message(STATUS "  MPI launcher:${MPIEXEC}")
message(STATUS "  Oversubscribe:   '${MPI_OVERSUBSCRIBE_FLAG}'")

# ==============================================================================
# SUMMARY
# ==============================================================================

message(STATUS "I/O Test Configuration:")
message(STATUS "  Binary tests:        Enabled")
message(STATUS "  HDF5 tests: ${VPIC_ENABLE_HDF5}")
message(STATUS "  MPI oversubscribe:   ' ${MPI_OVERSUBSCRIBE_FLAG}'")