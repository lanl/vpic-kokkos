# ==============================================================================
# HDF5 I/O Tests
# ==============================================================================
# Tests HDF5 I/O correctness in both M2M and M2O modes
# Requires: VPIC_ENABLE_HDF5=ON, Python3, h5py, numpy

# ------------------------------------------------------------------------------
# Build Test Deck Executables
# ------------------------------------------------------------------------------
# build_a_vpic creates standalone executables from input deck (.cxx) files
# The macro signature: build_a_vpic(executable_name path_to_deck.cxx)

build_a_vpic(test_io_hdf5_single ${CMAKE_CURRENT_SOURCE_DIR}/write_correctness.cxx)

build_a_vpic(test_io_hdf5_4rank ${CMAKE_CURRENT_SOURCE_DIR}/write_correctness_4rank.cxx)

# ------------------------------------------------------------------------------
# Test 1: Single-Rank HDF5 Verification
# ------------------------------------------------------------------------------
# Tests M2M and M2O modes with 1 MPI rank (M2O degenerates to M2M behavior)
# Grid: 4×4×4 local (same as global since single rank)

add_test(
  NAME vpic_io_hdf5_single
  COMMAND ${CMAKE_COMMAND}
    -DMPIEXEC=${MPIEXEC}
    -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
    -DMPI_RANKS=1
    -DMPI_OVERSUBSCRIBE=${MPI_OVERSUBSCRIBE_FLAG}
    -DTEST_EXEC=$<TARGET_FILE:test_io_hdf5_single>
    -DVERIFY=${Python3_EXECUTABLE}
    -DVERIFY_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/verify_hdf5.py
    "-DVERIFY_ARGS=--mode;single"
    -P${CMAKE_SOURCE_DIR}/cmake/run_and_verify_mpi.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_tests_properties(vpic_io_hdf5_single PROPERTIES
  LABELS "io;hdf5;correctness"
  TIMEOUT 300
)

# ------------------------------------------------------------------------------
# Test 2: 4-Rank HDF5 Verification
# ------------------------------------------------------------------------------
# Tests M2M and M2O modes with 4 MPI ranks (2×2×1 topology)
# Grid: 8×8×4 global (4×4×4 local per rank)
# This is the critical test for parallel HDF5 collective I/O

add_test(
  NAME vpic_io_hdf5_4rank
  COMMAND ${CMAKE_COMMAND}
    -DMPIEXEC=${MPIEXEC}
    -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
    -DMPI_RANKS=4
    -DMPI_OVERSUBSCRIBE=${MPI_OVERSUBSCRIBE_FLAG}
    -DTEST_EXEC=$<TARGET_FILE:test_io_hdf5_4rank>
    -DVERIFY=${Python3_EXECUTABLE}
    -DVERIFY_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/verify_hdf5.py
    "-DVERIFY_ARGS=--mode;4rank"
    -P ${CMAKE_SOURCE_DIR}/cmake/run_and_verify_mpi.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_tests_properties(vpic_io_hdf5_4rank PROPERTIES
  LABELS "io;hdf5;correctness;parallel"
  TIMEOUT 300
)

# Only run 4-rank test if parallel HDF5 is available
if(NOT HDF5_IS_PARALLEL)
  set_tests_properties(vpic_io_hdf5_4rank PROPERTIES DISABLED TRUE)
  message(STATUS "HDF5 4-rank test disabled (requires parallel HDF5)")
endif()

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
# Install test decks and verification scripts for manual testing

install(FILES
  write_correctness.cxx
  write_correctness_4rank.cxx
  verify_hdf5.py
  verify_hdf5_4rank.py
  DESTINATION share/vpic/tests/io/hdf5
  COMPONENT tests
)

# ------------------------------------------------------------------------------
# Usage Documentation
# ------------------------------------------------------------------------------
# To run tests manually:
#
# 1. Build test executables:
#    cmake --build build --target test_io_hdf5_single test_io_hdf5_4rank
#
# 2. Single-rank test:
#    cd build
#    ./test_io_hdf5_single
#    python3 ../src/tests/io/hdf5/verify_hdf5.py --mode single
#
# 3. 4-rank test:
#    mpirun -np 4 --oversubscribe ./test_io_hdf5_4rank
#    python3 ../src/tests/io/hdf5/verify_hdf5_4rank.py --mode 4rank
#
# 4. Via CMake/CTest:
#    ctest -R vpic_io_hdf5 -V
#
# 5. Debug mode (keep output):
#    ctest -R vpic_io_hdf5_single_debug -V