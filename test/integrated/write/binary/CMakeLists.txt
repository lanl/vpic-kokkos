# ==============================================================================
# Binary I/O Tests (Always Enabled)
# ==============================================================================
# Tests binary I/O correctness against HDF5 reference outputs
# Requires: Python3, h5py, numpy

# ------------------------------------------------------------------------------
# Build Test Deck Executables
# ------------------------------------------------------------------------------
# build_a_vpic creates standalone executables from input deck (.cxx) files
# The macro signature: build_a_vpic(executable_name path_to_deck.cxx)

build_a_vpic(test_io_binary_single ${CMAKE_CURRENT_SOURCE_DIR}/write_correctness.cxx)

build_a_vpic(test_io_binary_4rank ${CMAKE_CURRENT_SOURCE_DIR}/write_correctness_4rank.cxx)

# Test 1: Single-Rank Binary Verification
add_test(
  NAME vpic_io_binary_single
  COMMAND ${CMAKE_COMMAND}
    -DMPIEXEC=${MPIEXEC}
    -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
    -DMPI_RANKS=1
    -DMPI_OVERSUBSCRIBE=${MPI_OVERSUBSCRIBE_FLAG}
    -DTEST_EXEC=$<TARGET_FILE:test_io_binary_single>
    -DVERIFY=${Python3_EXECUTABLE}
    -DVERIFY_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/verify_binary.py  # ← SAME SCRIPT
    "-DVERIFY_ARGS=--mode;single"
    -P ${CMAKE_SOURCE_DIR}/cmake/run_and_verify_mpi.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Test 2: 4-Rank Binary Verification
add_test(
  NAME vpic_io_binary_4rank
  COMMAND ${CMAKE_COMMAND}
    -DMPIEXEC=${MPIEXEC}
    -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
    -DMPI_RANKS=4
    -DMPI_OVERSUBSCRIBE=${MPI_OVERSUBSCRIBE_FLAG}
    -DTEST_EXEC=$<TARGET_FILE:test_io_binary_4rank>
    -DVERIFY=${Python3_EXECUTABLE}
    -DVERIFY_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/verify_binary.py  # ← SAME SCRIPT, DIFFERENT MODE
    "-DVERIFY_ARGS=--mode;4rank"
    -P ${CMAKE_SOURCE_DIR}/cmake/run_and_verify_mpi.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_tests_properties(vpic_io_binary_4rank PROPERTIES
  LABELS "io;binary;correctness;multirank"
  TIMEOUT 600
)

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
# Install test decks and verification scripts for manual testing

install(FILES
  write_correctness.cxx
  write_correctness_4rank.cxx
  verify_binary.py
  verify_binary_4rank.py
  DESTINATION share/vpic/tests/io/binary
  COMPONENT tests
)

# ------------------------------------------------------------------------------
# Usage Documentation
# ------------------------------------------------------------------------------
# To run tests manually:
#
# 1. Build test executables:
#    cmake --build build --target test_io_binary_single test_io_binary_4rank
#
# 2. Single-rank test:
#    cd build
#    mpirun -np 1 ./test_io_binary_single
#    python3 ../src/tests/io/binary/verify_binary.py --mode single
#
# 3. 4-rank test:
#    mpirun -np 4 --oversubscribe ./test_io_binary_4rank
#    python3 ../src/tests/io/binary/verify_binary_4rank.py --mode 4rank
#
# 4. Via CMake/CTest:
#    ctest -R vpic_io_binary -V