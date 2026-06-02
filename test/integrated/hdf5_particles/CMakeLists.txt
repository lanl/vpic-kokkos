# add the tests
set(MPI_NUM_RANKS 1)

set(TEST_HDF5 "hdf5_particles")
set(TEST_BINARY "binary_particles")
find_package(Python3 COMPONENTS Interpreter)


if(VPIC_ENABLE_HDF5)
    # Build
    build_a_vpic(${TEST_HDF5} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_HDF5}.deck)
    build_a_vpic(${TEST_BINARY} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_BINARY}.deck)

    # Add TESTs
    add_TEST(${TEST_HDF5} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPI_NUM_RANKS} ${MPIEXEC_PREFLAGS} ./${TEST_HDF5} ${MPIEXEC_POSTFLAGS} ${ARGS} --kokkos-num-threads=1)
    add_TEST(${TEST_BINARY} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPI_NUM_RANKS} ${MPIEXEC_PREFLAGS} ./${TEST_BINARY} ${MPIEXEC_POSTFLAGS} ${ARGS} --kokkos-num-threads=1)

    #and check that the output is ok
    if(VPIC_ENABLE_VARIABLE_CHARGE)
        add_TEST(particles_compare_binary_hdf5 python3 ${CMAKE_CURRENT_SOURCE_DIR}/readback.py ${CMAKE_CURRENT_BINARY_DIR} --variable-charge)
    else()
        add_TEST(particles_compare_binary_hdf5 python3 ${CMAKE_CURRENT_SOURCE_DIR}/readback.py ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endif()
