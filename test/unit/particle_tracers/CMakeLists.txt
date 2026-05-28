# add the tests
set(MPI_NUM_RANKS 1)

set(TESTS "particle_tracers" "pcai_tracers")
find_package(Python3 COMPONENTS Interpreter)

if(VPIC_ENABLE_TRACER_PARTICLES)
    # Build
    # TODO: This method of doing the tests is really bad at rebuilding them properly
    add_definitions(-DELECTRON_REFERENCE_POSITION_FILE="${CMAKE_CURRENT_SOURCE_DIR}/electron_pos_reference")
    add_definitions(-DION2_REFERENCE_POSITION_FILE="${CMAKE_CURRENT_SOURCE_DIR}/I2_pos_reference")
    foreach(test ${TESTS})
        build_a_vpic(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.deck)
    endforeach()

    # Add test
    foreach(test ${TESTS})
        add_test(${test} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./${test} ${MPIEXEC_POSTFLAGS} ${ARGS})
    endforeach()

    #and check that the output is ok
    add_TEST(verify_tracer_write python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_hdf5.py ${CMAKE_CURRENT_BINARY_DIR})
endif()
