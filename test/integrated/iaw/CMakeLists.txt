# add the tests
set(MPI_NUM_RANKS 1)

find_package(Python3 COMPONENTS Interpreter)
build_a_vpic(iaw ${CMAKE_CURRENT_SOURCE_DIR}/iaw.cxx)
add_TEST(iaw ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPI_NUM_RANKS} ${MPIEXEC_PREFLAGS} --map-by socket ./iaw ${MPIEXEC_POSTFLAGS} ${ARGS} --kokkos-num-threads=32)
add_TEST(verify_iaw python3 ${CMAKE_CURRENT_SOURCE_DIR}/verifyIAW.py ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(verify_iaw PROPERTIES DEPENDS iaw)


