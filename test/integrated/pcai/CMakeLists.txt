# add the tests
set(MPI_NUM_RANKS 1)

find_package(Python3 COMPONENTS Interpreter)
build_a_vpic(pcai ${CMAKE_CURRENT_SOURCE_DIR}/pcai.cxx)
add_TEST(pcai ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPI_NUM_RANKS} ${MPIEXEC_PREFLAGS} ./pcai ${MPIEXEC_POSTFLAGS} ${ARGS})
add_TEST(verify_pcai python3 ${CMAKE_CURRENT_SOURCE_DIR}/verifyPCAI.py ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(verify_pcai PROPERTIES DEPENDS pcai)

