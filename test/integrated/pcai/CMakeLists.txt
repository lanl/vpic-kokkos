# add the tests
set(MPI_NUM_RANKS 1)

cmake_host_system_information(RESULT ncores QUERY NUMBER_OF_PHYSICAL_CORES)
if(NOT ncores EQUAL 0)
  if(ncores GREATER 32)
    set(KOKKOS_NUM_THREADS 32)
  else()
    set(KOKKOS_NUM_THREADS ${ncores})
  endif()
else()
  set(KOKKOS_NUM_THREADS 2)
endif()

find_package(Python3 COMPONENTS Interpreter)
build_a_vpic(pcai ${CMAKE_CURRENT_SOURCE_DIR}/pcai.cxx)
add_TEST(pcai ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPI_NUM_RANKS} ${MPIEXEC_PREFLAGS} --bind-to socket ./pcai ${MPIEXEC_POSTFLAGS} ${ARGS} --kokkos-num-threads=${KOKKOS_NUM_THREADS})
add_TEST(verify_pcai python3 ${CMAKE_CURRENT_SOURCE_DIR}/verifyPCAI.py ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(verify_pcai PROPERTIES DEPENDS pcai)

