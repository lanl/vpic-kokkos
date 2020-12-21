# add the tests
set(MPI_NUM_RANKS 1)
set(ARGS "1 1")

list(APPEND DEFAULT_ARG_TESTS cyl-orbit cyl-standing-wave)
list(APPEND ALL_TESTS ${DEFAULT_ARG_TESTS})

foreach(test ${ALL_TESTS})
  build_a_vpic(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.deck)
endforeach()

foreach(test ${DEFAULT_ARG_TESTS})
  add_test(${test} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${test} ${MPIEXEC_POSTFLAGS} ${ARGS})
endforeach()
