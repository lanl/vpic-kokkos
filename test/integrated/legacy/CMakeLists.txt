# add the tests
set(MPI_NUM_RANKS 1)
set(ARGS "1 1")

list(APPEND DEFAULT_ARG_TESTS accel cyclo inbndj interpe) # outbndj)
list(APPEND ALL_TESTS ${DEFAULT_ARG_TESTS}) # pcomm)

foreach(test ${ALL_TESTS})
  build_a_vpic(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.deck)
endforeach()

foreach(test ${DEFAULT_ARG_TESTS})
  add_test(${test} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${test} ${MPIEXEC_POSTFLAGS} ${ARGS})
endforeach()

# TODO: re-enable PCOM once MPI is ready
#add_test(pcomm ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} pcomm ${MPIEXEC_POSTFLAGS} ${ARGS})
