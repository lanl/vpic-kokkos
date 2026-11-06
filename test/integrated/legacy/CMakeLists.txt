# add the tests
set(ARGS 1 1)

list(APPEND DEFAULT_ARG_TESTS accel cyclo inbndj interpe) # outbndj)
list(APPEND ALL_TESTS ${DEFAULT_ARG_TESTS} pcomm)

foreach(test ${ALL_TESTS})
  build_a_vpic(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.deck)
endforeach()

foreach(test ${DEFAULT_ARG_TESTS})
  vpic_add_test(NAME ${test} COMMAND ./${test} ${ARGS})
endforeach()

# Particle communication: one particle crosses the corners of a 2x2x2
# decomposition and its exact state is checked on the receiving rank.
vpic_add_test(NAME pcomm NP 8 COMMAND ./pcomm ${ARGS})
