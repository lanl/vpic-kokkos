LIST(APPEND TESTS chex_driver)

foreach (test ${TESTS})
    add_executable(${test} ./${test}.cc)
    target_link_libraries(${test} vpic Kokkos::kokkos)
    add_test(NAME ${test} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./${test})
endforeach(test)
