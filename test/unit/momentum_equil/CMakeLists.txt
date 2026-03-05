LIST(APPEND TESTS momentum_equil_driver)

list(APPEND beam_file "${CMAKE_CURRENT_SOURCE_DIR}/beam_energy.csv")
add_definitions(-DBEAM_ENERGY_FILE="${beam_file}")

foreach (test ${TESTS})
    add_executable(${test} ./${test}.cc)
    target_link_libraries(${test} vpic Kokkos::kokkos)
    add_test(NAME ${test} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./${test})
endforeach(test)
