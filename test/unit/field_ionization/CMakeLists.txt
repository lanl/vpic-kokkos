LIST(APPEND TESTS field_ionization_benchmark)

list(APPEND gold_file "${CMAKE_CURRENT_SOURCE_DIR}/SMILIE_hydrogen_analytic.txt")
add_definitions(-DGOLD_FILE="${gold_file}")

foreach (test ${TESTS})
    add_executable(${test} ./${test}.cxx)
    target_link_libraries(${test} vpic Kokkos::kokkos)
    add_test(NAME ${test} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./${test})
endforeach(test)
