LIST(APPEND TESTS weibel_driver)

list(APPEND gold_file "${CMAKE_CURRENT_SOURCE_DIR}/energies_gold")
add_definitions(-DGOLD_ENERGY_FILE="${gold_file}")

foreach (test ${TESTS})
    #build_a_vpic(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cc)
    #add_test(NAME ${test} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./${test})

    add_executable(${test} ./${test}.cc)
    target_link_libraries(${test} vpic ${LINK_LIBS})
    add_test(NAME ${test} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./${test})
endforeach(test)
