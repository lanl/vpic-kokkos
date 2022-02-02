# WARNING: None of these tests test correctness, only that they don't die.

set(MPIEXEC_NUMPROC 1)

# TODO: If would be good if we could detect that the machine supports the
# desired number of threads
set(MPIEXEC_NUMPROC_PARALLEL 2)
set(NUM_THREADS 8)

set(ARGS "")

list(APPEND DEFAULT_ARG_TESTS
    simple # This test is a simple run which should not die
    dump # This is a simple run which should dump restart files
    reconnection_test # This is a simple reconnection run
    )

list(APPEND SIMPLE_DECK simple) # Reuse existing deck and start half way
list(APPEND RESTART_DECK dump) # Reuse existing deck and start half way
list(APPEND RESTART_BINARY restore)
list(APPEND CHECKPOINT_FILE "${CMAKE_CURRENT_BINARY_DIR}/checkpt_test.1")
list(APPEND RESTART_ARGS --restore ${CHECKPOINT_FILE})

foreach(test ${DEFAULT_ARG_TESTS})
    build_a_vpic(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.deck)
endforeach()

foreach(test ${DEFAULT_ARG_TESTS})
    add_test(${test} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_NUMPROC}
        ${MPIEXEC_PREFLAGS} ./${test} ${MPIEXEC_POSTFLAGS} ${ARGS})
endforeach()

# TODO: re-enable parallel run
# Try a parallel run
#set (PARALLEL_TEST parallel)
#build_a_vpic(${PARALLEL_TEST} ${CMAKE_CURRENT_SOURCE_DIR}/simple.deck)
#add_test(${PARALLEL_TEST} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
#${MPIEXEC_NUMPROC_PARALLEL} ${MPIEXEC_PREFLAGS} ${PARALLEL_TEST}
#${MPIEXEC_POSTFLAGS} ${ARGS})

# Try a threaded run
#set (THREADED_TEST threaded)
#list(APPEND THREADED_ARGS --tpp ${NUM_THREADS})

#build_a_vpic(${THREADED_TEST} ${CMAKE_CURRENT_SOURCE_DIR}/simple.deck)
#add_test(${THREADED_TEST} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_NUMPROC}
#${MPIEXEC_PREFLAGS} ${THREADED_TEST} ${MPIEXEC_POSTFLAGS} ${THREADED_ARGS})

set(generate_restore "generate_${RESTART_BINARY}")
build_a_vpic(${generate_restore} ${CMAKE_CURRENT_SOURCE_DIR}/${RESTART_DECK}.deck)
add_test(${generate_restore} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
${MPIEXEC_NUMPROC} ${MPIEXEC_PREFLAGS} ./${generate_restore}
${MPIEXEC_POSTFLAGS})

# Run using the restore file
# TODO: caps?
set(perform_restore "perform_${RESTART_BINARY}")
build_a_vpic(${perform_restore} ${CMAKE_CURRENT_SOURCE_DIR}/${RESTART_DECK}.deck)
add_test(${perform_restore} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
${MPIEXEC_NUMPROC} ${MPIEXEC_PREFLAGS} ./${perform_restore}
${MPIEXEC_POSTFLAGS} ${RESTART_ARGS})

# Link the two tests
set(RESTORE_LABEL "restore_group")
set_tests_properties(${perform_restore} PROPERTIES DEPENDS ${generate_restore})
#set_property(TEST ${generate_restore} PROPERTY FIXTURES_SETUP ${RESTORE_LABEL})
#set_property(TEST ${perform_restore} PROPERTY FIXTURES_REQUIRED ${RESTORE_LABEL})

# See if we can build a deck with the 2 stage compile
# TODO: add a test that runs the generated deck
add_test(
    NAME perform_2stage_compile
    COMMAND ${CMAKE_BINARY_DIR}/bin/vpic ${CMAKE_CURRENT_SOURCE_DIR}/${SIMPLE_DECK}.deck
)

set(TWOSTAGE_OUT ${SIMPLE_DECK}.deck.${CMAKE_SYSTEM_NAME})
add_test(run_2stage_compile ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
    ${MPIEXEC_NUMPROC} ${MPIEXEC_PREFLAGS} ./${TWOSTAGE_OUT}
)

set_tests_properties(perform_2stage_compile PROPERTIES FIXTURES_SETUP 2stage_fixture)
set_tests_properties(run_2stage_compile PROPERTIES FIXTURES_REQUIRED 2stage_fixture)
