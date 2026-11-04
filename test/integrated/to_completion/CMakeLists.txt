# WARNING: None of these tests test correctness, only that they don't die.

list(APPEND DEFAULT_ARG_TESTS
    simple # This test is a simple run which should not die
    dump # This is a simple run which should dump restart files
    reconnection_test # This is a simple reconnection run
    )

list(APPEND SIMPLE_DECK simple) # Reuse existing deck and start half way
list(APPEND RESTART_DECK dump) # Reuse existing deck and start half way
list(APPEND RESTART_BINARY restore)

foreach(test ${DEFAULT_ARG_TESTS})
    build_a_vpic(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.deck)
endforeach()

foreach(test ${DEFAULT_ARG_TESTS})
    vpic_add_test(NAME ${test} COMMAND ./${test})
endforeach()

# Parallel runs. simple slices the domain in y and reconnection_test in x.
vpic_add_test(NAME simple_np2 NP 2 COMMAND ./simple)
vpic_add_test(NAME simple_np4 NP 4 COMMAND ./simple)
vpic_add_test(NAME reconnection_test_np2 NP 2 COMMAND ./reconnection_test)

# Threaded and hybrid runs (skipped unless Kokkos has a threaded host backend)
vpic_add_test(NAME simple_t4 THREADS 4 COMMAND ./simple)
vpic_add_test(NAME simple_np2_t2 NP 2 THREADS 2 COMMAND ./simple)

# Test Restart (restore) functionality. The restart files are written per
# rank, so a restore must use the same number of ranks as the dump. Each rank
# count gets its own working directory so the checkpoints do not collide.
set(generate_restore "generate_${RESTART_BINARY}")
build_a_vpic(${generate_restore} ${CMAKE_CURRENT_SOURCE_DIR}/${RESTART_DECK}.deck)

# Run using the restore file
# TODO: caps?
set(perform_restore "perform_${RESTART_BINARY}")
build_a_vpic(${perform_restore} ${CMAKE_CURRENT_SOURCE_DIR}/${RESTART_DECK}.deck)

foreach(np 1 2)
    if(np EQUAL 1)
        set(suffix "")
    else()
        set(suffix "_np${np}")
    endif()
    set(restore_dir ${CMAKE_CURRENT_BINARY_DIR}/restore${suffix})
    file(MAKE_DIRECTORY ${restore_dir})
    set(checkpoint_file "${restore_dir}/checkpt_test.1")

    vpic_add_test(NAME ${generate_restore}${suffix} NP ${np}
        COMMAND $<TARGET_FILE:${generate_restore}>)
    vpic_add_test(NAME ${perform_restore}${suffix} NP ${np}
        COMMAND $<TARGET_FILE:${perform_restore}> --restore ${checkpoint_file})

    # Link the two tests
    set_tests_properties(${generate_restore}${suffix} ${perform_restore}${suffix}
        PROPERTIES WORKING_DIRECTORY ${restore_dir})
    set_tests_properties(${generate_restore}${suffix} PROPERTIES
        FIXTURES_SETUP restore_fixture${suffix})
    set_tests_properties(${perform_restore}${suffix} PROPERTIES
        FIXTURES_REQUIRED restore_fixture${suffix})
endforeach()

# See if we can build a deck with the 2 stage compile
# TODO: add a test that runs the generated deck
add_test(
    NAME perform_2stage_compile
    COMMAND ${CMAKE_BINARY_DIR}/bin/vpic ${CMAKE_CURRENT_SOURCE_DIR}/${SIMPLE_DECK}.deck
)

set(TWOSTAGE_OUT ${SIMPLE_DECK}.deck.${CMAKE_SYSTEM_NAME})
vpic_add_test(NAME run_2stage_compile COMMAND ./${TWOSTAGE_OUT})

set_tests_properties(perform_2stage_compile PROPERTIES FIXTURES_SETUP 2stage_fixture)
set_tests_properties(run_2stage_compile PROPERTIES FIXTURES_REQUIRED 2stage_fixture)
