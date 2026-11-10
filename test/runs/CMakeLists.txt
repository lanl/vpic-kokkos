set(VPIC_TEST_GROUP runs)

# These check only that a run completes, not its results.
# simple slices the domain in y, reconnection_test in x.
foreach(test simple dump reconnection_test)
  build_a_vpic(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.deck)
  vpic_add_test(NAME ${test} COMMAND ${test})
endforeach()

vpic_add_test(NAME simple_np2 NP 2 COMMAND simple)
vpic_add_test(NAME simple_np4 NP 4 COMMAND simple)
vpic_add_test(NAME reconnection_test_np2 NP 2 COMMAND reconnection_test)

# Threaded and hybrid runs (skipped unless Kokkos has a threaded host backend)
vpic_add_test(NAME simple_t4 THREADS 4 COMMAND simple)
vpic_add_test(NAME simple_np2_t2 NP 2 THREADS 2 COMMAND simple)

# Restart: generate_restore dumps a checkpoint, perform_restore restarts from
# it. Restart files are written per rank, so both must use the same number of
# ranks, and each pair shares a working directory.
build_a_vpic(generate_restore ${CMAKE_CURRENT_SOURCE_DIR}/dump.deck)
build_a_vpic(perform_restore ${CMAKE_CURRENT_SOURCE_DIR}/dump.deck)
foreach(np 1 2)
  if(np EQUAL 1)
    set(suffix "")
  else()
    set(suffix "_np${np}")
  endif()
  set(restore_dir ${CMAKE_CURRENT_BINARY_DIR}/run/restore${suffix})
  vpic_add_test(NAME generate_restore${suffix} NP ${np}
    WORKING_DIRECTORY ${restore_dir}
    COMMAND generate_restore)
  vpic_add_test(NAME perform_restore${suffix} NP ${np}
    WORKING_DIRECTORY ${restore_dir}
    COMMAND perform_restore --restore ${restore_dir}/checkpt_test.1)
  set_tests_properties(generate_restore${suffix} PROPERTIES
    FIXTURES_SETUP restore_fixture${suffix})
  set_tests_properties(perform_restore${suffix} PROPERTIES
    FIXTURES_REQUIRED restore_fixture${suffix})
endforeach()

# Build a deck with the installed-style 2-stage compile script, then run it.
# This is a compile, not a run, so it gets more time than the other tests
# (~50 s with nvcc).
set(twostage_dir ${CMAKE_CURRENT_BINARY_DIR}/run/2stage_compile)
file(MAKE_DIRECTORY ${twostage_dir})
add_test(NAME perform_2stage_compile
  COMMAND ${CMAKE_BINARY_DIR}/bin/vpic ${CMAKE_CURRENT_SOURCE_DIR}/simple.deck
  WORKING_DIRECTORY ${twostage_dir})
set_tests_properties(perform_2stage_compile PROPERTIES
  LABELS "serial;runs" TIMEOUT 600
  FIXTURES_SETUP 2stage_fixture)
vpic_add_test(NAME run_2stage_compile WORKING_DIRECTORY ${twostage_dir}
  COMMAND ${twostage_dir}/simple.deck.${CMAKE_SYSTEM_NAME})
set_tests_properties(run_2stage_compile PROPERTIES
  FIXTURES_REQUIRED 2stage_fixture)
