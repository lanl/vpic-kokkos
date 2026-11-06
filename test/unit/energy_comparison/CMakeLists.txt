# Each test has its own gold file: weibel_legacy is the original test and
# deck, weibel_driver uses a decomposition-independent particle load
set(gold_weibel_legacy "${CMAKE_CURRENT_SOURCE_DIR}/energies_gold")
set(gold_weibel_driver "${CMAKE_CURRENT_SOURCE_DIR}/energies_gold_driver")

# Each run writes ./energies, so every test gets its own working directory
# (they would otherwise overwrite each other under ctest -j)
foreach (test weibel_driver weibel_legacy)
    add_executable(${test} ./${test}.cc)
    target_link_libraries(${test} vpic Kokkos::kokkos)
    target_compile_definitions(${test} PRIVATE GOLD_ENERGY_FILE="${gold_${test}}")
    set(run_dir ${CMAKE_CURRENT_BINARY_DIR}/${test}_run)
    file(MAKE_DIRECTORY ${run_dir})
    vpic_add_test(NAME ${test} COMMAND $<TARGET_FILE:${test}>)
    set_tests_properties(${test} PROPERTIES WORKING_DIRECTORY ${run_dir})
endforeach(test)

# weibel_driver's particle load does not depend on the decomposition, so it
# is checked against its gold energies at any number of ranks and threads.
# weibel_legacy only matches its gold file on one rank.
foreach(config np2:2:1 np4:4:1 t4:1:4 np2_t2:2:2)
    string(REPLACE ":" ";" config ${config})
    list(GET config 0 suffix)
    list(GET config 1 np)
    list(GET config 2 threads)
    set(run_dir ${CMAKE_CURRENT_BINARY_DIR}/weibel_driver_${suffix})
    file(MAKE_DIRECTORY ${run_dir})
    vpic_add_test(NAME weibel_driver_${suffix} NP ${np} THREADS ${threads}
        COMMAND $<TARGET_FILE:weibel_driver>)
    if(TEST weibel_driver_${suffix})
        set_tests_properties(weibel_driver_${suffix} PROPERTIES
            WORKING_DIRECTORY ${run_dir})
    endif()
endforeach()
