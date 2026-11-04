LIST(APPEND TESTS weibel_driver)

list(APPEND gold_file "${CMAKE_CURRENT_SOURCE_DIR}/energies_gold")
add_definitions(-DGOLD_ENERGY_FILE="${gold_file}")

foreach (test ${TESTS})
    add_executable(${test} ./${test}.cc)
    target_link_libraries(${test} vpic Kokkos::kokkos)
    vpic_add_test(NAME ${test} COMMAND ./${test})
endforeach(test)

# The particle load does not depend on the decomposition, so the same gold
# energies must be reproduced at any number of ranks and threads. Each run
# writes ./energies, so give each its own working directory.
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
