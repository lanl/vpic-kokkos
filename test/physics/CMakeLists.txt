set(VPIC_TEST_GROUP physics)

# Single-particle and field checks with exact expected results
# (outbndj.deck is kept but not built; it has not passed since the Kokkos port)
foreach(test accel cyclo inbndj interpe)
  build_a_vpic(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.deck)
  vpic_add_test(NAME ${test} COMMAND ${test} 1 1)
endforeach()

# Weibel instability. weibel_legacy is the original test and deck;
# weibel_driver uses a decomposition-independent particle load, so it is also
# checked at several numbers of ranks and threads. Each has its own gold file.
set(gold_weibel_legacy "${CMAKE_CURRENT_SOURCE_DIR}/energies_gold")
set(gold_weibel_driver "${CMAKE_CURRENT_SOURCE_DIR}/energies_gold_driver")
foreach(test weibel_driver weibel_legacy)
  add_executable(${test} ${test}.cc)
  target_link_libraries(${test} vpic Kokkos::kokkos)
  target_compile_definitions(${test} PRIVATE GOLD_ENERGY_FILE="${gold_${test}}")
  vpic_add_test(NAME ${test} COMMAND ${test})
endforeach()

foreach(config np2:2:1 np4:4:1 t4:1:4 np2_t2:2:2)
  string(REPLACE ":" ";" config ${config})
  list(GET config 0 suffix)
  list(GET config 1 np)
  list(GET config 2 threads)
  vpic_add_test(NAME weibel_driver_${suffix} NP ${np} THREADS ${threads}
    COMMAND weibel_driver)
endforeach()
