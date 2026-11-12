set(VPIC_TEST_GROUP kernels)

add_executable(particle_sorting particle_sorting.cc)
target_link_libraries(particle_sorting vpic Kokkos::kokkos)
vpic_add_test(NAME particle_sorting COMMAND particle_sorting)

# Charge and hydro deposition, checked against a double-precision sum. Each
# also checks the legacy implementation when the legacy particle array is
# built. A threaded run (a different summation order) must pass too.
foreach(test rho_p hydro_p)
  add_executable(${test} ${test}.cc)
  target_link_libraries(${test} vpic Kokkos::kokkos)
  vpic_add_test(NAME ${test} COMMAND ${test})
  vpic_add_test(NAME ${test}_t4 THREADS 4 COMMAND ${test})
endforeach()
