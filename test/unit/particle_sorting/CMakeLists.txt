add_executable(particle_sorting sort_tests.cc)
target_link_libraries(particle_sorting vpic Kokkos::kokkos)
vpic_add_test(NAME particle_sorting COMMAND ./particle_sorting)
