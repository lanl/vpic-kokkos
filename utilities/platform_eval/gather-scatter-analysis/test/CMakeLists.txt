add_executable(particle_sorting sort_tests.cpp)
target_link_libraries(particle_sorting Kokkos::kokkos)
add_test(NAME particle_sorting COMMAND particle_sorting)

