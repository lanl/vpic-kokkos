add_executable(particle_sorting sort_tests.cc)
target_link_libraries(particle_sorting vpic Kokkos::kokkos)
add_test(NAME particle_sorting COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} particle_sorting ${MPIEXEC_POSTFLAGS})
