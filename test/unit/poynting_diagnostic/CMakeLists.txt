add_executable(poynting_diagnostic ./poynting_diagnostic.cc)
target_link_libraries(poynting_diagnostic vpic Kokkos::kokkos)
add_test(NAME poynting_diagnostic COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./poynting_diagnostic)
set_tests_properties(poynting_diagnostic PROPERTIES ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
