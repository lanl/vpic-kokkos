add_executable(rho_p ./rho_p.cc)
target_link_libraries(rho_p vpic Kokkos::kokkos)
add_test(NAME rho_p COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./rho_p)
