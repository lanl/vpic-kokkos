if(VPIC_USE_LEGACY_PARTICLE_ARRAY)
add_executable(rho_p ./rho_p.cc)
target_link_libraries(rho_p vpic Kokkos::kokkos)
add_test(NAME rho_p COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./rho_p)
add_executable(hydro_p ./hydro_p.cc)
target_link_libraries(hydro_p vpic Kokkos::kokkos)
add_test(NAME hydro_p COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./hydro_p)
endif(VPIC_USE_LEGACY_PARTICLE_ARRAY)

add_executable(field_comm field_comm.cpp)

target_link_libraries(field_comm vpic Kokkos::kokkos)

add_test(NAME field_comm_1x1x1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1  ${MPIEXEC_PREFLAGS} ./field_comm ${MPIEXEC_POSTFLAGS} 1 1 1 --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME field_comm_2x2x2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8  ${MPIEXEC_PREFLAGS} ./field_comm ${MPIEXEC_POSTFLAGS} 2 2 2 --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME field_comm_3x3x3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 27 --oversubscribe ${MPIEXEC_PREFLAGS} ./field_comm ${MPIEXEC_POSTFLAGS} 3 3 3 --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)
