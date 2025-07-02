
add_executable(halo_exchange halo_exchange_tests.cc)

target_link_libraries(halo_exchange vpic Kokkos::kokkos)

add_test(NAME halo_exchange_1x1x1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1  ${MPIEXEC_PREFLAGS} ./halo_exchange ${MPIEXEC_POSTFLAGS} 1 1 1 --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME halo_exchange_2x2x2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8  ${MPIEXEC_PREFLAGS} ./halo_exchange ${MPIEXEC_POSTFLAGS} 2 2 2 --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME halo_exchange_3x3x3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 27 --oversubscribe ${MPIEXEC_PREFLAGS} ./halo_exchange ${MPIEXEC_POSTFLAGS} 3 3 3 --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)

