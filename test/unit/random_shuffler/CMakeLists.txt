
add_executable(random_shuffle random_shuffle_tests.cc)

target_link_libraries(random_shuffle vpic Kokkos::kokkos)

add_test(NAME random_shuffle_1x1x1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1  ${MPIEXEC_PREFLAGS} ./random_shuffle ${MPIEXEC_POSTFLAGS} 1 1 1 --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)
add_test(NAME random_shuffle_2x2x2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1  ${MPIEXEC_PREFLAGS} ./random_shuffle ${MPIEXEC_POSTFLAGS} 2 2 2 --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)
#add_test(NAME random_shuffle_3x3x3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 27 --oversubscribe ${MPIEXEC_PREFLAGS} ./random_shuffle ${MPIEXEC_POSTFLAGS} 3 3 3 --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)

