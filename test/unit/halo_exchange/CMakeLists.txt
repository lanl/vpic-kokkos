
add_executable(halo_exchange halo_exchange_tests.cc)

target_link_libraries(halo_exchange vpic Kokkos::kokkos)

add_test(NAME halo_exchange COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 27 ./halo_exchange --kokkos-num-threads=1 --kokkos-map-device-id-by=mpi_rank)

