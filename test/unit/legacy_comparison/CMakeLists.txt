if(VPIC_USE_LEGACY_PARTICLE_ARRAY)
add_executable(rho_p ./rho_p.cc)
target_link_libraries(rho_p vpic Kokkos::kokkos)
vpic_add_test(NAME rho_p COMMAND ./rho_p)
add_executable(hydro_p ./hydro_p.cc)
target_link_libraries(hydro_p vpic Kokkos::kokkos)
vpic_add_test(NAME hydro_p COMMAND ./hydro_p)
endif(VPIC_USE_LEGACY_PARTICLE_ARRAY)

set(COMM_ARGS --kokkos-map-device-id-by=mpi_rank)

# Exchange fields across every topology <tx>x<ty>x<tz>, run on tx*ty*tz ranks
add_executable(field_comm field_comm.cpp)
target_link_libraries(field_comm vpic Kokkos::kokkos)
foreach(topology 1x1x1 2x1x1 1x2x1 1x1x2 2x2x2 3x3x3)
  string(REPLACE "x" ";" dims ${topology})
  list(GET dims 0 tx)
  list(GET dims 1 ty)
  list(GET dims 2 tz)
  math(EXPR np "${tx} * ${ty} * ${tz}")
  vpic_add_test(NAME field_comm_${topology} NP ${np} THREADS 1
    COMMAND ./field_comm ${tx} ${ty} ${tz} ${COMM_ARGS})
endforeach()

add_executable(field_local_bc field_local_bc.cpp)
target_link_libraries(field_local_bc vpic Kokkos::kokkos)
vpic_add_test(NAME field_local_bc NP 8 THREADS 1
  COMMAND ./field_local_bc 2 2 2 ${COMM_ARGS})
