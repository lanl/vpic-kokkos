# One source, two configurations; see the header of grid_heating.cc
set(MPI_NUMPROC 1)

foreach(variant symmetric asymmetric)
    set(test grid_heating_${variant})

    add_executable(${test} ./grid_heating.cc)
    target_link_libraries(${test} vpic Kokkos::kokkos)
    if(variant STREQUAL "asymmetric")
        target_compile_definitions(${test} PRIVATE GRID_HEATING_ASYMMETRIC)
    endif()

    add_test(NAME ${test}
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPI_NUMPROC}
                     ${MPIEXEC_PREFLAGS} ./${test} --tpp 16)

    # OpenMPI binds a lone rank to one core, which would put all 16 threads
    # there
    set_tests_properties(${test} PROPERTIES
        ENVIRONMENT "OMPI_MCA_hwloc_base_binding_policy=none")
endforeach()
