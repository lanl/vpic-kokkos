#------------------------------------------------------------------------------#
# Helpers for registering VPIC tests that run under MPI and/or host threads
#------------------------------------------------------------------------------#

# Open MPI refuses to start more ranks than it detects cores unless told to
# oversubscribe, and binds each rank to a single core by default, which
# serializes host threads.  Detect it once so the flags below are only added
# when they are understood.
if(NOT DEFINED VPIC_MPI_IS_OPENMPI)
  execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version
    OUTPUT_VARIABLE mpiexec_version ERROR_VARIABLE mpiexec_version
    RESULT_VARIABLE mpiexec_result)
  set(is_openmpi OFF)
  if(mpiexec_result EQUAL 0 AND mpiexec_version MATCHES "Open MPI|OpenRTE")
    set(is_openmpi ON)
  endif()
  set(VPIC_MPI_IS_OPENMPI ${is_openmpi} CACHE BOOL
    "mpiexec is Open MPI (adds --oversubscribe/--bind-to none to tests)")
endif()

# Host-thread tests are only meaningful with a threaded Kokkos host backend
set(VPIC_HAS_HOST_THREADS OFF)
if(Kokkos_ENABLE_OPENMP OR Kokkos_ENABLE_THREADS)
  set(VPIC_HAS_HOST_THREADS ON)
endif()

# Every test finishes in seconds; a hung MPI run should fail, not stall CI
set(VPIC_TEST_TIMEOUT 120 CACHE STRING "Timeout in seconds for each VPIC test")

# vpic_add_test(NAME <name> [NP <ranks>] [THREADS <threads>]
#               COMMAND <exe> [args...] [LABELS <labels...>])
#
# Runs <exe> under ${MPIEXEC_EXECUTABLE} with NP ranks (default 1).  When
# THREADS is given, each rank runs that many Kokkos host threads; such tests
# are skipped if Kokkos has no threaded host backend.  The test reserves
# NP*THREADS ctest processors, so "ctest -j" does not oversubscribe the
# machine, is labeled "serial", "mpi" and/or "threads" for "ctest -L", and
# fails after VPIC_TEST_TIMEOUT seconds.
function(vpic_add_test)
  cmake_parse_arguments(VT "" "NAME;NP;THREADS" "COMMAND;LABELS" ${ARGN})
  if(NOT VT_NAME OR NOT VT_COMMAND)
    message(FATAL_ERROR "vpic_add_test: NAME and COMMAND are required")
  endif()
  if(NOT VT_NP)
    set(VT_NP 1)
  endif()

  set(labels ${VT_LABELS})
  set(launch ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${VT_NP})
  set(thread_args "")
  set(nthreads 1)

  if(VT_NP GREATER 1)
    list(APPEND labels mpi)
    if(VPIC_MPI_IS_OPENMPI)
      list(APPEND launch --oversubscribe)
    endif()
  endif()

  if(VT_THREADS)
    if(VT_THREADS GREATER 1 AND NOT VPIC_HAS_HOST_THREADS)
      message(STATUS "Skipping test ${VT_NAME}: Kokkos has no threaded host backend")
      return()
    endif()
    set(nthreads ${VT_THREADS})
    set(thread_args --kokkos-num-threads=${VT_THREADS})
    if(VT_THREADS GREATER 1)
      list(APPEND labels threads)
      if(VPIC_MPI_IS_OPENMPI)
        list(APPEND launch --bind-to none)
      endif()
    endif()
  endif()

  if(NOT labels)
    set(labels serial)
  endif()

  list(GET VT_COMMAND 0 exe)
  list(REMOVE_AT VT_COMMAND 0)

  add_test(NAME ${VT_NAME}
    COMMAND ${launch} ${MPIEXEC_PREFLAGS} ${exe} ${MPIEXEC_POSTFLAGS}
            ${VT_COMMAND} ${thread_args})

  math(EXPR nprocs "${VT_NP} * ${nthreads}")
  set_tests_properties(${VT_NAME} PROPERTIES
    TIMEOUT ${VPIC_TEST_TIMEOUT}
    PROCESSORS ${nprocs}
    LABELS "${labels}"
    ENVIRONMENT "OMP_NUM_THREADS=${nthreads};OMP_PROC_BIND=false")
endfunction()
