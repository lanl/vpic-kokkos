#------------------------------------------------------------------------------#
# Helpers for registering VPIC tests that run under MPI and/or host threads
#------------------------------------------------------------------------------#

# Open MPI takes its slot count from the Slurm task count, or else from the
# physical cores (not hyperthreads), so "srun -N1 ctest" gives it one slot.  It
# refuses to start more ranks than slots unless told to oversubscribe, and it
# binds each rank to a single core by default, which serializes host threads.
# Detect it once so the flags below are only added when they are understood.
#
# Open MPI's mpi.h defines OPEN_MPI.  mpiexec --version is only a fallback:
# Ubuntu's Open MPI 5 packages leave out PRRTE's help files, so it prints a
# "Sorry! You were supposed to get help about: version" message instead.
if(NOT DEFINED VPIC_MPI_IS_OPENMPI)
  include(CheckCXXSymbolExists)
  set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
  set(CMAKE_REQUIRED_QUIET ON)
  check_cxx_symbol_exists(OPEN_MPI "mpi.h" VPIC_MPI_H_IS_OPENMPI)
  unset(CMAKE_REQUIRED_LIBRARIES)
  unset(CMAKE_REQUIRED_QUIET)
  execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version
    OUTPUT_VARIABLE mpiexec_version ERROR_VARIABLE mpiexec_version)
  set(is_openmpi OFF)
  if(VPIC_MPI_H_IS_OPENMPI OR mpiexec_version MATCHES "Open MPI|OpenRTE")
    set(is_openmpi ON)
  endif()
  message(STATUS "mpiexec is Open MPI: ${is_openmpi}")
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
#               [WORKING_DIRECTORY <dir>]
#               COMMAND <exe> [args...] [LABELS <labels...>])
#
# Runs <exe> (a target name or a path) under ${MPIEXEC_EXECUTABLE} with NP
# ranks (default 1).  When THREADS is given, each rank runs that many Kokkos
# host threads; such tests are skipped if Kokkos has no threaded host backend.
# The test reserves NP*THREADS ctest processors, so "ctest -j" does not
# oversubscribe the machine, and fails after VPIC_TEST_TIMEOUT seconds.
#
# Labels for "ctest -L": "serial", "mpi" and/or "threads", the group set by
# VPIC_TEST_GROUP in the calling directory (kernels, communication, physics,
# runs), and any given in LABELS.
#
# Each test runs in its own directory, <binary dir>/run/<name>, unless
# WORKING_DIRECTORY is given, so tests that write files with the same names
# do not overwrite each other under "ctest -j".
function(vpic_add_test)
  cmake_parse_arguments(VT "" "NAME;NP;THREADS;WORKING_DIRECTORY" "COMMAND;LABELS" ${ARGN})
  if(NOT VT_NAME OR NOT VT_COMMAND)
    message(FATAL_ERROR "vpic_add_test: NAME and COMMAND are required")
  endif()
  if(NOT VT_NP)
    set(VT_NP 1)
  endif()

  set(labels "")
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
  if(VPIC_TEST_GROUP)
    list(APPEND labels ${VPIC_TEST_GROUP})
  endif()
  list(APPEND labels ${VT_LABELS})

  if(NOT VT_WORKING_DIRECTORY)
    set(VT_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/run/${VT_NAME})
  endif()
  file(MAKE_DIRECTORY ${VT_WORKING_DIRECTORY})

  list(GET VT_COMMAND 0 exe)
  list(REMOVE_AT VT_COMMAND 0)
  if(TARGET ${exe})
    set(exe $<TARGET_FILE:${exe}>)
  endif()

  add_test(NAME ${VT_NAME}
    COMMAND ${launch} ${MPIEXEC_PREFLAGS} ${exe} ${MPIEXEC_POSTFLAGS}
            ${VT_COMMAND} ${thread_args}
    WORKING_DIRECTORY ${VT_WORKING_DIRECTORY})

  math(EXPR nprocs "${VT_NP} * ${nthreads}")
  set_tests_properties(${VT_NAME} PROPERTIES
    TIMEOUT ${VPIC_TEST_TIMEOUT}
    PROCESSORS ${nprocs}
    LABELS "${labels}"
    ENVIRONMENT "OMP_NUM_THREADS=${nthreads};OMP_PROC_BIND=false")
endfunction()
