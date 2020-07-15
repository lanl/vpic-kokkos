#------------------------------------------------------------------------------#
# Copyright (c) 2014 Los Alamos National Security, LLC
# All rights reserved.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.9)
cmake_policy(SET CMP0074 NEW)

#~----------------------------------------------------------------------------~#
# VPIC project configuration
#~----------------------------------------------------------------------------~#

project(vpic CXX)

#------------------------------------------------------------------------------#
# If a C++11 compiler is available, then set the appropriate flags
#------------------------------------------------------------------------------#

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#------------------------------------------------------------------------------#
# MPI detection
#------------------------------------------------------------------------------#

# TODO: it seems like Cmake version 3.12 (and likely others) have a bug/change
# where it has a hard time detecting MPI. Cmake version. 3.9.0 works great
# though..
find_package(MPI REQUIRED)
list(APPEND MPI_CXX_COMPILE_FLAGS -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX )
include_directories(${MPI_C_INCLUDE_PATH})
include_directories(${MPI_CXX_INCLUDE_PATH})

#------------------------------------------------------------------------------#
# Add build options
#------------------------------------------------------------------------------#
option(ENABLE_INTEGRATED_TESTS "enable integrated tests" OFF)

option(ENABLE_UNIT_TESTS "enable unit tests" OFF)

option(USE_V4_ALTIVEC "Enable V4 Altivec" OFF)

option(USE_V4_PORTABLE "Enable V4 Portable" OFF)

option(USE_V4_SSE "Enable V4 SSE" OFF)

option(ENABLE_OPENSSL "Enable OpenSSL support for checksums" OFF)

option(ENABLE_KOKKOS_OPENMP "Enable Kokkos OpenMP" ON)

option(ENABLE_KOKKOS_CUDA "Enable Kokkos CUDA" OFF)

option(BUILD_INTERNAL_KOKKOS "Have VPIC build it's own Kokkos rather than look in $CMAKE_PREFIX_PATH" OFF)

option(VPIC_DUMP_ENERGIES "If the code should dump energies diagnostic" OFF)

add_definitions(-DUSE_KOKKOS)
set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -DUSE_KOKKOS") # Set it here for ./deck/ files

# Set a build option to compare zigzag current accumulation to the old method.
option(ACCUMULATE_J_ZIGZAG "enable zigzag method on accumulate_j in move_p_kokkos" OFF)
if(ACCUMULATE_J_ZIGZAG)
    add_definitions(-DACCUMULATE_J_ZIGZAG)
    set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -DACCUMULATE_J_ZIGZAG")
endif()

#------------------------------------------------------------------------------#
# Create include and link aggregates
#
# NOTE: These must be set before creating the compile scripts below.
#------------------------------------------------------------------------------#

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(VPIC_CPPFLAGS)
if(MPI_CPPFLAGS)
  string(REPLACE ";" " " string_cppflags "${MPI_CPPFLAGS}")
  set(VPIC_CPPFLAGS "${string_cppflags}")
endif(MPI_CPPFLAGS)

# Append extra implicit includes
string(REPLACE ";" " -I" string_cppflags "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}")
set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -I${string_cppflags}")
message(${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
message(${VPIC_CPPFLAGS})


string(REPLACE ";" " -I" string_includes "${MPI_C_INCLUDE_PATH}")
if(NOT ${string_includes} STREQUAL "")
  set(VPIC_CXX_FLAGS "-I${string_includes} ${MPI_C_LINK_FLAGS}")
endif(NOT ${string_includes} STREQUAL "")
set(VPIC_CXX_FLAGS "${CMAKE_CXX11_STANDARD_COMPILE_OPTION} ${VPIC_CXX_FLAGS}")

# Add Debug flags to VPIC_CXX_FLAGS
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(VPIC_CXX_FLAGS "${VPIC_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

# Add RelWithDebInfo flags to VPIC_CXX_FLAGS
if("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  set(VPIC_CXX_FLAGS "${VPIC_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")

string(REPLACE ";" " " string_libraries "${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES}")
set(VPIC_CXX_LIBRARIES "${string_libraries}")

#------------------------------------------------------------------------------#
# OpenMP
#------------------------------------------------------------------------------#

find_package(OpenMP)

# FIXME: Sometimes OpenMP is not found fully, when it really should be. Is this
# a CMAKE issue?
if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(VPIC_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${VPIC_CXX_FLAGS}")
  #set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
else()
  message("OpenMP not found...")
endif()

# Allow files to be references from the root, (eg #include "src/vpic/vpic.h")
include_directories(${PROJECT_SOURCE_DIR})

#------------------------------------------------------------------------------#
# Kokkos
#------------------------------------------------------------------------------#
set(Kokkos_ENABLE_DEBUG OFF)

option(VPIC_KOKKOS_DEBUG "Enable Debug In Kokkos" OFF)

if (VPIC_KOKKOS_DEBUG)
    set(Kokkos_ENABLE_DEBUG ON)
endif(VPIC_KOKKOS_DEBUG)

set(Kokkos_ENABLE_EXPLICIT_INSTANTIATION OFF)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Default allow serial execution")

if(ENABLE_KOKKOS_OPENMP)
  if (OPENMP_FOUND)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Use OpenMP if found")
  endif()
else()
    set(Kokkos_ENABLE_OPENMP OFF)
endif()

if(ENABLE_KOKKOS_CUDA)
  find_package(CUDA)
  if (CUDA_FOUND)
    add_definitions(-DENABLE_KOKKOS_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Use Cuda with Kokkos")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable cuda lambda support")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")

    # turn off -std=gnu++ to avoid error: identifier "__float128" is undefined issue
    # see https://github.com/UoB-HPC/BabelStream/issues/16
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Don't use intel intrinsics if we are using CUDA
    option(USE_V4_ALTIVEC "Enable V4 Altivec" OFF)
    option(USE_V4_PORTABLE "Enable V4 Portable" OFF)
    option(USE_V4_SSE "Enable V4 SSE" OFF)

  endif()
endif()

# Disable platform specific options
if (APPLE)
  set(Kokkos_ENABLE_LIBRT OFF CACHE BOOL "librt is not available on macOS")
endif()



# TODO: This is only need for non-linux platforms
set(Kokkos_ENABLE_LIBDL OFF)
set(Kokkos_ENABLE_PROFILING OFF) # NO libDL => no profiling


if (BUILD_INTERNAL_KOKKOS)
  message("Building our own kokkos..")
  add_subdirectory(kokkos)
  set(Kokkos_ENABLE_LIBDL OFF)
  set(Kokkos_ENABLE_PROFILING OFF) # NO libDL => no profiling
  #include_directories(${Kokkos_INCLUDE_DIRS_RET})
  #set(KOKKOS_INCLUDE_DIRS ${Kokkos_INCLUDE_DIRS_RET})
else()
  # CONSIDER ADDING --expt-extended-lambda
#  find_package(KOKKOS REQUIRED)
  find_package(Kokkos REQUIRED)

  #get_target_property(KOKKOS_INCLUDE_DIRS Kokkos::kokkos INTERFACE_INCLUDE_DIRECTORIES)
  #get_target_property(KOKKOS_LIB_DIR Kokkos::kokkos INTERFACE_LINK_LIBRARIES)
  #message(${KOKKOS_INCLUDE_DIRS})
  #link_directories(${KOKKOS_INCLUDE_DIRS}/../lib) # TODO: bad idea..
endif(BUILD_INTERNAL_KOKKOS)


get_target_property(KOKKOS_INCLUDE_DIRS Kokkos::kokkoscore INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(KOKKOS_INCLUDE_CONTAINERS Kokkos::kokkoscontainers INTERFACE_INCLUDE_DIRECTORIES)

# TODO: this gives the external include path
#get_target_property(KOKKOS_LIB_DIRS Kokkos::kokkos INTERFACE_LINK_DEPENDS)

#get_target_property(KOKKOS_LIB_DIRS Kokkos::kokkos INTERFACE_LINK_DEPENDS)
#get_target_property(KOKKOS_LIB_CONTAINERS Kokkos::kokkoscontainers INTERFACE_LINK_DEPENDS)

set(KOKKOS_CORE_INCLUDES_FILE ${PROJECT_BINARY_DIR}/kokkos_core_includes)
set(KOKKOS_CONTAINER_INCLUDES_FILE ${PROJECT_BINARY_DIR}/kokkos_container_includes)
#set(KOKKOS_CORE_LIB_FILE ${PROJECT_BINARY_DIR}/kokkos_core_lib)
#set(KOKKOS_CONTAINER_LIB_FILE ${PROJECT_BINARY_DIR}/kokkos_container_lib)

# For an "external" build these will be the same, but for an internal we need to gather both
file(GENERATE OUTPUT ${KOKKOS_CORE_INCLUDES_FILE} CONTENT "${KOKKOS_INCLUDE_DIRS}")
file(GENERATE OUTPUT ${KOKKOS_CONTAINER_INCLUDES_FILE} CONTENT "${KOKKOS_INCLUDE_CONTAINERS}")

#file(GENERATE OUTPUT ${KOKKOS_CORE_LIB_FILE} CONTENT "${KOKKOS_LIB_DIRS}")
#file(GENERATE OUTPUT ${KOKKOS_CONTAINER_LIB_FILE} CONTENT "${KOKKOS_LIB_CONTAINERS}")

#include_directories(${KOKKOS_INCLUDE_DIRS})
#string(REPLACE ";" " " KOKKOS_INCLUDE_DIRS "${KOKKOS_INCLUDE_DIRS}")
#message(dirs: ${KOKKOS_INCLUDE_DIRS})

#FOREACH (P ${KOKKOS_INCLUDE_DIRS})
    #message(${P})
    #include_directories(${P})
    #link_directories(${P})
    #ENDFOREACH(P)

# TODO: bad idea.. at best, this is just going to add it to the last one in the
# array
#link_directories(${KOKKOS_INCLUDE_DIRS}/../lib)
#link_directories(${KOKKOS_INCLUDE_DIRS}/lib)

# Libs
string(REPLACE ";" " " string_libraries "${KOKKOS_LIBS} ${KOKKOS_LDFLAGS}")
set(VPIC_CXX_LIBRARIES "${VPIC_CXX_LIBRARIES} ${string_libraries}")

# Includes (TODO: this is definitely abuse of the CXX variable..)
#string(REPLACE ";" " -I" string_libraries "-I${KOKKOS_INCLUDE_DIRS}")
#set(VPIC_CXX_LIBRARIES "${VPIC_CXX_LIBRARIES} ${string_libraries}")

#string(REPLACE ";" " -L" string_libraries "-L${KOKKOS_INCLUDE_DIRS}")
#set(VPIC_CXX_LIBRARIES "${VPIC_CXX_LIBRARIES} ${string_libraries}")


#------------------------------------------------------------------------------#
# OpenSSL
#------------------------------------------------------------------------------#

if(ENABLE_OPENSSL)
  find_package(OpenSSL REQUIRED)

  include_directories(${OPENSSL_INCLUDE_DIR})
  string(REPLACE ";" " " string_libraries "${OPENSSL_LIBRARIES}")
  set(VPIC_CXX_LIBRARIES "${VPIC_CXX_LIBRARIES} ${string_libraries}")
endif(ENABLE_OPENSSL)

find_package(Threads REQUIRED)

#------------------------------------------------------------------------------#
# Act on build options set in project.cmake
#------------------------------------------------------------------------------#

if (VPIC_DUMP_ENERGIES)
    add_definitions(-DDUMP_ENERGIES)
endif(VPIC_DUMP_ENERGIES)

set(USE_V4)
if(USE_V4_ALTIVEC)
  add_definitions(-DUSE_V4_ALTIVEC)
  set(USE_V4 True)
endif(USE_V4_ALTIVEC)

if(USE_V4_PORTABLE)
  add_definitions(-DUSE_V4_PORTABLE)
  set(USE_V4 True)
endif(USE_V4_PORTABLE)

if(USE_V4_SSE)
  add_definitions(-DUSE_V4_SSE)
  set(USE_V4 True)
endif(USE_V4_SSE)

if(ENABLE_OPENSSL)
  add_definitions(-DENABLE_OPENSSL)
endif(ENABLE_OPENSSL)

#------------------------------------------------------------------------------#
# Output log variables
#------------------------------------------------------------------------------#
FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    EXECUTE_PROCESS(
        COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE   VPIC_GIT_COMMIT_HASH
        )
ELSE()
    SET(VPIC_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
string(TIMESTAMP _build_time_stamp)

MESSAGE(STATUS "VPIC Revision = '${VPIC_GIT_COMMIT_HASH}'")
#add_definitions("-DVPIC_GIT_HASH=${VPIC_GIT_COMMIT_HASH}")

configure_file (
    "${PROJECT_SOURCE_DIR}/cmake/git_version.h.in"
    "${PROJECT_BINARY_DIR}/include/git_version.h"
)

include_directories(${PROJECT_BINARY_DIR}/include) # for generated git_version.h

#------------------------------------------------------------------------------#
# Handle vpic compile script last
#------------------------------------------------------------------------------#

# Allow files to be references from the root, (eg #include "src/vpic/vpic.h")
#include_directories(${PROJECT_SOURCE_DIR})

if(BUILD_SHARED_LIBS)
    set(VPIC_CXX_FLAGS "-rdynamic ${VPIC_CXX_FLAGS}")
endif(BUILD_SHARED_LIBS)

if(ENABLE_COVERAGE_BUILD)
    set(VPIC_CXX_FLAGS "${VPIC_CXX_FLAGS} --coverage")
endif(ENABLE_COVERAGE_BUILD)

# process Makefile.run.in to get a simple Makefile.run for a run. Points to
# local built exe wrapper, and has example deck/platform.
configure_file(
    ${CMAKE_SOURCE_DIR}/sample/Makefile.run.in
    ${CMAKE_BINARY_DIR}/bin/Makefile.run
)

# install script
configure_file(
    ${CMAKE_SOURCE_DIR}/bin/vpic.in
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vpic-install
    @ONLY
)

install(FILES ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vpic-install
  DESTINATION bin
  RENAME vpic
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    )

install(FILES ${CMAKE_SOURCE_DIR}/deck/main.cc
  DESTINATION share/vpic)
install(FILES ${CMAKE_SOURCE_DIR}/deck/wrapper.cc
  DESTINATION share/vpic)

# local script
configure_file(
    ${CMAKE_SOURCE_DIR}/bin/vpic-local.in
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vpic
    @ONLY
)

file(COPY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vpic
  DESTINATION ${CMAKE_BINARY_DIR}/bin
  FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

#------------------------------------------------------------------------------#
# Add library target
#------------------------------------------------------------------------------#

file(GLOB_RECURSE VPIC_SRC src/*.c src/*.cc)
file(GLOB_RECURSE VPIC_NOT_SRC src/util/v4/test/v4.cc src/util/rng/test/rng.cc)
list(REMOVE_ITEM VPIC_SRC ${VPIC_NOT_SRC})
option(NO_LIBVPIC "Don't build a libvpic, but all in one" OFF)
if(NO_LIBVPIC)
  set(VPIC_EXPOSE "INTERFACE")
  add_library(vpic INTERFACE)
else()
  set(VPIC_EXPOSE "PRIVATE")
  # TODO: this does a bad job of detecting when changes are need in ./bin/vpic
  # for example if you delete ./bin/vpic it won't get rebuilt
  add_library(vpic ${VPIC_SRC})
  set(VPIC_SRC)
  install(TARGETS vpic LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endif()
target_include_directories(vpic INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(vpic ${VPIC_EXPOSE} ${MPI_CXX_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OPENSSL_LIBRARIES} ${CMAKE_DL_LIBS} Kokkos::kokkos)
target_compile_options(vpic ${VPIC_EXPOSE} ${MPI_C_COMPILE_FLAGS})

macro(build_a_vpic name deck)
  if(NOT EXISTS ${deck})
    message(FATAL_ERROR "Could not find deck '${deck}'")
  endif()
  add_executable(${name} ${CMAKE_SOURCE_DIR}/deck/main.cc ${CMAKE_SOURCE_DIR}/deck/wrapper.cc ${VPIC_SRC})
  target_link_libraries(${name} vpic Kokkos::kokkos)
  target_compile_definitions(${name} PRIVATE INPUT_DECK=${deck})
endmacro(build_a_vpic)

if(USER_DECKS)
  foreach(DECK ${USER_DECK})
    get_filename_component(NAME "${DECK}" NAME_WE)
    get_filename_component(DECKDIR "${DECK}" DIRECTORY)
    if(NOT DECKDIR)
      set(DECK "${CMAKE_CURRENT_BINARY_DIR}/${DECK}")
    endif()
    build_a_vpic(${NAME} ${DECK})
  endforeach()
endif()

#------------------------------------------------------------------------------#
# Add VPIC integrated test mechanism
#------------------------------------------------------------------------------#

if(ENABLE_INTEGRATED_TESTS)
  enable_testing()
  add_subdirectory(test/integrated)

endif(ENABLE_INTEGRATED_TESTS)

if(ENABLE_UNIT_TESTS)

  enable_testing()

  # NOTE: This can also be done by a subdirectory (useful when we have multiple tests)
  # Add includes for Catch2
  set(TEST_DIR "./test/include")
  include_directories(${TEST_DIR})

  # Add tests individually
  if (USE_V4)
    add_executable(v4 src/util/v4/test/v4.cc)
    add_test(NAME v4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./v4)
  endif(USE_V4)

  # RNG tests
  add_executable(rng src/util/rng/test/rng.cc)
  target_link_libraries(rng vpic Kokkos::kokkos)
  get_target_property(OUT rng LINK_LIBRARIES)

  add_test(NAME rng COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./rng)

  add_subdirectory(test/unit)

endif(ENABLE_UNIT_TESTS)
#~---------------------------------------------------------------------------~-#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#~---------------------------------------------------------------------------~-#
