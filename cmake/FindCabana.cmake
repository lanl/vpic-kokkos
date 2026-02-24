# - Try to find Cabana
# Once done this will define
#  Cabana_FOUND - System has Cabana
#  Cabana_INCLUDE_DIR - The Cabana include directories
#  Cabana_LIBRARY - The libraries needed to use Cabana

set(Cabana_INSTALL_DIR "" CACHE STRING "Cabana install directory")
if(Cabana_INSTALL_DIR)
  message(STATUS "Cabana_INSTALL_DIR ${Cabana_INSTALL_DIR}")
endif()

find_path(Cabana_INCLUDE_DIR Cabana_Core.hpp PATHS "${Cabana_INSTALL_DIR}/include")

find_library(Cabana_LIBRARY cabanacore PATHS "${Cabana_INSTALL_DIR}/lib64")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set Cabana_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(
    Cabana
    DEFAULT_MSG
    Cabana_LIBRARY Cabana_INCLUDE_DIR
)

mark_as_advanced(Cabana_INCLUDE_DIR Cabana_LIBRARY )
