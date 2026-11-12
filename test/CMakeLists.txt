# VPIC tests, grouped by what they check. Each test is labeled with its group,
# so for example "ctest -L kernels" runs the quick single-component checks.
#
#   kernels        single components (sorting, deposition) checked directly
#   communication  multi-rank field and particle exchange
#   physics        short simulations checked against known results
#   runs           simulations and the 2-stage build that must run to completion
#
# The rng and v4 tests live next to their sources in src/util and are added by
# the top-level CMakeLists.txt.

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(kernels)
add_subdirectory(communication)
add_subdirectory(physics)
add_subdirectory(runs)
