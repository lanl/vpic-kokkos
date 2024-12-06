#ifndef VPIC_DUMP_H_
#define VPIC_DUMP_H_

#include <array>

// TODO: should this be an enum?
namespace dump_type {
  const int grid_dump = 0;
  const int field_dump = 1;
  const int hydro_dump = 2;
  const int particle_dump = 3;
  const int restart_dump = 4;
  const int history_dump = 5;
} // namespace

#endif
