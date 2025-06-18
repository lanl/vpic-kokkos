#ifndef VPIC_DUMP_STRATEGY_H_
#define VPIC_DUMP_STRATEGY_H_

// C++ headers
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// VPIC headers
#include "../util/io/FileIO.h"
#include "../util/util_base.h"
#include "../util/io/FileUtils.h"
#include "../field_advance/field_advance.h"
#include "../fluid_advance/fluid_advance.h"
#include "../sf_interface/sf_interface.h"
#include "../species_advance/species_advance.h"
#include "dump.h"
#include "dumpmacros.h"

#ifdef VPIC_ENABLE_HDF5
#include "hdf5.h"             // from the lib
#include "hdf5_header_info.h" // from vpic
#endif

// Forward declarations
class vpic_simulation;
struct DumpParameters;

typedef enum DumpStrategyID
{
  DUMP_STRATEGY_BINARY = 0,
  DUMP_STRATEGY_HDF5 = 1,
} DumpStrategyID;

//------------------------------------------------------------------------------
// class Dump_Strategy
// functions to dump fields, hydro, and particle

class Dump_Strategy
{
public:
  int rank, nproc;

  Dump_Strategy(int _rank, int _nproc) : rank(_rank), nproc(_nproc)
  {}

  virtual ~Dump_Strategy(){};

  virtual void dump_fields(
      const char *fbase,
      int step,
      grid_t *grid,
      field_array_t *field_array,
      int ftag) = 0;
  virtual void dump_hydro(
      const char *fbase,
      int step,
      species_t *sp,
      grid_t *grid,
      hydro_array_t *hydro_array,
      interpolator_array_t *interpolator_array,
      int ftag) = 0;
  virtual void dump_particles(
      const char *fbase,
      int step,
      species_t *sp,
      grid_t *grid,
      interpolator_array_t *interpolator_array,
      int ftag) = 0;
  virtual void dump_fluids(
      const char *fbase,
      int step,
      fluid_species_t *fsp,
      grid_t *grid,
      int ftag) = 0;
  virtual void field_dump(
      DumpParameters& dumpParams,
      int step,
      grid_t *grid,
      field_array_t *field_array) = 0;
  virtual void hydro_dump(
      DumpParameters& dumpParams,
      int step,
      species_t *sp,
      grid_t *grid,
      hydro_array_t *hydro_array,
      interpolator_array_t *interpolator_array) = 0;
  virtual void fluid_dump(
      DumpParameters& dumpParams,
      int step,
      fluid_species_t *fsp,
      grid_t *grid) = 0;
};

// functions to create and delete the dump strategy
Dump_Strategy *new_dump_strategy(DumpStrategyID dump_strategy_id, vpic_simulation *vpic_simu);
void delete_dump_strategy(Dump_Strategy *ds);

//------------------------------------------------------------------------------
// class BinaryDump
// functions to dump fields, hydro, and particle in binary format

class BinaryDump : public Dump_Strategy
{
public:
  using Dump_Strategy::Dump_Strategy; // inherit constructor

  // TODO: now we pass rank and step, ftag has odd semantics
  void dump_fields(
      const char *fbase,
      int step,
      grid_t *grid,
      field_array_t *field_array,
      int ftag);
  void dump_hydro(
      const char *fbase,
      int step,
      species_t *sp,
      grid_t *grid,
      hydro_array_t *hydro_array,
      interpolator_array_t *interpolator_array,
      int ftag);
  void dump_particles(
      const char *fbase,
      int step,
      species_t *sp,
      grid_t *grid,
      interpolator_array_t *interpolator_array,
      int ftag);
  void dump_fluids(
      const char *fbase,
      int step,
      fluid_species_t *fsp,
      grid_t *grid,
      int ftag);
  void field_dump(
      DumpParameters& dumpParams,
      int step,
      grid_t *grid,
      field_array_t *field_array);
  void hydro_dump(
      DumpParameters& dumpParams,
      int step,
      species_t *sp,
      grid_t *grid,
      hydro_array_t *hydro_array,
      interpolator_array_t *interpolator_array);
  void fluid_dump(
      DumpParameters& dumpParams,
      int step,
      fluid_species_t *fsp,
      grid_t *grid);
};

#ifdef VPIC_ENABLE_HDF5

struct field_dump_flag_t
{
  std::vector<std::string> flag_keys = {
    "ex", "ey", "ez", "div_e_err",
    "cbx", "cby", "cbz", "pe",
    "tcax", "tcay", "tcaz", "rhob",
    "jfx", "jfy", "jfz", "rhof",
    "jfxold", "jfyold", "jfzold", "rhofold",
    "cbx0", "cby0", "cbz0", "te0",
    "tx", "ty", "tz", "te",
    "ox", "oy", "oz", "oe","div_b_err"
  };

  std::unordered_map<std::string, bool> flags = {
    {"ex", true}, {"ey", true}, {"ez", true}, {"div_e_err", true},
    {"cbx", true}, {"cby", true}, {"cbz", true}, {"pe", true},
    {"tcax", true}, {"tcay", true}, {"tcaz", true}, {"rhob", true},
    {"jfx", true}, {"jfy", true}, {"jfz", true}, {"rhof", true},
    {"jfxold", true}, {"jfyold", true}, {"jfzold", true}, {"rhofold", true},
    {"cbx0", true}, {"cby0", true}, {"cbz0", true}, {"te0", true},
    {"tx", true}, {"ty", true}, {"tz", true}, {"te", true},
    {"ox", true}, {"oy", true}, {"oz", true}, {"oe", true}, {"div_b_err", true}
  };

  void disableE() {
    flags["ex"] = false, flags["ey"] = false, flags["ez"] = false;
  }

  void disableCB() {
    flags["cbx"] = false, flags["cby"] = false, flags["cbz"] = false;
  }

  void disableCB0() {
    flags["cbx0"] = false, flags["cby0"] = false, flags["cbz0"] = false;
  }

  void disableTCA() {
    flags["tcax"] = false, flags["tcay"] = false, flags["tcaz"] = false;
  }

  void disableJF() {
    flags["jfx"] = false, flags["jfy"] = false, flags["jfz"] = false;
  }

  void disableJFOLD() {
    flags["jfxold"] = false, flags["jfyold"] = false, flags["jfzold"] = false;
  }

  void disableT() {
    flags["tx"] = false, flags["ty"] = false, flags["tz"] = false;
  }

  void disableO() {
    flags["ox"] = false, flags["oy"] = false, flags["oz"] = false;
  }

  void disableALL() {
    for (auto& [key,_] : flags)
      flags[key] = false;
  }

  void enableE() {
    flags["ex"] = true, flags["ey"] = true, flags["ez"] = true;
  }

  void enableCB() {
    flags["cbx"] = true, flags["cby"] = true, flags["cbz"] = true;
  }

  void enableCB0() {
    flags["cbx0"] = true, flags["cby0"] = true, flags["cbz0"] = true;
  }

  void enableTCA() {
    flags["tcax"] = true, flags["tcay"] = true, flags["tcaz"] = true;
  }

  void enableJF() {
    flags["jfx"] = true, flags["jfy"] = true, flags["jfz"] = true;
  }

  void enableJFOLD() {
    flags["jfxold"] = true, flags["jfyold"] = true, flags["jfzold"] = true;
  }

  void enableT() {
    flags["tx"] = true, flags["ty"] = true, flags["tz"] = true;
  }

  void enableO() {
    flags["ox"] = true, flags["oy"] = true, flags["oz"] = true;
  }

  void enableALL() {
    for (auto& [key,_] : flags)
      flags[key] = true;
  }

  bool enabledE() {
    return flags["ex"] && flags["ey"] && flags["ez"];
  }

  bool enabledCB() {
    return flags["cbx"] && flags["cby"] && flags["cbz"];
  }

  bool enabledCB0() {
    return flags["cbx0"] && flags["cby0"] && flags["cbz0"];
  }

  bool enabledTCA() {
    return flags["tcax"] && flags["tcay"] && flags["tcaz"];
  }

  bool enabledJF() {
    return flags["jfx"] && flags["jfy"] && flags["jfz"];
  }

  bool enabledJFOLD() {
    return flags["jfxold"] && flags["jfyold"] && flags["jfzold"];
  }

  bool enabledT() {
    return flags["tx"] && flags["ty"] && flags["tz"];
  }

  bool enabledO() {
    return flags["ox"] && flags["oy"] && flags["oz"];
  }
};

struct hydro_dump_flag_t
{
  std::vector<std::string> flag_keys = {
    "jx", "jy", "jz", "rho",
    "px", "py", "pz", "rho_m",
    "txx", "tyy", "tzz",
    "tyz", "tzx", "txy",
#ifdef VARIABLE_CHARGE
    "qmin", "qmax"
#else
    "pad"
#endif
  };

  std::unordered_map<std::string, bool> flags = {
    {"jx", true}, {"jy", true}, {"jz", true}, {"rho", true},
    {"px", true}, {"py", true}, {"pz", true}, {"rho_m", true},
    {"txx", true}, {"tyy", true}, {"tzz", true},
    {"tyz", true}, {"tzx", true}, {"txy", true},
#ifdef VARIABLE_CHARGE
    {"qmin", true}, {"qmax", true},
#else
    {"pad", true}
#endif
  };

  void disableJ() {
    flags["jx"] = false, flags["jy"] = false, flags["jz"] = false;
  }

  void disableP() {
    flags["px"] = false, flags["py"] = false, flags["pz"] = false;
  }

  void disableTD() { //Stress diagonal
    flags["txx"] = false, flags["tyy"] = false, flags["tzz"] = false;
  }

  void disableTOD() { //Stress off-diagonal
    flags["tyz"] = false, flags["tzx"] = false, flags["txy"] = false;
  }

  void disableALL() {
    for (auto& [key,_] : flags)
      flags[key] = false;
  }

  void enableJ() {
    flags["jx"] = true, flags["jy"] = true, flags["jz"] = true;
  }

  void enableP() {
    flags["px"] = true, flags["py"] = true, flags["pz"] = true;
  }

  void enableTD() { //Stress diagonal
    flags["txx"] = true, flags["tyy"] = true, flags["tzz"] = true;
  }

  void enableTOD() { //Stress off-diagonal
    flags["tyz"] = true, flags["tzx"] = true, flags["txy"] = true;
  }

  void enableALL() {
    for (auto& [key,_] : flags)
      flags[key] = true;
  }

  bool enabledJ() {
    return flags["jx"] && flags["jy"] && flags["jz"];
  }

  bool enabledP() {
    return flags["px"] && flags["py"] && flags["pz"];
  }

  bool enabledTD() {
    return flags["txx"] && flags["tyy"] && flags["tzz"];
  }

  bool enabledTOD() {
    return flags["tyz"] && flags["tzx"] && flags["txy"];
  }
};

//------------------------------------------------------------------------------
// class HDF5Dump
// functions to dump fields, hydro, and particle in HDF5 format

class HDF5Dump : public Dump_Strategy
{
public:
  int field_interval;
  int hydro_interval;
  int fluid_interval;
  int num_step;
  size_t stride_x = 1; // stride along each direction
  size_t stride_y = 1;
  size_t stride_z = 1;
  size_t stride_particle = 1; // stride for particle dump
  std::unordered_map<species_id, size_t> tframe_map;

  HDF5Dump(int _rank, int _nproc, int _ns, int _fieldi, int _hydroi, int _fluidi) :
    Dump_Strategy(_rank, _nproc), num_step(_ns), field_interval(_fieldi),
    hydro_interval(_hydroi), fluid_interval(_fluidi) {}

  hydro_dump_flag_t hydro_dump_flag;
  field_dump_flag_t field_dump_flag;

  void set_strides(size_t sx, size_t sy, size_t sz) {
    stride_x = sx;
    stride_y = sy;
    stride_z = sz;
  };

  void set_stride_particle(size_t stride) {
    stride_particle = stride;
  }

  void dump_fields(
      const char *fbase,
      int step,
      grid_t *grid,
      field_array_t *field_array,
      int ftag);
  void dump_hydro(
      const char *fbase,
      int step,
      species_t *sp,
      grid_t *grid,
      hydro_array_t *hydro_array,
      interpolator_array_t *interpolator_array,
      int ftag);
  void dump_particles(
      const char *fbase,
      int step,
      species_t *sp,
      grid_t *grid,
      interpolator_array_t *interpolator_array,
      int ftag);
  void dump_fluids(
      const char *fbase,
      int step,
      fluid_species_t *fsp,
      grid_t *grid,
      int ftag);
  void field_dump(
      DumpParameters& dumpParams,
      int step,
      grid_t *grid,
      field_array_t *field_array);
  void hydro_dump(
      DumpParameters& dumpParams,
      int step,
      species_t *sp,
      grid_t *grid,
      hydro_array_t *hydro_array,
      interpolator_array_t *interpolator_array);
  void fluid_dump(
      DumpParameters& dumpParams,
      int step,
      fluid_species_t *fsp,
      grid_t *grid);
};
#endif  // #define VPIC_ENABLE_HDF5

#endif
