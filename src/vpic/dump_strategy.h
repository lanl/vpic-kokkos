#ifndef VPIC_DUMP_STRATEGY_H_
#define VPIC_DUMP_STRATEGY_H_

// C++ headers
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cassert>

// VPIC headers
#include "../util/io/FileIO.h"
#include "../util/util_base.h"
#include "../util/io/FileUtils.h"
#include "../field_advance/field_advance.h"
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
      hydro_array_t *hydro_array,
      species_t *sp,
      interpolator_array_t *interpolator_array,
      grid_t *grid,
      int ftag) = 0;
  virtual void dump_particles(
      const char *fbase,
      species_t *sp,
      grid_t *grid,
      int step,
      interpolator_array_t *interpolator_array,
      int ftag) = 0;
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
      hydro_array_t *hydro_array,
      species_t *sp,
      interpolator_array_t *interpolator_array,
      grid_t *grid,
      int ftag);
  void dump_particles(
      const char *fbase,
      species_t *sp,
      grid_t *grid,
      int step,
      interpolator_array_t *interpolator_array,
      int ftag);
};

#ifdef VPIC_ENABLE_HDF5

struct field_dump_flag_t
{
  bool ex = true, ey = true, ez = true, div_e_err = true;
  bool cbx = true, cby = true, cbz = true, div_b_err = true;
  // External (potential) magnetic field
  bool cbx0 = true, cby0 = true, cbz0 = true, tmpsm = true;
  // hybrid: tcax multiplies hypereta, tcay multiplies eta, tcaz multiplies E field
  bool tcax = true, tcay = true, tcaz = true, rhob = true;
  // Free current and charge density
  bool jfx = true, jfy = true, jfz = true, rhof = true;
  // Free current and charge density
  bool jfxold = true, jfyold = true, jfzold = true, rhofold = true;
  // Electron temperature + temp storage
  bool tx = true, ty = true, tz = true, te = true;
  // For B field solve/smoothing
  bool ox = true, oy = true, oz = true, oe = true;

  void disableE() {
    ex = false, ey = false, ez = false;
  }

  void disableCB() {
    cbx = false, cby = false, cbz = false;
  }

  void disableCB0() {
    cbx0 = false, cby0 = false, cbz0 = false;
  }

  void disableTCA() {
    tcax = false, tcay = false, tcaz = false;
  }

  void disableJF() {
    jfx = false, jfy = false, jfz = false;
  }

  void disableJFOLD() {
    jfxold = false, jfyold = false, jfzold = false;
  }

  void disableT() {
    tx = false, ty = false, tz = false;
  }

  void disableO() {
    ox = false, oy = false, oz = false;
  }

  void disableALL() {
    ex = false, ey = false, ez = false, div_e_err = false;
    cbx = false, cby = false, cbz = false, div_b_err = false;
    cbx0 = false, cby0 = false, cbz0 = false, tmpsm = false;
    tcax = false, tcay = false, tcaz = false, rhob = false;
    jfx = false, jfy = false, jfz = false, rhof = false;
    jfxold = false, jfyold = false, jfzold = false, rhofold = false;
    tx = false, ty = false, tz = false, te = false;
    ox = false, oy = false, oz = false, oe = false;
  }

  void enableE() {
    ex = true, ey = true, ez = true;
  }

  void enableCB() {
    cbx = true, cby = true, cbz = true;
  }

  void enableCB0() {
    cbx0 = true, cby0 = true, cbz0 = true;
  }

  void enableTCA() {
    tcax = true, tcay = true, tcaz = true;
  }

  void enableJF() {
    jfx = true, jfy = true, jfz = true;
  }

  void enableJFOLD() {
    jfxold = true, jfyold = true, jfzold = true;
  }

  void enableT() {
    tx = true, ty = true, tz = true;
  }

  void enableO() {
    ox = true, oy = true, oz = true;
  }

  void enableALL() {
    ex = true, ey = true, ez = true, div_e_err = true;
    cbx = true, cby = true, cbz = true, div_b_err = true;
    cbx0 = true, cby0 = true, cbz0 = true, tmpsm = true;
    tcax = true, tcay = true, tcaz = true, rhob = true;
    jfx = true, jfy = true, jfz = true, rhof = true;
    jfxold = true, jfyold = true, jfzold = true, rhofold = true;
    tx = true, ty = true, tz = true, te = true;
    ox = true, oy = true, oz = true, oe = true;
  }

  bool enabledE() {
    return ex && ey && ez;
  }

  bool enabledCB() {
    return cbx && cby && cbz;
  }

  bool enabledCB0() {
    return cbx0 && cby0 && cbz0;
  }

  bool enabledTCA() {
    return tcax && tcay && tcaz;
  }

  bool enabledJF() {
    return jfx && jfy && jfz;
  }

  bool enabledJFOLD() {
    return jfxold && jfyold && jfzold;
  }

  bool enabledT() {
    return tx && ty && tz;
  }

  bool enabledO() {
    return ox && oy && oz;
  }
};

struct hydro_dump_flag_t
{
  bool jx = true, jy = true, jz = true, rho = true;
  bool px = true, py = true, pz = true, ke = true;
  bool txx = true, tyy = true, tzz = true;
  bool tyz = true, tzx = true, txy = true;

  void disableJ() {
    jx = false, jy = false, jz = false;
  }

  void disableP() {
    px = false, py = false, pz = false;
  }

  void disableTD() { //Stress diagonal
    txx = false, tyy = false, tzz = false;
  }

  void disableTOD() { //Stress off-diagonal
    tyz = false, tzx = false, txy = false;
  }

  void disableALL() {
    jx = false, jy = false, jz = false, rho = false;
    px = false, py = false, pz = false, ke = false;
    txx = false, tyy = false, tzz = false;
    tyz = false, tzx = false, txy = false;
  }

  void enableJ() {
    jx = true, jy = true, jz = true;
  }

  void enableP() {
    px = true, py = true, pz = true;
  }

  void enableTD() { //Stress diagonal
    txx = true, tyy = true, tzz = true;
  }

  void enableTOD() { //Stress off-diagonal
    tyz = true, tzx = true, txy = true;
  }

  void enableALL() {
    jx = true, jy = true, jz = true, rho = true;
    px = true, py = true, pz = true, ke = true;
    txx = true, tyy = true, tzz = true;
    tyz = true, tzx = true, txy = true;
  }

  bool enabledJ() {
    return jx && jy && jz;
  }

  bool enabledP() {
    return px && py && pz;
  }

  bool enabledTD() {
    return txx && tyy && tzz;
  }

  bool enabledTOD() {
    return tyz && tzx && txy;
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
  int num_step;
  std::unordered_map<species_id, size_t> tframe_map;

  HDF5Dump(int _rank, int _nproc, int _ns, int _fi, int _hi) :
    Dump_Strategy(_rank, _nproc), num_step(_ns), field_interval(_fi),
    hydro_interval(_hi) {}

  // TODO: replace these with a common dump interface
  // Declare vars to use
  hydro_dump_flag_t hydro_dump_flag;
  field_dump_flag_t field_dump_flag;

  void dump_fields(
      const char *fbase,
      int step,
      grid_t *grid,
      field_array_t *field_array,
      int ftag);
  void dump_hydro(
      const char *fbase,
      int step,
      hydro_array_t *hydro_array,
      species_t *sp,
      interpolator_array_t *interpolator_array,
      grid_t *grid,
      int ftag);
  void dump_particles(
      const char *fbase,
      species_t *sp,
      grid_t *grid,
      int step,
      interpolator_array_t *interpolator_array,
      int ftag);
};
#endif  // #define VPIC_ENABLE_HDF5

#endif
