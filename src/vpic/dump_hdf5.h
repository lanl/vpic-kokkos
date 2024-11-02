/*
 * Written by:
 *   Bin Dong, Ph.D.
 *   Scientific Data Management Group
 *   Scientific Data Division
 *   Lawrence Berkeley National Lab
 *
 *  Xiaocan Li - revised for hybrid-VPIC, 11/2024
 *
 */

#ifndef dump_hdf5_h
#define dump_hdf5_h

#include "hdf5.h"

/* Flags for dumping fields using HDF5 */
struct field_dump_flag_t
{
  bool ex = true, ey = true, ez = true, div_e_err = true;
  bool cbx = true, cby = true, cbz = true, div_b_err = true;
  bool cbx0 = true, cby0 = true, cbz0 = true, tmpsm = true; // External (potential) magnetic field
  bool tcax = true, tcay = true, tcaz = true, rhob = true; // hybrid: tcax multiplies hypereta, tcay multiplies eta, tcaz multiplies E field 
  bool jfx = true, jfy = true, jfz = true, rhof = true; // Free current and charge density
  bool jfxold = true, jfyold = true, jfzold = true, rhofold = true; // Free current and charge density
  bool tx = true, ty = true, tz = true, te = true; // Electron temperature + temp storage
  bool ox = true, oy = true, oz = true, oe = true; // For B field solve/smoothing
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

/* Flags for dumping hydro using HDF5 */
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
#endif // dump_hdf5_h
