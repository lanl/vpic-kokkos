#!/usr/bin/env python3
# Visualization for the plasma-sphere verification deck.
#
# Reads the (r, theta, phi) .gda fields produced by translate_faster.f90 (same
# 3D layout as examples/mirror). Produces:
#   1. A poloidal (r,theta) slice of ion density ni at a chosen phi and time.
#   2. The radial density profile n(r) (theta,phi-averaged) at several times.
#   3. A time series of the perturbation amplitude, to show it oscillates
#      rather than growing/running away.
#
# Usage:  python3 plots_plasma_sphere.py [time_slice]

import numpy as np
import os
import struct
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dir = "../../build/data/"

# ----------------------------------------------------------------------------
def loadinfo(d):
    """Read grid dims and physical extents from data/info (written by translate)."""
    with open(d + "info", "rb") as fd:
        arr = struct.unpack("fIIIfffff", fd.read()[:36])
    nr, nth, nz = arr[1], arr[2], arr[3]   # nx=r, ny=theta, nz=phi
    rmax, thmax, phimax = arr[6], arr[7], arr[8]
    print(f"Grid: nr={nr} ntheta={nth} nphi={nz} | rmax={rmax} thetamax={thmax} phimax={phimax}")
    return nr, nth, nz, rmax, thmax, phimax

def num_time_slices(d, q, nr, nth, nz):
    return os.path.getsize(d + q + ".gda") // (nr * nth * nz * 4)

def load_slice(d, q, tslice, nr, nth, nz, iphi=0):
    """One phi-layer of a time slice. On-disk order is Fortran (r,theta,phi),
    i.e. C-shape (nphi, ntheta, nr). Returns (r, theta) array."""
    with open(d + q + ".gda", "rb") as fd:
        fd.seek(4 * tslice * nr * nth * nz, 1)
        a = np.fromfile(fd, dtype=np.float32, count=nr * nth * nz)
    a = np.reshape(a, (nz, nth, nr))   # (phi, theta, r)
    a = a[iphi, :, :]                  # (theta, r)
    return np.transpose(a)             # (r, theta)

def radial_profile(d, q, tslice, nr, nth, nz):
    """theta- and phi-averaged radial profile n(r)."""
    with open(d + q + ".gda", "rb") as fd:
        fd.seek(4 * tslice * nr * nth * nz, 1)
        a = np.fromfile(fd, dtype=np.float32, count=nr * nth * nz)
    a = np.reshape(a, (nz, nth, nr))   # (phi, theta, r)
    return a.mean(axis=(0, 1))         # average over phi, theta -> (r,)

# ----------------------------------------------------------------------------
nr, nth, nz, rmax, thmax, phimax = loadinfo(dir)
nt = num_time_slices(dir, "ni", nr, nth, nz)
print(f"Number of time slices: {nt}")

tslice = nt - 1
if len(sys.argv) > 1:
    tslice = int(sys.argv[1])

rv  = np.linspace(0.0, rmax, nr)
thv = np.linspace(0.0, thmax, nth)

# ============================================================
# FIGURE 1: poloidal (r,theta) density slice at phi=0
# ============================================================
ni = load_slice(dir, "ni", tslice, nr, nth, nz, iphi=0)  # (r, theta)
# Map (r,theta) -> Cartesian (R_cyl, Z) for a poloidal view: R=r sin th, Z=r cos th
R, TH = np.meshgrid(rv, thv, indexing="ij")
Xp = R * np.sin(TH)
Zp = R * np.cos(TH)

fig1, ax1 = plt.subplots(figsize=(6, 8))
im1 = ax1.pcolormesh(Xp, Zp, ni, cmap="Spectral_r", shading="auto")
ax1.set_xlabel("r sin(theta)")
ax1.set_ylabel("r cos(theta)")
ax1.set_title(f"Ion density ni  (phi=0 slice, t-slice {tslice})")
ax1.set_aspect("equal")
fig1.colorbar(im1, ax=ax1, label="ni", shrink=0.6)
fig1.tight_layout()
fig1.savefig("plot_density_poloidal.png", dpi=200)
print("Saved plot_density_poloidal.png")

# ============================================================
# FIGURE 2: radial profile n(r) at several times
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))
sample_ts = sorted(set(int(f) for f in np.linspace(0, nt - 1, min(nt, 6))))
for ts in sample_ts:
    prof = radial_profile(dir, "ni", ts, nr, nth, nz)
    ax2.plot(rv, prof, label=f"t-slice {ts}")
ax2.set_xlabel("r")
ax2.set_ylabel("<ni>  (theta,phi averaged)")
ax2.set_title("Radial density profile vs time")
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("plot_radial_profile.png", dpi=200)
print("Saved plot_radial_profile.png")

# ============================================================
# FIGURE 3: perturbation amplitude time series (oscillation check)
# ============================================================
# Amplitude = max deviation of the radial profile from its time-mean baseline.
profiles = np.array([radial_profile(dir, "ni", ts, nr, nth, nz) for ts in range(nt)])
baseline = profiles.mean(axis=0)
amp = np.sqrt(((profiles - baseline) ** 2).mean(axis=1))  # RMS radial perturbation
tvec = np.arange(nt)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(tvec, amp, "-o", ms=3)
ax3.set_xlabel("time slice")
ax3.set_ylabel("RMS density perturbation")
ax3.set_title("Perturbation amplitude vs time (should oscillate / stay bounded)")
fig3.tight_layout()
fig3.savefig("plot_amplitude_timeseries.png", dpi=200)
print("Saved plot_amplitude_timeseries.png")

print(f"amplitude: start={amp[0]:.3e}  max={amp.max():.3e}  end={amp[-1]:.3e}")
