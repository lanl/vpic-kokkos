#!/usr/bin/env python3

import numpy as np
import os, re, struct, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

build = "../../build"
fdir  = build + "/fields"
hdir  = build + "/hydro"

BOILER = 23
BLKHDR = 24

def read_dims_from_info():
    """nr,ntheta,nphi and physical extents from build/data/info if present,
    else fall back to reading nc from a dump header."""
    info = build + "/data/info"
    if os.path.exists(info):
        with open(info, "rb") as fd:
            arr = struct.unpack("fIIIfffff", fd.read()[:36])
        return arr[1], arr[2], arr[3], arr[6], arr[7], arr[8]
    return None

def read_nc(path):
    """Return GHOSTED cell counts (nx+2,ny+2,nz+2). The dump header stores the
    interior (nx,ny,nz); on-disk arrays include one ghost layer per side, so the
    array dims are nx+2 etc. (Verified vs file size: only nc+2 gives integer
    variable count.)"""
    with open(path, "rb") as fd:
        fd.seek(BOILER + 4 + 4 + 4)   # skip v0,itype,ndim
        nx, ny, nz = struct.unpack("3i", fd.read(12))
    return nx + 2, ny + 2, nz + 2

def read_var(path, ivar, nc):
    """Read variable index ivar (0-based) as a (ncz,ncy,ncx) array."""
    ncx, ncy, ncz = nc
    ncell = ncx * ncy * ncz
    with open(path, "rb") as fd:
        fd.seek(BOILER + BLKHDR + ivar * ncell * 4)
        a = np.fromfile(fd, dtype=np.float32, count=ncell)
    return a.reshape(ncz, ncy, ncx)

def time_indices(d):
    """Sorted list of dump time indices from directory names T.<n>."""
    ts = []
    for name in os.listdir(d):
        m = re.match(r"T\.(\d+)$", name)
        if m:
            ts.append(int(m.group(1)))
    return sorted(ts)

IVAR_NE = 3

ts = time_indices(hdir)
if not ts:
    print("No hydro dumps found under", hdir); sys.exit(1)
print(f"Found {len(ts)} time slices: T.{ts[0]} .. T.{ts[-1]}")

# Grid dims from the first hydro dump header
nc = read_nc(f"{hdir}/T.{ts[0]}/Hhydro.{ts[0]}.0")
ncx, ncy, ncz = nc
nr, nth, nphi = ncx - 2, ncy - 2, ncz - 2
print(f"Grid (incl ghosts) nc = {nc} -> nr={nr} ntheta={nth} nphi={nphi}")

info = read_dims_from_info()
if info:
    _, _, _, Rmax, thmax, phimax = info
else:
    Rmax, thmax, phimax = float(nr), np.pi, 2*np.pi
rv  = np.linspace(0.0, Rmax, nr)
thv = np.linspace(0.0, thmax, nth)

def hydro_ne(tidx):
    """Interior density (r,theta,phi) for a time index."""
    path = f"{hdir}/T.{tidx}/Hhydro.{tidx}.0"
    ne = read_var(path, IVAR_NE, nc) 
    return ne[1:1+nphi, 1:1+nth, 1:1+nr]

def radial_profile(tidx):
    ne = hydro_ne(tidx)
    return ne.mean(axis=(0, 1))

# time index to display
tsel = ts[-1]
if len(sys.argv) > 1:
    want = int(sys.argv[1])
    tsel = min(ts, key=lambda t: abs(t - want))

ne = hydro_ne(tsel)              # (nphi, nth, nr)
ni_rt = ne[0, :, :].T            # phi=0 layer -> (nr, nth)
R, TH = np.meshgrid(rv, thv, indexing="ij")
Xp = R * np.sin(TH)
Zp = R * np.cos(TH)

fig1, ax1 = plt.subplots(figsize=(6, 8))
im1 = ax1.pcolormesh(Xp, Zp, ni_rt, cmap="Spectral_r", shading="gouraud")
ax1.set_xlabel("r sin(theta)")
ax1.set_ylabel("r cos(theta)")
ax1.set_title(f"Ion density  (phi=0, T.{tsel})")
ax1.set_aspect("equal")
fig1.colorbar(im1, ax=ax1, label="ni", shrink=0.6)
fig1.tight_layout()
fig1.savefig("plot_density_poloidal.png", dpi=200)
print("Saved plot_density_poloidal.png")

fig2, ax2 = plt.subplots(figsize=(8, 5))
sample = [ts[i] for i in sorted(set(int(f) for f in np.linspace(0, len(ts)-1, min(len(ts), 6))))]
for t in sample:
    ax2.plot(rv, radial_profile(t), label=f"T.{t}")
ax2.set_xlabel("r"); ax2.set_ylabel("<ni> (theta,phi avg)")
ax2.set_title("Radial density profile vs time")
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("plot_radial_profile.png", dpi=200)
print("Saved plot_radial_profile.png")

profiles = np.array([radial_profile(t) for t in ts])
baseline = profiles.mean(axis=0)
amp = np.sqrt(((profiles - baseline) ** 2).mean(axis=1))
tphys = np.array(ts)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(tphys, amp, "-o", ms=3)
ax3.set_xlabel("time step"); ax3.set_ylabel("RMS density perturbation")
ax3.set_title("Perturbation amplitude vs time (should oscillate / stay bounded)")
fig3.tight_layout()
fig3.savefig("plot_amplitude_timeseries.png", dpi=200)
print("Saved plot_amplitude_timeseries.png")

print(f"amplitude: start={amp[0]:.3e}  max={amp.max():.3e}  end={amp[-1]:.3e}")
