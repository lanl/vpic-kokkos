import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# Stretched-grid IAW plot. Assumes the deck used
#   init_stretched_cartesian_grid(beta_x, 0, 0)   (x-stretch only)
# with a periodic domain [gx0, gx1], nx cells, topology_x = 1.
#
# Two things differ from the uniform plotsIAW.py:
#  1. Cell centers are NOT uniformly spaced -> use the tanh map for the x axis.
#  2. The equilibrium ion density is NOT 1 (it is ~ 1/jac, i.e. proportional to
#     the local cell size). So the perturbation is measured relative to the
#     equilibrium profile, and the |dn| norm is volume-weighted so it is the
#     physical L2 perturbation, not a per-cell-count artifact.

datadir = "../../build/data/"
nx   = 48
gx0, gx1 = -8.0, 8.0     # domain (deck: -0.5*Lx .. 0.5*Lx, Lx=16)
beta_x = 2.0             # deck: init_stretched_cartesian_grid(beta_x,0,0)
taui = 50                # run time in Cs/L units for the t axis

# ---- reconstruct the stretched grid geometry (matches grid.h init) ----------
ii = np.arange(nx)
xi = (ii + 0.5) / nx                      # uniform logical coordinate in (0,1)
if beta_x > 1e-10:
    xs    = np.tanh(beta_x*(xi-0.5))/np.tanh(beta_x*0.5)
    dx_dxi = beta_x/(np.tanh(beta_x*0.5)*np.cosh(beta_x*(xi-0.5))**2)
else:
    xs    = 2.0*xi - 1.0
    dx_dxi = np.full(nx, 2.0)
xc  = gx0 + (gx1-gx0)*(xs+1.0)*0.5        # physical cell centers
# physical cell width (volume weight for 1D). Constant factors cancel in ratios.
Vphys = (gx1-gx0)*0.5*dx_dxi*(1.0/nx)     # = h1 * dref, proportional to cell size

# ---- load ni.gda, infer nt from file size -----------------------------------
raw = np.fromfile(datadir+"ni.gda", dtype=np.float32)
nt  = raw.size // nx
ni  = raw[:nt*nx].reshape(nt, nx)
tv  = np.linspace(0, taui, num=nt)
print("nx=%d nt=%d beta_x=%g" % (nx, nt, beta_x))

# ---- equilibrium and perturbation -------------------------------------------
# The stretched equilibrium density is non-uniform; estimate it as the time
# mean (the oscillating wave averages out, leaving the equilibrium profile).
ni_eq = ni.mean(axis=0)
dni   = (ni - ni_eq) / ni_eq              # relative perturbation, per cell, per time

# Volume-weighted L2 norm of the perturbation: sqrt( sum_x w*dni^2 / sum_x w ),
# w = Vphys. This is the physical rms perturbation, independent of the
# non-uniform cell layout.
w = Vphys / Vphys.sum()
dn = np.sqrt(np.sum(w[None,:]*dni*dni, axis=1))

# ---- charge conservation check (should be ~flat) ----------------------------
Q = np.sum(ni*Vphys[None,:], axis=1); Q /= Q[0]
print("Q(t)/Q0 min/max:", Q.min(), Q.max())

# ---- plots ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(nrows=2)

# density perturbation in physical space (x axis is the stretched centers)
im = ax1.pcolormesh(tv, xc, dni.T)
ax1.set_ylabel('x  (stretched)')
fig.colorbar(im, ax=ax1, label='(ni-ni_eq)/ni_eq')

gamma = -0.093196
ax2.plot(tv, np.log10(dn), label='|dn| (vol-weighted)')
ax2.plot(tv, np.log10(dn[0]*np.exp(gamma*tv)), '--', label='exp(gamma t)')
ax2.set_xlabel('t * C_s/L')
ax2.set_ylabel('log10 |dn|')
ax2.legend()

plt.tight_layout()
plt.savefig('fig_stretch.png', dpi=300)
plt.show()
