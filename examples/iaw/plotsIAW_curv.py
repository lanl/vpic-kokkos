import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# 1D ion-acoustic wave along z on a CYLINDRICAL mesh.
# Deck (iaw_curv.cxx): r=x in [0.1, Lx], theta=y in [-pi,pi], z in [-Lz/2, Lz/2],
# grid nx(r) x ny(theta) x nz(z), wave seeded as a sinusoid in z (kz).
# gda records are nx*ny*nz float32 per time, x(r) fastest, then y(theta), then z.

datadir = "../../build/data/"
nx = 10   # r   (must match deck)
ny = 10   # theta
nz = 48   # z   (wave direction)
Lz = 16.0
taui = 50

# infer nt from file size so we never mismatch the run
raw = np.fromfile(datadir + "ni.gda", dtype=np.float32)
ncell = nx*ny*nz
nt = raw.size // ncell
print("nx=%d ny=%d nz=%d nt=%d" % (nx, ny, nz, nt))

zv = np.linspace(-Lz/2, Lz/2, num=nz)
tv = np.linspace(0, taui, num=nt)

######### loadSlice: return z-profile (nz, nt), averaged over the r,theta cross-section
def loadSlice(dir, q):
    arr = np.fromfile(dir + q + ".gda", dtype=np.float32)[:nt*ncell]
    arr = arr.reshape(nt, nz, ny, nx)   # z slowest, r(x) fastest
    arr = arr.mean(axis=(2, 3))         # average over theta(y) and r(x) -> (nt, nz)
    return arr.T                        # -> (nz, nt)
######### end loadSlice

Q = {}
for q in ["ni", "Ez", "Uiz"]:
    Q[q] = loadSlice(datadir, q)

ni = Q["ni"]     # (nz, nt)

# On a cylindrical mesh the equilibrium ni is not 1 (it reflects the r,theta
# average of a jac-weighted density); measure the z-perturbation relative to the
# time-averaged equilibrium z-profile so a flat wave reads as zero perturbation.
ni_eq = ni.mean(axis=1, keepdims=True)
dni   = (ni - ni_eq) / ni_eq
dn    = np.sqrt(np.mean(dni*dni, axis=0))   # rms over z, per time

print("ni range:", ni.min(), ni.max(), " (all-zero would mean bad read)")

gamma = -0.093196
fig, (ax1, ax2) = plt.subplots(nrows=2)
im = ax1.pcolormesh(tv, zv, dni)
ax1.set_ylabel('z')
fig.colorbar(im, ax=ax1, label='(ni-ni_eq)/ni_eq')

ax2.plot(tv, np.log10(dn), label='|dn| rms(z)')
ax2.plot(tv, np.log10(dn[0]*np.exp(gamma*tv)), '--', label='exp(gamma t)')
ax2.set_xlabel('t * C_s/L')
ax2.set_ylabel('log10 |dn|')
ax2.legend()

plt.tight_layout()
plt.savefig('fig_curv.png', dpi=300)
plt.show()
