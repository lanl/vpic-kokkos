import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pywt

# ============================================================================
# Configuration
# ============================================================================
datadir = "../../build/data/"
nx = 48  # 96
nt = 100

pi = np.pi

# Spatial and temporal grids
xv = np.linspace(0, 16, num=nx)
tv = np.linspace(0, 50, num=nt)
if (nx > 1): dx = xv[1] - xv[0]
if (nt > 1): dt = tv[1] - tv[0]

# Growth/damping rate
gamma = -0.093196

# ============================================================================
# Wavelet Denoising Function
# ============================================================================
def wclean(arr, wavn, alpha):
    """
    Wavelet denoising function using adaptive thresholding.
    
    Parameters:
    -----------
    arr : ndarray
        Input array to denoise
    wavn : str
        Wavelet name (e.g., 'coif3')
    alpha : float
        Threshold multiplier
    """
    cs = pywt.wavedecn(arr, wavn, mode='symmetric', level=None, axes=None)
    levs = len(cs)
    coef = np.concatenate(cs[0])
    
    for x in range(1, levs):
        for n in cs[x]:
            coefn = np.concatenate(cs[x][n])
            coef = np.concatenate((coef, coefn))
    
    thr2 = alpha * np.sqrt(np.var(coef) * np.log(len(coef)))
    thr = alpha * 2 * thr2
    
    while (thr / thr2 > 1.05):
        thr = thr2
        thr2 = alpha * np.sqrt(np.var(coef[np.abs(coef) < thr]) * np.log(len(coef)))
    
    for x in range(1, levs):
        for n in cs[x]:
            inds = np.abs(cs[x][n]) < thr
            cs[x][n][inds] = 0
    
    return pywt.waverecn(cs, wavn, mode='symmetric', axes=None)

# ============================================================================
# Data Loading Function
# ============================================================================
def loadSlice(dir, q, sl, nx, ny):
    """
    Load a 2D slice of data from binary file.
    
    Parameters:
    -----------
    dir : str
        Directory path
    q : str
        Quantity name
    sl : int
        Slice index
    nx, ny : int
        Grid dimensions
    """
    fstr = dir + q + ".gda"
    fd = open(fstr, "rb")
    fd.seek(4 * sl * nx * ny, 1)
    arr = np.fromfile(fd, dtype=np.float32, count=nx * ny)
    fd.close()
    arr = np.reshape(arr, (ny, nx))
    arr = np.transpose(arr)
    return arr

# ============================================================================
# Load Data
# ============================================================================
print("Loading simulation data...")
cmap = plt.get_cmap("RdBu_r")

Q = {}
quantities = ["ni", "Ex", "Uix"]

for slice_idx in range(0, 1):
    for q in quantities:
        tmp = loadSlice(datadir, q, slice_idx, nx, nt)
        Q[q] = tmp
        print("Loaded quantity: {}".format(q))

print("Data loaded successfully.")
print("\nDensity data (first few values):")
print(Q["ni"])

# ============================================================================
# Compute Diagnostics
# ============================================================================
# RMS density perturbation (deviation from equilibrium)
density_perturbation = np.sqrt(np.sum((1 - Q["ni"])**2, axis=0))

# Envelope for comparison
exponential_fit = 0.08 * np.exp(gamma * tv[0:50])

# ============================================================================
# Create Figure
# ============================================================================
fig = plt.figure(figsize=(8, 6))
# Top panel: Spatiotemporal evolution
ax1 = plt.subplot(2, 1, 1)
im1 = ax1.pcolormesh(tv, xv, Q["ni"], cmap=cmap, shading='auto')
ax1.set_ylabel('Position x/L', fontsize=11)
ax1.set_title('Ion Density Evolution', fontsize=12)
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Ion Density', rotation=270, labelpad=20)

# Bottom panel: Growth/damping rate
ax2 = plt.subplot(2, 1, 2)
ax2.plot(tv, np.log10(density_perturbation), linewidth=2.5, 
         label='RMS Density Perturbation')
ax2.plot(tv[0:50], np.log10(exponential_fit), linewidth=2, 
         label='Envelope (gamma = {:.4f})'.format(gamma))
ax2.set_xlabel('Time', fontsize=11)
ax2.set_ylabel('log10(Density Perturbation)', fontsize=11)
ax2.set_title('Density Perturbation Damping Rate', fontsize=12)
ax2.legend(loc='best', framealpha=0.9, fontsize=10)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save figure
plt.savefig('ion_density_evolution.png', dpi=300, bbox_inches='tight')
print("\nFigure saved as 'ion_density_evolution.png'")

plt.show()