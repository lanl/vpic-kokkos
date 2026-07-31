import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
datadir = "../../build/data/"
nx = 64  # Number of spatial points
nt = 300  # Number of time steps

# Spatial and temporal grids
# Compute physical x coordinates for stretched Cartesian grid
Lx = 10.5  # since di=1 in normalized units
Ly = 1.0
Lz = 1.0
beta_x = 2.0
beta_y = 0.0
beta_z = 0.0
# Stretching function
def stretch_map(xi, beta):
    if abs(beta) < 1e-12:
        return 2.0 * xi - 1.0
    else:
        return np.tanh(beta * (xi - 0.5)) / np.tanh(beta * 0.5)
gx0 = -0.5 * Lx
gx1 = 0.5 * Lx
xi_centers = (np.arange(nx) + 0.5) / nx
x_stretched = stretch_map(xi_centers, beta_x)
xv = gx0 + (gx1 - gx0) * (x_stretched + 1.0) * 0.5
tv = np.linspace(0, 60, num=nt)
dx = xv[1] - xv[0] if nx > 1 else 0  # Note: dx is not constant in stretched grid, but not used elsewhere
dt = tv[1] - tv[0] if nt > 1 else 0
# Growth rate for exponential fit
gamma = 0.162

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
print("Loading data...")
Q = {}
quantities = ["Uiy", "Uiz", "aniso", "By", "Bz"]
for q in quantities:
    tmp = loadSlice(datadir, q, 0, nx, nt)
    Q[q] = tmp
print("Data loaded successfully.")

# ============================================================================
# Compute Diagnostics
# ============================================================================
# RMS current density components
J_y_rms = np.sqrt(np.sum(Q["Uiy"]**2, axis=0))
J_z_rms = np.sqrt(np.sum(Q["Uiz"]**2, axis=0))

# RMS magnetic field components
B_y_rms = np.sqrt(np.sum(Q["By"]**2, axis=0))
B_z_rms = np.sqrt(np.sum(Q["Bz"]**2, axis=0))

# Pressure anisotropy (spatial average)
pressure_anisotropy = np.mean(Q["aniso"], axis=0)

# Exponential growth fit
exponential_fit = 0.019 * np.exp(gamma * tv)
exponential_fit_B = 0.012 * np.exp(gamma * tv)

# ============================================================================
# Create Comprehensive Figure
# ============================================================================
fig = plt.figure(figsize=(14, 10))

# Define colormap
cmap = plt.get_cmap("RdBu_r")

# ------------ Row 1: Spatiotemporal Evolution ------------
# Current density J_y
ax1 = plt.subplot(3, 2, 1)
im1 = ax1.pcolormesh(tv, xv, Q["Uiy"], cmap=cmap, shading='auto')
ax1.set_ylabel('Position (ion inertial length)', fontsize=11)
ax1.set_title('Current Density Jy Evolution', fontsize=11)
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Jy', rotation=0, labelpad=15)

# Current density J_z
ax2 = plt.subplot(3, 2, 2)
im2 = ax2.pcolormesh(tv, xv, Q["Uiz"], cmap=cmap, shading='auto')
ax2.set_ylabel('Position (ion inertial length)', fontsize=11)
ax2.set_title('Current Density Jz Evolution', fontsize=11)
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Jz', rotation=0, labelpad=15)

# ------------ Row 2: Growth Rates ------------
# Current density growth
ax3 = plt.subplot(3, 2, 3)
ax3.plot(tv, np.log10(J_y_rms), linewidth=2, label='RMS Current Jy')
ax3.plot(tv, np.log10(J_z_rms), linewidth=2, label='RMS Current Jz')
ax3.plot(tv, np.log10(exponential_fit), linewidth=2, 
         label='Envelope (gamma = {})'.format(gamma))
ax3.set_xlabel('Time', fontsize=11)
ax3.set_ylabel('log10(RMS Current)', fontsize=11)
ax3.set_title('Current Density Growth Rate', fontsize=11)
ax3.legend(loc='best', framealpha=0.9)
ax3.set_xlim([0, 60])
ax3.set_ylim([-2, 1])

# Magnetic field growth
ax4 = plt.subplot(3, 2, 4)
ax4.plot(tv, np.log10(B_y_rms), linewidth=2, label='RMS Field By')
ax4.plot(tv, np.log10(B_z_rms), linewidth=2, label='RMS Field Bz')
ax4.plot(tv, np.log10(exponential_fit_B), linewidth=2, 
         label='Envelope (gamma = {})'.format(gamma))
ax4.set_xlabel('Time', fontsize=11)
ax4.set_ylabel('log10(RMS Magnetic Field)', fontsize=11)
ax4.set_title('Magnetic Field Growth Rate', fontsize=11)
ax4.legend(loc='best', framealpha=0.9)
ax4.grid(alpha=0.3)
ax4.set_xlim([0, 60])
ax4.set_ylim([-2, 1])

# ------------ Row 3: Pressure Anisotropy ------------
ax5 = plt.subplot(3, 1, 3)
ax5.plot(tv, pressure_anisotropy, linewidth=2.5, label='Pressure Anisotropy')
ax5.axhline(y=1.0, linewidth=1.5, 
            alpha=0.7, label='Isotropic (P_perp = P_parallel)')
ax5.set_xlabel('Time', fontsize=11)
ax5.set_ylabel('P_perp / P_parallel', fontsize=11)
ax5.set_title('Pressure Anisotropy Evolution', fontsize=11)
ax5.legend(loc='best', framealpha=0.9)
ax5.grid(alpha=0.3)
ax5.set_xlim([0, 60])

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
plt.savefig('plasma_current_anisotropy_analysis.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'plasma_current_anisotropy_analysis.png'")

# plt.show()