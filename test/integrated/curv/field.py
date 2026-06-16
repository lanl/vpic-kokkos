#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load trajectory
traj = np.loadtxt('../../../build/trajectory.txt')
x_traj = traj[:, 1]
y_traj = traj[:, 2]

# Load fields
fields = np.loadtxt('../../../build/fields.txt')
fx, fy = fields[:, 3], fields[:, 4]
jfx, jfy = fields[:, 7], fields[:, 8]
ex, ey = fields[:, 9], fields[:, 10]
cbz = fields[:, 13]

# Compute magnitudes
j_mag = np.sqrt(jfx**2 + jfy**2)
e_mag = np.sqrt(ex**2 + ey**2)

# Create grid for interpolation (use unique x/y values to determine grid size)
nx = len(np.unique(fx))
ny = len(np.unique(fy))
xi = np.linspace(fx.min(), fx.max(), nx)
yi = np.linspace(fy.min(), fy.max(), ny)
XI, YI = np.meshgrid(xi, yi)

# Interpolate fields onto grid
J_grid = griddata((fx, fy), j_mag, (XI, YI), method='cubic')
E_grid = griddata((fx, fy), e_mag, (XI, YI), method='cubic')
B_grid = griddata((fx, fy), cbz, (XI, YI), method='cubic')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# J field with trajectory
im1 = axes[0].pcolormesh(XI, YI, J_grid, cmap='viridis', shading='auto')
axes[0].plot(x_traj, y_traj, 'w-', linewidth=2, alpha=0.8)
axes[0].plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Current density |J|')
axes[0].axis('equal')
axes[0].legend()
plt.colorbar(im1, ax=axes[0])

# E field with trajectory
im2 = axes[1].pcolormesh(XI, YI, E_grid, cmap='plasma', shading='auto')
axes[1].plot(x_traj, y_traj, 'w-', linewidth=2, alpha=0.8)
axes[1].plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('E field |E|')
axes[1].axis('equal')
axes[1].legend()
plt.colorbar(im2, ax=axes[1])

# B field with trajectory
im3 = axes[2].pcolormesh(XI, YI, B_grid, cmap='RdBu_r', shading='auto')
axes[2].plot(x_traj, y_traj, 'w-', linewidth=2, alpha=0.8)
axes[2].plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('B field (z component)')
axes[2].axis('equal')
axes[2].legend()
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('fields.png', dpi=150)
print("Saved fields.png")
plt.show()