#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

# Command line argument for field type
field_type = 'E'  # default
if len(sys.argv) > 1:
    field_type = sys.argv[1].upper()
    if field_type not in ['E', 'B', 'J']:
        print(f"Unknown field type '{field_type}'. Use E, B, or J")
        sys.exit(1)

# Read trajectory data
data = np.loadtxt("../../../build/orbit_trajectory.txt")

step = data[:, 0]
x1 = data[:, 1]
y1 = data[:, 2]
r1 = data[:, 3]
theta1 = data[:, 4]
ux1 = data[:, 5]
uy1 = data[:, 6]

# Read field data
try:
    field_data = np.loadtxt("../../../build/fields.txt")
    # Columns: i j k x y r theta jfx jfy ex ey cbx cby cbz
    field_x = field_data[:, 3]
    field_y = field_data[:, 4]
    field_jfx = field_data[:, 7]
    field_jfy = field_data[:, 8]
    field_ex = field_data[:, 9]
    field_ey = field_data[:, 10]
    field_cbx = field_data[:, 11]
    field_cby = field_data[:, 12]
    field_cbz = field_data[:, 13]

    # Select field based on command line argument
    if field_type == 'E':
        field_mag = np.sqrt(field_ex**2 + field_ey**2)
        field_label = '|E| field magnitude'
    elif field_type == 'B':
        field_mag = np.sqrt(field_cbx**2 + field_cby**2 + field_cbz**2)
        field_label = '|B| field magnitude'
    elif field_type == 'J':
        field_mag = np.sqrt(field_jfx**2 + field_jfy**2)
        field_label = '|J| current density magnitude'

    has_fields = True

    # Debug: print field statistics
    print(f"\n{field_type}-field statistics:")
    print(f"  Min: {field_mag.min():.6e}")
    print(f"  Max: {field_mag.max():.6e}")
    print(f"  Mean: {field_mag.mean():.6e}")
    print(f"  Median: {np.median(field_mag):.6e}")
    print(f"  95th percentile: {np.percentile(field_mag, 95):.6e}")
    print(f"  Non-zero points: {np.sum(field_mag > 1e-10)}/{len(field_mag)}")
except:
    print("Warning: Could not load fields.txt, skipping field visualization")
    has_fields = False

# Plot x-y trajectory for both particles with field heatmap
fig, ax = plt.subplots(figsize=(12, 10))

if has_fields:
    # Reshape field data to 2D grid for fast plotting
    # Assuming regular grid structure
    nr = len(np.unique(field_data[:, 0]))  # number of r points
    nt = len(np.unique(field_data[:, 1]))  # number of theta points

    field_mag_grid = field_mag.reshape(nt, nr)
    field_x_grid = field_x.reshape(nt, nr)
    field_y_grid = field_y.reshape(nt, nr)

    # Use pcolormesh for fast rendering
    mesh = ax.pcolormesh(field_x_grid, field_y_grid, field_mag_grid,
                         cmap='plasma', shading='auto', alpha=0.6)
    cbar = plt.colorbar(mesh, ax=ax, label=f'{field_label}')

ax.plot(x1, y1, 'b-', linewidth=2, alpha=0.9, label='Particle 1 (+q)', zorder=10)
ax.plot(x1[0], y1[0], 'go', markersize=12, zorder=11)
ax.plot(x1[-1], y1[-1], 'ro', markersize=12, zorder=11)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title(f'Two-Particle Orbit Trajectories with {field_type}-field', fontsize=14)
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
output_file = f'orbit_xy_{field_type}.png'
plt.savefig(output_file, dpi=150)
print(f"Saved {output_file}")
print(f"Usage: python orbit.py [E|B|J]  (default: E)")
plt.show()
