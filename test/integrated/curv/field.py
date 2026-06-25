#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Read trajectory data
data = np.loadtxt("../../../build/trajectory.txt")

step = data[:, 0]
x1 = data[:, 1]
y1 = data[:, 2]
r1 = data[:, 3]
theta1 = data[:, 4]
ux1 = data[:, 5]
uy1 = data[:, 6]

# Read field data
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

# Calculate field magnitudes
bz_field = field_cbz
e_mag = np.sqrt(field_ex**2 + field_ey**2)
j_mag = np.sqrt(field_jfx**2 + field_jfy**2)

# Print field statistics
print("\nField statistics:")
print(f"\nB_z field:")
print(f"  Min: {bz_field.min():.6e}")
print(f"  Max: {bz_field.max():.6e}")
print(f"  Mean: {bz_field.mean():.6e}")

print(f"\n|E| field:")
print(f"  Min: {e_mag.min():.6e}")
print(f"  Max: {e_mag.max():.6e}")
print(f"  Mean: {e_mag.mean():.6e}")

print(f"\n|J| field:")
print(f"  Min: {j_mag.min():.6e}")
print(f"  Max: {j_mag.max():.6e}")
print(f"  Mean: {j_mag.mean():.6e}")

# Create figure with 3 subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Reshape field data to 2D grid
print(field_data)
nr = len(np.unique(field_data[:, 0]))  # number of r points
nt = len(np.unique(field_data[:, 1]))  # number of theta points

field_x_grid = field_x.reshape(nt, nr)
field_y_grid = field_y.reshape(nt, nr)

# Prepare each field
fields = [
    (bz_field.reshape(nt, nr), 'B_z', 'RdBu_r'),
    (e_mag.reshape(nt, nr), '|E|', 'plasma'),
    (j_mag.reshape(nt, nr), '|J|', 'viridis')
]

# Plot each field
for ax, (field_grid, label, cmap) in zip(axes, fields):
    mesh = ax.pcolormesh(field_x_grid, field_y_grid, field_grid,
                        cmap=cmap, shading='auto', alpha=0.6)
    cbar = plt.colorbar(mesh, ax=ax, label=label)
    
    # Plot trajectory
    ax.plot(x1, y1, 'b-', linewidth=2, alpha=0.9, label='Particle 1 (+q)', zorder=10)
    ax.plot(x1[0], y1[0], 'go', markersize=10, zorder=11, label='Start')
    ax.plot(x1[-1], y1[-1], 'ro', markersize=10, zorder=11, label='End')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'{label} Field', fontsize=14)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')

plt.suptitle('Particle Orbit with Field Overlays', fontsize=16, y=1.02)
plt.tight_layout()

output_file = 'orbit_fields_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved {output_file}")
plt.show()