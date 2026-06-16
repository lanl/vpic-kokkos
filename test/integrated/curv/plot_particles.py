#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob

# Find all particle files
files = sorted(glob.glob("../../../build/data/particle_positions_step_*.txt"))

if not files:
    print("No particle files found!")
    exit(1)

print(f"Found {len(files)} particle files")

# Read all positions
all_positions = {}  # particle_id -> list of (step, x, y)

for filename in files:
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse step from header
    step = int(lines[0].split('=')[1].split()[0])

    # Parse particle data
    for p_idx, line in enumerate(lines[2:]):
        if line.strip():
            data = [float(x) for x in line.split()]
            x = data[0]
            y = data[1]

            if p_idx not in all_positions:
                all_positions[p_idx] = []
            all_positions[p_idx].append((step, x, y))
    print(f'read {filename}')

# Plot trajectories
plt.figure(figsize=(10, 10))

from tqdm import tqdm
for p_idx, trajectory in tqdm(all_positions.items()):
    trajectory = sorted(trajectory, key=lambda t: t[0])  # Sort by step
    steps = [t[0] for t in trajectory]
    xs = [t[1] for t in trajectory]
    ys = [t[2] for t in trajectory]

    plt.plot(xs, ys, alpha=0.5, linewidth=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Particle trajectories ({len(all_positions)} particles, {len(files)} timesteps)')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.savefig('particle_trajectories.png', dpi=150)
print("Saved particle_trajectories.png")
plt.show()
