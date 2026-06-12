#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Read trajectory data
data = np.loadtxt("../../../build/trajectory.txt")

step = data[:, 0]
x = data[:, 1]
y = data[:, 2]
r = data[:, 3]
theta = data[:, 4]
ux = data[:, 5]
uy = data[:, 6]

# Plot x-y trajectory
plt.figure(figsize=(8, 8))
plt.plot(x, y, 'b-', linewidth=1, alpha=0.7)
plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Particle Trajectory (Cylindrical Coordinates)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('trajectory_xy.png', dpi=150)
print("Saved trajectory_xy.png")
plt.show()
