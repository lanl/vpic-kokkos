import numpy as np
import matplotlib.pyplot as plt

# Data from output
r = [0.505, 0.504985, 0.504971, 0.504957, 0.504945, 0.504934, 0.504924, 0.504914,
     0.504906, 0.504899, 0.504893, 0.504887, 0.504883, 0.50488, 0.504878, 0.504876,
     0.504876, 0.504877, 0.504879, 0.504881, 0.504885, 0.50489, 0.504896, 0.504902,
     0.50491, 0.504919, 0.504928, 0.504939, 0.504951, 0.504963, 0.504977, 0.504992]

theta = [0.0314158, 0.0304196, 0.0294234, 0.0284271, 0.0274308, 0.0264344, 0.025438,
         0.0244416, 0.0234451, 0.0224487, 0.0214522, 0.0204557, 0.0194591, 0.0184626,
         0.0174661, 0.0164695, 0.015473, 0.0144764, 0.0134799, 0.0124833, 0.0114868,
         0.0104903, 0.00949378, 0.0084973, 0.00750083, 0.0065044, 0.00550799, 0.00451161,
         0.00351526, 0.00251895, 0.00152268, 0.00052645]

# Convert to Cartesian
x = np.array(r) * np.cos(np.array(theta)*10)
y = np.array(r) * np.sin(np.array(theta)*10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot trajectory in Cartesian
ax1.plot(x, y, 'b.-', markersize=3)
ax1.plot(x[0], y[0], 'go', markersize=8, label='Start')
ax1.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Particle Trajectory (Cartesian)')
ax1.axis('equal')
ax1.grid(True)
ax1.legend()

# Plot in cylindrical
ax2.plot(theta, r, 'b.-', markersize=3)
ax2.plot(theta[0], r[0], 'go', markersize=8, label='Start')
ax2.plot(theta[-1], r[-1], 'ro', markersize=8, label='End')
ax2.set_xlabel('theta (rad)')
ax2.set_ylabel('r')
ax2.set_title('Particle Trajectory (Cylindrical)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('/vast/home/cgraham/vpic-kokkos/test/integrated/cyclo/cyclo_trajectory.png', dpi=150)
print("Plot saved")
