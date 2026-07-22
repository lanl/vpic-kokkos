#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../../../build/convergence_results.txt')
dt_array = data[:, 0]
err_array = data[:, 1]

plt.figure(figsize=(10, 7))

plt.loglog(dt_array, err_array)

mid_idx = len(dt_array) // 2
dt_ref = dt_array[mid_idx]
err_ref = err_array[mid_idx]
C1 = err_ref / dt_ref
plt.loglog(dt_array, C1 * dt_array, '--', linewidth=2, label='dt', color='green')

C2 = err_ref / dt_ref**2
plt.loglog(dt_array, C2 * dt_array**2, '--', linewidth=2, label='dt^2', color='red')

plt.xlabel('dt',)
plt.ylabel('Rerr')
plt.title(f'Cyclo convergence using cyl grid and particle push', fontsize=16)
plt.legend()

plt.savefig('convergence_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to convergence_plot.png")

if len(dt_array) >= 3:
    log_dt = np.log(dt_array)
    log_err = np.log(err_array)
    slope = (log_err[-1] - log_err[0]) / (log_dt[-1] - log_dt[0])

print("\n{:>12s} {:>15s}".format("dt", "Rerr"))
print("-" * 30)
for dt, err in zip(dt_array, err_array):
    print(f"{dt:12.6f} {err:15.6e}")