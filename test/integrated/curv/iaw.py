import numpy as np, matplotlib.pyplot as plt
d=np.loadtxt('../../../build/wave.txt')
plt.semilogy(d[:,0], d[:,1])
plt.xlabel('Time')
plt.ylabel('Ex')
plt.title('iaw')
plt.grid()
plt.savefig('landau.png', dpi=300)
plt.show()