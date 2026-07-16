import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pywt

datadir = "../../build/data/"
nx = 48
ny = 48
nt = 100

pi = np.pi

# Grid parameters for stretched Cartesian (from your simulation)
betax = 2
betay = 2
betaz = 0.0

# Global domain bounds (from your simulation)
Lx = 32.0
Ly = 32.0
gx0, gx1 = -0.5*Lx, 0.5*Lx  # [-16, 16]
gy0, gy1 = -0.5*Ly, 0.5*Ly  # [-16, 16]

def stretched_coordinate(xi_normalized, beta, g0, g1):
    """
    Transform normalized computational coordinate [0,1] to physical coordinate.
    Matches the C++ init_stretched_cartesian_grid transformation.
    """
    if beta > 1e-10:
        # Tanh stretching
        x_stretched = np.tanh(beta * (xi_normalized - 0.5)) / np.tanh(beta * 0.5)
    else:
        # No stretching (uniform)
        x_stretched = 2.0 * xi_normalized - 1.0
    
    # Map from [-1,1] to [g0,g1]
    x_physical = g0 + (g1 - g0) * (x_stretched + 1.0) * 0.5
    return x_physical

# Create physical coordinate arrays for CELL EDGES
# Edge i is at computational coordinate i/nx (no +0.5 offset for edges!)
xi_edges = np.arange(nx + 1) / float(nx)  # 0/48, 1/48, 2/48, ..., 48/48
yi_edges = np.arange(ny + 1) / float(ny)

xv = stretched_coordinate(xi_edges, betax, gx0, gx1)
yv = stretched_coordinate(yi_edges, betay, gy0, gy1)

tv = np.linspace(0,50,num=nt)
if (nx>1): dx = xv[1]-xv[0]
if (nt>1): dt = tv[1]-tv[0]

######### loadSlice function
def loadSlice(dir,q,sl,nx,ny):
	fstr = dir + q + ".gda"
	fd = open(fstr,"rb")
	fd.seek(4*sl*nx*ny,1)
	arr = np.fromfile(fd,dtype=np.float32,count=nx*ny)
	fd.close
	arr = np.reshape(arr,( ny, nx))
	arr = np.transpose(arr)
	return arr
######### end loadSlice

cmap = plt.get_cmap("Spectral")

Q = {}


for slice in range(0,100,5):
	qs = ["By","Ex"]
	for q in qs:
		tmp = loadSlice(datadir,q,slice,nx,ny)
		Q[q] = tmp
	
	fig, (ax1,ax2) = plt.subplots(nrows=2)
	im = ax1.pcolormesh(xv,yv,Q["By"])
	fig.colorbar(im, ax=ax1)
	im2 = ax2.pcolormesh(yv,xv,Q["Ex"],cmap=cmap)
	fig.colorbar(im2, ax=ax2)    
	plt.show()
	plt.savefig('fig.png', dpi=300)