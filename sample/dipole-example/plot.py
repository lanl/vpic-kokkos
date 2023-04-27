import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

datadir = "./data/"
nx = 800
ny = 1600


######### loadSlice function
def loadSlice(dir,q,sl,nx,ny):
	fstr = dir + q + ".gda"
	fd = open(fstr,"rb")
	fd.seek(4*sl*nx*ny,1)
	arr = np.fromfile(fd,dtype=np.float32,count=nx*ny)
	fd.close
	arr = np.reshape(arr,( ny, nx))
	#arr = arr[1:ny-1,1:nx-1]
	return arr
######### end loadSlice

cmap = plt.get_cmap("Spectral")

Q = {}

qs = ["ni","by"]

for slice in range(90,91):
	for q in qs:
		tmp = loadSlice(datadir,q,slice,nx,ny)
		Q[q] = tmp
	fig, (ax1,ax2) = plt.subplots(nrows=2)
	im = ax1.pcolormesh(Q["ni"],cmap=cmap)
	fig.colorbar(im, ax=ax1)
	im2 = ax2.pcolormesh(Q["by"],cmap=cmap)
	fig.colorbar(im, ax=ax2)    
	plt.show()


