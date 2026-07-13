import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import struct

######### loadinfo function
def loadinfo(dir):
	fstr = dir + "info"
	fd = open(fstr,"rb")
	infocontent = fd.read()
	fd.close
	arr = struct.unpack("fIIIfffff", infocontent[:36]) 
	infoarr=np.zeros(2);
	infoarr[0] = arr[1]
	infoarr[1] = arr[3]
	print(infoarr)
	return infoarr
######### end loadSlice


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

cmap = plt.get_cmap("Spectral_r")

Q = {}
qs = ["ni"]



dir = "./data/"

Lx=300
Lz=30
infoarr = loadinfo(dir)
nx = int(infoarr[0])
nz = int(infoarr[1])
xv = np.linspace(0,Lx,nx)
zv = np.linspace(0,Lz,nz)

cnt=0	
for slice in range(0,45,1):	
	for q in qs:
		tmp = loadSlice(dir,q,slice,nx,nz)
		Q[q] = tmp
	fig, (ax1) = plt.subplots(nrows=1)
	im = ax1.pcolormesh(Q["ni"],cmap=cmap)
	fig.colorbar(im, ax=ax1)    
	plt.show()