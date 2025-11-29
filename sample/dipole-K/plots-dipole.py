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
	print(arr)
	infoarr=np.zeros(4);
	infoarr[0] = arr[1]
	infoarr[1] = arr[3]
	infoarr[2] = arr[6]
	infoarr[3] = arr[8]
	#print(infoarr)
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
	#arr = np.transpose(arr)
	return arr
######### end loadSlice


cmap = plt.get_cmap("Spectral_r")

Q = {}
qs = ["ni","uix","bx","Ay_int"]


dir = "./data/"
infoarr = loadinfo(dir)
nx = int(infoarr[0])
nz = int(infoarr[1])
Lx = infoarr[2]
Lz = infoarr[3]
xv1 = np.linspace(0,Lx,nx+1)
zv1 = np.linspace(0,Lz,nz+1)
xv = np.linspace(0,Lx,nx)
zv = np.linspace(0,Lz,nz)
zv = zv - np.mean(zv)
zv1 = zv1 - np.mean(zv1)
levs = np.linspace(-10500,10000,200)

cnt=0	
for slice in range(30,100,4):	
	for q in qs:
		tmp = loadSlice(dir,q,slice,nx,nz)
		Q[q] = tmp
		cnt=cnt+1
  
	fig, (ax1) = plt.subplots(nrows=1)
	im = ax1.pcolormesh(xv1,zv1,Q["ni"],cmap=cmap)
	plt.rcParams['contour.negative_linestyle'] = 'solid'
	plt.colorbar(im)
	im = ax1.contour(xv,zv,Q["Ay_int"],levs,colors='k')
	#im = ax2.pcolormesh(xv,zv,Q["uix"],cmap=cmap)
	#im = ax3.pcolormesh(xv,zv,Q["bx"],cmap=cmap)
	plt.title('Density')

	plt.show()


