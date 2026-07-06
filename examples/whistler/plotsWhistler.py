import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pywt

datadir = "../../build/data/"
nx = 48#96
ny = 48
nt = 100

pi = np.pi

xv = np.linspace(0,16,num=nx)
yv = np.linspace(0,16,num=ny)
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

#yv,xv = np.meshgrid(np.linspace(0,7.5*pi,num=ny),
#                np.linspace(0,5*pi,num=nx))

cmap = plt.get_cmap("Spectral")

Q = {}


for slice in range(0,100,5):
	qs = ["By","Ex"]
	for q in qs:
		tmp = loadSlice(datadir,q,slice,nx,ny)
		Q[q] = tmp
	
	#bxw = wclean(arr=Q["den"],wavn="coif3",alpha=1)
		
	fig, (ax1,ax2) = plt.subplots(nrows=2)
	im = ax1.pcolormesh(xv,yv,Q["By"])
	#im = ax1.pcolormesh(yv,xv,Q["ni"],cmap=cmap)
	fig.colorbar(im, ax=ax1)
	#im = ax1.plot(xv,Q["ni"][:,0])
	im2 = ax2.pcolormesh(yv,xv,Q["Ex"],cmap=cmap)
	fig.colorbar(im2, ax=ax2)    
	plt.show()

