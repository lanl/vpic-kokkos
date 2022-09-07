import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pywt

#datadir = "./data/"
datadirs = ["./data/","./NO-SMOOTH/data/"]
nx = 48#96
nt = 100

pi = np.pi

xv = np.linspace(0,16,num=nx)
tv = np.linspace(0,50,num=nt)
if (nx>1): dx = xv[1]-xv[0]
if (nt>1): dt = tv[1]-tv[0]

########## wavelet denoising function
def wclean(arr,wavn,alpha):
	cs = pywt.wavedecn(arr, wavn, mode='symmetric', level=None,axes=None)
	levs = len(cs)
	coef = np.concatenate(cs[0])
	for x in range(1,levs):
		for n in cs[x]:
			coefn = np.concatenate(cs[x][n])
			coef = np.concatenate((coef,coefn))	
	thr2 = alpha*np.sqrt(np.var(coef)*np.log(len(coef)))
	thr = alpha*2*thr2
	while (thr/thr2 > 1.05):
		thr = thr2;
		thr2 = alpha*np.sqrt(np.var(coef[np.abs(coef)<thr])*np.log(len(coef)))
	for x in range(1,levs):
		for n in cs[x]:
			inds = np.abs(cs[x][n]) < thr
			cs[x][n][inds] = 0
	return pywt.waverecn(cs, wavn, mode='symmetric', axes=None);
########## end wavelet denoising


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

fig, (ax1,ax2) = plt.subplots(nrows=2)
for run in datadirs:
	print(run)
	qs = ["ni","Ex","Uix"]
	for q in qs:
		tmp = loadSlice(run,q,0,nx,nt)
		Q[q] = tmp
	
	#bxw = wclean(arr=Q["den"],wavn="coif3",alpha=1)
		
#	fig, (ax1,ax2) = plt.subplots(nrows=2)
	im = ax1.pcolormesh(tv,xv,Q["ni"])
	dn = np.sqrt(np.sum((1-Q["ni"])*(1-Q["ni"]),axis=0))#/float(nx)
	im = ax2.plot(tv,np.log10(dn),label=run)
gamma = -0.093196
im = ax2.plot(tv[0:50],np.log10(0.08*np.exp(gamma*tv[0:50])))
ax1.set_ylabel('x/L')
ax2.set_ylabel('|dn|')
ax2.set_xlabel('t * C_s/L')
ax2.legend()
plt.show()


