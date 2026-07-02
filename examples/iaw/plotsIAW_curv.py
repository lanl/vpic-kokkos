import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pywt

datadir = "./data/"
# curvilinear deck: wave is along z (nz=48), transverse ny=5, nx=1
nx = 1
ny = 5
nz = 48
nt = 100

pi = np.pi

zv = np.linspace(0,16,num=nz)
tv = np.linspace(0,50,num=nt)
if (nz>1): dz = zv[1]-zv[0]
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
# gda records are nx*ny*nz floats per time, x fastest then y then z.
# Extract the z-profile (the wave direction) at fixed x=0, averaged over y.
def loadSlice(dir,q,nx,ny,nz,nt):
	fstr = dir + q + ".gda"
	fd = open(fstr,"rb")
	ncell = nx*ny*nz
	arr = np.fromfile(fd,dtype=np.float32,count=ncell*nt)
	fd.close
	arr = np.reshape(arr,(nt, nz, ny, nx))   # z slowest, x fastest
	arr = arr.mean(axis=(2,3))               # average over transverse y,x -> (nt, nz)
	arr = np.transpose(arr)                  # -> (nz, nt)
	return arr
######### end loadSlice

cmap = plt.get_cmap("Spectral")

Q = {}


for slice in range(0,1):
	qs = ["ni","Ez","Uiz"]
	for q in qs:
		tmp = loadSlice(datadir,q,nx,ny,nz,nt)
		Q[q] = tmp

	#bxw = wclean(arr=Q["den"],wavn="coif3",alpha=1)

	fig, (ax1,ax2) = plt.subplots(nrows=2)
	im = ax1.pcolormesh(tv,zv,Q["ni"])

print(Q["ni"])

gamma = -0.093196

dn = np.sqrt(np.sum((1-Q["ni"])*(1-Q["ni"]),axis=0))#/float(nz)
im = ax2.plot(tv,np.log10(dn))
im = ax2.plot(tv[0:50],np.log10(0.08*np.exp(gamma*tv[0:50])))
ax1.set_ylabel('z/L')
ax2.set_ylabel('|dn|')
ax2.set_xlabel('t * C_s/L')
plt.savefig('fig_curv.png', dpi=300)
plt.show()
