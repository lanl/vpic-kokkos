import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pywt

datadir = "./data/"
nx = 64
nt = 300

pi = np.pi

xv = np.linspace(0,10.5,num=nx)
#tv = np.linspace(0,100,num=nt)
tv = np.linspace(0,60,num=nt)
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
for slice in range(0,1):
	qs = ["Uiy","Uiz","aniso","By","Bz"]
	for q in qs:
		tmp = loadSlice(datadir,q,slice,nx,nt)
		Q[q] = tmp
	
fig, (ax1,ax2) = plt.subplots(nrows=2)
im1 = ax1.pcolormesh(tv,xv,Q["Uiy"])
#ax1.set_xlabel('t*w_ci')
ax1.set_ylabel('d|Ui_y|')

gamma = 0.162
#gamma = 0.0785

duy = np.sqrt(np.sum((Q["Uiy"])*(Q["Uiy"]),axis=0))
duz = np.sqrt(np.sum((Q["Uiz"])*(Q["Uiz"]),axis=0))
dby = np.sqrt(np.sum((Q["By"])*(Q["By"]),axis=0))
dbz = np.sqrt(np.sum((Q["Bz"])*(Q["Bz"]),axis=0))

aniso=np.mean(Q["aniso"],axis=0)
#print(aniso)
#print(dn)
im2a = ax2.plot(tv,np.log10(duy),label='d|Uiy|')
im2b = ax2.plot(tv,np.log10(duz),label='d|Uiz|')
#im = ax2.plot(tv,np.log10(0.009*np.exp(gamma*tv)))
im2c = ax2.plot(tv,np.log10(0.019*np.exp(gamma*tv)))  
ax2.set_xlabel('t*w_ci')
ax2.set_ylabel('d|Ui|')
ax2.legend()

#plt.xlim([0, 80])
plt.xlim([0, 60])
plt.ylim([-2, 1])



fig2, (ax3,ax4) = plt.subplots(nrows=2)
im3 = ax3.plot(tv,aniso)
im4a = ax4.plot(tv,np.log10(dby),label='d|By|')
im4b = ax4.plot(tv,np.log10(dbz),label='d|Bz|')
im4c = ax4.plot(tv,np.log10(0.012*np.exp(gamma*tv)))

ax4.set_xlabel('t*w_ci')
ax3.set_ylabel('P_perp/P_par')
ax4.set_ylabel('d|B|')
ax4.legend()

plt.xlim([0, 60])
plt.ylim([-2, 1])

plt.show()


