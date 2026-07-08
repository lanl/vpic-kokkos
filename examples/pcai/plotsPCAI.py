import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

datadir = "../../build/data/"
nx = 64
nt = 300

pi = np.pi

xv = np.linspace(0,10.5,num=nx)
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
	# Load all quantities including electric field
	qs = ["Uiy","Uiz","aniso","By","Bz","Ey","Ez"]
	for q in qs:
		tmp = loadSlice(datadir,q,slice,nx,nt)
		Q[q] = tmp
	
# Original velocity plot
fig, (ax1,ax2) = plt.subplots(nrows=2)
im1 = ax1.pcolormesh(tv,xv,Q["Uiy"])
ax1.set_ylabel('$x/d_i$')
ax1.set_title('$U_{iy}(x,t)$')

gamma = 0.162

# Calculate RMS magnitudes
duy = np.sqrt(np.sum((Q["Uiy"])*(Q["Uiy"]),axis=0))
duz = np.sqrt(np.sum((Q["Uiz"])*(Q["Uiz"]),axis=0))
dby = np.sqrt(np.sum((Q["By"])*(Q["By"]),axis=0))
dbz = np.sqrt(np.sum((Q["Bz"])*(Q["Bz"]),axis=0))
dey = np.sqrt(np.sum((Q["Ey"])*(Q["Ey"]),axis=0))
dez = np.sqrt(np.sum((Q["Ez"])*(Q["Ez"]),axis=0))

aniso = np.mean(Q["aniso"],axis=0)

im2a = ax2.plot(tv,np.log10(duy),label='$\\delta|U_{iy}|$')
im2b = ax2.plot(tv,np.log10(duz),label='$\\delta|U_{iz}|$')
im2c = ax2.plot(tv,np.log10(0.019*np.exp(gamma*tv)),label='envelope',ls='--',color='black')  
ax2.set_xlabel('$t\\omega_{ci}$')
ax2.set_ylabel('$\\log_{10}(\\delta|U_i|)$')
ax2.legend()

plt.xlim([0, 60])
plt.ylim([-2, 1])
plt.tight_layout()
plt.savefig('pcai-1.png', dpi=300)

# Magnetic field plot
fig2, (ax3,ax4) = plt.subplots(nrows=2)
im3 = ax3.plot(tv,aniso)
im4a = ax4.plot(tv,np.log10(dby),label='$\\delta|B_y|$')
im4b = ax4.plot(tv,np.log10(dbz),label='$\\delta|B_z|$')
im4c = ax4.plot(tv,np.log10(0.012*np.exp(gamma*tv)),label='envelope',ls='--',color='black')

ax4.set_xlabel('$t\\omega_{ci}$')
ax3.set_ylabel('$P_\\perp/P_\\parallel$')
ax4.set_ylabel('$\\log_{10}(\\delta|B|)$')
ax4.legend()

plt.xlim([0, 60])
plt.ylim([-2, 1])
plt.tight_layout()
plt.savefig('pcai.png', dpi=300)

# NEW: Electric field plot (clean version - no fuzzy colormap)
fig3 = plt.figure(figsize=(8,5))
ax5 = fig3.add_subplot(111)

im5a = ax5.plot(tv,np.log10(dey),label='$\\delta|E_y|$',linewidth=2)
im5b = ax5.plot(tv,np.log10(dez),label='$\\delta|E_z|$',linewidth=2)
# Adjust envelope amplitude to match E field (tune as needed)
im5c = ax5.plot(tv,np.log10(0.015*np.exp(gamma*tv)),label='envelope',ls='--',color='black',linewidth=2)

ax5.set_xlabel('$t\\omega_{ci}$',fontsize=14)
ax5.set_ylabel('$\\log_{10}(\\delta|E|)$',fontsize=14)
ax5.set_title('Electric Field Growth Rate',fontsize=14)
ax5.legend(fontsize=12)
ax5.grid(alpha=0.3)

plt.xlim([0, 60])
plt.ylim([-2, 1])
plt.tight_layout()
plt.savefig('pcai-E.png', dpi=300)

plt.show()