#!/usr/projects/hpcsoft/toss2/common/anaconda/2.1.0-python-2.7/bin/python
#-------------------------------------------------------------------------------
import numpy as np
import ReadFile
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter 

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
nx, ny, nz, xvals, yvals, zvals = ReadFile.read_griddata()
dx = xvals[1]-xvals[0]
dz = zvals[1]-zvals[0]

print('nx,ny,nz=',nx,ny,nz)

dt=1.0
#-------------------------------------------------------------------------------
# Read in dump files and iterate over list
#-------------------------------------------------------------------------------

ayval = np.zeros(100)
#minpos1=np.zeros(100)
#minpos2=np.zeros(100)
for it_flt in range(100):
#    it_des = int(it_flt) 
#    print it_des
    #---------------------------------------------------------------------------
    # Read in non-averaged magnetic field terms 
    #---------------------------------------------------------------------------
    Ay = ReadFile.read_xyz(nx,ny,nz,it_flt,'Ay')
    
    #    Ay = gaussian_filter(Ay, sigma=sigval)
    ayval[it_flt] = Ay[int(nx/2),0,int(nz/2)]

print(ayval)
    
rate=np.zeros(100)
rate[0]=0
for it_flt in range(100-1):
    rate[it_flt+1] = (ayval[it_flt+1] - ayval[it_flt])/dt

#plt.plot(rate)
ratesm4 = gaussian_filter(rate,sigma=4)
plt.plot(ratesm4)
plt.title('Reconnection rate')
plt.ylabel('dA/dt (B_0 v_A)')
plt.xlabel('t*w_ci')

plt.show()
#fig = plt.figure(figsize=(9,9)

