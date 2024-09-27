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

#-------------------------------------------------------------------------------
# Read in dump files and iterate over list
#-------------------------------------------------------------------------------

opointpos = np.zeros(100)
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

    minindex1 = np.argmin(Ay[0:int(nx/2),0,int(nz/2)])
    minindex2 = np.argmin(Ay[int(nx/2):int(nx),0,int(nz/2)])

    opointpos[it_flt] = 0.5*(abs(xvals[minindex1]) + abs(xvals[minindex2+int(nx/2)]))

plt.plot(opointpos)
plt.title('O-point position')
plt.xlabel('t*w_ci')
plt.ylabel('x/d_i')

plt.show()


#fig = plt.figure(figsize=(9,9)

