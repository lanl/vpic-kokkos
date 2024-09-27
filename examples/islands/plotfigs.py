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
	arr = struct.unpack("fIIIffffff", infocontent[:40]) 
	infoarr=np.zeros(6);
	infoarr[0] = arr[1]
	infoarr[1] = arr[2]
	infoarr[2] = arr[3]
	infoarr[3] = arr[6]
	infoarr[4] = arr[7]
	infoarr[5] = arr[8]
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


######### Make a gif
def makeGIF(imdir, basename, slicenums, imageext):
    images = [(imdir + basename + '_' + str(index) + imageext) for index in slicenums]
    filename = imdir+'../'+basename+'.gif'
    with open(imdir + basename+'_list.txt','w') as fil:
        for item in images:
            fil.write("%s\n" % item)
    os.chdir(imdir)
    os.system('convert @'+basename+'_list.txt '+filename)
########

cmap = plt.get_cmap("Spectral")

Q = {}

qs = ["ni","Ay"]

dir = "./data/"

infoarr = loadinfo(dir)
nx = int(infoarr[0])
nz = int(infoarr[2])
Lx = int(infoarr[3])
Lz = int(infoarr[5])
#Rmin = int(infoarr[6])

xv = np.linspace(0,Lx,nx)-Lx/2.0
zv = np.linspace(0,Lz,nz)-Lz/2.0

cnt=0	
for slice in range(0,200,1):	
        for q in qs:
                tmp = loadSlice(dir,q,slice,nx,nz)
                Q[q] = tmp
                cnt=cnt+1

        fig, (ax1) = plt.subplots(nrows=1)
        im = ax1.pcolor(xv,zv,np.transpose(Q["ni"]),shading="nearest")    
        im2 = ax1.contour(xv,zv,np.transpose(Q["Ay"]),levels=20,colors='white')
        #im = ax2.pcolor(zv,xv,Q["Ay"],shading="nearest")    
        #plt.show()
        plt.title(('slice='+str(slice)))
        plt.savefig(('figs/ni_Ay_'+str(slice).zfill(3)+'.png'))
        plt.close()

