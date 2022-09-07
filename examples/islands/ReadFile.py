# ReadFile module

#-------------------------------------------------------------------------------
def read_griddata():
# Read in the basic griddata for a run
# Returns: 
#    nx, ny, nz (the size of the grid)
#    xvals, yvals, zvals (linear arrays of positions on the grid)
#-------------------------------------------------------------------------------
    import struct 
    import numpy as np
    f=open('./data/info','rb')
    f.seek(0)
    temp=f.read(4)
    dval=struct.unpack("<l",temp)
    temp=f.read(dval[0])
    linein=struct.unpack("<lll",temp)
    nx = linein[0]
    ny = linein[1]
    nz = linein[2]
    temp=f.read(4)
    #next line
    temp=f.read(4) 
    dval=struct.unpack("<l",temp)
    temp=f.read(dval[0])
    linein=struct.unpack("<fff",temp)
    Lx = linein[0]
    Ly = linein[1]
    Lz = linein[2]
    temp=f.read(4) 

#    xvals = np.linspace(-xmax/2.0, xmax*(float(nx-1)/float(nx))-xmax/2.0, nx)
#    yvals = np.linspace(-ymax/2.0, ymax*(float(ny-1)/float(ny))-ymax/2.0, ny)
#    zvals = np.linspace(-zmax/2.0, zmax*(float(nz-1)/float(nz))-zmax/2.0, nz)
#    xvals = np.linspace(-xmax/2.0, xmax/2.0, nx)
#    yvals = np.linspace(-ymax/2.0, ymax/2.0, ny)
#    zvals = np.linspace(-zmax/2.0, zmax/2.0, nz)
    xvals = np.zeros([nx],dtype='f')
    yvals = np.zeros([ny],dtype='f')
    zvals = np.zeros([nz],dtype='f')
    dx = Lx/nx
    dy = Ly/ny
    dz = Lz/nz
    for ix in range(nx):
        xvals[ix] = (ix+0.5)*dx - Lx/2.0 
    for iy in range(ny):
        yvals[iy] = (iy+0.5)*dy - Ly/2.0 
    for iz in range(nz):
        zvals[iz] = (iz+0.5)*dz - Lz/2.0 

    return nx, ny, nz, xvals, yvals, zvals

#-------------------------------------------------------------------------------
def read_xyzt(nx, ny, nz, var):
# Read in a 4D array consisting of [x, y, z, t] data 
# The dimensions in x, y, and z are needed (nx, ny, nz)
# Returns:
#     datavals[x,y,z,t]
#-------------------------------------------------------------------------------
    import struct 
    import numpy as np

    f=open('./data/'+var+'.gda','rb')
    f.seek(0,2)
    file_size=f.tell()
    nt = file_size/(nx*ny*nz*4)

    datavals = np.zeros([nx,ny,nz,nt],dtype='f')
    
    f.seek(0)
    for jt in range(nt):
        for jz in range(nz):
            for jy in range(ny):
                for jx in range(nx):
                    temp = f.read(4)
                    fval = struct.unpack("<f",temp)
                    datavals[jx,jy,jz,jt]=fval[0]

    return datavals

#-------------------------------------------------------------------------------
def read_xyz(nx, ny, nz, it_des, var):
# Read in a 3D array over x, y, z at a given time index specified by it_des
# Needs to know the dimensions nx, ny, nz
# Returns:
#    datavals[x,y,z] at the desired time index
# If the desired time index is not present, returns 0
#-------------------------------------------------------------------------------
    import struct 
    import numpy as np

    f=open('./data/'+var+'.gda','rb')
    f.seek(0,2)
    file_size=f.tell()
    nt = file_size/(nx*ny*nz*4)

    if it_des>=nt:
        it_des = nt-1
    elif it_des<=0:
        it_des = 0
    datavals = np.zeros([nx,ny,nz],dtype='f')

    f.seek(nx*ny*nz*it_des*4)
    for jz in range(nz):
        for jy in range(ny):
            for jx in range(nx):
                temp = f.read(4)
                fval = struct.unpack("<f",temp)
                datavals[jx,jy,jz]=fval[0]

    return datavals

#-------------------------------------------------------------------------------
def read_xyz_sep(nx, ny, nz, it_des, var):
# Read in a 3D array over x, y, z at a given dump time specified by it_des
# Needs to know the dimensions nx, ny, nz
# Returns:
#    datavals[x,y,z] at the desired time index
# If the desired time index is not present, returns 0
#-------------------------------------------------------------------------------
    import struct 
    import numpy as np

    f=open('./data/'+var+'_'+str(it_des)+'.gda','rb')
    f.seek(0,2)
    file_size=f.tell()
    nt = file_size/(nx*ny*nz*4)

    datavals = np.zeros([nx,ny,nz],dtype='f')

    f.seek(0)
    for jz in range(nz):
        for jy in range(ny):
            for jx in range(nx):
                temp = f.read(4)
                fval = struct.unpack("<f",temp)
                datavals[jx,jy,jz]=fval[0]

    return datavals

#-------------------------------------------------------------------------------
def read_xy(nx, ny, nz, iz_des, it_des, var):
# Read in a 2D array over x, z at a given dump time specified by it_des
# and for a given y-plane specified by iy_des
# Needs to know the dimensions nx, ny, nz
# Returns:
#    datavals[x,z] at the desired time index
# If the desired time index is not present, returns 0
#-------------------------------------------------------------------------------
    import struct 
    import numpy as np

    f=open('./data/'+var+'_'+str(it_des)+'.gda','rb')
    datavals = np.zeros([nx,ny],dtype='f')
    f.seek((iz_des*nx*ny)*4)
    for jy in range(ny):
        for jx in range(nx):
            temp = f.read(4)
            fval = struct.unpack("<f",temp)
            datavals[jx,jy]=fval[0]

    return datavals

#-------------------------------------------------------------------------------
def read_xz(nx, ny, nz, iy_des, it_des, var):
# Read in a 2D array over x, z at a given dump time specified by it_des
# and for a given y-plane specified by iy_des
# Needs to know the dimensions nx, ny, nz
# Returns:
#    datavals[x,z] at the desired time index
# If the desired time index is not present, returns 0
#-------------------------------------------------------------------------------
    import struct 
    import numpy as np

    f=open('./data/'+var+'_'+str(it_des)+'.gda','rb')
    datavals = np.zeros([nx,nz],dtype='f')
    f.seek(0)
    for jz in range(nz):
        f.seek((jz*nx*ny+iy_des*nx)*4)
        for jx in range(nx):
            temp = f.read(4)
            fval = struct.unpack("<f",temp)
            datavals[jx,jz]=fval[0]

    return datavals



#-------------------------------------------------------------------------------
def read_x(nx, ny, nz, iy_des, iz_des, it_des, var):
# Read in a 1D array over x at a given dump time specified by it_des
# and for a given y and z location specified by iy_des and iz_des
# Needs to know the dimensions nx, ny, nz
# Returns:
#    datavals[x] at the desired time index
# If the desired time index is not present, returns 0
#-------------------------------------------------------------------------------
    import struct 
    import numpy as np

    f=open('./data/'+var+'_'+str(it_des)+'.gda','rb')
    datavals = np.zeros([nx],dtype='f')
    f.seek(0)
    f.seek((iz_des*nx*ny+iy_des*nx)*4)
    for jx in range(nx):
        temp = f.read(4)
        fval = struct.unpack("<f",temp)
        datavals[jx]=fval[0]

    return datavals

#-------------------------------------------------------------------------------
def read_kymagen(ny):
# Read in magnetic energies in '../dis/ky_magen.gda' 
# Returns:
#    ky_en[jy,jt]
#-------------------------------------------------------------------------------
    import struct 
    import numpy as np

    f=open('../dis/ky_magen.gda','rb')
    f.seek(0,2)
    file_size=f.tell()
    nt = file_size/(ny*4)

    ky_en = np.zeros([ny,nt],dtype='f')

    f.seek(0)
    for jt in range(nt):
        for jy in range(ny):
            temp = f.read(4)
            fval = struct.unpack("<f",temp)
            ky_en[jy,jt]=fval[0]

    return ky_en

def read_ky_ne(ny):
# Read in amplitude and phase of perturbed ne '../dis/ky_ne_(amp/phs).gda' 
# Returns:
#    ky_ne_amp[jy,jt]
#    ky_ne_phs[jy,jt]
#-------------------------------------------------------------------------------
    import struct 
    import numpy as np

    f=open('../dis/ky_ne_amp.gda','rb')
    f.seek(0,2)
    file_size=f.tell()
    nt = file_size/(ny*4)
    ky_ne_amp = np.zeros([ny,nt],dtype='f')
    f.seek(0)
    for jt in range(nt):
        for jy in range(ny):
            temp = f.read(4)
            fval = struct.unpack("<f",temp)
            ky_ne_amp[jy,jt]=fval[0]

    f=open('../dis/ky_ne_phs.gda','rb')
    f.seek(0,2)
    file_size=f.tell()
    nt = file_size/(ny*4)
    ky_ne_phs = np.zeros([ny,nt],dtype='f')
    f.seek(0)
    for jt in range(nt):
        for jy in range(ny):
            temp = f.read(4)
            fval = struct.unpack("<f",temp)
            ky_ne_phs[jy,jt]=fval[0]


    return ky_ne_amp, ky_ne_phs

