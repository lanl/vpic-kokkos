import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import struct

######### loadinfo function - UPDATED FOR CYLINDRICAL
def loadinfo(dir):
    fstr = dir + "info.bin"
    fd = open(fstr,"rb")
    infocontent = fd.read()
    fd.close()
    
    # Updated binary format for cylindrical: topology_r, topology_z, Lr, Lz, nr, nz, dt, mime, mi, vthe, vthi, status_interval
    # Old format had 9 values for 3D: topology_x, topology_y, topology_z, Lx, Ly, Lz, nx, ny, nz
    # New format has 6 values for 2D cylindrical: topology_r, topology_z, Lr, Lz, nr, nz
    
    arr = struct.unpack("dddddd", infocontent[:48])  # 6 doubles (8 bytes each = 48 bytes)
    
    infoarr = np.zeros(4)
    infoarr[0] = arr[4]  # nr (number of cells in r)
    infoarr[1] = arr[5]  # nz (number of cells in z)
    infoarr[2] = arr[2]  # Lr (radial extent)
    infoarr[3] = arr[3]  # Lz (axial extent)
    
    print("Grid dimensions (nr x nz):", int(infoarr[0]), "x", int(infoarr[1]))
    print("Box size (Lr x Lz):", infoarr[2], "x", infoarr[3])
    
    return infoarr
######### end loadinfo


######### loadSlice function - UPDATED FOR CYLINDRICAL
def loadSlice(dir, q, sl, nr, nz):
    """
    Load a slice of data for cylindrical (R-Z) geometry
    Note: Data is now nr x nz (radial x axial)
    """
    fstr = dir + q + ".gda"
    fd = open(fstr,"rb")
    fd.seek(4*sl*nr*nz, 1)
    arr = np.fromfile(fd, dtype=np.float32, count=nr*nz)
    fd.close()
    arr = np.reshape(arr, (nz, nr))  # nz is second dimension
    arr = np.transpose(arr)
    return arr
######### end loadSlice

cmap = plt.get_cmap("Spectral_r")

Q = {}
qs = ["ni"]

dir = "../../build/data/"

# Load grid info (returns [nr, nz, Lr, Lz])
infoarr = loadinfo(dir)
nr = int(infoarr[0])
nz = int(infoarr[1])
Lr = infoarr[2]
Lz = infoarr[3]

# Create coordinate vectors for cylindrical geometry
rv = np.linspace(0, Lr, nr)   # Radial coordinate (0 to Lr)
zv = np.linspace(-0.5*Lz, 0.5*Lz, nz)  # Axial coordinate (centered at 0)

cnt = 0	
for slice in range(0, 45, 1):	
    for q in qs:
        tmp = loadSlice(dir, q, slice, nr, nz)
        Q[q] = tmp
    
    fig, (ax1) = plt.subplots(nrows=1, figsize=(10, 6))
    
    # Create meshgrid for proper plotting
    R, Z = np.meshgrid(rv, zv, indexing='ij')
    
    # Plot with proper axis labels
    im = ax1.pcolormesh(Z, R, Q["ni"], cmap=cmap, shading='auto')
    ax1.set_xlabel('Z (axial)', fontsize=12)
    ax1.set_ylabel('R (radial)', fontsize=12)
    ax1.set_title(f'Ion Density - Cylindrical (R-Z), Slice {slice}', fontsize=14)
    fig.colorbar(im, ax=ax1, label='Ion Density')
    
    # Optional: Add axis line at r=0
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()