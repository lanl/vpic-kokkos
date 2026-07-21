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
	arr = struct.unpack("fIIIfffff", infocontent[:36]) 
	infoarr=np.zeros(2);
	infoarr[0] = arr[1]  # nr (was nx)
	infoarr[1] = arr[3]  # nz (was ny, but now nz)
	print("Grid: nr =", infoarr[0], ", nz =", infoarr[1])
	return infoarr
######### end loadinfo


######### loadSlice function  
def loadSlice(dir, q, sl, nr, nz):
	"""
	Load a slice from cylindrical data
	sl: theta slice index (usually 0 for axisymmetric)
	nr: number of radial cells
	nz: number of axial cells
	"""
	fstr = dir + q + ".gda"
	fd = open(fstr,"rb")
	fd.seek(4*sl*nr*nz, 1)
	arr = np.fromfile(fd, dtype=np.float32, count=nr*nz)
	fd.close
	arr = np.reshape(arr, (nz, nr))  # Data stored as (z, r)
	arr = np.transpose(arr)          # Transpose to (r, z)
	return arr
######### end loadSlice

cmap = plt.get_cmap("Spectral_r")

Q = {}
qs = ["ni"]

dir = "../../build/data/"

# CYLINDRICAL coordinates now: (r, theta, z)
Lr = 36      # Radial extent (was Lz)
Lz = 360     # Axial extent (was Lx)

infoarr = loadinfo(dir)
nr = int(infoarr[0])  # Radial cells
nz = int(infoarr[1])  # Axial cells

# Create coordinate arrays
rv = np.linspace(0, Lr, nr)  # Radial coordinates
zv = np.linspace(-Lz/2, Lz/2, nz)  # Axial coordinates (centered)

# Create meshgrid for proper cylindrical plotting
R, Z = np.meshgrid(rv, zv, indexing='ij')

def get_num_slices(dir, q, nr, nz):
    """
    Determine the number of theta slices available in the file
    """
    fstr = dir + q + ".gda"
    file_size = os.path.getsize(fstr)
    bytes_per_slice = nr * nz * 4  # 4 bytes per float32
    num_slices = file_size // bytes_per_slice
    return num_slices

num_slices = get_num_slices(dir, "ni", nr, nz)
print(f"Number of available slices: {num_slices}")

cnt = 0	
for slice in range(0, num_slices)[::-1]:  # Usually just slice=0 for axisymmetric
	for q in qs:
		tmp = loadSlice(dir, q, slice, nr, nz)
		Q[q] = tmp
	
	fig, (ax1) = plt.subplots(nrows=1, figsize=(10, 4))
	
	# Plot in r-z plane (cylindrical cross-section)
	im = ax1.pcolormesh(Z, R, Q["ni"], cmap=cmap, shading='auto')
	
	ax1.set_xlabel('z')
	ax1.set_ylabel('r')
	ax1.set_title(f'Ion Density - Cylindrical (r-z plane)')
	ax1.set_aspect('equal')  # Equal aspect ratio
	
	fig.colorbar(im, ax=ax1, label='ni')
	
	plt.tight_layout()
	plt.show()


# Optional: Plot as a "full" cylindrical view (mirror top and bottom)
def plot_cylindrical_full(Q, R, Z, quantity='ni'):
	"""
	Plot full cylindrical view by mirroring around axis
	"""
	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Top half (positive r)
	im1 = ax.pcolormesh(Z, R, Q[quantity], cmap=cmap, shading='auto')
	
	# Bottom half (negative r, mirrored)
	im2 = ax.pcolormesh(Z, -R, Q[quantity], cmap=cmap, shading='auto')
	
	ax.set_xlabel('z (axial position)')
	ax.set_ylabel('r (radial position)')
	ax.set_title(f'{quantity} - Full Cylindrical View')
	ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
	ax.set_aspect('equal')
	
	fig.colorbar(im1, ax=ax, label=quantity)
	plt.tight_layout()
	plt.show()

# Uncomment to use full cylindrical plot:
# plot_cylindrical_full(Q, R, Z, 'ni')