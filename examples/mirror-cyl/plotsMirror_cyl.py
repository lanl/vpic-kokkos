import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import struct
import sys


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

def loadFieldLines(dir, sl, nr, nz):
	"""
	Load magnetic field lines (Ay) from Fortran output
	sl: time slice/record number
	nr: number of radial cells (was nx in Fortran)
	nz: number of axial cells (was nz in Fortran)
	"""
	fstr = dir + "Ay_int.gda"
	
	# Check if file exists
	if not os.path.exists(fstr):
		print(f"Warning: {fstr} not found. Run Fortran code first to generate field lines.")
		return None
	
	try:
		fd = open(fstr, "rb")
		fd.seek(4*sl*nr*nz, 1)
		arr = np.fromfile(fd, dtype=np.float32, count=nr*nz)
		fd.close()
		
		if len(arr) != nr*nz:
			print(f"Warning: Expected {nr*nz} values, got {len(arr)}")
			return None
			
		arr = np.reshape(arr, (nz, nr))  # Data stored as (z, r)
		arr = np.transpose(arr)          # Transpose to (r, z)
		return arr
	except Exception as e:
		print(f"Error loading field lines: {e}")
		return None

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
slice = num_slices - 1
if len(sys.argv) > 1:
	slice = int(sys.argv[1])
for q in qs:
	tmp = loadSlice(dir, q, slice, nr, nz)
	Q[q] = tmp

# cartesian starts at ni = 0.06

fig, (ax1) = plt.subplots(nrows=1, figsize=(10, 4))

# Plot in r-z plane (cylindrical cross-section)
im = ax1.pcolormesh(Z, R, Q["ni"], cmap=cmap, shading='auto')
Ay = loadFieldLines(dir, slice, nr, nz)

# Main plot with field lines
fig, (ax1) = plt.subplots(nrows=1, figsize=(12, 5))

# Plot density in r-z plane (cylindrical cross-section)
im = ax1.pcolormesh(Z, R, Q["ni"], cmap=cmap, shading='auto')

# Overlay field lines if available
if Ay is not None:
	# Number of field lines to draw
	num_lines = 20
	
	# Draw field lines as contours of constant Ay
	contours = ax1.contour(Z, R, Ay, levels=num_lines, colors='black', 
	                       linewidths=1.0, alpha=0.6, linestyles='solid')
	
	# Optional: add labels to some field lines
	# ax1.clabel(contours, inline=True, fontsize=8, fmt='%1.2f')
	
	print(f"Drew {num_lines} field lines")
	print(f"Ay range: [{np.min(Ay):.3e}, {np.max(Ay):.3e}]")
else:
	print("Field lines not plotted - Ay data not available")

ax1.set_xlabel('z')
ax1.set_ylabel('r/x')
ax1.set_title(f'density')
ax1.set_aspect('equal')  # Equal aspect ratio

fig.colorbar(im, ax=ax1, label='ni')

fig.tight_layout()
fig.savefig('plot.png', dpi=300)
# plt.show()


# Optional: Plot as a "full" cylindrical view (mirror top and bottom)
def plot_cylindrical_full(Q, R, Z, quantity='ni'):
	"""
	Plot full cylindrical view by mirroring around axis
	"""
	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Top half (positive r)
	im1 = ax.pcolormesh(Z, R, Q[quantity], cmap=cmap, shading='auto', vmin=0, vmax=1.0)
	
	# Bottom half (negative r, mirrored)
	im2 = ax.pcolormesh(Z, -R, Q[quantity], cmap=cmap, shading='auto', vmin=0, vmax=1.0)
	
	ax.set_xlabel('z (axial position)')
	ax.set_ylabel('r (radial position)')
	ax.set_title(f'{quantity} - Full Cylindrical View')
	ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
	ax.set_aspect('equal')
	
	cbar = fig.colorbar(im1, ax=ax, label=quantity)
	cbar.set_clim(0, 1)
	plt.tight_layout()
	plt.show()

# Uncomment to use full cylindrical plot:
# plot_cylindrical_full(Q, R, Z, 'ni')