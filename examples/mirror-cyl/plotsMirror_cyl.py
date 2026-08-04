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
qs = ["ni", "bx", "by", "bz", "ex", "ey", "ez", "pe"]  # All quantities we need

dir = "../../build/data/"

# CYLINDRICAL coordinates now: (r, theta, z)
Lr = 36      # Radial extent (was Lz)
Lz = 360     # Axial extent (was Lx)

infoarr = loadinfo(dir)
nr = int(infoarr[0])  # Radial cells
nz = int(infoarr[1])  # Axial cells

# Create coordinate arrays
rv = np.linspace(0, Lr, nr)  # Radial coordinates  np.linspace(0.1*Lr, 0.5*Lr, nr)
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

# Load all quantities
for q in qs:
	tmp = loadSlice(dir, q, slice, nr, nz)
	Q[q] = tmp
	print(f"{q} range: [{np.min(Q[q]):.3e}, {np.max(Q[q]):.3e}]")

# Load field lines
Ay = loadFieldLines(dir, slice, nr, nz)

# ============================================================
# FIGURE 1: Ion Density with Field Lines
# ============================================================
fig1, ax1 = plt.subplots(figsize=(12, 5))

im1 = ax1.pcolormesh(Z, R, Q["ni"], cmap=cmap, shading='auto')
# ax1.set_facecolor('black')


# Overlay field lines if available
if Ay is not None:
	num_lines = 20
	contours = ax1.contour(Z, R, Ay, levels=num_lines, colors='black', 
	                       linewidths=0.1, linestyles='solid')
	print(f"Drew {num_lines} field lines on density plot")
else:
	print("Field lines not plotted - Ay data not available")

ax1.set_xlabel('z (axial position)', fontsize=12)
ax1.set_ylabel('r (radial position)', fontsize=12)
ax1.set_title(f'Ion Density (slice {slice})', fontsize=14)
ax1.set_aspect('equal')
cbar1 = fig1.colorbar(im1, ax=ax1, label='ni', shrink=0.3)

fig1.tight_layout()
fig1.savefig('plot_density.png', dpi=300)
print("Saved density plot to plot_density.png")

# ============================================================
# FIGURE 2: All Field Components (Bx, By, Bz, Ex, Ey, Ez)
# ============================================================
fig2, axes = plt.subplots(2, 3, figsize=(10, 2))
axes = axes.flatten()

# Define which fields to plot
field_names = ['bx', 'by', 'bz', 'ex', 'ey', 'ez']
field_titles = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']
field_labels = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']

# Color maps - use RdBu_r for diverging data (fields can be positive/negative)
field_cmap = 'RdBu_r'

# Plot each field component
for idx, (field, title, label) in enumerate(zip(field_names, field_titles, field_labels)):
	ax = axes[idx]
	
	# Get symmetric color limits for better visualization
	vmax = np.max(np.abs(Q[field]))
	vmin = -vmax
	
	im = ax.pcolormesh(Z, R, Q[field], cmap=field_cmap, shading='auto',
	                   vmin=vmin, vmax=vmax)
	
	# Overlay field lines if available
	if Ay is not None:
		contours = ax.contour(Z, R, Ay, levels=15, colors='black', 
		                      linewidths=0.5, alpha=0.4, linestyles='solid')
	
	ax.set_xlabel('z', fontsize=10)
	ax.set_ylabel('r', fontsize=10)
	ax.set_title(title, fontsize=12, pad=10)
	ax.set_aspect('equal')
	
	# Add colorbar with smaller size
	cbar = fig2.colorbar(im, ax=ax, shrink=0.2, pad=0.02)
	cbar.set_label(label, fontsize=9)
	cbar.ax.tick_params(labelsize=8)

fig2.suptitle(f'Electromagnetic Field Components (slice {slice})', fontsize=16, y=0.995)
fig2.tight_layout()
fig2.savefig('plot_fields.png', dpi=300)
print("Saved field components plot to plot_fields.png")

# plt.show()