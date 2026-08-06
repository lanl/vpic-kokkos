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
	infoarr=np.zeros(6);
	infoarr[0] = arr[1]  # nr    (nx)
	infoarr[1] = arr[2]  # ntheta (ny)
	infoarr[2] = arr[3]  # nz
	infoarr[3] = arr[6]  # xmax (radial extent)
	infoarr[4] = arr[7]  # ymax (theta extent)
	infoarr[5] = arr[8]  # zmax (axial extent)
	print("Grid: nr =", infoarr[0], ", ntheta =", infoarr[1], ", nz =", infoarr[2],
	      "| xmax =", infoarr[3], ", zmax =", infoarr[5])
	return infoarr
######### end loadinfo


######### loadSlice function
def loadSlice(dir, q, tslice, nr, nth, nz, ith=0):
	"""
	Load one theta-layer of a time slice from cylindrical (r,theta,z) data.
	Each time slice on disk is a full 3D field of nr*nth*nz floats written in
	Fortran order (fastest: r, then theta, then z), i.e. C-shape (nz, nth, nr).
	tslice: time-slice index; ith: theta layer to extract (0 for axisymmetric).
	"""
	fstr = dir + q + ".gda"
	fd = open(fstr,"rb")
	fd.seek(4*tslice*nr*nth*nz, 1)
	arr = np.fromfile(fd, dtype=np.float32, count=nr*nth*nz)
	fd.close
	arr = np.reshape(arr, (nz, nth, nr))  # Fortran (r,theta,z) -> C (z,theta,r)
	arr = arr[:, ith, :]                  # pick theta layer -> (z, r)
	arr = np.transpose(arr)               # -> (r, z)
	return arr
######### end loadSlice

cmap = plt.get_cmap("Spectral_r")

Q = {}
qs = ["ni", "bx", "by", "bz", "ex", "ey", "ez", "pe"]  # All quantities we need

dir = "../../build/data/"

# CYLINDRICAL coordinates now: (r, theta, z) -- extents autodetected from info
infoarr = loadinfo(dir)
nr  = int(infoarr[0])  # Radial cells
nth = int(infoarr[1])  # Azimuthal (theta) cells
nz  = int(infoarr[2])  # Axial cells
Lr = infoarr[3]        # Radial extent (xmax)
Lz = infoarr[5]        # Axial extent (zmax)

# Create coordinate arrays
rv = np.linspace(0, Lr, nr)  # Radial coordinates  np.linspace(0.1*Lr, 0.5*Lr, nr)
zv = np.linspace(-Lz/2, Lz/2, nz)  # Axial coordinates (centered)

# Create meshgrid for proper cylindrical plotting
R, Z = np.meshgrid(rv, zv, indexing='ij')

def loadFieldLines(dir, tslice, nr, nth, nz, ith=0):
	"""
	Load magnetic field lines (Ay) from Fortran output, one theta layer.
	Same 3D (r,theta,z) layout as the field .gda files.
	"""
	fstr = dir + "Ay_int.gda"

	# Check if file exists
	if not os.path.exists(fstr):
		print(f"Warning: {fstr} not found. Run Fortran code first to generate field lines.")
		return None

	try:
		fd = open(fstr, "rb")
		fd.seek(4*tslice*nr*nth*nz, 1)
		arr = np.fromfile(fd, dtype=np.float32, count=nr*nth*nz)
		fd.close()

		if len(arr) != nr*nth*nz:
			print(f"Warning: Expected {nr*nth*nz} values, got {len(arr)}")
			return None

		arr = np.reshape(arr, (nz, nth, nr))  # Fortran (r,theta,z) -> C (z,theta,r)
		arr = arr[:, ith, :]                  # pick theta layer -> (z, r)
		arr = np.transpose(arr)               # -> (r, z)
		return arr
	except Exception as e:
		print(f"Error loading field lines: {e}")
		return None

def get_num_slices(dir, q, nr, nth, nz):
    """
    Determine the number of time slices available in the file
    (each time slice is a full 3D field of nr*nth*nz floats).
    """
    fstr = dir + q + ".gda"
    file_size = os.path.getsize(fstr)
    bytes_per_slice = nr * nth * nz * 4  # 4 bytes per float32
    num_slices = file_size // bytes_per_slice
    return num_slices

num_slices = get_num_slices(dir, "ni", nr, nth, nz)
print(f"Number of available time slices: {num_slices}")

cnt = 0
slice = num_slices - 1
if len(sys.argv) > 1:
	slice = int(sys.argv[1])

# Load all quantities (theta=0 layer)
for q in qs:
	tmp = loadSlice(dir, q, slice, nr, nth, nz)
	Q[q] = tmp
	print(f"{q} range: [{np.min(Q[q]):.3e}, {np.max(Q[q]):.3e}]")

# Load field lines
Ay = loadFieldLines(dir, slice, nr, nth, nz)

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

ax1.set_xlabel('z', fontsize=12)
ax1.set_ylabel('r', fontsize=12)
ax1.set_title(f'density (slice {slice})', fontsize=14)
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