import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pywt

datadir = "../../build/data/"
nx = 48#96
ny = 48
nt = 100

pi = np.pi

# Stretched grid parameters (must match VPIC deck)
beta_x = 5.0
beta_y = 5.0
beta_z = 0.0

# Domain bounds (must match VPIC deck: -Lx/2 to Lx/2, -Ly/2 to Ly/2)
Lx = 32.0
Ly = 32.0
gx0 = -Lx/2.0
gx1 = Lx/2.0
gy0 = -Ly/2.0
gy1 = Ly/2.0

# Create uniform computational coordinates for cell centers
xi_uniform = np.linspace(0, 1, nx, endpoint=False) + 0.5/(nx)
eta_uniform = np.linspace(0, 1, ny, endpoint=False) + 0.5/(ny)

def apply_stretching(xi, beta):
    """Apply tanh stretching to uniform coordinate"""
    if beta > 1e-10:
        stretched = np.tanh(beta * (xi - 0.5)) / np.tanh(beta * 0.5)
    else:
        stretched = 2.0 * xi - 1.0
    return stretched

def compute_scale_factors(xi, eta, beta_x, beta_y):
    """Compute scale factors h_i = 0.5 * d(stretched)/d(computational)"""
    
    # Derivatives of stretching transformation
    if beta_x > 1e-10:
        dx_dxi = beta_x / (np.tanh(beta_x * 0.5) * np.cosh(beta_x * (xi - 0.5))**2)
    else:
        dx_dxi = 2.0 * np.ones_like(xi)
    
    if beta_y > 1e-10:
        dy_deta = beta_y / (np.tanh(beta_y * 0.5) * np.cosh(beta_y * (eta - 0.5))**2)
    else:
        dy_deta = 2.0 * np.ones_like(eta)
    
    # Scale factors (dimensionless, h=1 for uniform grid)
    h1 = 0.5 * dx_dxi
    h2 = 0.5 * dy_deta
    h3 = 1.0  # No stretching in z for beta_z = 0
    
    return h1, h2, h3

# Get stretched coordinates (actual physical cell center locations)
x_stretched = apply_stretching(xi_uniform, beta_x)
y_stretched = apply_stretching(eta_uniform, beta_y)

# Map to physical domain
xv = gx0 + (gx1 - gx0) * (x_stretched + 1.0) * 0.5
yv = gy0 + (gy1 - gy0) * (y_stretched + 1.0) * 0.5

# Compute scale factors at each grid point
h1_array, h2_array, h3 = compute_scale_factors(xi_uniform, eta_uniform, beta_x, beta_y)

# Create 2D arrays of scale factors for field conversion
H1, H2 = np.meshgrid(h1_array, h2_array, indexing='ij')

tv = np.linspace(0,50,num=nt)
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

def convert_to_physical(field_data, scale_factors, is_contravariant):
    """
    Convert field components to physical components.
    
    Args:
        field_data: Field component array (contravariant or covariant)
        scale_factors: Scale factor array (h_i)
        is_contravariant: True for B (contravariant), False for E (covariant)
    
    Returns:
        Physical field component
    """
    if is_contravariant:
        # B is contravariant: B_physical = h_i * B^i
        return field_data * scale_factors
    else:
        # E is covariant: E_physical = E_i / h_i
        return field_data / scale_factors

#yv,xv = np.meshgrid(np.linspace(0,7.5*pi,num=ny),
#                np.linspace(0,5*pi,num=nx))

cmap = plt.get_cmap("Spectral")

Q = {}

for slice in range(0,100,5):
	qs = ["By","Ex"]
	for q in qs:
		tmp = loadSlice(datadir,q,slice,nx,ny)
		Q[q] = tmp
	
	# Convert to physical components
	# By is contravariant (multiply by h2)
	# Ex is covariant (divide by h1)
	By_physical = Q["By"]
	Ex_physical = Q["Ex"]
	
	#bxw = wclean(arr=Q["den"],wavn="coif3",alpha=1)
		
	fig, (ax1,ax2) = plt.subplots(nrows=2)
	im = ax1.pcolormesh(xv,yv,By_physical)
	#im = ax1.pcolormesh(yv,xv,Q["ni"],cmap=cmap)
	fig.colorbar(im, ax=ax1)
	#im = ax1.plot(xv,Q["ni"][:,0])
	im2 = ax2.pcolormesh(yv,xv,Ex_physical,cmap=cmap)
	fig.colorbar(im2, ax=ax2)    
	plt.show()
	plt.savefig('fig.png', dpi=300)