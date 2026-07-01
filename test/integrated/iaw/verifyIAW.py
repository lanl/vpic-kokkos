import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

########## wavelet denoising function
def wclean(arr,wavn,alpha):
	cs = pywt.wavedecn(arr, wavn, mode='symmetric', level=None,axes=None)
	levs = len(cs)
	coef = np.concatenate(cs[0])
	for x in range(1,levs):
		for n in cs[x]:
			coefn = np.concatenate(cs[x][n])
			coef = np.concatenate((coef,coefn))	
	thr2 = alpha*np.sqrt(np.var(coef)*np.log(len(coef)))
	thr = alpha*2*thr2
	while (thr/thr2 > 1.05):
		thr = thr2;
		thr2 = alpha*np.sqrt(np.var(coef[np.abs(coef)<thr])*np.log(len(coef)))
	for x in range(1,levs):
		for n in cs[x]:
			inds = np.abs(cs[x][n]) < thr
			cs[x][n][inds] = 0
	return pywt.waverecn(cs, wavn, mode='symmetric', axes=None);
########## end wavelet denoising

def check_var(var_name, x, y, expected, beg, end):
    var_peaks, var_props = find_peaks(y)
    var_valid = False
    if len(var_peaks) > 1:
      #var_peaks, var_props = find_peaks(y)
      var_peaks = list(filter(lambda i: beg <= i <= end, var_peaks))
      print(var_name + " Peak indices: ")
      print(var_peaks)
      print(var_name + " Peak values: ")
      print(y[var_peaks])
      print(var_name + " Expected values: ")
      print(expected[var_peaks])
      var_isclose = [np.isclose(y[idx], expected[idx], atol=0.10, rtol=0.25) for idx in var_peaks] 
      var_abs = [np.abs(expected[idx] - y[idx]) for idx in var_peaks]
      var_rel = [np.abs(expected[idx] - y[idx]) / expected[idx] for idx in var_peaks]
      print(var_name + " isclose: ")
      print(var_isclose)
      print(var_name + " abs error: ")
      print(var_abs)
      print(var_name + " rel error: ")
      print(var_rel)
      var_valid = np.all(var_isclose)
      print(var_name + " pass? " + str(var_valid))
      
      #expected_slope = (expected[end] - expected[beg])/(end-beg)
      #var_slope      = (y[var_peaks[-1]] - y[var_peaks[0]])/(var_peaks[-1] - var_peaks[0])
      #print("Expected slope: " + str(expected_slope))
      #print(var_name + " slope: " + str(var_slope))
      #var_slope_rel = np.abs(expected_slope - var_slope) / expected_slope
      #print(var_name + " slope rel error: " + str(var_slope_rel))
      #var_valid = (var_slope_rel < 0.25) and np.all(var_isclose)
      #print(var_name + " pass? " + str(var_valid))
    return var_peaks, var_valid

if len(sys.argv) != 2:
    sys.stderr.write("Usage: "+ str(sys.argv[0]) +" rundir\n")
    sys.exit(1)

rundir = sys.argv[1]
datadir = rundir + "/data/"
field_dir = rundir + "/fields/"

nx = 48
ny = 1
nz = 1
nt = 101
nfield_vars = 32

xv = np.linspace(0,16,num=nx)
tv = np.linspace(0,50,num=nt)
if (nx>1): dx = xv[1]-xv[0]
if (nt>1): dt = tv[1]-tv[0]

data = {"Uix":   np.zeros((nt, nx,ny,nz), dtype=np.float32),
        "Ex":   np.zeros((nt, nx,ny,nz), dtype=np.float32),
        "ni":    np.zeros((nt, nx,ny,nz), dtype=np.float32)}

# Read data from fields dumps
for t in range(0,nt):
    step = t*25
    field_file = open(field_dir + "T." + str(step) + "/fields." + str(step) + ".0", "rb")
    ftemp = np.fromfile(field_file, dtype=np.float32, count=nfield_vars*(nx+2)*(ny+2)*(nz+2), offset=123)
    ftemp = np.reshape(ftemp, (nx+2, ny+2, nz+2, nfield_vars), order='F')
    idx = int(step / 25)
    data["Ex"][idx, :,:,:]  = ftemp[1:nx+1, 1:ny+1, 1:nz+1, 0]
    data["Uix"][idx, :,:,:] = ftemp[1:nx+1, 1:ny+1, 1:nz+1, 20]
    data["ni"][idx, :,:,:]  = ftemp[1:nx+1, 1:ny+1, 1:nz+1, 23]

cmap = plt.get_cmap("Spectral")

Q = {}
qs = ["Ex", "Uix", "ni"]
for q in qs:
    tmp = data[q][:,:,:,:]
    tmp = np.reshape(tmp, (nt, nx))
    Q[q] = np.transpose(tmp)
    
print(Q["ni"])
        
gamma = -0.093196

fig, (ax1,ax2) = plt.subplots(nrows=2)
im = ax1.pcolormesh(tv,xv,Q["ni"])

expected = np.log10(0.08*np.exp(gamma*tv))
dn = np.sqrt(np.sum((1-Q["ni"])*(1-Q["ni"]),axis=0))#/float(nx)
dn_log10 = np.log10(dn)

im = ax2.plot(tv,dn_log10)
im = ax2.plot(tv[0:50],expected[0:50])
ax1.set_ylabel('x/L')
ax2.set_ylabel('|dn|')
ax2.set_xlabel('t * C_s/L')
ax1.set_xlim((0, 25))

beg = 0
end = len(tv)
for idx in range(len(tv)):
  #if tv[idx] <= 15:
  #  beg = idx
  #  continue
  if tv[idx] >= 25:
    end = idx
    break
print(beg)
print(end)

dn_peaks, dn_valid = check_var("|dn|", tv, dn_log10, expected, beg, end) 
print(dn_peaks)
if len(dn_peaks) > 0:
  im2d = ax2.plot(tv[dn_peaks], dn_log10[dn_peaks], 'x', color='blue')

plt.xlim([0, 25])
#plt.ylim([-2, 1])

plt.show()

if dn_valid:
    sys.exit(0)
else:
    sys.exit(1)


