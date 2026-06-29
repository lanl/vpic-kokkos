import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def check_var(var_name, x, y, expected, beg, end):
    var_peaks, var_props = find_peaks(y, height=-0.75)
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
hydro_dir = rundir + "/hydro/"

nx = 64
ny = 1
nz = 1
nt = 151
nfield_vars = 32
nhydro_vars = 10

pi = np.pi

xv = np.linspace(0,10.5,num=nx)
#tv = np.linspace(0,100,num=nt)
tv = np.linspace(0,30,num=nt)
if (nx>1): dx = xv[1]-xv[0]
if (nt>1): dt = tv[1]-tv[0]

data = {"Uiy":   np.zeros((nt, nx,ny,nz), dtype=np.float32),
        "Uiz":   np.zeros((nt, nx,ny,nz), dtype=np.float32),
        "By":    np.zeros((nt, nx,ny,nz), dtype=np.float32),
        "Bz":    np.zeros((nt, nx,ny,nz), dtype=np.float32),
        "aniso": np.zeros((nt, nx,ny,nz), dtype=np.float32)}

# Read data from fields and hydro dumps
for t in range(0,nt):
    step = t*20
    field_file = open(field_dir + "T." + str(step) + "/fields." + str(step) + ".0", "rb")
    ftemp = np.fromfile(field_file, dtype=np.float32, count=nfield_vars*(nx+2)*(ny+2)*(nz+2), offset=123)
    ftemp = np.reshape(ftemp, (nx+2, ny+2, nz+2, nfield_vars), order='F')
    idx = int(step / 20)
    data["By"][idx, :,:,:]  = ftemp[1:nx+1, 1:ny+1, 1:nz+1, 5]
    data["Bz"][idx, :,:,:]  = ftemp[1:nx+1, 1:ny+1, 1:nz+1, 6]
    data["Uiy"][idx, :,:,:] = ftemp[1:nx+1, 1:ny+1, 1:nz+1, 17]
    data["Uiz"][idx, :,:,:] = ftemp[1:nx+1, 1:ny+1, 1:nz+1, 18]

    hydro_file = open(hydro_dir + "T." + str(step) + "/Hhydro." + str(step) + ".0", "rb")
    htemp = np.fromfile(hydro_file, dtype=np.float32, count=nhydro_vars*(nx+2)*(ny+2)*(nz+2), offset=123)
    htemp = np.reshape(htemp, (nx+2, ny+2, nz+2, nhydro_vars), order='F')
    txx = htemp[1:nx+1, 1:ny+1, 1:nz+1, 4]
    tyy = htemp[1:nx+1, 1:ny+1, 1:nz+1, 5]
    tzz = htemp[1:nx+1, 1:ny+1, 1:nz+1, 6]
    data["aniso"][idx, :,:,:] = np.divide(np.add(tyy,tzz),(2.0*txx))
    field_file.close
    hydro_file.close

cmap = plt.get_cmap("Spectral")

Q = {}
qs = ["Uiy","Uiz","aniso","By","Bz"]
for q in qs:
    tmp = data[q][:,:,:,:]
    tmp = np.reshape(tmp, (nt, nx))
    Q[q] = np.transpose(tmp)
    
fig, (ax1,ax2) = plt.subplots(nrows=2)
im1 = ax1.pcolormesh(tv,xv,Q["Uiy"])
#ax1.set_xlabel('t*w_ci')
ax1.set_ylabel('d|Ui_y|')

gamma = 0.162
#gamma = 0.0785

duy = np.sqrt(np.sum((Q["Uiy"])*(Q["Uiy"]),axis=0))
duz = np.sqrt(np.sum((Q["Uiz"])*(Q["Uiz"]),axis=0))
dby = np.sqrt(np.sum((Q["By"])*(Q["By"]),axis=0))
dbz = np.sqrt(np.sum((Q["Bz"])*(Q["Bz"]),axis=0))

aniso=np.mean(Q["aniso"],axis=0)
#print(tv)
#print(aniso)
#print(dn)
beg = 0
end = len(tv)
for idx in range(len(tv)):
  if tv[idx] <= 15:
    beg = idx
    continue
  if tv[idx] >= 30:
    end = idx
    break
duy_log10 = np.log10(duy)
duz_log10 = np.log10(duz)
expected = np.log10(0.019*np.exp(gamma*tv))

im2a = ax2.plot(tv,duy_log10,label='d|Uiy|')
im2b = ax2.plot(tv,duz_log10,label='d|Uiz|')
#im = ax2.plot(tv,np.log10(0.009*np.exp(gamma*tv)))
im2c = ax2.plot(tv,expected)  

duy_peaks, duy_valid = check_var("d|Uiy|", tv, duy_log10, expected, beg, end) 
duz_peaks, duz_valid = check_var("d|Uiz|", tv, duz_log10, expected, beg, end) 
if len(duy_peaks) > 0:
  im2d = ax2.plot(tv[duy_peaks], duy_log10[duy_peaks], 'x', color='blue')
if len(duz_peaks) > 0:
  im2e = ax2.plot(tv[duz_peaks], duz_log10[duz_peaks], 'x', color='orange')

ax2.set_xlabel('t*w_ci')
ax2.set_ylabel('d|Ui|')
ax2.legend()

#plt.xlim([0, 80])
plt.xlim([0, 30])
plt.ylim([-2, 1])



expected  = np.log10(0.012*np.exp(gamma*tv))
dby_log10 = np.log10(dby)
dbz_log10 = np.log10(dbz)

fig2, (ax3,ax4) = plt.subplots(nrows=2)
im3 = ax3.plot(tv,aniso)
im4a = ax4.plot(tv,dby_log10,label='d|By|')
im4b = ax4.plot(tv,dbz_log10,label='d|Bz|')
im4c = ax4.plot(tv,expected)

dby_peaks, dby_valid = check_var("d|By|", tv, dby_log10, expected, beg, end) 
dbz_peaks, dbz_valid = check_var("d|Bz|", tv, dbz_log10, expected, beg, end) 
if len(dby_peaks) > 0:
  im4d = ax4.plot(tv[dby_peaks], dby_log10[dby_peaks], 'x', color='blue')
if len(dby_peaks) > 0:
  im4e = ax4.plot(tv[dbz_peaks], dbz_log10[dbz_peaks], 'x', color='orange')

ax4.set_xlabel('t*w_ci')
ax3.set_ylabel('P_perp/P_par')
ax4.set_ylabel('d|B|')
ax4.legend()

plt.xlim([0, 30])
plt.ylim([-2, 1])

#plt.show()

if duy_valid and duz_valid and dby_valid and dbz_valid:
    sys.exit(0)
else:
    sys.exit(1)

