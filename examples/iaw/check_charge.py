# Charge-conservation check for stretched/curvilinear IAW.
# On a stretched grid rhof = q/(8*jac), so equilibrium ni is NON-uniform and
# Sum(ni) is NOT the conserved quantity. The conserved total charge is
#     Q(t) = Sum_cell rhof(cell) * Vphys(cell),   Vphys = 8*jac ∝ h1*h2*h3.
# For a 1D x-stretch (ny=nz=1) Vphys(cell) ∝ h1(cell). We reconstruct the
# relative cell volume from the equilibrium: at t=0 (uniform logical load)
# ni0 = rhof(t=0) already = q/(8 jac) up to a constant, so weight_cell = 1/ni0
# recovers Vphys and Q(t)=Sum ni(t)/ni0 should be constant.
import numpy as np, os
datadir="../../build/data/"; nx=48
ni=np.fromfile(datadir+"ni.gda",dtype=np.float32)
nt=ni.size//nx; ni=ni[:nt*nx].reshape(nt,nx)
ni0=ni[0].copy(); ni0[ni0==0]=np.nan          # t=0 profile ~ 1/Vphys (relative)
w = 1.0/ni0                                     # cell weight ∝ Vphys
Q = np.nansum(ni*w, axis=1)                      # total charge (relative units)
Q /= Q[0]
idx=np.linspace(0,nt-1,10).astype(int)
print("record :", " ".join("%7d"%i for i in idx))
print("Q(t)/Q0:", " ".join("%7.4f"%Q[i] for i in idx))
print()
print("If Q(t)/Q0 stays ~1.0 -> charge CONSERVED (ni growth is relaxation to stretched equilibrium).")
print("If Q(t)/Q0 grows      -> real non-conservation bug.")
print("raw Sum(ni)/Sum(ni0) for comparison:", 
      " ".join("%.3f"%(ni[i].sum()/ni[0].sum()) for i in idx))
