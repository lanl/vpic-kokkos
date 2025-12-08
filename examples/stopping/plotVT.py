"""
Plot velocity and temperature for two ion species

typedef struct hydro {
  float jx, jy, jz, rho; // Current and charge density => <q v_i f>, <q f>
  float px, py, pz, rho_m; // Momentum and mass density (changed from ke_density)
  float txx, tyy, tzz;   // Stress diagonal            => <p_i v_j f>, i==j
  float tyz, tzx, txy;   // Stress off-diagonal        => <p_i v_j f>, i!=j
#if VARIABLE_CHARGE
  float qmin, qmax;      // Minimum and maximum charge within a cell
#else
  float _pad[2];         // 16-byte align
#endif

"""

import matplotlib.pylab as plt
import numpy as np
import h5py, os, sys, subprocess
from scipy import constants as con

# load matplotlib style
if (subprocess.check_output(["whoami"])[:-1] == b'mlavell'):
  ws_dir = os.getenv('WS_DIR')
  plt.style.use(f'{ws_dir}/extras/mjl.mplstyle')

#################################################################
# Load data
#################################################################

speices = ["ion", "beam"]
species_mass = [1.0, 1.0]
species_charge = [1.0, 1.0]

num_step = 50000
interval = 200
steps = np.arange(0, num_step + interval, interval)

velc = np.zeros((2, np.size(steps)))
temp = np.zeros((2, np.size(steps)))

for i_species in range(2):
    step_cntr = 0
    for step in steps:
        step_data_dir = f'hydro_hdf5/T.{step}'
        fname = f'{step_data_dir}/hydro_{speices[i_species]}_{step}.h5'
        file = h5py.File(fname, 'r')
        group = file[f'Timestep_{step}']
        
        # print(speices[i_species], group['rho'])
        # rho = group['rho'][0,0,0]             # charge density
        rho_m = group['rho_m'][0,0,0]         # mass density
        # ndens = rho / species_mass[i_species] # number density

        # calculate temperature from stress tensor and density
        # sigma_ij = -p * delta_ij + sigma_ij'  -->  sigam_ii = -p
        # t = (pxx + pyy + pzz) / (3.0 * ndens)
        txx = group['txx'][0,0,0] 
        tyy = group['tyy'][0,0,0]
        tzz = group['tzz'][0,0,0]
        ke2 = (txx + tyy + tzz) / rho_m    

        vx = group['px'][0,0,0] / (species_mass[i_species] * rho_m)
        vy = group['py'][0,0,0] / (species_mass[i_species] * rho_m)
        vz = group['pz'][0,0,0] / (species_mass[i_species] * rho_m)

        t = (ke2 - species_mass[i_species] * (vx**2.0 + vy**2.0 + vz**2.0)) / (3.0)

        jz = group['jz'][0,0,0]
        vz = jz / rho_m
        
        velc[i_species, step_cntr] = vz
        temp[i_species, step_cntr] = t

        step_cntr += 1

#################################################################
# Change units
#################################################################

def calc_nu_SI(m1_, m2_, Z1_, Z2_, n_, T_, lnL_):
  mu = (m1_ * m2_) / (m1_ + m2_)
  num = (Z1_ * con.e)**2.0 * (Z2_ * con.e)**2.0 * n_ * lnL_ 
  den = 8.0 * np.pi * con.epsilon_0**2.0 * mu**0.5 * (con.e * T_)**1.5
  return num / den

# fully-iononized carbon
mass_kg = 12.0 * con.atomic_mass
charge = 6.0
ndens_m3 = 1.0e26
T0_eV = 500.0
lnL = 10.0

nu = calc_nu_SI(mass_kg, mass_kg, charge, charge, ndens_m3, T0_eV, lnL)
# cvar0 = 2.0 ** 0.5 * nu_SI
tau = 1.0 / nu
dt = tau / 50.0
print(f'nu={nu:.2e},  dt={dt:.2e}')

time_ps = steps * dt * 1.0e12

vth0_kms = (con.e * T0_eV / mass_kg) ** 0.5 * 1.0e-3
T0_keV = T0_eV * 1.0e-3

vbeam = 655
print(f'vth0 = {vth0_kms},  vbeam/vth0 = {vbeam/vth0_kms}')

#################################################################
# Plot velocity and temperature
#################################################################

# create figure
fig, ax = plt.subplots(1, 2, figsize=(9, 4))

ax[0].set_ylabel('Velocity (km/s)')
ax[1].set_ylabel('Temperature (keV)')

for axis in ax:
    axis.set_xlabel('Time (ps)')
    axis.tick_params(axis='both', pad=8)
    axis.grid()
    # axis.set_xlim([time_ps[0], time_ps[-1]])
    axis.set_xlim([0.0, 100.0])

ax[0].plot(time_ps, velc[0, :] * vth0_kms, label='Background')
ax[0].plot(time_ps, velc[1, :] * vth0_kms, label='Beam')

ax[1].plot(time_ps, temp[0, :] * T0_keV, label='Background')
ax[1].plot(time_ps, temp[1, :] * T0_keV, label='Beam')

# inlude kinetic solution from Rambo and Procassini 1995
# (solid curves in Fig 4a and Fig 4b)
plot_rambo = True
if plot_rambo:
  v_rambo = np.loadtxt('rambo_soln/rambo_v_alpha.csv', delimiter=',')
  T_rambo = np.loadtxt('rambo_soln/rambo_T_alpha.csv', delimiter=',')

  cms_to_kms = 1.0e-5
  ax[0].plot(v_rambo[::2, 0], v_rambo[::2, 1] * cms_to_kms, 'xk', ms=8, mew=2, label='RP95')
  ax[1].plot(T_rambo[::2, 0], T_rambo[::2, 1], 'xk', ms=8, mew=2, label='RP95')
  
ax[0].legend()
fig.tight_layout(pad=0.5, rect=[0, 0, 1, 1])

# plt.savefig('figures/momt_equil_soln.png')
plt.show()
