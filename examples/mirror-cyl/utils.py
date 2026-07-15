import matplotlib.pylab as plt
import numpy as np
import h5py, os, sys, string
from scipy import constants as con
import ffmpeg  # pip install ffmpeg-python
from collections import defaultdict
import matplotlib.colors as colors

# load matplotlib style
#ws_dir = os.getenv('WS_DIR')
#plt.style.use(f'{ws_dir}/extras/mjl.mplstyle')


ref_b0  = 3.0e4 # Central cell magnetic field in Gauss
ref_n0  = 1e14 # Central cell reference density in cm^-3
# ref_mi  = 3.344e-24 # Ion mass in grams
ref_mi  = 1.6605e-24 # Atomic mass unit in grams
ref_qi  = 4.803e-10 # Ion charge in esu (Gaussian CGS unit)
ref_c   = 2.998e10; # Speed of light in cm/s
ref_vA0 = ref_b0/(4*np.pi*ref_n0*ref_mi)**0.5 # Alfven speed in cm/s
ref_di = ref_c/(4*np.pi*ref_n0*ref_qi*ref_qi/ref_mi)**0.5 # Ion skin depth in cm
ref_E0 = ref_mi*ref_vA0*ref_vA0 # Reference energy in erg
ref_wci = ref_vA0 / ref_di

ref_Ti0 = 40 # keV

dt = 0.01
Lx = 36
Ly = 4
Lz = 360

nx = 64
ny = 1
nz = 256

# # Plasma parameters
# mime   = 1.0
# TiTe   = 1.0
# Z      = 1.0
# vthe   = 0.05
# beta_e = 0.03

# # Reference units
# n_ref = 1.0e20
# w_pe = (n_ref * con.e**2 / (con.m_e * con.epsilon_0))**0.5
# d_e = con.c / w_pe
# w_pi = w_pe / (mime)**0.5
# d_i = con.c / w_pi
# vthi = vthe * (TiTe / mime)**0.5

# wpe_wce = (beta_e)**0.5 / (vthe * 2.0**0.5) # Electron plasma freq/Electron Cyclotron
# w_ce = w_pe / wpe_wce
# w_ci = w_ce / (mime)**0.5

# dt = 0.01 / w_ci
# Lx = 300.0 * d_i
# Ly = d_i
# Lz = 30.0 * d_i

# nx = 300
# ny = 1
# nz = 30

# grid_lx = np.linspace(0, Lx, nx)
# grid_ly = np.linspace(0, Ly, ny)
# grid_lz = np.linspace(0, Lz, nz)

# m_e = con.m_e
# m_i = m_e * mime

# b0 = 1.0 #m_e * con.c * w_ce / con.e

# #################################################################
# # Reference unit dictionary
# #################################################################

# # number density
# rho_keys = ['rhof', 'rhofold', 'rho', 'rho_m', \
#             'n_q0', 'n_q1', 'n_q2', 'n_q3', 'n_q4', 'n_q5',]
# rho_conv_dict = dict.fromkeys(rho_keys, n_ref)
# rho_unit_dict = dict.fromkeys(rho_keys, r'm$^{-3}$')

# # momentum
# momt_keys = ['px', 'py', 'pz']
# momt_conv_dict = dict.fromkeys(momt_keys, m_i * con.c)
# momt_unit_dict = dict.fromkeys(momt_keys, r'kg$\cdot$m/s')

# # mass density
# den_keys = ['den'] 
# den_conv_dict = dict.fromkeys(den_keys, n_ref * m_i)
# den_unit_dict = dict.fromkeys(den_keys, r'kg/m$^{-3}$')

# # pressure
# prs_keys = ['prs'] 
# prs_conv_dict = dict.fromkeys(prs_keys, 1.0)
# prs_unit_dict = dict.fromkeys(prs_keys, 'bar')

# # velocity
# vel_keys = ['ux', 'uy', 'uz'] 
# vel_conv_dict = dict.fromkeys(vel_keys, con.c)
# vel_unit_dict = dict.fromkeys(vel_keys, 'm/s')

# # magnetic field
# bfield_keys = ['cbx', 'cby', 'cbz', 'cbx0', 'cby0', 'cbz0', 'div_b_err']
# bfield_conv_dict = dict.fromkeys(bfield_keys, b0)
# bfield_unit_dict = dict.fromkeys(bfield_keys, 'T')

# # electric field
# efield_keys = ['ex', 'ey', 'ez']
# efield_conv_dict = dict.fromkeys(efield_keys, b0 * con.c)
# efield_unit_dict = dict.fromkeys(efield_keys, 'V/m')

# # current
# current_keys = ['jx', 'jy', 'jz']
# current_conv_dict = dict.fromkeys(current_keys, 1.0)
# current_unit_dict = dict.fromkeys(current_keys, 'A')

# # default
# def_keys = ['txx', 'tyy', 'tzz', 'tyz', 'tzx', 'txy', 'qmin', 'qmax', 'pad', 'tmp']
# def_conv_dict = dict.fromkeys(def_keys, 1.0)
# def_unit_dict = dict.fromkeys(def_keys, 'a.u.')


# # Dictionary for changing units from simulation units to physical units
# conv_unit_dict = rho_conv_dict | momt_conv_dict | den_conv_dict | prs_conv_dict | \
#                  vel_conv_dict | efield_conv_dict | bfield_conv_dict  | current_conv_dict | \
#                  def_conv_dict

# # Dictionary for getting unit string
# unit_labl_dict = rho_unit_dict | momt_unit_dict | den_unit_dict | prs_unit_dict | \
#                  vel_unit_dict | efield_unit_dict | bfield_unit_dict | current_unit_dict | \
#                  def_unit_dict


# #################################################################
# # Physics functions
# #################################################################

# def calc_b0(TeV_, m_, Z_):
#     vth2 = con.e * TeV_ / m_
#     return con.e ** 2.0 * Z_ ** 2 / (4.0 * np.pi * con.epsilon_0 * m_ * vth2)


# def calc_deBroglie_wavelength(TeV_):
#     return (2.0 * np.pi * con.hbar ** 2.0 / (con.m_e * con.e * TeV_)) ** 0.5


# def calc_impact_parameter(Z_, TeV_):
#     return Z_ * con.e ** 2.0 / (con.e * TeV_)


# # function returns Debye length for single species
# def calc_Debye_length(n_, Z_, TeV_):
#     r_min = (4.0 * np.pi * n_ / 3.0) ** (-1.0 / 3.0)  # min mean interatomic distance
#     debye_length = (n_ * (con.e * Z_) ** 2.0 / (con.epsilon_0 * con.e * TeV_)) ** (-0.5)
#     return max(r_min, debye_length)


# def calc_Coulomb_log(TeV_, n_, m_, Z_):
#     b0 = calc_b0(TeV_, m_, Z_)
#     lambda_deBroglie = calc_deBroglie_wavelength(TeV_)
    
#     rmin = (4.0 * np.pi * n_ / 3.0) ** (-1.0 / 3.0)  # interatomic spacing
#     lDebye = calc_Debye_length(n_, Z_, TeV_)  # species Debye length

#     bmin2 = b0 ** 2.0 + lambda_deBroglie ** 2.0
#     bmax2 = rmin ** 2.0 + lDebye ** 2.0

#     return 0.5 * np.log(1.0 + bmax2 / bmin2)


# def calc_nu_momentum(m1_, m2_, Z1_, Z2_, v1_, n2_, lnL_):
#     nu = 0.0
#     if (v1_ != 0.0 or n2_ != 0.0):
#         mu = (m1_ * m2_) / (m1_ + m2_) # reduced mass
#         coef = (Z1_ * Z2_ * con.e ** 2.0 / (4.0 * np.pi * con.epsilon_0)) ** 2.0
#         nu = coef * 4.0 * np.pi * n2_ / (mu * m1_ * v1_ ** 3.0) * lnL_
#     return nu


# def calc_mfp(m1_, m2_, Z1_, Z2_, v1_, n2_, lnL_):
#     nu = calc_nu_momentum(m1_, m2_, Z1_, Z2, n1_, v1_, n2_, lnL_)
#     #v1 = (2.0 * con.e * T1 / m1) ** 0.5
#     return v1_ / nu_12


# def calc_nu_ei_spitzer(Z_, n_i_, lnL_, T_K_):
#     return ((4.0 / 3.0) * np.sqrt(2.0 * np.pi / con.m_e) * (Z_**2 * con.e**4 * n_i_ * lnL_) /
#                 (4.0 * np.pi * con.epsilon_0)**2 / (con.k * T_K_)**(3.0 / 2.0))

# #################################################################
# # Plasma formulary equations
# #################################################################

# """ 
# Plasma formulary equations for relaxation rates.
# Primed quantity in formulary is background.
#     mu = mi / mp 
#     Z = qi / e
#     eps = test particle energy in eV
#     T = background temperature in eV
#     n = background density in cm^-3
# """

# def calc_nu_slowing_ii_fast(mu_t_, mu_b_, Z_b_, Z_t_, eps_t_, n_b_, lnL_):
#     """
#     Ion(test)-ion(background) slowing down relaxation rate
#     in limit of fast target from plasma formulary.
#     """
#     t1 = 9.0e-8 * (mu_t_**-1.0 + mu_b_**-1.0) * mu_t_**0.5 / eps_t_**1.5
#     t2 = n_b_ * (Z_b_ * Z_t_)**2.0 * lnL_
#     return t1 * t2

# def calc_nu_transverse_ii_fast(mu_t_, Z_b_, Z_t_, eps_t_, n_b_, lnL_):
#     """
#     Ion(test)-ion(background) transverse diffusion relaxation rate
#     in limit of fast target from plasma formulary.
#     """
#     t1 = 1.8e-7 * mu_t_**-0.5 * eps_t_**-1.5
#     t2 = n_b_ * (Z_b_ * Z_t_)**2.0 * lnL_
#     return t1 * t2

# def calc_nu_parallel_ii_fast(mu_t_, mu_b_, Z_b_, Z_t_, eps_t_, n_b_, T_b_, lnL_):
#     """
#     Ion(test)-ion(background) parallel diffusion relaxation rate
#     in limit of fast target from plasma formulary.
#     """
#     t1 = 9.0e-8 * mu_t_**0.5 * mu_b_**-1.0 * T_b_ * eps_t_**-2.5
#     t2 = n_b_ * (Z_b_ * Z_t_)**2.0 * lnL_
#     return t1 * t2

# def calc_nu_energy_ii_fast(mu_t_, mu_b_, Z_b_, Z_t_, eps_t_, n_b_, T_b_, lnL_):
#     """
#     Ion(test)-ion(background) energy transfer relaxation rate
#     in limit of fast target from plasma formulary.
#     """
#     nu_s = calc_nu_slowing_ii_fast(mu_t_, mu_b_, Z_b_, Z_t_, eps_t_, n_b_, lnL_)
#     nu_t = calc_nu_transverse_ii_fast(mu_t_, Z_b_, Z_t_, eps_t_, n_b_, lnL_)
#     nu_p = calc_nu_parallel_ii_fast(mu_t_, mu_b_, Z_b_, Z_t_, eps_t_, n_b_, T_b_, lnL_)
#     return 2.0 * nu_s - nu_t - nu_p
