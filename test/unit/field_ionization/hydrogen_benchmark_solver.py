# Brandon M. Medina
# This script solves the system of coupled ODEs
# that describes the Hydrogen benchmark problem
# and calculates a gold file.

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

# Global Constants
q_e       = 1.60217663e-19;  # coulombs
m_e       = 9.1093837e-31;   # kilograms
c         = 299792458;       # m/s
epsilon_0 = 8.85418782e-12;  # F/m
alpha     = 0.00729735;      # fine structure constant
h_bar     = 1.054571817e-34; # J *s
lambda_0  = 0.8e-6;          # m
I_L       = 1e14 / 1e-2**2;   # W/m^2

nu        = c/lambda_0; # [Hz]
omega_0   = 2*np.pi*nu;  # [radians/sec], laser (angular) frequency  
E_max     = np.sqrt( (2*I_L)/(c * epsilon_0) ); # SI units
m         = 0; # In SMILIE, the ionization rate is comuted for abs(m)= 0 only
l         = 0;
Z_star    = 0; # initial charge state
I_Z_star_eV     = 13.6; # eV
I_Z_star_atomic = I_Z_star_eV * 0.036749322176; # Z^star ionization potential, atomic units (Hartree)
n_star          = (Z_star + 1)/np.sqrt(2*I_Z_star_atomic); # effective principle quantum number
l_star          = n_star - 1; # angular momentum
A_n_l           = ( 2**(2*n_star) ) / ( n_star * math.gamma(n_star+l_star+1) * math.gamma(n_star-l_star) );
B_l_m           = ( (2*l+1)*math.factorial(l+abs(m)) ) / ( 2**abs(m)*math.factorial(abs(m))*math.factorial(l-abs(m))  );
t_to_SI         = 2.4188843265857e-17; 
E_to_SI         =  (alpha**3*m_e**2*c**3)/(q_e*h_bar);
    

# Define the ODE system
def coupledODEs(N, t):

    # Extract variables
    N0 = N[0]
    N1 = N[1]
    
    # Ionization rate 
    gamma_0 = (h_bar/(alpha**2*m_e*c**2))**(-1)*A_n_l * B_l_m*I_Z_star_atomic*(2*(2*I_Z_star_atomic)**(3/2)/np.sqrt((E_max/E_to_SI*math.cos(omega_0*t))**2))**((2*n_star-m-1))*math.exp(-(2*(2*I_Z_star_atomic)**(3/2))/(3*np.sqrt((E_max/E_to_SI*math.cos(omega_0*t))**2)))

    # System of ODEs
    dN0dt = - gamma_0*N0
    dN1dt =   gamma_0*N0
    
    # Return derivatives
    return [dN0dt, dN1dt]

# Set initial conditions and time span
N_initial = [1000, 0]  # Initial values for x and y

timesteps = np.arange(0,620);           # FIXME: get from simulation
timestep_to_SI = 4.29926779703711293e-17;
t = timesteps * timestep_to_SI; # Time span for integration


# Solve the ODEs
sol = odeint(coupledODEs, N_initial, t)

# Extract the solutions
N0 = sol[:, 0]
N1 = sol[:, 1]

# Calculate the average charge state
avg_charge_state = N1/N_initial[0];

# Write solutions to a CSV file
data = np.column_stack((t, avg_charge_state))
np.savetxt('hydrogen_benchmark_gold.csv', data, delimiter=',', header='', comments='')

'''
# Plot the average charge state
plt.plot(c*t/lambda_0, avg_charge_state, 'k', linewidth=2)
plt.xlabel('c*t/$\lambda$')
plt.ylabel('<Z>')
plt.show()
# Plot the solutions
plt.plot(c*t/lambda_0, N0, 'b', linewidth=2, label='N0')
plt.plot(c*t/lambda_0, N1, 'r', linewidth=2, label='N1')
plt.xlabel('c*t/$\lambda$')
plt.ylabel('Solution')
plt.legend()
plt.title('Solutions of the Coupled ODEs')
plt.show()
'''
