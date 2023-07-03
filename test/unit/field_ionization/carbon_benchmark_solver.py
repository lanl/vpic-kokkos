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
I_L       = 5e16 / 1e-2**2;   # W/m^2

t_to_SI   = 2.4188843265857e-17; 
E_to_SI   =  (alpha**3*m_e**2*c**3)/(q_e*h_bar);
nu        = c/lambda_0; # [Hz]
omega_0   = 2*np.pi*nu;  # [radians/sec], laser (angular) frequency  
E_max     = np.sqrt( (2*I_L)/(c * epsilon_0) ); # SI units
m         = 0; # In SMILIE, the ionization rate is comuted for abs(m)= 0 only
l         = 0;
I_Z_star_eV     = [11.26030, 24.38332, 47.8878, 64.4939, 392.087, 489.99334]; # eV
pulse_FWHM  = 5/nu;  # smilie sets this
pulse_mean  = 10/nu; # need to shift the gaussian (want the max at the end of the sim)                                                                                                                          
pulse_sigma = pulse_FWHM/( 2*math.sqrt(2*math.log(2) )); # sigma for gaussian function

    

# Define the ODE system
def coupledODEs(N, t):

    # Extract variables
    N0 = N[0]
    N1 = N[1]
    N2 = N[2]
    N3 = N[3]
    N4 = N[4]
    N5 = N[5]
    N6 = N[6]

    
    # Ionization rates 
    #gamma_0 = (h_bar/(alpha**2*m_e*c**2))**(-1)*A_n_l * B_l_m*I_Z_star_atomic*(2*(2*I_Z_star_atomic)**(3/2)/np.sqrt((E_max/E_to_SI*math.cos(omega_0*t))**2))**((2*n_star-m-1))*math.exp(-(2*(2*I_Z_star_atomic)**(3/2))/(3*np.sqrt((E_max/E_to_SI*math.cos(omega_0*t))**2)))
    gamma = np.zeros(7)
    for i,val in enumerate(I_Z_star_eV):
        Z_star    = i; # initial charge state
        I_Z_star_atomic = I_Z_star_eV[i] * 0.036749322176; # Z^star ionization potential, atomic units (Hartree)
        n_star          = (Z_star + 1)/np.sqrt(2*I_Z_star_atomic); # effective principle quantum number
        l_star          = n_star - 1; # angular momentum
        A_n_l           = ( 2**(2*n_star) ) / ( n_star * math.gamma(n_star+l_star+1) * math.gamma(n_star-l_star) );
        B_l_m           = ( (2*l+1)*math.factorial(l+abs(m)) ) / ( 2**abs(m)*math.factorial(abs(m))*math.factorial(l-abs(m))  );
        gamma[i] = (h_bar/(alpha**2*m_e*c**2))**(-1)*A_n_l*B_l_m*I_Z_star_atomic*(2*(2*I_Z_star_atomic)**(3/2)/math.sqrt((E_max/E_to_SI*math.cos(omega_0*t)*math.exp(-(t-pulse_mean)*(t-pulse_mean)/(2.*pulse_sigma*pulse_sigma)))**2))**((2*n_star-m-1))*math.exp(-(2*(2*I_Z_star_atomic)**(3/2))/(3*math.sqrt((E_max/E_to_SI*math.cos(omega_0*t)*math.exp(-(t-pulse_mean)*(t-pulse_mean)/(2.*pulse_sigma*pulse_sigma)))**2)))





    

    # System of ODEs
#    dN0dt =              - gamma_0*N0
#    dN1dt =   gamma_0*N0 - gamma_1*N1
#    dN2dt =   gamma_1*N1 - gamma_2*N2
#    dN3dt =   gamma_2*N2 - gamma_3*N3
#    dN4dt =   gamma_3*N3 - gamma_4*N4
#    dN5dt =   gamma_4*N4 - gamma_5*N5
#    dN6dt =   gamma_5*N5

    dN0dt =               - gamma[0]*N0
    dN1dt =   gamma[0]*N0 - gamma[1]*N1
    dN2dt =   gamma[1]*N1 - gamma[2]*N2
    dN3dt =   gamma[2]*N2 - gamma[3]*N3
    dN4dt =   gamma[3]*N3 - gamma[4]*N4
    dN5dt =   gamma[4]*N4 - gamma[5]*N5
    dN6dt =   gamma[5]*N5



        
    
    # Return derivatives
    return [dN0dt, dN1dt, dN2dt, dN3dt, dN4dt, dN5dt, dN6dt]

# Set initial conditions and time span
N_initial = [1000, 0, 0, 0, 0, 0, 0]  # Initial values for x and y

timesteps = np.arange(0,620);           # FIXME: get from simulation
timestep_to_SI = 4.29926779703711293e-17;
t = timesteps * timestep_to_SI; # Time span for integration


# Solve the ODEs
sol = odeint(coupledODEs, N_initial, t)

# Extract the solutions
N0 = sol[:, 0]
N1 = sol[:, 1]
N2 = sol[:, 2]
N3 = sol[:, 3]
N4 = sol[:, 4]
N5 = sol[:, 5]
N6 = sol[:, 6]


# Write solutions to a CSV file
data = np.column_stack((t, N0/N_initial[0], N1/N_initial[0], N2/N_initial[0], N3/N_initial[0], N4/N_initial[0], N5/N_initial[0], N6/N_initial[0]))
np.savetxt('carbon_benchmark_gold.csv', data, delimiter=',', header='', comments='')

'''
# Plot the average charge state
plt.plot(c*t/lambda_0, avg_charge_state, 'k', linewidth=2)
plt.xlabel('c*t/$\lambda$')
plt.ylabel('<Z>')
plt.show()
'''
# Plot the solutions
plt.plot(c*t/lambda_0, N0/N_initial[0], 'k', linewidth=2, label='Z* = 0')
plt.plot(c*t/lambda_0, N1/N_initial[0], 'r', linewidth=2, label='Z* = 1')
plt.plot(c*t/lambda_0, N2/N_initial[0], 'b', linewidth=2, label='Z* = 2')
plt.plot(c*t/lambda_0, N3/N_initial[0], 'g', linewidth=2, label='Z* = 3')
plt.plot(c*t/lambda_0, N4/N_initial[0], 'c', linewidth=2, label='Z* = 4')
plt.plot(c*t/lambda_0, N5/N_initial[0], 'm', linewidth=2, label='Z* = 5')
plt.plot(c*t/lambda_0, N6/N_initial[0], 'y', linewidth=2, label='Z* = 6')
plt.xlabel('c*t/$\lambda$')
plt.ylabel('N_{Z*}(t)/N_0(t=0)')
plt.xlim([4, 10])
plt.legend()
plt.show()

