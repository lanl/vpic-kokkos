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
c_au      = 137.02;          # atomic units
epsilon_0 = 8.85418782e-12;  # F/m
alpha     = 0.00729735;      # fine structure constant
h_bar     = 1.054571817e-34; # J *s
lambda_0  = 0.8e-6;          # m
I_L       = 1e14 / 1e-2**2;   # W/m^2

nu        = c/lambda_0; # [Hz]
omega_0   = 2*np.pi*nu;  # [radians/sec], laser (angular) frequency
omega_eV  = 1.2398 / (lambda_0 / 1e-6); # eV
omega_au  = omega_eV * 0.036749; # energy, Hartree units  
E_max     = np.sqrt( (2*I_L)/(c * epsilon_0) ); # SI units
m         = 0; # In SMILIE, the ionization rate is comuted for abs(m)= 0 only
l         = 0;
n         = 1;
Z_star    = 0; # initial charge state
Z         = Z_star+1; # final ionization state
I_Z_star_eV     = 13.6; # eV
I_Z_star_atomic = I_Z_star_eV * 0.036749322176; # Z^star ionization potential, atomic units (Hartree)
n_star          = (Z_star + 1)/np.sqrt(2*I_Z_star_atomic); # effective principle quantum number
l_star          = n_star - 1; # angular momentum
A_n_l           = ( 2**(2*n_star) ) / ( n_star * math.gamma(n_star+l_star+1) * math.gamma(n_star-l_star) );
B_l_m           = ( (2*l+1)*math.factorial(l+abs(m)) ) / ( 2**abs(m)*math.factorial(abs(m))*math.factorial(l-abs(m))  );
K               = math.floor(I_Z_star_atomic / omega_au) + 1
    
t_to_SI         = 2.4188843265857e-17; 
E_to_SI         =  (alpha**3*m_e**2*c**3)/(q_e*h_bar);
Gamma_conversion = 1.0/(h_bar/(alpha**2*m_e*c**2)); # multiply au to get sec^-1 

E_M_au = omega_au * math.sqrt(2 * I_Z_star_atomic); # atomic units
E_T_au = (I_Z_star_atomic**2.0) / (4 * Z); # atomic units
E_B_au = (2*(2)**(3.0/2.0))/(3)*I_Z_star_atomic**(3/2)/(2.*n_star - abs(m) - 1.); # atomic units

# variable to save values
gamma_0_values = []
E_au_values    = []

# Define the ODE system
def coupledODEs(N, t):

    # Extract variables
    N0 = N[0]
    N1 = N[1]

    # field at time t
    E_au = np.sqrt((E_max/E_to_SI*math.cos(omega_0*t))**2)

    # Ionization rate
    if E_au <= E_M_au:
        #  If K! is too large then the multiphoton ionisation rate is zero
        if K < 30: # K! ~ 2.6525e+32 so this limit should be sufficient to ignore MPI ionization 
            # MPI
            T_K = 4.80 * pow(1.30, 2 * K) * pow(2 * K + 1, -1) * pow(K, -1.0 / 2.0)
            sigma_K_au = pow(c_au * math.gamma(K + 1) ** 2 * pow(n, 5) * pow(omega_au, (10 * K - 1) / 3), -1) * T_K * pow(E_au, 2 * K - 2)
            flux = c_au * pow(E_au, 2.0) / (8 * np.pi * omega_au)
            gamma_MPI = sigma_K_au * pow(flux, K) * Gamma_conversion
        else:
            gamma_MPI = 0; 
        # ADK
        gamma_ADK =  Gamma_conversion*A_n_l * B_l_m*I_Z_star_atomic*(2*(2*I_Z_star_atomic)**(3/2)/E_au)**((2*n_star-m-1))*math.exp(-(2*(2*I_Z_star_atomic)**(3/2))/(3*E_au))
        # decide if ADK or MPI
        gamma_0 = min(gamma_MPI, gamma_ADK)   
    elif E_au > E_M_au and E_au <= E_T_au:
        # ADK
        gamma_0 =  Gamma_conversion*A_n_l * B_l_m*I_Z_star_atomic*(2*(2*I_Z_star_atomic)**(3/2)/E_au)**((2*n_star-m-1))*math.exp(-(2*(2*I_Z_star_atomic)**(3/2))/(3*E_au))
    elif E_au > E_T_au and E_au <= E_B_au:
        # ADK
        gamma_ADK =  Gamma_conversion*A_n_l * B_l_m*I_Z_star_atomic*(2*(2*I_Z_star_atomic)**(3/2)/E_au)**((2*n_star-m-1))*math.exp(-(2*(2*I_Z_star_atomic)**(3/2))/(3*E_au))
        # BSI correction
        T_0 = np.pi * Z / (abs(I_Z_star_atomic) * np.sqrt(2 * abs(I_Z_star_atomic))); # period of classical radial trajectories
        Gamma_ADK_au_threshold = A_n_l * B_l_m * I_Z_star_atomic * (2 * (2 * I_Z_star_atomic)**(3.0 / 2.0) / E_T_au)**(2 * n_star - abs(m) - 1) * math.exp(-2 * (2 * I_Z_star_atomic)**(3.0 / 2.0) / (3 * E_T_au));
        Gamma_ADK_SI_threshold = Gamma_ADK_au_threshold * Gamma_conversion;
        Gamma_cl_au = 1.0 / (np.pi * T_0) * (np.pi / 2.0 - math.asin((I_Z_star_atomic**2.0) / (4 * Z * E_au)) + (I_Z_star_atomic**2.0) / (4 * Z * E_au) * math.log((4 * Z * E_au - np.sqrt(16 * (Z**2.0) * (E_au**2.0) - (I_Z_star_atomic**4.0))) / (I_Z_star_atomic**2.0))); #oscillating field
        Gamma_cl_SI = Gamma_cl_au * Gamma_conversion;
        gamma_BSI = Gamma_cl_SI + Gamma_ADK_SI_threshold;
        # Decide if ADK or BSI
        gamma_0 = min(gamma_ADK, gamma_BSI)
    elif E_au > E_B_au:
        # BSI correction
        T_0 = np.pi * Z / (abs(I_Z_star_atomic) * np.sqrt(2 * abs(I_Z_star_atomic))); # period of classical radial trajectories
        Gamma_ADK_au_threshold = A_n_l * B_l_m * I_Z_star_atomic * (2 * (2 * I_Z_star_atomic)**(3.0 / 2.0) / E_T_au)**(2 * n_star - abs(m) - 1) * math.exp(-2 * (2 * I_Z_star_atomic)**(3.0 / 2.0) / (3 * E_T_au));
        Gamma_ADK_SI_threshold = Gamma_ADK_au_threshold * Gamma_conversion;
        Gamma_cl_au = 1.0 / (np.pi * T_0) * (np.pi / 2.0 - math.asin((I_Z_star_atomic**2.0) / (4 * Z * E_au)) + (I_Z_star_atomic**2.0) / (4 * Z * E_au) * math.log((4 * Z * E_au - np.sqrt(16 * (Z**2.0) * (E_au**2.0) - (I_Z_star_atomic**4.0))) / (I_Z_star_atomic**2.0))); #oscillating field
        Gamma_cl_SI = Gamma_cl_au * Gamma_conversion;
        gamma_0 = Gamma_cl_SI + Gamma_ADK_SI_threshold

    gamma_0_values.append(gamma_0)
    E_au_values.append(E_au)
  
    # System of ODEs
    dN0dt = - gamma_0*N0
    dN1dt =   gamma_0*N0
    
    # Return derivatives
    return [dN0dt, dN1dt]

# Set initial conditions and time span
N_initial = [1000, 0]  # Initial values for x and y

timesteps = np.arange(0,600);           # FIXME: get from simulation
timestep_to_SI = 4.29926779703711293e-17;
t = timesteps * timestep_to_SI; # Time span for integration

# Solve the ODEs
tolerance = 1e-12
sol = odeint(coupledODEs, N_initial, t, atol=tolerance, rtol=tolerance)

# Extract the solutions
N0 = sol[:, 0]
N1 = sol[:, 1]

# Calculate the average charge state
avg_charge_state = N1/N_initial[0];

# Write solutions to a CSV file
#header_str = f"% timestep = {timestep_to_SI}"
data = np.column_stack((t, N0/N_initial[0], N1/N_initial[0]))
np.savetxt('hydrogen_benchmark_gold.csv', data, delimiter=',', header='', comments='')


## Plot the average charge state
#plt.plot(c*t/lambda_0, avg_charge_state, 'k', linewidth=2)
#plt.xlabel('c*t/$\lambda$')
#plt.ylabel('<Z>')
#plt.show()
## Plot the solutions
#plt.plot(c*t/lambda_0, N0, 'b', linewidth=2, label='N0')
#plt.plot(c*t/lambda_0, N1, 'r', linewidth=2, label='N1')
#plt.xlabel('c*t/$\lambda$')
#plt.ylabel('Solution')
#plt.legend()
#plt.title('Solutions of the Coupled ODEs')
#plt.show()
## Plot gamma_0 vs E_au
#plt.figure()
#plt.scatter(E_au_values, gamma_0_values, marker='.')
#plt.axvline(E_M_au, color='red', linestyle='--', label='E_M_au')
#plt.axvline(E_B_au, color='green', linestyle='--', label='E_B_au')
#plt.axvline(E_T_au, color='blue', linestyle='--', label='E_T_au')
#plt.xlabel('E_au')
#plt.ylabel('gamma_0')
#plt.title('gamma_0 vs E_au')
#plt.show()




