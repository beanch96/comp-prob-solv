import numpy as np
from scipy.constants import k, N_A
import pandas as pd
import comp_part_func as fn

# Constants
epsilon = 0.0103 * 1.60218e-19       # Convert eV to Joules
sigma = 3.4e-10                      # Convert Angstrom to meters
molar_mass_Ar = 39.948               # Argon molar mass in g/mol
m_Ar = molar_mass_Ar / N_A / 1000    # Mass of one Argon atom in kg
V = (10e-10)**3                        # Volume in cubic meters (box size of 10 Angstroms)
L = (V)**(1/3)                       # Box length in meters


# Calculate internal energy U and heat capacity Cv
def calculate_thermodynamic_properties(T_range):
    """"
    Compute thermodynamic properties of internal energy and heat capacity by finding parition functions

    Parameters:
    T_range(array): Array of Temperatures
    """
    #Create list to append Z and lnZ
    Z_values = []    
    lnZ_values = []
    
    # Calculate partition function and its logarithm for all temperatures
    for T in T_range:
        Z_T = fn.total_partition_function(T)
        Z_values.append(Z_T)
        lnZ_values.append(np.log(Z_T))
    
    Z_values = np.array(Z_values)
    lnZ_values = np.array(lnZ_values)
    
    beta_range = 1 / (k * T_range)
    
    # Calculate internal energy U = -d(ln Z)/d beta
    U_values = -np.gradient(lnZ_values, beta_range)
    
    # Calculate heat capacity Cv = dU/dT
    U_values = np.array(U_values)
    Cv_values = np.gradient(U_values, T_range)
    
    return Z_values, U_values, Cv_values

# Temperature range
T_range = np.linspace(10, 1000, 10000)  

# Calculate partition function, internal energy, and heat capacity
Z_values, U_values, Cv_values = calculate_thermodynamic_properties(T_range)

#Create data for csv file
df = pd.DataFrame({
    'Temperature': T_range,
    'Internal Energy': U_values,
    'Cv': Cv_values
})

#Save data to excel file
df.to_csv('homework-4-grad/Internal Energy-Cv-Temperature.csv', index=False)
print("CSV file 'Internal Energy/Cv/Temperature.csv' created successfully.")

