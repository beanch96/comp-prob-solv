import numpy as np
from scipy.constants import k, h, N_A, pi
import pandas as pd

# Constants
epsilon = 0.0103 * 1.60218e-19       # Convert eV to Joules
sigma = 3.4e-10                      # Convert Angstrom to meters
molar_mass_Ar = 39.948               # Argon molar mass in g/mol
m_Ar = molar_mass_Ar / N_A / 1000    # Mass of one Argon atom in kg
V = (10e-10)**3                        # Volume in cubic meters (box size of 10 Angstroms)
L = (V)**(1/3)                       # Box length in meters


def lennard_jones_potential(r):
    """
    Calculate the Lennard-Jones potential energy between two atoms.

    Parameters:
    r (float): Distance between the two atoms.

    Returns:
    float: Potential energy V(r).
    """
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def total_partition_function(T):
    """
    Calculate the total partition function for a system of two Argon atoms interacting 
    via a Lennard-Jones potential in a cubic box, using the trapezoidal rule for integration.
    
    Parameters:
    T (float): Temperature in Kelvin at which the partition function is evaluated.
    
    Returns:
    Z_total (float): The total partition function at the specified temperature T.
    """
    r_min = sigma * 0.9            # Avoid r=0 to prevent singularity
    r_max = L                      #Max lenght of cube
    r_points = np.linspace(r_min, r_max, 1000)  #define r grid
    beta = 1 / (k * T)           
    lamba_val=np.sqrt(beta * h**2 / (2 * np.pi *m_Ar))   #thermal wavelenght 
    V_LJ = lennard_jones_potential(r_points)       #array of lennard jones potential values at different r values
    integrand = 4 * pi * r_points**2 * np.exp(-beta * V_LJ)   #integrand using volume element (change of variable) due to spherical coordinates
    Z_int = np.trapz(integrand, r_points)    #integration using trapezoidal rule
    Z_total=Z_int*(1/(lamba_val**6)) #multiply partition function with 1/thermal wavelenght^6
  
    return Z_total

def find_Z_values(T):
    """
    Calculate partititon function of values over a range of temperatures

    Parameters:
    T(float): Temperature in Kelving at which partition function is evaluated

    Returns:
    Z_values(array): Array of partition functions for each temperature 
    """
    Z_values = []
    for T in T:
        Z_T = total_partition_function(T)  #Call function to calculate partition function
        Z_values.append(Z_T)            #append values   
    Z_values = np.array(Z_values)        #Create array

    return Z_values

#Temperature range
T_range = np.linspace(10, 1000, 1000)

#Find Partition functions over temperature range
Z_values=find_Z_values(T_range)


#Create data for csv file
df = pd.DataFrame({
    'Temperature': T_range,
    'Z_values': Z_values,
})

#Save data to excel file
df.to_csv('homework-4-grad/Partition Function vs Temperature.csv', index=False)
print("CSV file 'Partition Functions_vs_Temperature.csv' created successfully.")

