
#Lennard-Jones potential is contained in a python file "Functions.py", which is imported into the program.  

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import Functions as fn
import os

def hard_sphere_potential(r,sigma):
    """
    Computes the hard sphere potential for a given distance.

    Parameters:
    r (float or array-like): The distance between particles.
    sigma (float): The hard sphere diameter.

    Returns:
    numpy.ndarray: An array with potential values according to Hard sphere potential
    """
    return np.where(r < sigma, 1000, 0)
    
def square_well_potential(r,sigma,epsilon,range):
    """
    Computes the square well potential for a given distance.

    Parameters:
    r (float or array-like): The distance between particles.
    sigma (float): The hard sphere diameter.
    epsilon (float): The depth of the attractive well.
    range (float): The range for the well.

    Returns:
    numpy.ndarray: An array with potential values according to the square well potential model.
    """
    return np.where(r < sigma, 1000, np.where(r <= range * sigma, -epsilon, 0)) 
    
def B2V(V,T,r):
    """
    Computes the second virial coefficient for a given potential model using numerical integration (trapezoid).

    Parameters:
    V (array-like): The potential energy values for distances in r_array.
    T (float): The temperature in Kelvin units.
    r (array-like): The distances at which the potential `V` is evaluated.

    Returns:
    float: The second virial coefficient.
    """
    K=8.617e-5
    N=6.022*10**23
    integrand= (np.exp(-V/(K*T))-1)*r**2
    coeff=-2*np.pi*N*trapezoid(integrand,r)
    return coeff

#Define Units
T = 100           
sigma = 3.4 
epsilon = 0.01 
range = 1.5
r_max = 5 * sigma  
r = np.linspace(1e-3, r_max, 1000)  

#Compute optimzed value for a hard-sphere potential
V_hard_sphere = hard_sphere_potential(r, sigma)
B2V_hard_sphere = B2V(V_hard_sphere, T, r)
print(f"B2V optimized value for a hard-sphere potential @100 K: {B2V_hard_sphere}")
#Compute optimzed value for a square-well potential
V_Square_Well = square_well_potential(r, sigma,epsilon,range)
B2V_square_well = B2V(V_Square_Well, T, r)
print(f"B2V optimized value for a Square-Well potential @100 K: {B2V_square_well}")
#Compute optimzed value for a Lennard-Jones potential
V_lennard=fn.lennard_jones(r,epsilon,sigma)
B2V_Lennard=B2V(V_lennard,T,r)
print(f"B2V optimized value for a Lennard-Jones potential @100 K: {B2V_Lennard}")

#Compute B2V for different temperatures

temperatures = np.linspace(100, 800, 8000)  # Range from 100 K to 800 K

# Arrays to store B2V values of different potentials
B2V_hard_sphere_values = []
B2V_lennard_jones_values = []
B2V_square_well_values = []

#Calculate T's
for T in temperatures:
    u_hard_sphere = hard_sphere_potential(r, sigma)
    u_lennard_jones = fn.lennard_jones(r, epsilon, sigma)
    u_square_well = square_well_potential(r, sigma, epsilon, range)
    
    B2V_hard_sphere_values.append(B2V(u_hard_sphere, T, r))
    
    B2V_lennard_jones_values.append(B2V(u_lennard_jones, T, r))
    
    B2V_square_well_values.append(B2V(u_square_well, T, r))

# Create a DataFrame
data = {
        'Temperature (K)': temperatures,
        'B2V Hard Sphere': B2V_hard_sphere_values,
        'B2V Lennard-Jones': B2V_lennard_jones_values,
        'B2V Square Well': B2V_square_well_values
    }
df = pd.DataFrame(data)

#Save csv file to "homework-2-2" folder
directory = 'homework-2-2'
if not os.path.exists(directory):  # Create directory if it doesn't exist
    os.makedirs(directory)
csv_filename = os.path.join(directory, 'B2V_vs_T.csv')  #name file
df.to_csv(csv_filename, index=False) #save csv file
#Indicate succesful operation 
print("CSV file saved to Directory:", directory)


# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(temperatures, B2V_hard_sphere_values, label="Hard-Sphere Potential", color="blue")
plt.plot(temperatures, B2V_lennard_jones_values, label="Lennard-Jones Potential", color="green")
plt.plot(temperatures, B2V_square_well_values, label="Square-Well Potential", color="red")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Temperature (K)")
plt.ylabel("$B_{2V}$ (m$^3$/mol)")
plt.title("$B_{2V}$ as a Function of Temperature \nfor Different Potentials")
plt.legend()
plt.grid(True)

#Save plot to "homework-2-2" folder
directory = 'homework-2-2'
if not os.path.exists(directory):                       #Create directory
    os.makedirs(directory)
plot_filename = os.path.join(directory, 'B2V_vs_T_comparison.png')   #Name File
plt.savefig(plot_filename)                                             #Save File
#Indicate succesful operation    
print("Plot saved to Directory:", directory)

# Create a Markdown file with a brief discussion comparing B2V with different potentials
markdown_content = """
# Comparison of B2V for Different Potentials

## Overview
The plot shows the variation of the second virial coefficient, B2V, as a function of temperature for three different potentials: Hard-Sphere, Lennard-Jones, and Square-Well.

## Hard-Sphere Potential
- **Behavior:** The B2V values for the Hard-Sphere potential remains constant and positive across all temperatures
- **Explanation:** The Hard-Sphere potential considers only repulsive interactions with a fixed diameter, therefore it won't change with temperature. 

## Lennard-Jones Potential
- **Behavior:** The B2V for the Lennard-Jones potential starts around -1.5 m^3/mol at lower temperatures and increases toward postiive values with increasing temperature. 
- **Explanation:** The Lennard-Jones potential includes both attractive and repulsive regions. At low temperatures, the attractive forces dominate(negative B2V), but at high temperatures the repulsive forces dominate(positive B2V)

## Square-Well Potential
- **Behavior:** Similar to the Lennard-Jones potential, the B2V for the Square-Well potential is negative, but it starts at around -2 m^3/mol at low temperatures, but increases and surpasses the Lennard-Jones values as Temperature increase.
- **Explanation:** The Square-Well potential, similarly to Lennard-Jones has an attractive and repulsive region. However, it is not capturing a more realistic behavior like Lennard-Jones potential.

## Summary
- The **Hard-Sphere** potential only models repulsion-therefore is independent of Temperature
- The **Lennard-Jones** and **Square-Well** potentials include both repulsive and attractive regions-therefore there is temperature dependance, with a behaviour of increasing B2V as Temperature increases. 
"""

# Directory and file path to save the Markdown file
directory = 'homework-2-2'
if not os.path.exists(directory):
    os.makedirs(directory)

markdown_file_path = os.path.join(directory, 'B2V_explain.md')

with open(markdown_file_path, 'w') as file:
    file.write(markdown_content)
markdown_file_path
print("Markdown File saved to Directory:", directory)
