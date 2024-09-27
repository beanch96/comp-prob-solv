#Code to find the distance between two Ar atoms that minimizes the Lennard-Jones potential
# and plot the potential energy curve
# 
#Lennard-Jones potential is contained in a python file "Functions.py", which is imported into the program.  

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import Functions as fn
import os

angstrom = "Å"   #Define angstrom symbol

# This function is a way to ensure that `fn.lennard_jones` function to adapt 
# its input to be compatible with optimization routines that require an array-like input, 
# such as "minimize(fxn,initial guess)".

def objective_function(r):
    """
    Computes the Lennard-Jones potential for a given distance.

    Parameters:
    r (array): A list or array containing the interatomic distance at index 0.

    Returns:
    float: The Lennard-Jones potential for the given distance.
    """
    return fn.lennard_jones(r[0])

guess = [4.0]
result = minimize(objective_function, guess)  # Minimize the Lennard-Jones potential
optimized_r = result.x[0]                     # Extract the optimized value of r

print(f"Optimized bond distance r12: {optimized_r:.4f} {angstrom}")
r_values = np.linspace(3.0, 6.0, 500)         #values between 3 and 6
potential_values = [fn.lennard_jones(r) for r in r_values]

# Plot the Lennard-Jones potential
plt.plot(r_values, potential_values, label='Lennard-Jones Potential')
plt.axvline(x=optimized_r, color='r', linestyle='--', label=f'Equilibrium Distance = {optimized_r:.4f} {angstrom}')
plt.scatter(optimized_r, fn.lennard_jones(optimized_r), color='r', zorder=5)
plt.xlabel(f'Distance r ({angstrom})')
plt.ylabel('Potential Energy V(r)')
plt.title('Lennard-Jones Potential for Argon Dimer')
plt.legend()
plt.grid(True)

#Save plot to "homework-2-1" folder
directory = 'homework-2-1'
if not os.path.exists(directory):            #Create Directory
    os.makedirs(directory)
plot_filename = os.path.join(directory, 'Lennard_Jones_Ar_Dimer.png')  #Title of file
plt.savefig(plot_filename)
#indicate succesful operation 
print("Plot saved to Directory:", directory)

#Create ".xyz" file
directory = 'homework-2-1'
filename = 'Argon_dimer.xyz'
if not os.path.exists(directory):            #Create Directory
    os.makedirs(directory)

# Define the coordinates of the atoms after optimization
atom1 = [0.0, 0.0, 0.0]
atom2 = [optimized_r, 0.0, 0.0]

# Write the XYZ file
with open(os.path.join(directory, filename), 'w') as file:
    # Number of atoms
    file.write(f"2\n")
    # Comment line
    file.write(f"Argon Dimer (Lennard-Jones optimization)\n")
    # Atom coordinates
    file.write(f"Ar {atom1[0]:.6f} {atom1[1]:.6f} {atom1[2]:.6f}\n")
    file.write(f"Ar {atom2[0]:.6f} {atom2[1]:.6f} {atom2[2]:.6f}\n")
#Indicate succesful operation    
print(f"XYZ file '{filename}' has been created in the directory '{directory}'.")

