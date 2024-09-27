#Code to find the distance between three Ar atoms that minimizes the Lennard-Jones potential,
#and optimize its geometry
# 
#Lennard-Jones potential is contained in a python file "Functions.py", which is imported into the program.  

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import Functions as fn
import os

angstrom = "Å"

# This function calculates the distances between three atoms in a trimer system,
# using the provided coordinates. It then computes the total potential energy as the
# sum of the Lennard-Jones potentials for each pair of atoms.

def total_potential(variables):
    """
    Computes the total Lennard-Jones potential energy of a trimer system.

    Parameters:
    variables (list): A list containing the following elements:
        - variables[0] (float): The distance between Atom 1 and Atom 2 (r12).
        - variables[1] (float): The x-coordinate of Atom 3.
        - variables[2] (float): The y-coordinate of Atom 3.

    Returns:
    float: The total Lennard-Jones potential energy of the trimer system.
    """
    r12 = variables[0]  # Distance between Atom 1 and Atom 2
    x3, y3 = variables[1], variables[2]  # Coordinates of Atom 3
    
    r13 = np.sqrt(x3**2 + y3**2)  # Distance between Atom 1 and Atom 3
    r23 = np.sqrt((x3 - r12)**2 + y3**2)  # Distance between Atom 2 and Atom 3
    
    # Total Lennard-Jones potential energy
    V_total = fn.lennard_jones(r12) + fn.lennard_jones(r13) + fn.lennard_jones(r23)
    return V_total


guess = [4.0, 4.0, 4.0]
result = minimize(total_potential, guess) # Use minimize function to optimize the geometry of Ar3

# Extract the optimized geometry
min_r12 = result.x[0]
min_x3 = result.x[1]
min_y3 = result.x[2]
min_potential = result.fun

# Output the results
print(f"Optimized distance r12: {min_r12:.4f} {angstrom}")
print(f"Optimized position of Atom 3: (x3, y3) = ({min_x3:.4f}, {min_y3:.4f}) {angstrom}")

# Coordinates after optimization
atom1 = [0.0, 0.0, 0.0]               
atom2 = [min_r12, 0.0, 0.0]        
atom3 = [min_x3, min_y3, 0.0]  

# Use the compute_bond_angle function to calculate angles
angle_123 = fn.compute_bond_angle(atom1, atom2, atom3)
angle_213 = fn.compute_bond_angle(atom2, atom1, atom3)
angle_132 = fn.compute_bond_angle(atom1, atom3, atom2)

print(f"Angle between Atom 1 - Atom 2 - Atom 3: {angle_123:.2f}°")
print(f"Angle between Atom 2 - Atom 1 - Atom 3: {angle_213:.2f}°")
print(f"Angle between Atom 1 - Atom 3 - Atom 2: {angle_132:.2f}°")

# Use the compute_bond_length function to calculate angles
bond_length_12 = fn.compute_bond_lenght(atom1, atom2)  
bond_length_13 = fn.compute_bond_lenght(atom1, atom3)  
bond_length_23 = fn.compute_bond_lenght(atom2, atom3)  

print(f"Bond length between Atom 1 and Atom 2: {bond_length_12:.3f} {angstrom}")
print(f"Bond length between Atom 1 and Atom 3: {bond_length_13:.3f} {angstrom}")
print(f"Bond length between Atom 2 and Atom 3: {bond_length_23:.3f} {angstrom}")

#Create ".xyz" file
directory = 'homework-2-1'
filename = 'Argon_trimer.xyz'

if not os.path.exists(directory):          #Create Directory
    os.makedirs(directory)

# Write the XYZ file
with open(os.path.join(directory, filename), 'w') as file:
    # Number of atoms
    file.write(f"3\n")
    # Comment line
    file.write(f"Argon Trimer (Lennard-Jones optimization)\n")
    # Atom coordinates
    file.write(f"Ar {atom1[0]:.6f} {atom1[1]:.6f} {atom1[2]:.6f}\n")
    file.write(f"Ar {atom2[0]:.6f} {atom2[1]:.6f} {atom2[2]:.6f}\n")
    file.write(f"Ar {atom3[0]:.6f} {atom3[1]:.6f} {atom3[2]:.6f}\n")
#Indicate succesful operation   
print(f"XYZ file '{filename}' has been created in the directory '{directory}'.")

#Comment on molecular geometry
Comment="""As we can see, the bond angles are 60°, and the bond length between each atom is 3.816 Å,
therefore we can conlcude that the argon trimer forms an optimzed arrangement of an equilateral triangle"""

print(Comment)
