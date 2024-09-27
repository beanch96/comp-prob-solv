import numpy as np
import math


def compute_bond_angle(coord1, coord2, coord3):
    """
    Computes the bond angle between three atoms based on their 3D coordinates.
    
    Parameters:
    coord1 (list): List of 3D coordinates of atom 1. 
    coord2 (list): List of 3D coordinates of atom 2.
    coord3 (list): List of 3D coordinates of atom 3.
    
    Returns:
    float: bond angle between three atoms in 3D space.
    String: Angle Classification
    """
    #Convert list into array for easier mathematical operations
    A = np.array(coord1)                 
    B = np.array(coord2)
    C = np.array(coord3)
    
    #Calculate vectors AB and BC
    AB=A-B                               
    BC=C-B

    dotProduct= np.dot(AB,BC)            #dot product of vectors
    Mag_AB= np.linalg.norm(AB)           #Calculate magnitude of vectors
    Mag_BC= np.linalg.norm(BC)
    
    #calculate angles
    radians= np.arccos((dotProduct)/(Mag_AB*Mag_BC))      
    degrees= np.degrees(radians)

    return degrees

def compute_bond_lenght(coord1, coord2):
    """
    Computes the bond lenght between two atoms based on their 3D coordinates.
    
    Parameters:
    coord1 (list): List of 3D coordinates of atom 1. 
    coord2 (list): List of 3D coordinates of atom 2.
    
    Returns:
    float: bond lenght between two atoms in 3D space.
    Boolean: True if lenght below 2, False is lenght is above 2
    """
    dx=coord2[0]-coord1[0]
    dy=coord2[1]-coord1[1]
    dz=coord2[2]-coord1[2]

    lenght=math.sqrt(dx**2 + dy**2 + dz**2)                          
    
    return lenght

def lennard_jones(r, epsilon=0.01, sigma=3.4):
    """
    Calculate the Lennard-Jones potential energy between two atoms.

    Parameters:
    r (float): Distance between the two atoms.
    epsilon (float): Depth of the potential well (default is 0.01).
    sigma (float): Distance at which the potential is zero (default is 3.4).

    Returns:
    float: Potential energy V(r).
    """
    # if r == 0:
    #     raise ValueError("Distance r cannot be zero.")
    if np.any(r <= 0):
        raise ValueError("Distance r cannot be zero or negative.")
    potential= 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    
    return potential

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def Laplacian(n,L):
    """
    Computes the Laplacian operator based on number of points in grid.
    
    Parameters:
    n (int): Number of points on the grid. 
    L (int): Range of the space
    
    
    Returns:
    Array: Laplacian Matrix.
    """
    I=np.identity(n)                          #Create an identity Matrix of n points
    sup_diag=np.diag(np.ones(n-1),1)          #Create a super diagonal with "1" above diagonal
    sub_diag=np.diag(np.ones(n-1),-1)         #Create a sub diagonal with "1" below diagonal
    I_off=sup_diag+sub_diag                   #Create off diagonal
    dx=(L / (n-1))                            #Delta x of grid
    laplacian=(1/(dx**2))*(-2*I+I_off)        #Calculate the Laplacian
    return laplacian

