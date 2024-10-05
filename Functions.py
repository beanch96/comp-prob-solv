#Important functions used for Computational Chemistry Class Assignment 

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

def ols_slope(x, y):
    """
    Calculate the slope of the Ordinary Least Squares (OLS) regression line.

    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.

    Returns:
    float: The slope of the regression line.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator

def ols_intercept(x, y):
    """
    Calculate the intercept of the Ordinary Least Squares (OLS) regression line.

    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.

    Returns:
    float: The intercept of the regression line.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = ols_slope(x, y)
    return y_mean - slope * x_mean

def ols(x, y):
    """
    Calculate the slope and intercept of the Ordinary Least Squares (OLS) regression line.

    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.

    Returns:
    tuple: A tuple containing the slope and intercept of the regression line.
    """
    slope = ols_slope(x, y)
    intercept = ols_intercept(x, y)
    return slope, intercept


def se_slope(x, variance):
    """
    Calculate the standard error of the slope in an Ordinary Least Squares (OLS) regression.

    Parameters:
    x (array-like): The independent variable values.
    variance (float): The variance of the residuals.

    Returns:
    float: The standard error of the slope.
    """
    numerator = variance
    x_mean = np.mean(x)
    denominator = np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)


def se_intercept(x, variance):
    """
    Calculate the standard error of the intercept in an Ordinary Least Squares (OLS) regression.

    Parameters:
    x (array-like): The independent variable values.
    variance (float): The variance of the residuals.

    Returns:
    float: The standard error of the intercept.
    """
    numerator = variance
    x_mean = np.mean(x)
    denominator = len(x) * np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)


from scipy.stats import t

def confidence_interval_slope(x, variance, confidence_level):
    """
    Calculate the confidence interval for the slope and intercept in an Ordinary Least Squares (OLS) regression.

    Parameters:
    x (array-like): The independent variable values.
    variance (float): The variance of the residuals.
    confidence_level (float): The desired confidence level (e.g., 0.95 for a 95% confidence interval).

    Returns:
    tuple: A tuple containing the confidence interval of the slope and intercept.
    """
    slope_error = se_slope(x, variance)
    int_error = se_intercept(x, variance)
    n_data_points = len(x)
    df = n_data_points - 2  # degrees of freedom
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)
    Conf_int_slope= critical_t_value * slope_error
    Conf_int_int=  critical_t_value * int_error
    
    return Conf_int_slope, Conf_int_int


def extract_data(class_data):
    """
    Extract temperature and enthalpy data from a DataFrame.

    Parameters:
    class_data (DataFrame): The DataFrame containing the class data. The temperature is expected in the 
    3rd column, and the enthalpy in the 4th column.

    Returns:
    tuple: A tuple containing the temperature nd enthalpy as two separate arrays.
    """
    Hv=class_data.iloc[:,3] #enthalpy in kcal/mol
    Tb=class_data.iloc[:,2] #Temperature in K
    Hv=4184*Hv  #enthalpy in J/mol
    return Tb, Hv