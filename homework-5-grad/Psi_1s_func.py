import numpy as np
from scipy.constants import pi
from scipy.stats import norm

def psi_1s(x, y, z, Z=1, ao=1):
    """
    Compute the 1s orbital wavefunction at a point (x,y,z)

    Parameters:
     x (float or array) : x-coordinate
     y (float or array) : y-coordinate 
     z (float or array) : z-coordinate 
    
     Returns:
     float or array: Value of the 1s wavefunction at x,y,z
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    Psi_1s = (1/np.sqrt(pi * ao**3)) * np.exp(-r / ao)
    return Psi_1s

def laplacian_psi_1s(x, y, z, Z=1, a0=1):
    """
    Compute the Laplacian of the hydrogen 1s orbital at a given point (x, y, z).

    Parameters:
    x (float or array): x-coordinate
    y (float or array): y-coordinate
    z (float or array): z-coordinate
    Z (int): Atomic number of the nucleus
    a0 (float): Bohr radius

    Returns:
    float or array: The value of the Laplacian of the 1s orbital at the given point.
    """

    r = np.sqrt(x**2 + y**2 + z**2)
    psi = psi_1s(x, y, z, Z, a0)

    # First derivative
    d_psi_dr = -np.exp(-r)/np.sqrt(pi)

    # Second derivative
    d2_psi_dr2 = np.exp(-r)/np.sqrt(pi)

    # Laplacian of the 1s orbital
    laplacian = d2_psi_dr2 + (2 / r) * d_psi_dr
    return laplacian

def random_integration(L,N,R):
    """
    Compute Monte Carlo integration using random points for the calculation of kinetic energy matrix of two 1s orbitals

    Parameters:
    L (int): Length of box
    N (int): Number of points for grid
    R (int): Separation distance between orbitals (0 being no separation)

    Returns:
    float: integration
    float: standard deviation
    """
    # Parameters
    np.random.seed(42)
    Z = 1  # Atomic number for hydrogen
    a0 = 1  # Bohr radius in atomic units
    V = (2 * L)**3  # Volume of the cubic region
   
    # Generate random points in the cubic region
    x = np.random.uniform(-L, L, N)
    y = np.random.uniform(-L, L, N)
    z = np.random.uniform(-L, L, N)
    
    # Compute the integrand at each point
    psi = psi_1s(x, y, z+(R/2), Z, a0)
    laplacian_psi = laplacian_psi_1s(x, y, z-(R/2), Z, a0)
    integrand = -0.5 * psi * laplacian_psi

    # Monte Carlo estimation of the integral
    mean_integrand = np.mean(integrand)
    variance_integrand = np.var(integrand)
   
        # Estimate K_ii
    Kii = V * mean_integrand                          #With random sampling, we are only calculating 1 cubic domain, therefore we need to multiply it by 8 for full integration
    std_dev = V * np.sqrt(variance_integrand / N)

    return Kii,std_dev

def important_integration(L,N,R):
    """
    Compute Monte Carlo integration using important sampling points for the calculation of kinetic energy matrix of two 1s orbitals

    Parameters:
    L (int): Length of box
    N (int): Number of points for grid
    R (int): Separation distance between orbitals (0 being no separation)

    Returns:
    float: integration
    float: standard deviation
    """

    np.random.seed(42)
    Z = 1  # Atomic number for hydrogen
    a0 = 1  # Bohr radius in atomic units

    scale_Factor=1.0

    # Generate random points in the cubic region
    x = norm.rvs(size=N, scale=scale_Factor)
    y = norm.rvs(size=N, scale=scale_Factor)
    z = norm.rvs(size=N, scale=scale_Factor)

    # Compute the integrand at each point
    psi = psi_1s(x, y, z+(R/2), Z, a0)
    laplacian_psi = laplacian_psi_1s(x, y, z-(R/2), Z, a0)
    numer = -0.5 * psi * laplacian_psi
    denom = denom=norm.pdf(x,scale=scale_Factor) * norm.pdf(y,scale=scale_Factor) * norm.pdf(z,scale=scale_Factor)
    integrand=numer/denom

    # Monte Carlo estimation of the integral
    mean_integrand = np.mean(integrand)              #There is no need to multiply it with volume because a gaussian function is sampling from all space (negative and positive)
    variance_integrand = np.var(integrand)            #If using another function like an exponential make sure to multiply with appropiate symmetry. 
    # Estimate K_ii
    Kii = mean_integrand
    std_dev = np.sqrt(variance_integrand / N)


    return Kii, std_dev
 



