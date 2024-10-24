import numpy as np
from scipy.constants import pi
from scipy.stats import expon

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

    # First derivative of psi_1s
    d_psi_dr = -np.exp(-r)/np.sqrt(pi)

    # Second derivative of psi_1s
    d2_psi_dr2 = np.exp(-r)/np.sqrt(pi)

    # Laplacian of the 1s orbital
    laplacian = d2_psi_dr2 + (2 / r) * d_psi_dr
    return laplacian

def random_integration(L,N,R):
    # Parameters
    np.random.seed(42)
    Z = 1  # Atomic number for hydrogen
    a0 = 1  # Bohr radius in atomic units
    V = (2 * L)**3  # Volume of the cubic region
   
    # Generate random points in the cubic region
    x = np.random.uniform(0, L, N)
    y = np.random.uniform(0, L, N)
    z = np.random.uniform(0, L, N)
    
    # Compute the integrand at each point
    psi = psi_1s(x, y, z+(R/2), Z, a0)
    laplacian_psi = laplacian_psi_1s(x, y, z-(R/2), Z, a0)
    integrand = -0.5 * psi * laplacian_psi

    # Monte Carlo estimation of the integral
    mean_integrand = np.mean(integrand)
    variance_integrand = np.var(integrand)
   
        # Estimate K_ii
    Kii = V * mean_integrand
    std_dev = V * np.sqrt(variance_integrand / N)

    return Kii,std_dev

def important_integration(L,N,R):
    np.random.seed(42)
    Z = 1  # Atomic number for hydrogen
    a0 = 1  # Bohr radius in atomic units
    V = 2**3  # Volume of the cubic region

    scale_Factor=1.5

    # Generate random points in the cubic region
    x = expon.rvs(size=N, scale=scale_Factor)
    y = expon.rvs(size=N, scale=scale_Factor)
    z = expon.rvs(size=N, scale=scale_Factor)

    # Compute the integrand at each point
    psi = psi_1s(x, y, z+(R/2), Z, a0)
    laplacian_psi = laplacian_psi_1s(x, y, z-(R/2), Z, a0)
    numer = -0.5 * psi * laplacian_psi
    denom = denom=expon.pdf(x,scale=scale_Factor) * expon.pdf(y,scale=scale_Factor) * expon.pdf(z,scale=scale_Factor)
    integrand=numer/denom

    # Monte Carlo estimation of the integral
    mean_integrand = np.mean(integrand)
    variance_integrand = np.var(integrand)
    # Estimate K_ii
    Kii = V * mean_integrand
    std_dev = V * np.sqrt(variance_integrand / N)


    return Kii, std_dev
 



