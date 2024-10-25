import numpy as np
from scipy.constants import pi
from scipy.stats import expon

def psi_2p_z(x,y,z):
    """
    Calculate the wavefunction of the 2p_z orbital at a given point (x, y, z).

    Parameters:
     x (float or np.array) : x-coordinate 
     y (float or np.array) : y-coordinate 
     z (float or np.array) : z-coordinate 
    
     Returns:
     float or np.array: Value of the 2p_z wavefunction at x,y,z
    """
    r=np.sqrt(x**2+y**2+z**2)  #define r in cartesian coordinates
    cos_theta=z/r              #define cos(theta) in cartesian coordinates
    N=1/(4*np.sqrt(2*pi))      #Normalization constant of 2pz orbital
    rad= (r)*np.exp(-r/(2))    #Radial part of 2pz orbital
    ang=cos_theta              #Angular part of 2pz orbital
    Y2pz=N*rad*ang             #2Pz wavefunction

    return Y2pz

def random_overlap_pz(N,R):
    """
    Calculate overlap integrap of two 2pz orbitals at a given separation distance and number of points to define grid by Monte Carlo method using random sampling. 

    Parameters:
    N (int) : number of points to define x,y,z grid
    R (int) : separation distance in atomic units

    Returns:
    float: The result of the overlap integral 
    float: The standard deviation of the integral
    """    
    np.random.seed(42)  #reproducibility

#limits in the grid-use of symmetry to simplify integration
    a=0
    b=20

    x=np.random.uniform(a,b,N)
    y=np.random.uniform(a,b,N)   #create random distribution in the x,y,z grid
    z=np.random.uniform(a,b,N)
    Y2pz_1=psi_2p_z(x,y,z-(R/2)) #shifting orbital to (0,0,-R/2)
    Y2pz_2=psi_2p_z(x,y,z+(R/2)) #shifting orbital to (0,0, R/2)
    integrand= Y2pz_1*Y2pz_2    #calculate integrand
    integral=np.mean(integrand)*(2*(b-a))**3 #calulate integral by symmetry  
    variance=np.var(integrand)*(2*(b-a))**3
    std_dev=(np.sqrt(variance))

    return integral,std_dev

def important_overlap_pz(N,R):
    """
    Calculate overlap integrap of two 2pz orbitals at a given separation distance and number of points to define grid by Monte Carlo method using important sampling. 

    Parameters:
    N (int) : number of points to define x,y,z grid
    R (int) : separation distance in atomic units

    Returns:
    float: The result of the overlap integral
    float: The standard deviation 
    """  
    np.random.seed(42)  #reproducibility
    scale_Factor=10

    x = expon.rvs(size=N, scale=scale_Factor)
    y = expon.rvs(size=N, scale=scale_Factor)     #use exponential decay function as sampling point
    z = expon.rvs(size=N, scale=scale_Factor)
    Y2pz_1=psi_2p_z(x,y,z-(R/2)) #shifting orbital to (0,0,-R/2)
    Y2pz_2=psi_2p_z(x,y,z+(R/2)) #shifting orbital to (0,0, R/2)
    numer= Y2pz_1*Y2pz_2
    denom=expon.pdf(x,scale=scale_Factor) * expon.pdf(y,scale=scale_Factor) * expon.pdf(z,scale=scale_Factor) #divide by exponential function 
    integrand=numer/denom
    integral=np.mean(integrand)*(2**3) #calulate integral by symmetry
    variance=np.var(integrand)*(2)**3
    std_dev=(np.sqrt(variance))
  
    return integral,std_dev

