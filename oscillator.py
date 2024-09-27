#Oscillator.py consist on evaluating the eigenvalues and wavefunctions
#of a 1-Dimensional harmonic and anharmonic oscillator. 

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import Functions as fn
import os

#Definition of variables in atomic units 
h=1
m=1
w=1
D=10
B=np.sqrt(1/(2*D))
x_0=0                         
L = 40  
n = 2000

x = np.linspace(-L/2, L/2, n)            #Discretizing space

def Harmonic_Pot(x,w,m):
    """
    Computes the harmonic potential and returns it as a diagonal matrix.

    Parameters:
    x (float or array-like): Space range.
    w (float): Angular frequency of the harmonic oscillator.
    m (float): Mass of the particle.

    Returns:
    numpy.ndarray: A diagonal matrix representing the harmonic potential.
    """
    harmonic=0.5 * m * w**2 * x**2
    matrix=np.diag(harmonic)
    return matrix

def Morse_Pot(x, D, B, x0):
    """
    Computes the Morse potential and returns it as a diagonal matrix.

    Parameters:
    x (float or array-like): Space range.
    D (float): Depth of the potential well, representing bond dissociation energy.
    B (float): Width parameter of the potential
    x0 (float): Equilibrium bond length.

    Returns:
    numpy.ndarray: A diagonal matrix representing the Morse potential.
    """
    morse=D*(1-np.exp(-B*(x-x0)))**2
    matrix=np.diag(morse)
    return matrix

def H_harmonic_fn(Laplacian, Harmonic_Pot):
    """
    Computes the Hamiltonian for a harmonic oscillator system.

    Parameters:
    Laplacian (function): A function to compute the Laplacian of the wavefunction.
    Harmonic_Pot (function): A function to compute the harmonic potential.

    Returns:
    numpy.ndarray: A matrix representing the Hamiltonian for the harmonic oscillator.
    """
    V_harmonic=Harmonic_Pot(x,w,m)
    H_harmonic= (-h**2/(2*m)) * Laplacian(n,L) + V_harmonic
    return H_harmonic

def H_anharmonic_fn(Laplacian, Morse_Pot):
    """
    Computes the Hamiltonian for an anharmonic oscillator system.

    Parameters:
    Laplacian (function): A function to compute the Laplacian of the wavefunction.
    Morse_Pot (function): A function to compute the anharmonic potential.

    Returns:
    numpy.ndarray: A matrix representing the Hamiltonian for the anharmonic oscillator.
    """
    V_anharmonic=Morse_Pot(x,D,B,x_0)
    H_anharmonic= (-h**2/(2*m)) * Laplacian(n,L) + V_anharmonic
    return H_anharmonic

#Calculate the hamiltonian of both potentials
#Laplacian function called from Functions.py
H_harmonic= H_harmonic_fn(fn.Laplacian, Harmonic_Pot)          
H_anharmonic = H_anharmonic_fn(fn.Laplacian, Morse_Pot)

eigenvalues, eigenvectors = np.linalg.eig(H_harmonic)          #Calculate eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)                       #Sort eigenvalues by their indices
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

first_10_eigenvalues = eigenvalues[:10]                         
first_10_eigenvectors = eigenvectors[:, sorted_indices[:10]]

#Output first 10 energy levels
print("Harmonic Potential Eigenvalues:")
print(first_10_eigenvalues)


V_harmonic=np.diag(Harmonic_Pot(x,w,m))                          #Reconstruct Harmonic potential by diagonalizing it-use in plot
dx= (x[1]-x[0])                                                         #Find volume of integration (1D)

plt.plot(x, V_harmonic, color="black", linestyle='-') 

for i in range(10):                                                      #Plot First 10 wavefunctions
    wavefunction= eigenvectors[:, i]                                   #Extract eigenvector i
    wavefunction= wavefunction/(np.sqrt(np.sum(wavefunction**2)*dx))    #Normalize Wavefunction
    y=wavefunction+eigenvalues[i]                                     #Multiply wavefunction by weight to ease visualization and add energy level
    plt.plot(x,y, label=f"v= {i}")                                    #Plot wavefunctions
    plt.axvline(x=0, color='black', linestyle='--')                   #Plot vertical line at x=0
    plt.hlines(eigenvalues[i],-25, 25, color="black")                 #Plot energy Level
    

plt.title("Wavefunctions of Harmonic Oscillator")
plt.xlabel("x")
plt.ylabel("Wavefunction")
plt.ylim(0,11)
plt.legend()

#Save plot to "homework-2-grad" folder
directory = 'homework-2-grad'                                         
if not os.path.exists(directory):                                              #Create directory 
    os.makedirs(directory)
plot_filename = os.path.join(directory, 'Wavefunctions_Harmonic_Pot.png')      #Name File
plt.savefig(plot_filename)                                                     #Save File

print(" Harmonic Potential Plot saved to Directory:", directory)               #Output job complete

eigenvalues_anharmonic, eigenvectors_anharmonic = np.linalg.eig(H_anharmonic)               #Calculate eigenvalues and eigenvectors
sorted_indices_anharmonic = np.argsort(eigenvalues_anharmonic)                   #Sort eigenvalues by their indices
eigenvalues_anharmonic = eigenvalues_anharmonic[sorted_indices_anharmonic]
eigenvectors_anharmonic = eigenvectors_anharmonic[:, sorted_indices_anharmonic]


first_10_eigenvalues_anharmonic = eigenvalues_anharmonic[:10]                         
first_10_eigenvectors_anharmonic = eigenvectors_anharmonic[:, sorted_indices[:10]]

#Output first 10 energy levels
print("Anharmonic Potential Eigenvalues:")
print(first_10_eigenvalues_anharmonic)

V_anharmonic=np.diag(Morse_Pot(x,D,B,0.0))                             #Reconstruct Harmonic potential by diagonalizing it-use in plot

plt.clf()                                                  #Remove previous figure to plot new figure
plt.plot(x, V_anharmonic, color="black", linestyle='-')
plt.axvline(x=0.0, color='black', linestyle='--')

dx= (x[1]-x[0])                                                            #Find volume of integration (1D)
for i in range(10):                                                      #Plot First 10 wavefunctions
    wavefunction= eigenvectors_anharmonic[:, i]                                   #Extract eigenvector i
    wavefunction= wavefunction/(np.sqrt(np.sum(wavefunction**2)*dx))    #Normalize Wavefunction
    y=wavefunction+eigenvalues_anharmonic[i]                                   #Multiply wavefunction by weight to ease visualization and add energy level
    plt.plot(x,y, label=f"v= {i}")                                    #Plot wavefunctions
    plt.hlines(eigenvalues_anharmonic[i],-20, 20, color="black")                 #Plot energy Level

plt.title("Wavefunctions Anharmonic Oscillator")
plt.xlabel("x")
plt.ylabel("Wavefunction")
plt.ylim(0,10)
plt.legend()

#Save plot to "homework-2-grad" folder
directory = 'homework-2-grad'
if not os.path.exists(directory):
    os.makedirs(directory)
plot_filename = os.path.join(directory, 'Wavefunctions_Anharmonic_Pot.png')
plt.savefig(plot_filename)

print(" Anharmonic Potential Plot saved to Directory:", directory)