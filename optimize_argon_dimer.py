import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

angstrom = "Å"

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
    if r == 0:
        raise ValueError("Distance r cannot be zero.")
    
    potential= 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    
    return potential

# Objective function wrapper for optimization (only takes r)
def objective_function(r):
    return lennard_jones(r[0])

initial_guess = [4.0]

# Minimize the Lennard-Jones potential
result = minimize(objective_function, initial_guess)

# Extract the optimized value of r
optimal_r = result.x[0]
optimal_potential = result.fun

# Generate r values between 3 and 6 A
r_values = np.linspace(3.0, 6.0, 500)
potential_values = [lennard_jones(r) for r in r_values]

# Plot the Lennard-Jones potential
plt.plot(r_values, potential_values, label='Lennard-Jones Potential')
plt.axvline(x=optimal_r, color='r', linestyle='--', label=f'Equilibrium Distance = {optimal_r:.4f} {angstrom}')
plt.scatter(optimal_r, lennard_jones(optimal_r), color='r', zorder=5)

# Labels and title
plt.xlabel(f'Distance r ({angstrom})')
plt.ylabel('Potential Energy V(r)')
plt.title('Lennard-Jones Potential for Argon Dimer')
plt.legend()
plt.grid(True)
plt.show()

