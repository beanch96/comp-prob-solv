import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import pandas as pd

def compute_work_adiabatic(V_i, V_f, P_i, gamma, num_points=1000):
    """
    Compute the work done during an adiabatic expansion.

    Parameters:
    V_i (float): Initial volume of the gas.
    V_f (float): Final volume of the gas.
    P_i (float): Initial pressure of the gas.
    gamma (float): Adiabatic variable.
    num_points (int): Number of points.

    Returns:
    float: The work done during the adiabatic process.
    """
    constant = P_i * V_i ** gamma
    V =np.linspace(V_i, V_f, num_points)
    P =constant / V ** gamma
    work = -trapezoid(P, V)
    return work

# Values
n = 1  # mol
R = 8.314  # J/(mol*K)
T = 300  # K
V_i = 0.1  # m^3
gamma = 1.4

P_i = n * R * T / V_i

# Final volumes from V_i to 3V_i
V_f_values = np.linspace(V_i, 3 * V_i, 100)

work_adiabatic = [compute_work_adiabatic(V_i, V_f, P_i, gamma) for V_f in V_f_values]

df = pd.DataFrame({
    'Final Volume (m^3)': V_f_values,
    'Work Done (J)': work_adiabatic
})

df.to_csv('homework-4-1/adiabatic_work.csv', index=False)
print("CSV file 'adiabatic_work.csv' created successfully.")