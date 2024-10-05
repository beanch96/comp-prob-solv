import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import pandas as pd
import os


def compute_work_isothermal(V_i, V_f, n, T, num_points=1000):
    """
    Copmute the work done during an isothermal expansion.

    Parameters:
    V_i (float): Initial volume of the gas.
    V_f (float): Final volume of the gas.
    n (float): Number of moles.
    T (float): Temperature of the gas (K).
    num_points (int): Number of points.

    Returns:
    float: The work done during the isothermal process.
    """
    R = 8.314
    V = np.linspace(V_i, V_f, num_points)
    P = n * R * T / V
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

# Calculate work for both processes
work_isothermal = [compute_work_isothermal(V_i, V_f, n, T) for V_f in V_f_values]


df = pd.DataFrame({
    'Final Volume (m^3)': V_f_values,
    'Work Done (J)': work_isothermal
})

df.to_csv('homework-4-1/isothermal_work.csv', index=False)
print("CSV file 'isothermal_work.csv' created successfully.")