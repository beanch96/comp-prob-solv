import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, eV
import os
import pandas as pd

def partition_function_NO_SOC(degeneracies, energies, temperature):
    """
    Compute the partition function for an isolated system(no splitting).

    Parameters:
    degeneracies (int): degeneracy for ground state.
    energies (int): Energy for ground state.
    temperature (float): The temperature of the system in Kelvin.

    Returns:
    array-like: The partition function for the system.
    """
    k_B = k / eV
    Z = degeneracies * np.exp(-np.array(energies) / (k_B * temperature))
    return Z

def partition_function_SOC(degeneracy1,degeneracy2, energy1, energy2, temperature):
    """
    Compute the partition function for a system with two energy levels due to spin-orbit coupling.

    Parameters:
    degeneracy1 (float): The degeneracy of the first energy level.
    degeneracy2 (float): The degeneracy of the second energy level.
    energy1 (float): The energy of the first level (in eV).
    energy2 (float): The energy of the second level (in eV).
    temperature (float): The temperature of the system in Kelvin.

    Returns:
    float: The partition function of the system.
    """
    k_B = k / eV
    Za = degeneracy1 * np.exp(-np.array(energy1) / (k_B * temperature))
    Zb=degeneracy2 * np.exp(-np.array(energy2) / (k_B * temperature))
    Z=Za+Zb
    return Z

def partition_function_SOC_CFS(degeneracies,energies, temperature):
    """
    Compute the partition function for a system with spin-orbit coupling and crystal field splitting.

    Parameters:
    degeneracies (array-like): An array of degeneracies for each energy level.
    energies (array-like): An array of energy levels (in eV).
    temperature (float): The temperature of the system in Kelvin.

    Returns:
    float: The partition function of the system.
    """
    k_B = k / eV
    Z=np.sum(degeneracies*np.exp(-np.array(energies) / (k_B * temperature)))
    return Z

#-------------ISOLATED CE3+---------------------------------------------------------------------------------------------------------#

degeneracies_NO_SOC = 14 
energies_NO_SOC = 0 
T = np.linspace(300, 2000, 1000)  

k_B = k / eV

# Calculate the partition function for each temperature by using a for loop
Z_Ce3plus_NO_SOC = np.array([partition_function_NO_SOC(degeneracies_NO_SOC, energies_NO_SOC, temp) for temp in T]) 
#Energy Calculations
U_NO_SOC = -np.gradient(np.log(Z_Ce3plus_NO_SOC), 1 / (k_B * T))  #internal energy
F_NO_SOC=-k_B*T*np.log(Z_Ce3plus_NO_SOC)  #Free-energy
S_NO_SOC= -np.gradient(F_NO_SOC,T)   #Entropy

#----------SOC CE3+-------------------------------------------------------------------------------------------------------------------#

Energy1_SOC=0
Energy2_SOC=0.28
degeneracy1_SOC=6
degeneracy2_SOC=8
T = np.linspace(300, 2000, 1000)  

#Calulcate partition function for each temperature
Z_Ce3plus_SOC = np.array([partition_function_SOC(degeneracy1_SOC,degeneracy2_SOC, Energy1_SOC, Energy2_SOC, temp) for temp in T])
U_SOC = -np.gradient(np.log(Z_Ce3plus_SOC), 1 / (k_B * T))   #internal energy
F_SOC=-k_B*T*np.log(Z_Ce3plus_SOC)         #free-energy
S_SOC= -np.gradient(F_SOC,T)               #entropy

#-------------SOC and CFS CE3+------------------------------------------------------------------------------------------------------#

Energies_SOC_CFS=[0,0.12,0.25,0.32,0.46]
degeneracies_SOC_CFS=[4,2,2,4,2]

T = np.linspace(300, 2000, 1000)  
#Calulate partition function for each temperature
Z_Ce3plus_SOC_CFS = np.array([partition_function_SOC_CFS(degeneracies_SOC_CFS,Energies_SOC_CFS,temp) for temp in T])
U_SOC_CFS = -np.gradient(np.log(Z_Ce3plus_SOC_CFS), 1 / (k_B * T))  #internal energy
F_SOC_CFS=-k_B*T*np.log(Z_Ce3plus_SOC_CFS)          #free-energy
S_SOC_CFS= -np.gradient(F_SOC_CFS,T)          #entropy


#---------------Create Excel file----------------------------------------------------------------------------------------------#

#Create data for csv file
df = pd.DataFrame({
    'Temperature': T,
    'Internal Energy_Isolated': U_NO_SOC,
    'Free Energy-Isolated': F_NO_SOC,
    'Entropy-Isolated': S_NO_SOC,
    'Internal Energy_SOC': U_SOC,
    'Free Energy_SOC': F_SOC,
    'Entropy_SOC': S_SOC,
    'Internal Energy_SOC_CFS': U_SOC_CFS,
    'Free Energy_SOC_CFS': F_SOC_CFS,
    'Entropy_SOC_CFS': S_SOC_CFS
})

#Save data to excel file
df.to_csv('homework-4-2/Energies_vs_Temperature.csv', index=False)
print("CSV file 'Energies_vs_Temperature.csv' created successfully.")

#---------------Create Plots----------------------------------------------------------------------------------------------#
plt.figure(figsize=(6, 4))
plt.plot(T, U_SOC_CFS, color='green',label='U-SOC&CFS')
plt.plot(T,U_NO_SOC,color="blue",label='U-Isolated')
plt.plot(T,U_SOC,"--",color="brown",label='U-SOC')
plt.xlabel('Temperature (K)')
plt.ylabel('Internal Energy (eV)')
plt.title('Internal Energy Comparison for Ce$^{3+}$')
plt.legend()
#Save plot to "homework-4-2"
directory = 'homework-4-2'
if not os.path.exists(directory):            #Create Directory
    os.makedirs(directory)
plot_filename = os.path.join(directory, "Internal Energy vs T.png")  #Title of file
plt.savefig(plot_filename)
#indicate succesful operation 
print("Internal Energy comparison Plot saved to Directory:", directory)

plt.figure(figsize=(6, 4))
plt.plot(T, F_SOC_CFS, color='green',label='F-SOC&CFS')
plt.plot(T,F_NO_SOC,color="blue",label='F-Isolated')
plt.plot(T,F_SOC,"--",color="brown",label='F-SOC')
plt.xlabel('Temperature (K)')
plt.ylabel('Internal Energy (eV)')
plt.title('Free Energy Comparison for Ce$^{3+}$')
plt.legend()
#Save plot to "homework-4-2"
directory = 'homework-4-2'
if not os.path.exists(directory):            #Create Directory
    os.makedirs(directory)
plot_filename = os.path.join(directory, "Free Energy vs T.png")  #Title of file
plt.savefig(plot_filename)
#indicate succesful operation 
print("Free Energy comparison Plot saved to Directory:", directory)


plt.figure(figsize=(8, 6))
plt.plot(T, S_SOC_CFS, color='green',label='S-SOC&CFS')
plt.plot(T,S_NO_SOC,color="blue",label='S-Isolated')
plt.plot(T,S_SOC,"--",color="brown",label='S-SOC')
plt.xlabel('Temperature (K)')
plt.ylabel('Internal Energy (eV)')
plt.title('Entropy Comparison for Ce$^{3+}$')
plt.legend()
#Save plot to "homework-4-2"
directory = 'homework-4-2'
if not os.path.exists(directory):            #Create Directory
    os.makedirs(directory)
plot_filename = os.path.join(directory, "Entropy vs T.png")  #Title of file
plt.savefig(plot_filename)
#indicate succesful operation 
print("Entropy  comparison Plot saved to Directory:", directory)