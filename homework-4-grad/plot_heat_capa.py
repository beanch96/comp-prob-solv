
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data= pd.read_csv("homework-4-grad\Internal Energy-Cv-Temperature.csv")    #read data from csv file

#extract data from csv file
T_range = data.iloc[:, 0]  
Cv = data.iloc[:, 2]  

# Find the temperature at which Cv is maximum 
T_dissociation = T_range[np.argmax(Cv)]
print(f"Dissociation temperature: {T_dissociation:.2f} K")

# Plot Cv vs Temperature
plt.figure(figsize=(8,6))
plt.plot(T_range, Cv, label='Cv(T)')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity Cv (J/K)')
plt.axvline(x=T_dissociation, color='r', linestyle='--', label=f'Peak at T={T_dissociation:.2f}k')
plt.title('Heat Capacity vs Temperature for Argon Dimer')
plt.legend()
plt.grid(True)

#Save plot to "homework-4-grad" folder
directory = 'homework-4-grad'
if not os.path.exists(directory):            #Create Directory
    os.makedirs(directory)
plot_filename = os.path.join(directory, "Cv vs T.png")  #Title of file
plt.savefig(plot_filename)
#indicate succesful operation 
print("Plot saved to Directory:", directory)
