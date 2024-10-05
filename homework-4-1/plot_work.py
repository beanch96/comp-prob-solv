import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data_isothermal=pd.read_csv('homework-4-1/isothermal_work.csv')
data_adiabatic=pd.read_csv('homework-4-1/adiabatic_work.csv')

V_f_values= data_isothermal.iloc[:,0]
work_isothermal=data_isothermal.iloc[:,1]
work_adiabatic=data_adiabatic.iloc[:,1]


plt.plot(V_f_values, work_isothermal, label='Isothermal Process')
plt.plot(V_f_values, work_adiabatic, label='Adiabatic Process')
plt.xlabel('Final Volume $V_f$ (m$^3$)')
plt.ylabel('Work Done (J)')
plt.title('Work vs Final Volume')
plt.legend()
plt.grid(True)

#Save plot
directory = 'homework-4-1'
plot_filename = os.path.join(directory, "Work vs Final Volume.png")  #Title of file
plt.savefig(plot_filename)
#indicate succesful operation 
print("Plot saved to Directory:", directory)