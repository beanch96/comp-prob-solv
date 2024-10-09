import Functions as fn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os


data = pd.read_csv("trouton.csv")  #read data from csv file

#Create dictionary of different class of substances in the file
#Used to differentiate data in order to plot with different colors
class_dictionary = {
    "Perfect liquids": "PL",
    "Imperfect liquids": "IL",
    "Liquids subject to quantum effects": "QL",
    "Metals": "M"
}

# Extract Temperature and Enthalpy
Tb1 = data.iloc[:, 2]  # Boiling Point in K
Hv1 = data.iloc[:, 3]  # Enthalpy in kcal/mol
Hv1 = 4184 * Hv1  # Convert enthalpy to J/mol

# Define the objective function: Sum of Squared Residuals (SSR)
def objective_function(params, Tb, Hv):
    a, b = params  # parameters (slope, intercept)
    predicted_Hv = a * Tb + b
    residuals = Hv - predicted_Hv
    ssr = np.sum(residuals ** 2)
    return ssr

# Initial guess for slope and intercept 
initial_guess = [1.0, 1.0]

# Use scipy.optimize.minimize to find the optimal parameters
result = minimize(objective_function, initial_guess, args=(Tb1, Hv1))

# Extract the optimal parameters
optimal_slope, optimal_intercept = result.x
print(f"Optimal Slope (a): {optimal_slope}")
print(f"Optimal Intercept (b): {optimal_intercept}")

#For plot convenience, determine sign of intercept y=mx+/-b
if optimal_intercept<0:
    sign="-"
else:
    sign="+"


#Create Figure
plt.figure(figsize=(6,4))
for class_name in class_dictionary.keys():
    class_data = data[data["Class"]== class_name]   #itirate over all types of substances
    Tb, Hv = fn.extract_data(class_data)  #call function to extract temperature and enthalpy(converted to J/mol) from each class
    plt.scatter(Tb, Hv, label=class_dictionary[class_name])  #plot all Hv vs T, with each class a different color as is different type of data 
    

#Plot regression line in blue
plt.plot(Tb1,optimal_slope*Tb1+optimal_intercept,"b-")
plt.annotate(f"H$_{{v}}$ = {optimal_slope:.2f}T$_{{B}}$ {sign} {abs(optimal_intercept):.2f}",
             xy=(1000,100000),
             xytext=(0,150000),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
              bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=1))


plt.xlabel('Boiling Point (K)')
plt.ylabel('Enthalpy of Vaporization (J/mol)')
plt.title("Trouton's Rule Optimization")
plt.legend()
ax = plt.gca()  # Get the current axis
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))   #Make y-axis in scientific notation to ease visibility

#Save plot to "homework-3-2" folder
directory = 'homework-3-2'
if not os.path.exists(directory):            #Create Directory
    os.makedirs(directory)
plot_filename = os.path.join(directory, "Trouton's Rule Optimization.png")  #Title of file
plt.savefig(plot_filename)
#indicate succesful operation 
print("Plot saved to Directory:", directory)
