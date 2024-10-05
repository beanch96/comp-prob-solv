#This script extracts thermodynamic from a csv file, classify the data, plot it, fit a linear
#regression analysis with its respective confidence interval

import Functions as fn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

data= pd.read_csv("trouton.csv")    #read data from csv file

#Create dictionary of different class of substances in the file
#Used to differentiate data in order to plot with different colors
class_dictionary = {
    "Perfect liquids": "PL",
    "Imperfect liquids": "IL",
    "Liquids subject to quantum effects": "QL",
    "Metals": "M"
}

#Create Figure
plt.figure(figsize=(6,4))
for class_name in class_dictionary.keys():
    class_data = data[data["Class"]== class_name]   #itirate over all types of substances
    Tb, Hv = fn.extract_data(class_data)  #call function to extract temperature and enthalpy(converted to J/mol) from each class
    plt.scatter(Tb, Hv, label=class_dictionary[class_name])  #plot all Hv vs T, with each class a different color as is different type of data 


#Extract Hv and Tb from data and convert to J/mol to fit regression line and confidence interval
Hv1=data.iloc[:,3] #enthalpy in kcal/mol
Tb1=data.iloc[:,2] #Temperature in K
Hv1=4184*Hv1  #enthalpy in J/mol

#call "ols" function to find slope and intercept of a fit line 
slope1, int1= fn.ols(Tb1,Hv1)
print(f"slope (a): {slope1}")
print(f"int (b): {int1}")

#For plot convenience, determine sign of intercept y=mx+/-b
if int1<0:
    sign="-"
else:
    sign="+"

#Plot regression line in blue
plt.plot(Tb1,slope1*Tb1+int1, "b-")
plt.annotate(f'H$_{{v}}$ = {slope1:.2f}T$_{{B}}$ {sign} {abs(int1):.2f}',
             xy=(1000,100000),
             xytext=(0,150000),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
              bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=1))

#Calculate Confidence interval:

line= slope1*Tb1+int1   #y=mx+b
residuals = Hv1 - line
sm_sq_res = np.sum(residuals**2)   #Calculate sum of squared residuals
variance= sm_sq_res/(len(residuals)-2) #Calculate Variance
#call function to calculate confidence for slope and intercept
Conf_int_slope, Conf_int_int = fn.confidence_interval_slope(Tb1, variance, 0.95) 
#Insert confidence intervals in plot by creating box in the bottom right corner of figure
plt.annotate(f"""95% Confidence Intervals:
Slope: {slope1:.3f} ± {Conf_int_slope:.3f}
Intercept: {int1:.3f} ± {Conf_int_int:.3f}""",
             xy=(1500, 0),
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=1))
              
plt.xlabel('Boiling Point (K)')
plt.ylabel('Enthalpy of Vaporization (J/mol)')
plt.title("Trouton's Rule")
plt.legend()
ax = plt.gca()  # Get the current axis
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))   #Make y-axis in scientific notation to ease visibility

#Save plot to "homework-3-1" folder
directory = 'homework-3-1'
if not os.path.exists(directory):            #Create Directory
    os.makedirs(directory)
plot_filename = os.path.join(directory, "Trouton's Rule.png")  #Title of file
plt.savefig(plot_filename)
#indicate succesful operation 
print("Plot saved to Directory:", directory)