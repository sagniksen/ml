# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:42:49 2020

@author: Sagnik Sen
"""

###Simple Linear Regression (y=BO+B1x1+E)
##Importing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

##Load Data

data=pd.read_csv("F:/Python/Simple Linear Regression/SLR.csv")
data

data.describe()

##First Regression
#Defining dependent and independent variables

y=data['GPA']
x1=data['SAT']

#Explore The Data

plt.scatter(x1,y)
plt.xlabel('SAT')
plt.ylabel('GPA')

#Regression
#y=b0+b1x1

x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
results.summary()

#Plotting Simple Regression Line

plt.scatter(x1,y)
yhat=0.2750+0.0017*x1 #Regression Line
fig=plt.plot(x1,yhat,lw=4,c='orange',label='Regression Line')
plt.xlabel('SAT',fontsize='20')
plt.ylabel('GPA',fontsize='20')

