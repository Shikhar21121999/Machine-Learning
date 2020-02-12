# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:23:11 2020

@author: LAPPY jr
"""
#My Random Forest Regression

#importin Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the dataset
ds=pd.read_csv('Position_Salaries.csv')
X=ds.iloc[:,1:-1].values
X=X.reshape(-1,1)
Y=ds.iloc[:,2].values
Y=Y.reshape(-1,1)

#Building The random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor as RFR
regressor=RFR(n_estimators=400,random_state=0)
regressor.fit(X,Y)

#Predicting the new results
y_pred=regressor.predict(np.array(6.5).reshape(-1,1))

#Visualizing the Regrression results for Random Forest Regression
X_grid=np.arange(start=min(X),stop=max(X),step=0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

