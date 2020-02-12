# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 20:51:30 2020

@author: LAPPY jr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#getting the dataset
ds=pd.read_csv('Position_Salaries.csv')
X=ds.iloc[:,1:2].values
X=X.reshape(-1,1)
Y=ds.iloc[:,2:].values
Y=Y.reshape(-1,1)

#Feature Scaling not required
##feature scaling
##in svr we need to manually apply 
#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#sc_y=StandardScaler()
#X=sc_x.fit_transform(X)
#Y=sc_y.fit_transform(Y)

#creating the regressor amd fitting it to data
from sklearn.tree import DecisionTreeRegressor as DTR
regressor=DTR(random_state=0)
regressor.fit(X,Y)

#Predicting the result
y_pred=regressor.predict(np.array(([6.5])).reshape(1,-1))
y_pred=regressor.predict(np.array(([6.5])).reshape(1,-1))
#y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array(([6.5])).reshape(1,-1))))

# Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
X_grid=np.arange(start=min(X),stop=max(X),step=0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

from sklearn import tree



