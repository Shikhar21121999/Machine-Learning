# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 23:59:40 2020

@author: LAPPY jr
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,2].values

##taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
#imputer=imputer.fit(X[:,1:3])
#X[:,1:3]=imputer.transform(X[:,1:4])

##Encoding Categorical data
#from sklearn.preprocessing import LabelEncoder as LE
#from sklearn.preprocessing import OneHotEncoder as OE
#labelencoder_x=LE()
#X[:,0]=labelencoder_x.fit_transform(X[:,0])
#onehotencoder=OE(categorical_features=[0])
#X=onehotencoder.fit_transform(X).toarray()

#makin the linear model
from sklearn.linear_model import LinearRegression as LR
linear_regressor=LR()
linear_regressor.fit(X,Y)

#makin the polynomial model
from sklearn.preprocessing import PolynomialFeatures as PF
poly_reg=PF(degree=8)
X_poly=poly_reg.fit_transform(X)
#polynomial model
linear_regressor2=LR()
linear_regressor2.fit(X_poly,Y)

#Visualizing the training set results using single linear regression
plt.scatter(X,Y,color='red')
plt.plot(X,linear_regressor.predict(X),color='black')
plt.title('Linear regressor model')
linear_regressor.predict(6.5.reshape(6.5,(-1,1)))
plt.scatter(linear_regressor.predict(6.5),b,color='green')

#Visualizing the training set results using polynomial regression
X_grid=np.arange(start=0,stop=10.1,step=0.001)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,linear_regressor2.predict(poly_reg.fit_transform(X_grid)),color='black')
plt.title(' Polynomial Linear regressor model')
linear_regressor2.predict(6.5)
plt.scatter(linear_regressor2.predict(6.5),Y,color='green')




