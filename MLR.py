# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:03:31 2020

@author: LAPPY jr
"""

#importing ibraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

##taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
#imputer=imputer.fit(X[:,1:3])
#X[:,1:3]=imputer.transform(X[:,1:4])

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import OneHotEncoder as OE
labelencoder_x=LE()
X[:,3]=labelencoder_x.fit_transform(X[:,3])
onehotencoder=OE(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#labelencoder_y=LE()
#Y=labelencoder_y.fit_transform(Y)

#no need to apply dummy variable trap as 
#library akes care of it itself

#splitting the dataset into train daata seet and test data set
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=0)

##feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#X_Train=sc_x.fit_transform(X_Train)
#X_Test=sc_x.transform(X_Test)
#
##Visualizing the training set results
#plt.scatter(X_Train,Y_Train)
#plt.scatter(X_Test,Y_Test,color='red')

#although not needed yet just for show avoiding dummy variable trap
X=X[:,1:]

#makin the linear model
from sklearn.linear_model import LinearRegression as LR
regressor=LR()
regressor.fit(X_Train,Y_Train)

#Predicting the values for the model
y_pred=regressor.predict(X_Test)

#Building the optimal model using Backward Elimination
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1) 
#axis determines values along which array is appended
#axis=0 for row and axis=1 for column

X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
X_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
X_opt=X[:,[0,3]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()





