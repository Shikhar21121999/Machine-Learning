# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:39:57 2020

@author: LAPPY jr
"""
#importing ibraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import OneHotEncoder as OE
labelencoder_x=LE()
X[:,0]=labelencoder_x.fit_transform(X[:,0])
onehotencoder=OE(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LE()
Y=labelencoder_y.fit_transform(Y)

#splitting the dataset into train daata seet and test data set
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_Train=sc_x.fit_transform(X_Train)
X_Test=sc_x.transform(X_Test)









