# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 22:29:34 2020

@author: LAPPY jr
"""
#Logistic Regression

#importin Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the dataset
ds=pd.read_csv('Social_Network_Ads.csv')
X=ds.iloc[:,[2,3]].values
Y=ds.iloc[:,4].values

#splitting the dataset into train daata seet and test data set
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Featur Scaling has to be done in case of Logistic Regression
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_Train=sc_x.fit_transform(X_Train)
X_Test=sc_x.transform(X_Test)

#Fitting Logistic Regression to the dataset
from sklearn.linear_model import LogisticRegression as LR
classifier=LR(random_state=0)
classifier.fit(X_Train,Y_Train)

#Predicting the test results
y_pred=classifier.predict(X_Test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix #here confusion_matrix is a function
cm=confusion_matrix(Y_Test,y_pred)

#Visualizing the training results using graphs
from matplotlib.colors import ListedColormap
X_set,Y_set=X_Train,Y_Train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
            np.arrange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))







