# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:58:50 2020

@author: LAPPY jr
"""

# My K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

#Splitting the data into training data and test data
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature scaling has to be done in K_nearest neighbors algo
#Feature Scaling
from sklearn.preprocessing import StandardScaler as SS
sc=SS()
X_Train=sc.fit_transform(X_Train)
X_Test=sc.transform(X_Test)

#Creating the model and fitting it to data
from sklearn.neighbors import KNeighborsClassifier as KNC
classifier = KNC(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_Train, Y_Train)

#Predicting the result
y_pred=classifier.predict(X_Test)

#Creatingt the confusion matrix for finding the accuracy of model
from sklearn.metrics import confusion_matrix           #here confusion_matrix is a method not a class
cm=confusion_matrix(Y_Test,y_pred)

#Visualizing the data using pyplot

#Viualizing the training set results
from matplotlib.colors import ListedColormap
X_set,Y_set=X_Train,Y_Train
x1,x2=np.meshgrid(np.array(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01)),
                  np.array(np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01)))

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()







