# -*- coding: utf-8 -*-
"""
Created on Sat May 26 06:22:13 2018

@author: admin
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()#making object for reg package
regressor.fit(X_train, y_train)#to fit the regressor to our training data

#predict the test results
y_pred =regressor.predict(X_test)
#Now if we compare y_Pred and y_test we can see the current salary and model predicted salary in y_pred
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#we have plotted the line where real salary in x axis and 
#predicted salary in y axis  and we observe thatfew obs which are on line means its quite accurate i.e. real salary approx equal to predcted salary
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#here model is same only scatter points are of training set
#a we have fit that is tarining set and here we are testing its efficiency in test set
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

























