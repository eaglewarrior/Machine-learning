# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values#we take all the lines we take all coluumn except last one independent country,age etc
y = dataset.iloc[:, 3].values#we take all lines but only last column it is dependent variable the purchased 
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
#here above we have replaced the missing values by average of the 
#other data we can also use mode etc 
#Encoding categorical data her categorical data is the coutries and purchased
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:, 0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
#do the same for y in y we have only one col so no need of iteration

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
#splitting the values test and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size= 0.2, random_state=0)

#feature scaling
#now it has encoded the text values to 0,1 category
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
#we dont need to scale dummy variable normally 







