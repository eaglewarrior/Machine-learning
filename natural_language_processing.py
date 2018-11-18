# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
#here we have to use delimeter as it is tsv not csv and to avoid  issues of double quote we use quoting =3
dataset = pd.read_csv('.../Natural-Language-Processing/Natural_Language_Processing/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')#most common words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    #remove the unwanted stuffs like $#2 etc keeping only letters
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #converting to lowercase
    review = review.lower()
    #splitting them ['see','word'] of see word like this kind
    review = review.split()
    #port stemmer removes the is a process for removing the 
    #commoner morphological and inflexional endings from words in English 
    #like loved will be converted to love 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
#max_features this gives highest 1500 or any number written in it ,its good for us when have to work for millions of reviwes
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)