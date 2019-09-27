import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path1 = "C:\\Users\\Andish\\PycharmProjects\\MACHINE LEARNING\\iris.data"

# assign colums
headernames = ['sepal-length','sepal-width','petal-length','petal-width','Class']

# read dataset to pandas dataframe
dataset = pd.read_csv(path1, names = headernames)
dataset.head()

# data preprocessing will be done with the help of following script lines
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

# divide the data into train and test split. 60% training & 40% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40)

# data scaling will be done as follows
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# train the model with the help of KNeighborsClassifier class of sklearn as follows
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X_train, Y_train)

# need to make prediction with the following script
Y_pred = classifier.predict(X_test)

# print the results as follows
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, Y_pred)
print("classification_report")
print(result1)
result2 = accuracy_score(Y_test, Y_pred)
print("Accuracy: ", result2)