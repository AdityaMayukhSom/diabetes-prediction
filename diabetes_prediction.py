# importing libraries for data processing

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

from sklearn.metrics import accuracy_score

# Data collection and analysis
# PIMA diabetes dataset

# loading the dataset into pandas dataframe

diabetes_dataset = pd.read_csv('./diabetes.csv')

# printing the first 5 rows of the dataset
# print(diabetes_dataset.head())

# number of rows and columns in the dataset
# print(diabetes_dataset.shape)

# getting the statistical measures of the data
# print(diabetes_dataset.describe())

# 0 --> non diabetic
# 1 --> diabetic

# print(diabetes_dataset['Outcome'].value_counts())
# print(diabetes_dataset.groupby('Outcome').mean())
# print(diabetes_dataset.groupby('Outcome').median())


# seperating the data and the labels

# X represents the data, Y represents the outcome
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
X = X.values  # only training with the values, not the feature names
# data standardization

# here we are using one instance of the StandardScalar object
scalar = StandardScaler()

scalar.fit(X)
standardized_data = scalar.transform(X)
# the above two steps are same as
# standardized_data = scalar.fit_transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

# now we need to split our data for training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)


# print(X.shape, X_train.shape, X_test.shape)
# print(Y.shape, Y_train.shape, Y_test.shape)

classifier = svm.SVC(kernel='linear')
# training the support vector machine classifier
classifier.fit(X_train, Y_train)


# model evaluation
# accuracy score the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('accuracy score of the training data is', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('accuracy score of the test data is', test_data_accuracy)

# making a predictive system

# answer should be 1 --> diabetic
input_data = (10, 168, 74, 0, 0, 38, 0.537, 34)
# change import data to numpy array
input_data_for_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting one instance
input_data_reshaped = input_data_for_numpy_array.reshape(1, -1)

# standardize the input data
# scalar is already fitted with the dataset X, so we need to only transform
std_data = scalar.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if prediction == 0:
    print('person is non diabetic')
else:
    print('person is diabetic')
