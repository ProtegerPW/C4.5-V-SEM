# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Function importing Dataset


def importdata():
    balance_data = pd.read_csv("student-mat.csv")

    # Printing the dataswet shape
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)

    # Printing the dataset obseravtions
    print("Dataset: ", balance_data)
    return balance_data

# Function to split the dataset


def splitdataset(balance_data):

    # Separating the target variable
    feature_cols = list(balance_data.columns)
    feature_cols.remove("Dalc")
    feature_cols.remove("Walc")
    X = balance_data[feature_cols]
    Y = balance_data["Dalc"]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1)

    # print("date in X: ", X)
    # print("date in Y: ", Y)
    # print("date in X_train: ", X_train)
    # print("date in X_test: ", X_test)

    return X, Y, X_train, X_test, y_train, y_test

# Function to perform training with giniIndex.


def train_using_gini(X_train, X_test, y_train):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier()

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.


def tarin_using_entropy(X_train, X_test, y_train):

    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy")

    # Performing training
    clf_entropy = clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):

    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy


def cal_accuracy(y_test, y_pred):

    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred)*100)

    print("Report : ",
          classification_report(y_test, y_pred))

# Driver code


def main():

    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    enc = OrdinalEncoder()
    le = LabelEncoder()

    X_train = enc.fit_transform(X_train)
    X_test = enc.fit_transform(X_test)
    y_train = le.fit_transform(y_train)

    # clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

    # Operational Phase
    # print("Results Using Gini Index:")

    # Prediction using gini
    # y_pred_gini = prediction(X_test, clf_gini)
    # cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    print("Accuracy:", metrics.accuracy_score(
        y_test, y_pred_entropy), len(y_pred_entropy))
    # cal_accuracy(y_test, y_pred_entropy)


# Calling main function
if __name__ == "__main__":
    main()
