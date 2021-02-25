# C4.5 algorithm

Implementation of C4.5 algorithm in Python

## Features
* There is no data normalisation - program finds out data type and calculate best value of split info parameter
* Dataset is divided into two sets: training and testing in the ration 7:3

## Dataset
As an example dataset is used student alcohol consumption from [this_site](https://www.kaggle.com/uciml/student-alcohol-consumption/)

## Launch program
In order to launch program you have to pass as an argument name of the dataset (in .csv format) and name of column which represents terminal class e.g.
`python3 C4.5.py student-mat.csv Dalc`

## Comments
* Program is not finished. There is still need to implement back-track pruning
