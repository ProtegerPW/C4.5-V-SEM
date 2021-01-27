# from __future__ import division

import sys
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

# import operator
# import copy
# import csv
# import time
# import random

from collections import Counter


class inputdata():
    def __init__(self):
        self.X = []
        self.Y = []


class decisionTreeNode():
    def __init__(self, classification, value, attribute_split_value, parent, child, height, is_leaf_node):

        self.classification = None
        self.value = None
        self.attribute_split = None
        self.attribute_split_value = None
        self.parent = parent
        self.child = []
        self.height = None
        self.is_leaf_node = True


# compute the decision tree recursively
def compute_decision_tree(dataset, parent_node):
    node = decisionTreeNode(None, None, None, parent_node, None, None, True)

    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1

    # count entropy of all possible decision in dataset
    main_entropy = count_main_entropy(dataset.Y)

    # each value belongs to the same class -> leaf
    if (main_entropy == 0.0):
        node.classification = dataset.Y[0]
        node.is_leaf_node = True
        return node
    else:
        node.is_leaf_node = False

    # attribuite which splits the dataset
    splitting_attribute = None

    # the information gain given by the best attribute
    global_max_gain = 0
    global_split_val = None

    # for each column of data calculate entropy
    for index, attr_index in enumerate(dataset.X.columns):

        uniq_param_list = dataset.X[attr_index].unique()
        uniq_param_list.sort()

        # if type is not numerical
        if(type(uniq_param_list[0]) is str):
            current_gain = optimal_inf_gain(
                attr_index, uniq_param_list, dataset, main_entropy)
            print("Current gain for " + attr_index +
                  " is equal: ", current_gain)

            if(current_gain > global_max_gain):
                global_max_gain = current_gain
                global_split_val = uniq_param_list
                splitting_attribute = attr_index
        else:
            # if type is numerical find the best split value
            local_max_gain = 0
            local_split_val = None

            for value in uniq_param_list:
                current_gain = num_inf_gain(
                    value, attr_index, uniq_param_list, dataset, main_entropy)

                if (current_gain > local_max_gain):
                    local_max_gain = current_gain
                    local_split_val = value

            if(local_max_gain > global_max_gain):
                global_max_gain = local_max_gain
                global_split_val = local_split_val
                splitting_attribute = attr_index

    print("Global_max_gain", global_max_gain)
    if(global_max_gain == 1.0):
        node.is_leaf_node = True
        node.classification = classify_leaf(dataset)
        return node

    node.attribute_split = splitting_attribute
    node.attribute_split_value = global_split_val

    if (type(global_split_val) == np.int64):

        right_child = inputdata()
        left_child = inputdata()

        for row in range(len(dataset.X)):
            if dataset.X.iloc[row][splitting_attribute] >= global_split_val:
                right_child.X.append(dataset.X.iloc[row])
                right_child.Y.append(dataset.Y.iloc[row])
            else:
                left_child.X.append(dataset.X.iloc[row])
                left_child.Y.append(dataset.Y.iloc[row])

        right_child.Y = pd.Series(right_child.Y, dtype=int)
        left_child.Y = pd.Series(left_child.Y, dtype=int)

        right_child.X = pd.DataFrame(right_child.X)
        left_child.X = pd.DataFrame(left_child.X)

        right_child.X.drop(splitting_attribute, 1, inplace=True)
        left_child.X.drop(splitting_attribute, 1, inplace=True)

        node.child.append(compute_decision_tree(right_child, node))
        node.child.append(compute_decision_tree(left_child, node))

        return node

    else:
        child = []
        for i in range(len(global_split_val)):
            child.append(None)
            child[i] = inputdata()

        for uniq in range(len(global_split_val)):
            for row in range(len(dataset.X)):
                if (dataset.X.iloc[row][splitting_attribute] == global_split_val[uniq]):
                    child[uniq].X.append(dataset.X.iloc[row])
                    child[uniq].Y.append(dataset.Y.iloc[row])

        for i in range(len(child)):
            child[i].X = pd.DataFrame(child[i].X)
            child[i].Y = pd.Series(child[i].Y, dtype=int)
            child[i].X.drop(splitting_attribute, axis=1, inplace=True)
            child[i].value = global_split_val[i]
            node.child.append(compute_decision_tree(child[i], node))

        return node


def classify_leaf(dataset):
    decision_list = dataset.Y.unique()
    decision_list.sort()
    count_decision = [None] * len(decision_list)
    for i in range(len(count_decision)):
        count_decision[i] = np.sum(dataset.Y == decision_list[i])

    biggest_index = count_decision.index(max(count_decision))
    return decision_list[biggest_index]


# Final evaluation of the data
def get_classification(row, node):
    if (node.is_leaf_node == True):
        return node.classification
    else:
        if(type(node.attribute_split_value) == np.int64):
            if (row[node.attribute_split] >= node.attribute_split_value):
                return get_classification(row, node.child[0])
            else:
                return get_classification(row, node.child[1])
        else:
            for i in range(len(node.child)):
                if (row[node.attribute_split] == node.child[i].value):
                    return get_classification(row, node.child[i])
                    break


def calculate_entropy(dataset):
    decision_list = dataset.Y.unique()
    decision_list.sort()

    length_uniq_decision = len(decision_list)

    ent_list = [None] * length_uniq_decision

    for decision in range(length_uniq_decision):
        ent_list[decision] = np.sum(
            dataset.Y == decision_list[decision]) + 0.00001

    sum_ent_list = sum(ent_list)
    for i in range(len(ent_list)):
        ent_list[i] = (-1)*(ent_list[i]/sum_ent_list) * \
            math.log(ent_list[i]/sum_ent_list, 2)

    return sum(ent_list)


def optimal_inf_gain(param, uniq_param_list, dataset, main_entropy):
    decision_list = dataset.Y.unique()
    decision_list.sort()

    length_uniq_param = len(uniq_param_list)
    length_uniq_decision = len(decision_list)

    prob_uniq_param = [None] * length_uniq_param
    ent_uniq_param = [None] * length_uniq_param
    ent_list = [None] * length_uniq_decision

    for count_param in range(length_uniq_param):
        prob_uniq_param[count_param] = np.sum(
            dataset.X[param] == uniq_param_list[count_param])

        for decision in range(length_uniq_decision):
            ent_list[decision] = np.sum((dataset.X[param] == uniq_param_list[count_param]) & (
                dataset.Y == decision_list[decision])) + 0.00001

        sum_ent_list = sum(ent_list)
        for i in range(len(ent_list)):
            ent_list[i] = (-1)*(ent_list[i]/sum_ent_list) * \
                math.log(ent_list[i]/sum_ent_list, 2)

        ent_uniq_param[count_param] = sum(ent_list)

    num_of_records = len(dataset.X)

    for i in range(len(ent_uniq_param)):
        ent_uniq_param[i] = (prob_uniq_param[i] /
                             num_of_records) * ent_uniq_param[i]

        prob_uniq_param[i] = (-1.0) * prob_uniq_param[i]/num_of_records * \
            math.log(prob_uniq_param[i]/num_of_records, 2)
        if sum(prob_uniq_param) == 0:
            return 0.0

    return ((main_entropy - sum(ent_uniq_param)) / sum(prob_uniq_param))


def num_inf_gain(value, param, uniq_param_list, dataset, main_entropy):

    upper_set = inputdata()
    lower_set = inputdata()
    split_ent = 0.0

    for row in range(len(dataset.X)):
        if dataset.X.iloc[row][param] >= value:
            upper_set.X.append(dataset.X.iloc[row])
            upper_set.Y.append(dataset.Y.iloc[row])
        else:
            lower_set.X.append(dataset.X.iloc[row])
            lower_set.Y.append(dataset.Y.iloc[row])

    upper_set.Y = pd.Series(upper_set.Y, dtype=int)
    lower_set.Y = pd.Series(lower_set.Y, dtype=int)

    if (len(upper_set.X) == 0 or len(lower_set.X) == 0):
        return -1.0

    size_of_dataset = len(dataset.X)
    size_of_upper_set = len(upper_set.X)
    size_of_lower_set = len(lower_set.X)

    split_ent += calculate_entropy(upper_set) * \
        size_of_upper_set/size_of_dataset

    split_ent += calculate_entropy(lower_set) * \
        size_of_lower_set/size_of_dataset

    split_info = (-1.0) * (size_of_upper_set/size_of_dataset * math.log(size_of_upper_set/size_of_dataset, 2) +
                           size_of_lower_set/size_of_dataset * math.log(size_of_lower_set/size_of_dataset, 2))

    return float((main_entropy - split_ent) / split_info)


def count_main_entropy(inputColumn):
    rowNum = len(inputColumn)

    numOfValues = inputColumn.unique()
    numOfValues.sort()
    entOfValues = [None] * len(numOfValues)

    for x in range(len(numOfValues)):
        entOfValues[x] = np.sum(inputColumn == numOfValues[x])

        entOfValues[x] = (-1)*(entOfValues[x]/rowNum) * \
            math.log(entOfValues[x]/rowNum, 2)
    return sum(entOfValues)


def splitdataset(balance_data, classifier):

    feature_cols = list(balance_data.columns)
    feature_cols.remove(classifier)
    X = balance_data[feature_cols]
    Y = balance_data[classifier]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1)

    return X_train, X_test, y_train, y_test


def run_decision_tree(fileName, classifierLabel):
    # read dataFrame from file
    df = pd.read_csv(fileName)
    print(df)

    # divide into train and test set
    train_set = inputdata()
    test_set = inputdata()

    train_set.X, test_set.X, train_set.Y, test_set.Y = splitdataset(
        df, classifierLabel)

    print("Number of training records: %d" % len(train_set.X))
    print("Number of test records: %d" % len(test_set.X))

    # create decision tree from the root
    root = compute_decision_tree(train_set, None)

    scores = []
    decisions = []
    for row in range(len(test_set.X)):
        decision = get_classification(test_set.X.iloc[row], root)
        decisions.append(decision)
        scores.append(decision == test_set.Y.iloc[row])

    accuracy = float(scores.count(True))/float(len(scores))
    print("Accuracy: %.4f" % accuracy)


if __name__ == "__main__":
    run_decision_tree(sys.argv[1], sys.argv[2])
