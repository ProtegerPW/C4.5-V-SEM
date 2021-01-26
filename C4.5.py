from __future__ import division

import sys
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

import operator
import copy
import csv
import time
import random

from collections import Counter


# csvdata class to store the csv data
class inputdata():
    def __init__(self):
        self.X = []
        self.Y = []

# the node class that will make up the tree


class decisionTreeNode():
    def __init__(self, classification, attribute_split_index, attribute_split_value, parent, child, height, is_leaf_node):

        self.classification = None
        self.attribute_split = None
        self.attribute_split_index = None
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
    if (main_entropy == 0):
        node.classification = dataset.Y[0]
        node.is_leaf_node = True
        return node
    else:
        node.is_leaf_node = False

    # attribuite which splits the dataset
    splitting_attribute = None

    # The information gain given by the best attribute
    global_max_gain = 0
    global_min_gain = 0.0001
    global_split_val = None

    # for each column of data calculate entropy
    for index, attr_index in enumerate(dataset.X.columns):

        local_max_gain = 0
        local_split_val = None

        uniq_param_list = dataset.X[attr_index].unique()
        uniq_param_list.sort()
        # print("Uniq param: ", uniq_param_list)
        # print("Type: ", type(uniq_param_list[0]))
       # print("Print type: ", type(uniq_param_list[0]) is str)

        # if type is not numerical
        if(type(uniq_param_list[0]) is str):
            current_gain = optimal_inf_gain(
                attr_index, uniq_param_list, dataset, main_entropy)
            print("Current gain is equal: ", current_gain)

            if(current_gain > global_max_gain):
                global_max_gain = current_gain
                global_split_val = uniq_param_list
                splitting_attribute = attr_index
        else:
            # if type is numerical
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

    if(global_max_gain <= global_min_gain):
        node.is_leaf_node = True
        node.classification = classify_leaf(dataset)
        return node

    print("Global attribute: ", splitting_attribute)
    print("Global split value: ", global_split_val)

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

        print(right_child.X)

        node.child.append(compute_decision_tree(right_child, node))
        node.child.append(compute_decision_tree(left_child, node))

        return node

    else:
        child = [inputdata()] * len(global_split_val)
        print("Type of childs: ", type(child[0]), len(child))
        print("Global_split_val: ", global_split_val)
        # print("Right child: ", type(right_child.X))
        for uniq in range(len(global_split_val)):
            for row in range(len(dataset.X)):
                if dataset.X.iloc[row][splitting_attribute] == global_split_val[uniq]:
                    child[uniq].X.append(dataset.X.iloc[row])
                    child[uniq].Y.append(dataset.Y.iloc[row])

        for i in range(len(child)):
            child[i].X = pd.DataFrame(child[i].X)
            child[i].Y = pd.Series(child[i].Y, dtype=int)
            print(child[i].X)
            node.child.append(compute_decision_tree(child[i], node))

        return node
        # for rest

    # node.attribute_split_index=splitting_attribute
    # node.attribute_split=dataset.attributes[splitting_attribute]
    # node.attribute_split_value=split_val

    # left_dataset=csvdata(classifier)
    # right_dataset=csvdata(classifier)

    # left_dataset.attributes=dataset.attributes
    # right_dataset.attributes=dataset.attributes

    # left_dataset.attribute_types=dataset.attribute_types
    # right_dataset.attribute_types=dataset.attribute_types

    # for row in dataset.rows:
    #     if (splitting_attribute is not None and row[splitting_attribute] >= split_val):
    #         left_dataset.rows.append(row)
    #     elif (splitting_attribute is not None):
    #         right_dataset.rows.append(row)

    # node.left_child=compute_decision_tree(left_dataset, node, classifier)
    # node.right_child=compute_decision_tree(right_dataset, node, classifier)

    return node

# # Classify dataset


def classify_leaf(dataset):
    attr_list = dataset.Y.unique()
    attr_list.sort()
    count_attr = [None] * len(attr_list)
    for i in len(count_attr):
        count_attr[i] = np.sum(dataset.Y == attr_list[i])

    biggest_index = count_attr.index(max_value)
    return attr_list[biggest_index]


# Final evaluation of the data
def get_classification(example, node, class_col_index):
    if (node.is_leaf_node == True):
        return node.classification
    else:
        if (example[node.attribute_split_index] >= node.attribute_split_value):
            return get_classification(example, node.left_child, class_col_index)
        else:
            return get_classification(example, node.right_child, class_col_index)

##################################################
# Calculate the entropy of the current dataset
##################################################


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

##################################################
# Calculate the gain of a particular attribute split
##################################################


def optimal_inf_gain(param, uniq_param_list, dataset, main_entropy):
    decision_list = dataset.Y.unique()
    print("Opt inf gain: ", type(dataset.Y))
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
        ent_uniq_param[i] = prob_uniq_param[i] / \
            num_of_records * ent_uniq_param[i]

        prob_uniq_param[i] = (-1) * prob_uniq_param[i]/num_of_records * \
            math.log(prob_uniq_param[i]/num_of_records, 2)

    return (main_entropy - sum(ent_uniq_param)) / sum(prob_uniq_param)


def num_inf_gain(value, param, uniq_param_list, dataset, main_entropy):

    upper_set = inputdata()
    lower_set = inputdata()
    split_ent = 0

    # print(type(dataset.X))
    # print(type(dataset.X.iloc[0]))
    # print(type(dataset.Y.iloc[2]))
    # print(type(upper_set.Y))

    for row in range(len(dataset.X)):
        if dataset.X.iloc[row][param] >= value:
            # print(dataset.X.iloc[row][param], value)
            upper_set.X.append(dataset.X.iloc[row])
            upper_set.Y.append(dataset.Y.iloc[row])
        else:
            lower_set.X.append(dataset.X.iloc[row])
            lower_set.Y.append(dataset.Y.iloc[row])

    upper_set.Y = pd.Series(upper_set.Y, dtype=int)
    lower_set.Y = pd.Series(lower_set.Y, dtype=int)

    if (len(upper_set.X) == 0 or len(lower_set.X) == 0):
        return -1

    split_ent += calculate_entropy(upper_set) * len(upper_set.X)/len(dataset.X)

    split_ent += calculate_entropy(lower_set) * len(lower_set.X)/len(dataset.X)

    return main_entropy - split_ent


def count_main_entropy(inputColumn):
    rowNum = len(inputColumn)
    print(rowNum)

    numOfValues = inputColumn.unique()
    numOfValues.sort()
    entOfValues = [None] * len(numOfValues)

    for x in range(len(numOfValues)):
        entOfValues[x] = np.sum(inputColumn == numOfValues[x])
        entOfValues[x] = (-1)*(entOfValues[x]/rowNum) * \
            math.log(entOfValues[x]/rowNum, 2)

    print(sum(entOfValues))
    return sum(entOfValues)


def validate_tree(node, dataset):
    total = len(dataset.rows)
    correct = 0
    for row in dataset.rows:
        # validate example
        correct += validate_row(node, row)
    return correct/total

# Validate row (for finding best score before pruning)


def validate_row(node, row):
    if (node.is_leaf_node == True):
        projected = node.classification
        actual = int(row[-1])
        if (projected == actual):
            return 1
        else:
            return 0
    value = row[node.attribute_split_index]
    if (value >= node.attribute_split_value):
        return validate_row(node.left_child, row)
    else:
        return validate_row(node.right_child, row)

##################################################
# Prune tree
##################################################


def prune_tree(root, node, validate_set, best_score):
    # if node is a leaf
    if (node.is_leaf_node == True):
        classification = node.classification
        node.parent.is_leaf_node = True
        node.parent.classification = node.classification
        if (node.height < 20):
            new_score = validate_tree(root, validate_set)
        else:
            new_score = 0

        if (new_score >= best_score):
            return new_score
        else:
            node.parent.is_leaf_node = False
            node.parent.classification = None
            return best_score
    # if its not a leaf
    else:
        new_score = prune_tree(root, node.left_child, validate_set, best_score)
        if (node.is_leaf_node == True):
            return new_score
        new_score = prune_tree(root, node.right_child, validate_set, new_score)
        if (node.is_leaf_node == True):
            return new_score

        return new_score


def splitdataset(balance_data, classifier):

    # Separating the target variable
    feature_cols = list(balance_data.columns)
    feature_cols.remove(classifier)
    X = balance_data[feature_cols]
    Y = balance_data[classifier]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1)

    # print("date in X: ", X)
    # print("date in Y: ", Y)
    # print("date in X_train: ", X_train)
    # print("date in X_test: ", X_test)

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

    print(root)

    #     # Classify the test set using the tree we just constructed
    #     results = []
    #     for instance in test_set.rows:
    #         result = get_classification(
    #             instance, root, test_set.class_col_index)
    #         results.append(str(result) == str(instance[-1]))

    #     # Accuracy
    #     acc = float(results.count(True))/float(len(results))
    #     print("accuracy: %.4f" % acc)

    #     # pruning code currently disabled
    #     # best_score = validate_tree(root, validate_set)
    #     # post_prune_accuracy = 100*prune_tree(root, root, validate_set, best_score)
    #     # print "Post-pruning score on validation set: " + str(post_prune_accuracy) + "%"
    #     accuracy.append(acc)
    #     del root

    # mean_accuracy = math.fsum(accuracy)/10
    # print("Accuracy  %f " % (mean_accuracy))
    # #print("Took %f secs" % (time.clock() - start))
    # # Writing results to a file (DO NOT CHANGE)
    # f = open("result.txt", "w")
    # f.write("accuracy: %.4f" % mean_accuracy)
    # f.close()


def preprocessing(dataset):
    for index, row in dataset.iterrows():
        for i in range(len(dataset.columns)):
            if (type(i) != str):
                print(i)
                row[i] = int(row[i])


if __name__ == "__main__":
    run_decision_tree(sys.argv[1], sys.argv[2])
