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
        self.rows = []
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

# convert attributes that are numeric to floats. In the wine dataset all the columns will be numeric except the last one


# def preprocessing(dataset):
#     for example in dataset.rows:
#         for x in range(len(dataset.rows[0])):
#             if dataset.attributes[x] == 'True':
#                 example[x] = float(example[x])


# compute the decision tree recursively
def compute_decision_tree(dataset, parent_node):
    node = decisionTreeNode(None, None, None, parent_node, None, None, True)

    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1

    # count_positives() will count the number of rows with classification "1"
    main_entropy = count_main_entropy(dataset.Y)

    # each value belongs to the same class -> leaf
    if (main_entropy == 0):
        node.classification = dataset.Y[0]
        node.is_leaf_node = True
        return node
    else:
        node.is_leaf_node = False

    # The index of the attribute we will split on
    splitting_attribute = None

    # The information gain given by the best attribute
    global_max_gain = 0
    global_min_gain = 0.01

    global_split_val = None

    # for each column of data
    for index, attr_index in enumerate(dataset.X.columns):

        local_max_gain = 0
        local_split_val = None

        uniq_param_list = dataset.X[attr_index].unique()
        uniq_param_list.sort()
        print("Uniq param: ", uniq_param_list)
        print("Type: ", type(uniq_param_list[0]))
       # print("Print type: ", type(uniq_param_list[0]) is str)

        # these are the values we can split on, now we must find the best one
        # attr_value_list = [example[attr_index] for example in dataset.X]
        # remove duplicates from list of all attribute values
        # attr_value_list = list(set(attr_value_list))

        # if(len(attr_value_list) > 100):
        #     attr_value_list = sorted(attr_value_list)
        #     total = len(attr_value_list)
        #     ten_percentile = int(total/10)
        #     new_list = []
        #     for x in range(1, 10):
        #         new_list.append(attr_value_list[x*ten_percentile])
        #     attr_value_list = new_list

        # if len(uniq_param_list) == 2) -> count for binary
        # else if type() is str -> count for nominal
        # else count for numeric

        if(type(uniq_param_list[0]) is str):
            current_gain = optimal_inf_gain(
                attr_index, uniq_param_list, dataset, main_entropy)
            print(current_gain)
        else:
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

    # for value in uniq_param_list:
    #     # calculate the gain if we split on this value
    #     # if gain is greater than local_max_gain, save this gain and this value
    #     current_gain=calculate_information_gain(
    #         attr_index, dataset.X, value, main_entropy)

    #         if (current_gain > local_max_gain):
    #             local_max_gain=current_gain
    #             local_split_val=val

    #     if (local_max_gain > maximum_info_gain):
    #         maximum_info_gain=local_max_gain
    #         split_val=local_split_val
    #         splitting_attribute=attr_index

    # if (maximum_info_gain <= minimum_info_gain or node.height > 20):
    #     node.is_leaf_node=True
    #     node.classification=classify_leaf(dataset, classifier)
    #     return node

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


def classify_leaf(dataset, classifier):
    ones = count_positives(dataset.rows, dataset.attributes, classifier)
    total = len(dataset.rows)
    zeroes = total - ones
    if (ones >= zeroes):
        return 1
    else:
        return 0


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

    # # get count of all the rows with classification 1
    # ones=count_positives(dataset.rows, dataset.attributes, classifier)

    # # get the count of all the rows in the dataset.
    # total_rows=len(dataset.rows)
    # # from the above two we can get the count of rows with classification 0 too

    # # Entropy formula is sum of p*log2(p). Referred the slides. P is the probability
    # entropy=0

    # # probability p of classification 1 in total data
    # p=ones / total_rows
    # if (p != 0):
    #     entropy += p * math.log(p, 2)
    # # probability p of classification 0 in total data
    # p=(total_rows - ones)/total_rows
    # if (p != 0):
    #     entropy += p * math.log(p, 2)

    # # from the formula
    # entropy=-entropy
    # return entropy

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

    # print(row)

    # print(row[param] >= value)

    #    if (row(2) >= value):
    #         # print(row[param])
    #         upper_set.X.append(row)
    #         print("Index: ", row)
    #         print(dataset.Y[index])
    #         tempVari = dataset.Y.iloc[index]
    #         upper_set.Y.append(tempVari)
    #     else:
    #         lower_set.X.append(row)
    # lower_set.Y.append(dataset.Y[index])

    # decision_list = dataset.Y.unique()
    # decision_list.sort()

    # length_uniq_param = len(uniq_param_list)
    # length_uniq_decision = len(decision_list)

    # ent_uniq_param = [None] * length_uniq_param
    # ent_list = [None] * length_uniq_decision

    # # for count_param in range(length_uniq_param):
    #   prob_uniq_param = np.sum(
    #        dataset.X[param] == param)

    #    for decision in range(length_uniq_decision):
    #         ent_list[decision] = np.sum((dataset.X[param] == uniq_param_list[count_param]) & (
    #             dataset.Y == decision_list[decision])) + 0.00001

    #     sum_ent_list = sum(ent_list)
    #     for i in range(len(ent_list)):
    #         ent_list[i] = (-1)*(ent_list[i]/sum_ent_list) * \
    #             math.log(ent_list[i]/sum_ent_list, 2)

    #     ent_uniq_param[count_param] = sum(ent_list)

    # num_of_records = len(dataset.X)
    # for i in range(len(ent_uniq_param)):
    #     ent_uniq_param[i] = prob_uniq_param[i] / \
    #         num_of_records * ent_uniq_param[i]

    #     prob_uniq_param[i] = (-1) * prob_uniq_param[i]/num_of_records * \
    #         math.log(prob_uniq_param[i]/num_of_records, 2)

    # return (main_entropy - sum(ent_uniq_param)) / sum(prob_uniq_param)
    # return


def calculate_information_gain(attr_index, dataset, value, main_entropy):

    classifier = dataset.attributes[attr_index]
    attr_entropy = 0
    total_rows = len(dataset.rows)
    gain_upper_dataset = csvdata(classifier)
    gain_lower_dataset = csvdata(classifier)
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    gain_upper_dataset.attribute_types = dataset.attribute_types
    gain_lower_dataset.attribute_types = dataset.attribute_types

    for example in dataset.rows:
        if (example[attr_index] >= val):
            gain_upper_dataset.rows.append(example)
        elif (example[attr_index] < val):
            gain_lower_dataset.rows.append(example)

    if (len(gain_upper_dataset.rows) == 0 or len(gain_lower_dataset.rows) == 0):
        return -1

    attr_entropy += calculate_entropy(gain_upper_dataset,
                                      classifier) * len(gain_upper_dataset.rows) / total_rows
    attr_entropy += calculate_entropy(gain_lower_dataset,
                                      classifier) * len(gain_lower_dataset.rows) / total_rows

    return main_entropy - attr_entropy

##################################################
# count number of rows with classification "1"
##################################################


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


# class inputdata():
#     def __init__(self, classifier):
#         self.rows = []
#         self.attributes = []
#         self.classifier = classifier
#         self.classifier_col_ind = None


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

    # preprocessing(df)

    # TODO preprocessing dataset ???
    train_set = inputdata()
    test_set = inputdata()

    train_set.X, test_set.X, train_set.Y, test_set.Y = splitdataset(
        df, classifierLabel)

    print("Number of training records: %d" % len(train_set.X))
    print("Number of test records: %d" % len(test_set.X))

    root = compute_decision_tree(train_set, None)

    # dataset = csvdata("")
    # training_set = csvdata("")
    # test_set = csvdata("")

    # # Load data set
    # # with open("data.csv") as f:
    # #    dataset.rows = [tuple(line) for line in csv.reader(f, delimiter=",")]
    # # print "Number of records: %d" % len(dataset.rows)
    # f = open("data.csv")
    # original_file = f.read()
    # rowsplit_data = original_file.splitlines()
    # dataset.rows = [rows.split(',') for rows in rowsplit_data]

    # dataset.attributes = dataset.rows.pop(0)
    # print(dataset.attributes)

    # # this is used to generalize the code for other datasets.
    # # true indicates numeric data. false in nominal data
    # dataset.attribute_types = ['true', 'true', 'true', 'true',
    #                            'true', 'true', 'true', 'true', 'true', 'true', 'true', 'false']

    # classifier = dataset.attributes[-1]
    # dataset.classifier = classifier

    # # find index of classifier
    # for a in range(len(dataset.attributes)):
    #     if dataset.attributes[a] == dataset.classifier:
    #         dataset.class_col_index = a
    #     else:
    #         dataset.class_col_index = range(len(dataset.attributes))[-1]

    # print("classifier is %d" % dataset.class_col_index)
    # # preprocessing the dataset
    # preprocessing(dataset)

    # training_set = copy.deepcopy(dataset)
    # training_set.rows = []
    # test_set = copy.deepcopy(dataset)
    # test_set.rows = []
    # validate_set = copy.deepcopy(dataset)
    # validate_set.rows = []
    # # Split training/test sets
    # # You need to modify the following code for cross validation.

    # # This is to create a validation set for post pruning
    # # dataset.rows = [x for i, x in enumerate(dataset.rows) if i % 10 != 9]
    # # validate_set.rows = [x for i, x in enumerate(dataset.rows) if i % 10 == 9]

    # K = 10
    # # Stores accuracy of the 10 runs
    # accuracy = []
    # #start = time.clock()
    # for k in range(K):
    #     print("Doing fold ", k)
    #     training_set.rows = [x for i, x in enumerate(
    #         dataset.rows) if i % K != k]
    #     test_set.rows = [x for i, x in enumerate(dataset.rows) if i % K == k]

    #     print("Number of training records: %d" % len(training_set.rows))
    #     print("Number of test records: %d" % len(test_set.rows))
    #     root = compute_decision_tree(training_set, None, classifier)

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
