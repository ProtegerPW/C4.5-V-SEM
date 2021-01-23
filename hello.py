import sys
import pandas as pd
import numpy as np
import math


def calculate_main_entropy(inputColumn):
    rowNum = len(inputColumn)
    print(rowNum)
    alcoholCon = [0, 0, 0, 0, 0]

    for x in range(5):
        alcoholCon[x] = np.sum(inputColumn == (x+1))
        alcoholCon[x] = (-1)*(alcoholCon[x]/rowNum) * \
            math.log(alcoholCon[x]/rowNum, 2)

    print(sum(alcoholCon))
    return sum(alcoholCon)


def calculate_parameters_entropy(inputTable):
    mainEntropy = calculate_main_entropy(inputTable["Dalc"])

    parametersList = list(inputTable.columns)
    splitInfoList = [None] * len(parametersList)

    for index, presentParam in enumerate(parametersList):
        if presentParam == "Dalc" or presentParam == "Walc":
            continue

        # zbiór unikalnych przyjmowanych parametrów
        uniqParam = inputTable[presentParam].unique()
        decisionParam = inputTable["Dalc"].unique()
        decisionParam.sort()
        # tablica do przechowywania ilości wystąpienia unikalnych parametrów
        entropyUniqParam = [None] * len(uniqParam)
        probUniqParam = [None] * len(uniqParam)

        for countParam in range(len(uniqParam)):
            probUniqParam[countParam] = np.sum(
                inputTable[presentParam] == uniqParam[countParam])

            entropyList = [None] * len(decisionParam)

            for decision in range(len(decisionParam)):

                # suma
                entropyList[decision] = np.sum((inputTable[presentParam] == uniqParam[countParam]) & (
                    inputTable["Dalc"] == decisionParam[decision])) + 0.00001
                # print(entropyList[decision])
                # np.sum(
                #     inputTable[presentParam] == uniqParam[countParam])
                # np.sum((
                # inputTable[presentParam] == uniqParam[countParam]) & (inputTable["Dalc"] == decisionParam[decision]))

            sumEntropyElem = sum(entropyList)
            for x in range(len(entropyList)):
                entropyList[x] = (-1)*(entropyList[x]/sumEntropyElem) * \
                    math.log(entropyList[x]/sumEntropyElem, 2)

            entropyUniqParam[countParam] = sum(entropyList)
        # print(entropyUniqParam)
        # print(probUniqParam)

        numOfRows = len(inputTable.index)
        for x in range(len(probUniqParam)):
            entropyUniqParam[x] = probUniqParam[x] / \
                numOfRows * entropyUniqParam[x]

            probUniqParam[x] = (-1) * probUniqParam[x]/numOfRows * \
                math.log(probUniqParam[x]/numOfRows, 2)

        splitInfoList[index] = (
            mainEntropy - sum(entropyUniqParam)) / sum(probUniqParam)

    print(splitInfoList)


def run_decision_tree(fileName):
    df = pd.read_csv(fileName)
    print(df)
    #saved_column = df["age"].sum()
    # print(saved_column)
    parametersList = calculate_parameters_entropy(df)


if __name__ == "__main__":
    run_decision_tree(sys.argv[1])
