from typing import List
from math import log

# import math
# import numpy as np
import pandas as pd
# from typing import *
from pandas import DataFrame

from classifier_tree import *

ILL = 'M'
HEALTHY = 'B'


def get_features_from_csv(file_name):
    df = pd.read_csv(file_name)
    return df.columns.tolist()


def get_data_from_csv(file_name):
    """Returns the data that is saved as a csv file in current folder.
    """
    # Using the function to load the data of file_name.csv into a Dataframe df
    df = pd.read_csv(file_name)
    # data_array = df.to_numpy()

    # print("Indexes numbers: ")
    # print(df.index)
    # print()
    # print("Columns names: ")
    # print(df.columns.tolist())
    # print("Columns dtypes: ")
    # print(df.dtypes)

    # Print the Dataframe
    # print(dataframe)
    # df_ = df.astype(dtype={'diagnosis': np.float64})
    # data_array = df_.to_numpy()
    # data_array = df.to_numpy(dtype={'diagnosis': np.float64})

    # data_array = np.loadtxt(file_name, delimiter=',', skiprows=1)
    # print("data_array: ")
    # print(data_array)
    # Return the Dataframe
    return df
    # return data_array


# gets a data frame, a feature and a threshold and returns the 2 dataframes split by the threshold value
def filter_dataframe_by_threshold(dataframe, feature: str, threshold: float):
    below_threshold = dataframe[feature] < threshold
    above_threshold = dataframe[feature] >= threshold
    entries_below = dataframe[below_threshold]
    entries_above = dataframe[above_threshold]
    return entries_below, entries_above


# calculates a list of predictions done by a given classifier on a given test data frame
def calc_predictions(test_data: DataFrame, classifier):
    print('calc_predictions function: ')
    predictions = []
    for index, patient_entry in test_data.iterrows():
        print('real diagnosis is: ', patient_entry['diagnosis'], '\n')
        print(patient_entry, '\n')
        prediction = dt_classify(patient_entry, classifier)
        print('prediction by dt_classify is: ', prediction, '\n')
        predictions.append(prediction)
    return predictions


# calculates the accuracy of a given classifier on a test data frame
def calc_accuracy(test_dataframe: DataFrame, classifier):
    print('calc_accuracy function: ')
    real_classifications = test_dataframe['diagnosis'].tolist()
    predictions = calc_predictions(test_dataframe, classifier)
    # true_positive = 0
    # true_negative = 0
    assert len(real_classifications) == len(predictions)
    total = len(real_classifications)
    counter = 0
    for item in real_classifications:
        print('real classification item: ', item, '\n')
        print('predictions[item]: ', predictions[item], '\n')
        if item == predictions[item]:
            counter += 1
        print('counter of accuracy is: ', counter, '\n')
    accuracy = (counter * 1.0)/total
    print('accuracy is: ', accuracy, '\n')
    return accuracy


# the classification algorithm. gets an object to classify and the classification tree.
def dt_classify(patient_entry, tree_node: Node):
    if tree_node.children is None or len(tree_node.children) == 0:
        # if it's a leaf, we just return the classification stored there.
        return tree_node.classification

    # for subtree in tree_node.children:
    # TODO: maybe need get_feature()?
    patient_feature_value = patient_entry[tree_node.feature]
    if patient_feature_value >= tree_node.threshold:
        return dt_classify(patient_entry, tree_node.children[1])
    return dt_classify(patient_entry, tree_node.children[0])

    # if patient_entry[subtree_tuple.get_feature()] == subtree_tuple[0]:
    #     return dt_classify(patient_entry, subtree_tuple[1])


# def h_entropy(values: List[]):
#     return -1


def classification_probability(examples: DataFrame, diagnosis):
    # print('classification_probability function: ')
    if examples is None or len(examples) == 0:
        return 0
    count = len(examples[examples['diagnosis'] == diagnosis])
    return (count * 1.0) / len(examples)


def classification_entropy(examples: DataFrame, diagnosis):
    # print('calc_classification_entropy function: ')
    diagnosis_probability = classification_probability(examples, diagnosis)
    log_value = 0
    if diagnosis_probability > 0:
        log_value = log(diagnosis_probability, 2)
    return diagnosis_probability * log_value


def group_entropy(examples: DataFrame):
    # print('calc_group_entropy function: ')
    ill_diagnosis_entropy = classification_entropy(examples, ILL)
    healthy_diagnosis_entropy = classification_entropy(examples, HEALTHY)
    return - ill_diagnosis_entropy - healthy_diagnosis_entropy


# returns the information gain of a specific feature
def ig(examples: DataFrame, feature: str, threshold: float):
    # calc H(E)1
    h_e = group_entropy(examples)
    entries_below, entries_above = filter_dataframe_by_threshold(examples, feature, threshold)
    entries_below_entropy = (len(entries_below) / len(examples)) * group_entropy(entries_below)
    entries_above_entropy = (len(entries_above) / len(examples)) * group_entropy(entries_above)
    return h_e - entries_below_entropy - entries_above_entropy


def max_feature_ig(examples: DataFrame, feature: str):
    # info_gain_list = []
    # for feature in features:
    #     current_info_gain = ig(examples, feature)
    #     info_gain_list.append(current_info_gain)
    # # print('max(info_gain_list)', max(info_gain_list))
    # return max(info_gain_list)
    # sorted_examples = examples[feature].sort_values()

    # sorting the feature of the examples in current node.
    sorted_features = examples[feature].sort_values().tolist()
    thresholds = []

    # defining k-1 thresholds
    for i in range(1, len(sorted_features)):
        thresholds.append((sorted_features[i] + sorted_features[i - 1]) / 2)
    # thresholds = [(sorted_features[i] + sorted_features[i - 1]) / 2 for i in sorted_features]

    # going through thresholds and checking the k-1 binary characters
    curr_max_ig = - 1.0
    maximizing_threshold = -1.0
    for threshold in thresholds:
        curr_ig = ig(examples, feature, threshold)
        # according to the FAQ- in the decision tree, if two splitting threshold values
        # yield the same entropy we should pick the first one (the smallest)
        if curr_ig > curr_max_ig:
            curr_max_ig = curr_ig
            maximizing_threshold = threshold

    return maximizing_threshold, curr_max_ig


# returns the feature with the maximum gain.
# if 2 features has the same maximum gain, we will return the one with the bigger index.
def max_ig(examples: DataFrame, features: List[str]):
    # info_gain_list = []
    # for feature in features:
    #     current_info_gain = ig(examples, feature)
    #     info_gain_list.append(current_info_gain)
    # # print('max(info_gain_list)', max(info_gain_list))
    # return max(info_gain_list)
    if features is None or len(features) == 0:
        return ""
    curr_max_feature = ""
    curr_max_ig = - 1.0
    curr_max_threshold = - 1.0
    # ignoring the first feature. (the diagnosis feature)
    for i in range(1, len(features)):
        curr_feature_threshold, curr_feature_ig = max_feature_ig(examples=examples, feature=features[i])
        if curr_feature_ig >= curr_max_ig:
            curr_max_ig = curr_feature_ig
            curr_max_feature = features[i]
            curr_max_threshold = curr_feature_threshold

    return curr_max_threshold, curr_max_feature
