
from typing import List
from math import log
import pandas as pd
from pandas import DataFrame
from classifier_tree import *

# defining the letters for the 2 available diagnosis for convenient use later
ILL = 'M'
HEALTHY = 'B'


# exporting the features from a given csv file and returns a list of it
def get_features_from_csv(file_name):
    df = pd.read_csv(file_name)
    return df.columns.tolist()


# converting a given csv file in current folder to a dataframe
def get_data_from_csv(file_name):
    # Using the function to load the data of file_name.csv into a Dataframe df
    df = pd.read_csv(file_name)
    return df


# gets a data frame, a feature and a threshold and returns the 2 dataframes split by the threshold value
def filter_dataframe_by_threshold(dataframe, feature: str, threshold: float):
    below_threshold = dataframe[feature] < threshold
    above_threshold = dataframe[feature] >= threshold
    entries_below = dataframe[below_threshold]
    entries_above = dataframe[above_threshold]
    return entries_below, entries_above


# calculates a list of predictions done by a given classifier on a given test data frame
def calc_predictions(test_data: DataFrame, classifier: Node):
    predictions = []
    for index, patient_entry in test_data.iterrows():
        prediction = dt_classify(patient_entry, classifier)
        predictions.append(prediction)
    return predictions


# calculates the accuracy of a given classifier on a test data frame
def calc_accuracy(test_dataframe: DataFrame, classifier: Node):
    real_classifications = test_dataframe['diagnosis'].tolist()
    predictions = calc_predictions(test_dataframe, classifier)
    total = len(real_classifications)
    counter = 0
    for index, item in enumerate(real_classifications):
        if item == predictions[index]:
            counter += 1
    accuracy = (counter * 1.0)/total
    return accuracy


# calculates the loss of a given classifier on a test data frame
def calc_loss(test_dataframe: DataFrame, classifier: Node):
    false_positive = 0
    # prediction is ill but real classification is healthy
    false_negative = 0
    # prediction is healthy but real classification is ill

    real_classifications = test_dataframe['diagnosis'].tolist()
    predictions = calc_predictions(test_dataframe, classifier)
    total = len(real_classifications)
    for index, real in enumerate(real_classifications):
        if predictions[index] == ILL and real == HEALTHY:
            false_positive += 1
        if predictions[index] == HEALTHY and real == ILL:
            false_negative += 1
    loss = (false_positive * 0.1 + false_negative)/total
    return loss


# the classification algorithm. gets an object to classify and the classification tree.
def dt_classify(patient_entry, tree_node: Node):
    if tree_node.get_children() is None or len(tree_node.get_children()) == 0\
            or (tree_node.get_children()[0] is None and tree_node.get_children()[1] is None):
        # if it's a leaf, we just return the classification stored there.
        return tree_node.get_classification()

    # for subtree in tree_node.children:
    patient_feature_value = patient_entry[tree_node.get_feature()]
    if patient_feature_value >= tree_node.get_threshold():
        return dt_classify(patient_entry, tree_node.get_children()[1])
    return dt_classify(patient_entry, tree_node.get_children()[0])


# calculating the probability for a specific classification in a given data
def classification_probability(examples: DataFrame, diagnosis):
    if examples is None or len(examples) == 0:
        return 0
    count = len(examples[examples['diagnosis'] == diagnosis])
    return (count * 1.0) / len(examples)


# calculating the entropy for a specific classification
def classification_entropy(examples: DataFrame, diagnosis):
    diagnosis_probability = classification_probability(examples, diagnosis)
    log_value = 0
    if diagnosis_probability > 0:
        log_value = log(diagnosis_probability, 2)
    return diagnosis_probability * log_value


# calculating the entropy for a group of examples
def group_entropy(examples: DataFrame):
    ill_diagnosis_entropy = classification_entropy(examples, ILL)
    healthy_diagnosis_entropy = classification_entropy(examples, HEALTHY)
    return - ill_diagnosis_entropy - healthy_diagnosis_entropy


# returns the information gain of a specific feature
def ig(examples: DataFrame, feature: str, threshold: float):
    # calc H(E)
    h_e = group_entropy(examples)
    entries_below, entries_above = filter_dataframe_by_threshold(examples, feature, threshold)
    entries_below_entropy = (len(entries_below) / len(examples)) * group_entropy(entries_below)
    entries_above_entropy = (len(entries_above) / len(examples)) * group_entropy(entries_above)
    return h_e - entries_below_entropy - entries_above_entropy


# returns the maximum information gain of a specific feature (for dynamic ID3)
def max_feature_ig(examples: DataFrame, feature: str):

    # sorting the feature of the examples in current node.
    sorted_features = examples[feature].sort_values().tolist()
    thresholds = []

    # defining k-1 thresholds
    for i in range(1, len(sorted_features)):
        thresholds.append((sorted_features[i] + sorted_features[i - 1]) / 2)

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
