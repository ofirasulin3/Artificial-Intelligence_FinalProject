from typing import List

# import numpy as np
import pandas as pd
# from typing import *
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


def calc_accuracy(test_data_array, classifications):
    # true_positive = 0
    # true_negative = 0
    total = test_data_array.len
    counter = 0
    for item in classifications:
        if item == test_data_array[item][0]:
            counter += 1
    accuracy = (counter * 1.0)/total
    return accuracy


# the classification algorithm. gets an object to classify and the classification tree.
def dt_classify(patient_entry, tree: Node):
    if len(tree.children) == 0:
        # if it's a leaf, we just return the classification stored there.
        return tree.classification
    for subtree_tuple in tree.children:
        if patient_entry[subtree_tuple.get_feature()] == subtree_tuple[0]:
            return dt_classify(patient_entry, subtree_tuple[1])


def ig(examples, feature: str):
    return -1


def max_ig(examples, features: List[str]):
    info_gain_list = []
    for feature in features:
        current_info_gain = ig(feature, examples)
        info_gain_list.append(current_info_gain)
    # print('max(info_gain_list)', max(info_gain_list))
    return max(info_gain_list)
