# from typing import List
from pandas import DataFrame

import helpers
# from classifier_tree import *
from helpers import *


class ID3:
    # def __init__(self):
    # def __init__(self, train_array, test_array):
    # self.train_array = train_array
    # self.test_data = test_array

    def id3_algo(self, examples: DataFrame, features: List[str]):
        c = majority_class(examples)
        return self.td_idt_algo(examples, features, c, max_ig)

    def td_idt_algo(self, examples: DataFrame, features: List[str], default_val, select_feature):
        if len(examples) == 0:
            # Empty leaf. Use default classification
            return Node(feature=None, children=None, classification=default_val)
        c = majority_class(examples)

        # if features are empty and there's noise return a leaf
        if len(features) == 0:
            return Node(feature=None, children=None, classification=c)
        consistent_node = True
        for index, patient_entry in examples.iterrows():
            if patient_entry['diagnosis'] != c:
                consistent_node = False
                break
        # if all patients have the same classification return a leaf
        if consistent_node:
            return Node(feature=None, children=None, classification=c)

        threshold, f = select_feature(examples, features)
        entries_below, entries_above = filter_dataframe_by_threshold(examples, f, threshold)

        # divide the tree to 2 subtrees according to the threshold
        subtree1 = self.td_idt_algo(entries_below, features, c, select_feature)
        subtree2 = self.td_idt_algo(entries_above, features, c, select_feature)
        # add them as children to current node.
        children_tuple = (subtree1, subtree2)

        return Node(feature=f, children=children_tuple, classification=c, threshold=threshold)

    def id3_pruning(self, examples: DataFrame, features: List[str], pruning_m):
        c = majority_class(examples)
        return self.td_idt_algo(examples, features, c, max_ig, pruning_m)

    def td_idt_pruning(self, examples: DataFrame, features: List[str], default_val, select_feature, pruning_m):
        if len(examples) < pruning_m:
            # there is enough examples in current node so we can stop
            return Node(feature=None, children=None, classification=default_val)

        if len(examples) == 0:
            # Empty leaf. Use default classification
            return Node(feature=None, children=None, classification=default_val)
        c = majority_class(examples)

        # if features are empty and there's noise return a leaf
        if len(features) == 0:
            return Node(feature=None, children=None, classification=c)
        consistent_node = True
        for index, patient_entry in examples.iterrows():
            if patient_entry['diagnosis'] != c:
                consistent_node = False
                break
        # if all patients have the same classification return a leaf
        if consistent_node:
            return Node(feature=None, children=None, classification=c)

        threshold, f = select_feature(examples, features)
        entries_below, entries_above = filter_dataframe_by_threshold(examples, f, threshold)

        # divide the tree to 2 subtrees according to the threshold
        subtree1 = self.td_idt_algo(entries_below, features, c, select_feature)
        subtree2 = self.td_idt_algo(entries_above, features, c, select_feature)
        # add them as children to current node.
        children_tuple = (subtree1, subtree2)

        return Node(feature=f, children=children_tuple, classification=c, threshold=threshold)


def majority_class(patients):
    ill = 0
    healthy = 0
    for index, patient_entry in patients.iterrows():
        if patient_entry['diagnosis'] == HEALTHY:
            healthy += 1
        else:
            ill += 1
    return ILL if ill > healthy else HEALTHY


if __name__ == '__main__':
    # filter_by_threshold = train_data['radius_mean'] < 20
    # data_below = train_data[filter_by_threshold]
    # print(data_below, '\n', '\n')

    # print(test_data['radius_mean'], '\n')
    # # print(test_data.sort_values(by='radius_mean'), '\n')

    # ex1:
    train_data = helpers.get_data_from_csv('train.csv')
    test_data = helpers.get_data_from_csv('test.csv')
    features_data = get_features_from_csv('train.csv')
    id3_instance = ID3()
    # print('\n building classifier_tree (id3_algo): \n')
    classifier_tree = id3_instance.id3_algo(train_data, features_data)
    # print('\n\n calculating_predictions: \n')
    # print('\n\n accuracy: \n')
    accuracy = helpers.calc_accuracy(test_data, classifier_tree)
    print(accuracy)
