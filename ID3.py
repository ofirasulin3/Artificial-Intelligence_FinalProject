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

        features_empty = len(features) == 0
        # if features are empty and there's noise return a leaf
        if features_empty:
            return Node(feature=None, children=None, classification=c)
        consistent_node = True
        for patient in examples:
            if patient != c:
                consistent_node = False
                break
        # if all patients have the same classification return a leaf
        if consistent_node:
            return Node(feature=None, children=None, classification=c)

        threshold, f = select_feature(examples, features)
        # entries_below = examples[f] < threshold
        # entries_above = examples[f] >= threshold
        entries_below, entries_above = filter_dataframe_by_threshold(examples, f, threshold)

        # for item in feature.values
        # subtree = ...
        # value = ...
        # subtree_tuple = [value, subtree]
        # subtrees.append[subtree_tuple]

        # subtrees_tuples = []
        # subtree1 = (threshold, self.td_idt_algo(entries_below, features, c, select_feature))
        # subtrees_tuples.append(subtree1)
        # subtree2 = (threshold, self.td_idt_algo(entries_above, features, c, select_feature))
        # subtrees_tuples.append(subtree2)

        subtree1 = self.td_idt_algo(entries_below, features, c, select_feature)
        subtree2 = self.td_idt_algo(entries_above, features, c, select_feature)
        children_tuple = (subtree1, subtree2)

        return Node(feature=f, children=children_tuple, classification=c, threshold=threshold)


def majority_class(patients):
    ill = 0
    not_ill = 0
    for patient in patients:
        if patient == HEALTHY:
            not_ill += 1
        else:
            ill += 1
    return ILL if ill > not_ill else HEALTHY


if __name__ == '__main__':

    # print(train_data)

    # filter_by_threshold = train_data['radius_mean'] < 20
    # data_below = train_data[filter_by_threshold]
    # print(data_below, '\n', '\n')

    # print(test_data['radius_mean'], '\n')
    # print(test_data['radius_mean'].sort_values().tolist(), '\n')
    # # print(test_data.sort_values(by='radius_mean'), '\n')
    # sorted_features = test_data['radius_mean'].sort_values().tolist()
    # thresholds = []
    # for i in range(1, len(sorted_features)):
    #     thresholds.append((sorted_features[i] + sorted_features[i - 1]) / 2.0)
    # print('thresholds: \n', thresholds)

    # predictions = []
    # for patient_entry in test_data.iterrows():
    #     print(patient_entry, '\n')
    #     prediction = dt_classify(patient_entry, classifier)
    #     predictions.append(prediction)
    #

    train_data = helpers.get_data_from_csv('train.csv')
    # print('train_data_array:\n', train_data_array, '\n')

    test_data = helpers.get_data_from_csv('test.csv')
    # print('test_data:\n', test_data)
    # print('len(test_data): ', len(test_data))
    features_data = get_features_from_csv('train.csv')
    id3_instance = ID3()

    classifier_tree = id3_instance.id3_algo(train_data, features_data)
    # classifier = fit(train_array)
    predictions = calc_predictions(test_data, classifier_tree)

    accuracy = helpers.calc_accuracy(test_data, predictions)

    print('\n\n\n\n\nfinal accuracy: ', accuracy)
