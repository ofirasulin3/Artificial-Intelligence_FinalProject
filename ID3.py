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
        print('id3_algo function: ')
        c = majority_class(examples)
        # print('in id3_algo, c majority_class(examples) is: ', c, '\n')
        return self.td_idt_algo(examples, features, c, max_ig)

    def td_idt_algo(self, examples: DataFrame, features: List[str], default_val, select_feature):
        print('td_idt_algo function: \n')
        # print('len(examples): ', len(examples), '\n')
        if len(examples) == 0:
            print('found that len(examples) is 0 \n')
            # Empty leaf. Use default classification
            return Node(feature=None, children=None, classification=default_val)
        c = majority_class(examples)
        # print('c majority_class(examples) is: ', c, '\n')

        # features_empty = len(features) == 0
        # if features are empty and there's noise return a leaf
        if len(features) == 0:
            # print('found that len(features) is 0 \n')
            return Node(feature=None, children=None, classification=c)
        consistent_node = True
        # for patient in examples:
        #     if patient['diagnosis'] != c:
        for index, patient_entry in examples.iterrows():
            if patient_entry['diagnosis'] != c:
                consistent_node = False
                break
        # if all patients have the same classification return a leaf
        if consistent_node:
            print('found consistent_node: \n')
            return Node(feature=None, children=None, classification=c)

        threshold, f = select_feature(examples, features)
        # entries_below = examples[f] < threshold
        # entries_above = examples[f] >= threshold
        entries_below, entries_above = filter_dataframe_by_threshold(examples, f, threshold)

        # print('examples: ', examples, '\n')
        # print('entries_below: ', entries_below, '\n')

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
        # print('subtree1 classification: ', subtree1.classification, '\n')
        subtree2 = self.td_idt_algo(entries_above, features, c, select_feature)
        # print('subtree2 classification: ', subtree2.classification, '\n')

        children_tuple = (subtree1, subtree2)
        # print('children_tuple: ', children_tuple, '\n')

        print('finito la comedia: \n')

        return Node(feature=f, children=children_tuple, classification=c, threshold=threshold)


def majority_class(patients):
    ill = 0
    healthy = 0
    # for patient in patients:
    #     if patient == HEALTHY:
    for index, patient_entry in patients.iterrows():
        if patient_entry['diagnosis'] == HEALTHY:
            healthy += 1
        else:
            ill += 1
    return ILL if ill > healthy else HEALTHY


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

    print('\nbuilding classifier_tree (id3_algo): \n')

    classifier_tree = id3_instance.id3_algo(train_data, features_data)
    # classifier = fit(train_array)
    print('\n\ncalculating_predictions: \n')

    # predictions = calc_predictions(test_data, classifier_tree)

    print('\n\naccuracy: \n')

    accuracy = helpers.calc_accuracy(test_data, classifier_tree)

    print('\n\n\n\n\nfinal accuracy: ', accuracy)
