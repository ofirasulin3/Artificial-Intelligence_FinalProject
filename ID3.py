# from typing import List
import helpers
# from classifier_tree import *
from helpers import *


class ID3:
    # def __init__(self):
    # def __init__(self, train_array, test_array):
    # self.train_array = train_array
    # self.test_data = test_array

    def id3_algo(self, examples, features: List[str]):
        c = majority_class(examples)
        self.td_idt_algo(examples, features, c, max_ig)

    def td_idt_algo(self, examples, features: List[str], default_val, select_feature):
        if examples.len == 0:
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

        f = select_feature(features, examples)
        subtrees = []
        # for item in feature.values
        # subtree = ...
        # value = ...
        # subtree_tuple = [value, subtree]
        # subtrees.append[subtree_tuple]

        return Node(feature=f, children=subtrees, classification=c)


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
    train_data = helpers.get_data_from_csv('train.csv')
    # print('train_data_array:\n', train_data_array, '\n')

    test_data = helpers.get_data_from_csv('test.csv')
    # print('test_data_array:\n', test_data_array)
    features_data = get_features_from_csv('train.csv')
    id3_instance = ID3()
    id3_instance.id3_algo(train_data, features_data)

    # classifier = fit(train_array)
    #
    # predictions = predict(classifier, test_data)
    #
    # accuracy = helpers.calc_accuracy(test_data, predictions)
    #
    # print(accuracy)
