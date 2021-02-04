# import classifier_tree
from ID3 import ID3
import helpers
from helpers import *
from math import sqrt


class KNNForest:
    # def __init__(self, n_trees: List[classifier_tree] = None, centroids: List[DataFrame] = None):
    def __init__(self, n_trees=None, centroids=None):
        self.n_trees = n_trees
        self.centroids = centroids

    # ID3 algorithm
    def knn_decision_tree_algo(self, examples: DataFrame, features: List[str], testing_data: DataFrame):
        print('knn_decision_tree_algo function: ')
        # c = majority_class(examples)
        self.knn_learning(examples, features)
        accuracy = self.knn_testing(examples, features, testing_data)
        return accuracy

    def knn_learning(self, examples: DataFrame, features: List[str]):
        print('knn_learning function: ')
        self.centroids = []
        self.n_trees = []
        number_of_trees = N_PARAM_FOR_KNN
        # K = 5
        p_param = P_PARAM_FOR_KNN
        for i in range(0, number_of_trees):
            sample_len = p_param * len(examples)
            curr_train_data = examples.sample(int(sample_len))
            self.n_trees.append(ID3().id3_algo(curr_train_data, features))
            self.centroids.append(curr_train_data.mean())
            # TODO: don't forget the first feature is 'M' or 'B'...

    def knn_testing(self, examples: DataFrame, features: List[str], testing_data: DataFrame):
        print('knn_testing function: ')
        count = 0
        # for person in test_data:
        for index, patient_entry in testing_data.iterrows():
            k_trees = self.get_k_trees(self.centroids, patient_entry, K_PARAM_FOR_KNN, features)
            chosen_k_trees = [self.n_trees[i] for i in k_trees]
            classification = classify_test_example(chosen_k_trees, patient_entry)
            if patient_entry['diagnosis'] == classification:
                count += 1

        return count * 1.0 / len(test_data)

    def get_k_trees(self, centroids, patient_entry, k_param, features: List[str]):
        print('get_k_trees function: ')
        distances = []
        for centroid in centroids:
            distances.append(calc_distance(patient_entry, centroid, features))
        trees = distances.sort(key=lambda k: distances[k])
        return trees[:k_param]


def classify_test_example(k_trees, patient_entry):
    print('classify_test_example function: ')
    classifications = []
    healthy = 0
    ill = 0
    for tree in k_trees:
        classification = dt_classify(patient_entry, tree)
        classifications.append(classification)
        if classification == HEALTHY:
            healthy += 1
        else:
            ill += 1
    return ILL if ill >= healthy else HEALTHY


def calc_distance(patient_entry: DataFrame, centroid: DataFrame, features: List[str]):
    print('calc_distance function: ')
    # for feature in features:
    distance = 0.0
    for i in range(1, len(features)):
        curr_dist = patient_entry[features[i]] - centroid[features[i]]
        curr_dist_squared = curr_dist ** 2
        distance += sqrt(curr_dist_squared)


if __name__ == '__main__':

    # getting data from csv files:
    train_data = helpers.get_data_from_csv('train.csv')
    test_data = helpers.get_data_from_csv('test.csv')
    features_data = get_features_from_csv('train.csv')

    # print(train_data.mean())

    # exercise 1:

    knn_instance = KNNForest().knn_decision_tree_algo(train_data, features_data, test_data)
    print(knn_instance)

    # ----------------------------------------------------------------------------------------------
