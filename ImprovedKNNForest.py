from ID3 import ID3
import helpers
from helpers import *
from math import sqrt


class ImprovedKNNForest:
    def __init__(self, n_trees=None, centroids=None):
        self.n_trees = n_trees
        self.centroids = centroids

    # This functions does the whole knn_decision_tree algorithm and returns the accuracty at the end
    def improved_knn_decision_tree_algo(self, examples: DataFrame, features: List[str], testing_data: DataFrame):
        self.improved_knn_learning(examples, features)
        accuracy = self.improved_knn_testing(examples, features, testing_data)
        return accuracy

    # This function does the learning part of the learning algorithm, with the examples data
    def improved_knn_learning(self, examples: DataFrame, features: List[str]):
        self.centroids = []
        self.n_trees = []
        number_of_trees = N_PARAM_FOR_KNN
        p_param = P_PARAM_FOR_KNN
        for i in range(0, number_of_trees):
            sample_len = p_param * len(examples)
            curr_train_data = examples.sample(int(sample_len))
            self.n_trees.append(ID3().id3_algo(curr_train_data, features))
            self.centroids.append(curr_train_data.mean())

    # This function does the testing part of the learning algorithm, with the testing_data,
    # and returns the accuracy it got
    def improved_knn_knn_testing(self, examples: DataFrame, features: List[str], testing_data: DataFrame):
        count = 0
        for index, patient_entry in testing_data.iterrows():
            distances = self.get_distances(self.centroids, patient_entry, features)
            indexes = range(len(distances))
            sorted_trees = sorted(indexes, key=lambda k: distances[k])
            k_trees = sorted_trees[:K_PARAM_FOR_KNN]
            chosen_k_trees = [self.n_trees[i] for i in k_trees]
            classification = improved_knn_classify_patient_example(chosen_k_trees, patient_entry, distances)
            if patient_entry['diagnosis'] == classification:
                count += 1

        return count * 1.0 / len(test_data)

    # get distances between centroids to current patient
    def improved_knn_get_distances(self, centroids, patient_entry, features: List[str]) -> List[float]:
        distances = []
        for centroid in centroids:
            distances.append(improved_knn_calc_distance(patient_entry, centroid, features))
        return distances


# getting a patient entry of data, and k trees,
# and classifying the patient according to the majority of the k trees decisions.
def improved_knn_classify_patient_example(k_trees, patient_entry, distances):
    classifications = []
    healthy = 0
    ill = 0
    healthy_weight = 0.0
    ill_weight = 0.0
    index = 0
    for tree in k_trees:
        classification = dt_classify(patient_entry, tree)
        classifications.append(classification)
        if classification == HEALTHY:
            healthy += 1
            healthy_weight += 1 / distances[index]
        else:
            ill += 1
            ill_weight += 1 / distances[index]
        index += 1
    return ILL if ill_weight >= healthy_weight else HEALTHY
    # return ILL if ill >= healthy else HEALTHY


# calculating the distance between a centroid of a tree to a given patient
def improved_knn_calc_distance(patient_entry: DataFrame, centroid: DataFrame, features: List[str]):
    distance = 0.0
    for i in range(1, len(features)):
        curr_dist = patient_entry[features[i]] - centroid[features[i]]
        curr_dist_squared = curr_dist ** 2
        distance += sqrt(curr_dist_squared)
    return distance


if __name__ == '__main__':

    # getting data from csv files:
    train_data = helpers.get_data_from_csv('train.csv')
    test_data = helpers.get_data_from_csv('test.csv')
    features_data = get_features_from_csv('train.csv')

    improved_knn_instance = ImprovedKNNForest().improved_knn_decision_tree_algo(train_data, features_data, test_data)
    print(improved_knn_instance)

    # ----------------------------------------------------------------------------------------------
