
import helpers
from helpers import *


class CostSensitiveID3:

    # CostSensitiveID3 algorithm
    def cost_sensitive_id3_algo(self, examples: DataFrame, features: List[str]):
        c = cost_sensitive_majority_class(examples)
        # adaption for cost sensitive algorithm: calling cost_sensitive_max_ig
        return self.cost_sensitive_td_idt_algo(examples, features, c, cost_sensitive_max_ig)

    # CostSensitive_TD_IDT algorithm for CostSensitiveID3 algorithm
    def cost_sensitive_td_idt_algo(self, examples: DataFrame, features: List[str], default_val, select_feature):
        if len(examples) == 0:
            # Empty leaf. Use default classification
            return Node(feature=None, children=None, classification=default_val)
        c = cost_sensitive_majority_class(examples)

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
        subtree1 = self.cost_sensitive_td_idt_algo(entries_below, features, c, select_feature)
        subtree2 = self.cost_sensitive_td_idt_algo(entries_above, features, c, select_feature)
        # add them as children to current node.
        children_tuple = (subtree1, subtree2)

        return Node(feature=f, children=children_tuple, classification=c, threshold=threshold)

    # ID3 algorithm with pre pruning
    def cost_sensitive_id3_pruning(self, examples: DataFrame, features: List[str], pruning_m):
        # print('m_pruning_val inside id3_pruning is: ', pruning_m, '\n')
        c = cost_sensitive_majority_class(examples)
        return self.cost_sensitive_td_idt_pruning(examples, features, c, cost_sensitive_max_ig, pruning_m)

    # TD_IDT algorithm for ID3 algorithm with pre pruning
    def cost_sensitive_td_idt_pruning(self, examples: DataFrame, features: List[str], default_val, select_feature,
                                      pruning_m):
        if len(examples) < pruning_m:
            # there is enough examples in current node so we can stop
            return Node(feature=None, children=None, classification=default_val)

        if len(examples) == 0:
            # Empty leaf. Use default classification
            return Node(feature=None, children=None, classification=default_val)
        c = cost_sensitive_majority_class(examples)

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
        subtree1 = self.cost_sensitive_td_idt_pruning(entries_below, features, c, select_feature, pruning_m)
        subtree2 = self.cost_sensitive_td_idt_pruning(entries_above, features, c, select_feature, pruning_m)
        # add them as children to current node.
        children_tuple = (subtree1, subtree2)

        return Node(feature=f, children=children_tuple, classification=c, threshold=threshold)


# gets patients entries and returns the most common diagnosis within them
def cost_sensitive_majority_class(patients):
    ill = 0
    healthy = 0
    for index, patient_entry in patients.iterrows():
        if patient_entry['diagnosis'] == HEALTHY:
            healthy += 1
        else:
            ill += 1
    # adaption for cost sensitive algorithm:
    return ILL if 3 * ill >= healthy else HEALTHY


# calculating the entropy for a group of examples
def cost_sensitive_group_entropy(examples: DataFrame):
    ill_diagnosis_entropy = classification_entropy(examples, ILL)
    healthy_diagnosis_entropy = classification_entropy(examples, HEALTHY)
    # adaption for cost sensitive algorithm:
    return - 10 * ill_diagnosis_entropy - healthy_diagnosis_entropy


# returns the information gain of a specific feature
def cost_sensitive_ig(examples: DataFrame, feature: str, threshold: float):
    # calc H(E)
    h_e = cost_sensitive_group_entropy(examples)
    entries_below, entries_above = filter_dataframe_by_threshold(examples, feature, threshold)
    entries_below_entropy = (len(entries_below) / len(examples)) * cost_sensitive_group_entropy(entries_below)
    entries_above_entropy = (len(entries_above) / len(examples)) * cost_sensitive_group_entropy(entries_above)
    return h_e - entries_below_entropy - entries_above_entropy


# returns the maximum information gain of a specific feature (for dynamic ID3)
def cost_sensitive_max_feature_ig(examples: DataFrame, feature: str):

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
        curr_ig = cost_sensitive_ig(examples, feature, threshold)
        # according to the FAQ- in the decision tree, if two splitting threshold values
        # yield the same entropy we should pick the first one (the smallest)
        if curr_ig > curr_max_ig:
            curr_max_ig = curr_ig
            maximizing_threshold = threshold

    return maximizing_threshold, curr_max_ig


# returns the feature with the maximum gain.
# if 2 features has the same maximum gain, we will return the one with the bigger index.
def cost_sensitive_max_ig(examples: DataFrame, features: List[str]):
    if features is None or len(features) == 0:
        return ""
    curr_max_feature = ""
    curr_max_ig = - 1.0
    curr_max_threshold = - 1.0
    # ignoring the first feature. (the diagnosis feature)
    for i in range(1, len(features)):
        curr_feature_threshold, curr_feature_ig = cost_sensitive_max_feature_ig(examples=examples, feature=features[i])
        if curr_feature_ig >= curr_max_ig:
            curr_max_ig = curr_feature_ig
            curr_max_feature = features[i]
            curr_max_threshold = curr_feature_threshold

    return curr_max_threshold, curr_max_feature


if __name__ == '__main__':

    # getting data from csv files:
    train_data = helpers.get_data_from_csv('train.csv')
    test_data = helpers.get_data_from_csv('test.csv')
    features_data = get_features_from_csv('train.csv')

    # ----------------------------------------------------------------------------------------------

    # exercise 4.3:

    cost_sensitive_id3_pruning_instance4_3 = CostSensitiveID3()
    classifier_tree = cost_sensitive_id3_pruning_instance4_3.cost_sensitive_id3_pruning(train_data, features_data,
                                                                                        pruning_m=2)
    loss = helpers.calc_loss(test_data, classifier_tree)
    print(loss)
