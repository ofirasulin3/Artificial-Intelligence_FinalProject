
from sklearn.model_selection import KFold
import helpers
from helpers import *
import matplotlib.pyplot as plt


class ID3:
    # def __init__(self):
    # def __init__(self, train_array, test_array):
    # self.train_array = train_array
    # self.test_data = test_array

    # ID3 algorithm
    def id3_algo(self, examples: DataFrame, features: List[str]):
        c = majority_class(examples)
        return self.td_idt_algo(examples, features, c, max_ig)

    # TD_IDT algorithm for ID3 algorithm
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

    # ID3 algorithm with pre pruning
    def id3_pruning(self, examples: DataFrame, features: List[str], pruning_m):
        # print('m_pruning_val inside id3_pruning is: ', pruning_m, '\n')
        c = majority_class(examples)
        return self.td_idt_pruning(examples, features, c, max_ig, pruning_m)

    # TD_IDT algorithm for ID3 algorithm with pre pruning
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
        subtree1 = self.td_idt_pruning(entries_below, features, c, select_feature, pruning_m)
        subtree2 = self.td_idt_pruning(entries_above, features, c, select_feature, pruning_m)
        # add them as children to current node.
        children_tuple = (subtree1, subtree2)

        return Node(feature=f, children=children_tuple, classification=c, threshold=threshold)


# to run exercise 3.3:
# search for "exercise 3.3" comment in the main, and read the instructions there.
def experiment(training_data: DataFrame, features_names):
    m_pruning_values = [2, 4, 8, 12, 16]

    accuracies_list = []
    avg = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=316251305)
    for m_pruning_val in m_pruning_values:
        accuracy_sum = 0
        size_for_avg = 0
        curr_accuracies_list = []
        for train_i, test_i in kf.split(training_data):

            train_info = training_data.iloc[train_i]

            test_info = training_data.iloc[test_i]

            curr_classifier_tree = ID3().id3_pruning(train_info, features_names, m_pruning_val)

            curr_accuracy = helpers.calc_accuracy(test_info, curr_classifier_tree)
            curr_accuracies_list.append(curr_accuracy)

            accuracy_sum += curr_accuracy
            size_for_avg += 1

        avg = accuracy_sum / size_for_avg
        accuracies_list.append(avg)

    # Data for plotting
    fig1, ax1 = plt.subplots()
    ax1.plot(m_pruning_values, accuracies_list)

    ax1.set(xlabel='M', ylabel='Accuracy', title='Accuracy as a function of M')
    ax1.grid()

    plt.show()


# gets patients entries and returns the most common diagnosis within them
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

    # getting data from csv files:
    train_data = helpers.get_data_from_csv('train.csv')
    test_data = helpers.get_data_from_csv('test.csv')
    features_data = get_features_from_csv('train.csv')

    # exercise 1:
    # id3_instance = ID3()
    # classifier_tree = id3_instance.id3_algo(train_data, features_data)
    # accuracy = helpers.calc_accuracy(test_data, classifier_tree)
    # print(accuracy)

    # exercise 3.3:
    # uncomment the coming line with "experiment" call, for running the experiment.
    # make sure the 3 lines at the start of main (for getting data from csv files) are uncommented

    # experiment(train_data, features_data)

    # exercise 3.4:
    # id3_pruning_instance3_4 = ID3()
    # classifier_tree = id3_pruning_instance3_4.id3_pruning(train_data, features_data, pruning_m=2)
    # accuracy = helpers.calc_accuracy(test_data, classifier_tree)
    # print(accuracy)

    # exercise 4.1:
    id3_pruning_instance3_4 = ID3()
    classifier_tree = id3_pruning_instance3_4.id3_pruning(train_data, features_data, pruning_m=2)
    loss = helpers.calc_loss(test_data, classifier_tree)
    print(loss)
