
# A class for holding a tree with classifying nodes.
# Will be used to build the classifier tree
class Node(object):
    def __init__(self, feature, children: tuple = None, classification: str = None, threshold: float = None):
        self.feature = feature
        self.children = children
        self.classification = classification
        self.threshold = threshold

    def get_feature(self):
        return self.feature

    def get_children(self):
        return self.children

    def get_classification(self):
        return self.classification

    def get_threshold(self):
        return self.threshold
