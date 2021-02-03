
# a class for holding the classifier tree.
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
