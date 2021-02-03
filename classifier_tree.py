

class Node(object):
    def __init__(self, feature, children: tuple = None, classification: str = None, threshold: float = None):
        self.feature = feature
        self.children = children
        self.classification = classification
        # self.value = value
        self.threshold = threshold

    def get_feature(self):
        return self.feature

    # def add_child(self, child):
    #     self.children.append(child)
