

class Node(object):
    def __init__(self, feature, children: list = None, classification: str = None, value=None):
        self.feature = feature
        self.children = children
        self.classification = classification
        self.value = value

    def get_feature(self):
        return self.feature

    # def add_child(self, child):
    #     self.children.append(child)
