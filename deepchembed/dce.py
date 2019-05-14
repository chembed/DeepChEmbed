"""
DeepChEmbed Models
"""

from dimreducer import DimReducer
from classifier import Classifier


class DCE():
    """
    class to build a deep chemical embedding model
    """

    def __init__(self, dim_reducer=None, classifier=None):
        """ """
        if dim_reducer is not None:
            self.set_dim_reducer(dim_reducer)
        else:
            self.dim_reducer = None

        if classifier is not None:
            self.set_classifier(classifier)
        else:
            self.classifier = None

        return

    def set_dim_reducer(self, dim_reducer):
        """ """
        assert isinstance(dim_reducer, DimReducer)
        self.dim_reducer = dim_reducer
        return

    def set_classifier(self, classifier):
        """ """
        assert isinstance(classifier, Classifier)
        self.classifier = classifier
        return
