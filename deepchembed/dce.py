"""
DeepChEmbed Models
"""

from dimreducer import DimReducer
from cluster import Cluster


class DCE():
    """
    class to build a deep chemical embedding model
    """

    def __init__(self, dim_reducer=None, cluster=None):
        """ """
        if dim_reducer is not None:
            self.set_dim_reducer(dim_reducer)
        else:
            self.dim_reducer = None

        if cluster is not None:
            self.set_classifier(classifier)
        else:
            self.cluster = None

        return

    def set_dim_reducer(self, dim_reducer):
        """ """
        assert isinstance(dim_reducer, DimReducer)
        self.dim_reducer = dim_reducer
        return

    def set_classifier(self, cluster):
        """ """
        assert isinstance(cluster, Cluster)
        self.cluster = cluster
        return
