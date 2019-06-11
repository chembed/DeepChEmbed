from abc import ABC, abstractmethod
from sklearn.cluster import KMeans as sklean_KMeans
from sklearn.metrics import cluster as cluster_metric
from keras.engine.topology import Layer, InputSpec
from keras import backend as K

class Cluster(ABC):
    """
    base class for Clustering model
    """

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = None

        return

    @staticmethod
    def no_label_metrics(input_feature, assigned_label, print_metric,
                         metric='euclidean'):
        """  https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation """
        no_label_metrics = {}
        no_label_metrics['silhouette_score'] = \
            cluster_metric.silhouette_score(input_feature,
                                            assigned_label,
                                            metric=metric)
        no_label_metrics['calinski_score'] = \
            cluster_metric.calinski_harabaz_score(input_feature,
                                                  assigned_label)
        # no_label_metrics['davie_bouldin_score'] = \
        #     cluster_metric.davies_bouldin_score(input_feature,
        #                                         assigned_label)
        if(print_metric):
            print('Metrics without ture labels')
            print("silhouette score: % s"
                  % no_label_metrics['silhouette_score'])
            print("calinski score: % s" % no_label_metrics['calinski_score'])
            # print("davie bouldin score: % s"
            #       % no_label_metrics['davie_bouldin_score'])

        return no_label_metrics

    @staticmethod
    def true_label_metrics(true_label, assigned_label, print_metric):
        """ https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation"""
        true_label_metrics = {}
        true_label_metrics['adjusted_rand_score'] = \
            cluster_metric.adjusted_rand_score(true_label, assigned_label)
        # true_label_metrics['adjusted_mutual_info_score'] = \
        #     cluster_metric.adjusted_mutual_info_score(true_label,
        #                                               assigned_label)
        # true_label_metrics['homogeneity_completeness_v_measure'] = \
        #     cluster_metric.homogeneity_completeness_v_measure(true_label,
        #                                                       assigned_label)
        true_label_metrics['fowlkes_mallows_score'] = \
            cluster_metric.fowlkes_mallows_score(true_label, assigned_label)

        if(print_metric):
            print("Metric with True label")
            print("adjusted rand score: % s "
                  % true_label_metrics['adjusted_rand_score'])
            # print("adjusted mutual info score: % s"
            #       % true_label_metrics['adjusted_mutual_info_score'])
            # print("homogeneity completeness v measure:" )
            # print(true_label_metrics['homogeneity_completeness_v_measure'])
            print("fowlkes_mallows : % s"
                  % true_label_metrics['fowlkes_mallows_score'])

        return true_label_metrics

class KMeans(Cluster):
    """
    Wrapper class utilizing the sklean KMeans class
    """

    def build_model(self, init='k-means++', tol=1e-6):
        """"""
        self.model = sklean_KMeans(n_clusters=self.n_clusters,
                                   init=init, tol=tol)
        return

    def train_model(self, input_feature, true_labels=None, get_metric=True,
                    print_metric=True):

        """ """
        self.model.fit(input_feature)

        avaiable_metrics = {}
        if(get_metric):
            avaiable_metrics.update(\
                self.no_label_metrics(input_feature, self.model.labels_,
                                      print_metric))
            if true_labels is not None:
                print('')
                avaiable_metrics.update(\
                    self.true_label_metrics(true_labels, self.model.labels_,
                                            print_metric))
        else:
            return self.model.labels_

        return (self.model.labels_, avaiable_metrics)


class KMeansLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        Layer.__init__(self,**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
