from abc import ABC, abstractmethod
from sklearn.cluster import KMeans as sklean_KMeans
from sklearn.metrics import cluster as cluster_metric

class Cluster(ABC):
    """
    base class for Clustering model
    """

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = None

        return

class KMeans(Cluster):
    """
    Wrapper class utilizing the sklean KMeans class
    """

    def bulid_kmeans_cluster(self, init='k-means++', tol=1e-6):
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
        no_label_metrics['davie_bouldin_score'] = \
            cluster_metric.davies_bouldin_score(input_feature,
                                                assigned_label)
        if(print_metric):
            print('Metrics without ture labels')
            print("silhouette score: % s"
                  % no_label_metrics['silhouette_score'])
            print("calinski score: % s" % no_label_metrics['calinski_score'])
            print("davie bouldin score: % s"
                  % no_label_metrics['davie_bouldin_score'])

        return no_label_metrics

    @staticmethod
    def true_label_metrics(true_label, assigned_label, print_metric):
        """ https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation"""
        true_label_metrics = {}
        true_label_metrics['adjusted_rand_score'] = \
            cluster_metric.adjusted_rand_score(true_label, assigned_label)
        true_label_metrics['adjusted_mutual_info_score'] = \
            cluster_metric.adjusted_mutual_info_score(true_label,
                                                      assigned_label)
        true_label_metrics['homogeneity_completeness_v_measure'] = \
            cluster_metric.homogeneity_completeness_v_measure(true_label,
                                                              assigned_label)
        true_label_metrics['fowlkes_mallows_score'] = \
            cluster_metric.fowlkes_mallows_score(true_label, assigned_label)

        if(print_metric):
            print("Metric with True label")
            print("adjusted rand score: % s "
                  % true_label_metrics['adjusted_rand_score'])
            print("adjusted mutual info score: % s"
                  % true_label_metrics['adjusted_mutual_info_score'])
            print("homogeneity completeness v measure:" )
            print(true_label_metrics['homogeneity_completeness_v_measure'])
            print("fowlkes_mallows : % s"
                  % true_label_metrics['fowlkes_mallows_score'])

        return true_label_metrics
