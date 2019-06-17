"""
DeepChEmbed Models
"""
from dimreducer import DeepAutoEncoder
from cluster import KMeansLayer
from cluster import KMeans
from keras import Model
from keras import optimizers
from keras.utils import normalize
import numpy as np

class DCE():
    """
    class to build a deep chemical embedding model
    """

    def __init__(self, autoencoder_dims, n_clusters, update_interval=50,
                 max_iteration=1e4, clustering_tol=1e-4, alpha=1.0):
        """ """
        self.autoencoder_dims = autoencoder_dims
        self.n_clusters       = n_clusters
        self.alpha            = alpha
        self.update_interval  = update_interval
        self.max_iteration    = max_iteration
        self.clustering_tol   = clustering_tol
        self.model            = None

        return

    def build_model(self, norm=True, act='relu'):
        """ """
        autoencoder = DeepAutoEncoder(self.autoencoder_dims, act)
        autoencoder.build_model(norm=norm)
        embeding = autoencoder.model.get_layer(name='embedding_layer').output
        clustering = KMeansLayer(self.n_clusters, alpha=self.alpha,
                                 name='clustering')(embeding)
        self.model = Model(inputs=autoencoder.model.input,
                           outputs=[clustering,autoencoder.model.output])

        return

    def train_model(self, data_train, norm_feature=True, training_prints=True,
                    compiled=False, clustering_loss='kld', decoder_loss='mse',
                    clustering_loss_weight=0.5,
                    hardening_method='simple',
                    hardening_order=2.0,
                    optimizer='adam', lr=0.001, decay=0.0):
        """ """
        if (not compiled):
            assert clustering_loss_weight <= 1 and clustering_loss_weight >= 0

            if optimizer == 'adam':
                dce_optimizer = optimizers.Adam(lr=lr,decay=decay)
            elif optimizer == 'sgd':
                dce_optimizer = optimizers.sgd(lr=lr,decay=decay)
            else:
                raise Exception('Input optimizer was not found')

            self.model.compile(loss={'clustering': clustering_loss,
                                     'decoder_output': decoder_loss},
                               loss_weights=[clustering_loss_weight,
                                             1 - clustering_loss_weight],
                               optimizer=dce_optimizer)

        if(norm_feature):
            data_train = normalize(data_train, axis=0, order=2)

        # initializing model by using sklean-Kmeans as guess
        kmeans_init = KMeans(n_clusters=self.n_clusters)
        kmeans_init.build_model()
        encoder  = Model(inputs=self.model.input,
                         outputs=self.model.get_layer(\
                         name='embedding_layer').output)
        kmeans_init.model.fit(encoder.predict(data_train))
        y_pred_last = kmeans_init.model.labels_
        self.model.get_layer(name='clustering').\
            set_weights([kmeans_init.model.cluster_centers_])

        assert hardening_method in ['simple', 'smoothstep']
        assert hardening_order >=1
        if hardening_method == 'simple':
            hardening_func = DCE.simple_hardening
        else:
            hardening_func = DCE.smoothstep_hardening

        loss = []
        delta_label = []

        for iteration in range(int(self.max_iteration)):

            if iteration % self.update_interval == 0:
                # updating centroid
                q, _ = self.model.predict(data_train)
                p = hardening_func(q, n=hardening_order)
                y_pred = q.argmax(1)
                delta_label_i = np.sum(y_pred != y_pred_last).\
                    astype(np.float32) / y_pred.shape[0]
                delta_label.append(delta_label_i)
                y_pred_last = y_pred

                # exam convergence
                if iteration > 0 and delta_label_i < self.clustering_tol:
                    print(str(delta_label_i) +' < ' + str(self.clustering_tol))
                    print('Reached tolerance threshold. Stopping training.')
                    break

            loss_i = self.model.train_on_batch(x=data_train,
                                               y=[p,data_train])
            loss.append(loss_i)

            if training_prints and iteration % self.update_interval == 0:
                print('Epoch: ' + str(iteration))
                print('  Total_loss = ' + str(loss_i[0]) +
                      ';Delta_label = ' + str(delta_label_i))
                print('  Clustering_loss = ' + str(loss_i[1]) +
                      '; Decoder_loss = ' + str(loss_i[2]))

        return [np.array(y_pred), np.array(loss).transpose(), np.array(delta_label)]

    @staticmethod
    def simple_hardening(q, n=2.0):
        """
        hardening distribution P by polynomial functions with order of n
        """
        weight = q ** n / q.sum(0)
        return (weight.T / weight.sum(1)).T

    @staticmethod
    def smoothstep_hardening(q, n=3):
        """
        hardening distribution P by smoothsetp harderning with order of n
        """
        assert n % 2 == 1
        functions = {
            1: lambda x: x,
            3: lambda x: (-2*x + 3) * x**2,
            5: lambda x: ((6*x - 15)*x + 10) * x**3,
            7: lambda x: (((-20*x + 70)*x - 84)*x + 35) * x**4,
            9: lambda x: ((((70*x - 315)*x + 540)*x -420)*x + 126) * x**5}

        weight = functions[n](q) / q.sum(0)
        return (weight.T / weight.sum(1)).T
