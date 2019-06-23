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
    HARDENING_FUNCS = {
        1: lambda x: x,
        3: lambda x: (-2*x + 3) * x**2,
        5: lambda x: ((6*x - 15)*x + 10) * x**3,
        7: lambda x: (((-20*x + 70)*x - 84)*x + 35) * x**4,
        9: lambda x: ((((70*x - 315)*x + 540)*x -420)*x + 126) * x**5}

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

    def train_model(self, data_train,
                    labels_train=None, data_test=None, labels_test=None,
                    verbose=1,
                    compiled=False, clustering_loss='kld',
                    decoder_loss='mse',clustering_loss_weight=0.5,
                    hardening_order=1, hardening_strength=2.0,
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

        # Prepare training: p disctribution methods
        if labels_train is None:
            # Unsupervised Learning
            assert hardening_order in DCE.HARDENING_FUNCS.keys()
            assert hardening_strength >= 1.0
            h_func = DCE.HARDENING_FUNCS[hardening_order]
        else:
            # Supervised Learning
            assert len(labels_train) == len(data_train)
            assert len(np.unique(labels_train)) == self.n_clusters
            p = np.zeros(shape=(len(labels_train), self.n_clusters))
            for i in range(len(labels_train)):
                p[i][labels_train[i]] = 1.0

            if data_test is not None:
                assert len(labels_test) == len(data_test)
                assert len(np.unique(labels_test)) == self.n_clusters
                p_test = np.zeros(shape=(len(labels_test), self.n_clusters))
                for i in range(len(labels_test)):
                    p_test[i][labels_test[i]] = 1.0

                validation_loss = []

        # training start:
        loss = []

        for iteration in range(int(self.max_iteration)):

            if iteration % self.update_interval == 0:
                # updating p for unsupervised learning process
                q, _ = self.model.predict(data_train)
                if labels_train is None:
                    p = DCE.hardening(q, h_func, hardening_strength)

                # get label change i
                y_pred = q.argmax(1)
                delta_label_i = np.sum(y_pred != y_pred_last).\
                    astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred

                # exam convergence
                if iteration > 0 and delta_label_i < self.clustering_tol:
                    print(str(delta_label_i) +' < ' + str(self.clustering_tol))
                    print('Reached tolerance threshold. Stopping training.')
                    break

            loss.append(self.model.train_on_batch(x=data_train,
                                                  y=[p,data_train]))
            if data_test is not None:
                validation_loss.append(self.model.test_on_batch(
                    x=data_test, y=[p_test,data_test]))

            if verbose > 0 and iteration % self.update_interval == 0:
                print('Epoch: ' + str(iteration))
                if verbose == 1:
                    print('  Total_loss = ' + str(loss[iteration][0]) +
                          ';Delta_label = ' + str(delta_label_i))
                    print('  Clustering_loss = ' + str(loss[iteration][1]) +
                          '; Decoder_loss = ' + str(loss[iteration][2]))

        if iteration == self.max_iteration - 1:
            print('Reached maximum iteration. Stopping training.')

        if data_test is None:
            return np.array(loss).T
        else:
            return [np.array(loss).T, np.array(validation_loss).T]

    @staticmethod
    def hardening(q, h_func, stength):
        """
        hardening distribution P by smoothsetp harderning with order of n
        """
        q = h_func(q)
        weight =  q ** stength / q.sum(0)
        return (weight.T / weight.sum(1)).T
