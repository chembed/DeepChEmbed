"""
DeepChEmbed Models
"""
from dimreducer import DeepAutoEncoder
from cluster import KMeansLayer
from cluster import KMeans
from keras import Model
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

    def build_model(self, norm=True):
        """ """
        autoencoder = DeepAutoEncoder(self.autoencoder_dims)
        autoencoder.build_model(norm=norm)
        embeding = autoencoder.model.get_layer(name='embedding_layer').output
        clustering = KMeansLayer(self.n_clusters, alpha=self.alpha,
                                 name='clustering')(embeding)
        self.model = Model(inputs=autoencoder.model.input,
                           outputs=[clustering,autoencoder.model.output])

        return

    def train_model(self, data_train, norm_featrue=True, training_prints=True,
                    compiled=False, clustering_loss='kld', decoder_loss='mse',
                    clustering_loss_weight=0.5, optimizer='adam'):
        """ """
        if (not compiled):
            assert clustering_loss_weight <= 1 and clustering_loss_weight >= 0
            self.model.compile(loss={'clustering': clustering_loss,
                                     'decoder_output': decoder_loss},
                               loss_weights=[clustering_loss_weight,
                                             1 - clustering_loss_weight],
                               optimizer=optimizer)

        if(norm_featrue):
            data_train = normalize(data_train, axis=0, order=2)

        # initializing model by using sklean-Kmeans as guess
        kmeans_init = KMeans(n_clusters=2)
        kmeans_init.build_model()
        encoder  = Model(inputs=self.model.input,
                         outputs=self.model.get_layer(\
                         name='embedding_layer').output)
        kmeans_init.model.fit(encoder.predict(data_train))
        y_pred_last = kmeans_init.model.labels_
        self.model.get_layer(name='clustering').\
            set_weights([kmeans_init.model.cluster_centers_])


        loss = []
        delta_label = []

        for iteration in range(int(self.max_iteration)):

            if iteration % self.update_interval == 0:
                # updating centroid
                q, _ = self.model.predict(data_train)
                p = DCE.target_distribution(q)
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
    def target_distribution(q):
        """
        target distribution P which enhances the discrimination of soft label Q
        """
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
