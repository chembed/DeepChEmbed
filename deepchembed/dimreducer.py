"""
Wrapper Module to integrate the models to reduce the intial feature dimensions
"""

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import normalize
from keras.layers.normalization import BatchNormalization


class DimReducer(ABC):
    """
    Base class of DimReducer
    """

    def __init__(self, input_size, output_size):
        self.input_size  = input_size
        self.output_size = output_size
        self.model = None

        return

class DeepAutoEncoder(DimReducer):
    """
    Inplementation of deep autoencoder

    Adapted from
        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded
        Clustering with Local Structure Preservation. IJCAI 2017.


    """

    def __init__(self, dims, act='relu'):
        """ Constructor of DeepAutoEncoder """
        assert len(dims) >= 2

        DimReducer.__init__(self, dims[0], dims[-1])

        self.set_dims(dims)

        self.act      = act

        return

    def set_dims(self, dims):
        """ """
        self.dims     = dims
        self.n_stacks = len(dims)
        self.n_layers = len(dims)*2 - 1
        return

    def build_model(self, norm=True):
        """ build an autoencoder using keras and tensorflow"""

        # input
        X_input = Input(shape=(self.input_size,), name='input')

        # normalization layer to accelarte convergence
        if(norm):
            X_encoder = BatchNormalization()(X_input)
        else:
            X_encoder = X_input

        # encoder
        for i in range(self.n_stacks - 2):
            X_encoder = Dense(self.dims[i + 1], activation=self.act,
                              name='encoder_%d' % i)(X_encoder)
        # embeding layer
        X_encoder  = Dense(self.output_size, activation=self.act,
                           name='embedding_layer')(X_encoder)
        X_decoder = X_encoder

        # decoder
        for i in range(self.n_stacks - 2, 0, -1):
            X_decoder = Dense(self.dims[i], activation=self.act,
                              name='decoder_%d' % i)(X_decoder)

        X_output = Dense(self.input_size, activation=self.act,
                          name='decoder_output')(X_decoder)

        self.model = Model(inputs=X_input, outputs=X_output)

        return

    def train_model(self, data_train, norm_featrue=True, compiled=False,
                    optimizer='adam',loss='mse',epochs=150):
        assert isinstance(self.model, Model)

        if(not compiled):
            self.model.compile(optimizer=optimizer, loss=loss)

        if(norm_featrue):
            data_train = normalize(data_train, axis=0, order=2)

        history = self.model.fit(data_train, data_train, epochs=epochs,
                                 batch_size=256,shuffle=True)
        return history
