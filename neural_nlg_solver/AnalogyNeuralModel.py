#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.layers import Dense, Dropout, Flatten, Input, Reshape
from keras.models import Model
import math

###############################################################################

__author__ = 'KAVEETA Vivatchai <vivatchai@fuji.waseda.jp, goodytong@gmail.com>'

__date__, __version__ = '25/01/2017', '0.1'  # First version

__description__ = 'Class for constructing, save, load neural network model'


###############################################################################

class FullyConnectedModel:
    """
    Fully-connected neural network object
    """

    model = None

    def save(self, save_path):
        """
        Save model to file
        :param save_path: path to save
        """

        self.model.save_weights(save_path)

    def load(self, load_path):
        """
        Load model from file
        :param load_path: path to load from
        """

        self.model.load_weights(load_path)

    def config(self, length, layers, optimizer, loss, activation, compile=True, verbose=False):
        """
        Build neural network model from input configuration
        :param compile: compile model after config
        :param optimizer: optimizer
        :param loss: loss function
        :param activation: activation function
        :param verbose: verbose mode
        :param length: length of strings in analogical equations
        :param layers: layer(s) configurations
        :return: network model object
        """

        # Input layer
        input_layer = Input(shape=(length[0], length[1] + length[2]), name='input')
        input_num = length[0] * (length[1] + length[2])
        output_num = length[3] * (length[1] + length[2])

        # flatten layer
        current_input = Flatten(name='flatten')(input_layer)

        # Dense layer(s)
        dense_id = 1
        dropout_id = 1
        hidden_num = math.floor((input_num + output_num) / 2)

        for layer in layers:
            current_input = Dense(hidden_num, name='dense_' + str(dense_id),
                                  activation=layer['activation'].lower())(current_input)
            dense_id += 1

            if layer['dropout'] > 0:
                current_input = Dropout(layer['dropout'], name='dropout_' + str(dropout_id))(current_input)
                dropout_id += 1

        # Last layer
        current_input = Dense(output_num, name='dense_' + str(dense_id),
                              activation=activation.lower())(current_input)

        # Output layer (Reshape)
        output_layer = Reshape((length[3], length[1] + length[2]), input_shape=(output_num,),
                               name='output')(current_input)

        # Create model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        if compile:
            self.model.compile(optimizer.lower(), loss.lower())

        # Print summary to console
        if verbose:
            self.model.summary()

        return self.model

    def train(self, train_src, train_trg, **kwargs):
        """
        Forward training to keras fit function
        :param train_src: training sources
        :param train_trg: training targets
        :param kwargs: arguments
        :return: training results (dict)
        """
        return self.model.fit(train_src, train_trg, **kwargs)

    def predict(self, test_src):
        """
        Predict the target from source matrices
        :param test_src: experiments source input
        :return: predicted experiments target input
        """

        return self.model.predict(test_src)

    @property
    def weights(self):
        return self.model.get_weights()

    @property
    def layers(self):
        return self.model.layers