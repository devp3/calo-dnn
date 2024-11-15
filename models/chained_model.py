import tensorflow as tf
# import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import logging

class ChainedModel(tf.keras.Model):
    # see https://www.tensorflow.org/api_docs/python/tf/keras/Model
    def __init__(self, config, c0, r0, r1, r2, r3, r4):
        super(ChainedModel, self).__init__()
        # define layers here
        self.config = config

        self.valid_features = [f for f in self.config.features if f not in self.config.ignore_train]
        self.valid_targets = [t for t in self.config.targets if t not in self.config.ignore_train]

        if self.config.input_shape == 'auto':
            self.config.input_shape = (len(self.valid_features),)
        if self.config.output_shape == 'auto':
            self.config.output_shape = len(self.valid_targets)

        logging.debug(f'(ChainedModel, __init__) input_shape: {self.config.input_shape}')
        logging.debug(f'(ChainedModel, __init__) output_shape: {self.config.output_shape}')

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.config.input_shape,))
        self.d100_1 = tf.keras.layers.Dense(100, activation='relu')
        self.d100_2 = tf.keras.layers.Dense(100, activation='relu')
        self.d50 = tf.keras.layers.Dense(50, activation='relu')
        self.d20 = tf.keras.layers.Dense(20, activation='relu')
        self.out = tf.keras.layers.Dense(self.config.output_shape, activation='linear')
        # self.drop2 = tf.keras.layers.Dropout(0.2)
        self.drop1_1 = tf.keras.layers.Dropout(0.1)
        self.drop1_2 = tf.keras.layers.Dropout(0.1)
        self.drop05 = tf.keras.layers.Dropout(0.05)

        self.classifier_out = tf.keras.layers.Dense(5, activation='sigmoid')

        self.classifier0 = c0       # classifier model
        self.regression0 = r0       # 0th regression model
        self.regression1 = r1
        self.regression2 = r2
        self.regression3 = r3
        self.regression4 = r4
       


    # possible solution: add parameter to call function to specify input shape
    # and then use that to define input layer shape but make sure input layer 
    # defined in __init__ can accept any input shape

    # treat x input into function as input tensor
    def call(self, x):

        if tf.math.argmax(self.classifier0(x), axis=1) == 0:
            return self.regression0(x)
        
        elif tf.math.argmax(self.classifier0(x), axis=1) == 1:
            return self.regression1(x)
        
        elif tf.math.argmax(self.classifier0(x), axis=1) == 2:
            return self.regression2(x)
        
        elif tf.math.argmax(self.classifier0(x), axis=1) == 3:
            return self.regression3(x)
        
        elif tf.math.argmax(self.classifier0(x), axis=1) == 4:
            return self.regression4(x)
        
        else:
            logging.error("Error in ChainedModel call function")

    # TODO: allow final layer to have different output shapes for n-dimensional
    # predictions

    @staticmethod
    def loss_object(prediction, label):
        loss_object = tf.keras.losses.MeanSquaredError()
        return loss_object(prediction, label)

    # def optimizer_object(prediction, )