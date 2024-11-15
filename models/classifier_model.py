import tensorflow as tf
# import tensorflow_addons as tfa
import pandas as pd
import logging

class ClassifierModel(tf.keras.Model):
    # see https://www.tensorflow.org/api_docs/python/tf/keras/Model
    def __init__(self, config):
        super(ClassifierModel, self).__init__()
        # define layers here
        self.config = config

        self.valid_features = [f for f in self.config.features if f not in self.config.ignore_train]
        self.valid_targets = [t for t in self.config.targets if t not in self.config.ignore_train]

        if self.config.input_shape == 'auto':
            self.config.input_shape = (len(self.valid_features),)
        if self.config.output_shape == 'auto':
            self.config.output_shape = len(self.valid_targets)

        logging.debug(f'(ClassifierModel, __init__) input_shape: {self.config.input_shape}')
        logging.debug(f'(ClassifierModel, __init__) output_shape: {self.config.output_shape}')

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.config.input_shape,))
        self.d100_1 = tf.keras.layers.Dense(100, activation='relu')
        self.d100_2 = tf.keras.layers.Dense(100, activation='relu')
        self.d50 = tf.keras.layers.Dense(50, activation='relu')
        self.d20 = tf.keras.layers.Dense(20, activation='relu')
        self.out = tf.keras.layers.Dense(len(self.config.bin_edges), activation='linear')
        # self.drop2 = tf.keras.layers.Dropout(0.2)
        self.drop1_1 = tf.keras.layers.Dropout(0.1)
        self.drop1_2 = tf.keras.layers.Dropout(0.1)
        self.drop05 = tf.keras.layers.Dropout(0.05)

    # possible solution: add parameter to call function to specify input shape
    # and then use that to define input layer shape but make sure input layer 
    # defined in __init__ can accept any input shape

    # treat x input into function as input tensor
    def call(self, x):
        # if type(x) == pd.DataFrame or type(x) == pd.Series:
        #     x = tf.convert_to_tensor(x)

        x = self.input_layer(x)
        x = self.d100_1(x)
        x = self.drop1_2(x)
        x = self.d100_2(x)
        x = self.drop1_2(x)
        x = self.d50(x)
        x = self.drop05(x)
        x = self.d20(x)
        return self.out(x)

    # TODO: allow final layer to have different output shapes for n-dimensional
    # predictions

    @staticmethod
    def loss_object(prediction, label):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        return loss_object(prediction, label)

    # def optimizer_object(prediction, )