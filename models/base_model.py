import tensorflow as tf
# import tensorflow_addons as tfa
import pandas as pd
import logging

class BaseModel(tf.keras.Model):
    # see https://www.tensorflow.org/api_docs/python/tf/keras/Model
    def __init__(self, config):
        super(BaseModel, self).__init__()
        # define layers here
        self.config = config

        self.valid_features = [f for f in self.config['features'] if f not in self.config['ignore_train']]
        self.valid_targets = [t for t in self.config['targets'] if t not in self.config['ignore_train']]
        if self.config['input_shape'] == 'auto':
            self.config['input_shape'] = (len(self.valid_features),)
        if self.config['output_shape'] == 'auto':
            self.config['output_shape'] = len(self.valid_targets)
        logging.debug(f"(BaseModel, __init__) input_shape: {self.config['input_shape']}")
        logging.debug(f"(BaseModel, __init__) output_shape: {self.config['output_shape']}")

        self._print_info()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=self.config['input_shape'])
        self.out = tf.keras.layers.Dense(self.config['output_shape'], activation='linear')

        self.d100_1 = tf.keras.layers.Dense(100, activation='relu')
        self.d100_2 = tf.keras.layers.Dense(100, activation='relu')
        # self.d100_3 = tf.keras.layers.Dense(100, activation='relu')
        # self.d100_4 = tf.keras.layers.Dense(100, activation='relu')
        # self.d100_5 = tf.keras.layers.Dense(100, activation='relu')
        # self.d100_6 = tf.keras.layers.Dense(100, activation='relu')

        self.d50_1 = tf.keras.layers.Dense(50, activation='relu')
        # self.d50_2 = tf.keras.layers.Dense(50, activation='relu')
        # self.d50_3 = tf.keras.layers.Dense(50, activation='relu')
        
        self.d20_1 = tf.keras.layers.Dense(20, activation='relu')
        # self.d20_2 = tf.keras.layers.Dense(20, activation='relu')
        # self.d20_3 = tf.keras.layers.Dense(20, activation='relu')

        # self.d5_1 = tf.keras.layers.Dense(5, activation='relu')
        # self.d5_2 = tf.keras.layers.Dense(5, activation='relu')

        self.drop1_1 = tf.keras.layers.Dropout(0.1)
        self.drop1_2 = tf.keras.layers.Dropout(0.1)
        # self.drop1_3 = tf.keras.layers.Dropout(0.1)
        # self.drop1_4 = tf.keras.layers.Dropout(0.1)
        # self.drop1_5 = tf.keras.layers.Dropout(0.1)
        # self.drop1_6 = tf.keras.layers.Dropout(0.1)

        self.drop05_1 = tf.keras.layers.Dropout(0.05)
        # self.drop05_2 = tf.keras.layers.Dropout(0.05)
        # self.drop05_3 = tf.keras.layers.Dropout(0.05)


        # # self.drop2 = tf.keras.layers.Dropout(0.2)
        # self.drop05 = tf.keras.layers.Dropout(0.05)
        # self.d1 = tf.keras.layers.Dense(1, activation='linear')
        # self.drop1_3 = tf.keras.layers.Dropout(0.1)
        # self.drop1_4 = tf.keras.layers.Dropout(0.1)
        # self.d20_1 = tf.keras.layers.Dense(20, activation='relu')

        # self.layer_normalization_1 = tf.keras.layers.LayerNormalization(axis=1)
        # self.layer_normalization_2 = tf.keras.layers.LayerNormalization(axis=1)

        # self.batch_normalization_1 = tf.keras.layers.BatchNormalization()
        # self.batch_normalization_2 = tf.keras.layers.BatchNormalization()

    # possible solution: add parameter to call function to specify input shape
    # and then use that to define input layer shape but make sure input layer 
    # defined in __init__ can accept any input shape

    def _print_info(self):
        logging.info('\033[1m Initializing Base Model\033[0m')
        logging.info(f"Input Shape: {self.config['input_shape']}")
        logging.info(f"Output Shape: {self.config['output_shape']}")
    

    # treat x input into function as input tensor
    def call(self, x):
        # if type(x) == pd.DataFrame or type(x) == pd.Series:
        #     x = tf.convert_to_tensor(x)

        x = self.input_layer(x)
        # x = self.batch_normalization_1(x)
        x = self.d100_1(x)
        x = self.drop1_1(x)
        x = self.d100_2(x)
        x = self.drop1_2(x)
        # x = self.batch_normalization_2(x)
        x = self.d50_1(x)
        x = self.drop05_1(x)
        x = self.d20_1(x)
        return self.out(x)

        # x1 = self.input_layer(x)
        # x1 = self.d100_1(x1)
        # x1 = self.drop1_2(x1)   
        # x1 = self.d100_2(x1)
        # x1 = self.drop1_2(x1)
        # x1 = self.d50(x1)
        # x1 = self.drop05(x1)
        # x1 = self.d20(x1)
        # x1 = self.d1(x1)
        # x2 = tf.keras.layers.Concatenate(axis=1)([x, x1])
        # x2 = self.d100_3(x2)
        # x2 = self.drop1_3(x2)
        # x2 = self.d100_4(x2)
        # x2 = self.drop1_4(x2)
        # x2 = self.d50_1(x2)
        # x2 = self.drop05_1(x2)
        # x2 = self.d20_1(x2)
        # return self.out(x2)




        # x1 = self.input_layer(x)

        # x2 = self.d100_1(x1)
        # x2 = self.drop1_1(x2)
        # x2 = self.d100_2(x2)
        # x2 = self.drop1_2(x2)
        # x2 = self.d50_1(x2)
        # x2 = self.drop05_1(x2)
        # x2 = self.d20_1(x2)
        # x2 = self.d5_1(x2)

        # x3 = self.d100_3(x1)
        # x3 = self.drop1_3(x3)
        # x3 = self.d100_4(x3)
        # x3 = self.drop1_4(x3)
        # x3 = self.d50_2(x3)
        # x3 = self.drop05_2(x3)
        # x3 = self.d20_2(x3)
        # x3 = self.d5_2(x3)

        # # may need a normalization or other processor before concat
        # x4 = tf.keras.layers.Concatenate(axis=1)([x1, x2, x3])
        # # x4 = tf.keras.layers.LayerNormalization(axis=1)(x4)
        # x4 = self.d100_5(x4)
        # x4 = self.drop1_5(x4)
        # x4 = self.d100_6(x4)
        # x4 = self.drop1_6(x4)
        # x4 = self.d50_3(x4)
        # x4 = self.drop05_3(x4)
        # x4 = self.d20_3(x4)

        # return self.out(x4)


    # TODO: allow final layer to have different output shapes for n-dimensional
    # predictions

    @staticmethod
    def loss_object(prediction, label):
        # loss_object = tf.keras.losses.MeanSquaredError()
        # loss_object = tf.keras.losses.MeanSquaredLogarithmicError()
        loss_object = tf.keras.losses.MeanAbsoluteError()
        # loss_object = tf.keras.losses.MeanSquaredLogarithmicError()
        return loss_object(prediction, label)
