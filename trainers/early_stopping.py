import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras

class EarlyStopping(keras.callbacks.Callback):

    def __init__(
        self,
        monitor='loss',
        min_delta=0,
        patience=0,

    ):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at. 
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, history):
        current_loss = history['loss'].iloc[-1]
        if np.less(current_loss, self.best):
            self.best = current_loss
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                logging.info(f'Epoch {epoch}: early stopping')
                logging.info(f'Best loss: {self.best}')
                logging.info(f'Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)
        
    def on_train_end(self):
        if self.stopped_epoch > 0:
            logging.info(f'Epoch {self.stopped_epoch + 1}: early stopping')
            