import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
import numpy as np
import logging
import pickle
import pandas as pd

from utils.truncate import truncate
from utils.handler import Handler
from plotters.performance import Performance

class ExampleTrainer:
    def __init__(
            self, 
            model, 
            train_data, 
            val_data, 
            config
        ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.optimizer = tf.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_rsquare = tfa.metrics.RSquare(name='train_rsquare')
        self.train_MAPE = tf.keras.metrics.MeanAbsolutePercentageError(name='train_MAPE')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_rsquare = tfa.metrics.RSquare(name='val_rsquare')
        self.val_MAPE = tf.keras.metrics.MeanAbsolutePercentageError(name='val_MAPE')
        self.history = pd.DataFrame(
            columns=['epoch', 'train_loss', 'train_rsquare', 'val_loss', 'val_rsquare']
        )
        self.handler = Handler(self.config)
        # self.early_stopping_enable = self.config.EarlyStopping.enable
        # self.early_stopping_monitor = self.config.EarlyStopping.monitor
        # self.early_stopping_min_delta = self.config.EarlyStopping.min_delta

        self.test_data = self.handler.load('test_data')

        logging.debug(f'train_data length: {len(self.train_data)}')
        logging.debug(f'val_data length: {len(self.val_data)}')


    # def early_stopping(self, monitor_metric='train_loss'):
        
        


    # def end_train(self):
    #     # called at the end of training



    @tf.function
    def train_step(self, x_train, y_train):
        logging.debug(f'x_train shape: {np.shape(x_train)}')
        logging.debug(f'y_train shape: {np.shape(y_train)}')
        logging.debug(f'x_train type: {type(x_train)}')
        logging.debug(f'y_train type: {type(y_train)}')
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            train_predictions = self.model(x_train) # predictions/logits for this minibatch


            # Compute the loss value for this minibatch.
            tloss = self.model.loss_object(y_train, train_predictions)

        # Use the gradient tape to automatically retrieve the gradients of the
        # trainable variables with respect to the loss.
        # self.model.trainable_variables is a list of all the 
        # trainable variables given in a list of length equal to the number of
        # layers(?). It includes the weights(kernel?) and biases of each layer.
        # The apply_gradients function adjusts the values of the trainable 
        # variables to minimize the loss.
        logging.debug(f'self.config.features: {self.config.features}')
        logging.debug(f'length of self.config.features: {len(self.config.features)}')
        gradients = tape.gradient(tloss, self.model.trainable_variables)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(tloss)
        self.train_rsquare(y_train, train_predictions) # true and predicted values
        self.train_MAPE(y_train, train_predictions)

        # TODO: evaluate validation metrics per epoch, not per train step

        logging.debug(f'train loss: {self.train_loss.result()}')
        logging.debug(f'train rsquare: {self.train_rsquare.result()}')

        return self.train_loss.result(), self.train_rsquare.result(), self.train_MAPE.result()

    @tf.function
    def val_step(self, x_val, y_val):
        val_predictions = self.model(x_val, training=False)

        vloss = self.model.loss_object(y_val, val_predictions)

        self.val_loss.update_state(vloss)
        self.val_rsquare.update_state(y_val, val_predictions)



    def train_epoch(self):
        # use the length of train_data to determine number of batches
        # if the input dataset does not use batching, then there may be problems
        loop = tqdm(range(len(self.train_data)))
        train_losses = []
        train_rsquares = []
        train_MAPEs = []

        # logging.debug(f'self.val_data shape: {tf.shape(self.val_data)}')


        
        # # save self.val_data as a pickle file
        # with open('val_data.pkl', 'wb') as f:
        #     pickle.dump(self.val_data, f)
        

        # x -> features, y -> targets
        # x_val = self.val_data.map(lambda x, y: x)
        # y_val = self.val_data.map(lambda x, y: y)


        # print("SUCCESSFULLY UNSTACKED VAL DATA")
        # logging.debug(f'x_val element_spec: {x_val.element_spec}')
        # logging.debug(f'y_val element_spec: {y_val.element_spec}')
        # logging.debug(f'x_val type: {type(x_val)}')
        # logging.debug(f'y_val type: {type(y_val)}')
        # logging.debug(f'x_val first/next row: {next(x_val.as_numpy_iterator())}')
        # logging.debug(f'y_val first/next row: {next(y_val.as_numpy_iterator())}')

        self.train_loss.reset_state()
        self.train_rsquare.reset_state()
        self.train_MAPE.reset_state()
        self.val_loss.reset_state()
        self.val_rsquare.reset_state()
        self.val_MAPE.reset_state()

        
        logging.debug(f'trainable variables type: {type(self.model.trainable_variables)}')
        logging.debug(f'trainable variables length: {len(self.model.trainable_variables)}')
        # logging.debug(f'trainable variables shape: {np.shape(self.model.trainable_variables)}')

        logging.debug('STARTING TRAINING LOOP')
        logging.debug(f'self.train_data is of type: {type(self.train_data)}')
        # logging.debug('self.train_data[0] is of type: ', type(self.train_data[0]))
        # logging.debug('self.train_data[1] is of type: ', type(self.train_data[1]))

        # Iterate over the batches in the dataset.
        for _, (batch_x, batch_y) in zip(loop, self.train_data):
            train_loss, train_rsquare, train_MAPE = self.train_step(batch_x, batch_y)
            train_losses.append(train_loss)
            train_rsquares.append(train_rsquare)
            train_MAPEs.append(train_MAPE)
        mean_train_loss = np.mean(train_losses)
        mean_train_rsquare = np.mean(train_rsquares)
        mean_train_MAPE = np.mean(train_MAPEs)

        for x_batch_val, y_batch_val in self.val_data:
            self.val_step(x_batch_val, y_batch_val)
        
        logging.debug('FINISHED VAL EVALUATION')

        val_loss_result, val_rsquare_result = self.val_loss.result(), self.val_rsquare.result()

        logging.debug('Saving training info')
        self.history = pd.concat([
            self.history,
            pd.DataFrame([[
                self.epoch,
                mean_train_loss,
                mean_train_rsquare,
                val_loss_result,
                val_rsquare_result
            ]], 
            columns=['epoch', 'train_loss', 'train_rsquare', 'val_loss', 'val_rsquare']
            )
        ],
            ignore_index=True
        )

        model = self.model

        self.handler.save(self.history, 'history')
        self.handler.save(model, 'model')

        # normalization_factors = self.handler.load('normalization_factors')
        # performance = Performance(self.config, normalization_factors)
        # performance.plot(
        #     model, 
        #     self.test_data, 
        #     post_title=f'Epoch {self.epoch:04d}', 
        #     plot_diagnostics=False,
        #     avoid_plotting=[
        #         'target_vs_predicted',
        #         'total_residuals',
        #         'feature_vs_residuals',
        #         'cross_feature_importance',
        #     ],
        #     comparison_ylim=(0, 150),   # for pointing resolution comparison plot
        #     )

        return mean_train_loss, mean_train_rsquare, val_loss_result, val_rsquare_result


    def train(self):
        epochs = self.config.num_epochs
        for epoch in range(1, epochs+1):
            self.epoch = epoch
            mean_train_loss, mean_train_rsquare, mean_val_loss, mean_val_rsquare = self.train_epoch()
            logging.debug(f"mean_train_loss: {mean_train_loss}")
            logging.debug(f"mean_train_rsquare: {mean_train_rsquare}")
            logging.debug(f"mean_val_loss: {mean_val_loss}")
            logging.debug(f"mean_val_rsquare: {mean_val_rsquare}")

            # checkpoint_filepath = self.config.checkpoint_path

            # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            #     filepath=checkpoint_filepath,
            #     save_weights_only=False,
            #     monitor='loss',
            #     save_best_weights=True,
            #     mode='min',
            #     save_freq='epoch',
            # )

            # tf.keras.callbacks.EarlyStopping(
            #     monitor='train_loss',
            #     min_delta=0.001,
            #     patience=2,
            #     verbose=1,
            #     mode='min',
            #     restore_best_weights=True,
            # )

            # self.early_stopping(epoch,)

            if self.config.display_precision != None and type(self.config.display_precision) == int:
                mean_train_loss = truncate(mean_train_loss, self.config.display_precision)
                mean_train_rsquare = truncate(mean_train_rsquare, self.config.display_precision)
                mean_val_loss = truncate(mean_val_loss, self.config.display_precision)
                mean_val_rsquare = truncate(mean_val_rsquare, self.config.display_precision)

            elif self.config.display_precision != None and type(self.config.display_precision) != int:
                raise TypeError('display_precision must be an integer')

            # TODO: Add MAPE to template
            # TODO: Add configurable display precision, not just truncation
            template = "\033[1m Epoch: {}\033[0m | \
Train Loss: {:.4f}, Train R-Squared: {:.4f}\n \
Validation Loss: {:.4f}, Validation R-Squared: {:.4f}"
            if epoch % self.config.verbose_epochs == 0:
                # TODO: Using logger instead of print function
                print(template.format(epoch, mean_train_loss, mean_train_rsquare, mean_val_loss, mean_val_rsquare), '\n')


# need to add line to remove columns from every dataframe if it is in the 
#   ignore_train list from config