import tensorflow as tf
import uproot
import pandas as pd
import numpy as np
import logging
import os, sys
import matplotlib.pyplot as plt

from functions.precompute import *
from utils.reorder import reorder
from functions.normalize import linear
from utils.handler import Handler
import cuts.cut as cut

class DatasetGenerator:
    def __init__(self, config, mode='tf', normed=True, checknormed=True):
        # data, info = tfds.load("mnist", with_info=True)
        self.config = config
        self.features = config.features
        self.targets = config.targets # should be list of strings
        data_path = config.data_path
        self.mode = mode
        self.normed = normed
        self.checknormed = checknormed
        # I can't define train, val, and test data here yet since 
        # self.train_data = train_data
        # self.val_data = val_data
        # self.test_data = test_data

        # We might want to assert this, but it must be later, after the loading
        # is complete.
        # assert isinstance(self.train_data, tf.data.Dataset)

        self.handler = Handler(config)
        self._set_bin_names()

    

    def _set_bin_names(self):
        """Use the indices of self.config.bin_edges to create a list of bin
        names, such that self.bin_names = ['bin0', 'bin1', 'bin2', ...]
        """
        self.bin_names = [f'bin{i}' for i in range(len(self.config.bin_edges))]



    def load(self):
        # self.handler = Handler(self.config)

        config = self.config
        raw, _ = self.root_loader() # not sure if this scheme is valid
        raw = self.call_precompute(raw, config.features + config.targets)
        raw = raw.sample(frac=config.load_fraction).dropna()

        raw = cut.apply(raw, name=self.config.cut_name)

        # 1) add extra column to indicate which bin the event falls into
        # 2) tell normalizer to ignore columns that start with 'bin'
        # 3) normalize
        # 4) inside splitter, 

        # 1) write one-hot function OHE: TV_z -> binary vector of dim len(target_bins)

        raw = self.add_ohe(raw) # adds more columns for one-hot encoding

        self.handler.save(raw, 'raw_data')
        
        # self.handler.save()

        # regression_data = [r0_data, ...] = [(r0_train, r0_test, r0_val), ...]
        # r0_train = (r0_train_features, r0_train_targets)
        regression_data = []

        if self.normed:
            raw_normed, factors = linear.norm(raw, check=self.checknormed, ignore=config.bin_names)
            # linear.save_factors(config.active_data_path + "/factors.csv", factors)
            logging.info("Normalized data.")
            
            self.handler.save(factors, 'normalization_factors')
            self.handler.save(raw_normed, 'normalized_data')

            r_raw_list = self.divide_regression(raw_normed) # list of raw regression dataframes
            c0_raw = self.divide_classification(raw_normed) # raw classification data

            for r in r_raw_list:
                r_train, r_test, r_val = self.splitter(r)
                regression_data.append((r_train, r_test, r_val))

            c0_train, c0_test, c0_val = self.splitter(c0_raw)

            # train_data, test_data, val_data = self.splitter(raw_normed)                

        else:
            logging.info("Data not normalized.")
            r_raw_list = self.divide_regression(raw) # list of raw regression dataframes
            c0_raw = self.divide_classification(raw) # raw classification data

            for r in r_raw_list:
                r_train, r_test, r_val = self.splitter(r)
                regression_data.append((r_train, r_test, r_val))

            c0_train, c0_test, c0_val = self.splitter(c0_raw)

        classification_data = (c0_train, c0_test, c0_val)

        self.handler.pickle_saver(
            classification_data, 
            os.path.join(self.config.base_path, self.config.active_data_path, "/classification_data.pkl")
        )

        self.handler.pickle_saver(
            regression_data, 
            os.path.join(self.config.base_path, self.config.active_data_path, "/regression_data.pkl")
        )

        # self.handler.save(train_data, 'train_data')
        # self.handler.save(test_data, 'test_data')
        
        for i, r in enumerate(regression_data):
            r_train, r_test, r_val = self.preprocessor(r) # r = (r_train, r_test, r_val)
            regression_data[i] = (r_train, r_test, r_val)

        c0_train, c0_test, c0_val = self.preprocessor((c0_train, c0_test, c0_val))
       

        # self.handler.save(train_data, 'train_data')
        # self.handler.save(test_data, 'test_data')

        # test_data.save(self.config.active_data_path + "/test_data")

        # (x_train, y_train), (x_test, y_test)
        # return (train_targets, test_targets), (train_features, test_features)

        logging.info("Data loaded.")
        
        return classification_data, regression_data
        # return train_data, test_data, val_data


    def root_loader(self):
        """_summary_

        Returns:
            features_raw: _description_
            features_pending: _description_
        """
        # return raw
        config = self.config
        features = self.features
        targets = self.targets

        variables = features + targets
        pending = [None]    # initialize pending list

        raw = pd.DataFrame()

        root_files = config.data_path    # might be single string or list of strings
        if isinstance(root_files, str):
            root_files = [root_files]

        for rf in root_files:
            # rf is the root file we are currently trying to open
            file_df = pd.DataFrame()
            with uproot.open(rf) as file:
                if config.superbranch:
                    file = file[config.superbranch]

                branches = file.keys()
                
                logging.debug('PRINTING FEATURES AND TARGETS')
                for branch in variables:
                    # for every branch in the variables we want to include
                    if branch in branches:
                        # if the branch is in the root file, add it to the file dataframe
                        # include only items that are in features+targets and in branches 
                        logging.debug(branch)

                        file_df[branch] = file[branch].array(library='np')

                        if isinstance(file_df[branch].iloc[0], bool) or isinstance(file_df[branch].iloc[0], np.bool_):
                            # if the branch is a boolean, convert it to an int
                            file_df[branch] = file_df[branch].apply(lambda x: int(x))

                pending = list(set(variables) - set(branches))
            raw = pd.concat([raw, file_df], ignore_index=True)

        logging.debug('PRINTING VARIABLES THAT NEED TO BE COMPUTED')
        for item in pending:
            logging.debug(item)

        return raw, pending


    def call_precompute(self, raw:pd.DataFrame, variables:list):
        """_summary_
        
        Args:
            raw (pd.DataFrame): input dataframe from raw root file. 
            variables (list): ordered list of variables to be included in final dataframe. 

        Returns:
            pd.DataFrame: raw dataframe with all variables included and ordered as specified in the config file.

        Raises:
            ValueError: _description_
            ValueError: _description_
        
        
        """

        # First, check if all the variables to be computed have a script 
        # already in the precompute directory. If not, raise an error.
        # Then, find a subset of the precompute variables that are composed
        # directly of the variables loaded from the initial root file. 
        # Then, remove those variables from the total list, add the 
        # newly computed variables to the list of available variables, and 
        # check again to see if any of the remaining uncomputed variables can be 
        # computed from the new set of available variables. 
        # Include a ticker in this loop that is reset by a successful 
        # computation of at least one variable and raised by one if no variable 
        # can be computed. If the ticker returns to the top of the loop in the 
        # state of 1 and there are variables in the "needing to be computed"
        # list, then raise an error (error: some variables could not be 
        # computed). 
        # TODO: add a spot in the config file of variables the network should
        # not be allowed to train on. This way, we can load additional variables'
        # needed for computation and plotting, but not have the network train on
        # them. This will require a change in how the network interprets how
        # many features it should to train on (size of input layer). 
        # # for training = # loaded -  # not allowed to train on

        # The final step should be to reorder the columns of the feature and 
        # target dataframes to match the order specified in the config file.

        # variables is the total list of variables for final dataframe
        # it is from the config file

        # Create a list of variables that are not in the raw dataframe but are
        # in the variables list. These are the pending variables that need to be 
        # computed.
        pending = []
        for item in variables:
            if item not in raw.columns:
                pending.append(item)

        # assemble the list of functions available in the precompute directory
        function_list = os.listdir('functions/precompute')
        function_list = [item for item in function_list if item.endswith('.py') and item != '__init__.py']
        function_list = [item[:-3] for item in function_list]

        # check that every function in the pending list is found in the function list
        # i.e. check that pending is a subset of function_list
        if not all(item in function_list for item in pending):
            error_message = 'Not all pending variables found in functions list.'
            error_message += 'Check variables in config or add precompute script.'
            raise ValueError(error_message)

        pending_dependencies = {}
        for item in pending:
            exec(f'pending_dependencies[item] = {item}.dependencies()')\
            # check that pending dependencies are not None 
            #   (i.e. did something import correctly)
            assert(pending_dependencies[item] is not None)

        # label each pending variable as either True (can be computed from raw)
        # or False (cannot be computed from raw)
        
        while len(pending) > 0:
            try:
                available = {}
                for item in pending:
                    if all(item in raw.keys() for item in pending_dependencies[item]):
                        available[item] = True
                    else:
                        available[item] = False

                for item in available:
                    if available[item]:
                        exec(f'raw = {item}.compute(raw)')
                        pending.remove(item)
                        available.pop(item)
                else:
                    error_message = 'Not all pending variables could be computed.'
                    error_message += 'Check dependencies of pending variables.'
                    raise ValueError(error_message)
            except: 
                pass

        assert len(pending) == 0, 'Not all pending variables could be computed.'
        assert set(raw.keys()) == set(variables), 'Not all variables were computed.'

        logging.debug('PRINTING RAW DATAFRAME KEYS')
        logging.debug(raw.keys())

        raw = reorder(raw, variables)

        return raw



    def splitter(self, data: pd.DataFrame):
        """Splits raw dataframe into train, test, and validation sets and 
        separates features and targets into separate dataframes.

        Args:
            data (pd.DataFrame): raw dataframe

        Returns:
            train_data (tuple): tuple of train targets and features dataframes
            test_data (tuple): tuple of test targets and features dataframes
            val_data (tuple): tuple of validation targets and features dataframes
        """        


        # need to split total data into train, test, and validation sets
        # need to separate features and targets
        # splitter is the 

        feature_variables = self.config.features
        target_variables = self.config.targets
        test_split = self.config.test_split
        validation_split = 1 - self.config.validation_split

        logging.debug(f'test_split = {test_split}')
        logging.debug(f'validation_split = {validation_split}')

        valtrain_dataset = data.sample(frac=test_split, random_state=0)
        test_dataset = data.drop(valtrain_dataset.index)

        train_dataset = valtrain_dataset.sample(frac=validation_split, random_state=0)
        val_dataset = valtrain_dataset.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()
        val_features = val_dataset.copy()

        train_targets = train_features[target_variables].copy()
        test_targets = test_features[target_variables].copy()
        val_targets = val_features[target_variables].copy()

        train_features = train_features.drop(target_variables, axis=1)
        test_features = test_features.drop(target_variables, axis=1)
        val_features = val_features.drop(target_variables, axis=1)

        logging.debug('PRINTING DATAFRAME SHAPES -- IN SPLITTER FUNCTION')
        logging.debug(f'train_targets.shape: {train_targets.shape}')
        logging.debug(f'train_features.shape: {train_features.shape}')
        logging.debug(f'test_targets.shape: {test_targets.shape}')
        logging.debug(f'test_features.shape: {test_features.shape}')
        logging.debug(f'val_targets.shape: {val_targets.shape}')
        logging.debug(f'val_features.shape: {val_features.shape}')

        # TODO: chance order of tuple from (targets, features) to (features, targets)
        #   to match the style of (x,y) since x->features and y->targets usually.
        #   This will require changing the loading behavior in numerous spots
        #   and updating documentation. 
        train_data, test_data, val_data = (train_features, train_targets), (test_features, test_targets), (val_features, val_targets)

        # return train_targets, test_targets, train_features, test_features
        return train_data, test_data, val_data


    def drop_ignore(self, data: pd.DataFrame):
        # drop any variables in the ignore list
        ignore = self.config.ignore_train
        data = data.drop(ignore, axis=1, errors='ignore')
        return data


    def preprocess(self, *data: tuple, shuffle=True):
        # expects a tuple of (features, targets) dataframes

        # batch will divide dataset into subsets of length batch_size
        # drop_remainder will drop the last batch if it is smaller than batch_size
        # buffer_size is the size of the buffer for the suffling algorithm
        #   buffer_size must be greater than or equal to the dataset size for 
        #   perfect suffling, but it should be fine if it is smaller

        # train_data, test_data, val_data

        # x_data = features, y_data = targets
        self.shuffle = shuffle

        for (x_data, y_data) in data:
            x_data = self.drop_ignore(x_data)
            y_data = self.drop_ignore(y_data)

            if self.mode == 'tf':
                xy_data = tf.data.Dataset.from_tensor_slices((x_data.values, y_data.values))

                if self.shuffle:
                    xy_data = xy_data.shuffle(buffer_size=1000)

                xy_data = xy_data.batch(self.config.batch_size, drop_remainder=False)

            if self.mode == 'pd':
                xy_data = (x_data, y_data)

            yield xy_data

        # train_targets, train_features = train_data
        # test_targets, test_features = test_data
        # val_targets, val_features = val_data
        
        # train_targets = self.drop_ignore(train_targets)
        # test_targets = self.drop_ignore(test_targets)
        # val_targets = self.drop_ignore(val_targets)
        # val_features = self.drop_ignore(val_features)
        # test_features = self.drop_ignore(test_features)
        # train_features = self.drop_ignore(train_features)

        # train_data = tf.data.Dataset.from_tensor_slices((train_features.values, train_targets.values))
        # train_data = train_data.shuffle(buffer_size=1000).batch(self.config.batch_size)

        # test_data = tf.data.Dataset.from_tensor_slices((test_features.values, test_targets.values))
        # test_data = test_data.shuffle(buffer_size=1000).batch(self.config.batch_size) # commenting out since we don't need to batch test data

        # val_data = tf.data.Dataset.from_tensor_slices((val_features.values, val_targets.values))
        # val_data = val_data.shuffle(buffer_size=1000).batch(self.config.batch_size) # commenting out since we don't need to batch val data

        # return train_data, test_data, val_data


    def preprocess_pd(self, *data):
        
        train_features, train_targets = train_data
        test_features, test_targets = test_data
        val_features, val_targets = val_data

        # used in pandas mode when we want the dataloader to return pandas dataframes
        train_data = pd.concat([train_features, train_targets], axis=1)
        test_data = pd.concat([test_features, test_targets], axis=1)
        val_data = pd.concat([val_features, val_targets], axis=1)
        return train_data, test_data, val_data
    


    def preprocessor(self, train_test_val: tuple, mode='tf'):
        """Runs the (train, test, val) tuple through the preprocessing pipeline
        in the data loader class. 

        Args:
            train_test_val (tuple): (train_data, test_data, val_data) 
            mode (str, optional): Output type ('tf'|'pd'). Defaults to 'tf' (tf.Tensor).

        Returns:
            tuple: train_data, test_data, val_data
        """

        train_data, test_data, val_data = train_test_val

        if mode == 'tf':
            train_data, test_data, val_data = self.preprocess(train_data, test_data, val_data)

        elif mode == 'pd':
            train_data, test_data, val_data = self.preprocess_pd(train_data, test_data, val_data)

        else:
            logging.error("Invalid mode: ", mode)
            logging.error("Mode must be 'tf' or 'pd'.")
            exit(1)
    
        return train_data, test_data, val_data
    

    def add_ohe(self, raw: pd.DataFrame):
        # accept the raw dataframe
        # returns the dataframe with additional columns for target one-hot encoding 

        trainable_targets = [t for t in self.config.targets if t not in self.config.ignore_train]
        if len(trainable_targets) > 1:
            logging.error('Only one target variable can be trained at this time.')
            logging.debug(f'trainable_targets: {trainable_targets}')
            sys.exit(1)

        if len(self.config.bin_names) != len(self.config.bin_values):
            logging.error('The number of bin names must match the number of bin values.')
            logging.debug(f'bin_names: {self.config.bin_names}')
            logging.debug(f'bin_values: {self.config.bin_values}')
            sys.exit(1)


        # TODO: make flexible OHE binning function for any number of bins
        raw[self.config.bin_names[0]] = raw[trainable_targets[0]].apply(lambda y: 1 if np.abs(y) <= self.config.bin_edges[0] else 0)
        raw[self.config.bin_names[1]] = raw[trainable_targets[0]].apply(lambda y: 1 if np.abs(y) > self.config.bin_edges[0] and np.abs(y) <= self.config.bin_edges[1] else 0)
        raw[self.config.bin_names[2]] = raw[trainable_targets[0]].apply(lambda y: 1 if np.abs(y) > self.config.bin_edges[1] and np.abs(y) <= self.config.bin_edges[2] else 0)
        raw[self.config.bin_names[3]] = raw[trainable_targets[0]].apply(lambda y: 1 if np.abs(y) > self.config.bin_edges[2] and np.abs(y) <= self.config.bin_edges[3] else 0)
        raw[self.config.bin_names[4]] = raw[trainable_targets[0]].apply(lambda y: 1 if np.abs(y) > self.config.bin_edges[3] and np.abs(y) <= self.config.bin_edges[4] else 0)
        raw[self.config.bin_names[5]] = raw[trainable_targets[0]].apply(lambda y: 1 if np.abs(y) > self.config.bin_edges[4] and np.abs(y) <= self.config.bin_edges[5] else 0)

        return raw

    def get_target_bin(self, y: float, ):
        # accepts a target dataframe or series

        bins = [0 for item in self.config.bin_edges]

        for i in range(len(self.config.bin_edges)):
            if i == len(self.config.bin_edges) - 1:     # last bin
                if np.abs(y) > self.config.bin_edges[i]: 
                    bins[i] = 1         # ex: activate if abs(y) in (1100,inf)
                    break

            if i == 0:                                  # first bin
                if np.abs(y) <= self.config.bin_edges[i]:
                    bins[i] = 1         # ex: activate if y in (-200,200)
                    break

            if np.abs(y) > self.config.bin_edges[i-1] and np.abs(y) <= self.config.bin_edges[i]:
                bins[i] = 1
        
        return bins
    


    def divide_regression(self, raw: pd.DataFrame, bin_names = None):
        # accept the raw dataframe with features, targets, and one-hot encoded targets
        # splits the raw dataframe into separate (features, targets) dataframes
        #   for training, testing, and validation, for each bin range (0-200, 200-400, etc.)
        #   based on the OHE bins

        # only for regression data sets

        bin_df_list = []    # list of raw_r0, raw_r1, raw_r2, etc. dataframes
        
        if bin_names is None:
            bin_names = self.config.bin_names

        for bname in bin_names:
            # return only the features and targets which have a 1 in the {bname} column
            #   (i.e. only the features and targets which fall into the bin range)
            bin_df = raw[raw[bname] == 1]   # only the rows which have a 1 in the {bname} column
            bin_df = bin_df.drop(columns=bin_names) # drop the OHE columns

            bin_df_list.append(bin_df)         

        return bin_df_list
    


    def divide_classification(self, raw: pd.DataFrame, bin_names = None):
        # compatible with 1-dim classification targets only

        if bin_names is None:
            bin_names = self.config.bin_names

        targets = [t for t in self.config.targets if t not in self.config.ignore_train]
        
        raw_ohe = raw.drop(columns=targets, errors='ignore') # drop the OHE columns

        return raw_ohe

    





    # def preprocess(self):
    #     self.train_data = self.train_data.map(
    #         DatasetGenerator.convert_types
    #     ).batch(self.config.batch_size)
    #     self.test_data = self.test_data.map(
    #         DatasetGenerator.convert_types
    #     ).batch(self.config.batch_size)
        
    # @staticmethod
    # def convert_types(batch):
    #     image, label = batch.values()
    #     image = tf.cast(image, tf.float32)
    #     image /= 255
    #     return image, label