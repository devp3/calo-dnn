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

        self.trainable_features = [f for f in self.features if f not in config.ignore_train]
        self.trainable_targets = [t for t in self.targets if t not in config.ignore_train]


    # Print iterations progress
    def printProgressBar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()


    def load(self):
        raw, _ = self.root_loader()
        raw = self.call_precompute(raw, self.config.features + self.config.targets)
        raw = raw.sample(frac=self.config.load_fraction).dropna()

        raw = cut.apply(raw, name=self.config.cut_name)

        self.handler.save(raw, 'raw_data')
        
        # self.handler.save()

        if self.normed:
            raw_normed, factors = linear.norm(raw, check=self.checknormed)
            # linear.save_factors(config.active_data_path + "/factors.csv", factors)
            logging.info("Normalized data.")
            
            self.handler.save(factors, 'normalization_factors')
            self.handler.save(raw_normed, 'normalized_data')

            train_data, test_data, val_data = self.splitter(raw_normed)                

        else:
            logging.info("Data not normalized.")
            train_data, test_data, val_data = self.splitter(raw)

        self.handler.save(train_data, 'train_data')
        self.handler.save(test_data, 'test_data')

        if self.mode == 'tf':
            train_data, test_data, val_data = self.preprocess(train_data, test_data, val_data)
        elif self.mode == 'pd':
            train_data, test_data, val_data = self.preprocess_pd(train_data, test_data, val_data)
        else:
            logging.error("Invalid mode: ", self.mode)
            logging.error("Mode must be 'tf' or 'pd'.")
            exit(1)

        # self.handler.save(train_data, 'train_data')
        # self.handler.save(test_data, 'test_data')

        # test_data.save(self.config.active_data_path + "/test_data")

        # (x_train, y_train), (x_test, y_test)
        # return (train_targets, test_targets), (train_features, test_features)

        logging.info("Data loaded.")
        
        return train_data, test_data, val_data


    def add_signal_type(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Parse filename to determine the signal type and add a pseudo-categorical
        feature column to the df dataframe representing the signal type.

        Signal Types:
            HyyHSM: 0
            HyyZSM: 1
            ZeeHSM: 2
            ZeeZSM: 3

        Args:
            df (pd.DataFrame): dataframe to add signal type column to
            filename (str): filename to parse to determine signal type

        Returns:
            pd.DataFrame: dataframe with signal type column added
        """

        if 'signal_type' in df.columns:
            logging.error(f'(DatasetGenerator, add_signal_type) signal_type already in dataframe columns.')
            exit(1)

        if 'HyyHSM' in filename:
            df['signal_type'] = 0
        elif 'HyyZSM' in filename:
            df['signal_type'] = 1
        elif 'ZeeHSM' in filename:
            df['signal_type'] = 2
        elif 'ZeeZSM' in filename:
            df['signal_type'] = 3
        else:
            logging.error(f'(DatasetGenerator, add_signal_type) Could not parse filename {filename} to determine signal type.')
            exit(1)

        return df
    

    def add_lifetime(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Parse filename to determine lifetime and add a pseudo-categorical
        feature column to the df dataframe representing the lifetime.
        
        Lifetimes:
            2ns: 0
            10ns: 1
            20ns: 2
            50ns: 3
            100ns: 4

        Args:
            df (pd.DataFrame): dataframe to add lifetime column to
            filename (str): filename to parse to determine lifetime

        Returns:
            pd.DataFrame: dataframe with lifetime column added
        """

        if 'lifetime' in df.columns:
            logging.error(f'(DatasetGenerator, add_lifetime) lifetime already in dataframe columns.')
            exit(1)
        
        if '2ns' in filename:
            df['lifetime'] = 0
        elif '10ns' in filename:
            df['lifetime'] = 1
        elif '20ns' in filename:
            df['lifetime'] = 2
        elif '50ns' in filename:
            df['lifetime'] = 3
        elif '100ns' in filename:
            df['lifetime'] = 4
        else:
            logging.error(f'(DatasetGenerator, add_lifetime) Could not parse filename {filename} to determine lifetime.')
            exit(1)
        
        return df
    

    def add_mass_point(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Parse filename to determine mass point and add a pseudo-categorical
        feature column to the df dataframe representing the mass point.
        
        Allowed Mass Points (GeV):
            100, 135, 175, 225, 275, 325, 375, 425, 475, 525, 625, 725, 775, 825, 925

        Args:
            df (pd.DataFrame): dataframe to add mass point column to
            filename (str): filename to parse to determine mass point

        Returns:
            pd.DataFrame: dataframe with mass point column added
        """

        filename_split = filename.split('_')
        mass_point = int(filename_split[filename_split.index('Hino') + 1])
        
        allowed_mass_points = [100, 135, 175, 225, 275, 325, 375, 425, 475, 525, 625, 725, 775, 825, 925]

        if 'mass_point' in df.columns:
            logging.error(f'(DatasetGenerator, add_mass_point) mass_point already in dataframe columns.')
            exit(1)
        
        if mass_point not in allowed_mass_points:
            logging.error(f'(DatasetGenerator, add_mass_point) Could not parse filename {filename} to determine mass point.')
            logging.debug(f'(DatasetGenerator, add_mass_point) Allowed mass points: {allowed_mass_points}')
            logging.debug(f'(DatasetGenerator, add_mass_point) Parsed mass point: {mass_point}')
            exit(1)

        df['mass_point'] = mass_point
        
        return df


    def root_loader(self):
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

        self.printProgressBar(0, len(root_files), prefix='Loading ROOT files:', suffix='Complete', length=50)

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

            # add the pseudo-categorical parsed variables
            if 'signal_type' in self.trainable_features + self.trainable_targets:
                file_df = self.add_signal_type(file_df, rf)
            if 'lifetime' in self.trainable_features + self.trainable_targets:
                file_df = self.add_lifetime(file_df, rf)
            if 'mass_point' in self.trainable_features + self.trainable_targets:
                file_df = self.add_mass_point(file_df, rf)

            raw = pd.concat([raw, file_df], ignore_index=True)
            self.printProgressBar(root_files.index(rf) + 1, len(root_files), prefix='Loading ROOT files:', suffix='Complete', length=50)

        logging.debug('PRINTING VARIABLES THAT NEED TO BE COMPUTED')
        for item in pending:
            logging.debug(item)

        # randomize the order of the dataframe, resetting the row indices
        logging.debug('(DatasetGenerator, root_loader) Randomizing rows in raw dataframe...')
        raw = raw.sample(frac=1).reset_index(drop=True)
        logging.debug('(DatasetGenerator, root_loader) Randomizing rows in raw dataframe...done.')

        return raw, pending


    def call_precompute(self, raw: pd.DataFrame, variables: list):
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