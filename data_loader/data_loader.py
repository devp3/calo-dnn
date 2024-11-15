import tensorflow as tf
import pandas as pd
import numpy as np
import os, sys, logging, time, uproot
import matplotlib.pyplot as plt

from functions.precompute import *
from utils.reorder import reorder
from functions.normalize import linear
from utils.handler import Handler
import cuts.cut as cut

# Typical usage:
# 


class Dataset:
    def __init__(
            self, 
            config,
            physics_type=None,
            id=None,
            **kwargs,
    ):
        self.config = config
        self.id = id                # unique run id
        self.kwargs = kwargs

        self.features = config.features
        self.targets = config.targets # should be list of strings
        self.trainable_features = [f for f in self.features if f not in config.ignore_train]
        self.trainable_targets = [t for t in self.targets if t not in config.ignore_train]

        self.data = pd.DataFrame()  # initialize empty dataframe
        self.physics_type = physics_type    # initialize physics type (e.g. 'signal', 'Zee')
        self.is_cut = False         # initialize cut status
        self.is_normed = False      # initialize normalization status
        self.cut_mode = None        # initialize cut mode (e.g. Zee, signal)

        self.handler = Handler(config)
        self.active_data_path = os.path.join(
            self.config.base_path, 
            self.config.active_data_path
        ) # TODO: make this naming less confusing


    def _make_auto_id(self):
        if self.id == None:
            self.id = int(time.time())



    def _attempt_auto_physics_type(self):
        # if physics_type not set, try to set it automatically
        possible_physics_types = {
            'Zee': 0,
            'signal': 0,
        }
        if self.physics_type == None:
            if 'ZeeMC' in ' '.join(self.load_path):
                possible_physics_types['Zee'] = 1
            elif 'ZeeZSM' in ' '.join(self.load_path):
                possible_physics_types['signal'] = 1
            elif 'ZeeHSM' in ' '.join(self.load_path):
                possible_physics_types['signal'] = 1
            elif 'HyyZSM' in ' '.join(self.load_path):
                possible_physics_types['signal'] = 1
            elif 'HyyHSM' in ' '.join(self.load_path):
                possible_physics_types['signal'] = 1
        
        if sum(possible_physics_types.values()) == 0:
            pass
        elif sum(possible_physics_types.values()) == 1:
            self.physics_type = list(possible_physics_types.keys())[list(possible_physics_types.values()).index(1)]
        elif sum(possible_physics_types.values()) > 1:
            # more than one physics type found, so no decision can be made
            pass

    
    def describe(self):
        description = {
            'physics_type': self.physics_type,
            'is_cut': self.is_cut,
            'cut_mode': self.cut_mode,
            'is_normed': self.is_normed,
            'normalization_mode': self.normalization_mode,
            'id': self.id,
        }



    def set_physics_type(self, physics_type):
        self.physics_type = physics_type
        if not isinstance(self.physics_type, str):
            logging.critical('physics_type must be a string')
            exit(0)



    def _decide_load_path(self, new, path):
        # default: new=True, path=None
        # if path is specified, then it should be a single file
        if new == True:     # new data, so load from data_path in config
            self.load_path = self.config.data_path
        elif new == False:  # not new data, so load from active_data_path
            if path == None:    # if path not specified, then return error
                logging.critical('(Dataset, load) Must specify path if new=False')
                exit(0)
            else:
                self.load_path = os.path.join(
                    self.config.base_path,
                    self.config.active_data_path,
                    path,
                )
        else:
            logging.critical('(Dataset, load) new must be True or False')
            exit(0)


    def load(
            self,
            new=True,
            path=None,      # rel path to data file. expecting one file.
            id=None,
            physics_type=None,
            inplace=True,
            savename=None,
            **kwargs,
        ):
        # data.load() -> loads data by processing root files
        # data.load(path='path/to/data') -> loads data from files in path (path not a filename)
        # data.load(new=False) -> loads data from active_data_path
        self.id = id        # unique run id
        self.physics_type = physics_type

        self._decide_load_path(new, path)
        self._attempt_auto_physics_type()

        if new == True:             # load new? -> load from root
            self.data = self._load_from_root()
            self.data = self.data.sample(frac=1).reset_index(drop=True) # shuffle data if new
        elif new == False:          # load preexisting? -> load from csv/pickle
            self.data = self._load_from_pandas() # load from csv or pickle
        else:
            logging.critical('(Dataset, load) new must be True or False')
            exit(0)

        if savename != None:
            logging.info(f'Saving {savename} to /active_data/')
            savename = savename.removesuffix('.pkl')
            self.data.to_pickle(f'{self.active_data_path}/{savename}.pkl')

        if not inplace: 
            return self.data
    


    def cut(
            self,
            mode=None,
            inplace=True,
            **kwargs,
        ):
        self.cut_mode = mode

        if self.cut_mode == None and self.physics_type != None:
            self.cut_mode = self.physics_type
            logging.info(f'No cut mode specified, so using physics type: {self.cut_mode}')
        if self.cut_mode == None and self.physics_type == None:
            logging.critical('(Dataset, cut) No cut mode specified, and no physics type specified.')
            exit(0)

        if not isinstance(self.cut_mode, str):
            logging.critical('(Dataset, cut) cut mode must be a string')
            exit(0)

        self.data = cut.apply(self.data, name=self.cut_mode, **kwargs)
        self.is_cut = True

        if not inplace:
            return self.data



    def normalize(
            self,
            mode='linear',
            factors=None,
            ignore: list = [],
            save: bool = False,
            rename: dict = {},
            inplace=True,
            path=None,
            **kwargs,
        ):
        # kwargs:
        #   - ignore (list): list of features to ignore
        self.normalization_mode = mode

        # if path != None:
        #     self.norm_factors = 

        if mode == 'linear':
            self.data, self.norm_factors = linear.norm(     # if factors is not None, then self.norm_factors = factors
                self.data, 
                factors=factors,    # if factors is not None, then it will be used instead of calculating new factors
                ignore=ignore, 
                rename=rename, 
                **kwargs
            )
            self.normalization_mode = mode
            self.is_normed = True
        else:
            logging.critical('(Dataset, normalize) normalization mode not recognized')
            exit(0)

        if save:
            self.norm_factors.to_csv(f'{self.active_data_path}/normalization_factors.csv')
        
        if not inplace:
            return self.data
    
    

    def denormalize(
            self,
            mode='linear',
            factors=None,
            **kwargs,
        ):
        # kwargs: 
        #   - ignore (list): list of features to ignore
        if self.is_normed == False and factors == None:
            logging.warning('(Dataset, denormalize) dataset not normalized, returning original data')
            return self.data

        if factors != None:
            self.norm_factors = factors     # this will allow self.norm_factors to be used again. is this a good idea?

        if self.norm_factors == None:
            logging.warning('(Dataset, denormalize) normalization factors not found. returning original data.')

        if mode == 'linear':
            self.data = linear.denorm(self.data, self.norm_factors, **kwargs)
            self.normalization_mode = None
            self.is_normed = False
        else:
            logging.critical('(Dataset, denormalize) normalization mode not recognized')
            exit(0)        
        
        return self.data
    


    def get_norm_factors(self):
        if self.is_normed:
            return self.norm_factors
        else:
            logging.warning('(Dataset, get_norm_factors) dataset not normalized')
            return None
        
    
    def _separate_features_targets(self):
        """Separates self.data dataframe into features and targets dataframes.

        Sets:
            self.data_features: features dataframe
            self.data_targets: targets dataframe

        """
        self.data_features = self.data.drop(self.trainable_targets, axis=1)
        self.data_targets = self.data[self.trainable_targets]



    def _splitter(self):
        """Splits self.data dataframe into train, test, and validation sets and 
        separates features and targets into separate dataframes.

        Sets:
            self.train_data: (x_train, y_train)
            self.test_data: (x_test, y_test)
            self.val_data: (x_val, y_val)

        for x->features, y->targets.

        """


        target_variables = self.config.targets
        test_split = self.config.test_split
        validation_split = 1 - self.config.validation_split

        logging.debug(f'test_split = {test_split}')
        logging.debug(f'validation_split = {validation_split}')

        valtrain_dataset = self.data.sample(frac=test_split, random_state=0)
        test_dataset = self.data.drop(valtrain_dataset.index)

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

        logging.debug(f'(Dataset, _splitter) train_targets.shape: {train_targets.shape}')
        logging.debug(f'(Dataset, _splitter) train_features.shape: {train_features.shape}')
        logging.debug(f'(Dataset, _splitter) test_targets.shape: {test_targets.shape}')
        logging.debug(f'(Dataset, _splitter) test_features.shape: {test_features.shape}')
        logging.debug(f'(Dataset, _splitter) val_targets.shape: {val_targets.shape}')
        logging.debug(f'(Dataset, _splitter) val_features.shape: {val_features.shape}')

        # (x,y) since x->features and y->targets
        self.train_data = (train_features, train_targets)
        self.test_data = (test_features, test_targets)
        self.val_data = (val_features, val_targets)

        self.split_data = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data,
        }



    def preprocess(
            self, 
            outtype='tf',
            split=True,
            inplace=False,
            batch=True,
            **kwargs,
        ):
        """Return (x_test, y_test), (x_train, y_train), (x_val, y_val) tuples of
        either tf.data.Dataset objects or pandas dataframes split according to 
        ratios in config.

        Args:
            outtype (str, optional): 'tf' for tf.data.Dataset objects or 'pd' for pandas dataframes. Defaults to 'tf'.
            split (bool, optional): True to split data into train, test, and validation sets. False to skip train/test/val splitting. Defaults to True.

        Sets:
            self.preprocessed_data (dict): dictionary of preprocessed data (keys are typically 'train', 'test', 'val', or 'all').

        Yields:
            (tuple): (x, y) for test, train, and validation or single (x, y) for all data if split=False
        """        

        # create tf.data.Dataset objects that are ready to be fed into the model

        # expects a tuple of (features, targets) dataframes

        # batch will divide dataset into subsets of length batch_size
        # drop_remainder will drop the last batch if it is smaller than batch_size
        # buffer_size is the size of the buffer for the suffling algorithm
        #   buffer_size must be greater than or equal to the dataset size for 
        #   perfect suffling, but it should be fine if it is smaller

        # train_data, test_data, val_data

        if split:
            self._splitter()
        elif not split:
            self._separate_features_targets()
            self.split_data = {'all': (self.data_features, self.data_targets)}
        else:
            logging.critical('(Dataset, preprocess) split not recognized')
            logging.debug(f'split = {split}')
            exit(0)

        if outtype not in ['tf', 'pd']:
            logging.critical('(Dataset, preprocess) outtype must be tf or pd')
            exit(0)

        self.preprocessed_data = {}

        # x_data = features, y_data = targets
        for key, (x_data, y_data) in self.split_data.items():
            x_data = self._drop_ignore(x_data)
            y_data = self._drop_ignore(y_data)

            if outtype == 'tf':
                xy_data = tf.data.Dataset.from_tensor_slices((x_data.values, y_data.values))
                if batch:
                    xy_data = xy_data.batch(self.config.batch_size, drop_remainder=False)
                self.preprocessed_data[key] = xy_data

            elif outtype == 'pd':
                xy_data = (x_data, y_data)
                self.preprocessed_data[key] = xy_data

            else:
                logging.critical('(Dataset, preprocess) outtype not recognized')
                exit(0)

            if not inplace:
                yield xy_data

    

    def get_preprocessed_data(self):
        return self.preprocessed_data


    def rename(
            self,
            rename_dict: dict,  # {old_name: new_name}
            inplace=True,   # don't return anything
            remove=False,   # remove columns not in rename_dict
            reverse=False,  # reverse the rename_dict
        ):
        self.rename_dict = rename_dict
        
        if reverse:
            self.rename_dict = {v: k for k, v in self.rename_dict.items()}
        
        for column in self.data.columns:
            if column in self.rename_dict.keys():
                self.data.rename(columns={column: self.rename_dict[column]}, inplace=True)
            elif remove:
                self.data.drop(column, axis=1, inplace=True)
            else:
                pass

        if not inplace:
            return self.data



    # def predict(
    #         self, 
    #         model: tf.keras.Model,
    #         normalized: bool = False,
    #         inplace: bool = False,
    #     ):
    #     self.preprocess(split=False, batch=True, inplace=True)
        
    #     print(self.preprocessed_data['all'])

    #     self.predictions = model.predict(next(self.preprocessed_data['all']))

    #     if not normalized:
    #         self.predictions = linear.denorm(self.predictions, self.norm_factors)

    #     if not inplace:
    #         return self.predictions



    def get_id(self):
        return self.id
    

    def get_physics_type(self):
        return self.physics_type
    

    def to_pd(
            self, 
            only_trainable=False,
        ):
        if only_trainable:
            return self.data[self.trainable_features + self.trainable_targets]
        else:
            return self.data
    

    def to_tf(
            self,
            only_trainable=False,
        ):
        print('not implemented yet')
        pass



    def _drop_ignore(self, data: pd.DataFrame):
        # drop any variables in the ignore list
        ignore = self.config.ignore_train
        data = data.drop(ignore, axis=1, errors='ignore')
        return data
    


    def _load_from_root(self):
        # can handle multiple files in directory
        raw, _ = self._root_loader()
        raw = self._call_precompute(raw, self.features + self.targets)
        raw = raw.sample(frac=self.config.load_fraction).dropna()
        return raw
    


    def _load_from_pandas(self):
        is_csv = False
        is_pickle = False

        if os.path.isdir(self.load_path):
            logging.critical(f'(Dataset, load) Load path cannot be a directory. Must be single file.')
            logging.debug(f'(Dataset, load) Load path: {self.load_path}')
            exit(0)

        if not os.path.isfile(self.load_path):
            logging.critical(f'(Dataset, load) File not found: {self.load_path}')
            exit(0)

        if self.load_path.endswith('.csv'):
            is_csv = True
        elif self.load_path.endswith('.pkl'):
            is_pickle = True
        elif self.load_path.endswith('.pickle'):
            is_pickle = True
        else:
            logging.critical(f'(Dataset, load) File type not recognized: {self.load_path}')
            exit(0)
        
        if is_csv and not is_pickle:
            return pd.read_csv(self.load_path)
        elif is_pickle and not is_csv:
            return pd.read_pickle(self.load_path)
        else:
            logging.critical(f'(Dataset, load) Issue with loading file: {self.load_path}')
            logging.debug(f'is_csv: {is_csv}, is_pickle: {is_pickle}')
            exit(0)



    def _printProgressBar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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


    def _root_loader(self):
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

        self._printProgressBar(0, len(root_files), prefix='Loading ROOT files:', suffix='Complete', length=50)

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

                        file_df[branch] = file[branch].array(library='np') # type: ignore

                        if isinstance(file_df[branch].iloc[0], bool) or isinstance(file_df[branch].iloc[0], np.bool_):
                            # if the branch is a boolean, convert it to an int
                            file_df[branch] = file_df[branch].apply(lambda x: int(x))

                pending = list(set(variables) - set(branches))

            raw = pd.concat([raw, file_df], ignore_index=True)
            self._printProgressBar(root_files.index(rf) + 1, len(root_files), prefix='Loading ROOT files:', suffix='Complete', length=50)

        logging.debug('PRINTING VARIABLES THAT NEED TO BE COMPUTED')
        for item in pending:
            logging.debug(item)

        # randomize the order of the dataframe, resetting the row indices
        logging.debug('(DatasetGenerator, root_loader) Randomizing rows in raw dataframe...')
        raw = raw.sample(frac=1).reset_index(drop=True)
        logging.debug('(DatasetGenerator, root_loader) Randomizing rows in raw dataframe...done.')

        return raw, pending


    def _call_precompute(self, raw: pd.DataFrame, variables: list):
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