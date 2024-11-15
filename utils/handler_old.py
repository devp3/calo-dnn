import os, sys, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import h5py
import pickle



# This processor is meant to handle the safe loading, saving, and moving 
# of files. It is meant to be an intermediate step between the processors and
# the save directory. 

# TODO: Add a function to warn the user via command line if a save/copy action
# is making a copy of a file that is larger than a certain size.


# Do not accept matplotlib objects. These will be saved as a pdf/png/jpg in 
# active plots. The user can then archive them if they wish. MPL's savefig
# function should be used within the plotter. 

# LOAD
# dataframe -> pickle


# MAIN FUNCTIONS:
# 1) load - returns data from active or archive directory (does not change any file)
#   - look in active directory first, then look in archive directory (TODO)
# 2) save - writes to active directory (savetype='archive' will copy to archive)
# 3) archive - copies from active to archive directory


# exmaple usage:
# hander = Handler(config)
# my_dataframe = handler.load('dataframe') # returns dataframe from active directory
# Looks for dataframe_
# handler.save(my_dataframe)


# Active directory special names:
# raw_data - raw data of all loaded variables, no processing, no cuts, no normalization
# normalized_data - data after normalization, but before cuts and other processing
# processed_data - data after all processing, cuts, and normalization
# train_data - data used for training
# test_data - data used for testing
# normalization_factors - normalization factors used to normalize the data
# history - history of training
# model - tensorflow model used for training





# save objects using name: <variablename>_<datetime>_<objecttype>.<extension>


class Handler():
    def __init__(self, config, loc='active', name=None, savetype='default'):
        self.config = config
        # self.object = object
        self.loc = loc  # TODO: need to change behavior and not use loc
        self.name = name
        self.savetype = savetype
        self._validate_loc()        # make sure loc is either 'active' or 'archive'
        self._validate_name()       # make sure name argument is being used correctly
        self._get_basepath()        # get the base path given the instructions
        self._validate_basepath()   # make sure the base bath is valid


    def _validate_loc(self):
        # self.loc must be either 'active' or 'archive'
        if self.loc not in ['active', 'archive']:
            logging.error("Invalid location, ", self.loc, ". Must be 'active' or 'archive'.")
            logging.error("Chosing 'active' as default.")
            self.loc = 'active'

    def _set_mode(self):
        if type(self.object) == str:
            self.mode = 'save'
        else:
            self.mode = 'load'

    def _validate_name(self):
        # in 'load' mode, the name of the object to be loaded is specified in 
        # in the object argument, not the name argument. The name argument is
        # only used in 'save' mode, so if it is specified in 'load' mode, it 
        # can be assumed the user is mistakenly using the wrong argument. 
        if self.mode == 'load':
            if self.name is not None:
                logging.warning("Name argument is not used in 'load' mode. Ignoring.")
                self.name = None
        elif self.mode == 'save':
            if self.name is None:
                logging.debug("In save mode and no name argument specified. Using string in object argument as name.")
            elif self.name is not None:
                logging.debug("In save mode and name argument specified. Using name argument as name.")
    

    def _get_basepath(self):
        if self.loc == 'active':
            self.basepath = self.config.active_data_path
            logging.debug("Basepath location is active data path.")
            logging.debug("Basepath set to: ", self.basepath)
                
        elif self.loc == 'archive':
            self.basepath = self.config.archive_data_path
            logging.debug("Basepath location is archive data path.")
            logging.debug("Basepath set to: ", self.basepath)

        else: 
            logging.warning("Invalid location: ", self.loc)
            logging.warning("Location must be 'active' or 'archive'.")
            logging.warning("Chosing 'active' as default.")
            self.basepath = self.config.active_data_path


    def _validate_basepath(self):
        # check if path is not None
        # check if path is a string
        # check if path is a valid path
        # check is path is a directory
        if self.basepath is None:
            logging.critical("Invalid path: ", self.basepath)
            logging.critical("Path is None. Cannot complete action.")
            exit(1)
        elif type(self.basepath) != str:
            logging.critical("Invalid path: ", self.basepath)
            logging.critical("Path is not a string. Cannot complete action.")
            exit(1)
        elif not os.path.exists(self.basepath):
            logging.critical("Invalid path: ", self.basepath)
            logging.critical("Path does not exist. Cannot complete action.")
            exit(1)
        elif not os.path.isdir(self.basepath):
            logging.critical("Invalid path: ", self.basepath)
            logging.critical("Path is not a directory. Cannot complete action.")
            exit(1)
        else:
            logging.debug("Path is valid: ", self.basepath)
    

    def analyze_object(self):
        # determine the datatype of the object
        # determine the size of the object, used to raise warnings if large
        # determine the shape of the object, if applicable

        if isinstance(self.object, pd.DataFrame):
            self.object_type = 'dataframe'
        elif isinstance(self.object, np.ndarray):
            self.object_type = 'numpy'
        elif isinstance(self.object, list):
            logging.warning("Object is list type. It will be converted to a numpy array and saved.")
            self.object = np.array(self.object)
            self.object_type = 'list'
        elif isinstance(self.object, dict):
            self.object_type = 'dict'
        elif isinstance(self.object, str):
            self.object_type = 'string'
            logging.warning("Object is string type. Is this intended?")
        elif isinstance(self.object, int):
            self.object_type = 'int'
            logging.warning("Object is int type. Is this intended?")
        elif isinstance(self.object, float):
            self.object_type = 'float'
            logging.warning("Object is float type. Is this intended?")
        elif isinstance(self.object, bool):
            self.object_type = 'bool'
            logging.warning("Object is bool type. Is this intended?")
        elif isinstance(self.object, tuple):
            self.object_type = 'tuple'
            logging.warning("Object is tuple type. Is this intended?")
        elif isinstance(self.object, set):
            self.object_type = 'set'
            logging.warning("Object is set type. Is this intended?")
        elif isinstance(self.object, type(None)):
            self.object_type = 'None'
            logging.warning("Object is None type. Nothing will be saved.")
        else:
            logging.error("Object type not recognized. Cannot .")
            exit(1)


    def load(self, object):
        # self._set_mode()            # set the mode to 'save' or 'load' from type(object)


        if self.loc == 'active':
            return self.load_active()

        if self.loc == 'archive':
            return self.load_archive()


    def load_active(self):
        # check if object is a string
        if type(self.object) != str:
            logging.error("Object argument is not a string. Cannot load.")
            exit(1)

        if os.path.isfile(os.path.join(self.basepath, self.object)):
            self.path_to_object = os.path.join(self.basepath, self.object)
        elif os.path.isfile(os.path.join(self.basepath, self.object + '.pkl')):
            self.path_to_object = os.path.join(self.basepath, self.object + '.pkl')
        else:
            logging.critical("Object not found in active data path.")
            logging.debug("Object: ", self.object)
            logging.debug("Basepath: ", self.basepath)
            exit(1)
    
        try: 
            with open(self.path_to_object, 'rb') as f:
                data = pickle.load(f)
            self.object_type = type(data)
            return data

        except:
            logging.critical("Error loading object: ", self.path_to_object)
            logging.debug("Object: ", self.object)
            logging.debug("Basepath: ", self.basepath)
            exit(1)


    def load_archive(self):
        # check if object is a string
        if type(self.object) != str:
            logging.error("Object argument is not a string. Cannot load.")
            exit(1)
        
        if os.path.isfile(os.path.join(self.basepath, self.object)):
            self.path_to_object = os.path.join(self.basepath, self.object)
        elif os.path.isfile(os.path.join(self.basepath, self.object + '.pkl')):
            self.path_to_object = os.path.join(self.basepath, self.object + '.pkl')
        else: 
            logging.critical("Object not found in archive data path.")
            logging.debug("Object: ", self.object)
            logging.debug("Basepath: ", self.basepath)
            exit(1)
        
        try:
            with open(self.path_to_object, 'rb') as f:
                data = pickle.load(f)
            self.object_type = type(data)
            return data
        
        except:
            logging.critical("Error loading object: ", self.path_to_object)
            logging.debug("Object: ", self.object)
            logging.debug("Basepath: ", self.basepath)
            exit(1)

    
    def save(self):
        # file_name = self._generate_file_name()
        file_name = str(self.object) + '.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(self.object, f)
        

    
    def archive(self):
    
        return

