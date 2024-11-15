import os, sys, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime
import tensorflow as tf
from tensorflow import keras


# CALLABLE FUNCTIONS
# Loading
# load -> returns unpickled object
# load 


class Handler():
    def __init__(self, config):
        self.config = config

    
    def load(self, name, mode='auto'):
        default_files_pkl = {
            'raw_data': 'raw dataframe of all loaded variables, no preprocessing or normalization.',
            'normalized_data': 'raw dataframe after normalization, but without any other preprocessing.',
            'preprocessed_data': 'dataframe after all preprocessing',
            'train_data': 'data used for training',
            'test_data': 'data used for testing',
            'normalization_factors': 'normalization factors used to normalize the data',
            'history': 'history of the training process',
        }
        default_files_tf = {
            'model': 'tensorflow model',
            'checkpoint': 'checkpoint used during training',
        }
        

        if mode == 'auto':
            # expect name to be a string specifying which default temp file to load
            # name must be one member of the default_files dictionary
            logging.debug("Mode: auto")
            if name[-4:] == '.pkl':
                # slice off the .pkl extension if it exists
                logging.debug(f"Removing .pkl extension from {name}")
                name = name[:-4]
            if name in default_files_pkl:
                logging.debug(f"Found name {name} in default pickle files dictionary")
                path = os.path.join(
                    self.config.base_path,
                    self.config.active_data_path,
                    f"{name}.pkl"
                )
                logging.debug(f"Path: {path}")
                if os.path.isfile(path):
                    logging.debug(f"File {name}.pkl exists in active_directory")
                    logging.info(f"Loading {name}.pkl from active_directory")
                    self.data = self.pickle_loader(path)
                else: 
                    logging.error(f"File {name}.pkl does not exist in active_directory")
                    logging.error("Exiting...")
                    sys.exit(1)
            elif name in default_files_tf:
                logging.debug(f"Found name {name} in default tensorflow files dictionary")
                path = os.path.join(
                    self.config.base_path,
                    self.config.active_data_path,
                    name                        # no extension because it's a directory
                )
                if os.path.isdir(path):
                    # TODO: implement loading of checkpoints
                    try:
                        assert name != 'model'
                    except AssertionError as err:
                        logging.error("Trying to load a tensorflow object that is not a model. ")
                        logging.error("This functionality has not been implemented yet.")
                        logging.error("Probably tried to load a checkpoint.")
                        logging.error('Exiting...')

                    logging.debug(f'Directory {name} exists in active_directory')
                    logging.info(f'Loading {name} from active_directory')
                    self.data = self.model_loader(path)

            
            else:
                logging.error(f"Name {name} not found in default files dictionary")
                logging.error("Available default files:")
                for key in default_files_pkl:
                    logging.error(f"\t{key}: {default_files_pkl[key]}")
                logging.error("Exiting...")
                sys.exit(1)
        
        elif mode == 'active':
            # expect name to be a string specifying which file in active_data to load
            # name can be anything as long as the file exists
            logging.debug("Mode: active")
            path = os.path.join(
                self.config.base_path,
                self.config.active_data_path,
                f"{name}.pkl"
            )
            logging.debug(f"Path: {path}")
            if os.path.isfile(path):
                logging.debug(f"File {name}.pkl exists in active_directory")
                logging.info(f"Loading {name}.pkl from active_directory")
                self.data = self.pickle_loader(path)
            else:
                logging.error(f"File {name}.pkl does not exist in active_directory")
                logging.error("Exiting...")
                sys.exit(1)
        
        elif mode == 'archive':
            # expect name to be a string specifying which file in archive_data to load
            # name can be anything as long as the file exists (can be subdirectory)
            logging.debug("Mode: archive")
            path = os.path.join(
                self.config.archive_path,
                f"{name}.pkl"
            )
            logging.debug(f"Path: {path}")
            if os.path.isfile(path):
                logging.debug(f"File {name}.pkl exists in archive_directory")
                logging.info(f"Loading {name}.pkl from archive_directory")
                self.data = self.pickle_loader(path)

        else:
            logging.error(f"Mode {mode} not recognized")
            logging.error("Exiting...")
            sys.exit(1)
            



        return self.data



    # TODO: saving this for later -> need to think a little more about the best 
    # way to structure the user input to the main save function. 
    def save(self, object, name:str, objecttype=None, mode='auto'):
        # saving object
        # in mode = 'auto', name specifies which type of default file to save
        # in mode = 'active', name specifies the name of the file to save in active_data
        # in mode = 'archive', name specifies the name of the file to save in 
        #   the archive directory

        default_files_pkl = {
            'raw_data': 'raw dataframe of all loaded variables, no preprocessing or normalization.',
            'normalized_data': 'raw dataframe after normalization, but without any other preprocessing.',
            'preprocessed_data': 'dataframe after all preprocessing',
            'train_data': 'data used for training',
            'test_data': 'data used for testing',
            'normalization_factors': 'normalization factors used to normalize the data',
            'history': 'history of the training process',
        }
        default_files_tf = {
            'model': 'tensorflow model',
            'checkpoint': 'checkpoint used during training',
        }

        # if objecttype != None:
        #     logging.debug(f"Object type specified: {objecttype}")

        # if name.find('model') != -1:
        #     logging.debug("Found 'model' in save name")
        #     objecttype = 'model'

        # elif name.find('callback') != -1:
        #     logging.debug("Found 'callback' in save name")
        #     objecttype = 'checkpoint'
        
        # elif name.find('checkpoint') != -1:
        #     logging.debug("Found 'checkpoint' in save name")
        #     objecttype = 'checkpoint'
        
        # elif name.find('normalized') != -1 or name.find('normed') != -1:
        #     logging.debug("Found 'normalized' in save name")
        #     objecttype = 'normalized_data'

        # elif name.find('preprocessed') != -1:
        #     logging.debug("Found 'preprocessed' in save name")
        #     objecttype = 'preprocessed_data'
        
        # elif name.find('train') != -1:
        #     logging.debug("Found 'train' in save name")
        #     objecttype = 'train_data'
        
        # elif name.find('test') != -1:
        #     logging.debug("Found 'test' in save name")
        #     objecttype = 'test_data'

        # elif name.find('history') != -1:
        #     logging.debug("Found 'history' in save name")
        #     objecttype = 'history'        
        
        # elif name.find('normalization') != -1:
        #     logging.debug("Found 'normalization' in save name")
        #     objecttype = 'normalization_factors'

        # else:
        #     logging.error(f"Object type not specified and no special keywords found in save name")
        #     logging.error("Exiting...")
        #     sys.exit(1)


        if mode == 'auto':
            # expect name to be a string specifying which default temp file to save
            # name must be one member of the default_files_pkl or default_files_tf
            # dictionaries
            logging.debug("Mode: auto")
            # if objecttype != None:
            #     logging.debug(f"Objecttype: {objecttype}")
            #     logging.warning(f"Expected objecttype None, but got {objecttype}. Ignoring objecttype.")

            if name[-4:] == '.pkl':
                # slice off the .pkl extension if it exists
                logging.debug(f"Removing .pkl extension from {name}")
                name = name[:-4]
            
            if name in default_files_pkl:
                logging.debug(f"Found name {name} in default pickle files dictionary")
                path = os.path.join(
                    self.config.base_path,
                    self.config.active_data_path,
                    f"{name}.pkl"
                )
                logging.debug(f"Path: {path}")
                logging.debug(f"Saving {name}.pkl to active_directory")
                self.pickle_saver(object, path)
            elif name in default_files_tf:
                logging.debug(f"Found name {name} in default tensorflow files dictionary")
                path = os.path.join(
                    self.config.base_path,
                    self.config.active_data_path,
                    name                        # no extension because it's a directory
                )
                logging.debug(f"Path: {path}")
                logging.debug(f"Saving {name} to active_directory")
                self.tf_saver(object, path, name)
            else:
                logging.error(f"Name {name} not found in default files dictionaries")
                logging.error("Exiting...")
                sys.exit(1)


        elif mode == 'active':
            # expect name to be a string specifying the name of the file to save
            #   in active_data
            # we will also need input from objecttype to specify the type of object
            logging.debug("Mode: active")

            if objecttype == None:
                logging.error(f"Objecttype None not allowed in mode active. Must specify object type.")
                logging.error("Exiting...")
                sys.exit(1)

            if name[-4:] == '.pkl':
                # slice off the .pkl extension if it exists
                logging.debug(f"Removing .pkl extension from {name}")
                name = name[:-4]

            if objecttype in default_files_pkl:
                logging.debug(f"Found {objecttype} in default pickle files dictionary")
                path = os.path.join(
                    self.config.base_path,
                    self.config.active_data_path,
                    f"{name}.pkl"
                )
                logging.debug(f"Path: {path}")
                logging.debug(f"Saving {name}.pkl to active_directory")
                self.pickle_saver(object, path)
            elif objecttype in default_files_tf:
                logging.debug(f"Found {objecttype} in default tensorflow files dictionary")
                path = os.path.join(
                    self.config.base_path,
                    self.config.active_data_path,
                    name                        # no extension because it's a directory
                )
                logging.debug(f"Path: {path}")
                logging.debug(f"Saving {name} to active_directory")
                self.tf_saver(object, path, objecttype)
            else:
                logging.error(f"Object type {objecttype} not found in default files dictionaries")
                logging.error("Exiting...")
                sys.exit(1)
            

        elif mode == 'archive':
            # expect name to be a string specifying the name of the file to save
            #   in archive_data
            # we will also need input from objecttype to specify the type of object
            logging.debug("Mode: archive")

            if objecttype == None:
                logging.error(f"Objecttype None not allowed in mode archive. Must specify object type.")
                logging.error("Exiting...")
                sys.exit(1)

            if name[-4:] == '.pkl':
                # slice off the .pkl extension if it exists
                logging.debug(f"Removing .pkl extension from {name}")
                name = name[:-4]

            if objecttype in default_files_pkl:
                logging.debug(f"Found {objecttype} in default pickle files dictionary")
                path = os.path.join(
                    self.config.base_path,
                    self.config.archive_data_path,
                    f"{name}.pkl"
                )
                logging.debug(f"Path: {path}")
                logging.debug(f"Saving {name}.pkl to archive_directory")
                self.pickle_saver(object, path)
            elif objecttype in default_files_tf:
                logging.debug(f"Found {objecttype} in default tensorflow files dictionary")
                path = os.path.join(
                    self.config.base_path,
                    self.config.archive_data_path,
                    name                        # no extension because it's a directory
                )
                logging.debug(f"Path: {path}")
                logging.debug(f"Saving {name} to archive_directory")
                self.tf_saver(object, path, objecttype)
            else:
                logging.error(f"Object type {objecttype} not found in default files dictionaries")
                logging.error("Exiting...")
                sys.exit(1)
        
        else:
            logging.error(f"Mode {mode} not recognized. Exiting...")
            sys.exit(1)




    def pickle_loader(self, path:str):
        if not os.path.isfile(path):
            logging.error(f"File {path} does not exist")
            logging.error("Exiting...")
            sys.exit(1)
        else: 
            logging.debug(f"Loading {path}...")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data

    @staticmethod
    def pickle_saver(data, path:str):
        # try:
        logging.debug(f"Saving {path}...")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        # except:
        #     logging.error(f"Error saving {path} in pickle_saver.")
        #     logging.error("Exiting...")
        #     sys.exit(1)
    

    # TODO: validate this method
    def model_saver(self, model, path:str):
        logging.debug(f"Saving {path}...")
        model.save(path)
        # tf.keras.models.save_model(model, path)
        # tf.saved_model.save(model, path) # bad method, returns error


    # TODO: validate this method
    def model_loader(self, path:str):
        if not os.path.exists(path):
            logging.error(f"File {path} does not exist")
            logging.error("Exiting...")
            sys.exit(1)
        else:
            logging.debug(f"Loading {path}...")
            model = keras.models.load_model(path)
            return model


    # checkpoints are also saved in the trainer by a callback
    # TODO: validate this method
    def checkpoint_saver(self, model, path:str):
        logging.debug(f"Saving checkpoint {path}...")
        model.save_weights(path)


    # TODO: validate this method
    def checkpoint_loader(self, model, path:str):
        # model is a tf.keras.Model
        if not os.path.exists(path):
            logging.error(f"File {path} does not exist")
            logging.error("Exiting...")
            sys.exit(1)
        else:
            logging.debug(f"Loading {path}...")
            model.load_weights(path)
            return model


    def tf_saver(self, model, path:str, name:str):
        # for all modes, tf_saver will simply pass the path arg to 
        #   model/checkpoint saver
        # the path arg is supposed to contain the filename of the object to save
        # the name arg is only supposed to specify 'model' or 'checkpoint'

        if name == 'model':
            self.model_saver(model, path)
        elif name == 'checkpoint':
            self.checkpoint_saver(model, path)
        else:
            logging.error(f"Name {name} not recognized")
            logging.error("Exiting...")
            sys.exit(1)

    

    def tf_loader(self, path: str, name:str, model=None):
        if name == 'model':
            return self.model_loader(path)
        elif name == 'checkpoint':
            return self.checkpoint_loader(model, path)
        else:
            logging.error(f"Name {name} not recognized")
            logging.error("Exiting...")
            sys.exit(1)

    
    # def tf_loader(self, path:str, type:str):


    # def archive(self):
