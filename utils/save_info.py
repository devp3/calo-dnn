import logging, os, sys, time, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from utils.handler import Handler

# saves:
# 1) copy of model summary
# 2) copy of config
# 3) copy of cuts file


class SaveInfo:
    def __init__(self, config):
        self.config = config
        self.base_path = self.config.base_path
        self.active_plots_path = os.path.join(
            self.base_path, 
            self.config.active_plots_path
        )
        self.active_data_path = os.path.join(
            self.base_path, 
            self.config.active_data_path
        )
        self.info_path = os.path.join(
            self.active_data_path,
            'info'
        )
        self.archive_path = self.config.archive_path
        self.handler = Handler(self.config)

    def save(self):
        
        # create a new directory in self.active_data_path

        self._make_info_dir()

        self._save_model_summary()

        self._save_config()

        self._save_cuts()

        pass


    def _make_info_dir(self):
        if not os.path.exists(self.info_path):
            os.makedirs(self.info_path)


    def _save_model_summary(self):
        # returns a string of the model summary

        self.model = keras.models.load_model(
            os.path.join(
                self.active_data_path,
                'model'
            ),
            compile=False
        )   # loads the model in active_data named 'model'
        self.model.summary(print_fn=self._summary_printer)

    
    def _summary_printer(self, s):
        model_name = self.model.name
        summary_path = os.path.join(
            self.info_path,
            f'{model_name}_summary.txt'
        )
        with open(summary_path, 'a') as f:
            print(s, file=f)


    def _get_config(self):
        pass

    def _get_cuts(self):
        pass



    def _save_config(self):
        pass


    def _save_cuts(self):
        pass