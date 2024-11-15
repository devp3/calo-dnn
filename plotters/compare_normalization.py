import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from utils.find_bin_count import find_bin_count
from functions.normalize.linear import norm as linear

# need to compare original, normed1, normed2, ... for each different normalizer


class CompareNormalization:
    def __init__(self, config):
        logging.info("CompareNormalization plotter initialized")
        self.config = config


    def save_path(self, name):
        try:
            assert isinstance(name, str)
        except AssertionError as err:
            logging.error("name must be a string")

        self.save_plot_path = os.path.join(
            self.config.base_path,
            self.config.active_plots_path,
            name
        )


    def plot(self, df, yscale='linear'):
        self.df = df
        self.yscale = yscale
        logging.info("Plotting Compare Normalization")

        for column in df.columns:
            self.plot_compare(df, column)



    def plot_compare(self, df, column):
        # column is the name of the column in df to plot

        try:
            assert isinstance(column, str)
        except AssertionError as err:
            logging.error("column must be a string")
            logging.debug(f"column = {column}")
            logging.debug(f"type(column) = {type(column)}")
            logging.error("Skipping this instance of plot_compare...")
            return 

        bins = find_bin_count(df[column])
