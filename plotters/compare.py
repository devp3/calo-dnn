import logging, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Compare:
    def __init__(self, config):
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
    

    def plot(self, df1, df2, names:list ,yscale='linear'):
        # Feed in dataframes to compare in data
        # Feed in list of names of dataframes in order
        # Columns of df1 will be compared to columns of df2 in the order they 
        # appear in the dataframe

        self.df1 = df1
        self.df2 = df2
        self.names = names
        self.yscale = yscale

        if len(df1.columns) != len(df2.columns):
            logging.error("Dataframes do not have the same number of columns")
            logging.error('Exiting...')
            sys.exit(1)

        for i in range(len(df1.columns)):
            logging.debug(f'i = {i}')
            # get the first column of df1 as a dataframe
            c1 = df1.iloc[:, i:i+1]
            # get the first column of df2 as a dataframe
            c2 = df2.iloc[:, i:i+1]

            logging.debug(f'c1.head(): \n{c1.head()}')
            logging.debug(f'c2.head(): \n{c2.head()}')

            self.plot_compare(c1, c2)



    def plot_compare(self, c1, c2):


        if c1.columns[0] != c2.columns[0]:
            col_name = f'{c1.columns[0]}_{c2.columns[0]}'
        else:
            col_name = f'{c1.columns[0]}'
        
        self.save_path(f'compare_{self.names[0]}_{self.names[1]}_cols_{col_name}.png')

        fig = plt.figure(figsize=(10, 6))

        plt.hist(
            c1, 
            bins=100, 
            label=f'{c1.columns[0]} from {self.names[0]}', 
            histtype='step', 
            fill=False,
            density=True,
        )
        plt.hist(
            c2,
            bins=100,
            label=f'{c2.columns[0]} from {self.names[1]}',
            histtype='step',
            fill=False,
            density=True,
        )
        if self.yscale != None:
            plt.yscale(self.yscale)
        plt.legend()
        plt.xlabel(f'{c1.columns[0]} and {c2.columns[0]}')
        # plt.ylabel('Density')
        plt.savefig(self.save_plot_path)
        logging.debug(f"Saved plot to {self.save_plot_path}")
        plt.close()