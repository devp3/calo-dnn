import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from utils.find_bin_count import find_bin_count


class Kinematics:
    def __init__(self, config, vars=['pT', 'eta', 'phi', 'E']):
        logging.info("Kinematics plotter initialized")
        self.vars = vars
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


    def plot(self, df, title=None, yscale='linear'):
        self.df = df
        self.columns = df.columns
        self.yscale = yscale
        logging.info("Plotting kinematics")

        if title is not None:
            self.title = title
            logging.info(f"Using plotter title: {self.title}")
        else:
            self.title = None
            logging.info("Using default plotter titles")

        var_index = self.find_variables()

        self.plot_pt(self.df, var_index)
        self.plot_eta(self.df, var_index)
        self.plot_phi(self.df, var_index)
        self.plot_E(self.df, var_index)


    def find_variables(self):
        # check that every variable is in the dataframe
        # when found, add the column to the list of variables
        logging.info("Finding variables")

        self.vars = [item.lower() for item in self.vars]
        self.columns = [item.lower() for item in self.columns]
        
        logging.debug(f"Variables to plot: {self.vars}")
        logging.debug(f"Columns in dataframe: {self.columns}")

        alternative_names = {
            'pt': ['pT', 'el1_pt', 'ph1_pt', 'el2_pt', 'ph2_pt'],
            'eta': [
                'eta', 'el1_etas1', 'el2_etas1', 'el1_etas2', 'el2_etas2', 
                'el1_eta', 'el2_eta', 'ph1_eta', 'ph2_eta', 'ph1_etas1', 
                'ph2_etas1', 'ph1_etas2', 'ph2_etas2'
                ],
            'phi': ['el1_phi', 'el2_phi', 'ph1_phi', 'ph2_phi'],
            'e': ['el1_E', 'el2_E', 'ph1_E', 'ph2_E'],
        }
        var_index = {}
        for var in self.vars:
            if var in self.columns:
                logging.debug(f"Found variable string, {var} in dataframe")
                var_index[var] = self.df.columns.get_loc(var)
            elif var in alternative_names.keys():
                logging.debug(f"Variable {var} is in the alternative names dictionary.")
                for alt in alternative_names[var]:
                    alt = alt.lower()
                    if alt in self.columns:
                        logging.debug(f"Found alternative variable string, {alt}, in dataframe")
                        # find the first instance of alt in the dataframe, then break
                        logging.debug(f"alt = {alt}")
                        logging.debug(f"var = {var}")
                        var_index[var] = self.columns.index(alt)
                        break
                if var not in var_index:
                    logging.debug(f"Variable {var} found in alternative names dictionary, but could not pair to a variable in the dataframe.")
                    logging.warning(f"Could not find variable string, {var} in dataframe or alternative names.")
                    logging.warning(f"Skipping variable: {var}")
                    self.vars.remove(var)
                    continue
            else:
                logging.warning(f"Could not find variable string, {var} in dataframe or alternative names.")
                logging.warning(f"Skipping variable: {var}")
                self.vars.remove(var)
                continue
        
        logging.debug(f"Variable index dictionary: {var_index}")
        logging.info("Found variables in dataframe.")

        return var_index


    def plot_pt(self, df, var_index):
        self.save_path('pt.png')


        if 'pt' not in var_index:
            # if find_variables could not find the pT variable, skip plotting
            logging.warning("Not plotting pT because it was not found in the dataframe.")
            return
        else:
            logging.debug("Plotting pT")

        column = df.iloc[:, var_index['pt']]

        # bins = find_bin_count(column, binmax=100)
        bins = 100

        plt.hist(
            column, 
            bins=bins, 
            alpha=0.5,
            label='pT',
            color='blue',
            # edgecolor='black',
        )
        plt.yscale(self.yscale)
        plt.title(f"{self.title} pT")
        plt.xlabel("pT (GeV)")
        plt.ylabel("Events / Bin")
        plt.savefig(self.save_plot_path)
        plt.close()


    def plot_eta(self, df, var_index):
        self.save_path('eta.png')

        if 'eta' not in var_index:
            # if find_variables could not find the eta variable, skip plotting
            logging.warning("Not plotting eta because it was not found in the dataframe.")
            return
        else:
            logging.debug("Plotting eta")

        column = df.iloc[:, var_index['eta']]

        # bins = find_bin_count(column)
        bins = 100

        plt.hist(
            column, 
            bins=bins, 
            alpha=0.5,
            label='eta',
            color='blue',
            # edgecolor='black',
        )
        plt.yscale(self.yscale)
        plt.title(f"{self.title} eta")
        plt.xlabel("eta")
        plt.ylabel("Events / Bin")
        plt.savefig(self.save_plot_path)
        plt.close()


    def plot_phi(self, df, var_index):
        self.save_path('phi.png')

        if 'phi' not in var_index:
            # if find_variables could not find the phi variable, skip plotting
            logging.warning("Not plotting phi because it was not found in the dataframe.")
            return
        else:
            logging.debug("Plotting phi")

        column = df.iloc[:, var_index['phi']]

        # bins = find_bin_count(column)
        bins = 100

        plt.hist(
            column, 
            bins=bins, 
            alpha=0.5,
            label='phi',
            color='blue',
            # edgecolor='black',
        )
        plt.yscale(self.yscale)
        plt.title(f"{self.title} phi")
        plt.xlabel("phi")
        plt.ylabel("Events / Bin")
        plt.savefig(self.save_plot_path)
        plt.close()
    

    def plot_E(self, df, var_index):
        self.save_path('E.png')

        if 'e' not in var_index:
            # if find_variables could not find the E variable, skip plotting
            logging.warning("Not plotting E because it was not found in the dataframe.")
            return
        else:
            logging.debug("Plotting E")

        column = df.iloc[:, var_index['e']]

        # bins = find_bin_count(column)
        bins = 100

        plt.hist(
            column, 
            bins=bins, 
            alpha=0.5,
            label='E',
            color='blue',
            # edgecolor='black',
        )
        plt.yscale(self.yscale)
        plt.title(f"{self.title} Energy")
        plt.xlabel("E (GeV)")
        plt.ylabel("Events / Bin")
        plt.savefig(self.save_plot_path)
        plt.close()

# logging output variable string name and name of column being used