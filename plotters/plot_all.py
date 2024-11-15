import logging, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.find_bin_count import find_bin_count

class PlotAll:
    def __init__(self, config):
        logging.info('PlotAll plotter initialized')
        self.config = config

        # column: [title, x-axis label]
        self.labels = {
            'el1_pt': [r'Electron 1 $p_T$', r'$p_T$ [GeV]'],
            'el2_pt': [r'Electron 2 $p_T$', r'$p_T$ [GeV]'],
            'el1_eta': [r'Electron 1 $\eta$', r'$\eta$'],
            'el2_eta': [r'Electron 2 $\eta$', r'$\eta$'],
            'el1_phi': [r'Electron 1 $\phi$', r'$\phi$'],
            'el2_phi': [r'Electron 2 $\phi$', r'$\phi$'],
            'el1_E': [r'Electron 1 $E$', r'$E$ [GeV]'],
            'el2_E': [r'Electron 2 $E$', r'$E$ [GeV]'],
            'el1_etas1': [r'Electron 1 First Layer $\eta$', r'electron 1 $\eta_{s1}$'],
            'el2_etas1': [r'Electron 2 First Layer $\eta$', r'electron 2 $\eta_{s1}$'],
            'el1_etas2': [r'Electron 1 Second Layer $\eta$', r'electron 1 $\eta_{s2}$'],
            'el2_etas2': [r'Electron 2 Second Layer $\eta$', r'electron 2 $\eta_{s2}$'],
            'el1_maxEcell_E': [r'Electron 1 Maximum Energy Cell Energy', r'electron 1 $E_{max}$ [GeV]'],
            'el2_maxEcell_E': [r'Electron 2 Maximum Energy Cell Energy', r'electron 2 $E_{max}$ [GeV]'],
            'el1_maxEcell_x': [r'Electron 1 Maximum Energy Cell $x$', r'electron 1 $x_{max}$ [mm]'],
            'el2_maxEcell_x': [r'Electron 2 Maximum Energy Cell $x$', r'electron 2 $x_{max}$ [mm]'],
            'el1_maxEcell_y': [r'Electron 1 Maximum Energy Cell $y$', r'electron 1 $y_{max}$ [mm]'],
            'el2_maxEcell_y': [r'Electron 2 Maximum Energy Cell $y$', r'electron 2 $y_{max}$ [mm]'],
            'el1_maxEcell_z': [r'Electron 1 Maximum Energy Cell $z$', r'electron 1 $z_{max}$ [mm]'],
            'el2_maxEcell_z': [r'Electron 2 Maximum Energy Cell $z$', r'electron 2 $z_{max}$ [mm]'],
            'el1_maxEcell_t': [r'Electron 1 Maximum Energy Cell Time', r'electron 1 $t_{max}$ [ns]'],
            'el2_maxEcell_t': [r'Electron 2 Maximum Energy Cell Time', r'electron 2 $t_{max}$ [ns]'],
            'el1_maxEcell_R': [r'Electron 1 Maximum Energy Cell $R$', r'electron 1 $R_{max}$ [mm]'],
            'el2_maxEcell_R': [r'Electron 2 Maximum Energy Cell $R$', r'electron 2 $R_{max}$ [mm]'],
            'el1_sinh_etas12': [r'Electron 1 Difference of $\sinh(\eta)$ between Layers 1 & 2', r'electron 1 $\sinh(\eta_{s1})-\sinh(\eta_{s2})$'],
            'el2_sinh_etas12': [r'Electron 2 Difference of $\sinh(\eta)$ between Layers 1 & 2', r'electron 2 $\sinh(\eta_{s1})-\sinh(\eta_{s2})$'],
            'el12_maxEcell_R': [r'Difference Between Max E Cell $R$ of Electrons 1 & 2', r'el1_maxEcell_R - el2_maxEcell_R [mm]'],
            'el1_diff_etas12': [r'Electron 1 $\eta$ Difference Between Layers 1 & 2', r'Electron 1 $\eta_{s1} - \eta_{s2}$'],
            'el2_diff_etas12': [r'Electron 2 $\eta$ Difference Between Layers 1 & 2', r'Electron 2 $\eta_{s1} - \eta_{s2}$'],
            'el12_diff_maxEcell_x': [r'Difference Between Max E Cell $x$ of Electrons 1 & 2', r'el1_maxEcell_x - el2_maxEcell_x [mm]'],
            'el12_diff_maxEcell_y': [r'Difference Between Max E Cell $y$ of Electrons 1 & 2', r'el1_maxEcell_y - el2_maxEcell_y [mm]'],
            'el12_diff_maxEcell_z': [r'Difference Between Max E Cell $z$ of Electrons 1 & 2', r'el1_maxEcell_z - el2_maxEcell_z [mm]'],
            'el1_dca': [r'Electron 1 Distance of Closest Approach', r'electron 1 $dca$ [mm]'],
            'el2_dca': [r'Electron 2 Distance of Closest Approach', r'electron 2 $dca$ [mm]'],
            'el1_phis1': [r'Electron 1 First Layer $\phi$', r'electron 1 $\phi_{s1}$'],
            'el2_phis1': [r'Electron 2 First Layer $\phi$', r'electron 2 $\phi_{s1}$'],
            'el1_phis2': [r'Electron 1 Second Layer $\phi$', r'electron 1 $\phi_{s2}$'],
            'el2_phis2': [r'Electron 2 Second Layer $\phi$', r'electron 2 $\phi_{s2}$'],
            'TV_x': [r'Truth Vertex $x$', r'Truth Vertex $x$ [mm]'],
            'TV_y': [r'Truth Vertex $y$', r'Truth Vertex $y$ [mm]'],
            'TV_z': [r'Truth Vertex $z$', r'Truth Vertex $z$ [mm]'],
            'TV_R': [r'Truth Vertex $R$', r'Truth Vertex $R$ [mm]'],
            'el1_f1': [r'Ratio of the First Layer Energy to the Total Cluster Energy', r'electron 1 $f_1$'],
            'el2_f1': [r'Ratio of the First Layer Energy to the Total Cluster Energy', r'electron 2 $f_1$'],
            'el1_f3': [r'Ratio of the Third Layer Energy to the Total Cluster Energy', r'electron 1 $f_3$'],
            'el2_f3': [r'Ratio of the Third Layer Energy to the Total Cluster Energy', r'electron 2 $f_3$'],
        }


    def save_path(self, name: str):
        try:
            assert isinstance(name, str)
        except AssertionError as err:
            logging.error('name must be a string')
        
        if self.pretitle is None:
            self.save_plot_path = os.path.join(
                self.config.base_path,
                self.config.active_plots_path,
                name
            )
        else:
            logging.debug(f'self.pretitle = {self.pretitle}')
            self.save_plot_path = os.path.join(
                self.config.base_path,
                self.config.active_plots_path,
                self.pretitle + '_' + name
            )

    
    def get_label(self, column):
        '''Return the (title, x-axis label) list for a given column according to
        the labels dictionary. If column not found in dictionary, return 
        (column, column). 

        Args:
            column (str): name of the column in the dataframe

        Returns:
            tuple: (plot title, x-axis label)
        '''

        try:
            assert isinstance(column, str)
        except AssertionError as err:
            logging.error('column must be a string')
            logging.debug(f'column = {column}')
            logging.debug(f'type(column) = {type(column)}')
            logging.error('Exiting...')
            sys.exit(1)

        if column in self.labels:
            return tuple(self.labels[column])
        else:
            logging.debug(f'Could not find label for {column}')
            return (column, column)
    

    def plot(self, df: pd.DataFrame, pretitle=None, yscale='linear', filetype='png'):
        self.df = df
        self.yscale = yscale
        self.pretitle = pretitle
        logging.info('Plotting All Columns in DataFrame...')

        logging.debug(f'Columns to plot: {df.columns}')

        for column in df.columns:
            logging.debug(f'Now plotting {column} ...')
            self.save_path(column + '.' + filetype)
            logging.debug(f'Saving to {self.save_plot_path}')
            self.default_plot(df, column, pretitle)
            
            logging.debug('Finished plotting {column}')



    def default_plot(self, df: pd.DataFrame, column: str, pretitle: str):
        # column is the name of the column in df to plot

        try:
            assert isinstance(column, str)
        except AssertionError as err:
            logging.error('column must be a string')
            logging.debug(f'column = {column}')
            logging.debug(f'type(column) = {type(column)}')
            logging.error('Skipping this instance of default_plot...')
            return 

        # bins = find_bin_count(df[column], binmax=100)
        bins = 100

        (auto_title, self.xlabel) = self.get_label(column)

        if pretitle is not None:
                self.title = pretitle + ' ' + auto_title
                logging.debug(f'Using plotter pretitle: {pretitle}')
        else:
            self.title = auto_title
            logging.debug('Using default plotter titles')

        plt.figure()
        plt.hist(
            df[column],
            bins=bins,
            label=column,
            # alpha=0.5,

        )
        plt.yscale(self.yscale)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel('Events/Bin')
        plt.legend()
        plt.savefig(self.save_plot_path)
        plt.close()