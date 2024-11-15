import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from utils.find_bin_count import find_bin_count

class History:
    def __init__(self, config):
        logging.info("History plotter initialized")
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
        self.yscale = yscale
        logging.info("Plotting History")

        if title is not None:
            self.title = title
            logging.info(f"Using plotter title: {self.title}")
        else:
            self.title = None
            logging.info("Using default plotter titles")

        self.plot_loss()
        self.plot_rsquare()

    
    def plot_loss(self):
        self.save_path('loss_history.png')

        # plot the loss
        plt.figure(figsize=(10, 8))
        plt.plot(
            self.df['train_loss'], 
            label=r'Train Loss', 
            marker='o',
            color='red',    
            markerfacecolor='none',
        )
        plt.plot(
            self.df['val_loss'], 
            label=r'Validation Loss', 
            marker='x', 
            color='green',
            markerfacecolor='none',
        )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale(self.yscale)
        plt.legend()
        if self.title is not None:
            plt.title(f"Loss: {self.title}")
        else:
            plt.title("Loss")
        plt.savefig(self.save_plot_path)
        plt.close()


    def plot_rsquare(self):
        self.save_path('rsquare_history.png')

        # plot the loss
        plt.figure(figsize=(10, 8))
        plt.plot(
            self.df['train_rsquare'], 
            label=r'Train R$^2$', 
            marker='o',
            color='red',
            markerfacecolor='none',
        )
        plt.plot(
            self.df['val_rsquare'], 
            label=r'Validation R$^2$', 
            marker='x',
            color='green',
            markerfacecolor='none',
        )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale(self.yscale)
        plt.legend()
        if self.title is not None:
            plt.title(f"Loss: {self.title}")
        else:
            plt.title("Loss")
        plt.savefig(self.save_plot_path)
        plt.close()

    