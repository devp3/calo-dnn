import logging, sys
import numpy as np
import pandas as pd

def find_bin_count(column:pd.Series, binmax=100):
    """
    Find the optimal number of bins for a histogram using the Freedman-Diaconis rule.

    Parameters:
        column (pandas.Series or pandas.DataFrame): The column to find the optimal number of bins for.

    Returns:
        bins (int): The optimal number of bins for the histogram.
    """
    if not isinstance(column, pd.Series) and not isinstance(column, pd.DataFrame):
        logging.critical("Input to find_bin_count must be a pandas Series or DataFrame")
        exit(1)

    min = float(column.min())
    max = float(column.max())
    IQR = float(column.quantile(0.75) - column.quantile(0.25))
    n = len(column)     # number of data points

    # Use the Freedman-Diaconis rule to find the optimal bin width
    h = np.abs(2 * IQR * n**(-1/3)) # bin width

    bins = round((max - min) / h)  # number of bins, round to nearest integer

    
    if bins > binmax:
        logging.debug(f'Desired bin count of {bins} is too large, setting bins = {binmax}')
        logging.debug(f"desired bin count = {bins}")
        logging.debug(f"forced bin count = {binmax}")
        bins = binmax

    logging.debug(f"bin count = {bins}")
    logging.debug(f"type(bins) = {type(bins)}")

    return bins


