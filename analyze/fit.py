import os, logging, sys
import scipy
import scipy.stats as stats
import numpy as np
import pandas as pd


class Fit:
    def __init__(self):
        logging.debug("Fit initialized")
        self.param_cov = None
        self.param_optimize = None
        self.p0 = None
        self.data = None
        self.curve = None
        self.bins = None



    def fit(self, data, p0='auto', curve='gaussian', bins=50):
        """Fit a distribution to the data and return the optimal parameters and
        covariance matrix. The fit is performed using scipy.optimize.curve_fit.

        Args:
            data (list-like): A flat list-like object of the input data. 
            p0 (list, str, optional): Intial parameter guess. Defaults to 'auto'. (A, mu, sigma)
            curve (str, optional): Type of curve to be fitted. Defaults to 'gaussian'.
            bins (int, optional): Number of bins to be used in fitting procedure. Defaults to 50.

        Returns:
            tuple: optimal parameters, covariance matrix, x_hist bin centers (np.ndarray)
        """
        self.data = data
        self.p0 = p0
        self.curve = curve
        self.bins = bins

        # data.to_csv('x_data.csv')

        if isinstance(data, pd.DataFrame):
            x_data = data.to_numpy()
            logging.debug("Data is a pandas DataFrame")
        elif isinstance(data, pd.Series):
            x_data = data.to_numpy()
            logging.debug("Data is a pandas Series")
        elif isinstance(data, np.ndarray) or isinstance(data, list):
            x_data = data
            logging.debug("Data is a list or numpy array")
        else:
            logging.debug("Data is not a pandas DataFrame, Series, list, or numpy array.")
            logging.debug("Accepting data for now.")
            x_data = data
        
        # np.savetxt('x_data.txt', x_data)

        hist, bin_edges = np.histogram(x_data, bins=bins)
        hist = hist/sum(hist)

        n = len(hist)
        x_hist = np.zeros((n), dtype=float)
        for i in range(n):
            x_hist[i] = (bin_edges[i+1] + bin_edges[i])/2

        y_hist = hist

        if curve == 'gaussian':
            self.fit_function = self.gauss

        if p0 == 'auto':
            # get initial guesses for parameters as self.p0
            self.get_p0(x_hist, y_hist) 

        self.param_optimize, self.param_cov = scipy.optimize.curve_fit(
            self.fit_function,
            x_hist,
            y_hist,
            p0=self.p0,
            maxfev=5000,
        )
        return self.param_optimize, self.param_cov, x_hist
    


    def predict(self, x: float):
        """Output the predicted value of the fitted function at x. Uses the 
        optimal parameters from the previously-called fit.

        Args:
            x (float): Input value to the fitted function.

        Returns:
            float: Predicted value of the fitted function at x.
        """
        try:
            assert self.param_optimize is not None
        except AssertionError as err:
            logging.error("No fit has been performed. Please run fit() first.")
            raise err
        
        return self.fit_function(x, *self.param_optimize)



    def get_p0(self, x_hist: np.ndarray, y_hist: np.ndarray):
        """Generates 

        Args:
            x_hist (np.ndarray): _description_
            y_hist (np.ndarray): _description_
        """        

        if self.curve == 'gaussian':

            max_y = max(y_hist)
            mean0 = sum(x_hist*y_hist)/sum(y_hist)                  
            sigma0 = sum(y_hist*(x_hist-mean0)**2)/sum(y_hist) 

            self.p0 = [max_y, mean0, sigma0]



    @staticmethod
    def gauss(x: float, A: float, mu: float, sigma: float): # p is a list of parameters
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
        # return A/np.sqrt(sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))
    
