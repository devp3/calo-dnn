import os, logging, sys, datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import seaborn as sns
import pickle, scipy, math, itertools

from utils.handler import Handler
from utils.find_bin_count import find_bin_count
from analyze.fit import Fit
import functions.normalize.linear as linear
from data_loader.example_data_loader import DatasetGenerator

# This plotter will generate several key performance plots
# 1. Target vs. Predicted value


# A better way to handle plot naming and titling is to define a base_name for 
# each kind of plot, and then define title, post_title, pretitle, and extension, 
# as variables, like self.pretitle and self.title, then have functions that
# assemble the file name and plot titles from these, handling things like
# spaces vs. underscores, etc. differently. 

# There is a better way to do performance plotting. In the main or plot script:
# import plotters.performance as Performance
# performance = Performance(config) # initialize performance plotter
# performance.plot_target_vs_predicted() # plot target vs. predicted
# performance.plot_total_residuals()
# performance.plot_feature_importance()



class Performance:
    def __init__(
            self, 
            config, 
            normalization_factors: pd.DataFrame,
            **kwargs
        ):
        logging.info("Performance plotter initialized")
        self.config = config
        self.normalization_factors = normalization_factors

        self.target_list = list(self.config.targets)
        self.target_list = [item for item in self.target_list if item not in self.config.ignore_train]

        self.loader = DatasetGenerator(self.config)
        self.diagnostics_path = os.path.join(
            self.config.base_path,
            self.config.active_plots_path,
            'diagnostics'
        )
        self._initialize_diagnostics()

        allowed_kwargs = {
            'name',
            'title',
            'data',
            'model',
        }


    def _initialize_diagnostics(self):
        if not os.path.exists(self.diagnostics_path):
            os.makedirs(self.diagnostics_path)


    def _initialize_feature_vs_residuals(self):
        fvr_path = os.path.join(
            self.config.base_path,
            self.config.active_plots_path,
            'feature_vs_residuals'
        )
        if 'feature_vs_residuals' not in self.avoid_plotting:
            if not os.path.exists(fvr_path):
                os.makedirs(fvr_path)

    
    def _drop_ignore_train(self, df: pd.DataFrame):
        df = df.drop(columns=self.config.ignore_train, errors='ignore')
        return df


    def save_path(self, name, is_diagnostic=False):
        try:
            assert isinstance(name, str)
        except AssertionError as err:
            logging.error("name must be a string")

        if self.post_title != None:
            post_title = self.post_title
            post_title = post_title.replace(' ', '_')
            if name.find('.') != -1: # name has an extension
                name_parts = name.split('.')
                name_parts.insert(1, post_title)
                name_parts.insert(1, '_')
                name_parts.insert(-1, '.')
                name = ''.join(name_parts)
            else:                   # name has no extension
                name = f'{name}_{post_title}'

        if is_diagnostic:
            self.save_plot_path = os.path.join(
                self.diagnostics_path,
                name
            )

        else:
            self.save_plot_path = os.path.join(
                self.config.base_path,
                self.config.active_plots_path,
                name
            )


    
    def denormalize(self, df: pd.DataFrame, denorm_factors: pd.DataFrame):
        # denormalize the columns of data in df using the denorm_factors
        return linear.denorm(df, denorm_factors)
        


    def predict(self, features: pd.DataFrame):
        # the second features is just a placeholder needed for preprocess
        tensors = next(self.loader.preprocess((features, features), shuffle=False))

        prediction_keys = self.test_targets.keys()
        prediction_keys = [item for item in prediction_keys if item not in self.config.ignore_train]

        predictions = pd.DataFrame(
            columns=prediction_keys,
        )
        for x_batch, y_batch in tensors:
            batch_pred = self.model(x_batch, training=False)
            batch_pred = batch_pred.numpy()
            batch_pred = pd.DataFrame(
                batch_pred, 
                columns=self.target_list, 
            )
            predictions = pd.concat([predictions, batch_pred], ignore_index=True)

        predictions = predictions.set_index(features.index)
            
        return predictions


    def plot_all(self, title=None, yscale='linear', norm='linear'):

        self.plot_target_vs_predicted()     # target vs. predicted value 2D hist
        self.plot_total_residuals()         # residuals histogram w/ fitted guassian
        self.plot_binned_residuals()        # residuals histogram binned by target value
        self.plot_feature_importance()      # feature importance plot
        self.plot_feature_vs_residuals()    # feature vs. residuals 2d hist


    def _validate_avoid_plotting(self):
        allowed = [
            'target_vs_predicted', 
            'total_residuals',
            'binned_residuals',
            'feature_importance',
            'feature_vs_residuals',
            'target_combinations',
            'zR_residuals',
            'cross_feature_importance',
        ]
        if isinstance(self.avoid_plotting, list):
            pass
        elif isinstance(self.avoid_plotting, type):
            pass
        elif isinstance(self.avoid_plotting, str):
            pass
        elif self.avoid_plotting == None:
            pass
        else: 
            logging.error('Invalid input for avoid_plotting argument in performance.')
            sys.exit(1)
        
        if any(True for item in self.avoid_plotting if item not in allowed):
            logging.warning(f'Some elements of avoid_plotting are not allowed.')
            logging.debug(f'self.avoid_plotting: {self.avoid_plotting}')
            logging.debug(f'allowed: {allowed}')
        

    def plot(
            self, 
            model: tf.keras.Model, 
            test_data: tuple, 
            title=None, 
            post_title=None,
            yscale='linear', 
            norm='linear',
            plot_diagnostics=True,      # plot the detail plots
            avoid_plotting=[], 
            comparison_xlim=None,
            comparison_ylim=None,
        ):
        self.model = model
        self.test_data = test_data
        self.yscale = yscale
        self.post_title = post_title
        self.plot_diagnostics = plot_diagnostics
        self.avoid_plotting = avoid_plotting 
        self.comparison_xlim = comparison_xlim
        self.comparison_ylim = comparison_ylim

        # test_data is all test data (features and targets)
        # test_features is all feature columns of test data
        # test_targets is all target columns of test data

        self._validate_avoid_plotting()
        self._initialize_feature_vs_residuals()

        self.test_features = self.test_data[0].copy()
        self.test_targets = self.test_data[1].copy()

        self.test_data_tensors = next(self.loader.preprocess((self.test_features, self.test_targets), shuffle=False))
        

        self.test_targets = self.denormalize(self.test_targets, self.normalization_factors)
        self.test_features = self.denormalize(self.test_features, self.normalization_factors)

        # logging.debug(f'shape of test_features: {self.test_features.shape}')

        # self.test_features = tf.data.Dataset.from_tensor_slices(self.test_features.values)
        # self.test_features = self.test_features.batch(self.config.batch_size)

        prediction_keys = self.test_targets.keys()
        prediction_keys = [item for item in prediction_keys if item not in self.config.ignore_train]

        self.predictions = pd.DataFrame(
            columns=prediction_keys,
            # index=self.test_targets.index,
        )
        logging.debug(f'(plot) self.predictions.head(): {self.predictions.head()}')

        for x_batch, y_batch in self.test_data_tensors:
            predictions = self.model(x_batch, training=False)
            predictions = predictions.numpy()
            # logging.debug(f'predictions[0]: {predictions[0]}')
            predictions = pd.DataFrame(
                predictions, 
                columns=prediction_keys, 
            )
            self.predictions = pd.concat([self.predictions, predictions], ignore_index=True)

        self.predictions = self.predictions.set_index(self.test_targets.index)

        # logging.debug(f'self.predictions.head(): {self.predictions.head()}')
        # logging.debug(f'self.predictions.describe(): {self.predictions.describe()}')

        self.predictions = self.denormalize(self.predictions, self.normalization_factors)

        combined = pd.concat([self.test_targets, self.predictions], axis=1)

        combined.to_csv(os.path.join(self.config.base_path, 'predictions.csv'))

        logging.info("Plotting Performance")

        if 'target_vs_predicted' not in self.avoid_plotting:
            self.plot_target_vs_predicted()     # target vs. predicted value 2D hist
        if 'total_residuals' not in self.avoid_plotting:
            self.plot_total_residuals()         # residuals histogram w/ fitted guassian
        if 'binned_residuals' not in self.avoid_plotting:
            self.plot_binned_residuals()        # residuals histogram binned by target value
        if 'feature_importance' not in self.avoid_plotting:
            self.plot_feature_importance()      # feature importance plot
        if 'feature_vs_residuals' not in self.avoid_plotting:
            self.plot_feature_vs_residuals()    # feature vs. residuals 2d hist
        if 'target_combinations' not in self.avoid_plotting:
            self.plot_target_combinations()    # target combinations 2d hist
        if 'zR_residuals' not in self.avoid_plotting:
            self.plot_zR_residuals()            # zR residuals and other plots
        # if 'cross_feature_importance' not in self.avoid_plotting:
        #     self.plot_cross_feature_importance()

        

    def model_predict(self, test_features: pd.DataFrame, model=None):
        """Returns the predictions of the model for the given test features. Inp

        Args:
            test_features (pd.DataFrame): dataframe of test features 

        Returns:
            _type_: _description_
        """
        if model == None:
            return self.model(test_features)
        else:
            return model(test_features)



    def get_residuals(self, targets: pd.DataFrame, predictions: pd.DataFrame):
        """Computes residuals for between two dataframes of targets and 
        prediction (target - prediction). 

        Args:
            targets (pd.DataFrame): true target values
            predictions (pd.DataFrame): predicted values

        Returns:
            pd.DataFrame: residuals
        """
        logging.debug(f'(get_residuals) targets.head(): {targets.head()}')
        logging.debug(f'(get_residuals) predictions.head(): {predictions.head()}')

        res = targets - predictions

        logging.debug(f'(get_residuals) res.head(): {res.head()}')
        return res



    def plot_target_vs_predicted(self):
        """Plots a 2D histogram of the true vs. predicted values for each target
        variable. 
        """        

        for target in self.target_list:
            # plot_min = min(self.test_targets[target].min(), self.predictions[target].min())
            # plot_max = max(self.test_targets[target].max(), self.predictions[target].max())
            # plot_edge = min(abs(plot_min), abs(plot_max))

            xup = self.test_targets[target].max()
            xdown = self.test_targets[target].min()
            yup = self.predictions[target].max()
            ydown = self.predictions[target].min()

            xedge = min(abs(xup), abs(xdown))
            yedge = min(abs(yup), abs(ydown))

            xedge = min(xedge, yedge)
            yedge = xedge

            self.save_path(f'target_vs_predicted_{target}.png')

            logging.debug(f'(plot_target_vs_predicted) Target column description: {self.test_targets[target].describe()}')
            logging.debug(f'(plot_target_vs_predicted) Target column head: {self.test_targets[target].head()}')
            logging.debug(f'(plot_target_vs_predicted) Predicted column description: {self.predictions[target].describe()}')
            logging.debug(f'(plot_target_vs_predicted) Predicted column head: {self.predictions[target].head()}')

            # logging.debug(f'number of nan in self.test_target[{target}]: {self.test_targets[target].isna().sum()}')
            # logging.debug(f'number of nan in self.predictions[{target}]: {self.predictions[target].isna().sum()}')

            if self.post_title == None: 
                post_title_string = ''
            else:
                post_title_string = rf' ({self.post_title})'
            title_string = rf'True vs. Predicted {target}{post_title_string}'

            plt.figure(figsize=(10, 8))
            plt.hist2d(
                self.test_targets[target],
                self.predictions[target],
                bins=(50,50),
            )
            plt.plot(np.linspace(- xedge, xedge), np.linspace(- yedge, yedge), color='r', linestyle='--')
            plt.xlabel(rf'True {target}')
            plt.ylabel(rf'Predicted {target}')
            plt.title(title_string)
            plt.colorbar(label='Events/Bin')
            plt.xlim(- xedge, xedge)
            plt.ylim(- yedge, yedge)
            plt.savefig(self.save_plot_path)
            plt.close()



    def plot_total_residuals(self):

        fit = Fit()

        for target in self.target_list:
            plot_bins = 250

            self.save_path(f'total_residuals_{target}.png')

            x_data = self.get_residuals(self.test_targets[target], self.predictions[target])

            logging.debug(f'(plot_total_residuals) for target {target}, x_data.describe(): \n{x_data.describe()}')
            logging.debug(f'(plot_total_residuals) for target {target}, x_data.head(): \n{x_data.head()}')

            # x_data.to_csv('total_residuals.csv')
            # self.test_targets.to_csv('targets.csv')
            # self.predictions.to_csv('predictions.csv')
            # self.test_features.to_csv('features.csv')

            param_optimized, param_cov, x_hist = fit.fit(x_data, bins=plot_bins, p0=[0.1, 0, 100])

            weights = np.ones_like(x_data)/len(x_data)
            x_hist_smooth = np.linspace(min(x_hist), max(x_hist), 500)

            if self.post_title == None:
                post_title_string = ''
            else:
                post_title_string = rf' ({self.post_title})'
            title_string = rf'Total Residuals for {target}{post_title_string}'

            fig, ax = plt.subplots(figsize=(10, 8))
            plt.hist(x_data, weights=weights, bins=plot_bins)
            plt.plot(
                x_hist_smooth,
                fit.predict(x_hist_smooth), # anticipating needing to add loop over vector input
                'r.:',
                # kde=False,
                # fit=stats.norm,
                # fit_kws={'color': 'red'},
                label='Gaussian Fit',
            )
            textstr = '\n'.join((
                r'$\mu=%.2f$   mm' % (param_optimized[-2], ),
                r'$\sigma=%.2f$   mm' % (np.abs(param_optimized[-1]), ),
                rf'$n$ = {len(x_data)} events',
                ))
            props = dict(
                boxstyle='round', 
                facecolor='white', 
                alpha=0.5
                )
            ax.text(0.05, 0.95, textstr, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
            plt.xlabel(rf'Residuals of {target} (mm)')
            plt.ylabel(rf'Count/Bin')
            plt.title(title_string)
            plt.savefig(self.save_plot_path)
            plt.close()

    
    def plot_binned_residuals(self):
        # TODO: This is bad practice for OOP. 
        #       There should be some other way to dynamically input these values
        #       from a separate python variable or saved data file somewhere
        #       else. Maybe something in the /configs/ directory?
        # pointing_bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]#, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380] #mm
        # pointing_bins = [200, 220, 240, 260, 280, 300, 320, 340, 360, 380] #mm
        # TODO: Make this pointing bin parameters, (min, max, step), a config or class parameter
        pointing_bins = list(range(
            self.config.pointing_bins['lower'], 
            self.config.pointing_bins['upper'],
            self.config.pointing_bins['step']
        )) #mm

        # pointing_bins = [0.0, 20., 40., 60, 80., 100.,120.,  300.,  500., 2000.]

        pointing_resolution_zee_mc = [12.04277, 12.36026, 13.29376, 15.05451, 17.49589, 19.43386, 22.89452] #mm
        pointing_mean_zee_mc = [9.725815, 29.1946, 48.67379, 68.20341, 87.74008, 107.2668, 130.0455] #mm

        # pointing_mean_zee_mc = np.linspace(9, 700, 20)
        # pointing_resolution_zee_mc = [15+(item**2) * (175/(700)**2) for item in pointing_mean_zee_mc]

        # add DNN Zee pointing resolution

        fit = Fit()

        fit_info = pd.DataFrame(
                columns=[
                    'target', 'measure_target',
                    'bin_lower', 'bin_upper', 
                    'A', 'A_uncertainty',
                    'mu', 'mu_uncertainty',
                    'sigma', 'sigma_uncertainty', 
                    'mean_target', 'mean_target_uncertainty',
                    'bin_population'
                ]
            )

        # "target" is what we're plotting on the x-axis. one plot per "target"
        # "measure_target" are the series of residuals we're fitting (items in plot legend).
        #   one data point per "measure_target" per "target" bin per "target" plot

        for target in self.target_list:
            # We will create plots using bins of <target> on the x-axis.
            # Then we will slice the data by bins of <target>.
            # Then, for each bin of <target>, we will fit the residuals for all 
            #   that fall into that bin of <target> for each target. 
            for measure_target in self.target_list:     # the target we are measuring the residuals of
    
                for i in range(len(pointing_bins) - 1):
                    lower = pointing_bins[i]
                    upper = pointing_bins[i + 1]

                    logging.debug(f'(plot_binned_residuals) In plot_binned_residuals, starting {target} binned from {lower} to {upper} mm')



                    x_data = self.get_residuals(self.test_targets[measure_target], self.predictions[measure_target])

                    # make df of events that fall into the <target> bin to find their indices
                    target_in_bin = self.test_targets[target]
                    target_in_bin = target_in_bin[(target_in_bin >= lower) & (target_in_bin < upper)]

                    x_data = x_data.loc[target_in_bin.index]
                    logging.debug(f'(plot_binned_residuals) length of x_data: {len(x_data)}')

                    param_optimized, param_cov, x_hist = fit.fit(x_data, p0=[1, 0, 100], bins=150)
                    logging.debug(f'(plot_binned_residuals) param_optimized: \n{param_optimized}')
                    logging.debug(f'(plot_binned_residuals) param_cov: \n{param_cov}')
                    logging.debug(f'(plot_binned_residuals) Uncertainty on A: {np.sqrt(param_cov[0][0])}')
                    logging.debug(f'(plot_binned_residuals) Uncertainty on mu: {np.sqrt(param_cov[1][1])}')
                    logging.debug(f'(plot_binned_residuals) Uncertainty on sigma: {np.sqrt(param_cov[2][2])}')

                    fit_info = pd.concat([
                        fit_info, 
                        pd.DataFrame([[
                            target, measure_target,
                            lower, upper,
                            param_optimized[0], np.sqrt(param_cov[0][0]),
                            param_optimized[1], np.sqrt(param_cov[1][1]),
                            np.abs(param_optimized[2]), np.sqrt(param_cov[2][2]),
                            target_in_bin.mean(), target_in_bin.std(),
                            len(x_data)
                        ]], 
                        columns=[
                            'target', 'measure_target',
                            'bin_lower', 'bin_upper', 
                            'A', 'A_uncertainty',
                            'mu', 'mu_uncertainty',
                            'sigma', 'sigma_uncertainty', 
                            'mean_target', 'mean_target_uncertainty',
                            'bin_population'
                        ]
                        )
                    ],
                        ignore_index=True
                    )

                    # pseudocode:
                    # [event for event in test_targets[measure_target] if test_targets[target][event.index] between lower and upper]
                    self.save_path(
                        f'binned_residuals_of_{measure_target}_for_{target}_in_{lower}-{upper}.png',
                        is_diagnostic=True,     # this is a diagnostic plot
                    )

                    weights = np.ones_like(x_data)/len(x_data)

                    if self.plot_diagnostics:
                        if self.post_title == None:
                            post_title_string = ''
                        else:
                            post_title_string = f' ({self.post_title})'
                        title_string = rf'Residuals of {measure_target} for events with {target} between {lower} and {upper} mm{post_title_string}'

                        fig, ax = plt.subplots(figsize=(10, 8))
                        x_hist_smooth = np.linspace(min(x_hist), max(x_hist), 500)
                        plt.hist(x_data, weights=weights, bins=150)
                        plt.plot(
                            x_hist_smooth,
                            fit.predict(x_hist_smooth), 
                            'r.:',
                            label='Gaussian Fit',
                        )
                        textstr = '\n'.join((
                            r'$\mu=%.2f$' % (param_optimized[-2], ),
                            r'$\sigma=%.2f$' % (np.abs(param_optimized[-1]), ),
                            rf'$n$ = {len(x_data)} events',

                            ))
                        props = dict(
                            boxstyle='round', 
                            facecolor='white', 
                            alpha=0.5
                            )
                        ax.text(
                            0.05, 0.95, 
                            textstr, fontsize=14, transform=ax.transAxes, 
                            verticalalignment='top', bbox=props
                        )
                        plt.xlabel(rf'Residuals of {measure_target} (mm)')
                        plt.ylabel(rf'Count/Bin')
                        plt.title(title_string)
                        plt.savefig(self.save_plot_path)
                        plt.close()

            # plotting comparison of pointing resolution and z average

            self.save_path(f'comparison_pointing_resolution_vs_{target}_average.png')

            if self.post_title == None:
                post_title_string = ''
            else:
                post_title_string = f' ({self.post_title})'
            title_string = f'Resolution vs {target} Average{post_title_string}'

            color_list = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

            plt.figure(figsize=(10, 8))
            for i, measure_target in enumerate(self.target_list):
                fi = fit_info.loc[fit_info['measure_target'] == measure_target]
                fi = fi.loc[fi['target'] == target]
                plt.errorbar(
                    fi['mean_target'],
                    fi['sigma'], 
                    xerr=(fi['bin_upper'] - fi['bin_lower'])/2,
                    yerr=fi['sigma_uncertainty'],
                    capsize=3,
                    fmt='none',     # 'o' for points
                    label=f'DNN {fi["measure_target"].iloc[0]}',
                    color=color_list[i+1],
                )
            if target == 'TV_z':
                plt.scatter(
                    pointing_mean_zee_mc,
                    pointing_resolution_zee_mc,
                    label='Zee MC',
                    color=color_list[0],
                )
            plt.legend()
            plt.xlabel(f'{target} average (mm)')
            plt.ylabel('Resolution (mm)')
            if self.comparison_xlim != None:
                plt.xlim(self.comparison_xlim[0], self.comparison_xlim[1])
            if self.comparison_ylim != None:
                plt.ylim(self.comparison_ylim[0], self.comparison_ylim[1])
            plt.title(title_string)
            plt.grid(True, color = "grey", linewidth = "1.4", linestyle = "dotted")
            plt.savefig(self.save_plot_path)
            plt.close()

            
            fit_info.to_csv(
                os.path.join(
                    self.config.base_path,
                    self.config.active_data_path,
                    f'binned_residuals_{target}.csv'
                )
            )



    def plot_feature_importance(self):

        x_test = self.test_data[0]  # initialize unnormed test data
        y_test = self.test_data[1]

        x_test = self._drop_ignore_train(x_test)
        y_test = self._drop_ignore_train(y_test)

        metrics = pd.DataFrame(
            columns=['feature', 'target', 'mse', 'r_squared']
        )

        fit = Fit()

        # remember x -> features, y -> targets

        for target in self.target_list:
            for feature in x_test.columns:

                x_test_rand = x_test.copy()         # x_test with one row randomized
                # x_test_rand = x_test_rand.sample(frac=1)#.reset_index(drop=True)

                logging.debug(f'(plot_feature_importance) x_test_rand: \n{x_test_rand.head()}')
                logging.debug(f'(plot_feature_importance) y_test: \n{y_test.head()}')

                test_targets_small = y_test.loc[x_test_rand.index]

                x_test_rand[feature] = np.random.permutation(x_test_rand[feature].values)

                logging.debug(f'(plot_feature_importance) the column {target} of x_test_rand has been randomized')
                logging.debug(f'(plot_feature_importance) x_test_rand: \n{x_test_rand.head()}')

                test_predictions_randomized = self.predict(x_test_rand)
                logging.debug(f'(plot_feature_importance) test_predictions_randomized: \n{test_predictions_randomized.head()}')

                logging.debug(f'(plot_feature_importance) test_targets_small: \n{test_targets_small.head()}')

                # test_predictions_randomized = test_predictions_randomized.set_index(test_targets_small.index)
                # logging.debug(f'test_predictions_randomized: \n{test_predictions_randomized.head()}')

                error = self.get_residuals(test_targets_small[target], test_predictions_randomized[target])
                logging.debug(f'(plot_feature_importance) error: \n{error.head()}')

                if self.plot_diagnostics:
                    if self.post_title == None:
                        post_title_string = ''
                    else:
                        post_title_string = f' ({self.post_title})'
                    title_string = rf'Residuals of {target} with {feature} randomized(mm){post_title_string}'

                    self.save_path(
                        f'error_hist_{target}_with_{feature}_randomized.png',
                        is_diagnostic=True,     # this is a diagnostic plot
                    )

                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.hist(error, bins=100)
                    plt.xlabel(rf'Residuals of {target} with {feature} randomized(mm)')
                    plt.ylabel(rf'Count/Bin')
                    plt.title(title_string)
                    # plt.xlim(-800, 800)
                    plt.savefig(self.save_plot_path)
                    plt.close()

                # param_optimized, param_cov, x_hist = fit.fit(error, p0=[4000, 0, 200], bins=100)

                # mse = param_optimized[2]

                mse = float((error**2).mean())
                logging.debug(f'(plot_feature_importance) mse: {mse}')

                x = test_targets_small[target]
                y = test_predictions_randomized[target]

                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

                r_squared = r_value**2

                # corr_matrix = np.corrcoef(test_targets_small[target], test_predictions_randomized[target])
                # corr = corr_matrix[0,1]
                # r_squared = corr**2
                logging.debug(f'(plot_feature_importance) r_squared: {r_squared}')

                if self.plot_diagnostics:
                    self.save_path(
                        f'predictions_{target}_with_{feature}_randomized.png',
                        is_diagnostic=True,     # this is a diagnostic plot
                    )

                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.hist2d(x, y, bins=(100,100))
                    plt.xlabel(rf'True targets')
                    plt.ylabel(rf'Predictions with {feature} randomized')
                    plt.title(rf'Predictions with {feature} randomized vs True targets')
                    # plt.xlim(-800, 800)
                    plt.savefig(self.save_plot_path)
                    plt.close()

                # r_squared_numerator = float((error**2).sum())
                # r_squared_denominator = float(((test_targets_small[target] - test_targets_small[target].mean()).sum())**2)
                # logging.debug(f'r_squared_numerator: {r_squared_numerator}')
                # logging.debug(f'r_squared_denominator: {r_squared_denominator}')

                # r_squared = 1 - (r_squared_numerator / r_squared_denominator)
                
                metrics = pd.concat(
                    [
                        metrics,
                        pd.DataFrame([[feature, target, mse, r_squared]], 
                                     columns=['feature', 'target', 'mse', 'r_squared'])
                    ], 
                    ignore_index=True
                )

        metrics.to_csv(
            os.path.join(
                self.config.base_path,
                self.config.active_data_path,
                "feature_importance_metrics.csv"
            )
        )

        for target in self.target_list:
            # evaluating change in performance for each feature
            self.save_path(f'feature_importance_MSE_{target}.png')

            if self.post_title == None:
                post_title_string = ''
            else:
                post_title_string = f' ({self.post_title})'
            title_string = f'Feature Importance for {target} Comparing MSE{post_title_string}'

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                x_test.columns, 
                metrics[metrics['target'] == target]['mse'].values,
            )
            plt.xlabel('MSE when feature is randomized')
            plt.ylabel('Feature')
            plt.title(title_string)
            plt.savefig(self.save_plot_path)
            plt.close()

            self.save_path(f'feature_importance_R_squared_{target}.png')

            if self.post_title == None:
                post_title_string = ''
            else:
                post_title_string = f' ({self.post_title})'
            title_string = f'Feature Importance for {target} Comparing R Squared{post_title_string}'

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                x_test.columns,
                metrics[metrics['target'] == target]['r_squared'].values,
            )
            plt.xlabel(rf'$R^2$ when feature is randomized')
            plt.ylabel('Feature')
            plt.title(title_string)
            plt.savefig(self.save_plot_path)
            plt.close()

            
    def plot_feature_vs_residuals(self):

        for target in self.target_list:
        
            x_data = self.get_residuals(self.test_targets[target], self.predictions[target])

            for f in self.test_features.columns:
                self.save_path(f'feature_vs_residuals/{f}_vs_log_abs_{target}_residuals.png')

                plt.figure(figsize=(10,10))
                plt.hist2d(np.log10(abs(x_data)), self.test_features[f], bins=(100,100))
                plt.xlabel(rf'$\log10(abs({target} \ residuals))$')
                plt.ylabel(f'{f}')
                plt.colorbar(label='Events/Bin')
                plt.savefig(self.save_plot_path)
                plt.close()



    def randomize_feature(self, x_test: pd.DataFrame, *features):
        """_summary_

        Args:
            x_test (pd.DataFrame): features dataframe to have randomization applied
            *features (str): arbitrary number of features to randomize, 'none' to skip

        Returns:
            pd.DataFrame: features dataframe with randomization applied
        """

        x_test = x_test.copy()

        logging.debug(f'(randomize_feature) features: {features}')

        for feature in features:
            if feature == 'none':
                continue            # skip if feature is 'none'
            x_test[feature] = np.random.permutation(x_test[feature].values)

        return x_test
    

    def heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw=None, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if ax is None:
            ax = plt.gca()

        if cbar_kw is None:
            cbar_kw = {}

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=False, bottom=True,
                    labeltop=False, labelbottom=True)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar


    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                        textcolors=("black", "white"),
                        threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts



    # Print iterations progress
    def printProgressBar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()



    def _make_target_combinations(self):
        return list(itertools.combinations(self.target_list, 2))



    def plot_target_combinations(self):
        if len(self.target_list) < 2:
            # No need to plot if there is only one target
            return

        combinations = self._make_target_combinations()

        if len(combinations) > 20:
            logging.warning(f'Number of target combinations is {len(combinations)}.')
        
        for i, (target1, target2) in enumerate(combinations):
            self.save_path(f'targets_{target1}_vs_{target2}_2D_hist.png')

            y1 = self.test_targets[target1]
            y2 = self.test_targets[target2]

            plt.figure(figsize=(10,8))
            plt.hist2d(y1, y2, bins=100, cmap='viridis', norm=mpl.colors.LogNorm())
            plt.xlabel(rf'{target1}')
            plt.ylabel(f'{target2}')
            plt.colorbar(label='Events/Bin')
            plt.savefig(self.save_plot_path)
            plt.close()




    def plot_zR_residuals(self):
        if 'TV_z' not in self.target_list:
            return
        if 'TV_R' not in self.target_list:
            return

        rho_target = np.sqrt(self.test_targets['TV_R']**2 + self.test_targets['TV_z']**2)
        rho_predictions = np.sqrt(self.predictions['TV_R']**2 + self.predictions['TV_z']**2)
        rho_residuals = self.get_residuals(rho_target, rho_predictions)

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'$\rho$ Residuals{post_title_string}'

        self.save_path(f'rho_residuals.png')
        plt.figure(figsize=(10,8))
        plt.hist(rho_residuals, bins=100)
        plt.xlabel(r'$\rho$ Residuals (mm)')
        plt.ylabel('Events/Bin')
        plt.title(title_string)
        plt.yscale('log')
        plt.savefig(self.save_plot_path)
        plt.close()

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'$\rho$ Predicted vs. $\rho$ True{post_title_string}'

        self.save_path(f'rho_predicted_vs_rho_true.png')

        plt.figure(figsize=(10,8))
        plt.hist2d(rho_target, rho_predictions, norm=mpl.colors.LogNorm(), bins=(100, 100), cmap='viridis')
        plt.title(title_string)
        plt.xlabel(r'$\rho$ True')
        plt.ylabel(r'$\rho$ Predicted')
        plt.colorbar(label='Events/Bin')
        plt.savefig(self.save_plot_path)
        plt.close()

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'$\rho$ True vs. $\rho$ Residuals{post_title_string}'

        self.save_path(f'rho_residuals_vs_rho.png')
        plt.figure(figsize=(10,8))
        plt.hist2d(rho_target, rho_residuals, norm=mpl.colors.LogNorm(), bins=(100, 100), cmap='viridis')
        plt.title(title_string)
        plt.xlabel(r'$\rho$ True')
        plt.ylabel(r'$\rho$ Residuals')
        plt.colorbar(label='Events/Bin')
        plt.savefig(self.save_plot_path)
        plt.close()

        TV_z_residuals = self.get_residuals(self.test_targets['TV_z'], self.predictions['TV_z'])
        TV_R_residuals = self.get_residuals(self.test_targets['TV_R'], self.predictions['TV_R'])

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'TV_R Residuals vs TV_z Residuals{post_title_string}'

        self.save_path(f'TV_R_residuals_vs_TV_z_residuals.png')
        plt.figure(figsize=(10,8))
        plt.hist2d(TV_z_residuals, TV_R_residuals, norm=mpl.colors.LogNorm(), bins=(100, 100), cmap='viridis')
        plt.title(title_string)
        plt.xlabel(r'TV_z Residuals (mm)')
        plt.ylabel(r'TV_R Residuals (mm)')
        plt.colorbar(label='Events/Bin')
        plt.savefig(self.save_plot_path)
        plt.close()

        TV_z_residuals = abs(TV_z_residuals)
        TV_R_residuals = abs(TV_R_residuals)
        summed_residuals = TV_z_residuals + TV_R_residuals

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'Sum of TV_z and TV_R Residuals{post_title_string}'

        self.save_path(f'summed_residuals_TV_z_and_TV_R.png')
        plt.figure(figsize=(10,8))
        plt.hist(summed_residuals, bins=100)
        plt.xlabel(r'abs(TV_z residuals) + abs(TV_R residuals) (mm)')
        plt.ylabel('Events/Bin')
        plt.title(title_string)
        plt.yscale('log')
        plt.savefig(self.save_plot_path)
        plt.close()

        theta_target = np.arctan(self.test_targets['TV_R'] / self.test_targets['TV_z'])
        theta_predictions = np.arctan(self.predictions['TV_R'] / self.predictions['TV_z'])
        theta_target.loc[theta_target < 0] += np.pi
        theta_predictions.loc[theta_predictions < 0] += np.pi

        theta_residuals = self.get_residuals(theta_target, theta_predictions)

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'$\theta$ Residuals{post_title_string}'

        self.save_path(f'theta_residuals.png')

        plt.figure(figsize=(10,8))
        plt.hist(theta_residuals, bins=100)
        plt.xlabel(r'$\theta$ Residuals (rad)')
        plt.ylabel('Events/Bin')
        plt.title(title_string)
        plt.yscale('log')
        plt.savefig(self.save_plot_path)
        plt.close()

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'$\theta$ Predicted vs. $\theta$ True{post_title_string}'

        self.save_path(f'theta_predicted_vs_theta_true.png')

        plt.figure(figsize=(10,8))
        plt.hist2d(theta_target, theta_predictions, norm=mpl.colors.LogNorm(), bins=(100, 100), cmap='viridis')
        plt.title(title_string)
        plt.xlabel(r'$\theta$ True')
        plt.ylabel(r'$\theta$ Predicted')
        plt.colorbar(label='Events/Bin')
        plt.savefig(self.save_plot_path)
        plt.close()

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'Residuals of $\theta${post_title_string}'

        self.save_path(f'theta_residuals_vs_theta.png')

        plt.figure(figsize=(10,8))
        plt.hist2d(theta_target, theta_residuals, norm=mpl.colors.LogNorm(), bins=(100, 100), cmap='viridis')
        plt.title(title_string)
        plt.xlabel(r'$\theta$ True')
        plt.ylabel(r'$\theta$ Residuals')
        plt.colorbar(label='Events/Bin')
        plt.savefig(self.save_plot_path)
        plt.close()

        eta_target = -np.log(np.tan(theta_target/2))
        eta_predictions = -np.log(np.tan(theta_predictions/2))
        eta_residuals = self.get_residuals(eta_target, eta_predictions)

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'$\eta$ Residuals{post_title_string}'

        self.save_path(f'eta_residuals.png')

        plt.figure(figsize=(10,8))
        plt.hist(eta_residuals, bins=100)
        plt.xlabel(r'$\eta$ Residuals')
        plt.ylabel('Events/Bin')
        plt.title(title_string)
        plt.yscale('log')
        plt.savefig(self.save_plot_path)
        plt.close()

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'Predicted $\eta$ vs. True $\eta${post_title_string}'

        self.save_path(f'predicted_eta_vs_eta.png')

        plt.figure(figsize=(10,8))
        plt.hist2d(eta_target, eta_predictions, norm=mpl.colors.LogNorm(), bins=(100, 100), cmap='viridis')
        plt.title(title_string)
        plt.xlabel(r'$\eta$ True')
        plt.ylabel(r'$\eta$ Predicted')
        plt.colorbar(label='Events/Bin')
        plt.savefig(self.save_plot_path)
        plt.close()

        if self.post_title == None:
            post_title_string = ''
        else:
            post_title_string = f' ({self.post_title})'
        title_string = rf'Residuals of $\eta${post_title_string}'
        
        self.save_path(f'eta_residuals_vs_eta.png')

        plt.figure(figsize=(10,8))
        plt.hist2d(eta_target, eta_residuals, norm=mpl.colors.LogNorm(), bins=(100, 100), cmap='viridis')
        plt.title(title_string)
        plt.xlabel(r'$\eta$ True')
        plt.ylabel(r'$\eta$ Residuals')
        plt.colorbar(label='Events/Bin')
        plt.savefig(self.save_plot_path)
        plt.close()




    def _make_df_triangle(self, df):
        # set all values above the diagonal to NaN in the dataframe df
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                df.iloc[i, j] = np.NaN
        return df
    

    def _reverse_row_order(self, df):
        # reverse the order of the rows in the dataframe df
        df = df.iloc[::-1]


    def plot_cross_feature_importance(self):
        x_test = self.test_data[0]
        y_test = self.test_data[1]

        x_test = self._drop_ignore_train(x_test)
        y_test = self._drop_ignore_train(y_test)

        # x_test = x_test.sample(frac=0.01)
        # y_test = y_test.loc[x_test.index]

        for target in self.target_list:

            r2_scores = pd.DataFrame(
                columns=x_test.columns,
                index=x_test.columns,
            )
            r2_scores['none'] = np.NaN          # add 'none' column for no randomization
            r2_scores.loc['none'] = np.NaN      # add 'none' row for no randomization
            mse_scores = pd.DataFrame(
                columns=x_test.columns,
                index=x_test.columns,
            )
            mse_scores['none'] = np.NaN          # add 'none' column for no randomization
            mse_scores.loc['none'] = np.NaN      # add 'none' row for no randomization

            n = len(r2_scores.columns)
            # task_count = n**2 - n*(n-1)/2   # number of unique squares in the matrix
            task_count = 0
            for i in range(n):
                task_count += n - i
            
            logging.info(f"Processing CFI for {target}...")
            self.printProgressBar(0, task_count, prefix='Processing CFI:', suffix='Complete', length=50)
            counter = 0
            # make a 2D version of plot_feature_importance
            for f1 in r2_scores.columns:
                for f2 in r2_scores.index:
                    if math.isnan(r2_scores.loc[f2, f1]):  
                        if counter == task_count:
                            break

                        if not r2_scores.isnull().values.any() and not mse_scores.isnull().values.any():
                            # if neither r2_scores nor mse_scores have any NaN values, then we can break out of the loop
                            break

                        # randomize both f1 and f2 and see how it affects the model
                        x_test_rand = x_test.copy()    # to be randomized for features f1 and f2

                        test_targets_small = y_test.loc[x_test_rand.index]

                        # randomize f1 variable
                        x_test_rand = self.randomize_feature(x_test_rand, f1, f2)

                        test_predictions_randomized = self.predict(x_test_rand)
                        # test_predictions_randomized = test_targets_small[target]**2 # TEMP

                        error = self.get_residuals(test_targets_small[target], test_predictions_randomized[target])

                        mse = np.sqrt(float((error**2).mean()))

                        x = test_targets_small[target]
                        y = test_predictions_randomized[target]

                        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

                        r_squared = r_value**2

                        # r2_scores.set_value(f1, f2, r_squared)      # better than doing r2_scores[f1][f2] = r_squared
                        # r2_scores.set_value(f2, f1, r_squared)
                        # mse_scores.set_value(f1, f2, mse)
                        # mse_scores.set_value(f2, f1, mse)

                        r2_scores.at[f2, f1] = r_squared
                        r2_scores.at[f1, f2] = r_squared
                        mse_scores.at[f2, f1] = mse
                        mse_scores.at[f1, f2] = mse

                        logging.debug(f'Randomized {f1} and {f2} and got R2 = {r_squared} and MSE = {mse}')
                        logging.debug(f'r_scores.head(): \n{r2_scores.head()}')
                        logging.debug(f'mse_scores.head(): \n{mse_scores.head()}')
                        logging.debug(f'counter = {counter} and task_count = {task_count}')

                        if counter > task_count:
                            logging.warning(f'counter > task_count! counter = {counter} and task_count = {task_count}')
                        
                        self.printProgressBar(counter + 1, task_count, prefix='Processing CFI:', suffix='Complete', length=50)
                        counter += 1
                        
            
            r2_scores = r2_scores.astype(float)
            mse_scores = mse_scores.astype(float)
            # r2_scores = self._reverse_row_order(r2_scores)
            # mse_scores = self._reverse_row_order(mse_scores)
            # r2_scores = self._make_df_triangle(r2_scores)
            # mse_scores = self._make_df_triangle(mse_scores)

            # sns.heatmap(r2_scores, annot=True, fmt='.2f')

            # Plot R2 Heatmap
            self.save_path(f'cross_feature_importance_R2_{target}.png')

            if self.post_title == None:
                post_title_string = ''
            else:
                post_title_string = f' ({self.post_title})'
            title_string = rf'Cross Feature Importance for {target} Comparing R$^2${post_title_string}'

            fig, ax = plt.subplots(figsize=(15, 15))
            im, cbar = self.heatmap(self._make_df_triangle(r2_scores), r2_scores.index, r2_scores.columns, ax=ax,
                                    cbarlabel="R$^2$ Score", cmap='YlOrRd_r')
            texts = self.annotate_heatmap(im, valfmt="{x:.2f}")
            fig.tight_layout()
            plt.savefig(self.save_plot_path)


            self.save_path(f'cross_feature_importance_MSE_{target}.png')

            # Plot MSE Heatmap
            if self.post_title == None:
                post_title_string = ''
            else:
                post_title_string = f' ({self.post_title})'
            title_string = rf'Cross Feature Importance for {target} Comparing MSE${post_title_string}'

            fig, ax = plt.subplots(figsize=(15, 15))
            im, cbar = self.heatmap(self._make_df_triangle(mse_scores), mse_scores.index, mse_scores.columns, ax=ax,
                                    cbarlabel="sqrt(MSE) Score", cmap='YlOrRd_r')
            texts = self.annotate_heatmap(im, valfmt="{x:.2f}")
            fig.tight_layout()
            plt.savefig(self.save_plot_path)

            # get the first last row of r2_scores and mse_scores


            r2_scores = r2_scores / r2_scores.iloc[-1]
            mse_scores = mse_scores / mse_scores.iloc[-1]

            self.save_path(f'cross_feature_importance_relative_R2_{target}.png')

            if self.post_title == None:
                post_title_string = ''
            else:
                post_title_string = f' ({self.post_title})'
            title_string = rf'Relative Cross Feature Importance for {target} Comparing Relative R$^2${post_title_string}'

            fig, ax = plt.subplots(figsize=(15, 15))
            im, cbar = self.heatmap(self._make_df_triangle(r2_scores), r2_scores.index, r2_scores.columns, ax=ax,
                                    cbarlabel="R$^2$ / R$^2$ (single feature) Score", cmap='YlOrRd_r')
            texts = self.annotate_heatmap(im, valfmt="{x:.2f}")
            fig.tight_layout()
            plt.savefig(self.save_plot_path)

            self.save_path(f'cross_feature_importance_relative_MSE_{target}.png')

            # Plot MSE Heatmap
            if self.post_title == None:
                post_title_string = ''
            else:
                post_title_string = f' ({self.post_title})'
            title_string = rf'Relative Cross Feature Importance for {target} Comparing MSE${post_title_string}'

            fig, ax = plt.subplots(figsize=(15, 15))
            im, cbar = self.heatmap(self._make_df_triangle(mse_scores), mse_scores.index, mse_scores.columns, ax=ax,
                                    cbarlabel="MSE / MSE (single feature) Score", cmap='YlOrRd_r')
            texts = self.annotate_heatmap(im, valfmt="{x:.2f}")
            fig.tight_layout()
            plt.savefig(self.save_plot_path)


                
                    

