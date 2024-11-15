# configs

## Purpose and Scope

This directory contains configuration files for the neural-llp codebase. The configuration files are used to specify the parameters for the training and analysis processes. The configuration files are written in JSON format. The purpose of having a config directory is to allow the user to easily swap config files without having to modify the codebase. Notable configurations are saved here and can be easily loaded in the future. 

## Usage

The configuration files are meant to be accessed by the user somewhat frequently. Certain config files are meant to be kept constant (ex. network hyperparameters), while others are meant to be modified frequently (ex. input variables). 

You may want to test the network on several different sets of input variables. In this case, you would create a new config file for each configuration. In the config file, you should not listing multiple input variable sets in the same file, perhaps by doing something like creating multiple superlists of input variables.

## Structure

This directory should be composed of a list of config files. Do not add subdirectories without modifying the config parser located in utils. 

## Naming

Name files in the following format: '[usage]_config.json'. Please try to be as descriptive as possible in the shortest length.