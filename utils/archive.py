import os
import sys
import pandas as pd
import numpy as np
import csv, json, yaml, pickle
import datetime


# Write a flexible function to archive data to the box drive
# It will be able to save data in a variety of formats
# It will be able to move files, rename files, and create directories

# archive_path
#   - plots
#       - 02-25-23
#           - 1677386658            # unix timestamp
#               - loss_history.png
#               - rsquare_history.png
#               - compare_normalization.png
#               - ...
#           - 1677386659
#               - ...
#       - 02-26-23
#           - ...
#   - data                          # dump all data here
#       - 02-25-23
#           - 1677386658
#               - train_data
#               - val_data
#               - test_data
#               - raw_data
#               - normalized_data
#               - preprocessed_data
#               - normalization_factors
#               - history
#               - model
#               - checkpoint        # multiple checkpoints?
#               - ...
#       - 02-26-23
#           - 1677386659
#               - ...
#   - figures                       # dump all figures here
#       - 02-25-23                  # figures are for nice, polished plots or
#           - loss_history.png      # figures that are public-facing or in the
#           - figure_1.png          # process of being polished
#       - 02-26-23_meeting
#           - figure_1.png
#           - ...
#       - 02-26-23_presentation
#           - figure_1.png
#           - ...
#       - thesis_figs
#           - figure_1.png
#           - ...
#       - ...


# RULES:
#   - If the file already exists, do NOT overwrite it unless...
#       - Overwriting is allowed only if overwrite=True argument is passed
#       - Warn before overwriting
#   - Instead of overwriting, create a new file with a timestamp
#   - If the directory does not exist, create it
#   - If the size of the directory/file is greater than 200 MB, warn that it 
#       is large and may take a moment. 
