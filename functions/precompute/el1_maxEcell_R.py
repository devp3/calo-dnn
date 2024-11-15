import os
import numpy as np

def dependencies():
    return ['el1_maxEcell_x', 'el1_maxEcell_y']

def compute(df):

    df['el1_maxEcell_R'] = np.sqrt(df['el1_maxEcell_x']**2 + df['el1_maxEcell_y']**2)   

    return df

# TODO: Add function to return description of el1_sinh_etas12