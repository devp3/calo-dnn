import os
import numpy as np

def dependencies():
    return ['ph1_maxEcell_x', 'ph1_maxEcell_y']

def compute(df):

    df['ph1_maxEcell_R'] = np.sqrt(df['ph1_maxEcell_x']**2 + df['ph1_maxEcell_y']**2)   

    return df

# TODO: Add function to return description of el1_sinh_etas12