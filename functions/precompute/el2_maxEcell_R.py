import os
import numpy as np

def dependencies():
    return ['el2_maxEcell_x', 'el2_maxEcell_y']

def compute(df):

    df['el2_maxEcell_R'] = np.sqrt(df['el2_maxEcell_x']**2 + df['el2_maxEcell_y']**2)   

    return df
