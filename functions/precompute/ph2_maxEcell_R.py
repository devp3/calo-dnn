import os
import numpy as np

def dependencies():
    return ['ph2_maxEcell_x', 'ph2_maxEcell_y']

def compute(df):

    df['ph2_maxEcell_R'] = np.sqrt(df['ph2_maxEcell_x']**2 + df['ph2_maxEcell_y']**2)   

    return df
