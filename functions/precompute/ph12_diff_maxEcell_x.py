import os
import numpy as np

def dependencies():
    return ['ph1_maxEcell_x', 'ph2_maxEcell_x']

def compute(df):

    df['ph12_diff_maxEcell_x'] = df['ph1_maxEcell_x'] - df['ph2_maxEcell_x']

    return df

