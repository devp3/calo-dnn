import os
import numpy as np

def dependencies():
    return ['ph1_maxEcell_y', 'ph2_maxEcell_y']

def compute(df):

    df['ph12_diff_maxEcell_y'] = df['ph1_maxEcell_y'] - df['ph2_maxEcell_y']

    return df

