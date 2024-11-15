import os
import numpy as np

def dependencies():
    return ['el1_maxEcell_y', 'el2_maxEcell_y']

def compute(df):

    df['el12_diff_maxEcell_y'] = df['el1_maxEcell_y'] - df['el2_maxEcell_y']

    return df

