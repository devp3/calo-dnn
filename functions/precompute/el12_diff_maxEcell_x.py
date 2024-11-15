import os
import numpy as np

def dependencies():
    return ['el1_maxEcell_x', 'el2_maxEcell_x']

def compute(df):

    df['el12_diff_maxEcell_x'] = df['el1_maxEcell_x'] - df['el2_maxEcell_x']

    return df

