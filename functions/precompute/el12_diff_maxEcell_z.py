import os
import numpy as np

def dependencies():
    return ['el1_maxEcell_z', 'el2_maxEcell_z']

def compute(df):

    df['el12_diff_maxEcell_z'] = df['el1_maxEcell_z'] - df['el2_maxEcell_z']

    return df

