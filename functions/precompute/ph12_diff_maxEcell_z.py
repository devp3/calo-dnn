import os
import numpy as np

def dependencies():
    return ['ph1_maxEcell_z', 'ph2_maxEcell_z']

def compute(df):

    df['ph12_diff_maxEcell_z'] = df['ph1_maxEcell_z'] - df['ph2_maxEcell_z']

    return df

