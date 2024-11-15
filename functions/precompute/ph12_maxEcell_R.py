import os
import numpy as np

def dependencies():
    return ['ph1_maxEcell_R', 'ph2_maxEcell_R']

def compute(df):

    df['ph12_maxEcell_R'] = df['ph1_maxEcell_R'] - df['ph2_maxEcell_R']

    return df

