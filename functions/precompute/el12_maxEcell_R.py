import os
import numpy as np

def dependencies():
    return ['el1_maxEcell_R', 'el2_maxEcell_R']

def compute(df):

    df['el12_maxEcell_R'] = df['el1_maxEcell_R'] - df['el2_maxEcell_R']

    return df

