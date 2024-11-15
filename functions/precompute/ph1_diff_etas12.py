import os
import numpy as np

def dependencies():
    return ['ph1_etas1', 'ph1_etas2']

def compute(df):

    df['ph1_diff_etas12'] = df['ph1_etas1'] - df['ph1_etas2']

    return df

