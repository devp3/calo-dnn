import os
import numpy as np

def dependencies():
    return ['ph2_etas1', 'ph2_etas2']

def compute(df):

    df['ph2_diff_etas12'] = df['ph2_etas1'] - df['ph2_etas2']

    return df

