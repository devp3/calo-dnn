import os
import numpy as np

def dependencies():
    return ['el2_etas1', 'el2_etas2']

def compute(df):

    df['el2_diff_etas12'] = df['el2_etas1'] - df['el2_etas2']

    return df

