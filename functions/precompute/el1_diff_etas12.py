import os
import numpy as np

def dependencies():
    return ['el1_etas1', 'el1_etas2']

def compute(df):

    df['el1_diff_etas12'] = df['el1_etas1'] - df['el1_etas2']

    return df

