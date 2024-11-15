import os
import numpy as np

def dependencies():
    return ['el1_etas1', 'el1_etas2']

def compute(df):

    df['el1_sinh_etas12'] = np.sinh(df['el1_etas1']) - np.sinh(df['el1_etas2'])    

    return df

# TODO: Add function to return description of el1_sinh_etas12