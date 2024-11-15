import os
import numpy as np

def dependencies():
    return ['el2_etas1', 'el2_etas2']

def compute(df):

    df['el2_sinh_etas12'] = np.sinh(df['el2_etas1']) - np.sinh(df['el2_etas2'])    

    return df

# TODO: Add function to return description of el1_sinh_etas12