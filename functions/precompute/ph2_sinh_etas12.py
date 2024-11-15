import os
import numpy as np

def dependencies():
    return ['ph2_etas1', 'ph2_etas2']

def compute(df):

    df['ph2_sinh_etas12'] = np.sinh(df['ph2_etas1']) - np.sinh(df['ph2_etas2'])    

    return df

# TODO: Add function to return description of el1_sinh_etas12