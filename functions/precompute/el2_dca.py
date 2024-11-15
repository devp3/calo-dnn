import os

def dependencies():
    return ['el2_sinh_etas12']

def compute(df):
    R1 = 1544.0 # mm
    R2 = 1761.0 # mm

    df['el2_dca'] = (R1 * R2)/(R1 - R2) * df['el2_sinh_etas12']
    
    return df