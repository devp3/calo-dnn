import numpy as np

def dependencies():
    return ['PV_x', 'PV_y']

def compute(df):
    df['PV_R'] = np.sqrt(df['PV_x']**2 + df['PV_y']**2)
    return df