import numpy as np

def dependencies():
    return ['TV_x', 'TV_y']

def compute(df):
    df['TV_R'] = np.sqrt(df['TV_x']**2 + df['TV_y']**2)
    return df