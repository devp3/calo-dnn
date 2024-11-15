import numpy as np

def dependencies():
    return ['ph1_phi', 'ph2_phi']

def compute(df):
    cos_dphi = np.cos(df['ph1_phi'] - df['ph2_phi'])
    acos_dphi = np.arccos(cos_dphi)
    df['dphi_ph'] = acos_dphi

    return df