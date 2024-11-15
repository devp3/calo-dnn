def dependencies():
    return ['el1_sinh_etas12']

def compute(df):
    R1 = 1544.0 # mm
    R2 = 1761.0 # mm

    df['el1_dca'] = (R1 * R2)/(R1 - R2) * df['el1_sinh_etas12']
    
    return df
