#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging, sys

names = [
    '925_2ns_ZeeZSM',
    '925_10ns_ZeeZSM',
    '825_2ns_ZeeZSM',
    '825_10ns_ZeeZSM',
    '775_2ns_ZeeZSM',
    '775_10ns_ZeeZSM',
    '725_2ns_ZeeZSM',
    '725_10ns_ZeeZSM',
    '625_2ns_ZeeZSM',
    '625_20ns_ZeeZSM',
    '625_10ns_ZeeZSM',
    '525_50ns_ZeeZSM',
    '525_50ns_ZeeHSM',
    '525_2ns_ZeeZSM',
    '525_2ns_ZeeHSM',
    '525_2ns_HyyZSM',
    '525_2ns_HyyHSM',
    '525_20ns_ZeeZSM',
    '525_20ns_ZeeHSM',
    '525_10ns_ZeeZSM',
    '525_10ns_ZeeHSM',
    '525_10ns_HyyZSM',
    '525_10ns_HyyHSM',
    '475_2ns_HyyHSM',
    '475_10ns_HyyHSM',
    '425_2ns_ZeeZSM',
    '425_2ns_ZeeHSM',
    '425_2ns_HyyZSM',
    '425_2ns_HyyHSM',
    '425_20ns_ZeeZSM',
    '425_20ns_ZeeHSM',
    '425_10ns_ZeeZSM',
    '425_10ns_ZeeHSM',
    '425_10ns_HyyZSM',
    '425_10ns_HyyHSM',
    '375_2ns_HyyHSM',
    '375_10ns_HyyHSM',
    '325_50ns_ZeeZSM',
    '325_50ns_ZeeHSM',
    '325_2ns_ZeeZSM',
    '325_2ns_ZeeHSM',
    '325_2ns_HyyZSM',
    '325_2ns_HyyHSM',
    '325_20ns_ZeeZSM',
    '325_20ns_ZeeHSM',
    '325_20ns_HyyZSM',
    '325_20ns_HyyHSM',
    '325_10ns_ZeeZSM',
    '325_10ns_ZeeHSM',
    '325_10ns_HyyZSM',
    '325_10ns_HyyHSM',
    '275_2ns_HyyHSM',
    '275_20ns_HyyHSM',
    '275_10ns_HyyHSM',
    '225_50ns_HyyZSM',
    '225_50ns_HyyHSM',
    '225_2ns_ZeeZSM',
    '225_2ns_ZeeHSM',
    '225_2ns_HyyZSM',
    '225_2ns_HyyHSM',
    '225_20ns_ZeeZSM',
    '225_20ns_ZeeHSM',
    '225_20ns_HyyZSM',
    '225_20ns_HyyHSM',
    '225_10ns_ZeeZSM',
    '225_10ns_ZeeHSM',
    '225_10ns_HyyZSM',
    '225_10ns_HyyHSM',
    '175_2ns_ZeeZSM',
    '175_2ns_ZeeHSM',
    '175_2ns_HyyZSM',
    '175_2ns_HyyHSM',
    '175_20ns_ZeeZSM',
    '175_20ns_ZeeHSM',
    '175_20ns_HyyZSM',
    '175_20ns_HyyHSM',
    '175_10ns_ZeeZSM',
    '175_10ns_ZeeHSM',
    '175_10ns_HyyZSM',
    '175_10ns_HyyHSM',
    '135_50ns_ZeeZSM',
    '135_50ns_ZeeHSM',
    '135_50ns_HyyZSM',
    '135_50ns_HyyHSM',
    '135_2ns_ZeeZSM',
    '135_2ns_ZeeHSM',
    '135_2ns_HyyZSM',
    '135_2ns_HyyHSM',
    '135_20ns_ZeeZSM',
    '135_20ns_ZeeHSM',
    '135_20ns_HyyZSM',
    '135_20ns_HyyHSM',
    '135_10ns_ZeeZSM',
    '135_10ns_ZeeHSM',
    '135_10ns_HyyZSM',
    '135_10ns_HyyHSM',
    '100_2ns_ZeeZSM',
    '100_20ns_ZeeZSM',
    '100_10ns_ZeeZSM',
]

def get_params(names):
    params = pd.DataFrame(columns=['mass', 'lifetime', 'type'])

    for name in names:
        mass, lifetime, type = name.split('_')
        lifetime = lifetime.replace('ns', '')
        mass = int(mass)
        lifetime = int(lifetime)
        p = pd.DataFrame({'mass': mass, 'lifetime': lifetime, 'type': type}, index=[0])
        params = pd.concat([params, p], ignore_index=True)

    return params

# params['type'].map({'ZeeZSM': 'r', 'ZeeHSM': 'b', 'HyyZSM': 'g', 'HyyHSM': 'y'})
# params['type'].map({'ZeeZSM': 5, 'ZeeHSM': 50, 'HyyZSM': 100, 'HyyHSM': 150})

def plot_params(params):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        params[params.type == 'ZeeZSM']['mass'], 
        params[params.type == 'ZeeZSM']['lifetime'], 
        edgecolors='r',
        facecolor='none',
        linewidths=2,
        s=5,
        marker='o',
        label='ZeeZSM',
        # alpha=0.25,
    )
    ax.scatter(
        params[params.type == 'ZeeHSM']['mass'], 
        params[params.type == 'ZeeHSM']['lifetime'], 
        edgecolors='b',
        facecolor='none',
        linewidths=2,
        s=40,
        marker='o',
        label='ZeeHSM',
        # alpha=0.25,
    )
    ax.scatter(
        params[params.type == 'HyyZSM']['mass'], 
        params[params.type == 'HyyZSM']['lifetime'], 
        edgecolors='g',
        facecolor='none',
        linewidths=2,
        s=110,
        marker='o',
        label='HyyZSM',
        # alpha=0.25,
    )
    ax.scatter(
        params[params.type == 'HyyHSM']['mass'], 
        params[params.type == 'HyyHSM']['lifetime'], 
        edgecolors='y',
        facecolor='none',
        linewidths=2,
        s=200,
        marker='o',
        label='HyyHSM',
        # alpha=0.25,
    )
    ax.grid()
    ax.set_xlabel('mass (GeV)')
    ax.set_ylabel('lifetime (ns)')
    ax.set_title('Signal Parameters')
    ax.legend()
    plt.savefig('../signal_params.png', dpi=300)
    plt.show()
    plt.close()


params = get_params(names)
plot_params(params)
# %%
