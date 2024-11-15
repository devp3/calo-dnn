import logging, sys, os
import numpy as np
import pandas as pd


def apply(raw, name='Zee'):
    if name=='Zee':
        return Zee_cuts(raw)
    elif name=='signal':
        return signal_cuts(raw)
    else:
        logging.error(f'cut name {name} not recognized')
        sys.exit(1)


def signal_cuts(raw):
    logging.debug(f'(signal_cuts) initial length of raw = {raw.shape[0]}')

    raw = raw[(raw.PassTrig_g35_loose_g25_loose == 1) | (raw.PassTrig_g35_medium_g25_medium == 1)]

    logging.debug(f'(signal_cuts) after trigger cut length of raw = {raw.shape[0]}')

    raw = raw[(raw.ph1_pt > 40) & (raw.ph2_pt > 30)]

    logging.debug(f'(signal_cuts) after pt cut length of raw = {raw.shape[0]}')

    # remove bad points with -99999.9 error float
    raw = raw[(raw.ph1_maxEcell_t != -99999.9) & (raw.ph2_maxEcell_t != -99999.9)]

    logging.debug(f'(signal_cuts) after maxEcell_t cut length of raw = {raw.shape[0]}')

    raw = raw[(abs(raw.ph1_t_unblinded) < 12) & (abs(raw.ph2_t_unblinded) < 12)]

    logging.debug(f'(signal_cuts) after t cut length of raw = {raw.shape[0]}')

    raw = raw[(raw.ph1_maxEcell_E > 5) & (raw.ph2_maxEcell_E > 5)]

    logging.debug(f'(signal_cuts) after maxEcell_E cut length of raw = {raw.shape[0]}')

    raw = raw[(raw.Hcand_M > 60) & (raw.Hcand_M < 135)]

    logging.debug(f'(signal_cuts) after Hcand_M cut length of raw = {raw.shape[0]}')

    # require that TV_z occur within +-3740 mm, value from Dev's thesis
    # this requires that they occur just before the calorimeters
    raw = raw[(abs(raw.TV_z) < 3740.0)]

    logging.debug(f'(signal_cuts) after TV_z cut length of raw = {raw.shape[0]}')

    raw = raw[(np.sqrt(raw.TV_x**2 + raw.TV_y**2) < 1200.0)]

    logging.debug(f'(signal_cuts) after TV_r cut length of raw = {raw.shape[0]}')

    raw = raw[(abs(raw.ph1_eta) < 1.37) | (abs(raw.ph2_eta) < 1.37)]

    logging.debug(f'(signal_cuts) after eta cut length of raw = {raw.shape[0]}')

    raw = raw[abs(raw.dEta_ph) > .1]

    logging.debug(f'(signal_cuts) after dEta_ph cut length of raw = {raw.shape[0]}')

    # raw = raw[(abs(raw.TV_z) < 200)]

    # raw = raw[(np.abs(raw.TV_z) > 400) & (np.abs(raw.TV_z) < 700)]

    # raw = raw[(raw.ph1_maxEcell_t < 0.1) | (raw.ph2_maxEcell_t < 0.1)]

    return raw


def Zee_cuts(raw):
    logging.debug(f'(Zee_cuts) initial length of raw = {raw.shape[0]}')

    raw = raw[(abs(raw.el1_eta) < 1.37) | (abs(raw.el2_eta) < 1.37)]

    logging.debug(f'(Zee_cuts) after eta cut length of raw = {raw.shape[0]}')

    raw = raw[(raw.Zcand_M > 60) & (raw.Zcand_M < 135)]

    logging.debug(f'(Zee_cuts) after Zcand_M cut length of raw = {raw.shape[0]}')

    raw = raw[(raw.Zcand_pt > 70) & (raw.dPhi_el < 2.4)]

    logging.debug(f'(Zee_cuts) after Zcand_pt and dPhi_el cut length of raw = {raw.shape[0]}')

    raw = raw[(abs(raw.el1_pt) > 40) & (abs(raw.el2_pt) > 30)]

    logging.debug(f'(Zee_cuts) after pt cut length of raw = {raw.shape[0]}')

    raw = raw[(abs(raw.el1_t) < 12) & (abs(raw.el2_t) < 12)]

    logging.debug(f'(Zee_cuts) after t cut length of raw = {raw.shape[0]}')

    raw = raw[(raw.el1_maxEcell_E > 5) & (raw.el2_maxEcell_E > 5)]

    logging.debug(f'(Zee_cuts) after maxEcell_E cut length of raw = {raw.shape[0]}')

    raw = raw[(abs(raw.dEta_el) > .1)]

    logging.debug(f'(Zee_cuts) after dEta_el cut length of raw = {raw.shape[0]}')

    return raw