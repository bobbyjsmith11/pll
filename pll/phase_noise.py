"""

# Phase Noise JSON format
    :frequency (float): VCO/PLL frequency in Hz
    :offset (list of floats): phase noise offset frequency points
    :phase_noise (list of floats): phase noise points
    :tune_ppm_per_volt (float): tuning sensitivity in ppm per volt (optional)
    :tune_Hz_per_volt (float): tuning sensitivity in Hz per volt (optional)
"""

import json
import numpy as np

def load_pn_json(fname):
    """

    """
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

def load_pn_files_directory(mydir):
    """
    """
    pass


def translate_phase_noise(in_dict, freq_Hz):
    """
    return the frequency offset and translated phase noise
    to the target frequency
    """
    base_freq_Hz = in_dict['frequency']
    pn_factor = 20*np.log10(freq_Hz/base_freq_Hz)
    freq_ar = []
    pn_ar = []
    for i in range(len(in_dict['offset'])):
        freq_ar.append(in_dict['offset'][i])
        pn_ar.append(in_dict['phase_noise'][i] + pn_factor)
    return freq_ar, pn_ar

