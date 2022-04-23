"""
================
plot_pn_json.py
================

plot a vco phase noise json file


"""


from pll.phase_noise import load_pn_json 
import matplotlib.pyplot as plt
import numpy as np
import json

import argparse

def plot_pn_contribs(in_dict, title="add title", ylim=[-160,-60], ytick=10):
    """
    plot all of the phase noise contributors
    :Arguments:
        in_dict (dict): 
    """
    for k in in_dict.keys():
        if k == 'offset':
            continue
        plt.semilogx(in_dict['offset'], in_dict[k], label=k)
    plt.title(title)
    plt.legend()
    plt.yticks(np.arange(ylim[0], ylim[1], ytick))
    plt.ylim(ylim)
    plt.xlabel("offset (Hz)")
    plt.ylabel("phase noise (dBc/Hz)")
    plt.grid(True)
    plt.show() 


def plot_pn_file(in_dict):
    """
    plot a dict with 'offset' and 'phase_nose' keys
    """
    plt.semilogx(d['offset'], d['phase_noise'])
    plt.title(d['name'])
    plt.xlabel("offset (Hz)")
    plt.ylabel("phase noise (dBc/Hz)")
    plt.grid(True)
    plt.show() 


def plot_loop_response(in_dict, title="add title"):
    """
    plot all of the phase noise contributors
    :Arguments:
        in_dict (dict):
            keys:   f (frequency)
                    pm (phase margin)
                    ol_g_db (open loop gain in dB)
                    cl_g_db (open loop gain in dB)
    """
    fig = plt.figure(figsize=(8, 7))
    
    # noise figure
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()
    ax1.semilogx(in_dict['f'], in_dict['ol_g_db'], label='ol gain')
    ax2.semilogx(in_dict['f'], in_dict['pm'], color='r', label='phase margin')
    ax1.set_ylabel("dB")
    ax2.set_ylabel("degrees")
    ax1.legend(loc=7)
    ax2.legend(loc=1)
    ax1.set_title(title)
    plt.tight_layout()
    # plt.ylim(0, 20)
    ax1.grid(True, which='both')

    # closed loop
    ax3 = fig.add_subplot(212, sharex=ax1)
    ax3.semilogx(in_dict['f'], in_dict['cl_g_db'], label='cl gain')
    ax3.set_ylabel("dB")
    # ax3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    ax3.legend(loc=7)
    ax3.grid(True, which='both')
    # plt.yticks(np.arange(0, 50, 5))
    # plt.ylim(0, 50)


    plt.xlabel("offset Hz")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('osc_json', type=str, 
        help='oscillator json file', default='data/ref_pll1.json')

    args = parser.parse_args()

    d = load_pn_json(args.osc_json)
    



