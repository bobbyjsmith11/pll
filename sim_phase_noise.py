"""
==================
sim_phase_noise.py
==================

simulate the phase noise components in a PLL


"""


from pll.phase_noise import *
from plot_json import plot_pn_contribs 
from pll.pll_calcs import *
import numpy as np

import argparse


def simulate_pll_phase_noise(ref_json, vco_json, pll_json, 
                             pll_fom, fmin=1e3, fmax=10e6, 
                             num_points=100, flicker=-500):
    """
    :Argumnets:
        ref_json (str): json file for reference
        vco_json (str): json file for vco
        pll_json (str): json file for pll paramters
        pll_fom float): figure-of-merit for PLL
        fmin (float): minimum offset
        fmax (float): maximum offset
        num_points (int): number of points
        flicker (int): number of points
    """
    ref = load_pn_json(ref_json)
    vco = load_pn_json(vco_json)
    pll = load_pn_json(pll_json)
    # translate the phase noise to the comparison frequency
    # calculates noise after the R divider
    f, pn = translate_phase_noise(ref, pll['fpfd'])
    freq, pn_r = interp_semilogx(f, pn, x_range=[fmin, fmax], 
                num_points=num_points)

    # translate vco phase noise to fcarrier
    f, pn = translate_phase_noise(vco, pll['fvco'])
    freq, pn_vco = interp_semilogx(f, pn, x_range=[fmin, fmax], 
                num_points=num_points)


    d = sim_phase_noise(freq, pn_r, pn_vco, pll_fom, flicker, pll)
    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref_json', type=str, 
        help='reference phase noise json', default='data/asvtx12.json')
    parser.add_argument('-v', '--vco_json', type=str, 
        help='vco phase noise json', default='data/navassa_vco.json')
    parser.add_argument('-p', '--pll_json', type=str, 
        help='json file for pll parmaeters', default='data/pll1.json')
    parser.add_argument('--pll_fom', type=float, 
        help='PLL figure-of-merit in dBc/Hz', default=-221)
    parser.add_argument('--pll_flick', type=float, 
        help='PLL flicker in dBc/Hz', default=-500)
    parser.add_argument('--fvco', type=float, 
        help='carrier frequency in Hz', default=2000e6)
    parser.add_argument('--fstart', type=float, 
        help='starting offset in Hz', default=100)
    parser.add_argument('--fstop', type=float, 
        help='ending offset in Hz', default=10e6)
    parser.add_argument('--y_max', type=float, 
        help='top of y scale for phase noise (dBc/Hz)', default=-60)
    parser.add_argument('--y_min', type=float, 
        help='bottom of y scale for phase noise (dBc/Hz)', default=-160)
    parser.add_argument('--y_tick', type=float, 
        help='increment on the y scale', default=10)
    parser.add_argument('--num_points', type=int, 
        help='number of points in sim', default=1000)
    parser.add_argument('--title', type=str, 
        help='plot title', default='add a title')
    args = parser.parse_args()
    pn_dict = simulate_pll_phase_noise(args.ref_json, args.vco_json, 
                     args.pll_json, args.pll_fom, 
                     fmin=args.fstart, fmax=args.fstop, 
                     num_points=args.num_points, flicker=args.pll_flick)
    print(pn_dict)
    # plot_pn_contribs(pn_dict, title=title_str, ylim=[args.y_min, args.y_max], 
    #                  ytick=args.y_tick)
    plot_pn_contribs(pn_dict, title=args.title, ylim=[-160, -60], ytick=10)
