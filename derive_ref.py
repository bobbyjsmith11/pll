"""
=============
derive_ref.py
=============

Derive the phase noise requirements of the reference


"""


from pll.phase_noise import load_pn_json 
from pll.pll_calcs import *
import numpy as np
import json

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pll_json', type=str, 
        help='filter json', default='data/pll1.json')
    parser.add_argument('--pn_spec_json', type=str, 
        help='phase noise spec json', default='data/navassa_wb.json')
    parser.add_argument('--margin_db', type=float, 
        help='margin to stay below spec in dB', default=0)
    parser.add_argument('--out_div', type=float, 
        help='output divider after the PLL', default=1)
     
    parser.add_argument('--fstart', type=float, 
        help='starting offset in Hz', default=100)
    parser.add_argument('--fstop', type=float, 
        help='ending offset in Hz', default=10e6)
    parser.add_argument('--num_points', type=int, 
        help='number of points in sim', default=100)

    parser.add_argument('-f', '--frequency', type=float, 
        help='reference frequency in Hz', default=100e6)
    parser.add_argument('--name', type=str, 
        help='name key you wish to write to the json file', 
        default="my phase noise spec")
    parser.add_argument('--out_file', type=str, 
        help='save destination', default="data/myphasenoise.json")
    args = parser.parse_args()

    spec = load_pn_json(args.pn_spec_json)
    f, pn_spec = interp_semilogx(spec['offset'], spec['phase_noise'],
                        x_range=[args.fstart,args.fstop], 
                        num_points=args.num_points)

    pll = load_pn_json(args.pll_json)
    print(json.dumps(pll, indent=2))
    t2 = pll['t2']
    a0 = pll['a0']
    a1 = pll['a1']
    a2 = pll['a2']
    a3 = pll['a3']
    kphi = pll['kphi']
    kvco = pll['kvco']
    R = pll['R']
    N = pll['N']
    cl_db = get_closed_loop_transfer_fn(f, kphi, kvco, R, N, t2, 
            a0, a1, a2=a2, a3=a3) - 20*np.log10(args.out_div)
    pn_ref_max = []
    for k in range(len(f)):
        pn_ref_max.append(pn_spec[k] - cl_db[k] - args.margin_db)
    print("{:<20}{:<20}{:<20}{:<20}".format("FREQ", "PN_SPEC", 
            "CL_GAIN_DB", "PN_REF_MAX"))
    for k in range(len(f)):
        print("{:<20.0f}{:<20.2f}{:<20.2f}{:<20.2f}".format(f[k], pn_spec[k], 
                cl_db[k], pn_ref_max[k]))

    d = {}
    d['name'] = args.name
    d['frequency'] = args.frequency
    d['offset'] = f
    d['phase_noise'] = pn_ref_max
    
    with open(args.out_file, "w") as write_file:
        json.dump(d, write_file, indent=2)
    # print(json.dumps(d, indent=2))



