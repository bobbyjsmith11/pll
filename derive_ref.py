"""
=============
derive_ref.py
=============

Derive the phase noise requirements of the reference


"""


from pll.phase_noise import load_pn_json 
from pll.pll_calcs import *
import numpy as np

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flt_json', type=str, 
        help='filter json', default='data/filter.json')
    parser.add_argument('--kphi', type=float, 
        help='charge pump current', default=5e-3)
    parser.add_argument('--kvco', type=float, 
        help='tuning sensitivity', default=1e6)
    parser.add_argument('-r', '--R', type=float, 
        help='R divider', default=1)
    parser.add_argument('-n', '--N', type=float, 
        help='N divider', default=40)
    parser.add_argument('--pn_spec_json', type=str, 
        help='phase noise spec json', default='data/navassa_wb.json')
    parser.add_argument('--margin_db', type=float, 
        help='margin to stay below spec in dB', default=0)
    
    parser.add_argument('--offset_start', type=int, 
        help='starting offset in Hz', default=100)
    parser.add_argument('--offset_stop', type=int, 
        help='ending offset in Hz', default=10e6)
    parser.add_argument('--num_points', type=int, 
        help='number of points in sim', default=100)

    args = parser.parse_args()

    spec = load_pn_json(args.pn_spec_json)
    f, pn_spec = interp_semilogx(spec['offset'], spec['phase_noise'],
                        x_range=[args.offset_start,args.offset_stop], 
                        num_points=args.num_points)

    flt = load_pn_json(args.flt_json)
    cl_db = get_closed_loop_gain(f, args.kphi, args.kvco, args.R, args.N, filt=flt)
    print(cl_db)
    # pn_ref_max = []
    # for k in range(len(f)):
    #     pn_ref_max.append(pn_spec[k] - cl_db[k] - args.margin_db)
    # print("{:<10}{:<10}{:<10}{:<10}".format("FREQ", "PN_SPEC", "CL_GAIN_DB", "PN_REF_MAX"))
    # for k in range(len(f)):
    #     print("{:<10.0f}{:<10.2f}{:<10.2f}{:<10.2f}".format(f[k], pn_spec[k], cl_db[k], pn_ref_max[k]))




