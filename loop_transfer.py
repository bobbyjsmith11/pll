"""
=============
loop_transfer.py
=============

Get the loop transfer function


"""


from pll.phase_noise import load_pn_json 
from plot_json import plot_loop_response 
from pll.pll_calcs import *
import numpy as np
import json

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pll_json', type=str, 
        help='pll json', default='data/pll.json')
    parser.add_argument('--fstart', type=int, 
        help='starting offset in Hz', default=100)
    parser.add_argument('--fstop', type=int, 
        help='ending offset in Hz', default=10e6)
    parser.add_argument('--num_points', type=int, 
        help='number of points in sim', default=100)
    parser.add_argument('--title', type=str, 
        help='plot title', default='add a title')
    
    parser.add_argument("--save", dest="save_json", action='store_true')
    parser.add_argument('-o', '--out_file', type=str, 
        help='file to save as json', default='data/xfer.json')

    args = parser.parse_args()
    pll = load_pn_json(args.pll_json)
    # print(flt)
    a = int(np.log10(args.fstart))
    b = int(np.log10(args.fstop))
    f = np.logspace(a, b, args.num_points)
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
    cl_g_db = get_closed_loop_transfer_fn(f, kphi, kvco, R, N, t2, a0, a1, a2=a2, a3=a3)
    g = get_open_loop_transfer_fn(f, kphi, kvco, N, t2, a0, a1, a2=a2, a3=a3, log=False)
    ol_g_db = 20*np.log10(np.absolute(g))
    pm = 180 + np.angle(g)*180/np.pi
    d = {}
    d['f'] =        list(f)
    d['cl_g_db'] =  list(cl_g_db)
    d['ol_g_db'] =  list(ol_g_db) 
    d['pm'] =       list(pm) 
    print("{:<14}{:<14}{:<14}{:<14}".format("FREQ", "OPEN_GAIN", "PHASE_MARGIN", "CLOSED"))
    for k in range(len(f)):
        print("{:<14.0f}{:<14.2f}{:<14.2f}{:<14.2f}".format(f[k], ol_g_db[k], pm[k], cl_g_db[k]))

    if args.save_json:
        with open(args.out_file, "w") as write_file:
            json.dump(d, write_file, indent=2)
    
    plot_loop_response(d, title=args.title)



