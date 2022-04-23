"""
=============
loop_filter.py
=============

get loop filter components and all PLL values
to a json file


"""


from pll.pll_calcs import *
import json

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bw', type=float, 
        help='loop filter bw in Hz', default=500e3)
    parser.add_argument('-p', '--pm', type=float, 
        help='phase margin in degrees', default=55)
    parser.add_argument('--kphi', type=float, 
        help='charge pump current in A/radian', default=5e-3)
    parser.add_argument('--kvco', type=float, 
        help='tuning sensitivity in Hz/V', default=1e6)
    parser.add_argument('-n', '--N', type=float, 
        help='N divider', default=200)
    parser.add_argument('-r', '--R', type=float, 
        help='R divider', default=1)
    parser.add_argument('--fref', type=float, 
        help='reference frequency', default=10e6)
    parser.add_argument('-g', '--gamma', type=float, 
        help='gamma value', default=1.115)
    parser.add_argument('--t31', type=float, 
        help='ratio of t3 to t1 (3rd and 4th only)', default=0.6)
    parser.add_argument('--t43', type=float, 
        help='ratio of t4 to t3 (4th only)', default=0.4)
    parser.add_argument('--order', type=int, 
        help='order of filter (2, 3 or 4)', default=2)
    parser.add_argument('-o', '--out_file', type=str, 
        help='file to save as json', default='data/mypll.json')

    args = parser.parse_args()
    if args.order == 2:
        t31 = 0
        t43 = 0
    elif args.order == 3:
        t31 = args.t31 
        t43 = 0
    else:
        t31 = args.t31 
        t43 = args.t43 
    flt = get_loop_filter_comps(args.bw, args.pm, args.N, args.kphi, 
            args.kvco, gamma=args.gamma, t31=t31, t43=t43, order=args.order)
    flt['bw'] = args.bw
    flt['pm'] = args.pm
    flt['N'] = args.N
    flt['R'] = args.R
    flt['fref'] = args.fref
    flt['fpfd'] = args.fref/args.R
    flt['fvco'] = (args.fref/args.R)*args.N
    flt['kphi'] = args.kphi
    flt['kvco'] = args.kvco
    flt['gamma'] = args.gamma
    flt['t31'] = args.t31
    flt['t43'] = args.t43
    
    with open(args.out_file, "w") as write_file:
        json.dump(flt, write_file, indent=2)
    print(json.dumps(flt, indent=2))


