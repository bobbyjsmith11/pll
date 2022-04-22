"""
=============
loop_filter.py
=============

get loop filter components


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
    parser.add_argument('-g', '--gamma', type=float, 
        help='gamma value', default=1.115)
    parser.add_argument('--order', type=int, 
        help='order of filter (2, 3 or 4)', default=2)
    parser.add_argument('-o', '--out_file', type=str, 
        help='file to save as json', default='data/myfilter.json')

    args = parser.parse_args()

    if args.order == 2:
        pll = PllSecondOrderPassive(args.bw,
                                     args.pm,
                                     args.kphi,
                                     args.kvco,
                                     args.N,
                                     gamma=args.gamma)
    elif args.order == 3:
        pll = PllThirdOrderPassive(args.bw,
                                    args.pm,
                                    args.kphi,
                                    args.kvco,
                                    args.N,
                                    gamma=args.gamma)
    elif args.order == 4:
        pll = PllFourthOrderPassive(args.bw,
                                     args.pm,
                                     args.kphi,
                                     args.kvco,
                                     args.N,
                                     gamma=args.gamma)
    else:
        raise ValueError('{} invalid value for order'.format(args.order))

    flt = pll.calc_components()
    with open(args.out_file, "w") as write_file:
        json.dump(flt, write_file, indent=2)
    print(json.dumps(flt, indent=2))


