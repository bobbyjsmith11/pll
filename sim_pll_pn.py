from pll.phase_noise import *
from pll.pll_plots import *
from pll.pll_calcs import *
import numpy as np

import argparse

def quick_comparison(fcomp=100e6):
    pn_b = load_pn_json('data/BVCB100MDEACCBN.json')
    pn_nv = load_pn_json('data/navassa_wb.json')
    pn_abc = load_pn_json('data/asvtx12.json')
    mytitle = "Phase noise comparison at {:<0.3f} MHz".format(fcomp/1e6)
    plot_phase_noise([pn_nv, pn_abc, pn_b], title=mytitle, freq_comp=fcomp)

def simulate_pll_phase_noise(ref_json, vco_json, pll_fom, fpfd, fcarrier,
                             fmin=1e3, fmax=10e6, loop_bw=100e3, pm=55,  
                             num_points=100, flicker=-500):
    """
    :param ref_json: json file for reference
    :param ref_vco: json file for vco
    :param pll_fom: figure-of-merit for PLL
    :param loop_bw: PLL loop filter cutoff
    :param pm: PLL phase margin
    :param fpfd: comparison frequency in Hz
    :param fcarrier: VCO frequency in Hz
    :param fmin: minimum offset
    :param fmax: maximum offset
    :param num_points: int number of points
    """
    ref = load_pn_json(ref_json)
    # translate the phase noise to the comparison frequency
    # calculates noise after the R divider
    f, pn = translate_phase_noise(ref, fpfd)
    freq, pn_r = interp_semilogx(f, pn, x_range=[fmin, fmax], num_points=num_points)

    vco = load_pn_json(vco_json)
    # translate vco phase noise to fcarrier
    f, pn = translate_phase_noise(vco, fcarrier)
    freq, pn_vco = interp_semilogx(f, pn, x_range=[fmin, fmax], num_points=num_points)

    # dummy constants
    KPHI = 5e-3
    KVCO = 1e6
    N = fcarrier/fpfd
    pll3 = PllThirdOrderPassive(loop_bw, pm, KPHI, KVCO, N)
    c = pll3.calc_components()
    c['flt_type'] = 'passive'
    R = 1
    f, ref_pn, vco_pn, ic_pn, ic_flick, total_pn = simulate_phase_noise(freq,
                                                                        pn_r,
                                                                        pn_vco,
                                                                        pll_fom,
                                                                        flicker,
                                                                        KPHI,
                                                                        KVCO,
                                                                        fpfd,
                                                                        N,
                                                                        R, filt=c)
    ret_lst = []
    ret_lst.append({'name': 'reference', 'offset': f, 'phase_noise': ref_pn})
    ret_lst.append({'name': 'vco', 'offset': f, 'phase_noise': vco_pn})
    ret_lst.append({'name': 'pll', 'offset': f, 'phase_noise': ic_pn})
    # ret_lst.append({'name': 'flicker', 'offset': f, 'phase_noise': ic_flick})
    ret_lst.append({'name': 'total', 'offset': f, 'phase_noise': total_pn})

    return ret_lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref_json', type=str, help='reference phase noise json', default='data/asvtx12.json')
    parser.add_argument('-v', '--vco_json', type=str, help='vco phase noise json', default='data/navassa_vco.json')
    parser.add_argument('--fpfd', type=float, help='comparison frequency in Hz', default=40e6)
    parser.add_argument('--loop_bw', type=float, help='PLL loop bandwidth', default=100e3)
    parser.add_argument('--pm', type=float, help='PLL phase margin in degrees', default=55)
    parser.add_argument('--pll_fom', type=float, help='PLL figure-of-merit in dBc/Hz', default=-221)
    parser.add_argument('--fvco', type=float, help='carrier frequency in Hz', default=2000e6)
    parser.add_argument('--offset_start', type=float, help='starting offset in Hz', default=100)
    parser.add_argument('--offset_stop', type=float, help='ending offset in Hz', default=10e6)
    parser.add_argument('--y_max', type=float, help='top of y scale for phase noise (dBc/Hz)', default=-60)
    parser.add_argument('--y_min', type=float, help='bottom of y scale for phase noise (dBc/Hz)', default=-160)
    parser.add_argument('--y_tick', type=float, help='increment on the y scale', default=10)
    parser.add_argument('--num_points', type=int, help='number of points in sim', default=1000)
    args = parser.parse_args()
    title_str = "fvco: {:<0.6f} MHz    ".format(args.fvco/1e6)
    title_str += "fpfd: {:<0.6f} MHz<br>".format(args.fpfd/1e6)
    title_str += "loop_bw: {:<0.3f} kHz    ".format(args.loop_bw/1e3)
    title_str += "phase margin: {:<0.1f} degrees<br>".format(args.pm)
    title_str += "PLL FOM: {:<0.1f} dBc/Hz    ".format(args.pll_fom)
    lst_pn = simulate_pll_phase_noise(args.ref_json,
                                      args.vco_json,
                                      args.pll_fom,
                                      args.fpfd,
                                      args.fvco,
                                      fmin=args.offset_start,
                                      fmax=args.offset_stop,
                                      loop_bw=args.loop_bw,
                                      pm=args.pm,
                                      num_points=args.num_points)
    plot_phase_noise(lst_pn, title=title_str, ylim=[args.y_min, args.y_max], ytick=args.y_tick)
