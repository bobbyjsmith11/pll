
from pll.phase_noise import *
from pll.pll_plots import *
import argparse

def quick_comparison(fcomp=100e6):
    pn_b = load_pn_json('data/BVCB100MDEACCBN.json')
    pn_nv = load_pn_json('data/navassa_wb.json')
    pn_abc = load_pn_json('data/asvtx12.json')
    mytitle = "Phase noise comparison at {:<0.3f} MHz".format(fcomp/1e6)
    plot_phase_noise([pn_nv, pn_abc, pn_b], title=mytitle, freq_comp=fcomp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', type=float, help='comparison frequency', default=None)
    args = parser.parse_args()
    quick_comparison(fcomp=args.freq)