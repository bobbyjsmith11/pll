"""
============
pll_plots.py
============

:Description:
    Plot functions to visualize phase locked loops



"""

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pylab as plt
import numpy as np

from . import pll_calcs

def plot_arctan():
    x = np.linspace(-np.pi/2,np.pi/2,1000)
    y1 = np.arctan(x)
    y2 = np.arctan2(1, 1/x)
    fig, ax = plt.subplots()
    ax.plot(x,y1,'r',label='arctan')
    ax.plot(x,y2,'g',label='arctan2')
    legend = ax.legend()
    plt.grid(True)
    plt.show()

def plot_pll_gains(pll, 
                   fstart_Hz=100, 
                   fstop_Hz=10e6, 
                   coeffs=None,
                   comps=None, 
                   num_pts=100):
    """
    Plot the open and closed loop response of the PLL
    :Args:
        :fstart_Hz (int):           start offset frequency in Hz
        :fstop_Hz (int):            stop offset frequency in Hz
        :bw_Hz (int):               bandwidth of loop filter
        :pm_deg (int or float):     phase margin of loop filter
        :order (int):               order of loop filter <2 | 3>
        :N (int or float):          loop multiplication factor
        :R (int or float):          divider before phase detector
        :kphi (float):              charge pump gain in amps/radian
        :kvco (int or float):       vco tuning sensitiviy in Hz/volt
        :ptsPerDec (int):           resoution of graph
    """
    
    # pll = pll_calcs.PhaseLockedLoop(bw_Hz, pm_deg, kphi, kvco, N, R=R, order=order, gamma=gamma, t31=t31, t43=t43, active=active)
    d = pll.get_loop_gains(coeffs=coeffs, comps=comps, fstart=fstart_Hz, fstop=fstop_Hz, num_pts=num_pts)
    
    f = d['freqs']
    g = d['gains']
    p = d['phases']
    fz = d['fzero']
    pz = d['pzero']
    ref_cl = d['ref_cl']
    vco_cl = d['vco_cl']
    
    # open loop gain and phase 
    p = np.array(p)
    plt.ion()
    fig = plt.figure(figsize=(10,10))
    
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title("open-loop response")
    cor1 = 'tab:blue'
    ax1.set_xlabel("offset (Hz)")
    ax1.set_ylabel("gain (db)", color=cor1)
    ax1.semilogx(f, g, color=cor1, label='open-loop gain')
    ax1.semilogx([fz], 0, color=cor1, marker='o', linestyle='None')
    ax1.annotate(xy=[fz, 0],
                 s="   fc = {:.1f}".format(fz))
    ax1.tick_params(axis='y', labelcolor=cor1)
     
    ax2 = ax1.twinx()
    cor2 = 'tab:red'
    ax2.set_ylabel('phase (deg)', color=cor2)
    ax2.semilogx(f, p, color=cor2, label='open-loop phase')
    ax2.tick_params(axis='y', labelcolor=cor2)
    ax2.semilogx([fz], np.array([pz]) - 180, color=cor2, marker='o', linestyle='None')
    ax2.annotate(xy=[fz, np.array([pz]) - 180],
                 s="   pm = {:.1f}".format(pz))

    ax1.set_xlim(fstart_Hz, fstop_Hz) 
    ax1.grid(which='minor', alpha=0.2) 
    ax1.grid(which='major', alpha=0.5) 

    # closed loop gain
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.set_title('closed-loop response')
    ax3.set_xlabel("offset (Hz)")
    ax3.set_ylabel("gain (dB)")
    ax3.semilogx(f, ref_cl, label='reference')
    ax3.semilogx(f, vco_cl, label='vco')

    ax3.set_xlim(fstart_Hz, fstop_Hz) 
    ax3.grid(which='minor', alpha=0.2) 
    ax3.grid(which='major', alpha=0.5) 
    ax3.legend() 
    plt.tight_layout()
    plt.show()
    return d

def plot_pll_phase_noise(pll, 
                         fstart_Hz=100, 
                         fstop_Hz=10e6, 
                         ref_pn_dict=None, 
                         vco_pn_dict=None, 
                         pllFom=-222, 
                         pllFlicker=-119,
                         coeffs=None,
                         comps=None, 
                         num_pts=100):
    """
    Plot the open and closed loop response of the PLL
    :Args:
        :fstart_Hz (int):           start offset frequency in Hz
        :fstop_Hz (int):            stop offset frequency in Hz
        :ref_pn_dict (dict):            keys: 'f', 'pn'
        :vco_pn_dict (dict):            keys: 'f', 'pn'
        :ref_freq_Hz (float):           reference frequency in Hz
        :fstart_Hz (float):             offset start phase noise
        :fstop_Hz (float):              offset start phase noise
        :pllFom (float):                PLL IC Figure of Merit
        :pllFlicker (float):            PLL IC flicker
        :coeffs (dict):                 if provided, use pll coefficients
        :comps (dict):                  if provided, use pll loop filter components
        :num_pts (int):                 number of phase noise points to simulate across band
    """
    
    pns = pll.get_phase_noise(ref_pn_dict=ref_pn_dict,
                              vco_pn_dict=vco_pn_dict,
                              fstart_Hz=fstart_Hz,
                              fstop_Hz=fstop_Hz,
                              pllFom=pllFom,
                              pllFlicker=pllFlicker,
                              coeffs=coeffs,
                              comps=comps,
                              num_pts=num_pts)
    fout = pll.get_fpfd()*pll.N
    plt.ion()
    fig = plt.figure(figsize=(10,8))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Phase Noise at {:.3f} GHz'.format(fout/1e9))
    ax.set_xlabel("offset (Hz)")
    ax.set_ylabel("dBc/Hz")

    ax.semilogx(pns['freqs'], pns['reference'], linestyle=':', label='reference')
    ax.semilogx(pns['freqs'], pns['pll_ic'], linestyle=':', label='pll_ic')
    ax.semilogx(pns['freqs'], pns['vco'], linestyle=':',label='vco')
    ax.semilogx(pns['freqs'], pns['composite'], label='composite')
    
    ax.set_xlim(fstart_Hz, fstop_Hz) 
    ax.grid(which='minor', alpha=0.2) 
    ax.grid(which='major', alpha=0.5) 
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return pns
    

def plot_pn_file(fname, numPts=100):
    """
    """
    d = pll_calcs.load_pn_file(fname)
    freq_pts = d['f']
    pn_pts = d['pn']
    
    freq_array = pll_calcs.logify_freq(min(freq_pts), max(freq_pts), numPts)
    pn = pll_calcs.interp_phase_noise(freq_pts, pn_pts, freq_array)

    fstart_Hz = min(freq_array)
    fstop_Hz = max(freq_array)
    
    plt.ion()
    fig = plt.figure(figsize=(10,8))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(fname)
    ax.set_xlabel("offset (Hz")
    ax.set_ylabel("dBc/Hz")

    ax.semilogx(freq_array, pn, marker='o')
    
    ax.set_xlim(fstart_Hz, fstop_Hz) 
    ax.grid(which='minor', alpha=0.2) 
    ax.grid(which='major', alpha=0.5) 
    
    plt.tight_layout()
    plt.show()
    return freq_array, pn

def get_interpolated_pn(freq_pts, pn_pts, numPts):
    """
    """

    # freq_pts = map(float, freq_pts.split(','))
    # pn_pts = map(float, pn_pts.split(','))
    # f = get_freq_points_per_decade(fstart, fstop, ptsPerDec)
    f, pns = pll_calcs.interp_semilogx(freq_pts, pn_pts, num_points=numPts)
    return f, pns 












