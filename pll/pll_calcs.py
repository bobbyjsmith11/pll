#!/usr/bin/env python
"""
=============
pll_calcs.py
=============

Description
-----------
    Several uses for this module, including:
        
        * Designing loop filters
        * evaulating the closed and open loop transfer functions
        * simulating PLL phase noise (in conjunction with phase_noise.py)

    Most of the functions in this module are based off the work of
    Dean Banerjee in his book titled 
    *Pll Performance, Simulation, and Design: 5th edition*
    (https://www.ti.com.cn/cn/lit/ml/snaa106c/snaa106c.pdf)


Usage Examples
--------------

    Returns a dict of loop filter components
    
    >>> bw = 10e3   # loop bw
    >>> pm = 70 # degrees
    >>> kphi = 1e-3 # charge pump current
    >>> kvco = 60e6 # tuning sensitivity
    >>> flt = get_loop_filter_comps(bw, pm, N, kphi, kvco, gamma=1.024, order=2)

"""

import math
import numpy as np


def interp_semilogx(x, y, num_points, x_range=None):
    """ return a paired list of values each with length num_points where
    the values are linearly interpolated with the x axis in the log scale.
    Essentially, given arrays x and y, increase the resolution of to num_points
    :param x: array of x values
    :param y: array of y values (x and y need to be of equal length)
    :param num_points: int number of points for the entire range
    :param x_range: array of 2 elements: [x_lo, x_hi]
                    this is the range of x values which will be returned
    :return:
        tuple (xx, yy)
    """
    # first, log-ify the x axis
    log_x = []
    for item in x:
        # x_new, y_new = interp_linear(log_x, y, x_interp)
        log_x.append(np.log10(item))  
    if x_range == None:
        xmin = min(log_x)
    else:
        xmin = np.log10(min(x_range))
    if x_range == None:
        xmax = max(log_x)
    else:
        xmax = np.log10(max(x_range))
    f_log = np.linspace(xmin, xmax, num_points)
    y_interp = []

    x_log = []
    for x_val in x:
        x_log.append(np.log10(x_val))

    f = []
    for xx in f_log:
        f.append(10 ** (xx))
        y_temp = interp_linear(x_log, y, xx)
        y_interp.append(y_temp[1])

    return f, y_interp


# def plot_interp_semilogx(x, y, num_points=10):
#     """
#     """
#     x2, y2 = interp_semilogx(x, y, num_points=num_points)
#     plt.semilogx(x, y, '-bo', x2, y2, 'ro')
#     plt.grid(True)
#     plt.show() 


# def plot_interp_linear(x, y, x_interp):
#     """
#     """
#     x2, y2 = interp_linear(x, y, x_interp)
#     plt.plot(x, y, '-bo', [x2], [y2], 'ro')
#     plt.grid(True)
#     plt.show() 


def interp_linear(x, y, x_interp):
    """ linearly interpolate between two points with the
        Parameters
            x (list) - x values
            y (list) - y values
        Returns
            tuple (x, y) where x is x_interp and y is the
                interpolated y value
    """
    if len(x) != len(y):
        raise ValueError('x and y arrays need to be the same length')
    x_interp = float(x_interp)
    
    # x_interp is below the lowest point in x array
    if x_interp < x[0]:
        # find the first slope and interpolate below
        m = (y[1]-y[0])/(x[1]-x[0])
        y_interp = (x_interp - x[0])*m + y[0]
        return x_interp, y_interp
    
    # x_interp is above the highest point in x array
    elif x_interp > x[-1]:      
        # find the last slope and interpolate above
        m = (y[-1]-y[-2])/(x[-1]-x[-2])
        y_interp = (x_interp - x[-1])*m + y[-1]
        return x_interp, y_interp
    # x_interp is between 2 points in array
    else:
        for n in range(1,len(x)):
            if x[n] > x_interp:
                j = n 
                i = n-1 
                break
            elif x[n] == x_interp:
                return x[n], y[n]
        m = (y[j]-y[i])/(x[j]-x[i])
        y_interp = (x_interp - x[i])*m + y[i]
        return x_interp, y_interp


def power_sum( pdb_lst ):
    """ take a list of powers in dBm, add them
        in the linear domain and return the sum
        in log
    """
    sum_lin = 0
    for pdb in pdb_lst:
        sum_lin += 10**(float(pdb)/10)*1e-3
    return 10*np.log10(sum_lin/1e-3)


def get_g_from_coefficients(f_ar, kphi, kvco, t2, a0, a1, a2=0, a3=0): 
    """
    Return G(s) as np.array(dtype=np.complex) given the
    loop filter coefficients.
    See p.302, formula 33.8 
    Arguments:
        f_ar: np.array of frequency points
        kphi: charge pump current in A/radian
        kvco: vco tuning sensitivity in Hz/V
        t2: time constant c2*r2
        a0: filter coefficient
        a1: filter coefficient
        a2: filter coefficient (3rd & 4th order only)
        a3: filter coefficient (4th order only)
    Returns:
        np.array for G(s)
    """
    s = np.array(f_ar)*2*np.pi*1j        
    z = (1 + s*t2)/(s*(a3*s**3 + a2*s**2 + a1*s + a0))
    g = kphi*kvco*z/s
    return g

def get_coefficients_from_comps(c1=0, r2=0, c2=0, r3=0, c3=3, r4=0, c4=0):
    """
    given loop filter component values, return the coefficients as
    a dict
    """
    t2 = r2*c2
    a0 = c1+c2+c3+c4
    a1 = r2*c2*(c1+c3+c4) + r3*(c1+c2)*(c3+c4) + r4*c4*(c1+c2+c3)
    a2 = r2*r3*c1*c2*(c3+c4) + r4*c4*(c2*c3*r3+c1*c3*r3+c1*c2*r2+c2*c3*r2)
    a3 = r2*r3*r4*c1*c2*c3*c4
    return {'t2':t2, 'a0':a0, 'a1':a1, 'a2':a2, 'a3':a3}


def get_closed_loop_transfer_fn(f_ar, kphi, kvco, R, N, t2, a0, a1, 
                             a2=0, a3=0, log=True, include_r=True): 
    """
    Return the closed loop response from the reference in to the 
    PLL output
    Arguments:
        f_ar: np.array of frequency points
        kphi: charge pump current in A/radian
        kvco: vco tuning sensitivity in Hz/V
        t2: time constant c2*r2
        a0: filter coefficient
        a1: filter coefficient
        a2: filter coefficient (3rd & 4th order only)
        a3: filter coefficient (4th order only)
    Returns:
        np.array 
    """
    g = get_g_from_coefficients(f_ar, kphi, kvco, t2, a0, a1, a2=a2, a3=a3)
    if include_r:
        cl_r = (1.0/R)*(g/(1+g/N))
    else:
        cl_r = (g/(1+g/N))
    if log:
        return 20*np.log10(np.absolute(cl_r))
    return cl_r

def get_open_loop_transfer_fn(f_ar, kphi, kvco, N, t2, a0, a1, 
                              a2=0, a3=0, log=True):
    """
    Return the closed loop response from the reference in to the 
    PLL output
    Arguments:
        f_ar: np.array of frequency points
        kphi: charge pump current in A/radian
        kvco: vco tuning sensitivity in Hz/V
        t2: time constant c2*r2
        a0: filter coefficient
        a1: filter coefficient
        a2: filter coefficient (3rd & 4th order only)
        a3: filter coefficient (4th order only)
    Returns:
        np.array 
    """
    g = get_g_from_coefficients(f_ar, kphi, kvco, t2, a0, a1, a2=a2, a3=a3)
    ol_r = g/N
    if log:
        return 20*np.log10(np.absolute(ol_r))
    return ol_r 


def get_vco_transfer_fn(f_ar, kphi, kvco, R, N, t2, a0, a1, 
                             a2=0, a3=0, log=True): 
    """
    Return the closed loop response from the VCO to the
    PLL output
    Arguments:
        f_ar: np.array of frequency points
        kphi: charge pump current in A/radian
        kvco: vco tuning sensitivity in Hz/V
        t2: time constant c2*r2
        a0: filter coefficient
        a1: filter coefficient
        a2: filter coefficient (3rd & 4th order only)
        a3: filter coefficient (4th order only)
    Returns:
        np.array 
    """
    g = get_g_from_coefficients(f_ar, kphi, kvco, t2, a0, a1, a2=a2, a3=a3)
    cl_vco = 1.0/(1+g/N)
    if log:
        return 20*np.log10(np.absolute(cl_vco))
    return cl_vco


def get_loop_filter_comps(bw, pm, N, kphi, kvco, gamma=1.115, t31=None, t43=None, order=2):
    """
    Return the loop filter component values as a dict
    Arguments:
        bw (float): loop bandwidth in Hz
        pm (float): phase margin in degrees
        N (int): loop multiplication factor
        gamma (float): optimization factor. Relates to the phase margin
                at the loop bw
        t31 (float): radio of pole t3 to t1. Only valid for 3rd
                and 4th order filters
        t43 (float): radio of pole t4 to t3. Only valid for 4th
                order filters
    """
    d = {}
    omega_c = 2*math.pi*bw

    # solve for time constants
    t1 = calc_t1(bw, pm, gamma=gamma, t31=t31, t43=t43, iters=100)
    t3 = t1*t31 
    t4 = t3*t43 
    t2 = gamma/((omega_c**2)*(t1 + t3 + t4))

    # solve for coefficients
    a0 = calc_a0(kphi, kvco, N, bw, t1, t2, t3=t3, t4=t4)
    a1 = a0*(t1+t3+t4)
    a2 = a0*(t1*t3+t1*t4+t3*t4)
    a3 = a0*t1*t3*t4

    # for 2nd order t31=0
    # no value for a1 for 2nd order
    if order == 2:
        a2=0
        a3=0

    if order == 2:
        c1 = a0*t1/t2
        c3 = 0
        r3 = 0
    elif order == 3:
        c1 = (a2/(t2**2))*(1+np.sqrt(1+(t2/a2)*(t2*a0-a1)))
        c3 = (-(t2**2)*(c1**2) + t2*a1*c1 - a2*a0)/((t2**2)*c1 - a2)
    elif order == 4:
        t2 = gamma/((omega_c**2)*(t1+t3+t4))

    if order == 3:
        r3 = a2/(c1*c3*t2)
    
    if (order == 2) or (order == 3):
        c2 = a0-c1-c3
        r2 = t2/c2
        c4 = 0
        r4 = 0

    if order == 4:
        # calculate C1 and R3
        c1_t3, r3_t3 = calc_c1_r3(a0,t1,t2,t3)
        c1_t4, r3_t4 = calc_c1_r3(a0,t1,t2,t4)
        c1 = (c1_t3 + c1_t4)/2
        r3 = (r3_t3 + r3_t4)/2
        
        # calculate C2 and R4
        c2, c3 = calc_c2_c3(a0, a1, a2, a3, t2, r3, c1)
        c4 = a0-c1-c2-c3
        r2 = t2/c2
        r4 = a3/(t2*r3*c1*c3*c4)

    # missing 4th order
    # c1, c3, r2, r3, c4

    d['t1'] = t1
    d['t2'] = t2
    d['t3'] = t3
    d['t4'] = t4
    d['a0'] = a0
    d['a1'] = a1
    d['a2'] = a2
    d['a3'] = a3
    d['c1'] = c1 
    d['c2'] = c2 
    d['c3'] = c3 
    d['c4'] = c4 
    d['r2'] = r2 
    d['r3'] = r3 
    d['r4'] = r4 
    return d
    
    # solve for components
    d['c1'] = self.calc_c1(d['a0'],
                           d['a1'],
                           d['a2'],
                           d['t2'])

    d['c3'] = self.calc_c3( d['a0'],
                            d['a1'],
                            d['a2'],
                            d['t2'],
                            d['c1'] )

    d['c2'] = d['a0'] - d['c1'] - d['c3']


    d['r2'] = d['t2']/d['c2']

    d['r3'] = d['a2']/(d['c1']*d['c3']*d['t2'])

    d['t4'] = 0
    d['a3'] = 0
    d['r4'] = 0
    d['c4'] = 0

    return d


def func_t1(x, bw, pm, t31=0.6, t43=0, gamma=1.136):
    """ simulate t1. This function is used to 
    numerically solve for T1. 
    Equation 22.31 in Dean Banerjee's Book
    :Parameters:
        x (float) - guess at t1
        bw (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        t31 (float) - ratio of t3 to t1
        t43 (float) - ratio of t4 to t3 (4th order only)
        gamma (float) - optimization factor (1.136)
    :Returns:
    updated value for t1 based on guess (float)
    """
    omega_c = 2*math.pi*bw
    phi = pm*np.pi/180
    val = np.arctan(gamma/(omega_c*x*(1+t31+t43+t43*t31)) ) - \
            np.arctan(omega_c*x) - \
            np.arctan(omega_c*x*t31) - \
            np.arctan(omega_c*x*t31*t43) - \
            phi
    return val


def calc_t1(bw, pm, gamma=1.024, t31=0, t43=0, iters=100):
    """
    :Arguments:
        bw (float): loop filter bandwith
        pm (float): phase margin in degrees
        gamma (float):  optimization factor (default=1.024)
        t31 (float): ratio of t3 to t1 (3rd and 4th only)
        t43 (float): ratio of t4 to t3 (4th only)
    """
    a = 1e-15   # initial guess for a
    b = 1.0       # initial guess for b
    fa = func_t1(a, bw, pm, t31=t31, t43=t43, gamma=gamma)
    fb = func_t1(b, bw, pm, t31=t31, t43=t43, gamma=gamma)
    for i in range(iters):
        t1 = (a+b)/2
        if (func_t1(t1, bw, pm, t31=t31, t43=t43, gamma=gamma) < 0):
            b = t1
        else:
            a = t1
    return t1


def calc_a0(kphi, kvco, N, bw, t1, t2, t3=0, t4=0):
    """
    return the a0 coefficient
    :Arguments:
        kphi (float): charge pump current in A/radian
        kvco (float): VCO tuning senstivity in Hz/V
        bw (float): loop filter bandwith
        t1 (float): pole t1
        t2 (float): pole t2
        t3 (float): pole t3 (3rd and 4th only)
        t4 (float): pole t4 (4th only)
    """
    omega_c = 2*math.pi*bw
    k1 = kphi*kvco/((omega_c**2)*(N))
    k2 = np.sqrt(
            (1+(omega_c*t2)**2)/((1+(omega_c*t1)**2)* \
            (1+(omega_c*t3)**2)*(1+(omega_c*t4)**2) ) 
                )
    return k1*k2

def calc_c1_r3(a0, t1, t2, tpole):
    """
    Returns value for C1 and R3 as a tuple.
    THIS FUNCTION IS FOR 4th ORDER LOOPS ONLY.
    :Arguments:
        a0 (float): coefficient
        t1 (float): pole t1
        t2 (float): pole t2
        tpole (float): pole to use (t3 or t4)
    """
    a1_t = a0*(t1+tpole)
    a2_t = a0*t1*tpole
    c1_t = (a2_t/(t2**2))*(1 + np.sqrt(1 + (t2/a2_t)*(t2*a0 - a1_t)) )
    c3_t = (-1*(t2**2)*(c1_t**2) + t2*a1_t*c1_t - a2_t*a0)/((t2**2)*c1_t - a2_t)
    r3_t = a2_t/(c1_t*c3_t*t2)
    return c1_t, r3_t


def calc_c2_c3(a0, a1, a2, a3, t2, r3, c1):
    """
    Returns value for C2 and R3 as a tuple.
    THIS FUNCTION IS FOR 4th ORDER LOOPS ONLY.
    :Arguments:
        a0 (float): coefficient
        a1 (float): coefficient
        a2 (float): coefficient
        a3 (float): coefficient
        t2 (float): pole t2
        r3 (float): resistor r3 in Ohms
        c3 (float): capacitor c3 in Farads
    """
    k0 = (a2/a3) - 1.0/t2 - 1.0/(c1*r3) - (a0 - c1)*t2*r3*c1/a3
    k1 = a1 - t2*a0 - a3/(t2*r3*c1) - (a0 - c1)*r3*c1
    a = a3/((t2*c1)**2)
    b = t2 + r3*(c1 - a0) + (a3/(t2*c1))*((1.0/t2) - k0)
    c = k1 - (k0*a3)/t2
    c2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    c3 = (t2*a3*c1)/(r3*(k0*t2*a3*c1 - c2*(a3 - r3*((t2*c1)**2))))
    return c2, c3

###################################################
# SIMULATIONS
###################################################

def sim_phase_noise(offset, ref_pn, vco_pn, pll_fom, pll_flick, pll_dict):
    """ 
    simulate an arbitrary phase-locked loop using either
    filter coefficients or component values. return 3 lists:
    f (frequencies), g_ol (open-loop gain), phases (open-loop phases)  
    :Arguments:
        f (list): offset frequencies in Hz
        ref_pn (list): reference phase noise
    """
    fpfd = pll_dict['fpfd'] 
    kphi = pll_dict['kphi'] 
    kvco = pll_dict['kvco'] 
    R = pll_dict['R'] 
    N = pll_dict['N'] 
    t2 = pll_dict['t2'] 
    a0 = pll_dict['a0'] 
    a1 = pll_dict['a1'] 
    a2 = pll_dict['a2'] 
    a3 = pll_dict['a3'] 

    cl_r_db = get_closed_loop_transfer_fn(offset, kphi, kvco, R, N, t2, 
            a0, a1, a2=a2, a3=a3, include_r=True)
    ref_pn_out = ref_pn + cl_r_db
    ref_pn = []
    ref_pn.extend(ref_pn_out)

    cl_ic_db = get_closed_loop_transfer_fn(offset, kphi, kvco, R, N, t2, 
            a0, a1, a2=a2, a3=a3, include_r=False)
    ic_pn_out = pll_fom + 10*np.log10(fpfd) + cl_ic_db
    ic_pn = []
    ic_pn.extend(ic_pn_out)

    ic_flick_out = pll_flick + 20*np.log10(fpfd) - 10*np.log10(offset) + \
                    cl_ic_db
    ic_flick = []
    ic_flick.extend(ic_flick_out)

    # # Closed-loop VCO transfer gain
    cl_vco_db = get_vco_transfer_fn(offset, kphi, kvco, R, N, t2, 
            a0, a1, a2=a2, a3=a3)
    vco_pn_out = vco_pn + cl_vco_db
    vco_pn = []
    vco_pn.extend(vco_pn_out)

    comp_pn = []
    for i in range(len(offset)):
        comp_pn.append(power_sum([ref_pn_out[i], vco_pn_out[i],
                                  ic_pn_out[i], ic_flick[i]]))

    d = {'offset': offset,
         'ref_pn': ref_pn,
         'vco_pn': vco_pn,
         'ic_pn': ic_pn,
         'ic_flick': ic_flick,
         'comp_pn': comp_pn}
    return d



