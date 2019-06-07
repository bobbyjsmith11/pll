#
#-*- coding: utf-8 -*-
# 
# -------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------
import math

import numpy as np

def solveForComponents(fc, pm, kphi, kvco, N, gamma, loop_type='passive2'):
    """
    :Parameters:
    loop_type (str) - 
        * passive2 - 2nd order passive
        * passive3 - 3rd order passive
        * passive4 - 4th order passive
        * active2 -  2nd order active
        * active3 -  3rd order active
        * active4 -  4th order active
    fc (float) - 0dB crossover frequency in Hz
    pm (float) - phase margin in degrees
    kphi (float) - charge pump gain in Amps per radian
    kvco (float) - vco tuning sensitivity in Hz/V
    N (int) - loop multiplication ratio
    gamma (float) - optimization factor (1.024 default)
    """
    
    if loop_type == 'passive2':
        pll = PhaseLockedLoop(fc, pm, kphi, kvco, N, gamma=gamma, active=False)
        coeffs = calc_coeffs_from_loop_vals(pll.get_loop_values())
        d =  calc_components_from_coeffs(coeffs, order=2, active=False)
    elif loop_type == 'passive3':
        pll = PhaseLockedLoop(fc, pm, kphi, kvco, N, order=3, t31=0.6, gamma=gamma, active=False)
        coeffs = calc_coeffs_from_loop_vals(pll.get_loop_values())
        d =  calc_components_from_coeffs(coeffs, order=3, active=False)
    elif loop_type == 'passive4':
        pll = PhaseLockedLoop(fc, pm, kphi, kvco, N, order=4, t31=0.4, t43=0.4, gamma=gamma, active=False)
        coeffs = calc_coeffs_from_loop_vals(pll.get_loop_values())
        d =  calc_components_from_coeffs(coeffs, order=4, active=False)
    return d 

class PhaseLockedLoop(object):
    """
    """
    def __init__(self, fc, pm, kphi, kvco, N, ref_freq_Hz=10e6, R=1, order=2, t31=0, t43=0, gamma=1.024, active=False):
        """
        :args:
            :fc (float):            loop bandwidth in Hz
            :pm (float):            phase margin in degrees 
            :kphi (float):          charge pump gain in amps per radian
            :kvco (float):          vco tuning sensitiviy in Hz/V
            :N (float):             loop multiplication ratio
            :R (float):             reference divider
            :order (int):           order of loop filter < 2 | 3 | 4>
            :t31 (float):           ratio of t3 to t1 (if 3rd order or greater)
            :t43 (float):           ratio of t4 to t4 (if 4th order)
            :gamma (float):         optimization factor
            :active (bool):         active filter if True
        """
        self.fc = fc
        self.pm = pm
        self.kphi = kphi
        self.kvco = kvco
        self.N = N
        self.R = R
        self.gamma = gamma
        self.order = order
        self.active = active
        self.t31 = t31
        self.t43 = t43
        self.ref_freq_Hz = ref_freq_Hz
        
    def get_loop_values(self):
        """
        """
        d = {}
        d['fc'] =       self.fc
        d['pm'] =       self.pm
        d['kphi'] =     self.kphi
        d['kvco'] =     self.kvco
        d['N'] =        self.N
        d['R'] =        self.R
        d['gamma'] =    self.gamma
        d['t31'] =      self.t31
        d['t43'] =      self.t43
        d['order'] =    self.order
        d['active'] =   self.active
        return d

    def get_loop_gains(self, coeffs=None, comps=None, fstart=100, fstop=10e6, num_pts=100):
        """
        get the closed loop gain from the reference input to the 
        vco output in dB.
        """
        if (coeffs == None) and (comps == None):
            coeffs = calc_coeffs_from_loop_vals(self.get_loop_values())
        elif (coeffs == None) and not(comps == None):
            coeffs = calc_coeffs_from_comps(comps, order=self.order, active=self.active)
        
        # f = get_log_freq_points(fstart, fstop, num_pts=num_pts)
        f = logify_freq(fstart, fstop, num_pts=num_pts)
        # loop filter impedance
        # z = (1 + s*t2)/(s*(a[3]*s**3 + a[2]*s**2 + a[1]*s + a[0])) 
        z = calculateZ(f,  
                       coeffs['t2'], 
                       coeffs['a0'], 
                       coeffs['a1'],
                       coeffs['a2'],
                       coeffs['a3'])
    
        # G(s)
        # g = kphi*kvco*z/s
        g = calculateG(f, z, self.kphi, self.kvco)
    
        # # Open-loop gain 
        g_ol = g/self.N
        g_ol_db = 20*np.log10(np.absolute(g_ol))
        # ph_ol = 180 + np.unwrap(np.angle(g_ol))*180/np.pi
        ph_ol = np.unwrap(np.angle(g_ol))*180/np.pi
    
        # # Closed-loop reference transfer gain
        cl_r = (1.0/self.R)*(g/(1+g/self.N))
        cl_r_db = 20*np.log10(np.absolute(cl_r))
   
        # # Gain of the IC figure of merit (used for phase noise)
        cl_ic = (g/(1+g/self.N))
        cl_ic_db = 20*np.log10(np.absolute(cl_r)) + 20*np.log10(self.R)
        
        # # Closed-loop VCO transfer gain
        cl_vco = 1.0/(1+g/self.N)
        cl_vco_db = 20*np.log10(np.absolute(cl_vco))
    
        # convert gains and phases to lists
        # cannot return numpy array to javascript
        g = []
        p = []
        g.extend(g_ol_db)
        p.extend(ph_ol)
        try:
            fz, pz = getInterpolatedFzeroPzero(f, g, p)
        except Exception as e:
            fz = 0
            pz = 0
        ref_cl = []
        vco_cl = []
        ref_cl.extend(cl_r_db)
        vco_cl.extend(cl_vco_db)
        d = { 'freqs':f,
              'gains':g,
              'phases':p,
              'fzero': fz,
              'pzero': pz,
              'ref_cl': ref_cl,
              'ic_cl': cl_ic_db,
              'vco_cl': vco_cl,
            }
        return d

    def get_phase_noise(self,   
                        ref_pn_dict=None, 
                        vco_pn_dict=None, 
                        ref_freq_Hz=10e6, 
                        fstart_Hz=100, 
                        fstop_Hz=10e6, 
                        pllFom=-229, 
                        pllFlicker=None,
                        coeffs=None,
                        comps=None, 
                        num_pts=100):
        """
        :Args:
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
        fpfd = self.ref_freq_Hz/self.R
        
        # # get interpolated phase noise at the simulation frequency points
        # f, refPn = interp_semilogx(ref_pn_dict['f'], ref_pn_dict['pn'], num_points=num_points)
        # f, vcoPn = interp_semilogx(vco_pn_dict['f'], vco_pn_dict['pn'], num_points=num_points)
        
        # get loop gains  
        gains = self.get_loop_gains(coeffs=coeffs, comps=comps, fstart=fstart_Hz, fstop=fstop_Hz, num_pts=num_pts)
      
        if ref_pn_dict == None:
            ref_pn_dict = {'f':     [fstart_Hz, fstop_Hz],
                           'pn':    [-200, -200]}
        if vco_pn_dict == None:
            vco_pn_dict = {'f':     [fstart_Hz, fstop_Hz],
                           'pn':    [-200, -200]}
        
        refPn = interp_phase_noise(ref_pn_dict['f'], ref_pn_dict['pn'], gains['freqs'])
        vcoPn = interp_phase_noise(vco_pn_dict['f'], vco_pn_dict['pn'], gains['freqs'])
        
        # multiply refPn by gains['ref_cl']
        refPnOut = np.array(refPn) + np.array(gains['ref_cl'])
        refPn = []
        refPn.extend(refPnOut)
    
        icPnOut = pllFom + 10*np.log10(fpfd) + np.array(gains['ic_cl'])
        icPn = []
        icPn.extend(icPnOut)    # need to make a list so it can go be sent via web
   
        icFlick = []
        if pllFlicker == None:
            icFlickerOut = -200*np.ones(len(gains['freqs']))
        else:
            icFlickerOut = pllFlicker + 20*np.log10(fpfd) - 10*np.log10(10e3/np.array(gains['freqs'])) + np.array(gains['ic_cl'])
        icFlick.extend(icFlickerOut)
    
        vcoPnOut = np.array(vcoPn) + np.array(gains['vco_cl'])
        vcoPn = []
        vcoPn.extend(vcoPnOut)
    
        compPn = []
        for i in range(len(gains['freqs'])):
            compPn.append(power_sum( [ refPnOut[i],
                                      vcoPnOut[i],
                                      icPnOut[i],
                                      icFlickerOut[i] ] ))
   
        d = { 'freqs':      gains['freqs'],
              'reference':  refPnOut,
              'vco':        vcoPnOut,
              'pll_ic':     icPn,
              'flicker':    icFlick,
              'composite':  compPn,
            }
        
        return d

def callSimulatePll(d):
    """
    """
    fstart =            d['fstart'] 
    fstop =             d['fstop'] 
    ptsPerDec =         d['ptsPerDec'] 
    kphi =              d['kphi'] 
    kvco =              d['kvco'] 
    N =                 d['N'] 
    R =                 d['R'] 
    flt_type =          d['flt_type'] 
    c1 =                d['c1'] 
    c2 =                d['c2'] 
    c3 =                d['c3'] 
    c4 =                d['c4'] 
    r2 =                d['r2'] 
    r3 =                d['r3'] 
    r4 =                d['r4'] 
    flt =   {
            'c1':c1,
            'c2':c2,
            'c3':c3,
            'c4':c4,
            'r2':r2,
            'r3':r3,
            'r4':r4,
            'flt_type':flt_type
            }

    if (r3 == 0) and (r4 == 0):
        order = 2
        gamma = 1.024
    elif (r3 != 0) and (r4 == 0):
        order = 3
        gamma = 1.136
    else:
        order = 4
        gamma = 1.136
    
    pll = PhaseLockedLoop(fc, 48, kphi, kvco, N, gamma=gamma, order=order, active=False)
    d = pll.get_loop_gains(comps=d, fstart=fstart, fstop=fstop, ptsPerDec=ptsPerDec)
    
    return d

def getInterpolatedFzeroPzero( f, g, p ):
    """ look at the points of f, g and p surrounding where
    g crosses zero and interpolate f and p at 0
    """
    ndx = getIndexZeroDbCrossover( g )
    f_zero_db = None 
    g_zero_db = None 
    p_zero_db = None 
    if ndx != None:
        f_zero_db = f[ndx]
        g_zero_db = g[ndx]
        p_zero_db = p[ndx]
    
    newf = f[ndx-1:ndx+1]
    newp = p[ndx-1:ndx+1]
    newg = g[ndx-1:ndx+1]
    mg = (newg[1] - newg[0])/(newf[1] - newf[0])
    mp = (newp[1] - newp[0])/(newf[1] - newf[0])
    
    fz = newf[0] - (newg[0]/mg)
    deltaf = fz - newf[0]   # distance from newf[0] to 0db crossover
    pz = 180 + mp*deltaf + newp[0]
    return fz, pz

def getIndexZeroDbCrossover( g ):
    for i in range(len(g)):
        if g[i] <= 0:
            return i
    return None

def plotSimulatePhaseNoise():
    kphi = 5e-3
    kvco = 60e6
    N = 200  
    R = 1  
    fpfd = 10e6/R

    flt = {
            'c1':368e-12,
            'c2':6.75e-9,
            'c3':76.6e-12,
            'c4':44.7e-12,
            'r2':526,
            'r3':1.35e3,
            'r4':3.4e3,
            'flt_type':"passive" 
           }

    f =         [ 10, 100, 1e3, 10e3, 100e3, 1e6, 10e6, 100e6 ]
    refPnIn =   [ -138, -158, -163, -165, -165, -165, -165, -165 ]
    vcoPnIn =   [ -10, -30, -60, -90, -120, -140, -160, -162 ]

    pllFom =        -227
    pllFlicker =    -268

    f, refPn, vcoPn, icPn, icFlick, comp = simulatePhaseNoise( f,
                                                               refPnIn,
                                                               vcoPnIn,
                                                               pllFom,
                                                               pllFlicker,
                                                               kphi,
                                                               kvco,
                                                               fpfd,
                                                               N,
                                                               R,
                                                               filt=flt )

    return f, refPn, vcoPn, icPn, icFlick, comp

def interp_semilogx(x, y, num_points):
    """ return a paired list of values each with length num_points where
    the values are linearly interpolated with the x axis in the log scale.
    Essentially, given arrays x and y, increase the resolution of to num_points
    Parameters:
        x (list) - x values (frequencies)
        y (list) - y values (phase noise or gain in dB)
    Note: x and y have a semilog X relationship.
    Returns:
        tuple of lists (freqs, values)
    """
    # first, log-ify the x axis
    log_x = []
    for item in x:
        log_x.append(math.log10(item))    # x_new, y_new = interp_linear(log_x, y, x_interp) 
    xmin = min(log_x)
    xmax = max(log_x)
    f_log = linspace(xmin, xmax, num_points)
    y_interp = []

    x_log = []
    for x_val in x:
        x_log.append(math.log10(x_val))
    
    f = []
    for xx in f_log:
        f.append(10**(xx))
        y_temp = interp_linear(x_log, y, xx)
        y_interp.append(y_temp[1])

    # f = [xx**(f_log) for xx in f_log]
    return f, y_interp

def linspace(a, b, num_points):
    """ return a list of linearly spaced values
    between a and b having num_points points
    """
    inc = (float(b) - float(a))/(num_points-1)
    ret_ar = []
    for i in range(num_points):
        ret_ar.append(a + i*inc)
    return ret_ar

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
    if x_interp < x[0]:         # x_interp is below the lowest point in x array
        # find the first slope and interpolate below
        m = (y[1]-y[0])/(x[1]-x[0])
        y_interp = (x_interp - x[0])*m + y[0]
        return x_interp, y_interp
    elif x_interp > x[-1]:      # x_interp is above the highest point in x array
        # find the last slope and interpolate above
        m = (y[-1]-y[-2])/(x[-1]-x[-2])
        y_interp = (x_interp - x[-1])*m + y[-1]
        return x_interp, y_interp
    else:                       # x_interp is between 2 points in array
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

def get_freq_points_per_decade(fstart, fstop, ptsPerDec):
    """ return an array of frequencies starting at the 
        nearest decade of 10 from fstart and ending at the 
        nearest decade of 10 at fstop. Each decade has
        ptsPerDec tpoints.
    :Arguments:
    fstart (float)
    fstop (float)
    ptsPerDec (int)
    """
    fstart = float(fstart)
    fstop = float(fstop)
    ptsPerDec = int(ptsPerDec)
    num_decades = round(math.log10(fstop/fstart)/math.log10(10),0)
    ar = []
    istart = int(math.log10(fstart)/math.log10(10))
    ar.append(10**istart)
    for i in range(istart,int(num_decades)+1):
        newDec = 10**i
        nextDec = 10**(i+1)
        inc = float((nextDec - newDec))/float(ptsPerDec-1)
        for j in range(1,ptsPerDec):
            val = newDec + j*inc
            ar.append(float(val))
    return ar    

def simulatePhaseNoise2(f, 
                        refPn,
                        vcoPn,
                        pllFom,
                        kphi,
                        kvco,
                        fpfd,
                        N,
                        R,
                        filt=None,
                        coeffs=None,
                        numPts=1000 ):
    """ simulate an arbitrary phase-locked loop using either
    filter coefficients or component values. return 3 lists:
    f (frequencies), g_ol (open-loop gain), phases (open-loop phases)  
    """
    if coeffs == None:
        c1 = filt['c1']
        c2 = filt['c2']
        r2 = filt['r2']
        if 'r3' not in filt.keys():
            r3 = 0
            c3 = 0
        else:
            c3 = filt['c3']
            r3 = filt['r3']

        if 'r4' not in filt.keys():
            r4 = 0
            c4 = 0
        else:
            c4 = filt['c4']
            r4 = filt['r4']

        coeffs = calculateCoefficients( c1=c1,
                                        c2=c2,
                                        c3=c3,
                                        c4=c4,
                                        r2=r2,
                                        r3=r3,
                                        r4=r4,
                                        flt_type=filt['flt_type'])
    a = coeffs
    t2 = filt['r2']*filt['c2']
    if len(a) == 2:
        # 2nd order
        a.append(0)    
        a.append(0)    
    elif len(a) == 3:
        # 3rd order
        a.append(0)    
    else:
        pass

    # get smoothed curves for each phase noise component

    freq, vcoPn = interp_semilogx( f, vcoPn, num_points=numPts )

    # loop filter impedance
    z = calculateZ( freq,  
                    t2, 
                    a[0], 
                    a[1],
                    a[2],
                    a[3] )

    # G(s)
    g = calculateG( freq, z, kphi, kvco )

    # # Closed-loop reference transfer gain
    cl_r = (1.0/R)*(g/(1+g/N))
    cl_r_db = 20*np.log10(np.absolute(cl_r))
    refPnOut = refPn + cl_r_db
    refPn = []
    refPn.extend( refPnOut )

    cl_ic = (g/(1+g/N))
    cl_ic_db = 20*np.log10(np.absolute(cl_r))
    icPnOut = pllFom + 10*np.log10(fpfd) + cl_ic_db
    icPn = []
    icPn.extend( icPnOut )

    # # Closed-loop VCO transfer gain
    cl_vco = 1.0/(1+g/N)
    cl_vco_db = 20*np.log10(np.absolute(cl_vco))
    vcoPnOut = vcoPn + cl_vco_db
    vcoPn = []
    vcoPn.extend( vcoPnOut )

    compPn = []
    for i in range(len(freq)):

        compPn.append(power_sum([refPnOut[i],
                                 vcoPnOut[i],
                                 icPnOut[i] ]))

    return freq, refPn, vcoPn, icPn, compPn


def simulatePhaseNoise(f, 
                       refPn,
                       vcoPn,
                       pllFom,
                       pllFlicker,
                       kphi,
                       kvco,
                       fpfd,
                       N,
                       R,
                       filt=None,
                       coeffs=None):
    """ simulate an arbitrary phase-locked loop using either
    filter coefficients or component values. return 3 lists:
    f (frequencies), g_ol (open-loop gain), phases (open-loop phases)  
    """
    if coeffs == None:
        c1 = filt['c1']
        c2 = filt['c2']
        r2 = filt['r2']
        if 'r3' not in filt.keys():
            r3 = 0
            c3 = 0
        else:
            c3 = filt['c3']
            r3 = filt['r3']

        if 'r4' not in filt.keys():
            r4 = 0
            c4 = 0
        else:
            c4 = filt['c4']
            r4 = filt['r4']

        coeffs = calculateCoefficients( c1=c1,
                                        c2=c2,
                                        c3=c3,
                                        c4=c4,
                                        r2=r2,
                                        r3=r3,
                                        r4=r4,
                                        flt_type=filt['flt_type'])
    a = coeffs
    t2 = filt['r2']*filt['c2']
    if len(a) == 2:
        # 2nd order
        a.append(0)    
        a.append(0)    
    elif len(a) == 3:
        # 3rd order
        a.append(0)    
    else:
        pass

    # loop filter impedance
    z = calculateZ( f,  
                    t2, 
                    a[0], 
                    a[1],
                    a[2],
                    a[3] )

    # G(s)
    g = calculateG( f, z, kphi, kvco )

    # # Closed-loop reference transfer gain
    cl_r = (1.0/R)*(g/(1+g/N))
    cl_r_db = 20*np.log10(np.absolute(cl_r))
    refPnOut = refPn + cl_r_db
    refPn = []
    refPn.extend( refPnOut )

    cl_ic = (g/(1+g/N))
    cl_ic_db = 20*np.log10(np.absolute(cl_r))
    icPnOut = pllFom + 10*np.log10(fpfd) + cl_ic_db
    icPn = []
    icPn.extend( icPnOut )

    icFlickerOut = pllFlicker + 20*np.log10(fpfd) - 10*np.log10(f) + cl_ic_db
    icFlick = []
    icFlick.extend( icFlickerOut )

    # # Closed-loop VCO transfer gain
    cl_vco = 1.0/(1+g/N)
    cl_vco_db = 20*np.log10(np.absolute(cl_vco))
    vcoPnOut = vcoPn + cl_vco_db
    vcoPn = []
    vcoPn.extend( vcoPnOut )

    compPn = []
    for i in range(len(f)):
        compPn.append(power_sum( [ refPnOut[i],
                                  vcoPnOut[i],
                                  icPnOut[i],
                                  icFlickerOut[i] ] ))

    return f, refPn, vcoPn, icPn, icFlick, compPn

def calculateCoefficients( c1=0, 
                           c2=0, 
                           c3=0, 
                           c4=0,
                           r2=0,
                           r3=0,
                           r4=0,
                           flt_type='passive'):
    """ return loop filter coeffiencients as list
    a[0] = a0, a[1] = a1, etc.
    """
    a = []
    if flt_type == 'passive':
        a.append( c1 + c2 + c3 + c4 )
        a.append( c2*r2*(c1 + c3 + c4) + r3*(c1 + c2)*(c3 + c4) +\
                  c4*r4*(c1 + c2 + c3) )
        a.append( c1*c2*r2*r3*(c3 + c4) +\
                  c4*r4*(c2*c3*r3 + c1*c3*r3 + c1*c2*r2 + c2*c3*r2) )
    else:
        a.append(c1 + c2)
        a.append( (c1*c2*r2) + (c1 + c2) * (c3*r3 + c4*r4 + c4*r3) )
        a.append( c3*c4*r3*r4 * (c1 + c2) + c1*c2*r2*(c3*r3 + c4*r4 + c4*r3) )
    a.append(c1*c2*c3*c4*r2*r3*r4)
    return a

def calculateZ(f, t2, a0, a1, a2=0, a3=0):
    """ given the frequency array and the filter coefficients, 
        return Z(s) as a np.array()
    """
    s = np.array(f)*2*math.pi*1j        ####################
    z = (1 + s*t2)/(s*(a3*s**3 + a2*s**2 + a1*s + a0))
    return z

def calculateG(f, z, kphi, kvco):
    """ given the loop filter impedance, kphi and kvco, return G(s)
    """
    s = np.array(f)*2*math.pi*1j        ###########
    g = kphi*kvco*z/s
    return g

def power_sum( pdb_lst ):
    """ take a list of powers in dBm, add them
        in the linear domain and return the sum
        in log
    """
    sum_lin = 0
    for pdb in pdb_lst:
        sum_lin += 10**(float(pdb)/10)*1e-3
    return 10*math.log10(sum_lin/1e-3)

def callGetInterpolatedPhaseNoise(d):
    """
    """
    fstart =        d['fstart'] 
    fstop =         d['fstop'] 
    numPts =        d['numPts'] 
    freq_pts =      d['freq_pts'] 
    pn_pts =        d['pn_pts'] 

    # freq_pts = map(float, freq_pts.split(','))
    # pn_pts = map(float, pn_pts.split(','))
    # f = get_freq_points_per_decade(fstart, fstop, ptsPerDec)
    f, pns = interp_semilogx(freq_pts, pn_pts, num_points=numPts )
    d = { 'freqs':f,
          'pns':pns,
        }

    return d

def calc_coeffs_from_loop_vals(d, order=2, active=False):
    """
    calculate coefficients from the pll loop values
    :Args:
        :d (dict):          dictionary returned from PhaseLockedLoop.get_values()
            :keys:
                :fc, pm, gamma, kphi, kvco, N, order, R, t31, t43, active:
        :order (int):       < 2 | 3 | 4 >
        :active (bool):     Use active filter equations if True
    :Returns:
        dict 
        :keys:
            :t1:    
            :t2:    
            :t3:    
            :t4:    
            :a0:
            :a1:
            :a2:
            :a3:
    """
    # local variables to make formulas easier to read
    fc = float(d['fc'])
    kphi = float(d['kphi'])
    kvco = float(d['kvco'])
    pm = float(d['pm'])
    N = float(d['N'])
    R = float(d['R'])
    gamma = float(d['gamma'])
    t31 = float(d['t31'])
    t43 = float(d['t43'])
    try:
        order = int(d['order'])
    except KeyError:
        order = order
    try:
        active = bool(int(d['active']))
    except KeyError:
        active = active
    ret = {}
    # find poles
    t1 = calc_t1(fc, pm, gamma, t31=t31, t43=t43, num_iters=100)
    t3 = t1*t31
    t4 = t1*t31*t43
    t2 = calc_t2(fc, t1, t3, t4, gamma=gamma)
    ret['t1'] = t1
    ret['t2'] = t2
    ret['t3'] = t3
    ret['t4'] = t4

    # calculate coefficients
    a0 = calc_a0(kphi, kvco, N, fc, t1, t2, t3=t3, t4=t4)
    a1 = calc_a1(a0, t1, t3, t4)
    a2 = calc_a2(a0, t1, t3, t4)
    a3 = calc_a3(a0, t1, t3, t4)
    ret['a0'] = a0
    ret['a1'] = a1
    ret['a2'] = a2
    ret['a3'] = a3
    
    return ret 

def calc_components_from_coeffs(d, order=2, active=False):
    """
    calculate component values from coefficients and poles
    :Args:
        :d (dict):          dictionary returned from PhaseLockedLoop.get_values()
            :keys:
                :t1, t2, t3, t4, a0, a1, a2, a3:
        :order (int):       < 2 | 3 | 4 >
        :active (bool):     Use active filter equations if True
    :Returns:
        dict with component values
            :keys:
                :c1, r2, c2, r3, c3, r4, c4:
    """
    # local variables to make formulas easier to read
    a0 = d['a0']
    a1 = d['a1']
    a2 = d['a2']
    a3 = d['a3']
    t1 = d['t1']
    t2 = d['t2']
    t3 = d['t3']
    t4 = d['t4']

    c = {}
    c['c1'] = 0
    c['r2'] = 0
    c['c2'] = 0
    c['r3'] = 0
    c['c3'] = 0
    c['r4'] = 0
    c['c4'] = 0

    # SOLVE C1
    if order == 2:
        c1 = a0*t1/t2
        c['c1'] = c1
    elif order == 3:
        c1 = (a2/(t2**2))*(1 + np.sqrt(1 + (t2/a2)*(t2*a0 - a1)))
        c['c1'] = c1
    elif order == 4:
        # actually solves C1 and R3 together
        
        # solve using pole t3
        a1_t3 = a0*(t1 + t3)
        a2_t3 = a0*t1*t3
        c1_t3 = (a2_t3/(t2**2))*(1 + np.sqrt(1 + (t2/a2_t3)*(t2*a0 - a1_t3)))
        c3_t3 = (-1*(t2**2)*(c1_t3**2) + t2*a1_t3*c1_t3 - a2_t3*a0)/((t2**2)*c1_t3 - a2_t3)
        r3_t3 = a2_t3/(c1_t3*c3_t3*t2)
        # c['a1_t3'] = a1_t3
        # c['a2_t3'] = a2_t3
        # c['c1_t3'] = c1_t3
        # c['c3_t3'] = c3_t3
        # c['r3_t3'] = r3_t3
        
        # solve using pole t4
        a1_t4 = a0*(t1 + t4)
        a2_t4 = a0*t1*t4
        c1_t4 = (a2_t4/(t2**2))*(1 + np.sqrt(1 + (t2/a2_t4)*(t2*a0 - a1_t4)) )
        c3_t4 = (-1*(t2**2)*(c1_t4**2) + t2*a1_t4*c1_t4 - a2_t4*a0)/((t2**2)*c1_t4 - a2_t4)
        r3_t4 = a2_t4/(c1_t4*c3_t4*t2)
        # c['a1_t4'] = a1_t4 
        # c['a2_t4'] = a2_t4
        # c['c1_t4'] = c1_t4
        # c['c3_t4'] = c3_t4
        # c['r3_t4'] = r3_t4

        c1 = (c1_t3 + c1_t4)/2
        r3 = (r3_t3 + r3_t4)/2
        c['c1'] = c1
        c['r3'] = r3
    else:
        raise ValueError("value passed to order of {} is not valid".format(order))
    
    # SOLVE C2 and C3
    if order == 2:
        c2 =  a0 - c1
        c['c2'] =  c2
    elif order == 3:
        c3 = ((-t2**2)*(c1**2) + t2*a1*c1 - a2*a0)/((t2**2)*c1 - a2)
        c2 = a0 - c1 - c3
        c['c3'] = c3
        c['c2'] = c2
        r3 = a2/(c1*c3*t2)
        c['r3'] = r3
    elif order == 4:
        k0 = (a2/a3) - 1.0/t2 - 1.0/(c1*r3) - (a0 - c1)*t2*r3*c1/a3
        k1 = a1 - t2*a0 - a3/(t2*r3*c1) - (a0 - c1)*r3*c1
        a_val = a3/((t2*c1)**2)
        b_val = t2 + r3*(c1 - a0) + (a3/(t2*c1))*((1.0/t2) - k0)
        c_val = k1 - (k0*a3)/t2
        c2 = (-b_val + np.sqrt(b_val**2 - 4*a_val*c_val))/(2*a_val)
        c3 = (t2*a3*c1)/(r3*(k0*t2*a3*c1 - c2*(a3 - r3*((t2*c1)**2))))
        c['c3'] = c3
        c['c2'] = c2

    # SOLVE FOR R2
    r2 =  t2/c2
    c['r2'] = r2

    # SOLVE FOR R4, C4
    if order == 4:
        c4 = a0 - c1 - c2 - c3
        c['c4'] = c4
        r4 = a3/(t2*r3*c1*c3*c4)
        c['r4'] = r4
    
    return c

####################################3
# 
#   1. Calculate t1
#   2. Calculate t3 (0 if 2nd order, t3=t31*t1 if 3rd order)
#   2. Calculate t4 (0 if 2nd or 3rd order, t4=t31*t1*t43 if 4th order)
#   3. Calculate t2
#   4. solve for a0
#   5. solve for a1
#   6. solve for a2
#   7. solve for a3

def calc_t1(fc, pm, gamma, t31=0, t43=0, num_iters=100):
    """ 
    numerically solve for t1 using the bisection method
    see: https://en.wikibooks.org/wiki/Numerical_Methods/Equation_Solving
    :Parameters:
    fc (float) - cutoff frequency in Hz
    pm (float) - phase margin in degrees
    gamma (float) - optimization factor (1.136)
    t31 (float) - ratio of t3 to t1. ONLY FOR 3rd order or greater
    t43 (float) - ratio of t4 to t3. ONLY FOR 4th order
    num_iters (int) - number of times to loop
    """
    a = 1e-15           # initial guess for a
    b = 1.0             # initial guess for b
    fa = func_t1(a, fc, pm, t31=t31, t43=t43, gamma=gamma)
    fb = func_t1(b, fc, pm, t31=t31, t43=t43, gamma=gamma)
    for i in range(num_iters):
        guess = (a+b)/2
        if (func_t1(guess, fc, pm, t31=t31, t43=t43, gamma=gamma) < 0):
            b = guess
        else:
            a = guess
    return guess

def calc_t2(fc, t1, t3=0, t4=0, gamma=1.024):
    """
    :Parameters:
        fc (float) - cutoff frequency in Hz
        t1 (float) - time constant t1 in seconds
        t3 (float) - time constant t3 in seconds. ONLY FOR 3rd order or greater
        gamma (float) - optimization factor (default=1.024)
    :Returns:
        t2 as float
    """
    omega_c = 2*np.pi*fc
    return gamma/((omega_c**2)*(t1 + t3 + t4))
 
def calc_t3(t1, t31):
    return t1*t31

def calc_t4(t1, t31, t43):
    return t1*t31*t43

def func_t1(x, fc, pm, t31=0.4, t43=0.4, gamma=1.136):
    """ 
    simulate t1. This function is used to 
    numerically solve for T1. 
    Equation 22.31 in Dean Banerjee's Book
    :Parameters:
    x (float) - guess at t1
    fc (float) - cutoff frequency in Hz
    pm (float) - phase margin in degrees
    t31 (float) - ratio of t3 to t1. ONLY FOR 3rd order or greater
    t43 (float) - ratio of t4 to t3. ONLY FOR 4th order
    gamma (float) - optimization factor (1.136)
    :Returns:
    updated value for t1 based on guess (float)
    """
    omega_c = 2*np.pi*fc
    phi = pm*np.pi/180
    val = np.arctan( gamma/(omega_c*x*(1+t31 + t31*t43)) ) - \
            np.arctan( omega_c*x ) - \
            np.arctan( omega_c*x*t31 ) -\
            np.arctan( omega_c*x*t31*t43 ) - phi
    return val

############################################################
# GET COEFFICIENTS FROM LOOP VALUES AND TIME CONSTANTS
############################################################
def calc_a0(kphi, kvco, N, fc, t1, t2, t3=0, t4=0):
    """
    return a0 coefficient
    :Args:
        :kphi (float):
        :kvco (float):
        :N (int):
        :fc (float):
        :t1 (float):
        :t2 (float):
        :t3 (float):
        :t4 (float):
    """
    omega_c = 2*np.pi*fc
    k1 = kphi*kvco/((omega_c**2)*(N))
    k2 = np.sqrt(
            (1+(omega_c*t2)**2)/((1+(omega_c*t1)**2)*(1+(omega_c*t3)**2)*(1+(omega_c*t4)**2) ) 
                )
    return k1*k2

def calc_a1(a0, t1, t3, t4):
    return a0*(t1 + t3 + t4)

def calc_a2(a0, t1, t3, t4):
    return a0*(t1*t3 + t1*t4 + t3*t4)

def calc_a3(a0, t1, t3, t4):
    return a0*t1*t3*t4

############################################################
# GET COEFFICIENTS FROM COMPONENTS
############################################################
def calc_coeffs_from_comps(c, order=2, active=False):
    """
    calculate coefficients from component values
    :Args:
        :c (dict):          dictionary returned from calc_components()
            :keys:
                :c1, c2, c3, c4, r2, r3, r4:
        :order (int):       < 2 | 3 | 4 >
        :active (bool):     Use active filter equations if True
    :Returns:
        dict with component values
            :keys:
                :t1, t2, t3, t4, a0, a1, a2, a3:
    """
    # local variables to make formulas easier to read
    c1 = c['c1']
    c2 = c['c2']
    c3 = c['c3']
    c4 = c['c4']
    r2 = c['r2']
    r3 = c['r3']
    r4 = c['r4']
    a0 = get_a0(c1, c2, r2, c3=c3, r3=r3, c4=c4, r4=r4, active=active) 
    a1 = get_a1(c1, c2, r2, c3=c3, r3=r3, c4=c4, r4=r4, active=active) 
    a2 = get_a2(c1, c2, r2, c3=c3, r3=r3, c4=c4, r4=r4, active=active) 
    a3 = get_a3(c1, c2, r2, c3=c3, r3=r3, c4=c4, r4=r4)
    
    d = {}
    d['a0'] = a0
    d['a1'] = a1
    d['a2'] = a2
    d['a3'] = a3
    # calculate poles
    t2 = r2*c2

        # solve t1, t3 and t4
    if active:
        t1 = c1*c2*r2/(c1 + c2)
        if order == 2:
            t3 = 0
            t4 = 0
        elif order == 3:
            t3 = c2*r3
            t4 = 0
        elif order == 4:
            x = c3*r3 + c4*r4 + c4*r3
            y = np.sqrt(x**2 - 4*(c3*c4*r3*r4))
            t3 = (x + y)/2
            t4 = (x - y)/2
    else:   # passive
        if order == 2: 
            t1 = c1*c2*r2/(c1 + c2)
            t3 = 0
            t4 = 0
        elif order == 3:
            t1 = (a1 + np.sqrt(a1**2 -4*a0*a2))/(2*a0)
            t3 = (a1 - np.sqrt(a1**2 -4*a0*a2))/(2*a0)
            t4 = 0
        elif order == 4:
            # A = 1
            # B = (a1/a0) - t1
            # C = a3/(a0*1)
            # coeffs = [A, B, C]
            # roots = np.roots(coeffs)
        
            A = (-2*a1/a0)
            B = ((a1**2)/(a0**2) + a2/a0)
            C = (a3/a0 - a1*a2/(a0**2))
            coeffs = [1, A, B, C]
            roots = np.roots(coeffs)
            t1 = 0
            t3 = 0
            t4 = 0
            d['coeffs'] = [A, B, C]
            d['roots'] = roots
       
    d['t1'] = t1
    d['t2'] = t2
    d['t3'] = t3
    d['t4'] = t4
     
    return d

def get_a0(c1, c2, r2, c3=0, r3=0, c4=0, r4=0, active=False):
    """
    return a0 coefficient based on component values
    """
    if active:
        return c1*c2
    else:
        return c1 + c2 + c3 + c4

def get_a1(c1, c2, r2, c3=0, r3=0, c4=0, r4=0, active=False):
    """
    return a1 coefficient based on component values
    """
    if active:
        return c1*c2*r2 + (c1+c2)*(c3*r3 + c4*r4 + c4*r3)
    else:
        return c2*r2*(c1+c3+c4) + r3*(c1+c2)*(c3+c4) + c4*r4*(c1+c2+c3)


def get_a2(c1, c2, r2, c3=0, r3=0, c4=0, r4=0, active=False):
    """
    return a2 coefficient based on component values
    """
    if active:
        return c3*c4*r3*r4*(c1+c2) + c1*c2*r2*(c3+r3 + c4*r4 + c4*r3)
    else:
        return c1*c2*r2*r3*(c3+c4) + c4*r4*(c2*c3*r3+c1*c3*r3+c1*c2*r2+c2*c3*r2)

def get_a3(c1, c2, r2, c3=0, r3=0, c4=0, r4=0):
    """
    return a3 coefficient based on component values
    """
    return c1*c2*c3*c4*r2*r3*r4

def load_pll_from_file(fpath):
    """
    load a pll object from file. return the PhaseLockedLoop() object and the dict
    """
    fo = open(fpath, 'r')
    d = {}
    lines = fo.readlines()
    fo.close()
    for line in lines:
        try:
            key = line.split(":")[0].strip()
            val = line.split(":")[1].strip()
        except:
            break
        d[key] = val
    order = int(d['order'])
    active = bool(int(d['active']))
    fc = float(d['fc'])
    pm = float(d['pm'])
    gamma = float(d['gamma'])
    N = float(d['N'])
    R = float(d['R'])
    kphi = float(d['kphi'])
    kvco = float(d['kvco'])
    t31 = float(d['t31'])
    t43 = float(d['t43'])
    p = PhaseLockedLoop(fc, pm, kphi, kvco, N, order=order, t31=t31, t43=t43, gamma=gamma)
    return p, d

def logify_freq(fstart, fstop, num_pts=100):
    """
    return a list of frequency points log scaled
    so the resolution looks good on a semilogx() plot.
    The frequency will start and finish at the nearest log10 value
    """
    n_start = int(np.floor(np.log10(fstart)))
    n_stop = int(np.ceil(np.log10(fstop)))
    f = np.logspace(n_start, n_stop, num_pts)
    return f

def interp_phase_noise(x, y, freq_array):
    """ 
    :Args:
        x (list) - x values (frequencies)
        y (list) - y values (phase noise or gain in dB)
        freq_array (list) - list of frequencies to interpolate the phase noise
    Note: x and y have a semilog X relationship.
    Returns:
        list of pn values
    """
    f_log = np.log10(freq_array)
    y_interp = []
    x_log = []
    
    for x_val in x:
        x_log.append(math.log10(x_val))
    
    for xx in f_log:
        y_temp = interp_linear(x_log, y, xx)
        y_interp.append(y_temp[1])
    
    return y_interp

def load_pn_file(fname):
    """
    """
    fo = open(fname, 'r')
    s = fo.read()
    fo.close()
    s = s.split("\n")
    d = {
         'f':[],
         'pn':[],
        }
    f = []
    pn = []
    i = 0
    for line in s:
        if line.startswith("#"):
            pass
        else:
            if ":" in line:
                key = line.split()[0]
                val = float(line.split()[1])
                d[key] = val
            else:
                try:
                    d['f'].append(float(line.split()[0]))
                except Exception as e:
                    return d
                try:
                    d['pn'].append(float(line.split()[1]))
                except Exception as e:
                    return d
    return d





