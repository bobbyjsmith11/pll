#
#-*- coding: utf-8 -*-
# 
# -------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------
import math

import numpy as np

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pylab as plt

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
        pll = PllSecondOrderPassive( fc,
                                     pm,
                                     kphi,
                                     kvco,
                                     N,
                                     gamma=gamma )
        d = pll.calc_components()
    elif loop_type == 'passive3':
        pll = PllThirdOrderPassive( fc,
                                    pm,
                                    kphi,
                                    kvco,
                                    N,
                                    gamma=gamma )
        d = pll.calc_components()
    elif loop_type == 'passive4':
        pll = PllFourthOrderPassive( fc,
                                     pm,
                                     kphi,
                                     kvco,
                                     N,
                                     gamma=gamma )
        d = pll.calc_components()
    return d 

class PllSecondOrderPassive( object ):
    """ The 2nd order passive phase locked loop object
    """
    def __init__(self,
                 fc,
                 pm,
                 kphi,
                 kvco,
                 N,
                 gamma=1.024):
        """
        :Parameters:
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        kphi (float) - charge pump gain in Amps per radian
        kvco (float) - vco tuning sensitivity in Hz/V
        N (int) - loop multiplication ratio

        gamma (float) - optimization factor (default=1.024)
        """
        self.fc = fc
        self.pm = pm
        self.kphi = kphi
        self.kvco = kvco
        self.N = N
        self.gamma = gamma

    def calc_components(self):
        """ return a dict with the component values """
        d = {}

        d['t1'] = self.calc_t1(self.fc, 
                               self.pm, 
                               self.gamma)

        d['t2'] = self.calc_t2(self.fc, 
                               d['t1'], 
                               self.gamma)

        d['a0'] = self.calc_a0(self.kphi, 
                               self.kvco, 
                               self.N, 
                               self.fc, 
                               d['t1'],
                               d['t2'])

        d['c1'] = self.calc_c1(d['a0'],
                               d['t1'],
                               d['t2'])

        d['c2'] = self.calc_c2(d['a0'], 
                               d['c1'])

        d['r2'] = self.calc_r2(d['c2'], 
                               d['t2'])

        d['a1'] = self.calc_a1(d['c1'], 
                               d['c2'],
                               d['r2'])
        d['a2'] = 0 
        d['a3'] = 0 
        d['r3'] = 0
        d['r4'] = 0
        d['c3'] = 0
        d['c4'] = 0
        d['t3'] = 0
        d['t4'] = 0

        return d

    def calc_t1(self, fc, pm, gamma=1.024):
        """
        :Parameters:
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        gamma (float) - optimization factor (default=1.024)

        """
        omega_c = 2*np.pi*fc
        phi = np.pi*pm/180
        t1 = (np.sqrt(((1+gamma)**2)*(np.tan(phi))**2 + 4*gamma) - (1+gamma)*np.tan(phi)) / (2*omega_c)
        return t1
       
    def calc_t2(self, fc, t1, gamma=1.024):
        """
        :Parameters:
        fc (float) - cutoff frequency in Hz
        t1 (float) - time constant t1 in seconds
        gamma (float) - optimization factor (default=1.024)
        """
        omega_c = 2*np.pi*fc
        return gamma/((omega_c**2)*t1)
    
    def calc_a0(self, kphi, kvco, N, fc, t1, t2):
        """
        :Parameters:
        kphi (float) - charge pump gain in Amps per radian
        kvco (float) - vco tuning sensitivity in Hz/V
        N (int) - loop multiplication ratio
        fc (float) - 0dB crossover frequency in Hz
        t1 (float) - time constant t1 in seconds
        t2 (float) - time constant t2 in seconds
        """
        omega_c = 2*np.pi*fc
        x = (kphi*kvco)/(N*omega_c**2)
        y_num = np.sqrt(1+(omega_c**2)*(t2**2))
        y_den = np.sqrt(1+(omega_c**2)*(t1**2))
        a0 = x*y_num/y_den
        return a0
    
    def calc_c1(self, a0, t1, t2):
        """
        :Parameters:
        a0 (float) - loop filter coefficient
        t1 (float) - time constant t1 in seconds
        (t2 (float) - time constant t2 in seconds
        """
        return a0*t1/t2
    
    def calc_c2(self, a0, c1):
        """
        :Parameters:
        a0 (float) - loop filter coefficient
        c1 (float) - capacitor in Farads
        """
        return a0-c1
    
    def calc_r2(self, c2, t2):
        """
        :Parameters:
        c2 (float) - capacitor in Farads
        t2 (float) - time constant t2 in seconds
        """
        return t2/c2

    def calc_a1(self, c1, c2, r2):
        """
        :Parameters:
        c1 (float) - capacitor in Farads
        c2 (float) - capacitor in Farads
        r2 (float) - resistor in Ohms
        """
        return c1*c2*r2

class PllThirdOrderPassive( PllSecondOrderPassive ):
    def __init__(self,
                 fc,
                 pm,
                 kphi,
                 kvco,
                 N,
                 gamma=1.136,
                 t31=0.6):
        """
        :Parameters:
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        kphi (float) - charge pump gain in Amps per radian
        kvco (float) - vco tuning sensitivity in Hz/V
        N (int) - loop multiplication ratio
        gamma (float) - optimization factor (default=1.136)
        t31 (float) - ratio of T3 to T1 (default=0.6)
        """
        self.fc = fc
        self.pm = pm
        self.kphi = kphi
        self.kvco = kvco
        self.N = N
        self.gamma = gamma
        self.t31 = t31

    def calc_components(self):
        """ return a dict with the component values """
        d = {}
        omega_c = 2*np.pi*self.fc

        # solve for time constants
        d['t1'] = self.calc_t1(self.fc, 
                               self.pm, 
                               self.gamma)

        d['t3'] = d['t1']*self.t31 
  
        d['t2'] = self.gamma/( (omega_c**2)*(d['t1'] + d['t3'] ) )
        

        # solve for coefficients
        d['a0'] = self.calc_a0(self.kphi, 
                               self.kvco, 
                               self.N, 
                               self.fc, 
                               d['t1'],
                               d['t2'],
                               d['t3'])

        d['a1'] = d['a0']*(d['t1'] + d['t3'])

        d['a2'] = d['a0']*d['t1']*d['t3']


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

  
    def calc_c3( self,
                 a0,
                 a1,
                 a2,
                 t2,
                 c1 ):
        return ( -(t2**2)*(c1**2) + t2*a1*c1 - a2*a0 )/( (t2**2)*c1 - a2 )

    def calc_c1( self,
                 a0,
                 a1, 
                 a2, 
                 t2 ):
        return (a2/(t2**2))*(1 + np.sqrt(1 + (t2/a2)*(t2*a0 - a1) ) )
    
    def calc_a0( self, 
                 kphi, 
                 kvco, 
                 N, 
                 fc, 
                 t1, 
                 t2, 
                 t3 ):
        omega_c = 2*np.pi*fc
        k1 = kphi*kvco/((omega_c**2)*(N))
        k2 = np.sqrt( (1+(omega_c*t2)**2)/((1+(omega_c*t1)**2)*(1+(omega_c*t3)**2) ) )
        return k1*k2

    def calc_t1(self, 
                fc, 
                pm, 
                gamma,
                t31=0.6,
                num_iters=100):
        """ numerically solve for t1 using the bisection method
            see: https://en.wikibooks.org/wiki/Numerical_Methods/Equation_Solving
        :Parameters:
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        gamma (float) - optimization factor (1.136)
        num_iters (int) - number of times to loop
        """
        a = 1e-15   # initial guess for a
        b = 1.0       # initial guess for b
        fa = self.func_t1(a,fc,pm,t31=t31,gamma=gamma)
        fb = self.func_t1(b,fc,pm,t31=t31,gamma=gamma)
        for i in range(num_iters):
            guess = (a+b)/2
            if (self.func_t1(guess,fc,pm,t31=t31,gamma=gamma) < 0):
                b = guess
            else:
                a = guess
        return guess

    def func_t1(self,
                x, 
                fc, 
                pm,
                t31=0.6, 
                gamma=1.136):
        """ simulate t1. This function is used to 
        numerically solve for T1. 
        Equation 22.31 in Dean Banerjee's Book
        :Parameters:
        x (float) - guess at t1
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        t31 (float) - ratio of t3 to t1
        gamma (float) - optimization factor (1.136)
        :Returns:
        updated value for t1 based on guess (float)
        """
        omega_c = 2*np.pi*fc
        phi = pm*np.pi/180
        val = np.arctan( gamma/(omega_c*x*(1+t31)) ) - \
                np.arctan( omega_c*x ) - \
                np.arctan( omega_c*x*t31 ) - phi
        return val


def test4thOrderPassive( t31=0.4, t43=0.4 ):
    fc = 10e3
    pm = 47.8
    kphi = 4e-3
    kvco = 20e6
    fout = 900e6
    fpfd = 200e3
    N = float(fout)/fpfd
    fstart = 10
    fstop = 100e6
    ptsPerDec = 100
    fref = 10e6
    R = int(fref/fpfd)
    # R = 1
    
    pll = PllFourthOrderPassive( fc,
                                 pm,
                                 kphi,
                                 kvco,
                                 N,
                                 gamma=1.115,
                                 t31=t31,
                                 t43=t43)
    

    d = pll.calc_components()
    # return d

    flt = {
            'c1':d['c1'],
            'c2':d['c2'],
            'c3':d['c3'],
            'c4':d['c4'],
            'r2':d['r2'],
            'r3':d['r3'],
            'r4':d['r4'],
            'flt_type':"passive" 
           }

    f,g,p,fz,pz,ref_cl,vco_cl = simulatePll( fstart, 
                                             fstop, 
                                             ptsPerDec,
                                             kphi,
                                             kvco,
                                             N,
                                             R,
                                             filt=flt)
    return d, fz, pz

class PllFourthOrderPassive( PllSecondOrderPassive ):
    def __init__(self,
                 fc,
                 pm,
                 kphi,
                 kvco,
                 N,
                 gamma=1.115,
                 t31=0.107,
                 t43=0.107):
        """
        :Parameters:
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        kphi (float) - charge pump gain in Amps per radian
        kvco (float) - vco tuning sensitivity in Hz/V
        N (int) - loop multiplication ratio
        gamma (float) - optimization factor (default=1.115)
        t31 (float) - ratio of T3 to T1 (default=0.4)
        t43 (float) - ratio of T4 to T3 (default=0.4)
            note: for a realizable solution, t31 + t43 <= 1
        """
        self.fc = fc
        self.pm = pm
        self.kphi = kphi
        self.kvco = kvco
        self.N = N
        self.gamma = gamma
        self.t31 = t31
        self.t43 = t43

    def calc_components(self):
        """ return a dict with the component values """
        d = {}
        omega_c = 2*np.pi*self.fc

        # solve for time constants
        d['t1'] = self.calc_t1(self.fc, 
                               self.pm, 
                               self.gamma,
                               t31=self.t31,
                               t43=self.t43)
        # d['t1'] = 4.0685e-6
        # print( 't1 = ' + str(d['t1']) )

        d['t3'] = d['t1']*self.t31 
        d['t4'] = d['t1']*self.t31*self.t43
  
        d['t2'] = self.gamma/( (omega_c**2)*(d['t1'] + d['t3'] + d['t4'] ) )

        # solve for coefficients
        d['a0'] = self.calc_a0(self.kphi, 
                               self.kvco, 
                               self.N, 
                               self.fc, 
                               d['t1'],
                               d['t2'],
                               d['t3'],
                               d['t4'])
    
        d['a1'] = d['a0']*(d['t1'] + d['t3'] + d['t4'])
        d['a2'] = d['a0']*(d['t1']*d['t3'] + d['t1']*d['t4'] + d['t3']*d['t4'])
        d['a3'] = d['a0']*d['t1']*d['t3']*d['t4']

        c1_t3, r3_t3 = self.calc_c1_r3(d['a0'],d['t1'],d['t2'],d['t3'])
        c1_t4, r3_t4 = self.calc_c1_r3(d['a0'],d['t1'],d['t2'],d['t4'])

        d['c1'] = (c1_t3 + c1_t4)/2
        d['r3'] = (r3_t3 + r3_t4)/2

        d['c2'], d['c3'] = self.calc_c2_c3( d['a0'],
                                   d['a1'],
                                   d['a2'],
                                   d['a3'],
                                   d['t2'],
                                   d['r3'],
                                   d['c1'] )
      
        d['c4'] = d['a0']- d['c1']- d['c2'] - d['c3']

        d['r2'] = d['t2']/d['c2']

        d['r4'] = d['a3']/(d['t2']*d['r3']*d['c1']*d['c3']*d['c4'])


        return d

    def calc_c2_c3( self,
                    a0,
                    a1,
                    a2,
                    a3,
                    t2,
                    r3,
                    c1 ):
        k0 = (a2/a3) - 1.0/t2 - 1.0/(c1*r3) - (a0 - c1)*t2*r3*c1/a3
        k1 = a1 - t2*a0 - a3/(t2*r3*c1) - (a0 - c1)*r3*c1
        a = a3/((t2*c1)**2)
        b = t2 + r3*(c1 - a0) + (a3/(t2*c1))*((1.0/t2) - k0)
        c = k1 - (k0*a3)/t2
        c2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        c3 = (t2*a3*c1)/(r3*(k0*t2*a3*c1 - c2*(a3 - r3*((t2*c1)**2))))
        return c2, c3

    def calc_c1_r3( self,
                    a0,
                    t1,
                    t2,
                    tpole):
        a1_t = a0*(t1+tpole)
        a2_t = a0*t1*tpole

        c1_t = (a2_t/(t2**2))*(1 + np.sqrt(1 + (t2/a2_t)*(t2*a0 - a1_t)) )
        c3_t = (-1*(t2**2)*(c1_t**2) + t2*a1_t*c1_t - a2_t*a0)/((t2**2)*c1_t - a2_t)
        r3_t = a2_t/(c1_t*c3_t*t2)

        return c1_t, r3_t

    def calc_a0( self, 
                 kphi, 
                 kvco, 
                 N, 
                 fc, 
                 t1, 
                 t2, 
                 t3,
                 t4):
        omega_c = 2*np.pi*fc
        k1 = kphi*kvco/((omega_c**2)*(N))
        k2 = np.sqrt(
                (1+(omega_c*t2)**2)/((1+(omega_c*t1)**2)*(1+(omega_c*t3)**2)*(1+(omega_c*t4)**2) ) 
                    )
        return k1*k2

    def calc_t1(self, 
                fc, 
                pm, 
                gamma,
                t31=0.4,
                t43=0.4,
                num_iters=100):
        """ numerically solve for t1 using the bisection method
            see: https://en.wikibooks.org/wiki/Numerical_Methods/Equation_Solving
        :Parameters:
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        gamma (float) - optimization factor (1.136)
        num_iters (int) - number of times to loop
        """
        a = 1e-15   # initial guess for a
        b = 1.0       # initial guess for b
        fa = self.func_t1(a,fc,pm,t31=t31,t43=t43,gamma=gamma)
        fb = self.func_t1(b,fc,pm,t31=t31,t43=t43,gamma=gamma)
        for i in range(num_iters):
            guess = (a+b)/2
            if (self.func_t1(guess,fc,pm,t31=t31,t43=t43,gamma=gamma) < 0):
                b = guess
            else:
                a = guess
        return guess

    def func_t1(self,
                x, 
                fc, 
                pm,
                t31=0.4, 
                t43=0.4, 
                gamma=1.136):
        """ simulate t1. This function is used to 
        numerically solve for T1. 
        Equation 22.31 in Dean Banerjee's Book
        :Parameters:
        x (float) - guess at t1
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        t31 (float) - ratio of t3 to t1
        gamma (float) - optimization factor (1.136)
        :Returns:
        updated value for t1 based on guess (float)
        """
        omega_c = 2*np.pi*fc
        phi = pm*np.pi/180
        val = np.arctan( gamma/(omega_c*x*(1+t31)) ) - \
                np.arctan( omega_c*x ) - \
                np.arctan( omega_c*x*t31 ) -\
                np.arctan( omega_c*x*t31*t43 ) - phi
        return val

class PllFourthOrderPassive2(PllSecondOrderPassive):
    def __init__(self,
                 fc,
                 pm,
                 kphi,
                 kvco,
                 N,
                 gamma=1.115):
        """
        :Parameters:
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        kphi (float) - charge pump gain in Amps per radian
        kvco (float) - vco tuning sensitivity in Hz/V
        N (int) - loop multiplication ratio
        gamma (float) - optimization factor (default=1.115)
        t31 (float) - ratio of T3 to T1 (default=0.4)
        t43 (float) - ratio of T4 to T3 (default=0.4)
            note: for a realizable solution, t31 + t43 <= 1
        """
        self.fc = fc
        self.pm = pm
        self.kphi = kphi
        self.kvco = kvco
        self.N = N
        self.gamma = gamma
        self.pole3 = fc*30
        self.pole4 = fc*10

    def calc_t1(self, 
                fc, 
                pm, 
                gamma,
                num_iters=100,
                plotme=False):
        """ numerically solve for t1 using the bisection method
            see: https://en.wikibooks.org/wiki/Numerical_Methods/Equation_Solving
        :Parameters:
        fc (float) - cutoff frequency in Hz
        pm (float) - phase margin in degrees
        gamma (float) - optimization factor (1.136)
        num_iters (int) - number of times to loop
        """
        a = 1e-15   # initial guess for a
        b = 1.0       # initial guess for b
        fa = self.func_t1(a,fc,pm,gamma=gamma)
        fb = self.func_t1(b,fc,pm,gamma=gamma)
        for i in range(num_iters):
            guess = (a+b)/2
            if (self.func_t1(guess,fc,pm,gamma=gamma) < 0):
                b = guess
            else:
                a = guess
            if plotme:
                x = np.linspace(a,b,1000)
                y = []
                for xx in x:
                    y.append(self.func_t1(xx,fc,pm,gamma=gamma) )
                fig, ax = plt.subplots()
                ax.plot(x,y,'r',label='func_t1')
                plt.grid(True)
                plt.show()
        return guess

    def func_t1(self,
                t1, 
                fc, 
                pm,
                gamma=1.115):
        """ simulate t1. This function is used to 
        numerically solve for T1. 
        """
        omega_c = 2*np.pi*fc
        phi = pm*np.pi/180
        t3 = 1.0/self.pole3
        t4 = 1.0/self.pole4
        # val = np.arctan2( 1.0, ( (omega_c)*(t1*t3*t4) )/gamma ) - \
        #       np.arctan2( 1.0, 1.0/omega_c*t1 ) - \
        #       np.arctan2( 1.0, 1.0/omega_c*t3 ) - \
        #       np.croarctan2( 1.0, 1.0/omega_c*t1*t4 ) - phi
        val = np.arctan( gamma/( (omega_c)*(t1*t3*t4) ) ) - \
              np.arctan( omega_c*t1 ) - \
              np.arctan( omega_c*t3 ) - \
              np.arctan( omega_c*t1*t4 ) - phi 
        return val

    def calc_components(self):
        """ return a dict with the component values """
        d = {}
        omega_c = 2*np.pi*self.fc

        d['pole3'] = self.pole3
        d['pole4'] = self.pole4
        # solve for time constants
        d['t1'] = self.calc_t1( self.fc,
                                self.pm,
                                gamma=self.gamma )
        d['pole1'] = 1.0/d['t1']

        d['t3'] = 1.0/self.pole3
        d['t4'] = 1.0/self.pole4
  
        d['t2'] = self.gamma/( (omega_c**2)*(d['t1'] + d['t3'] + d['t4'] ) )
        d['zero'] = 1.0/d['t2']
        # solve for coefficients
        # d['a0'] = self.calc_a0(self.kphi, 
        #                        self.kvco, 
        #                        self.N, 
        #                        self.fc, 
        #                        d['t1'],
        #                        d['t2'],
        #                        d['t3'],
        #                        d['t4'])
    
        # d['a1'] = d['a0']*(d['t1'] + d['t3'] + d['t4'])
        # d['a2'] = d['a0']*(d['t1']*d['t3'] + d['t1']*d['t4'] + d['t3']*d['t4'])
        # d['a3'] = d['a0']*d['t1']*d['t3']*d['t4']

        # c1_t3, r3_t3 = self.calc_c1_r3(d['a0'],d['t1'],d['t2'],d['t3'])
        # c1_t4, r3_t4 = self.calc_c1_r3(d['a0'],d['t1'],d['t2'],d['t4'])

        # d['c1'] = (c1_t3 + c1_t4)/2
        # d['r3'] = (r3_t3 + r3_t4)/2

        # d['c2'], d['c3'] = self.calc_c2_c3( d['a0'],
        #                            d['a1'],
        #                            d['a2'],
        #                            d['a3'],
        #                            d['t2'],
        #                            d['r3'],
        #                            d['c1'] )
      
        # d['c4'] = d['a0']- d['c1']- d['c2'] - d['c3']

        # d['r2'] = d['t2']/d['c2']

        # d['r4'] = d['a3']/(d['t2']*d['r3']*d['c1']*d['c3']*d['c4'])


        return d


def plot_arctan( ):
    x = np.linspace(-np.pi/2,np.pi/2,1000)

    y1 = np.arctan(x)
    y2 = np.arctan2(1, 1/x)

    fig, ax = plt.subplots()
    ax.plot(x,y1,'r',label='arctan')
    ax.plot(x,y2,'g',label='arctan2')
    legend = ax.legend()
    plt.grid(True)
    plt.show()


def callSimulatePll( d ):
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

    f, g, p, fz, pz, ref_cl, vco_cl = simulatePll( fstart,
                                                   fstop,
                                                   ptsPerDec,
                                                   kphi,
                                                   kvco,
                                                   N,
                                                   R,
                                                   filt=flt )
    
    d = { 'freqs':f,
          'gains':g,
          'phases':p,
          'fzero': fz,
          'pzero': pz,
          'ref_cl': ref_cl,
          'vco_cl': vco_cl,
        }
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

def simulatePll( fstart, 
                 fstop, 
                 ptsPerDec,
                 kphi,
                 kvco,
                 N,
                 R,
                 filt=None,
                 coeffs=None ):
    """ simulate an arbitrary phase-locked loop using either
    filter coefficients or component values. return 3 lists:
    f (frequencies), g_ol (open-loop gain), phases (open-loop phases)  
    """
    f = get_freq_points_per_decade(fstart, fstop, ptsPerDec)

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
    # z = (1 + s*t2)/(s*(a[3]*s**3 + a[2]*s**2 + a[1]*s + a[0])) 
    z = calculateZ( f,  
                    t2, 
                    a[0], 
                    a[1],
                    a[2],
                    a[3] )

    # G(s)
    # g = kphi*kvco*z/s
    g = calculateG( f, z, kphi, kvco )

    # # Open-loop gain 
    g_ol = g/N
    g_ol_db = 10*np.log10(np.absolute(g_ol))
    # ph_ol = 180 + np.unwrap(np.angle(g_ol))*180/np.pi
    ph_ol = np.unwrap(np.angle(g_ol))*180/np.pi

    # # Closed-loop reference transfer gain
    cl_r = (1.0/R)*(g/(1+g/N))
    cl_r_db = 20*np.log10(np.absolute(cl_r))

    # # Closed-loop VCO transfer gain
    cl_vco = 1.0/(1+g/N)
    cl_vco_db = 20*np.log10(np.absolute(cl_vco))

    # convert gains and phases to lists
    # cannot return numpy array to javascript
    g = []
    p = []
    g.extend(g_ol_db)
    p.extend(ph_ol)
    fz, pz = getInterpolatedFzeroPzero( f, g, p )
    ref_cl = []
    vco_cl = []
    ref_cl.extend(cl_r_db)
    vco_cl.extend(cl_vco_db)
    return f, g, p, fz, pz, ref_cl, vco_cl

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

    # print(type(f))
    # print(type(refPn))
    # print(type(vcoPn))
    # print(type(icPn))
    # print(type(icFlick))
    # print(type(comp))

    fig, ax = plt.subplots()
    ax.semilogx(f,refPn,'r',label='ref')
    ax.semilogx(f,vcoPn,'b',label='vco')
    ax.semilogx(f,icPn,'g',label='pll')
    ax.semilogx(f,icFlick,'c',label='flick')
    ax.semilogx(f,comp,'k',linewidth=2,label='total')
    legend = ax.legend()
    plt.grid(True)
    plt.show()
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
    # # return x_new, y_new
    # return log_x, y
    # # x_new, y_new = interp_linear(log_x, y, x_interp) 
    # # return x_new, y_new
    # x_new, y_new = interp_linear(x, y, x_interp) 
    # return x_new, y_new

def plot_interp_semilogx(x, y, num_points=10):
    """
    """
    x2, y2 = interp_semilogx(x, y, num_points=num_points)
    plt.semilogx(x, y, '-bo', x2, y2, 'ro')
    plt.grid(True)
    plt.show() 

def linspace(a, b, num_points):
    """ return a list of linearly spaced values
    between a and b having num_points points
    """
    inc = (float(b) - float(a))/(num_points-1)
    ret_ar = []
    for i in range(num_points):
        ret_ar.append(a + i*inc)
    return ret_ar

def plot_interp_linear(x, y, x_interp):
    """
    """
    x2, y2 = interp_linear(x, y, x_interp)
    plt.plot(x, y, '-bo', [x2], [y2], 'ro')
    plt.grid(True)
    plt.show() 

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

def simulatePhaseNoise2( f, 
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


def simulatePhaseNoise( f, 
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
                        coeffs=None ):
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


