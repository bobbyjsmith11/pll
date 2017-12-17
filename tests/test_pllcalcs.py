from unittest import TestCase

from pll.pll_calcs import *

class Test2ndOrderPassive(TestCase):
    """ The only real function of the class is to provide component values.
        Testing this function will indirectly test all underlying functions
        of the class.
    """

    def test_2nd_order_passive(self):
        """ Tests full operation of PllSecondOrderPassive. 
            Instantiate the class with some hard-coded values. Simulate the PLL
            by calling simulatePll function. Test that the phase margin (pm) and
            cutoff frequency (fc) are equal to the hard-coded values. 
        """
        fc = 100e3
        pm = 45.0

        gamma = 1.024
        kphi = 4.69e-3
        kvco = 10e6
        fstart = 1
        fstop = 100e6
        ptsPerDec = 100
        N = 200
        R = 4

        pll = PllSecondOrderPassive( fc,
                                     pm,
                                     kphi,
                                     kvco,
                                     N,
                                     gamma=gamma )


        d_test = pll.calc_components()
        pm_test, fc_test = get_pm_fc_from_actual_filter_components(d_test, fstart, fstop, ptsPerDec, kphi, kvco, N, R)

        self.assertAlmostEqual(pm,pm_test)
        self.assertAlmostEqual(fc,fc_test)


class Test3rdOrderPassive(TestCase):
    """ The only real function of the class is to provide component values.
        Testing this function will indirectly test all underlying functions
        of the class.
    """

    def test3rdOrderPassive(self):
        """ Tests full operation of PllThirdOrderPassive. 
            Instantiate the class with some hard-coded values. Simulate the PLL
            by calling simulatePll function. Test that the phase margin (pm) and
            cutoff frequency (fc) are equal to the hard-coded values. 
        """
        fc = 100e3
        pm = 45.0

        kphi = 5e-3
        kvco = 10e6
        N = 200
        fstart = 1
        fstop = 100e6
        ptsPerDec = 100
        R = 1

        pll = PllThirdOrderPassive( fc,
                                     pm,
                                     kphi,
                                     kvco,
                                     N,
                                     gamma=1.024,
                                     t31=0.6)

        d_test = pll.calc_components()
        pm_test, fc_test = get_pm_fc_from_actual_filter_components(d_test, fstart, fstop, ptsPerDec, kphi, kvco, N, R)

        self.assertAlmostEqual(pm,pm_test)
        self.assertAlmostEqual(fc,fc_test)


############ Helper functions ############333
def get_pm_fc_from_actual_filter_components(d, fstart, fstop, ptsPerDec, kphi, kvco, N, R):
    """ return pm and fc from simulating actual filter components
        Parameters
            d (dict) - returned from a call to calc_components in a pll class 

        Returns
            tuple(pm (float), fc (float))
    """
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
    return pz, fz
