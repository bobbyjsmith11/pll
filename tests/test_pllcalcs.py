from unittest import TestCase

from pll.pll_calcs import *


class TestGeneralFunctions(TestCase):

    def test_interp_linear_1(self):
        """ test the linear interpolator with a value within the x array
        """
        test_var = interp_linear([10,20], [1,2], 12)
        self.assertAlmostEqual(1.2, test_var[1])

    def test_interp_linear_2(self):
        """ test the linear interpolator with a value below the x array
        """
        test_var = interp_linear([1,2,3], [1,0,3], 1.5)
        self.assertAlmostEqual(0.5, test_var[1])

    def test_interp_linear_2(self):
        """ test the linear interpolator with a value above the x array
        """
        test_var = interp_linear([1,2,3], [1,2,3], 3.5)
        self.assertAlmostEqual(3.5, test_var[1])

    def test_freq_points_per_decade(self):
        """ tests that the get_freq_points_per_decade() function returns
        the correct array
        """
        f_good = list(range(10,100,10))
        f_good.extend(range(100,1000,100))
        f_good.extend(range(1000,11000,1000))
        [float(i) for i in f_good]
        f_test = get_freq_points_per_decade(10,10000,10)
        self.assertEqual( set(f_good), set(f_test))


class Test2ndOrderPassive(TestCase):
    """ The only real function of the class is to provide component values.
        Testing this function will indirectly test all underlying functions
        of the class.
    """

    def test_2nd_order_passive_phase_margin(self):
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

    def test_2nd_order_passive_loop_bandwidth(self):
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

        self.assertAlmostEqual(fc,fc_test)

class Test3rdOrderPassive(TestCase):
    """ The only real function of the class is to provide component values.
        Testing this function will indirectly test all underlying functions
        of the class.
    """

    def test_3rd_order_passive_phase_margin(self):
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

    def test_3rd_order_passive_loop_bandwidth(self):
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
