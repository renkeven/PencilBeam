"""
Luminous objects module. Takes in a luminosity versus halo mass file. Populates the catalog with the luminosities of these objects
"""

import numpy as np
from scipy import interpolate

class luminousObject(object):
    """
    Can be galaxy, quasars, GC's. Ensure that your luminosity v halo relation is in units of logMh vs Magnitude
    There is a function to add scatter around all of your objects, but it is up to the user to ensure that the resulting distribution is consistent with LFs!!!
    """
    def __init__(self, lvm):

        self.mass_range = lvm[:,0]
        self.lum_range = lvm[:,1]

        assert max(self.mass_range) < 20, 'Your units probably aren\'t in logMass'
        assert (self.lum_range < 0).any(), 'Your units are most likely in logLum instead of Magnitude'   

    def populate_median_catalog(self,halo_mass):
        """
        Populate our halo masses with luminous objects
        Check that our mass catalogue and LvM relation has the same units of mass.
        Returns our mass catalogue and the resulting luminosity array.
        """
        assert min(halo_mass) >= min(self.mass_range), 'Catalog contain halo masses outside the luminosity versus halo mass lower limit!'
        assert max(halo_mass) <= max(self.mass_range), 'Catalog contain halo masses outside the luminosity versus halo mass upper limit!' 

        mass_lum_relation = interpolate.interp1d(self.mass_range, self.lum_range)
        halo_lum = mass_lum_relation(halo_mass)

        return halo_mass, halo_lum

    def additional_scatter(self, halo_mass, var):
        """
        Jiggling the log luminosities with a normal distribution. 
        Returns the Magnitude of our object
        """

        _, halo_lum = self.populate_median_catalog(halo_mass)
        halo_lum = halo_lum/-2.5

        if var == 0:
            scatter_lum = halo_lum

        else:
            scatter_lum = np.random.normal(halo_lum,var)

        return -2.5 * scatter_lum

if __name__ == '__main__':
    """
    Checking that this program doesn't break
    """

    fake_mass = np.linspace(11,14,20)
    fake_lum = np.linspace(-23,-28,20)
    fake_lvm = np.array([fake_mass,fake_lum]).T

    fake_catalog = np.linspace(11.5,13.5,40)
   
    print(fake_lvm) 
    general_galaxy = luminousObject(fake_lvm)
    median_galaxy_lum = general_galaxy.additional_scatter(fake_catalog,0.2)
    print(median_galaxy_lum)







