"""
Example and fully functional code for using an n-body catalog, populating the halos with luminous objects (here, both galaxies and quasars) according to a luminosity versus halo mass relation, then drawing a light pencilbeam across a target halo (selected here as the most luminous) and collecting information on the contents. 

Ensure that the relation you use is the one after you account for scatter, and one that is suitable to use with your simulation. Personally, I would determine such a luminosity versus halo mass relation by calibrating to the fit of the mass function from the n-body (or empirical distribution itself) to the observed luminosity function. The relation with scatter can be determined by postprocessing using either deconvolution methods or Ren et al. (2019) approximation 
"""

from funcs import *
import numpy as np
from collections import Counter
import itertools

def load(name):
    return np.genfromtxt(name, comments='#')

def test_stack(halo_catalog, galaxy_lvm, qso_qvm, n, z, distance_modulus, beam_depth, beam_width, beam_length,
               gal_scatter, qso_scatter, dc, obs_gal_limit = 26.5, obs_qso_limit = 25.5, qso_select_limit = 24.0):
    """
    halo_catalog is a HaloCatalog object
    galaxy_lvm, qso_qvm is a LuminousObject object
    """
    assert isinstance(halo_catalog,halo_cat.haloCatalog) == True, 'Halo catalog needs to be a halo_cat.HaloCatalog object, refer to test_stack to see an example'
    assert isinstance(galaxy_lvm,luminousobjects.luminousObject) == True, 'Galaxy luminosity vs halo mass relation needs to be a luminousobject.LuminousObject object, refer to test_stack for example'
    assert isinstance(qso_qvm,luminousobjects.luminousObject) == True, 'QSO luminosity vs halo mass relation needs to be a luminousobject.LuminousObject object, refer to test_stack for example'

    target_summary_count = None
    random_summary_count = None
    target_full_catalog = []
    random_full_catalog = []

    for y in range(n):
        '''
        Basic method
        1. Populate halo masses with luminosities
        2. Select targets based on luminosities
        3. Pencilbeam centered across targets
        4. Recover index and luminosities of objects inside pencil beam
        '''
        #Populate halos with galaxies
        galaxy_catalog = galaxy_lvm.additional_scatter(halo_catalog.mass, gal_scatter)
        
        #Populate halos with QSOs and select the brightest quasars (brighter than qso_select_limit)
        qso_catalog = qso_qvm.additional_scatter(halo_catalog.mass, qso_scatter)
        dc_seed = np.random.rand(halo_catalog.count)
        active_qso = np.where(dc_seed < dc)[0]
        brightest_qso_indices = active_qso[qso_catalog[active_qso] < qso_select_limit - distance_modulus]

        object_counts = np.zeros((len(brightest_qso_indices),11,2))

        for j in range(len(brightest_qso_indices)):
            theta = np.random.choice([-1.,1.])*((np.pi/4.*np.random.random()) + np.pi/6.)
            phi = np.random.choice([-1.,1.])*((np.pi/4.*np.random.random()) + np.pi/6.)

            target_beam_indices, target_beam_center, target_beam_parameters = halo_catalog.light_beams(beam_depth, beam_width, beam_length, theta, phi, True, brightest_qso_indices[j])
            random_beam_indices, random_beam_center, random_beam_parameters = halo_catalog.light_beams(beam_depth, beam_width, beam_length, theta, phi, True)

            #Visibility of galaxies and quasars (accounting for duty cycle in quasars)
            active_qso_target_beam = target_beam_indices[dc_seed[target_beam_indices] < dc]
            active_qso_random_beam = random_beam_indices[dc_seed[random_beam_indices] < dc]       

            active_visible_galaxy_target = target_beam_indices[galaxy_catalog[target_beam_indices] < obs_gal_limit - distance_modulus]
            active_visible_galaxy_random = random_beam_indices[galaxy_catalog[random_beam_indices] < obs_gal_limit - distance_modulus]

            active_visible_qso_target = active_qso_target_beam[qso_catalog[active_qso_target_beam] < obs_qso_limit - distance_modulus]
            active_visible_qso_random = active_qso_random_beam[qso_catalog[active_qso_random_beam] < obs_qso_limit - distance_modulus]

            #Number of quasars visible after accounting for galaxies outshining quasars
            number_qso_fov_target = np.sum(np.isin(active_visible_qso_target,active_visible_galaxy_target,invert=True)) + \
                                    np.sum((galaxy_catalog[active_visible_qso_target[np.isin(active_visible_qso_target,active_visible_galaxy_target)]] - \
                                            qso_catalog[active_visible_qso_target[np.isin(active_visible_qso_target,active_visible_galaxy_target)]]) > 0)

            number_qso_fov_random = np.sum(np.isin(active_visible_qso_random,active_visible_galaxy_random,invert=True)) + \
                                    np.sum((galaxy_catalog[active_visible_qso_random[np.isin(active_visible_qso_random,active_visible_galaxy_random)]] - \
                                            qso_catalog[active_visible_qso_random[np.isin(active_visible_qso_random,active_visible_galaxy_random)]]) > 0)

            #Information for targeted light beam kept in 0th index, 3rd axis. Random 1st index, 3rd axis.
            object_counts[j,0,0] = len(np.unique(np.concatenate((active_visible_galaxy_target,active_visible_qso_target))))
            object_counts[j,1:4,0] = target_beam_center
            object_counts[j,4:6,0] = theta, phi
            object_counts[j,6:9,0] = target_beam_parameters
            object_counts[j,9,0] = number_qso_fov_target

            object_counts[j,0,1] = len(np.unique(np.concatenate((active_visible_galaxy_random,active_visible_qso_random))))
            object_counts[j,1:4,1] = random_beam_center
            object_counts[j,4:6,1] = theta, phi
            object_counts[j,6:9,1] = random_beam_parameters
            object_counts[j,9,1] = number_qso_fov_random

            #Look for repeated indices indicating that object has been counted twice (can occur by unlucky selection of angles)

            if not [index for index,count in Counter(target_beam_indices[galaxy_catalog[target_beam_indices] < obs_gal_limit - distance_modulus]).items() if count > 1]:
                object_counts[j,10,0] = 0
            else:
                object_counts[j,10,0] = 1

            if not [index for index,count in Counter(random_beam_indices[galaxy_catalog[random_beam_indices] < obs_gal_limit - distance_modulus]).items() if count > 1]:
                object_counts[j,10,1] = 0
            else:
                object_counts[j,10,1] = 1

            #Filling the pencil beam object content

            target_full_catalog_temp = np.zeros((len(target_beam_indices),8))
            random_full_catalog_temp = np.zeros((len(random_beam_indices),8))

            for i in range(len(target_beam_indices)):
                target_full_catalog_temp[i,0] = y + 1
                target_full_catalog_temp[i,1] = j + 1
                target_full_catalog_temp[i,2] = galaxy_catalog[target_beam_indices][i]
                target_full_catalog_temp[i,3] = qso_catalog[target_beam_indices][i]
                target_full_catalog_temp[i,4] = dc_seed[target_beam_indices][i]
                target_full_catalog_temp[i,5:8] = halo_catalog.position[target_beam_indices][i]

            for i in range(len(random_beam_indices)):
                random_full_catalog_temp[i,0] = y + 1
                random_full_catalog_temp[i,1] = j + 1
                random_full_catalog_temp[i,2] = galaxy_catalog[random_beam_indices][i]
                random_full_catalog_temp[i,3] = qso_catalog[random_beam_indices][i]
                random_full_catalog_temp[i,4] = dc_seed[random_beam_indices][i]
                random_full_catalog_temp[i,5:8] = halo_catalog.position[random_beam_indices][i]

            target_full_catalog.extend(target_full_catalog_temp)
            random_full_catalog.extend(random_full_catalog_temp)
        
        if target_summary_count is None:        
            target_summary_count = object_counts[:,:,0]
        else:
            target_summary_count = np.vstack((target_summary_count, object_counts[:,:,0]))        

        if random_summary_count is None:
            random_summary_count = object_counts[:,:,1]
        else:
            random_summary_count = np.vstack((random_summary_count, object_counts[:,:,1]))

    np.savetxt(f'test_output/dc{dc*100:.0f}q{qso_scatter*100:.0f}g{gal_scatter*100:.0f}_target_summary.dat', target_summary_count, delimiter=' ', fmt='%f')
    np.savetxt(f'test_output/dc{dc*100:.0f}q{qso_scatter*100:.0f}g{gal_scatter*100:.0f}_random_summary.dat', random_summary_count, delimiter=' ', fmt='%f')
    np.savetxt(f'test_output/dc{dc*100:.0f}q{qso_scatter*100:.0f}g{gal_scatter*100:.0f}_target_full.dat', target_full_catalog, delimiter=' ', fmt='%f')
    np.savetxt(f'test_output/dc{dc*100:.0f}q{qso_scatter*100:.0f}g{gal_scatter*100:.0f}_random_full.dat', random_full_catalog, delimiter=' ', fmt='%f')

    return

if __name__ == "__main__":
    """
    Stack to generate 'n' numbers of pencil beam pointings.
    """

    n = 5

    z = 6.0
    dmod = 51
    depth = 7
    width = 7
    length = 600
    var_g = 0.2
    var_q = 0.4
    dc = 0.1

    h, box_size = load('params')
    millennium_catalog = load('gfinalcut.csv')
    millennium_catalog[:,0] = millennium_catalog[:,0]*9.31e8/h

    #HaloCatalog object. Inputs are the (n,4) array of (mass,x,y,z) and box_size,h. Little_h is false by default meaning that output has h's accounted for already.
    halocatalog_object = halo_cat.haloCatalog(millennium_catalog, h, box_size)
    halocatalog_object.rubik_cube(1)
    halocatalog_object.apply_log_mass()    

    #Always good to check. Flag LO=True means we are checking our catalog is compatible to use with luminousObject Object.
    halocatalog_object.check_catalog_integrity(LO=True)

    #Constructing some fake relation between halo mass and galaxy luminosity
    fake_mass = np.linspace(10,15,20)
    fake_galaxy_lum = np.linspace(-23,-28,20)
    fake_lvm = np.array([fake_mass,fake_galaxy_lum]).T

    #Constructing some fake relation between halo mass and qso luminosity
    fake_qso_lum = np.linspace(-20,-31,20)
    fake_qvm = np.array([fake_mass,fake_qso_lum]).T

    #Creating LuminousObject objects from the relations
    galaxy_lvm_object = luminousobjects.luminousObject(fake_lvm)
    qso_qvm_object = luminousobjects.luminousObject(fake_qvm)    

    test_stack(halocatalog_object, galaxy_lvm_object, qso_qvm_object, n, z, dmod, depth, width, length, var_g, var_q, dc)


