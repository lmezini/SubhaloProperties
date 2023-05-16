import pandas as pd
from helpers.io_utils import hlist2pandas
from helpers.SimulationAnalysis import readHlist
import numpy as np
from io import StringIO
import os
from astropy.table import Table
from particle_practice import get_particles

rbins = np.logspace(-3, 0, 91)

"""
Create the halo profiles for host and subhalos by selecting particles within one virial radius of the host
Profiles will be in a format of particles per radial bin
"""

def get_profile(f,prof_type):
    for l in f:
        this_halo, host_id, block, _ = l.split()
        print(this_halo)

        # load host                                                          
        halos = readHlist('/home/cef41/Outputs/{}/halos_0.{}.ascii'.format(this_halo, block), ['ID', 'X', 'Y', 'Z', 'Rvir'])
        host = halos[halos['ID'] == int(host_id)][0]
        del halos
        # load particles                                                     
        path = '/home/cef41/Outputs/{}/'.format(this_halo)
        particles = get_particles(path,host['ID'])
        particles = Table.from_pandas(particles)
        
        # get distances                                                      
        d = np.zeros(len(particles))
        for ax in 'xyz':
            d1 = (particles[ax] - host[ax.upper()])
            d1 *= d1
            d += d1
        del d1
        print(d)
        d = np.sqrt(d, out=d)
        d /= (host['Rvir']*1.0e-3)
        
        if prof_type == 'host':
            d_host = d[particles['assigned_internal_haloid'] == particles['internal_haloid']]
            d_host = d_host[d_host < 1.]
            d_host_new = d[particles["external_haloid"] == particles["smallest_external_haloid"]]
            d_host_new = d_host_new[d_host_new < 1.]
            d_rvir = d[d < 1.]
            d_sub = d[particles['external_haloid'] != particles['smallest_external_haloid']]
            d_sub = d_sub[d_sub < 1.]
            del d, particles

            np.savez(this_halo+"_"+prof_type, rvir=np.histogram(d_rvir, rbins)[0], host=np.histogram(d_host, rbins)[0],sub=np.histogram(d_sub,rbins)[0],host_new=np.histogram(d_host_new,rbins)[0])

            del d_host, d_rvir, d_sub

        elif prof_type == 'sub':
             d_sub = d[d < 1.]
             np.savez(this_halo+"_"+prof_type,sub=np.histogram(d_sub,rbins)[0])

with open('halos_info.txt') as f:
    get_profile(f,'host')
