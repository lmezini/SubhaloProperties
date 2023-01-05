from helpers.io_utils import hlist2pandas
from helpers.SimulationAnalysis import readHlist
from get_particles import get_particles
import pandas as pd
from astropy.table import Table

with open('halos_info_subset.txt') as f:
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
    
        #whhost_no_ss = particles['assigned_internal_haloid'] == particles['internal_haloid']
        #whhost = particles["external_haloid"] == particles["smallest_external_haloid"]
        #whsub = particles["external_haloid"] != particles["smallest_external_haloid"]
        
        #particles[whhost_no_ss].write('/home/lom31/particle_stuff/particle_tables/{}_host_noSS.particle_table'.format(this_halo),format='ascii.commented_header',overwrite=True)
        #particles[whhost].write('/home/lom31/particle_stuff/particle_tables/{}_host.particle_table'.format(this_halo),format='ascii.commented_header',overwrite=True)
        #particles[whsub].write('/home/lom31/particle_stuff/particle_tables/{}_sub.particle_table'.format(this_halo),format='ascii.commented_header',overwrite=True)
        particles.write('/home/lom31/particle_stuff/particle_tables/{}_all.particle_table'.format(this_halo),format='ascii.commented_header',overwrite=True)
