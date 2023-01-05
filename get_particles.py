import pandas as pd
from helpers.io_utils import hlist2pandas
from helpers.SimulationAnalysis import readHlist
import numpy as np
from io import StringIO
import os
from astropy.table import Table

rbins = np.logspace(-3, 0, 91)
prof_type = 'host'
"""
Go through particle data files and collect particles associated with certain
halos
"""

def get_particles(loc,host_id = None):
    #Get Halo info
    data = hlist2pandas(loc + '/out_0.list')

    ##make keys all lowercase letters
    halos = data.rename(columns={c: c.lower() for c in data.columns})

    if host_id is None:
        #get id of most massive halo
        host_id = halos.at[halos['mvir'].idxmax(), 'id']
    
    #get host info from table
    host = next(halos.query('id == {}'.format(host_id)).itertuples())

    #get distance to host
    halos['dist_to_host'] = halos.eval('sqrt((x-@host.x)**2 + (y-@host.y)**2 + (z-@host.z)**2)')
    
    #find where the distance to host is less than rvir*0.002
    
    halos = halos.query('dist_to_host < (@host.rvir*0.002)') 
    halo_ids = set(halos['id'].tolist())
    part_table_header = 'x y z vx vy vz particle_id assigned_internal_haloid internal_haloid external_haloid'.split()
    
    part_table = StringIO() #create a file object for the particle table
    files = [f for f in os.listdir(loc) if f.startswith('halos_0.') and f.endswith('.particles')]
    for fname in files:
        with open(loc + fname) as f:
            for l in f:
                if l[0]=='#':
                    continue 
                eid = int(l.strip().rpartition(' ')[-1])
                if eid in halo_ids:
                    part_table.write(l)

    part_table.seek(0)
    particles=pd.read_csv(part_table, names = part_table_header,
            dtype=dict([(n, np.int if n.endswith('id') else np.float) for n in part_table_header]), engine='c', delim_whitespace=True, 
                          header=None, skipinitialspace=True,
                          na_filter=False, quotechar=' ')

    del part_table

    particles_host = particles[particles['external_haloid'] == host.id]

    particles = particles[['particle_id', 'external_haloid']]
    particles = pd.merge(particles, halos[['id', 'vmax']], 
                left_on='external_haloid', right_on='id')

    particles = particles.sort_values('vmax')
    particles = particles[['particle_id', 'external_haloid']]
    particles = particles[~particles.duplicated('particle_id')]
    particles = particles.rename(columns={'external_haloid': 'smallest_external_haloid'})
    particles_host = pd.merge(particles_host, particles, how='left', on='particle_id')
    particles_host['dist_to_host'] = particles_host.eval('sqrt((x-@host.x)**2 + (y-@host.y)**2 + (z-@host.z)**2)')

    #particles_host = particles_host[particles_host['dist_to_host']<0.5*host.rvir]

    return particles_host
