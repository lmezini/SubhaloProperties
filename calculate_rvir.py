import numpy as np
import numexpr as ne
import glob
from helpers.io_utils import hlist2pandas
from astropy.io import ascii
from astropy.table import Column, Table
import pandas as pd


particle_mass = 1.3e8 ##M_sun/h## 
h = 0.7
crit_density = 2.7754e11*h**2 ##M_sun/Mpc**3##
halo_names = []
host_ids = []

with open('halos_info_2.txt') as f:
    for l in f:
        this_halo, host_id, block, _ = l.split()
        halo_names.append(this_halo)
        host_ids.append(host_id)


def calc_density(r,mass):

    V = (4.0/3.0)*np.pi*r**3
    density = mass/V

    return density

def calc_density_iter(r,particle_mass,crit_density):
    
    r = np.sort(r)
    for i in range(len(r)):
        
        mass = (i+1)*particle_mass*0.7*0.7
        density = calc_density(r[i],mass)

        if density > 94*crit_density:
            continue
        else:
            return r[i],mass

def calc_density_binned(r,particle_mass,crit_density,nskip):
    r = np.sort(r)[::nskip]

    for i in range(len(r)):

        mass = nskip*(i+1)*particle_mass*0.7*0.7
        density = calc_density(r[i],mass)

        print(np.log10(density))

        if density > 94*crit_density:
            continue
        else:
            return r[i]

host_vals = ascii.read('host_og_vals_rhap.table', format = 'commented_header')

rvirs = host_vals['rvir']
mvirs = host_vals['mvir']
hostx = host_vals['hostx']
hosty = host_vals['hosty']
hostz = host_vals['hostz']
calc_rvir = np.zeros(96)
calc_mvir = np.zeros(96)
            
j = 0
for f in halo_names:
    print(f)

    #load particles
    fname = '/home/lom31/rhap_particles/particle_tables/{}_host_1rvir.particle_table'.format(f)
    particlevalues = ascii.read(fname, format = 'commented_header')
    
    r = np.sqrt((particlevalues['x']-hostx[j])**2+(particlevalues['y']-hosty[j])**2+(particlevalues['z']-hostz[j])**2)

    #nskip = int(len(r)/100)
    rvir,mvir = calc_density_iter(r,particle_mass,crit_density)

    #rvir = calc_density_binned(r,particle_mass,crit_density,nskip)
    print(rvir,rvirs[j]*0.001)

    calc_rvir[j]+=rvir
    calc_mvir[j]+=mvir
    np.savez('recalc_rvir_mvir_host.npz',rvir = calc_rvir, mvir=calc_mvir)
    j+=1
