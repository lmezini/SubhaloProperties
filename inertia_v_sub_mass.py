import numpy as np
from astropy.io import ascii
from numpy.linalg import norm,eig
import sys
from helpers.io_utils import hlist2pandas
from helpers.SimulationAnalysis import readHlist
from astropy.table import Column, Table

def calc_inertia(p,s=1.,q=1.,sub_mass=1.,tot_mass = 1.):
    #p is position vector
    #s,q are axis ratios
    r2 = (p[0]**2 + (p[1]/q)**2 + (p[2]/s)**2)
    Ixx = np.sum((p[0]*p[0])/r2)
    Iyy = np.sum((p[1]*p[1])/r2)
    Izz = np.sum((p[2]*p[2])/r2)
    Ixy = np.sum((p[0]*p[1])/r2)
    Iyz = np.sum((p[1]*p[2])/r2)
    Ixz = np.sum((p[0]*p[2])/r2)
    Iyx = Ixy
    Izy = Iyz
    Izx = Ixz
    mass_frac = sub_mass/tot_mass
    I = np.array(((Ixx,Ixy,Ixz),(Iyx,Iyy,Iyz),(Izx,Izy,Izz)))*mass_frac

    return I

def get_eigs(I,rvir):
    #return eigenvectors and eigenvalues
    w,v = eig(I)
    #sort in descending order
    odr = np.argsort(-1.*w)
    #sqrt of e values = a,b,c
    w = np.sqrt(w[odr])
    v = v.T[odr]
    #rescale so major axis = radius of original host
    ratio = rvir/w[0]
    w[0] = w[0]*ratio #this one is 'a'
    w[1] = w[1]*ratio #b
    w[2] = w[2]*ratio #c

    return w,v

def check_ortho(evect):
    #check if eigen vectors of inertia tensor are orthonormal
    
    #create indentity matrix
    a = np.zeros((3, 3))
    np.fill_diagonal(a, 1.)

    #take dot product of v and v.T
    #off diagonals are usually 1e-15 so round them to 0.
    m = np.abs(np.round(np.dot(evect,evect.T),1))

    #check if all elements are equal to identity matrix
    if np.any(a != m):
        print("not orthonormal")
        sys.exit(1)
    
def get_axis(w):
    #return axis ratios
    a,b,c = w
    s = c/a
    q = b/a
    
    return s,q

def transform(p,v):
    #convert to coords of ellipsoid
    #shape is 3 x Number of Particles
    p_new = np.zeros(np.shape(p))
    #loop over each of the 3 coorinates
    for i in range(3):
        p_new[i] += p[0]*v[i,0]+p[1]*v[i,1]+p[2]*v[i,2]
    return p_new

def cut_data(p1,p2,particles,s,q):
    #calculate particle distances in new coord system
    d = p2[0]**2 + (p2[1]/q)**2 + (p2[2]/s)**2
    cut = d<0.00017**2 
    d[cut] = 0.00017**2 #particle distances should not be below force resolution
    #determine which are within the bounds
    cut = d<=(rvir**2)
    p2= p2.T[cut].T #trimmed down in new coord system
    p = p1.T[cut].T #in orig coordinate system
    particles = particles[cut] #

    return p,p2,particles

def get_new_I(p,s,q,h_id,particles=None,halos=None,mass_norm = False,tot_mass=1.0):
    #return new inertia tensor using subset of particles within ellipsoid
    #p1,p2 are xyz data in old and new coord system
    #particles contains particle IDs
    
    #calculate new inertia tensor using orig coordinate system
    #r^2 is caluclated in eigen vector coordinate system

    if mass_norm:
        I_tot = np.zeros((101,3,3))
        #go through all halos in system
        #sort in ascending mass order
        halos = halos[np.argsort(halos['mvir'])]
        halos_set = halos[:-100]
        for h in halos_set:
            #select set of particles belonging to halo
            p_set = particles["smallest_external_haloid"] == h['id']
            p_temp = p.T[p_set].T
            h_mass = h['mvir']
            I = calc_inertia(p_temp,s,q,h_mass,tot_mass)
            I_tot[0] += I
        k = 1
        for h in halos[-100:]:
            #select set of particles belonging to halo
            p_set = particles["smallest_external_haloid"] == h['id']
            p1_temp = p1.T[p_set].T
            p2_temp = p2.T[p_set].T
            h_mass = h['mvir']
            I = calc_inertia(p1_temp,p2_temp,s,q,h_mass,tot_mass)
            I_tot[k] = I_tot[k-1]+I
            k+=1

        return I_tot

    else:
        I_tot = np.zeros((101,3,3))

        #go through all halos in system
        #sort in ascending mass order
        halos = halos[np.argsort(halos['mvir'])]
        
        #select host halo and calculate its contribution to I
        mask = halos['id'] == h_id
        host_halo = halos[mask]

        p_set = particles["smallest_external_haloid"] == host_halo['id']
        p_temp = p.T[p_set].T
        I = calc_inertia(p_temp,s,q)
        I_tot[0] += I
        host_loc = np.where(halos['id'] == h_id)[0][0]
        halos.remove_row(host_loc)

        #select all but 1000 most massive subhalos
        #calculate their contribution to I
        halos_set = halos[:-100]
        for h in halos_set:
            #select set of particles belonging to halo
            p_set = particles["smallest_external_haloid"] == h['id']
            p_temp = p.T[p_set].T
            I = calc_inertia(p_temp,s,q)
            I_tot[0] += I

        #select 1000 most massive subhalos and calculate I
        #with each addition of a more massive subhalo
        k = 1
        for h in halos[-100:]:
            #select set of particles belonging to halo
            p_set = particles["smallest_external_haloid"] == h['id']
            p_temp = p.T[p_set].T
            I = calc_inertia(p_temp,s,q)
            I_tot[k] = I_tot[k-1]+I
            k+=1
        
        return I_tot

j = 0
I_values = np.zeros((45,101,3,3))

with open('halos_info.txt') as f:
    for l in f:
        this_halo, host_id, block, _ = l.split()
        print(this_halo)

        # load host
        path = '/home/cef41/Outputs/{}/'.format(this_halo)
        hostvalues = hlist2pandas(path + '/out_0.list')
        hostvalues = hostvalues.rename(columns={c: c.lower() for c in hostvalues.columns})
        hostvalues = Table.from_pandas(hostvalues)
        # load particles                                                       
        fname = '/home/lom31/particle_stuff/particle_tables/{}_all.particle_table'.format(this_halo)
        particlevalues = ascii.read(fname, format = 'commented_header')

        #select which halos are bounded to host                                
        not_subs = []
        for h in hostvalues:
            p_set = particlevalues["smallest_external_haloid"] == h['id']
            if len(particlevalues[p_set]) == 0:
                h_loc = np.where(hostvalues['id'] == h['id'])[0][0]
                not_subs.append(h_loc)
        hostvalues.remove_rows(not_subs)

        host = hostvalues[hostvalues['id'] == int(host_id)]
        host_x = host['x']
        host_y = host['y']
        host_z = host['z']
        rvir = host['rvir']*1e-3 #convert to same units as x,y,z
        mvir = host['mvir']

        x = particlevalues['x'] - host_x
        y = particlevalues['y'] - host_y
        z = particlevalues['z'] - host_z
        pos = np.array((x,y,z))

        #set initial s,q values
        s,q = 1.,1.

        #recalculate inertia tensor (can normalize by halo mass)
        I = get_new_I(pos,s,q,int(host_id),particlevalues,hostvalues,mass_norm = False,tot_mass = mvir)

        I_values[j] += I
        j+=1
        np.save("inertia_v_submass_all_no_norm.npy",I_values)
