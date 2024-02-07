import numpy as np
from astropy.io import ascii
from numpy.linalg import norm,eig
import sys
from helpers.io_utils import hlist2pandas
from helpers.SimulationAnalysis import readHlist
from astropy.table import Column, Table


#calculate starting inertia tensor
def calc_sphere_inertia(p,s=1.,q=1.):
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
    I = np.array(((Ixx,Ixy,Ixz),(Iyx,Iyy,Iyz),(Izx,Izy,Izz)))

    return I

def calc_inertia(p,p2,s=1.,q=1.,mass=1.):
    #p is position vector
    #s,q are axis ratios
    r2 = (p2[0]**2 + (p2[1]/q)**2 + (p2[2]/s)**2)
    Ixx = np.sum((p[0]*p[0])/r2)
    Iyy = np.sum((p[1]*p[1])/r2)
    Izz = np.sum((p[2]*p[2])/r2)
    Ixy = np.sum((p[0]*p[1])/r2)
    Iyz = np.sum((p[1]*p[2])/r2)
    Ixz = np.sum((p[0]*p[2])/r2)
    Iyx = Ixy
    Izy = Iyz
    Izx = Ixz
    I = np.array(((Ixx,Ixy,Ixz),(Iyx,Iyy,Iyz),(Izx,Izy,Izz)))#/mass

    return I


def get_eigs(I):
    #return eigenvectors and eigenvalues
    w,v = eig(I)
    #sort in descending order
    odr = np.argsort(-1.*w)
    #sqrt of e values = a,b,c
    w_sq = w[odr] 
    w = np.sqrt(w[odr])
    v = v.T[odr]
    #rescale so major axis = radius of original host
    ratio = rvir/w[0]
    w[0] = w[0]*ratio #this one is 'a'
    w[1] = w[1]*ratio #b
    w[2] = w[2]*ratio #c

    return w,v,w_sq

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

def get_new_I(p1,p2,s,q,particles=None,halos=None,mass_norm = False):
    #return new inertia tensor using subset of particles within ellipsoid
    #p1,p2 are xyz data in old and new coord system
    #particles contains particle IDs
    
    #calculate new inertia tensor using orig coordinate system
    #r^2 is caluclated in eigen vector coordinate system

    I = calc_inertia(p,s,q)
      
    return I

j = 0
#I_values = np.zeros((45,3,3))

s_vals = np.zeros(45) #c to a
q_vals = np.zeros(45) #b to a

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

        I = calc_sphere_inertia(all_pos,s,q)
        err = 1.
        tol = .001 #keeping big while testing
        it = 0
        new_err = 1.
        old_err = 10.
        while new_err>tol:
            s_old,q_old = s,q
            old_err = new_err
            #get eigen vectors and values of inertia tensor
            w,v,w_sq = get_eigs(I)

            #check if vectors are orthonormal
            check_ortho(v)

            #get new s and q
            s,q = get_axis(w)
            #print('s,q: {},{}'.format(s,q))
            #rotate to frame of principle axis 
            data_prime = transform(data,v)

            #select which particles fall within new ellipsoid
            data,data_prime,particles = cut_data(data,data_prime,particles,s,q)
            print('# particles: {}'.format(np.shape(data)[1]))
            #recalculate inertia tensor (can normalize by halo mass)
            I = get_new_I(data,data_prime,s,q,particles,hostvalues,mass_norm = False)

            #compare err to tolerance
            err1 = abs(s_old-s)/s_old
            err2 = abs(q_old-q)/q_old
            new_err = max(err1,err2)

            it += 1
            if it > 9:
                break

        w,v,w_sq = get_eigs(I)

        #check if vectors are orthonormal
        check_ortho(v)

        #get new s and q
        s,q = get_axis(w)
        s_vals[j]+=s
        q_vals[j]+=q
        print(s,q)
        #this is yao's version
        #force_res = 0.00017
        #b_to_a, c_to_a, eig_A = calc_shape(rvir, [0,0,0], pos, force_res)
    

        #I_values[j] += I
        j+=1
        #np.save("mwm_all_inertia_tensor.npy",I_values)
        np.savez("mwm_w_sub_shape.npz",c_to_a=s_vals,b_to_a=q_vals)


