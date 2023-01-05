import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column
import sys
from numpy.linalg import norm,eig
import random
import glob
from helpers.io_utils import hlist2pandas

def rotate(v,thetax,thetay,thetaz):
    #angles are given in degrees
    #vect should be 1x3 dimensions

    v_new = np.zeros(np.shape(v))
    thetax = thetax*np.pi/180
    thetay = thetay*np.pi/180
    thetaz = thetaz*np.pi/180
    Rx = np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(thetax),-np.sin(thetax)],
                   [ 0, np.sin(thetax), np.cos(thetax)]])
  
    Ry = np.matrix([[ np.cos(thetay), 0, np.sin(thetay)],
                   [ 0           , 1, 0           ],
                   [-np.sin(thetay), 0, np.cos(thetay)]])
  
    Rz = np.matrix([[ np.cos(thetaz), -np.sin(thetaz), 0 ],
                   [ np.sin(thetaz), np.cos(thetaz) , 0 ],
                   [ 0           , 0            , 1 ]])
    
    R = Rx * Ry * Rz
    v_new += R*v

    return v_new

def transform(v1,v2,axis = None):
    #convert to coords of principal axis (v2)
    #Take transpose so that v1[0],v1[1],v1[2] are all x,y,z respectively
    v1 = v1.T
    v_new = np.zeros(np.shape(v1))

    #loop over each of the 3 coorinates
    if axis == None:
        for i in range(3):
            v_new[i] += v1[0]*v2[i,0]+v1[1]*v2[i,1]+v1[2]*v2[i,2]
        return v_new
    else:
        v_new[0] += v1[0]*v2[axis,0]+v1[1]*v2[axis,1]+v1[2]*v2[axis,2]
        return v_new

def random_rotation(I,pos):
    np.random.seed()
    tx=np.random.random()*90.0
    ty=np.random.random()*90.0
    tz=np.random.random()*90.0

    new_I = rotate(I,tx,ty,tz)
    new_pos = transform(pos,new_I).T

    hA = new_I[0]
    hB = new_I[1]

    new_pos = transform(pos,hv).T
    hA2 = np.repeat(hA,len(new_pos)).reshape(3,len(new_pos)).T
    hB2 = np.repeat(hB,len(new_pos)).reshape(3,len(new_pos)).T
    para1 = (new_pos*hA2/norm(hA)).sum(axis=1)
    para2 = (hA/norm(hA)).T

    t = np.arccos(abs((new_pos*hA2).sum(axis=1)/(norm(new_pos,axis=1)*norm(hA))))
    #p = np.arccos((perp*hB2).sum(axis=1)/(norm(hB)*norm(perp,axis=1)))

    return t

def rotation_err_bs(I_arr,pos,ang_cut,mass,n,n_rep=1000):
    #I_arr: principal axis of inertia tensor
    #pos_arr: particle postions
    #ang_cut: anglular separation
    #mvir: host mass
    #n: number of halos
    #n_rep: number of bootstrap
    fracs = np.zeros(n_rep)
    for j in range(n_rep):
        t = random_rotation(I_arr,pos)*180/np.pi
        frac = particle_mass*len(pos[t<ang_cut])/mass
        fracs[j]+=frac

    #std1 = np.std(np.median(np.log10(fracs),axis=1))
    #std2 = np.std(np.median(fracs,axis=1))
    #percentiles1 = np.percentile(np.log10(np.median(fracs,axis=1)),[15.9,84.1])
    #percentiles2 = np.percentile(np.median(fracs,axis=1),[15.9,84.1])

    return fracs

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

particle_mass = 1.3e8 ##M_sun/h##                                           
host_vals = ascii.read('host_og_vals_rhap.table', format = 'commented_header')
host_I = np.load('inertia_tensor_stuff/rhap_host_inertia_tensor_no_norm.npy')

rvirs = host_vals['rvir']
mvirs = host_vals['mvir']
hostx = host_vals['hostx']
hosty = host_vals['hosty']
hostz = host_vals['hostz']
hostvx = host_vals['hostvx']
hostvy =host_vals['hostvy']
hostvz = host_vals['hostvz']
host_shapes = host_vals['host_shapes']
host_spins = host_vals['host_spins']
host_cs = host_vals['host_cs']
hostJx = host_vals['hostJx']
hostJy = host_vals['hostJy']
hostJz = host_vals['hostJz']


ang_cut = [10.0, 20.0, 30.0, 40.0 ,50.0 ,60.0 ,70.0 ,80.0 ,90.0]

mass_frac_A_ang = np.zeros((len(host_vals),len(ang_cut)))
mass_frac_random = np.zeros((int(len(host_vals)*5),len(ang_cut)))
frac_errs = np.zeros((len(ang_cut),len(host_vals),1000))
percentiles = np.zeros((len(ang_cut),2))
percentiles5 = np.zeros((len(ang_cut),2))
halo_names = []
host_ids = []

with open('halos_info_2.txt') as f:
    for l in f:
        this_halo, host_id, block, _ = l.split()
        halo_names.append(this_halo)
        host_ids.append(host_id)


for j,f in enumerate(halo_names):
    print(f)
    # load host                                                                 
    path = '/home/lom31/rhap_particles/rhapsody/{}/rockstar/'.format(f)
    hostvalues = hlist2pandas(path + '/out_199.list')
    hostvalues = Table.from_pandas(hostvalues)

    fname = '/home/lom31/rhap_particles/particle_tables/{}_sub.particle_table'.format(f)
    particlevalues = ascii.read(fname, format = 'commented_header')
    ##Make sure chosen particles are within the virial radius##                 
    r = Column(np.sqrt(particlevalues['x']**2+particlevalues['y']**2+particlevalues['z']**2),name='r')

    particlevalues.add_column(r)

    whlimit = np.where(particlevalues['r']<=rvirs[j])
    particlevalues = particlevalues[whlimit]

    ##The following are arrays of floats##                                      
    ##Position in Mpc/h##                                                       
    x = particlevalues['x']
    y = particlevalues['y']
    z = particlevalues['z']
                                            
    ##Calculate Position##                                                      
    posx = (x - hostx[j])
    posy = (y - hosty[j])
    posz = (z - hostz[j])
    pos = np.array(list(zip(posx, posy, posz)))
    host_pos = [hostx[j],hosty[j],hostz[j]]

    hw, hv = get_eigs(host_I[j],rvirs[j])
        
    new_pos = transform(pos,hv).T
    hA = host_I[j][0]
    hA2 = np.repeat(hA,len(new_pos)).reshape(3,len(new_pos)).T
    hB = host_I[j][1]
    hB2 = np.repeat(hB,len(new_pos)).reshape(3,len(new_pos)).T
    hC = host_I[j][2]
    para1 = (new_pos*hA2/norm(hA)).sum(axis=1)
    para2 = (hA/norm(hA)).T
    para = np.array((para2[0]*para1,para2[1]*para1,para2[2]*para1))
    perp = new_pos-para.T

    t = np.arccos(abs((new_pos*hA2).sum(axis=1)/(norm(new_pos,axis=1)*norm(hA))))*180./np.pi
    p = np.arccos((perp*hB2).sum(axis=1)/(norm(hB)*norm(perp,axis=1)))
    for k in range(len(ang_cut)):
        mass_frac_A_ang[j][k]+=particle_mass*len(new_pos[t<ang_cut[k]])/mvirs[j]

    m = 0
    for n in range(5):
        rand_t = random_rotation(hv,pos)*180/np.pi
        for k in range(len(ang_cut)):
            mass_frac_random[m][k]+=particle_mass*len(new_pos[rand_t<ang_cut[k]])/mvirs[j]
        m+=1
    np.savez('rhap_mass_frac_ang_2.npz',A = mass_frac_A_ang, rand = mass_frac_random)

    """
        frac_errs[k][j] += rotation_err_bs(hv,pos,ang_cut[k],mvirs[j],len(halo_names),n_rep=1000)
        np.save('rhap_mass_frac_errs.npy',frac_errs)
    #percentiles1 = np.percentile(np.log10(np.median(frac_errs,axis=0)),[15.9,84.1])
    #percentile_res = np.percentile(np.median(frac_errs,axis=1),[15.9,84.1])          
    #percentiles[k][0]+=percentiles[0]
    #percentiles[k][1]+=percentiles[1]

    #percentile_res = np.percentile(np.median(frac_errs,axis=1),[0.00006,99.99994])
    #percentiles5[k][0]+=percentile_res[0]
    #percentiles5[k][1]+=percentile_res[1]

    #np.savez('rhap_mass_frac_err.npz', one_sig = percentiles, five_sig = percentiles5)
    """
