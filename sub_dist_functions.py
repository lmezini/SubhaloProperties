import numpy as np 
from scipy import stats
from scipy.stats import pearsonr,spearmanr
from numpy.linalg import norm,eig
from astropy.io import ascii
import random


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


def get_radii(sub_pos,host_e_vects):
    new_pos = transform(sub_pos,host_e_vects).T

    para = np.zeros(np.shape(new_pos))
    perp = np.zeros(np.shape(new_pos))
    #p = np.zeros(len(new_pos))
    #t = np.zeros(len(new_pos))
       
    hA = host_e_vects[0]
    hA2 = np.repeat(hA,len(new_pos)).reshape(3,len(new_pos)).T
    hB = host_e_vects[1]
    #hB2 = np.repeat(hB,len(new_pos)).reshape(3,len(new_pos)).T
    #hC = host_e_vects[2]

    para1 = (new_pos*hA2/norm(hA)).sum(axis=1)
    para2 = (hA/norm(hA)).T
    para = np.array((para2[0]*para1,para2[1]*para1,para2[2]*para1))
    perp = new_pos-para.T

    r = np.sqrt(np.sum(perp**2,axis=1))

    return r


def MC_err(func, n, n_rep=1000, val_min1 = 0.0, val_max1 = 1.0,val_min2 = 0.0, val_max2 = 1.0):

    """
    func options: pearson, spearman, uniform
    return: percentiles
    """
    if func == 'pearson':
        prsnr = np.zeros(n_rep)
        for i in range(n_rep):
            pdfs1 = val_min1 + np.random.random(n)*val_max1
            pdfs2 = val_min2 + np.random.random(n)*val_max2
            prsnr[i]+=stats.pearsonr(pdfs1,pdfs2)[0]

        percentiles = np.percentile(prsnr,[15.9,84.1],axis=0)

    if func == "uniform":
        cdfs = np.zeros((n_rep,n))
        for i in range(n_rep):
            cdfs[i]+=np.sort(np.random.uniform(val_min1, val_max1, n))
            cdfs[i][0] = 0.0
            cdfs[i][n-1] = 1.0
        err = np.std(cdfs,axis=0)

        percentiles = np.percentile(cdfs,[15.9,84.1],axis=0)

    
    if func == "spearman":
        spears = np.zeros(n_rep)
        for i in range(n_rep):
            pdfs1 = val_min1 + np.random.random(n)*val_max1
            pdfs2 = val_min2 + np.random.random(n)*val_max2
            spears[i]+=stats.spearmanr(pdfs1,pdfs2).correlation
        
        percentiles = np.percentile(cdfs,[15.9,84.1],axis=0)

    return percentiles

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

def random_rotation(hv,host_I,pos):
    
    np.random.seed()
    tx=np.random.random()*90.0
    ty=np.random.random()*90.0
    tz=np.random.random()*90.0

    new_I = rotate(host_I,tx,ty,tz)
    new_pos = transform(pos,hv).T
    
    para = np.zeros(np.shape(new_pos))
    perp = np.zeros(np.shape(new_pos))
    p = np.zeros(len(new_pos))
    t = np.zeros(len(new_pos))
    hA = new_I[0]
    hB = new_I[1]
    
    for i in range(len(new_pos)):
        para[i]+=np.dot(new_pos[i],hA/norm(hA))*(hA/norm(hA))
        perp[i]+=new_pos[i]-para[i]
        t[i]+=np.arccos(np.dot(new_pos[i],hA)/(norm(new_pos[i])*norm(hA)))
        p[i]+=np.arccos(np.dot(perp[i],hB)/(norm(hB)*norm(perp[i])))
    r = np.sqrt(np.sum(perp**2,axis=1))

    return r

def bs_mass_frac_err(mass_frac_arr,n_boot,n_samp,p1=15.9,p2=84.1):
    n_rad, n_halos = np.shape(mass_frac_arr.T)
    boot_vals = np.zeros((n_boot,n_samp,n_rad))
    for i in range(n_boot):
        for j in range(n_samp):
            indx = random.randint(0, n_halos-1)
            boot_vals[i][j]+=mass_frac_arr[indx]
    meds = np.median(boot_vals,axis=1)
    percentiles = np.percentile(meds,[p1,p2],axis=0)
    
    return percentiles