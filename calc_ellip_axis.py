import numpy as np 
from astropy.io import ascii
from numpy.linalg import eig
import sympy as sp

def calc_inertia(p,s,q):
    #p is position vector
    #s,q are axis ratios
    Ixx = np.sum(p[0]*p[0])
    Iyy = np.sum(p[1]*p[1])
    Izz = np.sum(p[2]*p[2])
    Ixy = np.sum(p[0]*p[1])
    Iyz = np.sum(p[1]*p[2])
    Ixz = np.sum(p[0]*p[2])
    Iyx = Ixy
    Izy = Iyz
    Izx = Ixz
    r2 = np.sum(p[0]**2 + (p[1]/q)**2 + (p[2]/s)**2)
    I = np.array(((Ixx,Ixy,Ixz),(Iyx,Iyy,Iyz),(Izx,Izy,Izz)))/r2
    
    return I

def get_axis(w):
    c,b,a = np.sqrt(w)
    s = c/a
    q = b/a
    
    return s,q,a

def get_eigs(I, rvir):
    #return sorted eigenvectors and eigenvalues
    w,v = eig(I)

    #sort by size
    odr = np.argsort(w)
    w = w[odr]
    v = v[odr]

    #rescale so major axis = radius of original halo
    ratio = (0.001*rvir)/w[2]
    w[2] = w[2]*ratio
    w[0] = w[0]*ratio
    w[1] = w[1]*ratio

    return w,v



#s,q = 1,1 #initial values for a sphere
#err = 1.0
#tol = 0.01
#data = np.array((x,y,z))

def calc_axis(data, rvir, s = 1, q =1, err = 1.0, tol = 1e-2):
    while err > tol:
        s_old = s
        q_old = q
        I = calc_inertia(data,s,q)
        w,v = get_eigs(I,rvir)
        s,q,a = get_axis(w)

        new_data = []
        for i in range(np.shape(data)[1]):
            if np.all(data[:,i] < new_axis):
                new_data.append(data[:,i])
        data = np.array(new_data).T    

        err1 = np.abs(s-s_old)/s_old
        err2 = np.abs(q-q_old)/q_old
        err = np.max(err1,err2)

        c,b,a = s*a,q*a,a

        return c,b,a,v
