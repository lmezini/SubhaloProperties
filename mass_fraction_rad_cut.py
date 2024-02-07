import os
import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column
import sys
from numpy.linalg import norm, eig
import random
import glob
from helpers.io_utils import hlist2pandas


def get_angle(v1, v2):
    c = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )  # -> cosine of the angle

    return c


def generate_random_z_axis_rotation():
    """Generate random rotation matrix about the z axis."""
    R = np.eye(3)
    x1 = np.random.rand()
    R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
    R[0, 1] = -np.sin(2 * np.pi * x1)
    R[1, 0] = np.sin(2 * np.pi * x1)

    return R


def uniform_random_rotation(x):
    """Apply a random rotation in 3D, with a distribution uniform over the
    sphere.

    Arguments:
        x: vector or set of vectors with dimension (n, 3), where n is the
            number of vectors

    Returns:
        Array of shape (n, 3) containing the randomly rotated vectors of x,
        about the mean coordinate of x.

    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """

    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()

    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = np.array([np.cos(x2) * np.sqrt(x3), np.sin(x2)
                 * np.sqrt(x3), np.sqrt(1 - x3)])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = np.mean(x, axis=0)

    return ((x - mean_coord) @ M) + mean_coord @ M


def get_random_particle_rad_dist(host_I, pos, rvir):
    """Transform particle position to randomly rotated coordinate system

    Args:
        host_I (_type_): host halo inertia tensor
        pos (_type_): particle position coordinates (x,y,z)
        rvir (_type_): halo virial radius

    Returns:
        _type_: 2d projected distance of particle from halo center
    """
    hw, hv = get_eigs(host_I, rvir)

    new_hv = uniform_random_rotation(hv)

    hA = new_hv[0]

    angle = np.dot(new_hv[0], hv[0]) / (
        np.linalg.norm(new_hv[0]) * np.linalg.norm(hv[0])
    )  # -> cosine of the angle

    hA2 = np.repeat(hA, len(pos)).reshape(3, len(pos)).T

    para1 = (pos * hA2 / norm(hA)).sum(axis=1)
    para2 = (hA / norm(hA)).T
    para = np.array((para2[0] * para1, para2[1] * para1, para2[2] * para1))
    perp = pos - para.T
    rad_dist = np.sqrt(np.sum(perp**2, axis=1))

    return rad_dist, angle


def get_eigs(I, rvir):

    # return eigenvectors and eigenvalues
    w, v = eig(I)
    # sort in descending order
    odr = np.argsort(-1.*w)
    # sqrt of e values = a,b,c
    w = np.sqrt(w[odr])
    v = v.T[odr]
    # rescale so major axis = radius of original host
    ratio = rvir/w[0]
    w[0] = w[0]*ratio  # this one is 'a'
    w[1] = w[1]*ratio  # b
    w[2] = w[2]*ratio  # c

    return w, v


particle_mass = 281981.0  # M_sun/h##
host_vals = ascii.read('host_og_vals_mw_new.table', format='commented_header')
host_I = np.load('inertia_tensor_stuff/mwm_host_inertia_tensor_no_norm_v2.npy')

rvirs = host_vals['rvir']
mvirs = host_vals['mvir']
hostx = host_vals['hostx']
hosty = host_vals['hosty']
hostz = host_vals['hostz']
hostvx = host_vals['hostvx']
hostvy = host_vals['hostvy']
hostvz = host_vals['hostvz']
host_shapes = host_vals['host_shapes']
host_spins = host_vals['host_spins']
host_cs = host_vals['host_cs']
hostJx = host_vals['hostJx']
hostJy = host_vals['hostJy']
hostJz = host_vals['hostJz']


rad_cut = np.logspace(np.log10(0.001), np.log10(1.), 100)

mass_frac_A = np.zeros((len(host_vals), len(rad_cut)))
mass_frac_random = np.zeros((int(len(host_vals)*5), len(rad_cut)))
random_angle = np.zeros((int(len(host_vals)*5), len(rad_cut)))

halo_names = []
host_ids = []

with open('halos_info.txt') as f:
    for l in f:
        this_halo, host_id, block, _ = l.split()
        halo_names.append(this_halo)
        host_ids.append(host_id)

m = 0
for j, f in enumerate(halo_names):
    print(f)
    # load host
    path = '/home/cef41/Outputs/{}/'.format(f)
    hostvalues = hlist2pandas(path + '/out_0.list')
    hostvalues = Table.from_pandas(hostvalues)

    fname = '/home/lom31/particle_stuff/particle_tables/{}_sub_1rvir.particle_table'.format(
        f)
    particlevalues = ascii.read(fname, format='commented_header')
    print(particlevalues.keys())
    ## Make sure chosen particles are within the virial radius##
    r = Column(np.sqrt((particlevalues['x']-hostx[j])**2 +
                       (particlevalues['y']-hosty[j])**2 +
                       (particlevalues['z']-hostz[j])**2), name='r')

    particlevalues.add_column(r)

    whlimit = np.where(particlevalues['r'] <= rvirs[j]*.001)
    particlevalues = particlevalues[whlimit]

    ## The following are arrays of floats##
    ## Position in Mpc/h##
    x = particlevalues['x']
    y = particlevalues['y']
    z = particlevalues['z']

    ## Calculate Position##
    posx = (x - hostx[j])
    posy = (y - hosty[j])
    posz = (z - hostz[j])
    pos = np.array(list(zip(posx, posy, posz)))
    host_pos = [hostx[j], hosty[j], hostz[j]]

    hw, hv = get_eigs(host_I[j], rvirs[j])

    hA = hv[0]
    hA2 = np.repeat(hA, len(pos)).reshape(3, len(pos)).T

    para1 = (pos * hA2 / norm(hA)).sum(axis=1)
    para2 = (hA / norm(hA)).T
    para = np.array((para2[0] * para1, para2[1] * para1, para2[2] * para1))
    perp = pos - para.T
    rad_dist = np.sqrt(np.sum(perp**2, axis=1))

    for k in range(len(rad_cut)):
        cut_off = 0.001*rvirs[j]*rad_cut[k]
        mass_frac_A[j][k] += particle_mass * \
            len(pos[rad_dist < cut_off])/mvirs[j]

    for n in range(5):
        rand_rad_dist, angle = get_random_particle_rad_dist(
            host_I[j], pos, rvirs[j])
        for k in range(len(rad_cut)):
            cut_off = 0.001*rvirs[j]*rad_cut[k]
            mass_frac_random[m][k] += particle_mass * \
                len(pos[rand_rad_dist < cut_off])/mvirs[j]
            random_angle[m][k] += angle
        m += 1

    np.savez('mwm_mass_frac_rad_cut.npz', A=mass_frac_A,
             rand=mass_frac_random, angle=random_angle)
