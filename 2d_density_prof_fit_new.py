import numpy as np
from astropy import units as u
from numpy.linalg import norm, eig
import math
from scipy.optimize import curve_fit
from astropy.io import ascii
from lenstronomy.Cosmo import nfw_param
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed


from halo_orientation_functions import (
    uniform_random_rotation,
    get_eigs,
    transform,
    rotate_position,
)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
z_cosmo = 0
n_angles = 500
n_core = 30


def sigma_nfw(r, rs, overdens):
    x = r / rs
    return np.piecewise(
        x,
        [x < 1, x == 1, x > 1],
        [
            lambda x: 2
            * rs
            * overdens
            / (x**2 - 1)
            * (1 - 2 / np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))),
            lambda x: 2 * rs * overdens / 3.0,
            lambda x: 2
            * rs
            * overdens
            / (x**2 - 1)
            * (1 - 2 / np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x)))),
        ],
    )


def calc_2d_shape(p):
    # p is position vector

    r2 = p[0] ** 2 + p[1] ** 2
    Ixx = np.sum((p[0] * p[0]) / r2)
    Iyy = np.sum((p[1] * p[1]) / r2)
    Ixy = np.sum((p[0] * p[1]) / r2)
    Iyx = Ixy

    I = np.array(((Ixx, Ixy), (Iyx, Iyy)))

    # return eigenvectors and eigenvalues
    w, v = eig(I)
    # sort in descending order
    odr = np.argsort(-1.0 * w)
    # sqrt of e values = a,b,c
    w = np.sqrt(w[odr])
    short = w[1]
    long = w[0]

    return long, short


def calc_halo_props(normalization, rs, z=None, cosmo=None):
    # normalization is in units of h^2*Mo/Mpc^3
    # convert from Kpc by multiplying by 1e9
    if cosmo == None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    if z == None:
        z = 0.0
    c = nfw_param.NFWParam(cosmo=cosmo).c_rho0(normalization, z)
    m200 = nfw_param.NFWParam(cosmo=cosmo).M200(rs * 1e-3, normalization, c)

    return c, m200


def rs_fit(nbins, r, rvir, particle_mass):
    # fit 2d NFW profile for scale radius and overdensity
    # calculate mass from fit

    # rvir in Kpc (halo radius)
    # r in Mpc (particle radial position)

    # return rs and normalization (units h^2 Mo/Kpc^3) (density)
    # fit out to 1/4 virial radius
    bin_edges = np.linspace(1, rvir/4., nbins)
    counts, bins = np.histogram(r * 1000.0, bin_edges)

    bin_area = np.pi * (
        bin_edges[1: len(bin_edges)] ** 2 -
        bin_edges[0: len(bin_edges) - 1] ** 2
    )
    rho = counts * particle_mass / bin_area
    bin_cens = (bin_edges[1:nbins] + bin_edges[0: nbins - 1]) / 2
    sigma = np.sqrt(counts) * particle_mass / bin_area
    sigma[np.where(sigma == 0)] = 1
    p, e = curve_fit(
        sigma_nfw, bin_cens, rho, sigma=sigma, bounds=[
            [0.001, 0.001], [2*rvir/3., np.inf]]
    )
    e0, e1 = np.sqrt(np.diag(e))

    return rho, bin_cens, sigma, p[0], p[1], e0, e1


def calc_q(halo_e_vals, radius):
    # q = minor/major axis
    # halo eigen values (major,semi-major,minor)
    q = halo_e_vals[2] / radius

    return q


def get_angle(v1, v2):
    c = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )  # -> cosine of the angle

    return c


def random_rotation_2d_pos(new_pos, new_hv):

    hA = new_hv[0]
    hB = new_hv[1]
    hC = new_hv[2]
    hA2 = np.repeat(hA, len(new_pos)).reshape(3, len(new_pos)).T
    hB2 = np.repeat(hB, len(new_pos)).reshape(3, len(new_pos)).T
    hC2 = np.repeat(hC, len(new_pos)).reshape(3, len(new_pos)).T

    para1 = (new_pos * hA2 / norm(hA)).sum(axis=1)
    para2 = (hA / norm(hA)).T
    para = np.array((para2[0] * para1, para2[1] * para1, para2[2] * para1))
    perp = new_pos - para.T

    para1 = (perp * hB2 / norm(hB)).sum(axis=1)
    para2 = (hB / norm(hB)).T
    b = np.array((para2[0] * para1, para2[1] * para1, para2[2] * para1))
    neg = np.where(para1 < 0)

    b_coord = np.sqrt(np.sum(b.T**2, axis=1))
    b_coord[neg] = b_coord[neg] * (-1)

    para1 = (perp * hC2 / norm(hC)).sum(axis=1)
    para2 = (hC / norm(hC)).T
    c = np.array((para2[0] * para1, para2[1] * para1, para2[2] * para1))
    c_coord = np.sqrt(np.sum(c.T**2, axis=1))

    neg = np.where(para1 < 0)
    c_coord[neg] = c_coord[neg] * (-1)

    return b_coord, c_coord


particle_mass = 281981.0  # M_sun/h##
host_vals = ascii.read("host_og_vals_mw_new.table", format="commented_header")
host_I = np.load("inertia_tensor_stuff/mwm_host_inertia_tensor_no_norm_v2.npy")

# halo_idx = [22,35]
rvirs = host_vals["rvir"]
mvirs = host_vals["mvir"]
hostx = host_vals["hostx"]
hosty = host_vals["hosty"]
hostz = host_vals["hostz"]

halo_names = []
host_ids = []

with open("halos_info.txt") as f:
    for l in f:
        this_halo, host_id, block, _ = l.split()
        halo_names.append(this_halo)
        host_ids.append(host_id)

# np.zeros((96, n_angles, 6))
# density_vals = np.zeros((96, n_angles, 3, 2999))


def do_fit(j, halo_names):
    fname = "/home/lom31/particle_stuff/particle_tables/{}_all_1rvir.particle_table".format(
        halo_names
    )
    particlevalues = ascii.read(fname, format="commented_header")

    ## The following are arrays of floats##
    ## Position in Mpc/h##
    x = particlevalues["x"]
    y = particlevalues["y"]
    z = particlevalues["z"]

    ## Calculate Position##
    posx = x - hostx[j]
    posy = y - hosty[j]
    posz = z - hostz[j]

    dist = np.sqrt((posx) ** 2 + (posy) ** 2 + (posz) ** 2)

    whlimit = np.where(dist <= 0.001 * rvirs[j])
    del dist
    particlevalues = particlevalues[whlimit]

    ## The following are arrays of floats##
    ## Position in Mpc/h##
    x = particlevalues["x"]
    y = particlevalues["y"]
    z = particlevalues["z"]

    posx = x - hostx[j]
    posy = y - hosty[j]
    posz = z - hostz[j]
    dist = np.sqrt((posx) ** 2 + (posy) ** 2 + (posz) ** 2)

    pos = np.array(list(zip(posx, posy, posz)))
    nbins = 1000  # int(math.ceil(len(dist)/200))

    r, new_pos, new_hv, angle = rotate_position(
        host_I[j], pos, 0.001 * rvirs[j])
    x, y = random_rotation_2d_pos(new_pos, new_hv)
    b, c = calc_2d_shape([x, y])
    q = c / b
    true_rho, bins, err, rs, rho, e_rs, e_rho = rs_fit(
        nbins, r, rvirs[j], particle_mass
    )
    # np.save('{}_{}_density.npy'.format(f,rotate_angles[i]),np.array((bins,true_rho,err)))
    c, m200 = calc_halo_props(rho * 1e9, rs, z=z_cosmo, cosmo=cosmo)

    fit_res = np.array([rs, rho, c, m200, q, angle, e_rs, e_rho])
    # fit_vals[j, i] += fit_res

    density_val = np.array([bins, true_rho])

    del particlevalues

    return fit_res, density_val


h_id = []
rs_arr = []
rho_arr = []
c_arr = []
m200_arr = []
q_arr = []
angle_arr = []
e_rs_arr = []
e_rho_arr = []

bins = []
true_rho = []
err = []

with Parallel(n_jobs=n_core) as parallel:
    for j in range(len(halo_names)):
        print(j)
        results = parallel(delayed(do_fit)(
            j, halo_names[j]) for i in range(n_angles))
        results = np.array(results).T

        fit_vals = results[0]
        fit_vals = np.vstack(fit_vals).T

        h_id.append(np.full(n_angles, j, dtype=int))
        rs_arr.append(fit_vals[0])
        rho_arr.append(fit_vals[1])
        c_arr.append(fit_vals[2])
        m200_arr.append(fit_vals[3])
        q_arr.append(fit_vals[4])
        angle_arr.append(fit_vals[5])
        e_rs_arr.append(fit_vals[6])
        e_rho_arr.append(fit_vals[7])

        np.savez(
            "mwm_lens_prop_fit_new_ang_w_err_all_quarter_rvir_w_h_id.npz",
            h_id=h_id,
            rs=rs_arr,
            rho=rho_arr,
            c=c_arr,
            m200=m200_arr,
            q=q_arr,
            angles=angle_arr,
            e_rs=e_rs_arr,
            e_rho=e_rho_arr,
        )

        density_vals = results[1]

        true_rho.append(density_vals)

        np.savez("mwm_lens_prop_fit_density_all_quarter_rvir_w_h_id.npz",
                 #         bins = bins,
                 true_rho=true_rho,
        )
