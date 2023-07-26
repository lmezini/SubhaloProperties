import numpy as np
import numexpr as ne
import scipy
from scipy.optimize import minimize_scalar
import glob
from helpers.io_utils import hlist2pandas
from astropy.io import ascii
from astropy.table import Column, Table
from calc_shape import calc_shape
import pandas as pd
import calc_ellip_axis


def calc_ang_momentum(position,velocity,h_mass=None,s_mass=None,weight=False):
    particle_mass = 1.3e8 ##M_sun/h##
    if weight == True:
        L = np.cross(position, velocity).sum(axis=0)*particle_mass*(s_mass/h_mass)
    else:
        L = np.cross(position, velocity).sum(axis=0)*particle_mass
    return L

def calc_orb_ang_momentum(position,velocity,s_mass):
 
    L = np.cross(position, velocity)*s_mass
    
    return L
    
def calc_spin_bullock(L,mvir,rvir):
    spin_unscale = np.float(np.sqrt((L*L).sum()))
    G = 4.302*(10**-9.) # in Mpc Msun^-1 (km/s)^2 
    spin_bullock = spin_unscale/(np.sqrt(2.*G*mvir**3.*rvir*1e-3)) #multiply by 1e-3 to convert Rvir to Mpc/h
    return spin_bullock
    
def calc_concentration(x, x_lim=1.0):
    x = np.asanyarray(x)
    x = x[x <= x_lim]
    x = x[x > 0]
    n = float(len(x))
    def obj_func(c_log):
        c = np.exp(c_log)
        k = c * x_lim + 1.0
        a = np.log((1.0/k + np.log(k) - 1.0)/(c*c))
        return a - ne.evaluate('sum(log(x/(1.0+c*x)**2.0))', {'c':c, 'x':x}, {})/n
    res = minimize_scalar(obj_func, bounds=(0, 7.0), method='bounded')
    if not res.success:
        raise ValueError('cannot obtain fit')
    return np.exp(res.x)

host_vals = ascii.read('host_og_vals_rhap.table', format = 'commented_header') 

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

sub_mass = np.zeros(96)

halo_names = []
host_ids = []
with open('halos_info_cut.txt') as f:
    for l in f:
        this_halo, host_id, block, _ = l.split()
        halo_names.append(this_halo)
        host_ids.append(host_id)

#glob.glob('/home/lom31/particle_stuff/particle_tables/*'):
    ##Get information about the particles##                          
j = 0
for f in halo_names:
    print(f)

    # load host    
    path = '/home/lom31/rhap_particles/rhapsody/{}/rockstar/'.format(f)
    hostvalues = hlist2pandas(path + '/out_199.list')
    #hostvalues.rename(columns={c: c.lower() for c in hostvalues.columns})
    hostvalues = Table.from_pandas(hostvalues)

    fname = '/home/lom31/rhap_particles/particle_tables/{}_all.particle_table'.format(f)
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
        ##Velocity in km/s##
    vx = particlevalues['vx']
    vy = particlevalues['vy']
    vz = particlevalues['vz']               

    ##Now lets do some calculations ##
    ##Calculate Position##
    posx = (x - hostx[j])
    posy = (y - hosty[j]) 
    posz = (z - hostz[j]) 
    pos = np.array(list(zip(posx, posy, posz)))
    host_pos = [hostx[j],hosty[j],hostz[j]]
    
        ##Calculate velocity##            
    velx = (vx - hostvx[j])
    vely = (vy - hostvy[j])
    velz = (vz - hostvz[j])
    vel = np.array(list(zip(velx, vely, velz)))
    
    """
    
    ##Calculate shape by calling the imported function##
    
    force_res = 0.013 #Mpc/h
    b_to_a, c_to_a, eig_A = calc_shape(rvirs[j]*1e-3, [0,0,0], pos, force_res) #all same units!
    calculated_shapes[j] = c_to_a
    
    calculated_a_ax_x[j] = eig_A[0]
    calculated_a_ax_y[j] = eig_A[1]
    calculated_a_ax_z[j] = eig_A[2]

        ##Calculate angular momentum##
    ang_mom = calc_ang_momentum(pos,vel)
    calculated_Jx[j] = ang_mom[0]
    calculated_Jy[j] = ang_mom[1]
    calculated_Jz[j] = ang_mom[2]                
    """

    #rvir in kpc
    #spin_bullock = calc_spin_bullock(ang_mom,mvirs[j],rvirs[j])
    #calculated_spins[j] = spin_bullock
    #np.save('rhap_mass_cut_spin_all.npy',calculated_spins)
#hostvalues = hostvalues[srtd][0:-400]
    #sub_ms = []
    #sub_js = []
    sub_pos = []
    #sub_vs = []
    #sub_Ls = []

    for h in hostvalues:
        #select set of particles belonging to halo
        p_set = particlevalues["smallest_external_haloid"] == h['ID']
        if len(particlevalues[p_set]) != 0:
            #v_temp = vel[p_set]
            #tot_sub_vel[j] += v_temp
            
            #p_temp = pos[p_set]
            #s_mass = h['Mvir']
            #if s_mass > 0.001*mvirs[j]:
            #    sub_mass[j]+=s_mass
            
            posx = (h['X'] - hostx[j])
            posy = (h['Y'] - hosty[j])
            posz = (h['Z'] - hostz[j])
            pos2 = np.array((posx, posy, posz))
            
            sub_pos.append(pos2)

            #velx = (h['VX'] - hostvx[j])
            #vely = (h['VY'] - hostvy[j])
            #velz = (h['VZ'] - hostvz[j])
            #vel2 = np.array((velx, vely, velz))
            #sub_vs.append(vel2)
            #ang_mom = calc_orb_ang_momentum(pos2,vel2,s_mass)
            #sub_Ls.append(ang_mom)
    
    np.save('sub_pos_{}.npy'.format(f),sub_pos)
    #np.save('rhap_sub_mass.npy',sub_mass)
    #calculated_Jx_w[j] += ang_mom_w[0]
        #calculated_Jy_w[j] += ang_mom_w[1]
        #calculated_Jz_w[j] += ang_mom_w[2]
            #sub_js.append(calc_ang_momentum(p_temp,v_temp))
    #np.save('rhap_sub_Ls/{}_sub_Ls.npy'.format(f),np.array(sub_Ls))
    #np.save('rhap_sub_vel/{}_sub_vel.npy'.format(f),np.array(sub_vs))
    #np.save('rhap_sub_js_2/{}_sub_js_2.npy'.format(f),np.array(sub_js))
    #np.save('rhap_sub_js_mass_cut/{}_sub_js_mass_cut.npy'.format(f),np.array(sub_js))
    #np.save('rhap_sub_mass_2/{}_sub_mass_2.npy'.format(f),np.array(sub_ms))
    #mass = particle_no*particle_mass
    #print(mass/mvirs[j])
        ##Calculate Bullock Spin##
    #spin_bullock = calc_spin_bullock(ang_mom,mvirs[j],rvirs[j])
    #calculated_spins[j] = spin_bullock
        
        ##Calculate concentration##
    #x = np.sqrt((pos*pos).sum(axis=-1))/(rvirs[j]*1e-3)
    #x_lim = 1.0
    #c = calc_concentration(x,x_lim)
    #calculated_cs[j] = c
    
    j += 1

"""
og_vals = np.vstack((rvirs,mvirs,hostx,hosty,hostz,hostvx,hostvy,
    hostvz,host_shapes,host_spins,host_cs,hostJx,hostJy,hostJz))

#print(calculated_spins,calculated_shapes,calculated_cs,calculated_Jx,calculated_Jy,calculated_Jz)
"""
#no_sub_vals = np.vstack((calculated_spins,calculated_shapes,calculated_cs,calculated_Jx,calculated_Jy,calculated_Jz,calculated_Jx_w,calculated_Jy_w,calculated_Jz_w,calculated_a_ax_x,calculated_a_ax_y,calculated_a_ax_z))
"""
host_df = pd.DataFrame(data=og_vals.T, columns=["rvir","mvir","hostx","hosty","hostz","hostvx","hostvy",
        "hostvz","host_shapes","host_spins","host_cs","hostJx","hostJy","hostJz"])

host_og_vals = Table.from_pandas(host_df)
host_og_vals.write('host_og_vals_rhap.table',format='ascii.commented_header',overwrite=True)

"""
#no_subs_df = pd.DataFrame(data=no_sub_vals.T, columns=["calc_spin","calc_shape","calc_cs","calc_Jx","calc_Jy","calc_Jz","calc_Jx_w","calc_Jy_w","calc_Jz_w","calc_a_ax_x","calc_a_ax_y","calc_a_ax_z"])
#host_no_subs_vals = Table.from_pandas(no_subs_df)
#del no_subs_df
#host_no_subs_vals.write('rhap_all_2.table',format='ascii.commented_header',overwrite=True)
