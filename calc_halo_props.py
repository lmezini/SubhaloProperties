import numpy as np 
import numexpr as ne
import scipy
from scipy.optimize import minimize_scalar
import glob
from astropy.io import ascii
from astropy.table import Column, Table
#from calc_shape import calc_shape
import pandas as pd


def calc_ang_momentum(position,velocity):
    particle_mass = 281981.0 ##M_sun/h##
    L = np.cross(position, velocity).sum(axis=0)*particle_mass
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
"""
##Get All of the host values##
i = 0
rvirs = np.zeros((45))
mvirs = np.zeros((45))
hostx = np.zeros((45))
hosty = np.zeros((45))
hostz = np.zeros((45))
hostvx = np.zeros((45))
hostvy = np.zeros((45))
hostvz = np.zeros((45))
host_shapes = np.zeros((45))
host_spins = np.zeros((45))
host_cs = np.zeros((45))
host_cs_part = np.zeros((45))
hostJx = np.zeros((45))
hostJy = np.zeros((45))
hostJz = np.zeros((45))
for file in sorted(glob.glob('/home/lom31/mwm_halo_tables/Halo*/*')):
    if file.endswith('.list_final'):
        print(file)
        hostvalues = ascii.read(file, format = 'commented_header')
        ##These values are entire columns in the list file for convenience##
        ##Reads in a single float##
        ##Rvir in kpc/h##
        halo_rvir = hostvalues['host_rvir'][1]
        rvirs[i] = halo_rvir      
        halo_mvir = hostvalues['host_mvir'][1]
        mvirs[i] = halo_mvir
        ##Position in Mpc/h##
        halox = hostvalues['host_x'][1]
        hostx[i] = halox
        haloy = hostvalues['host_y'][1]
        hosty[i] = haloy
        haloz = hostvalues['host_z'][1]
        hostz[i] = haloz
        ##Velocity in km/s##
        halovx = hostvalues['host_vx'][1]
        hostvx[i] = halovx
        halovy = hostvalues['host_vy'][1]
        hostvy[i] = halovy
        halovz = hostvalues['host_vz'][1]
        hostvz[i] = halovz
        ##Angular momentum in Msun*Mpc*km/s
        halojx = hostvalues['host_Jx_ES'][1]
        hostJx[i] = halojx
        halojy = hostvalues['host_Jy_ES'][1]
        hostJy[i] = halojy
        halojz = hostvalues['host_Jz_ES'][1]
        hostJz[i] = halojz
        
        host_shape = hostvalues['host_c_to_a_ES'][1]
        host_shapes[i] = host_shape
        host_spin = hostvalues['host_spin_bullock_ES'][1]
        host_spins[i] = host_spin
        host_c = hostvalues['host_rvir'][1]/hostvalues['host_Rs_ES'][1]
        host_cs[i] = host_c
                
        i+=1
"""
host_vals = ascii.read('host_og_vals.table', format = 'commented_header') 
print(host_vals)

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

calculated_spins = np.zeros((45))
calculated_shapes = np.zeros((45))
calculated_cs = np.zeros((45))
calculated_Jx = np.zeros((45))
calculated_Jy = np.zeros((45))
calculated_Jz = np.zeros((45))


halo_names = []
host_ids = []
with open('halos_info.txt') as f:
    for l in f:
        this_halo, host_id, block, _ = l.split()
        halo_names.append(this_halo)
        host_ids.append(host_id)


#glob.glob('/home/lom31/particle_stuff/particle_tables/*'):
    ##Get information about the particles##                          
j = 0
for f in halo_names:
    print(f)
    fname = 'Halo023_all.particle_table'#/home/lom31/particle_stuff/particle_tables/{}_all.particle_table'.format(f)
    particlevalues = ascii.read(fname, format = 'commented_header')
    ##Make sure chosen particles are within the virial radius##
    r = Column(np.sqrt(particlevalues['x']**2+particlevalues['y']**2+particlevalues['z']**2),name='r')
    particlevalues.add_column(r)
    
    particle_mass = 281981.0
    whlimit = np.where(particlevalues['r']<=rvirs[j])
    particle_no = np.size(whlimit)

    ##The following are arrays of floats##
    ##Position in Mpc/h##
    x = particlevalues[whlimit]['x']
    y = particlevalues[whlimit]['y']
    z = particlevalues[whlimit]['z']
        ##Velocity in km/s##
    vx = particlevalues[whlimit]['vx']
    vy = particlevalues[whlimit]['vy']
    vz = particlevalues[whlimit]['vz']               

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
        
        ##Calculate shape by calling the imported function##
    force_res = 0.00017 #Mpc/h
    #b_to_a, c_to_a, eig_A = calc_shape(rvirs[j]*1e-3, [0,0,0], pos, force_res) #all same units!
    #calculated_shapes[j] = c_to_a
        
        ##Calculate angular momentum##
    ang_mom = calc_ang_momentum(pos,vel)
    calculated_Jx[j] = ang_mom[0]
    calculated_Jy[j] = ang_mom[1]
    calculated_Jz[j] = ang_mom[2]                


    #mass = particle_no*particle_mass
    #print(mass/mvirs[j])
        ##Calculate Bullock Spin##
    spin_bullock = calc_spin_bullock(ang_mom,mvirs[j],rvirs[j])
    calculated_spins[j] = spin_bullock
        
        ##Calculate concentration##
    x = np.sqrt((pos*pos).sum(axis=-1))/(rvirs[j]*1e-3)
    x_lim = 1.0
    c = calc_concentration(x,x_lim)
    calculated_cs[j] = c
    j += 1

#og_vals = np.vstack((rvirs,mvirs,hostx,hosty,hostz,hostvx,hostvy,
#    hostvz,host_shapes,host_spins,host_cs,hostJx,hostJy,hostJz))

#print(calculated_spins,calculated_shapes,calculated_cs,calculated_Jx,calculated_Jy,calculated_Jz)
no_sub_vals = np.vstack((calculated_spins,calculated_shapes,calculated_cs,calculated_Jx,calculated_Jy,calculated_Jz))

#host_df = pd.DataFrame(data=og_vals.T, columns=["rvir","mvir","hostx","hosty","hostz","hostvx","hostvy",
#        "hostvz","host_shapes","host_spins","host_cs","hostJx","hostJy","hostJz"])
#host_og_vals = Table.from_pandas(host_df)
#host_og_vals.write('host_og_vals.table',format='ascii.commented_header')

no_subs_df = pd.DataFrame(data=no_sub_vals.T, columns=["calc_spin","calc_shape","calc_cs","calc_Jx","calc_Jy","calc_Jz"])
host_no_subs_vals = Table.from_pandas(no_subs_df)
del no_subs_df
host_no_subs_vals.write('host_all_vals_num_part_cut.table',format='ascii.commented_header',overwrite=True)
