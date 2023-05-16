import numpy as np 
import numexpr as ne
import scipy
from scipy.optimize import minimize_scalar
import glob
from astropy.io import ascii
from astropy.table import Column, Table
from calc_shape import calc_shape
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
num_subs = np.zeros((45))
scale = np.zeros((45))
scale_lmm = np.zeros((45))
scale_hm = np.zeros((45))
sub_mass = np.zeros((45))
max_mass_sub = np.zeros((45))
sat_dist = np.zeros((45))
sub_mass_far = np.zeros((45))

sub_mass_list = []

#for hlist
names =['scale','ID','desc_scale','desc_id','num_prog','pid','upid','desc_pid',
    'phantom','sam_mvir','Mvir','Rvir','Rs', 'vrms', 'mmp', 'scale_of_last_MM',
    'Vmax', 'X','Y','Z','VX','VY','VZ','JX','JY','JZ','Spin','Breadth_first_ID',
    'Depth_first_ID','Tree_root_ID','Orig_halo_ID','Snap_num',
    'Next_coprogenitor_depthfirst_ID','Last_progenitor_depthfirst_ID',
    'Rs_Klypin','Mvir_all','M200b','M200c','M500c','M2500c','Xoff','Voff',
    'spin_bullock','b_to_a','c_to_a','A[x]','A[y]','A[z]','b_to_a(500c)',
    'c_to_a(500c)','A[x](500c)','A[y](500c)','A[z](500c)','T/|U|','Macc',
    'Mpeak','Vacc','Vpeak','Halfmass_Scale','Acc_Rate_Inst','Acc_Rate_100Myr',
    'Acc_Rate_Tdyn']


#for file in sorted(glob.glob('/home/lom31/rhap_particles/rhapsody/Halo*/rockstar/*')):
#    if file.endswith('199.list'):
#        print(file)
#        hostvalues = ascii.read(file, format = 'commented_header')
halo_names = []
host_ids = []
with open('hlist_halo_ids.txt') as f:
    for l in f:
        j = 0
        this_halo, host_id = l.split()#, block, _ = l.split()
        halo_names.append(this_halo)
        host_ids.append(host_id)
        j+=1 

##These values are entire columns in the list file for convenience##
        ##Reads in a single float##
        ##Rvir in kpc/h##
for halo in halo_names:
    #print(halo)
    #hostvalues = ascii.read('/Users/lmezini/proj_2/Halos_Recalculated/{}/out_0.list'.format(halo), format = 'commented_header')
    hostvalues = ascii.read('/Users/lmezini/proj_2/rs_files/Halo{}/hlist.list'.format(halo),names=names)
    loc = int(np.where(hostvalues['ID']==int(host_ids[i]))[0][0])
    if type(loc)==int:
        print(i)
        halo_rvir = hostvalues['Rvir'][loc]
        halo_mvir = hostvalues['Mvir'][loc]
        mvirs[i] = halo_mvir
        rvirs[i] = halo_rvir
 
        #sub_mass[i] = np.sum(subs[whlimit]['Mvir'])
        #max_mass_sub[i] = np.max(subs[whlimit]['Mvir'])
        #num_subs[i] = np.shape(whlimit)[1]-1

        scale[i] = hostvalues[loc]['scale']
        scale_lmm[i] = hostvalues[loc]['scale_of_last_MM']
        scale_hm[i] = hostvalues[loc]['Halfmass_Scale']

        ##Position in Mpc/h##
        halox = hostvalues['X'][loc]
        hostx[i] = halox
        haloy = hostvalues['Y'][loc]
        hosty[i] = haloy
        haloz = hostvalues['Z'][loc]
        hostz[i] = haloz

        ##Velocity in km/s##
        halovx = hostvalues['VX'][loc]
        hostvx[i] = halovx
        halovy = hostvalues['VY'][loc]
        hostvy[i] = halovy
        halovz = hostvalues['VZ'][loc]
        hostvz[i] = halovz

        ##Angular momentum in Msun*Mpc*km/s
        halojx = hostvalues['JX'][loc]
        hostJx[i] = halojx
        halojy = hostvalues['JY'][loc]
        hostJy[i] = halojy
        halojz = hostvalues['JZ'][loc]
        hostJz[i] = halojz
        
        host_shape = hostvalues['c_to_a'][loc]
        host_shapes[i] = host_shape
        host_spin = hostvalues['spin_bullock'][loc]
        host_spins[i] = host_spin
        host_c = hostvalues['Rvir'][loc]/hostvalues['Rs'][loc]
        host_cs[i] = host_c
                

        hostvalues.remove_row(loc)
        whlimit = np.where(hostvalues['upid']==int(host_ids[i]))
        subs = hostvalues[whlimit]

        dist = np.sqrt((subs['X']-hostx[i])**2+(subs['Y']-hosty[i])**2+(subs['Z']-hostz[i])**2)
        whlimit = np.where(dist<halo_rvir*0.001)
        subs = subs[whlimit]

        #sub_mass_list.append(subs['Mvir']/halo_mvir)
        #np.save('mwm_sub_mass.npy',np.concatenate(sub_mass_list))

        whlimit = np.where(subs['Vmax']>10.)
        subs = subs[whlimit]
        whlimit = np.where(subs['Mvir']>0.001*halo_mvir)
        subs = subs[whlimit]
        #whlimit = np.where(subs['Mvir']<0.1*halo_mvir)
        #subs = subs[whlimit]
        
        num_subs[i] += int(len(subs))
    
        if num_subs[i] > 0:
            sat_dist[i] = len(subs[np.where(dist[whlimit]>(0.001*halo_rvir/2.))])/num_subs[i]
            sub_mass[i] = np.sum(subs['Mvir'])
            max_mass_sub[i] = np.max(subs['Mvir'])
            #sub_mass_far[i] = np.sum(subs[np.where(dist[whlimit]>(0.001*halo_rvir/2.))]['Mvir'])
        else:
            #avg_sat_dist[i] = 0.0#len(subs[np.where(dist[whlimit]<(halo_rvir/3000.))])/num_subs[i]
            #print(avg_sat_dist[i])
            #print(num_subs[i])
            sub_mass[i] = 0.0
            max_mass_sub[i] = 0.0

        i+=1
"""

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

calculated_spins = np.zeros((96))
calculated_shapes = np.zeros((96))
calculated_cs = np.zeros((96))
calculated_Jx = np.zeros((96))
calculated_Jy = np.zeros((96))
calculated_Jz = np.zeros((96))
calculated_mass = np.zeros((96))

halo_names = []
with open('halos_info_2.txt') as f:
    for l in f:
        this_halo, host_id, block, _ = l.split()
        halo_names.append(this_halo)

#glob.glob('/home/lom31/particle_stuff/particle_tables/*'):
    ##Get information about the particles##                          
j = 0
for f in halo_names:
    print(f)
    fname = '/home/lom31/rhap_particles/particle_tables/{}_all.particle_table'.format(f)
    particlevalues = ascii.read(fname, format = 'commented_header')
    ##Make sure chosen particles are within the virial radius##
    r = Column(np.sqrt(particlevalues['x']**2+particlevalues['y']**2+particlevalues['z']**2),name='r')

    particlevalues.add_column(r)

    whlimit = np.where(particlevalues['r']<=rvirs[j])
    Np = np.shape(whlimit)[1]
    print(Np)
    mvir = 1.3e8*Np
    calculated_mass[j] = mvir

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
    b_to_a, c_to_a, eig_A = calc_shape(rvirs[j]*1e-3, [0,0,0], pos, force_res) #all same units!
    calculated_shapes[j] = c_to_a
        
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
"""
og_vals = np.vstack((rvirs,mvirs,hostx,hosty,hostz,hostvx,hostvy,
    hostvz,host_shapes,host_spins,host_cs,hostJx,hostJy,hostJz,
    num_subs,scale,scale_lmm,scale_hm,sub_mass,max_mass_sub))

#print(calculated_spins,calculated_shapes,calculated_cs,calculated_Jx,calculated_Jy,calculated_Jz)
"""
no_sub_vals = np.vstack((calculated_spins,calculated_shapes,calculated_cs,calculated_Jx,calculated_Jy,calculated_Jz,))
"""
host_df = pd.DataFrame(data=og_vals.T, columns=["rvir","mvir","hostx","hosty","hostz","hostvx","hostvy",
        "hostvz","host_shapes","host_spins","host_cs","hostJx","hostJy","hostJz","num_subs","scale",
        "scale_lmm","scale_hm","sub_mass","max_mass_sub"])

host_og_vals = Table.from_pandas(host_df)
host_og_vals.write('host_og_vals_mw_new2.table',format='ascii.commented_header',overwrite=True)

"""
no_subs_df = pd.DataFrame(data=no_sub_vals.T, columns=["calc_spin","calc_shape","calc_cs","calc_Jx","calc_Jy","calc_Jz"])
host_no_subs_vals = Table.from_pandas(no_subs_df)
del no_subs_df
host_no_subs_vals.write('all_rhap_rock_mass.table',format='ascii.commented_header',overwrite=True)
"""
