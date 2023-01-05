import numpy as np 
import numexpr as ne
import scipy
from scipy.optimize import minimize_scalar
import glob
from astropy.io import ascii
from astropy.table import Column
from calc_shape import calc_shape

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
hostJx = np.zeros((45))
hostJy = np.zeros((45))
hostJz = np.zeros((45))
for file in sorted(glob.glob('/Users/catfielder/Documents/Research_Halos/Halos_Recalculated/Halo*/*')):
    if file.endswith('.list_final'):
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
    
calculated_spins = np.zeros((45))
calculated_shapes = np.zeros((45))
calculated_cs = np.zeros((45))
calculated_Jx = np.zeros((45))
calculated_Jy = np.zeros((45))
calculated_Jz = np.zeros((45))
j = 0
for file in sorted(glob.glob('/Users/catfielder/Documents/Research_Halos/ParticleDetail/Halo*/*')):
    ##Get information about the particles##                                        
    if file.endswith('_particle_table'):
        particlevalues = ascii.read(file, format = 'commented_header')
        ##Make sure chosen particles are within the virial radius##
        r = Column(np.sqrt(particlevalues['x']**2+particlevalues['y']**2+particlevalues['z']**2),name='r')
        particlevalues.add_column(r)
        whlimit = np.where(particlevalues['r']<rvirs[j])
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
        b_to_a, c_to_a, eig_A = calc_shape(rvirs[j]*1e-3, [0,0,0], pos, force_res) #all same units!
        calculated_shapes[j] = c_to_a
        
        ##Calculate angular momentum##
        ang_mom = calc_ang_momentum(pos,vel)
        calculated_Jx[j] = ang_mom[0]
        calculated_Jy[j] = ang_mom[1]
        calculated_Jz[j] = ang_mom[2]                
        
        ##Calculate Bullock Spin##
        spin_bullock = calc_spin_bullock(ang_mom,mvirs[j],rvirs[j])
        calculated_spins[j] = spin_bullock
        
        ##Calculate concentration##
        x = np.sqrt((pos*pos).sum(axis=-1))/(rvirs[j]*1e-3)
        x_lim = 1.0
        c = calc_concentration(x,x_lim)
        calculated_cs[j] = c
        
        j+=1

##Save those calculated array##        
np.save('calculated_spins_nosubs.npy', calculated_spins)
np.save('calculated_shapes_nosubs.npy', calculated_shapes)
np.save('calculated_cs_nosubs.npy', calculated_cs)
np.save('calculated_Jx_nosubs.npy', calculated_Jx)
np.save('calculated_Jy_nosubs.npy', calculated_Jy)
np.save('calculated_Jz.npy', calculated_Jz)
np.save('host_spins_nosubs.npy', host_spins)
np.save('host_shapes_nosubs.npy', host_shapes)
np.save('host_cs_nosubs.npy', host_cs)
np.save('host_Jx_nosubs.npy', hostJx)
np.save('host_Jy_nosubs.npy', hostJy)
np.save('host_Jz_nosubs.npy', hostJz)