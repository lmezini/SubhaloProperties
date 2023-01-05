import numpy as np 
from astropy.io import ascii
import argparse
from itertools import combinations
import time

rng = np.random.default_rng()

parser = argparse.ArgumentParser(description = 'pick which data set to use')
parser.add_argument('--dataset', dest = 'dataset')
#either mwm or rhapsody

#how many halos to drop
parser.add_argument('--numdrop', dest = 'numdrop',type=int)

args = parser.parse_args()

#Read halo info
if args.dataset == 'mwm':
    print('mwm')
    mass_frac_A_arr = np.load('mwm_mass_frac_A_arr.npy')
    #mass_frac_A_arr = np.delete(mass_frac_A_arr,[39,26,7,21,5,32],1)

elif args.dataset == 'rhapsody':
    print('rhapsody')
    mass_frac_A_arr = np.load('rhap_mass_frac_A_arr.npy')

meds_per_r = [] #list to store array of median values per radius threshold

startTime = time.time()

for x in mass_frac_A_arr:
#select set of mass fractions, x, for each r value
    
    #create array with index per halo
    ints = np.arange(0, len(x), 1, dtype=int)
    
    drop_indx = np.array(list(combinations(ints,args.numdrop)))
    numit = len(drop_indx)

    #create array with shape (numit,len(x)) which is just filled with
    #the array, x, numit times
    #a mask will be applied to this array to drop specific elements
    mass_frac_drop_arr = np.repeat(x,numit).reshape(len(x),numit).T

    #array with number of combinations of dropping elements and which ones to drop
    drop_arr = np.vstack((np.arange(0, numit, 1, dtype=int),drop_indx.T)).T
    mask = np.ones(np.shape(mass_frac_drop_arr),dtype=bool)
    mask[drop_arr.T[0],drop_arr.T[1:]]=False
    frac_set = mass_frac_drop_arr[mask].reshape(numit,len(x)-args.numdrop)

    #Calculate Median vs r
    meds = np.median(frac_set,axis=1)
    meds_per_r.append(meds)
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

np.save('{}_nit{}_ndrop_{}_med.npy'.format(args.dataset,numit,args.numdrop),np.array(meds_per_r))


#create file with arrays for many drop outs