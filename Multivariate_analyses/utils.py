'''
contains functions to help with analyses
'''

import h5py
import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os 
import glob
import time
from copy import deepcopy
import numpy as np
import pandas as pd 

from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import nibabel as nib

from brainiak import image, io
from brainiak.isc import isc, isfc, permutation_isc
from brainiak.isc import compute_summary_statistic
import matplotlib.pyplot as plt
import pandas as pd


def decode_variable(file, item):

    '''
    Converts matlab cell array in the form "<HDF5 object reference>" to list of strings.

    IN

    file: the path + filename 
    item: the variable in the dataset that needs to be decoded

    RETURNS

    readable_data: np array of strings
    '''

    # Open file                                                                                    
    myfile = h5py.File(file,'r')
    variable = myfile[item] # get the names variable

    readable_data = [] # store the ne


    for var in variable: # encode and decode the objects, 18 per subject
        for v in var: # Read the references  

            #print(v)
            ds = myfile[v]
            #print(ds)
            data = ds[:]

            # store the decoded data
            word = []
            
            for i in data:
                letter = str(chr(i))  # the chr() function returns the character that represents the specified unicode.
                word.append(letter)
            word = ''.join(word) # join list of strings
            
            readable_data.append(word)
            
    return np.array(readable_data)

'''

Functions for the fMRI level data

'''

def cleanup_level_names(s):
    '''
    Removes parts of the string to make it more orderly and easier to rearrange.
    '''
    for r in (('vgfmri3_', ''), ('*bf(1)', ''), ('_instance_', ' ')):
        s = s.replace(*r)
        
    return s



def get_in_shape_levels(B_s, names_s):
    
    '''
    Massages data into right shape to perform ISC: [TRs, voxels, subjects] - bunch of stacked matrices
    
    IN
    
    B: the bold data for subject s
    names: the order of the levels for subject s
    
    OUT
    
    dfOrdered: the ordered df, just to sanity check the reordering
    B_ordered: the ordered B array [levels, voxels]
    '''
    
    # --- clean up names --- # 
    clean_names = []

    for name in names_s:
        stripped_name = cleanup_level_names(name)
        clean_names.append(stripped_name)
        
    # --- split string at each white space --- #
    splitted = [words for segments in clean_names for words in segments.split(' ')] 
    
    num_runs = 6 # runs == sessions
    
    # store sessions, games and levels in separate lists 
    sessions = splitted[0::3] # get item 0,3,6,9 etc.
    games = splitted[1::3] # get item 1,4,7,10 etc.
    instances = splitted[2::3] # get item 2,5,8 etc.
    levels = num_runs * list(range(1,10))

    # --- read in B as pandas df --- #
    df = pd.DataFrame(B_s)
    df.insert(0, 'game', games) # insert block names as first col
    df.insert(1, 'session', sessions) 
    df.insert(2, 'instance', instances) # insert levels as second col
    
    # sort by game, session then instance
    dfOrdered = df.sort_values(by=['game','session','instance'], ascending=True) # return to check if order is the same
    dfOrdered.insert(3, 'level', levels) 
    
    dfBold = dfOrdered.drop(['game','instance','session',], axis=1) # don't need these columns anymore, got the Betas in the same order.
    B_ordered = dfBold.values # convert df to numpy array
    
    return clean_names, dfOrdered, B_ordered











