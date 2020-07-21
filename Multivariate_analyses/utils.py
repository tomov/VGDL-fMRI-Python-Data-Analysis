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
Functions for the fMRI blocks data
'''


def cleanup_block_names(s):

    for r in (('vgfmri3_', ''), ('*bf(1)', '')):
        s = s.replace(*r)
        
    return s


def get_in_shape_blocks(B_s, names_s):
    
    '''
    Massages data into right shape for the ISC: [TRs, voxels, subjects] - bunch of stacked matrices
    
    IN
    
    B: the bold data for subject s
    names: the order of the blocks for subject s
    
    OUT
    
    dfOrdered: the ordered df, just to sanity check the reordering
    B_ordered: the ordered B array [blocks, voxels]
    '''
    
    #print(B_s.shape)
    
    # cleanup the block names first, remove stuff
    
    block_names = []

    for name in names_s:
        stripped_name = cleanup_block_names(name)
        block_names.append(stripped_name)
    
    # split the strings by white space to separate the sessions and games
    splitted = [words for segments in block_names for words in segments.split(' ')]
    sessions = splitted[0::2] # append to separate lists
    blocks = splitted[1::2] 
    
    #print(block_names)
    # read in B as pandas df
    df = pd.DataFrame(B_s)
    df.insert(0, 'block', blocks) # insert block names as first col
    df.insert(1, 'session', sessions) # insert block names as first col
    
    # First look at the name (alphabetic order) then look at the session number
    dfOrdered = df.sort_values(by=['block','session'], ascending=True) # return for sanity checks
    dfBold = dfOrdered.drop(['block', 'session'], axis=1) # don't need these columns anymore, got the Betas in the same order.
    
    B_ordered = dfBold.values # convert df to numpy array
    
    return dfOrdered, B_ordered



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



def get_in_shape_levels(B_s, names_s, num_runs=6):
    
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



'''
ISC functions
'''

def plot_sub_isc_statmap(sub, data, brain_nii, mask_data, threshold=0.2):

    '''
    Computes and plots an ISC statmap for a given subject. 

    subject: participant number
    data: BOLD data array in the shape [n_TRs, n_voxels, n_subjects]
    brain_nii: nii image (mask)
    mask_data: the whole brain mask (boolean arr)
    '''

    isc_maps = isc(data, pairwise=False) # The output of ISC is a voxel by 
                           # participant matrix (showing the result of each individual with the group).

    #print(isc_maps.shape)

    # use the mask to find all the coordinates that represent the brain
    coords_sub = np.where(mask_data[sub] == 1) 

    # Make zeros 3D cube
    isc_vol = np.zeros(brain_nii.shape)

    # Map the ISC data for a subject into brain space
    isc_vol[coords_sub] = isc_maps[sub,:]

    # make a nii image of the isc map 
    isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)

    # plot the data as statmap
    f, ax = plt.subplots(1,1, figsize = (12, 5), dpi=100)
    plotting.plot_stat_map(
        isc_nifti, 
        threshold=threshold, 
        axes=ax
    )
    ax.set_title(f'ISC map for subject {sub+1}');




def plot_statmap_onesub(sub, isc_vol, isc_maps, coords, brain_nii, threshold=0.2, use_brainii=True):

    '''
    Plot the statistical map for a given subject.
    
    subject: participant number
    brain_nii: nii image (mask)
    coords: voxel coords from whole brain mask
    '''

    # Map the ISC data for a subject into brain space
    isc_vol[coords] = isc_maps[sub,:]

    # make a nii image of the isc map 
    isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)

    if use_brainii==True:

        # plot the data as statmap
        f, ax = plt.subplots(1,1, figsize = (12, 5))
        plotting.plot_stat_map(
            stat_map_img=isc_nifti, 
            bg_img=brain_nii,
            threshold=threshold, 
            axes=ax,
            black_bg=True,
            vmax=1,
        )
        ax.set_title(f'ISC map for subject {sub+1}'); 

    else:

        # plot the correlations on the MNI standards
        f, ax = plt.subplots(1,1, figsize = (12, 5))
        plotting.plot_stat_map(
            stat_map_img=isc_nifti,
            threshold=threshold, 
            axes=ax,
            black_bg=True,
            vmax=1,
        )
        ax.set_title(f'ISC map for subject {sub+1}'); 



def plot_statmap_avg(isc_vol, coords, collapsed_isc_corrs, brain_nii, use_brainii=True, threshold=0.2):

    '''
    Make a statistical map of the average/collapsed correlations.
    '''

    # Map the ISC data for a subject into brain space
    isc_vol[coords] =  collapsed_isc_corrs

    # make a nii image of the isc map 
    isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)

    if use_brainii==True:

        # plot the data as statmap
        f, ax = plt.subplots(1,1, figsize = (12, 5))
        plotting.plot_stat_map(
            stat_map_img=isc_nifti, 
            bg_img=brain_nii,
            threshold=threshold, 
            axes=ax,
            black_bg=True,
            vmax=1,
        )
        ax.set_title(f'ISC collapsed correlations'); 

    else:

        # plot the correlations on the MNI standards
        f, ax = plt.subplots(1,1, figsize = (12, 5))
        plotting.plot_stat_map(
            stat_map_img=isc_nifti,
            threshold=threshold, 
            axes=ax,
            black_bg=True,
            vmax=1,
        )
        ax.set_title(f'ISC collapsed correlations'); 


def surface_plot_avg(isc_vol, coords, brain_nii, collapsed_isc_corrs, view='medial', threshold=0.2):
    '''
    make a surface plot
    '''

    # get a surface
    fsaverage = datasets.fetch_surf_fsaverage5()

    # Map the ISC data for the first participant into brain space
    isc_vol[coords] = collapsed_isc_corrs
    # make a nii image of the isc map 
    isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)
    # make "texture" 
    texture = surface.vol_to_surf(isc_nifti, fsaverage.pial_left) 

    # plot 
    title_text = (f'Avg ISC map, {view} view')
    surf_map = plotting.plot_surf_stat_map(
        fsaverage.infl_left, texture, 
        hemi='left', view=view, 
        title= title_text, 
        threshold=threshold, cmap='RdYlBu_r', 
        colorbar=True,
        bg_map=fsaverage.sulc_left,
        vmax=1)













