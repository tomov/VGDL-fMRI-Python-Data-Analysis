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
from nibabel.affines import apply_affine
import pandas as pd
from scipy import stats
import seaborn as sns
import math


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
    
    dfBold = dfOrdered.drop(['game','instance','session','level'], axis=1) # don't need these columns anymore, got the Betas in the same order.
    B_ordered = dfBold.values # convert df to numpy array
    
    return clean_names, dfOrdered, B_ordered



'''
ISC functions
'''


def plot_correlations_onesub(sub, coords, isc_maps, brain_nii, mask_nii, threshold=0.2):

    '''
    Visualise the ISC correlations on the anatomical image.
    
    subject: participant number
    brain_nii: nii image (mask)
    coords: voxel coords from whole brain mask
    '''

    coords = tuple(coords) # needs to be a tuple

    # 1) create a volume from mask_nii
    isc_vol = np.zeros(mask_nii.shape) 


    # Map the ISC data for a subject into brain space
    isc_vol[coords] = isc_maps[sub,:]

    # 3) Create a nifti image from this with the affine from mask_nii
    isc_nifti = nib.Nifti1Image(isc_vol, mask_nii.affine, mask_nii.header)


    # plot the data as statmap
    f, ax = plt.subplots(1,1, figsize = (12, 5))
    plotting.plot_stat_map(
        stat_map_img=isc_nifti, 
        bg_img=brain_nii,
        threshold=threshold, 
        axes=ax,
        black_bg=True,
        vmax=1,
        cut_coords=[-30, -4, 5],
    )
    ax.set_title(f'ISC collapsed correlations'); 



def plot_correlations_collapsed(coords, collapsed_isc_corrs, brain_nii, mask_nii, threshold=0.2):

    '''
    Make a statistical map of the average/collapsed correlations.
    '''

    coords = tuple(coords) # needs to be a tuple

    # 1) create a volume from mask_nii
    isc_vol = np.zeros(mask_nii.shape) 

    # 2) Map the ISC data for a subject into brain space
    isc_vol[coords] = collapsed_isc_corrs

    # 3) Create a nifti image from this with the affine from mask_nii
    isc_nifti = nib.Nifti1Image(isc_vol, mask_nii.affine, mask_nii.header)


    # plot the data as statmap
    f, ax = plt.subplots(1,1, figsize = (12, 5))
    plotting.plot_stat_map(
        stat_map_img=isc_nifti, 
        bg_img=brain_nii,
        threshold=threshold, 
        axes=ax,
        black_bg=True,
        vmax=1,
        cut_coords=[-30, -4, 5],
    )
    ax.set_title(f'ISC map subject {sub+1}'); 


'''
@momchil Custom ISC functions
'''


def isc_onesub(data, s):

    ''' Intersubject correlation

    For each voxel, compute the pearson correlation between each subjects' response
    time series. Calculates the intersubject correlation using the leave-one-out approach. 


    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
            fMRI data for which to compute ISC
    s : subject/participant


    Returns
    -------
    iscs_r : [1, voxels ndarray]
        The pearson correlation coefficients for each voxel time series

    iscs_p : [1, voxels ndarray]
        The p values for each voxel time series
    
    '''
    # get number of voxels
    n_voxels = len(data[1])

    # for each subject should return a (1 x voxels matrix)
    iscs_r_sub = []
    iscs_p_sub = []

    betas_one_sub = data[:,:,s] # take the subject matrix 

    # remove this subjects' data from whole dataset and compute mean
    # TODO: can I compute the mean in this way or should transform first?
    mean_betas_rest = np.mean(np.delete(data, s, axis=2), axis=2) 

    # for each voxel (column) 
    for v in np.arange(n_voxels):

        # take the two columns that you want to correlate 
        voxel_v_sub = betas_one_sub[:, v] # all rows, one column at a time (voxel time series)
        voxel_v_rest = mean_betas_rest[:, v]

        r_vox_v, p_vox_v = array_correlation(x=voxel_v_sub, y=voxel_v_rest)

        # store the r and p value to subject arr
        iscs_r_sub.append(r_vox_v)
        iscs_p_sub.append(p_vox_v)

    return iscs_r_sub, iscs_p_sub


def array_correlation(x, y):

    ''' Column-wise pearson correlation between two arrays.

    scipy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html


    Parameters
    ----------
    x : one of the voxels from the given subject
            
    y : the corresponding voxel from the array of rest average 


    Returns
    -------
    r_vox_v : Pearson’s correlation coefficient

    p_vox_v : Two-tailed p-value.

    '''

    # correlation between left-out subject and mean of others
    r_vox_v, p_vox_v = stats.pearsonr(x, y) 

    return r_vox_v, p_vox_v


def get_significant_corrs(iscs_r_arr, iscs_pvalues_arr, alpha=0.05):

    ''' Check which correlations are significant


    Each r value corresponds to a p value. We only want to display the correlation coefficients that are significant,
    i.e. for which p < alpha. This function returns the significant correlations and fills in 0 otherwise. Can be used
    for the statistical map.

    Parameters
    ----------
    iscs_r_arr : an ndarray with the pearson correlations [subjects, voxels]

    iscs_pvalues_arr : an ndarray with the corresponding p values [subjects, voxels]
    
    Note that iscs_r_arr.shape == iscs_pvalues_arr


    Returns
    -------
    
    zeros_arr : an ndarray in the same shape as the input arrays with correlations where 
    p < alpha and zero otherwise.


    '''

    zeros_arr = np.zeros(iscs_r_arr.shape)

    # we do this for each row
    for row in range(len(iscs_pvalues_arr)):
        #print(row)
        
        # take the respective rows
        row_p = iscs_pvalues_arr[row,:]
        row_r = iscs_r_arr[row,:]
        
        #print(row_pvalues.shape, row_r.shape)
        
        # get the indices where the pvalues are significant
        sig_p_row_indices = np.where(row_p < alpha)
        
        #print(r_row[sig_p_row_indices])
        
        #print(zeros_arr[row,:].shape)
        
        # now use the zeros volume to use only the significant r values from 
        zeros_arr[row][sig_p_row_indices] = row_r[sig_p_row_indices] 

    return zeros_arr


def compute_avg_iscs(iscs, axis=0):

    '''Computes summary statistics for ISCs

    Computes the 'mean' across a set of ISCs in the following way:
        1) Take an ndarray with correlation coefficients [-1,1]
        2) Do a Fischer Z transform (arctanh) 
        3) Compute the mean
        4) Take the inverse Fisher transform (tanh)

    Parameters
    ----------

    iscs : [subjects, voxels] 
    list or ndarray, these are the isc values per subject

    Returns
    -------

    collapsed_isc_corrs : 1d array with the collapsed correlations for each voxel

    '''

    collapsed_isc_corrs = np.tanh(np.mean(np.arctanh(iscs), axis=axis))


    return collapsed_isc_corrs



def plot_statistical_map(coords, tstats, pvalues, brain_nii, mask_nii, theta=0.05, threshold=False, cut_coords=[42, 28, 26], vmax=None):

    '''
    Plots the t statistics onto the anatomical image. The pvalues are used to select a subset of
    t statistics (so for thresholding). 

    Parameters
    ----------
    
    coords: coordinates from the whole brain mask

    tstats: the t statistics from the r coefficients

    pvalues: the p values from the r coefficients
    
    brain_nii: the anatomical or structural image

    mask_nii: the whole brain mask

    theta: the cutoff point. Only display the t stats with p values < theta.

    vmax: Upper bound for plotting, passed to matplotlib.pyplot.imshow. By default vmax=None.
          (UB is specified by largest t value.)

    
    Returns
    ----------
        
    isc_nifti: the statistical map image (.nii) - use this for active visualisation 
    like

    view = plotting.view_img(stat_img, threshold=.2) >>> view 


    '''

    coords = tuple(coords) # needs to be a tuple in order to work

    # 1) create a volume from mask_nii
    isc_vol = np.zeros(mask_nii.shape) 


    if threshold==True:

        print(f'Display t statistics with a corresponding p < {theta}')

        # use theta to select the t statistics to plot
        # make arr to store the t stats with p < theta
        sig_arr = np.zeros(pvalues.shape)

        # get p indices smaller or equal to theta
        theta_indices = np.where(pvalues < theta)

        # get the t stats at these indices
        selected_tstats = tstats[theta_indices]

        # map the selected t statistics on the right places
        sig_arr[theta_indices] = selected_tstats

        # 2) Map the t statistics in voxel space
        isc_vol[coords] = sig_arr

        # 3) Create a nifti image from this with the affine from mask_nii
        isc_nifti = nib.Nifti1Image(isc_vol, mask_nii.affine, mask_nii.header)

        f, ax = plt.subplots(1,1, figsize = (12, 5), dpi=90)
        plotting.plot_stat_map(
            stat_map_img=isc_nifti, 
            bg_img=brain_nii,
            axes=ax,
            black_bg=True,
            cut_coords=cut_coords,
            vmax=vmax
        );

        ax.set_title(f't map with threshold p < {theta}'); 

    else:
        # plot all t statistics
        print('Display all t statistics')

        # 2) Map the t statistics in voxel space
        isc_vol[coords] = tstats

        # 3) Create a nifti image from this with the affine from mask_nii
        isc_nifti = nib.Nifti1Image(isc_vol, mask_nii.affine, mask_nii.header)

        f, ax = plt.subplots(1,1, figsize = (12, 5), dpi=90)
        plotting.plot_stat_map(
            stat_map_img=isc_nifti, 
            bg_img=brain_nii,
            axes=ax,
            black_bg=True,
            cut_coords=cut_coords,
            vmax=vmax
        );

        #ax.set_title(f't map with threshold p < {theta}'); 
    return isc_nifti


def prep_for_surface_plot(coords, tstats, pvalues, brain_nii, mask_nii, theta=0.05):

        
    '''
    Prepares the tstatistics for a surface plot. Default threshold is 0.05.

    
    Parameters
    ----------
        
    coords: coordinates from the whole brain mask

    tstats: the t statistics from the r coefficients

    pvalues: the p values from the r coefficients
    
    brain_nii: the anatomical or structural image

    mask_nii: the whole brain mask

    theta: the cutoff point. Only display the t stats with p values < theta.

    Returns
    -------

    isc_nifti: nifti image that should be used to make a 'texture'

    '''

    coords = tuple(coords) # needs to be a tuple
    
    # 1) create a volume from mask_nii
    isc_vol = np.zeros(mask_nii.shape) 
    
    print(f'Use t statistics with a corresponding p < {theta}')

    # use theta to select the t statistics to plot
    # make arr to store the t stats with p < theta
    sig_arr = np.zeros(pvalues.shape)

    # get p indices smaller or equal to theta
    theta_indices = np.where(pvalues < theta)

    # get the t stats at these indices
    selected_tstats = tstats[theta_indices]

    # map the selected t statistics on the right places
    sig_arr[theta_indices] = selected_tstats

    # 2) Map the t statistics in voxel space
    isc_vol[coords] = sig_arr

    # 3) Create a nifti image from this with the affine from mask_nii
    isc_nifti = nib.Nifti1Image(isc_vol, mask_nii.affine, mask_nii.header)
        
    return isc_nifti



def get_iscs_across_levels(levels_betas, T, num_subjects=8):

    '''
    
    Parameters
    ----------

    levels_betas: the betas from the levels. Shape is [54, voxels, subjects]
        
    num_subjects: the number of participants.

    T: the top voxel


    Returns
    -------
    
    '''

    print(f'Getting the intersubject corelations for voxel {T}')

    # === 1. Separate levels_betas for isc analyses ===

    betas_level_one = []
    betas_level_two = []
    betas_level_three = []
    betas_level_four = []
    betas_level_five = []
    betas_level_six = []
    betas_level_seven = []
    betas_level_eight = []
    betas_level_nine = []

    num_subjects = 8

    for s in range(num_subjects):
        #print(s)
        # take the array for that subject 
        levels_betas_sub = levels_betas[:,:,s]
        
        # level 1
        lvl_one_betas_sub = levels_betas_sub[0::9]
        betas_level_one.append(lvl_one_betas_sub)
        
        # level 2
        lvl_two_betas_sub = levels_betas_sub[1::9]
        betas_level_two.append(lvl_two_betas_sub)
        
        # level 3
        lvl_three_betas_sub = levels_betas_sub[2::9]
        betas_level_three.append(lvl_three_betas_sub)
        
        # level 4
        lvl_four_betas_sub = levels_betas_sub[3::9]
        betas_level_four.append(lvl_four_betas_sub)
        
        # level 5
        lvl_five_betas_sub = levels_betas_sub[4::9]
        betas_level_five.append(lvl_five_betas_sub)
        
        # level 6
        lvl_six_betas_sub = levels_betas_sub[5::9]
        betas_level_six.append(lvl_six_betas_sub)
        
        # level 7
        lvl_seven_betas_sub = levels_betas_sub[6::9]
        betas_level_seven.append(lvl_seven_betas_sub)
        
        # level 8
        lvl_eight_betas_sub = levels_betas_sub[7::9]
        betas_level_eight.append(lvl_eight_betas_sub)
        
        # level 9
        lvl_nine_betas_sub = levels_betas_sub[8::9]
        betas_level_nine.append(lvl_nine_betas_sub)
        
    # convert lists to np arrays
    betas_level_one = np.array(betas_level_one)
    betas_level_two = np.array(betas_level_two)
    betas_level_three = np.array(betas_level_three)
    betas_level_four = np.array(betas_level_four)
    betas_level_five = np.array(betas_level_five)
    betas_level_six = np.array(betas_level_six)
    betas_level_seven = np.array(betas_level_seven)
    betas_level_eight = np.array(betas_level_eight)
    betas_level_nine = np.array(betas_level_nine)

    # sanity check
    #print(betas_level_one.shape) # [subjects, games, voxels]

    # === 2. Swap axes to get data in right shape ===

    # do isc for each level
    # compute the isc correlations using the leave one out approach
    betas_level_one = np.swapaxes(betas_level_one, 0, 1) # need to get [TRs, voxels, subjects]
    betas_level_one = np.swapaxes(betas_level_one, 1, 2)

    betas_level_two = np.swapaxes(betas_level_two, 0, 1) # need to get [TRs, voxels, subjects]
    betas_level_two = np.swapaxes(betas_level_two, 1, 2)

    betas_level_three = np.swapaxes(betas_level_three, 0, 1) # need to get [TRs, voxels, subjects]
    betas_level_three = np.swapaxes(betas_level_three, 1, 2)

    betas_level_four = np.swapaxes(betas_level_four, 0, 1) # need to get [TRs, voxels, subjects]
    betas_level_four = np.swapaxes(betas_level_four, 1, 2)

    betas_level_five = np.swapaxes(betas_level_five, 0, 1) # need to get [TRs, voxels, subjects]
    betas_level_five = np.swapaxes(betas_level_five, 1, 2)

    betas_level_six = np.swapaxes(betas_level_six, 0, 1) # need to get [TRs, voxels, subjects]
    betas_level_six = np.swapaxes(betas_level_six, 1, 2)

    betas_level_seven = np.swapaxes(betas_level_seven, 0, 1) # need to get [TRs, voxels, subjects]
    betas_level_seven = np.swapaxes(betas_level_seven, 1, 2)

    betas_level_eight = np.swapaxes(betas_level_eight, 0, 1) # need to get [TRs, voxels, subjects]
    betas_level_eight = np.swapaxes(betas_level_eight, 1, 2)

    betas_level_nine = np.swapaxes(betas_level_nine, 0, 1) # need to get [TRs, voxels, subjects]
    betas_level_nine = np.swapaxes(betas_level_nine, 1, 2)


    # 3. === Pick a voxel (this voxel should be the most intense voxel from some ROI) ===

    # IMPORTANT: Momchils affine's matrix is the same so if the code says its voxel 5, 
    # I should get the 5th voxel in Python (which is actually #6 then bc of zero-indexing)
    
    topVox_betas_lvl_one = betas_level_one[:, T:T+1, :]

    topVox_betas_lvl_two = betas_level_two[:, T:T+1, :]

    topVox_betas_lvl_three = betas_level_three[:, T:T+1, :]

    topVox_betas_lvl_four = betas_level_four[:, T:T+1, :]

    topVox_betas_lvl_five = betas_level_five[:, T:T+1, :]

    topVox_betas_lvl_six = betas_level_six[:, T:T+1, :]

    topVox_betas_lvl_seven = betas_level_seven[:, T:T+1, :]

    topVox_betas_lvl_eight = betas_level_eight[:, T:T+1, :]

    topVox_betas_lvl_nine = betas_level_nine[:, T:T+1, :]

    # 4. === Get standard deviations and SE for error bars ===
    #  (std of betas from voxel) / sqrt(num_subjects)

    SEm_one = round((np.std(topVox_betas_lvl_one))/math.sqrt(num_subjects),2)

    SEm_two = round((np.std(topVox_betas_lvl_two))/math.sqrt(num_subjects),2)

    SEm_three = round((np.std(topVox_betas_lvl_three))/math.sqrt(num_subjects),2)

    SEm_four = round((np.std(topVox_betas_lvl_four))/math.sqrt(num_subjects),2)

    SEm_five = round((np.std(topVox_betas_lvl_five))/math.sqrt(num_subjects),2)

    SEm_six = round((np.std(topVox_betas_lvl_six))/math.sqrt(num_subjects),2)

    SEm_seven = round((np.std(topVox_betas_lvl_seven))/math.sqrt(num_subjects),2)

    SEm_eight = round((np.std(topVox_betas_lvl_eight))/math.sqrt(num_subjects),2)
    
    SEm_nine = round((np.std(topVox_betas_lvl_nine))/math.sqrt(num_subjects),2)
    
    # put them into a list for plotting
    errors = [SEm_one, SEm_two, SEm_three, SEm_four, SEm_five, SEm_six, SEm_seven, SEm_eight, SEm_nine]


    # === 5. Do the ISC for the chosen voxel ===
    # We obtain a scalar value for each level, because we collapse the vector of r coefficients (using Fischer Z first)
    isc_r_topVox_one = float(isc(topVox_betas_lvl_one, pairwise=False, tolerate_nans=True, summary_statistic='mean'))

    isc_r_topVox_two = float(isc(topVox_betas_lvl_two, pairwise=False, tolerate_nans=True, summary_statistic='mean'))

    isc_r_topVox_three = float(isc(topVox_betas_lvl_three, pairwise=False, tolerate_nans=True, summary_statistic='mean'))

    isc_r_topVox_four = float(isc(topVox_betas_lvl_four, pairwise=False, tolerate_nans=True, summary_statistic='mean'))

    isc_r_topVox_five = float(isc(topVox_betas_lvl_five, pairwise=False, tolerate_nans=True, summary_statistic='mean'))

    isc_r_topVox_six = float(isc(topVox_betas_lvl_six, pairwise=False, tolerate_nans=True, summary_statistic='mean'))

    isc_r_topVox_seven = float(isc(topVox_betas_lvl_seven, pairwise=False, tolerate_nans=True, summary_statistic='mean'))

    isc_r_topVox_eight = float(isc(topVox_betas_lvl_eight, pairwise=False, tolerate_nans=True, summary_statistic='mean'))

    isc_r_topVox_nine = float(isc(topVox_betas_lvl_nine, pairwise=False, tolerate_nans=True, summary_statistic='mean'))


    # === 6. Collect the correlation coefficients ===
    isc_r_values_levels = [isc_r_topVox_one, isc_r_topVox_two, isc_r_topVox_three, isc_r_topVox_four, isc_r_topVox_five,
                            isc_r_topVox_six, isc_r_topVox_seven, isc_r_topVox_eight, isc_r_topVox_nine]
    isc_r_values_levels = [round(i, 2) for i in isc_r_values_levels]                 

    return isc_r_values_levels, errors



def plot_r_values_levels(top_voxel, isc_r_values_levels, errors, levels=list(range(1,10))):

    '''
    Parameters
    ----------
        
    top_voxel: the most intense voxel for which you did the isc.

    isc_r_values_levels: a 1d array of len 9, one r per level.

    Returns
    -------
    
    a bar plot.
    '''

    f, ax = plt.subplots(1,1, figsize = (12, 5), dpi=70)
    f.suptitle(f'R values for voxel {top_voxel} across levels')
    ax.errorbar(levels, isc_r_values_levels, yerr=errors, fmt='o', color='Black', elinewidth=2, capthick=2, errorevery=1, alpha=1, ms=3, capsize=5, axes=ax)
    ax.bar(levels, isc_r_values_levels)
    ax.set_xticks(levels)
    ax.set_ylim(-1,1)
    ax.set_ylabel('r');


def get_vox_from_coords(coords_mat, native_coords):

    ''' 
    Returns the voxel number that belongs to a given coordinate.
    

    Parameters
    ----------
        
    coords_mat: whole brain mask coordinates in native voxel space.

    native_coords: the native coordinates that correspond to a certain voxel.

    Returns
    -------
    
    The voxel number

    '''

    # iterate through all coords from whole brain mask
    for v in range(len(coords_mat[0])):
        
        # x,y,z arrays are
        x_arr = coords_mat[0, :]
        y_arr = coords_mat[1, :]
        z_arr = coords_mat[2, :]
        
        # take coordinates for voxel v
        x = x_arr[v]
        y = y_arr[v]
        z = z_arr[v]

        if x == int(native_coords[0]) and y == int(native_coords[1]) and z == int(native_coords[2]):
            top_V = v
            print(f'The coordinates correspond to voxel: {top_V}.')

    return top_V




def find_top_voxel_in_roi(roi_mask, regions_thresholded_img, coords_mat, mask_nii, mean_nii):
    

    # convert to nifti object to plot
    roi_mask_nifti = nib.Nifti1Image(roi_mask, regions_thresholded_img.affine, regions_thresholded_img.header) 

    # The mask_nii mask overlayed on mean_nii
    plotting.plot_roi(roi_img=roi_mask_nifti, bg_img=mean_nii, black_bg=False, colorbar=True, title='The ROI (t statistics)');

    # check largest t values
    max_t_values = np.sort(roi_mask.flatten())[::-1][:20] # get top 20 t stats
    max_tstat = max_t_values[0] # take the highest t statistic

    #print(f'Largest values \n {max_t_values} \n')
    print(f'Largest t statistic: {round(max_tstat,2)}\n')
     
    top_vox_coords = np.where(roi_mask==max_tstat) # get x,y,z coords for voxel with the highest t statistic
    print(f'The coordinates of the voxel with the highest t value (native space): {top_vox_coords}')

    # get top vox corresponding to coordinates (return this)
    top_vox_in_roi = get_vox_from_coords(coords_mat, top_vox_coords) # corresponding voxel

    # === Sanity check: plot the coordinate back on brain ===

    # make all items in mask 1 execpt the top voxel, make that one 10
    nonzero_indices = np.nonzero(roi_mask)
    roi_boolean = np.zeros(roi_mask.shape)
    roi_boolean[nonzero_indices] = 1 # convert all nonzero indices to a one
    roi_boolean[top_vox_coords] = 10 # unfortunately can't see 1 voxel :(

    # put in simple lsit
    top_vox_coords = [int(top_vox_coords[0]), int(top_vox_coords[1]), int(top_vox_coords[2])]

    # translate back to mni space to check where the top voxel is 
    mni_coords = apply_affine(aff=mask_nii.affine, pts=top_vox_coords) # from cor2mni
    # reshape mni coordinates to add the markers
    mni_coords_rs = mni_coords.reshape(1,3)

    # convert to nifti object to plot
    roi_sanitycheck_nifti = nib.Nifti1Image(roi_boolean, regions_thresholded_img.affine, regions_thresholded_img.header) 
    roi_map = plotting.plot_stat_map(stat_map_img=roi_sanitycheck_nifti, bg_img=mean_nii, 
                            black_bg=False, cut_coords=mni_coords, title=f'Mapping the top voxel (v={top_vox_in_roi}) back onto brain');

    # Add marker
    roi_map.add_markers(
        marker_coords=mni_coords_rs, 
        marker_color='lawngreen',
        marker_size=40,
    )

    return top_vox_in_roi, mni_coords


def map_vox_on_brain(mean_nii, mask_nii, mni_coords, voxel=None):
	'''
	Maps a given voxel on the anatomical brain image using the mni coords.
	'''


	mni_coords = np.array(mni_coords)

	zeros_vol = np.zeros(mask_nii.shape)

	zeros_nifti = nib.Nifti1Image(zeros_vol, mask_nii.affine, mask_nii.header)

	vox_map = plotting.plot_stat_map(stat_map_img=zeros_nifti, bg_img=mean_nii, 
	                            black_bg=False, cut_coords=mni_coords, 
	                            title=f'');

	# reshape mni coordinates to add the markers 
	mni_coords_rs = mni_coords.reshape(1,3)

	# Add marker
	vox_map.add_markers(
	    marker_coords=mni_coords_rs, 
	    marker_color='lawngreen',
	    marker_size=40,
	)




def correlate_one_sub(R, encoding_voxels, updating_voxels):
    '''
    Correlates the time series between two sets of voxels.
    
    Parameters
    ----------
    
    R: the residuals for one subject
    updating_voxels: the indices of the updating voxels
    encoding_voxels: the indices of the encoding voxels
    
    Returns
    -------
    
    corr_matrix: asymmetric correlation matrix for that subject
    
    '''
    
    # select the voxels from the residuals
    updating_time_series = R[:, updating_voxels]
    encoding_time_series = R[:, encoding_voxels]

    N = len(encoding_time_series[1])
    corr_matrix = np.zeros((N,N))


    for i in range(N): # iterate through voxels (see img above)

        corr_col = []
        # pick the ith column from encoding 
        encoding_vec = encoding_time_series[:, i]

        # correlate it with each col (j) from updating time series
        for j in range(N):

            updating_vec = updating_time_series[:, j]

            # correlate column i with column j
            corr, _ = stats.pearsonr(encoding_vec, updating_vec)

            corr_col.append(round(corr,2)) 

        corr_matrix[:, i] = corr_col # insert correlations into correlation matrix
#         print(f'--- Theory encoding col {i} ---')
#         print(corr_matrix)

    return corr_matrix



def get_significant_stats(pvalues, tstats, alpha=0.01):

    '''
    Given an nd array of tstats and pvalues, return only the t statistics < alpha (significance level) 
    '''
    
    # make empty matrix for the significant t stats
    sig_M = np.zeros(pvalues.shape)

    # take indices of p values < alpha
    indices = np.where(pvalues < alpha)

    # get these t stats
    selected_tstats = tstats[indices]

    # map them in the right place
    sig_M[indices] = selected_tstats
    
    return sig_M











