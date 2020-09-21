# fMRI Data Analysis in Python

This repository contains Python code and Jupyter notebooks used to analyse fMRI data. 

### What's in the folder: Analyses_Daphne? 🤔

This folder contains the deliverables from Daphne, resulting from the summer research assistantship with Momchil from Jun-Aug 2020. 
You will find code on preprocessing data (🟠), how to find regions of interest (🔴), intersubject correlation (🟢) and functional connectivity analyses (🔵).

### Python scripts

- `utils.py` includes all the functions used for preprocessing the data, performing analyses, plotting and more. 
All the notebooks in this folder use the functions from `utils.py` to keep them nice and clean.

### Notebooks

(almost) each of the notebooks include a summary of what is done and are well documented. 
Some also include links that point to resources where you can learn more about the respective analysis.

#### Notebooks where I preprocess data or just stored for reference

- `Preprocess_smooth_betas.ipynb` is where the `.mat` files are preprocessed and the BOLD data is reordered. 
We looked at beta series for boxcars, blocks, levels, and games. 🟠

- `Startingpoint.ipynb` is a probably not very useful but kept it here for potential debugging. 

#### Analysis notebooks

- `ISC_nosmooth.ipynb` Doing an intersubject correlation (ISC) with the nonsmooth betas (row wise). 🟢

- `My_ISC_vs_brainiak_ISC.ipynb` Here we compare the results obtained from our own implemented intersubject correlation with 
the results from brainiaks intersubject correlation (`from brainiak.isc import isc`) 🟢

- `Smooth_betas_statisticalMaps.ipynb` This is were the statistical maps for the slides are generated. We perform one-sample 
t-tests and use the functions from `utils.py` to plot the t statistics on the anatomical brain image. 🟢

- In `From_vox_to_corrs.ipynb` we look at how the intersubject correlations evolve across levels. 🟢

- `Find_ROIs.ipynb` This notebook is used to find the regions of interest (ROIs) based on the statistical maps and extract 
the most intense voxel for each of the found ROIs. 🔴
 
- In `Running_on_cluster.ipynb` we get the data for all the subjects later used in `Functional_connectivity_within_subjects.ipynb`. 🔵

- `Functional_connectivity_within_subjects.ipynb` does a standard FC analysis, correlating across different brain areas 
(i.e. the ROIs found by Momchil & the ones from me) within a subjects brain. It also includes a bit on how to perform granger causality on voxel pairs. 🔵

- `Functional_connectivity_between_subjects.ipynb` as the name suggests, here we do the look at the functional connectivity
across subjects. 🔵

