#!/usr/bin/env python
# coding: utf-8

""" Create the training, validation and testing datasets (raster files) for machile learnig algorithms
using stratified random sampling.

NOTICE: run from 'rsml' environment (Python 3.8.13; GDAL 3.5.2 & matplotlib 3.6.0 from conda-forge)

author: Eduardo Jimenez <eduardojh@arizona.edu>
Changelog:
    May 18, 2023: Stratified random sampling to create training, validation, and testing datasets.
"""

import sys
import random
import numpy as np
import pandas as pd

NA_CLASS = 0  # In raster 0=NoData, other values are land cover classes

if len(sys.argv) == 3:
    # Check if arguments were passed from terminal
    args = sys.argv[1:]
    sys.path.insert(0, args[0])
    cwd = args[1]
    print(f"  Using RS_LIB={args[0]}")
    print(f"  Using CWD={args[1]}")
else:
    import os
    import platform
    system = platform.system()
    if system == 'Windows':
        # On Windows 10
        sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib/')
        cwd = 'D:/Desktop/CALAKMUL/ROI1/'
    elif system == 'Linux' and os.path.isdir('/vipdata/2023/CALAKMUL/ROI1/'):
        # On Ubuntu Workstation
        sys.path.insert(0, '/vipdata/2023/land_cover_analysis/lib/')
        cwd = '/vipdata/2023/CALAKMUL/ROI1/'
    elif system == 'Linux' and os.path.isdir('/VIP/engr-didan02s/DATA/EDUARDO/ML/'):
        # On Alma Linux Server
        sys.path.insert(0, '/home/eduardojh/Documents/land_cover_analysis/lib/')
        cwd = '/VIP/engr-didan02s/DATA/EDUARDO/ML/'
    else:
        print('  System not yet configured!')

import rsmodule as rs

#### 1. Set up file names
# fn_landcover = cwd + 'data/inegi_2018/usv250s7cw_ROI1_LC_KEY.tif'
fn_landcover = cwd + 'data/inegi_2018/usv250s7cw_ROI1_LC_KEY_grp.tif'  # use groups
fn_keys = cwd + 'data/inegi_2018/land_cover_groups.csv'
fn_sample = cwd + 'sampling_validation/dataset_sample_sizes.csv'
fn_stats = cwd + 'sampling_validation/usv250s7cw_ROI1_statistics.csv'
fn_lc_plot = cwd + 'sampling_validation/usv250s7cw_ROI1_percent_plot.png'
fn_testing_mask  = cwd + 'sampling_validation/usv250s7cw_ROI1_testing_mask.tif'  # create testing mask, training is the complement
fn_validation_mask = cwd + 'sampling_validation/usv250s7cw_ROI1_validation_mask.tif'

# Create a list of land cover keys and its area covered percentage
lc_frq = rs.land_cover_freq(fn_landcover, fn_keys, verbose=False)
print(f'  --Land cover freqencies: {lc_frq}')

lc_lbl = list(lc_frq.keys())
freqs = [lc_frq[x] for x in lc_lbl]  # pixel count
percentages = (freqs/sum(freqs))*100  # percent, based on pixel count

# Plot land cover percentage horizontal bar
print('  --Plotting land cover percentages...')
rs.plot_land_cover_hbar(lc_lbl, percentages, fn_lc_plot,
    title='INEGI Land Cover Classes in Calakmul Biosphere Reserve',
    xlabel='Percentage (based on pixel count)',
    ylabel='Land Cover (Grouped)',  # remove if not grouped
    xlims=(0,100))
# # training-validation-testing proportion is 70-10-20%
# testing_fraction = 0.2
# validation_fraction = 0.1

# #### Sample size == testing dataset
# # Use a dataframe to calculate sample size
# df = pd.DataFrame({'Key': lc_lbl, 'PixelCount': freqs, 'Percent': percentages})
# df['TestingPix'] = (df['PixelCount']*testing_fraction).astype(int)
# df['TestingPer'] = (df['TestingPix'] / df['PixelCount'])*100
# df['V8nPix'] = (df['PixelCount']*validation_fraction).astype(int)  # pixel count in validation dataset
# df['V8nPer'] = (df['V8nPix'] / df['PixelCount'])*100  # pixel percent of validation dataset

# #### 2. Create the testing mask
# raster_arr, nd, meta, gt, proj, epsg = rs.open_raster(fn_landcover)
# print(f'  --Opening raster : {fn_landcover}')
# print(f'  ----Metadata     : {meta}')
# print(f'  ----NoData       : {nd}')
# print(f'  ----Columns      : {raster_arr.shape[1]}')
# print(f'  ----Rows         : {raster_arr.shape[0]}')
# print(f'  ----Geotransform : {gt}')
# print(f'  ----Projection   : {proj}')
# print(f'  ----EPSG         : {epsg}')
# print(f'  ----Type         : {raster_arr.dtype}')

# rows, cols = raster_arr.shape
# print(f"  --Total pixels={rows*cols}, Values={sum(df['PixelCount'])}, NoData/Missing={rows*cols - sum(df['PixelCount'])}")

# raster_arr = raster_arr.astype(int)
# print(f"  --Before filling NoData: {np.unique(raster_arr)}")
# raster_arr = raster_arr.filled(0)  # replace masked constant "--" with zeros
# print(f"  --After filling NoData: {np.unique(raster_arr)}")
# # print(f'  --Check new array type: {raster_arr.dtype}')

# window_size = 7
# sample = {}  # to save the sample
# validation_sample = {}

# # Create a mask of the sampled regions
# sample_mask = np.zeros(raster_arr.shape, dtype=raster_arr.dtype)
# sample_mask_val = np.zeros(raster_arr.shape, dtype=raster_arr.dtype)

# # A window will be used for sampling, this array will hold the sample
# window_sample = np.zeros((window_size,window_size), dtype=int)

# nrows, ncols = raster_arr.shape

# max_trials = int(2e5)  # max of attempts to fill the sample size
# print(f'  --Max trials: {max_trials}')

# trials = 0  # attempts to complete the sample
# completed_test = {}  # classes which sample is complete
# completed_validation = {}

# for sample_key in list(df['Key']):
#     completed_test[sample_key] = False
#     completed_validation[sample_key] = False
# completed_samples = sum(list(completed_test.values())) + sum(list(completed_validation.values()))  # Values are all True if completed
# total_classes = len(completed_test.keys()) + len(completed_validation.keys())
# # print(completed_test)

# # Save the sampled points for testing and validation data sets
# sampled = []

# while (trials < max_trials and completed_samples < total_classes):
#     show_progress = (trials%10000 == 0)  # Step to show progress
#     if show_progress:
#         print(f'  --Trial {1 if trials == 0 else trials:>8} of {max_trials:>8} ', end='')

#     # 1) Generate a random point (row_sample, col_sample) to sample the array
#     #    Coordinates relative to array positions [0:nrows, 0:ncols]
#     #    Subtract half the window_size to avoid sampling too close to the edges, use window_size step to avoid overlapping
#     col_sample = random.randrange(0 + window_size//2, ncols - window_size//2, window_size)
#     row_sample = random.randrange(0 + window_size//2, nrows - window_size//2, window_size)

#     # Save the points previously sampled to avoid repeating and oversampling
#     point = (row_sample, col_sample)
#     if point in sampled:
#         trials +=1
#         continue
#     else:
#         sampled.append(point)

#     # 2) Generate a sample window around the random point, here create the boundaries,
#     #    these rows and columns will be used to slice the sample
#     win_col_ini = col_sample - window_size//2
#     win_col_end = col_sample + window_size//2 + 1  # add 1 to slice correctly
#     win_row_ini = row_sample - window_size//2
#     win_row_end = row_sample + window_size//2 + 1

#     assert win_col_ini < win_col_end, f"Incorrect slice indices on x-axis: {win_col_ini} < {win_col_end}"
#     assert win_row_ini < win_row_end, f"Incorrect slice indices on y-axis: {win_row_ini} < {win_row_end}"

#     # 3) Check if sample window is out of range, if so trim the window to the array's edges accordingly
#     #    This may not be necessary if half the window size is subtracted, but still
#     if win_col_ini < 0:
#         # print(f'    --Adjusting win_col_ini: {win_col_ini} to 0')
#         win_col_ini = 0
#     if win_col_end > ncols:
#         # print(f'    --Adjusting win_col_end: {win_col_end} to {ncols}')
#         win_col_end = ncols
#     if win_row_ini < 0:
#         # print(f'    --Adjusting win_row_ini: {win_row_ini} to 0')
#         win_row_ini = 0
#     if  win_row_end > nrows:
#         # print(f'    --Adjusting win_row_end: {win_row_end} to {nrows}')
#         win_row_end = nrows

#     # 4) Check and adjust the shapes of the arrays to slice and insert properly, only final row/column can be adjusted
#     window_sample = raster_arr[win_row_ini:win_row_end,win_col_ini:win_col_end]
#     # print(window_sample)
    
#     # 5) Get the unique values in sample (sample_keys) and its count (sample_freq)
#     sample_keys, sample_freq = np.unique(window_sample, return_counts=True)
#     classes_to_remove = []  # Avoid adding zeros or completed classes to the mask

#     # 6) Iterate over each class sample and add its respective pixel count to the sample
#     for sample_class, class_count in zip(sample_keys, sample_freq):
#         if sample_class == NA_CLASS:
#             # Sample is mixed with zeros, tag it to remove it and go to next sample_class
#             classes_to_remove.append(sample_class)
#             continue

#         # If sample is complete for testing and validation, pass
#         if completed_test.get(sample_class, False) and completed_validation.get(sample_class, False):
#             classes_to_remove.append(sample_class)  # do not add completed classes to the sample
#             continue

#         # If sample is complete for testing but not for validation, add to the later
#         if completed_test.get(sample_class, False) and not completed_validation.get(sample_class, False):
#             if validation_sample.get(sample_class) is None:
#                 validation_sample[sample_class] = class_count
#             else:
#                 sample_size = df[df['Key'] == sample_class]['V8nPix'].item()
                
#                 # If sample isn't completed, add the sampled window
#                 if validation_sample[sample_class] < sample_size:
#                     validation_sample[sample_class] += class_count
#                     # Check if last addition completed the sample
#                     if validation_sample[sample_class] >= sample_size:
#                         completed_validation[sample_class] = True  # this class' sample is now complete
#                 else:
#                     # This class' sample was completed already
#                     completed_validation[sample_class] = True
#                     classes_to_remove.append(sample_class)

#         # Else, add to testing sample
#         # Accumulate the pixel counts, chek first if general sample is completed
#         if sample.get(sample_class) is None:
#             sample[sample_class] = class_count
#         else:
#             # if sample[sample_class] < sample_sizes[sample_class]:
#             sample_size = df[df['Key'] == sample_class]['TestingPix'].item()

#             # If sample isn't completed, add the sampled window
#             if sample[sample_class] < sample_size:
#                 sample[sample_class] += class_count
#                 # Check if last addition completed the sample
#                 if sample[sample_class] >= sample_size:
#                     completed_test[sample_class] = True  # this class' sample is now complete
#                     # but do not add to classes_to_remove
#             else:
#                 # This class' sample was completed already
#                 completed_test[sample_class] = True  
#                 classes_to_remove.append(sample_class)

#     # Create an array containing all the sampled pixels by adding the sampled windows from each quadrant (or part)
#     sampled_window = np.zeros(window_sample.shape, dtype=raster_arr.dtype)
    
#     # Filter out classes with already complete samples
#     if len(classes_to_remove) > 0:
#         for single_class in classes_to_remove:
#             # Put a 1 on a complete class
#             filter_out = np.where(window_sample == single_class, 1, 0)
#             sampled_window += filter_out
        
#         # All values greater than zero are pixels to remove from mask, reverse it so 1's are the sample mask
#         sampled_window = np.where(sampled_window == 0, 1, 0)
#     else:
#         sampled_window = window_sample[:,:]
    
#     # Slice and insert sampled window
    
#     if completed_test.get(sample_class, False):
#         sample_mask_val[win_row_ini:win_row_end,win_col_ini:win_col_end] += sampled_window
#     else:
#         sample_mask[win_row_ini:win_row_end,win_col_ini:win_col_end] += sampled_window

#     trials += 1

#     completed_samples = sum(list(completed_test.values())) + sum(list(completed_validation.values()))  # Values are all True if completed
#     if show_progress:
#         print(f' (completed {completed_samples:>2}/{total_classes:>2} samples)')
#     if completed_samples >= total_classes:
#         print(f'\n  --All samples completed in {trials} trials! Exiting.\n')

# if trials == max_trials:
#     print('\n  --WARNING! Max trials reached, samples may be incomplete, try increasing max trials.')

# print(f'  --Testing sample:       {sample}')
# print(f'  --Completed testing:    {completed_test}')
# print(f'  --Validation sample:    {validation_sample}')
# print(f'  --Completed validation: {completed_validation}')

# print('\n  --WARNING! This may contain oversampling caused by overlapping windows!')
# df['TestPixSamp'] = [sample.get(x,0) for x in df['Key']]
# df['TestPerSamp'] = (df['TestPixSamp'] / df['TestingPix']) * 100
# df['TestFrac'] = (df['TestPixSamp'] / df['PixelCount']) * 100
# df['TestingOk'] = [completed_test[x] for x in df['Key']]

# df['ValPixSamp'] = [validation_sample.get(x,0) for x in df['Key']]
# df['ValPerSamp'] = (df['ValPixSamp'] / df['V8nPix']) * 100
# df['ValFrac'] = (df['ValPixSamp'] / df['PixelCount']) * 100
# df['ValidationOk'] = [completed_validation[x] for x in df['Key']]
# print(df)
# df.to_csv(fn_sample)  # save the sample sizes

# # Convert the sample_mask to 1's (indicating pixels to sample) and 0's
# sample_mask = np.where(sample_mask >= 1, 1, 0)
# print(f"  --Values in testing mask: {np.unique(sample_mask)}")  # should be 1 and 0
# sample_mask_val = np.where(sample_mask_val >= 1, 1, 0)
# print(f"  --Values in validation mask: {np.unique(sample_mask)}")  # should be 1 and 0

# # Create a raster with the sampled windows, this will be the testing mask (or sampling mask)
# rs.create_raster(fn_testing_mask, sample_mask, epsg, gt)
# rs.create_raster(fn_validation_mask, sample_mask_val, epsg, gt)

# print('  Done ;-)')