#!/usr/bin/env python
# coding: utf-8

""" Create the testing and training datasets (raster files) for machile learnig algorithms

NOTICE: run from 'rsml' environment (Python 3.8.13; GDAL 3.5.2 & matplotlib 3.6.0 from conda-forge)

author: Eduardo Jimenez <eduardojh@arizona.edu>
Changelog:
    Jan 13, 2023: Initial code inputs and plots of land cover percentages.
    Jan 17, 2023: Split raster into small parts for sampling.
    Jan 31, 2023: Random sampling works but still things to improve.
    Feb 21, 2023: Stratified random sampling is spatially balanced, sampling mask raster is generated.
    Mar  6, 2023: Moved to no-quadrants approach since its more efficient and unnecessary. Also removed groups reclassification.
    May 16, 2023: Updated to use grouped land cover classes, and minor fixes.
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
fn_landcover = cwd + 'data/inegi_2018/land_cover_ROI1.tif'  # use groups w/ancillary data
fn_keys = cwd + 'data/inegi_2018/land_cover_groups.csv'
fn_lc_plot = cwd + 'sampling/ROI1_percent_plot.png'

fn_training_mask = cwd + 'sampling/ROI1_training_mask.tif'
fn_train_labels =  cwd + 'sampling/ROI1_training_labels.tif'
fn_testing_mask  = cwd + 'sampling/ROI1_testing_mask.tif'  # create testing mask, training is the complement
fn_test_labels = cwd + 'sampling/ROI1_testing_labels.tif'
fn_sample = cwd + 'sampling/dataset_sample_sizes.csv'

# Create a list of land cover keys and its area covered percentage
assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
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
train_percent = 0.2  # training-testing proportion is 80-20%

#### Sample size == testing dataset
# Use a dataframe to calculate sample size
df = pd.DataFrame({'Key': lc_lbl, 'PixelCount': freqs, 'Percent': percentages})
df['TrainPixels'] = (df['PixelCount']*train_percent).astype(int)
# print(df['TrainPixels'])

# # Undersample largest classes to compensate for unbalance
# max_val = df['TrainPixels'].max()
# fix_val = (df.loc[df['TrainPixels'] == max_val, 'PixelCount']*(1-test_percent)).astype(int)
# df.loc[df['TrainPixels'] == max_val, 'TrainPixels'] = fix_val

# Now calculate percentages
df['TrainPercent'] = (df['TrainPixels'] / df['PixelCount'])*100
print(df)

#### 2. Create the testing mask
assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
raster_arr, nd, meta, gt, proj, epsg = rs.open_raster(fn_landcover)
print(f'  --Opening raster : {fn_landcover}')
print(f'  ----Metadata     : {meta}')
print(f'  ----NoData       : {nd}')
print(f'  ----Columns      : {raster_arr.shape[1]}')
print(f'  ----Rows         : {raster_arr.shape[0]}')
print(f'  ----Geotransform : {gt}')
print(f'  ----Projection   : {proj}')
print(f'  ----EPSG         : {epsg}')
print(f'  ----Type         : {raster_arr.dtype}')

rows, cols = raster_arr.shape
print(f"  --Total pixels={rows*cols}, Values={sum(df['PixelCount'])}, NoData/Missing={rows*cols - sum(df['PixelCount'])}")

raster_arr = raster_arr.astype(int)
print(f"  --Before filling NoData: {np.unique(raster_arr)}")
raster_arr = raster_arr.filled(0)  # replace masked constant "--" with zeros
print(f"  --After filling NoData: {np.unique(raster_arr)}")
# print(f'  --Check new array type: {raster_arr.dtype}')

window_size = 7
sample = {}  # to save the sample

# Create a mask of the sampled regions
sample_mask = np.zeros(raster_arr.shape, dtype=raster_arr.dtype)

# A window will be used for sampling, this array will hold the sample
window_sample = np.zeros((window_size,window_size), dtype=int)

nrows, ncols = raster_arr.shape

max_trials = int(2e5)  # max of attempts to fill the sample size
print(f'  --Max trials: {max_trials}')

trials = 0  # attempts to complete the sample
completed = {}  # classes which sample is complete

for sample_key in list(df['Key']):
    completed[sample_key] = False
completed_samples = sum(list(completed.values()))  # Values are all True if completed
total_classes = len(completed.keys())
# print(completed)

sampled_points = []

while (trials < max_trials and completed_samples < total_classes):
    show_progress = (trials%10000 == 0)  # Step to show progress
    if show_progress:
        print(f'  --Trial {1 if trials == 0 else trials:>8} of {max_trials:>8} ', end='')

    # 1) Generate a random point (row_sample, col_sample) to sample the array
    #    Coordinates relative to array positions [0:nrows, 0:ncols]
    #    Subtract half the window_size to avoid sampling too close to the edges, use window_size step to avoid overlapping
    col_sample = random.randrange(0 + window_size//2, ncols - window_size//2, window_size)
    row_sample = random.randrange(0 + window_size//2, nrows - window_size//2, window_size)

    # Save the points previously sampled to avoid repeating and oversampling
    point = (row_sample, col_sample)
    if point in sampled_points:
        trials +=1
        continue
    else:
        sampled_points.append(point)

    # 2) Generate a sample window around the random point, here create the boundaries,
    #    these rows and columns will be used to slice the sample
    win_col_ini = col_sample - window_size//2
    win_col_end = col_sample + window_size//2 + 1  # add 1 to slice correctly
    win_row_ini = row_sample - window_size//2
    win_row_end = row_sample + window_size//2 + 1

    assert win_col_ini < win_col_end, f"Incorrect slice indices on x-axis: {win_col_ini} < {win_col_end}"
    assert win_row_ini < win_row_end, f"Incorrect slice indices on y-axis: {win_row_ini} < {win_row_end}"

    # 3) Check if sample window is out of range, if so trim the window to the array's edges accordingly
    #    This may not be necessary if half the window size is subtracted, but still
    if win_col_ini < 0:
        # print(f'    --Adjusting win_col_ini: {win_col_ini} to 0')
        win_col_ini = 0
    if win_col_end > ncols:
        # print(f'    --Adjusting win_col_end: {win_col_end} to {ncols}')
        win_col_end = ncols
    if win_row_ini < 0:
        # print(f'    --Adjusting win_row_ini: {win_row_ini} to 0')
        win_row_ini = 0
    if  win_row_end > nrows:
        # print(f'    --Adjusting win_row_end: {win_row_end} to {nrows}')
        win_row_end = nrows

    # 4) Check and adjust the shapes of the arrays to slice and insert properly, only final row/column can be adjusted
    window_sample = raster_arr[win_row_ini:win_row_end,win_col_ini:win_col_end]
    # print(window_sample)
    
    # 5) Get the unique values in sample (sample_keys) and its count (sample_freq)
    sample_keys, sample_freq = np.unique(window_sample, return_counts=True)
    classes_to_remove = []  # Avoid adding zeros or completed classes to the mask

    # 6) Iterate over each class sample and add its respective pixel count to the sample
    for sample_class, class_count in zip(sample_keys, sample_freq):
        if sample_class == NA_CLASS:
            # Sample is mixed with zeros, tag it to remove it and go to next sample_class
            classes_to_remove.append(sample_class)
            continue

        if completed.get(sample_class, False):
            classes_to_remove.append(sample_class)  # do not add completed classes to the sample
            continue

        # Accumulate the pixel counts, chek first if general sample is completed
        if sample.get(sample_class) is None:
            sample[sample_class] = class_count
        else:
            # if sample[sample_class] < sample_sizes[sample_class]:
            sample_size = df[df['Key'] == sample_class]['TrainPixels'].item()

            # If sample isn't completed, add the sampled window
            if sample[sample_class] < sample_size:
                sample[sample_class] += class_count
                # Check if last addition completed the sample
                if sample[sample_class] >= sample_size:
                    completed[sample_class] = True  # this class' sample is now complete
                    # but do not add to classes_to_remove
            else:
                # This class' sample was completed already
                completed[sample_class] = True
                classes_to_remove.append(sample_class)

    # Create an array containing all the sampled pixels by adding the sampled windows from each quadrant (or part)
    sampled_window = np.zeros(window_sample.shape, dtype=raster_arr.dtype)
    
    # Filter out classes with already complete samples
    if len(classes_to_remove) > 0:
        for single_class in classes_to_remove:
            # Put a 1 on a complete class
            filter_out = np.where(window_sample == single_class, 1, 0)
            sampled_window += filter_out
        
        # All values greater than zero are pixels to remove from mask, reverse it so 1's are the sample mask
        sampled_window = np.where(sampled_window == 0, 1, 0)
    else:
        sampled_window = window_sample[:,:]
    
    # Slice and insert sampled window
    sample_mask[win_row_ini:win_row_end,win_col_ini:win_col_end] += sampled_window

    trials += 1

    completed_samples = sum(list(completed.values()))  # Values are all True if completed
    if show_progress:
        print(f' (completed {completed_samples:>2}/{total_classes:>2} samples)')
    if completed_samples >= total_classes:
        print(f'\n  --All samples completed in {trials} trials! Exiting.\n')

if trials == max_trials:
    print('\n  --WARNING! Max trials reached, samples may be incomplete, try increasing max trials.')

print(f'  --Sample sizes per class: {sample}')
print(f'  --Completed samples: {completed}')

print('\n  --WARNING! This may contain oversampling caused by overlapping windows!')
df['SampledPixels'] = [sample.get(x,0) for x in df['Key']]
df['SampledPercent'] = (df['SampledPixels'] / df['TrainPixels']) * 100
df['SampledPerClass'] = (df['SampledPixels'] / df['PixelCount']) * 100
df['SampleComplete'] = [completed[x] for x in df['Key']]
df.to_csv(fn_sample)
print(df)

# Convert the sample_mask to 1's (indicating pixels to sample) and 0's
sample_mask = np.where(sample_mask >= 1, 1, 0)
print(f"  --Values in mask: {np.unique(sample_mask)}")  # should be 1 and 0

# To undersample, flip the training/testing pixels of the biggest class
# flip_mask = np.where((raster_arr == 3) & (sample_mask == 1), 0, sample_mask)
# sample_mask = np.where((raster_arr == 3) & (sample_mask == 0), 1, flip_mask)

# Create a raster with actual labels (land cover classes)
sample_labels = np.where(sample_mask > 0, raster_arr, 0)
compl_labels = np.where(sample_mask == 0, raster_arr, 0)
rs.create_raster(fn_train_labels, sample_labels, epsg, gt)
rs.create_raster(fn_test_labels, compl_labels, epsg, gt)

# Create a raster with the sampled windows, this will be the sampling mask
rs.create_raster(fn_training_mask, sample_mask, epsg, gt)
rs.create_raster(fn_testing_mask, np.logical_not(sample_mask,), epsg, gt)

print('  Done ;-)')