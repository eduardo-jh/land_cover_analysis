#!/usr/bin/env python
# coding: utf-8

""" Create the training datasets (raster files) for machile learnig algorithms

NOTICE: run from 'rsml' environment (Python 3.8.13; GDAL 3.5.2 & matplotlib 3.6.0 from conda-forge)

author: Eduardo Jimenez <eduardojh@arizona.edu>
Changelog:
    Jan 13, 2023: Initial code inputs and plots of land cover percentages.
    Jan 17, 2023: Split raster into small parts for sampling.
    Jan 31, 2023: Random sampling works but still things to improve.
    Feb 21, 2023: Stratified random sampling is spatially balanced, sampling mask raster is generated.
    Mar 6, 2023: Moved to no-quadrants approach since its more efficient and unnecessary. Also removed groups reclassification.
"""

import sys
import platform
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

NA_CLASS = 0

# adding the directory with modules
system = platform.system()
if system == 'Windows':
    # On Windows laptop
    sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib/')
    cwd = 'D:/Desktop/CALAKMUL/ROI1/'
elif system == 'Linux':
    # On Ubuntu machine
    sys.path.insert(0, '/vipdata/2023/land_cover_analysis/lib/')
    cwd = '/vipdata/2023/CALAKMUL/ROI1/'
else:
    print('System not yet configured!')

import rsmodule as rs


def define_sample_size(stats_file: str, train_percent: float = 0.2) -> tuple:
    """ Defines the sample size per class using the percentage of coverage for each land cover class, which is read from file """
    sample_sizes = {}
    tr_keys = []
    tr_frq = []
    tr_size = []
    with open(stats_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        print(f'{header[0]:>3} {header[3]:>11} {header[4]:>12} {"Training Sample":>10}')
        for row in reader:
            key = int(row[0])  # Keys are landcover
            frq = int(row[3])
            per = float(row[4])  # Percentage

            # Number of pixels to sample per land cover class
            train_pixels = int(frq*train_percent)
            
            tr_keys.append(key)
            tr_frq.append(frq)
            tr_size.append(train_pixels)

            sample_sizes[key] = train_pixels
            
            print(f'{key:>3} {frq:>13} {per:>10.4f} {train_pixels:>10}')
    return sample_sizes, tr_keys, tr_frq, tr_size


#### 1. Analyze land cover classes percentages and create a training raster with (training labels)
fn_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY.tif'
fn_keys = cwd + 'training/usv250s7cw_ROI1_updated.txt'
fn_stats = cwd + 'training/usv250s7cw_ROI1_statistics.csv'
# fn_lc_plot = cwd + 'training/usv250s7cw_ROI1_plot.png'
fn_training_mask  = cwd + 'training/usv250s7cw_ROI1_train_mask.tif'

# Create a CSV file with the pixel count and percentage per land cover and land cover group
inegi_indices = (2, 1, 4)  # INEGI's land cover column, land cover key column, and group column
lc_desc, percentages, land_cover_groups, raster_arr, gt = rs.land_cover_percentages(fn_landcover, fn_keys, fn_stats, indices=inegi_indices)

# Plot land cover horizontal bar
# print('Plotting land cover percentages...')
# rs.plot_land_cover_hbar(lc_desc, percentages, fn_lc_plot,
#     title='INEGI Land Cover Classes in Calakmul Biosphere Reserve',
#     xlabel='Percentage (based on pixel count)',
#     xlims=(0,50))

# Put together all land cover classes by group
grp_filter, grp_percent = rs.land_cover_percentages_grp(land_cover_groups)

# Create a raster reclassified by land cover group
# Projection to create raster. SJR: 32612=WGS 84 / UTM zone 12N; CBR: 32616=WGS 84 / UTM zone 16N
epsg_proj = 32616

#### 2. Create the training mask

# Read percentage of coverage for each land cover clas
train_percent = 0.2
sample_sizes, tr_keys, tr_frq, tr_size = define_sample_size(fn_stats, train_percent)

# Open the raster to split
print(f'Openning {fn_landcover}...')
raster_arr, nd, meta, gt, proj = rs.open_raster(fn_landcover)
# print(f'{proj}: {type(proj)}')
# Get the raster extent
rows, cols = raster_arr.shape
ulx, xres, _, uly, _, yres = gt
extent = [ulx, ulx + xres*cols, uly, uly + yres*rows]

print(f'Opening raster : {fn_landcover}')
print(f'  Metadata     : {meta}')
print(f'  NoData       : {nd}')
print(f'  Columns      : {cols}')
print(f'  Rows         : {rows}')
print(f'  Extent       : {extent}')
print(f'  Geotransform : {gt}')
print(f'  Projection   : {proj}')
print(f'  Type         : {raster_arr.dtype}')

raster_arr = raster_arr.astype(int)
print(f"Before: {np.unique(raster_arr)}")
raster_arr = raster_arr.filled(0)  # replace masked constant "--" with zeros
print(f"After: {np.unique(raster_arr)}")
print(f'  New Type: {raster_arr.dtype}')

window_size = 7
sample = {}  # to save the sample

# Create a mask of the sampled regions
sample_mask = np.zeros(raster_arr.shape, dtype=raster_arr.dtype)

# A window will be used for sampling, this array will hold the sample
window_sample = np.zeros((window_size,window_size), dtype=int)

nrows, ncols = raster_arr.shape
max_trials = int(nrows*ncols*0.025)
print(f'  Max trials: {max_trials}')
max_trials = int(2e5)  # max of attempts to fill the sample size
print(f'  Raster: nrows={nrows}, ncols={ncols}')
print(f'  Max trials: {max_trials}')

trials = 0  # attempts to complete the sample
completed = {}  # classes which sample is complete

for sample_key in list(sample_sizes.keys()):
    completed[sample_key] = False
completed_samples = sum(list(completed.values()))  # Values are all True if completed
total_classes = len(completed.keys())

sampled_points = []

while (trials < max_trials and completed_samples < total_classes):
    show_progress = (trials%1000 == 0)  # Step to show progress
    if show_progress:
        print(f'  Trial {1 if trials == 0 else trials} of {max_trials} ', end='')

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
        # print(f'    Adjusting win_col_ini: {win_col_ini} to 0')
        win_col_ini = 0
    if win_col_end > ncols:
        # print(f'    Adjusting win_col_end: {win_col_end} to {ncols}')
        win_col_end = ncols
    if win_row_ini < 0:
        # print(f'    Adjusting win_row_ini: {win_row_ini} to 0')
        win_row_ini = 0
    if  win_row_end > nrows:
        # print(f'    Adjusting win_row_end: {win_row_end} to {nrows}')
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
            if sample[sample_class] < sample_sizes[sample_class]:
                sample[sample_class] += class_count  # Increase class count
            else:
                completed[sample_class] = True  # this class' sample is now complete
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
        print(f' (completed {completed_samples} of {total_classes})')
    if completed_samples >= total_classes:
        print(f'\nAll samples completes in {trials} trials! Exiting.')

if trials == max_trials:
    print('\nWARNING! Max trials reached, samples may be incomplete, try increasing max trials.')

# Another table to show the sampled pixels per class
print('Sample sizes per class:')
print(sample_sizes)
print(sample)
print(completed)
sampled = []
sampled_percent = []
sample_complete = []
for key in tr_keys:
    sampled.append(sample.get(key, 0))
    sample_complete.append(completed[key])
# Get the training percentage sampled = pixels actually sampled/sample size
sampled_percent = (np.array(sampled, dtype=float) / np.array(tr_size, dtype=float))*100
# Pixels actually sampled/pixels per land cover class
sampled_class = (np.array(sampled, dtype=float) / np.array(tr_frq, dtype=float))*100

print('\nWARNING! This may contain oversampling caused by overlapping windows!')
print(f"{'Class':>5}{'Frequency':>10}{'Sample Size':>12}{'Sampled':>10}{'Sampled/Size %':>18}{'Sampled/Class %':>18}{'Completed':>10}")
for i in range(len(tr_keys)):
    print(f'{tr_keys[i]:>5}{tr_frq[i]:>10}{tr_size[i]:>12}{sampled[i]:>10}{sampled_percent[i]:>15.2f}{sampled_class[i]:>15.2f}{sample_complete[i]:>10}')

# Convert the sample_mask to 1's (indicating pixels to sample) and 0's
sample_mask = np.where(sample_mask >= 1, 1, 0)
print(np.unique(sample_mask))  # should be 1 and 0

# Create a raster with the sampled windows, this will be the training mask (or sampling mask)
rs.create_raster(fn_training_mask, sample_mask, epsg_proj, gt)

print('Done ;-)')