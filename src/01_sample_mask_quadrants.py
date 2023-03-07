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
    Mar 6, 2023: Moved to no-quadrants approach since its more efficient and unnecessary, keeped as a backup!
                 The other file has up to date improvements that this one does not. Use with care!
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

            # Should be this added back???
            # # Set at least one pixel for sampling, hard to do in practice tho
            # if frq <= 1/train_percent and train_pixels <= 0:
            #     train_pixels = 1
            
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
fn_lc_plot = cwd + 'training/usv250s7cw_ROI1_plot.png'
fn_training_mask  = cwd + 'training/usv250s7cw_ROI1_train_mask.tif'
# Filenames of operations by group
fn_grp_landcover = cwd + 'training/groups/usv250s7cw_ROI1_LC_KEY_grp.tif'
fn_grp_keys = cwd + 'training/groups/usv250s7cw_ROI1_grp_keys.csv'
fn_grp_raster = cwd + 'training/groups/usv250s7cw_ROI1_grp'
fn_grp_plot = cwd + 'training/groups/usv250s7cw_ROI1_grp_plot.png'

fn_train_div_plot = cwd + 'training/sampling/usv250s7cw_ROI1_divided.png'

# Create a CSV file with the pixel count and percentage per land cover and land cover group
inegi_indices = (2, 1, 4)  # INEGI's land cover column, land cover key column, and group column
lc_desc, percentages, land_cover_groups, raster_arr, gt = rs.land_cover_percentages(fn_landcover, fn_keys, fn_stats, indices=inegi_indices)

# Plot land cover horizontal bar
print('Plotting land cover percentages...')
rs.plot_land_cover_hbar(lc_desc, percentages, fn_lc_plot,
    title='INEGI Land Cover Classes in Calakmul Biosphere Reserve',
    xlabel='Percentage (based on pixel count)',
    xlims=(0,50))

# Put together all land cover classes by group
grp_filter, grp_percent = rs.land_cover_percentages_grp(land_cover_groups)

# Create a raster reclassified by land cover group
# Projection to create raster. SJR: 32612=WGS 84 / UTM zone 12N; CBR: 32616=WGS 84 / UTM zone 16N
# epsg_proj = 32612 
epsg_proj = 32616
rs.reclassify_land_cover_by_group(raster_arr, gt, epsg_proj, grp_filter, fn_stats, fn_grp_keys, fn_grp_landcover, intermediate=fn_grp_raster)

print('Plotting land cover percentages by group...\n')
rs.plot_land_cover_hbar(grp_filter, grp_percent, fn_grp_plot,
    title='INEGI Land Cover Classes (by group) in Calakmul Biosphere Reserve',
    xlabel='Percentage (based on pixel count)',
    xlims=(0,50))

#### 2. Create the training mask

# Read percentage of coverage for each land cover clas
train_percent = 0.2
sample_sizes, tr_keys, tr_frq, tr_size = define_sample_size(fn_stats, train_percent)

# Split the raster into quadrants (or ninth squares): 2x2, 3x3, etc.
parts_per_side = 3

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
print(f'  New Type: {raster_arr.dtype}')

rows_per_square = rows//parts_per_side
cols_per_square = cols//parts_per_side

# Create a figure of the ROI splitted
fig = plt.figure(figsize=(24., 16.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

# Create a raster file per array part
print('\n === PROCESSING ROI BY PARTS (QUADRANTS) === \n')
part = 0  # quadrant or array part couter
im_list = []

window_size = 7

sample = {}  # to save the sample
part_samples = []

# Create a mask of the sampled regions
sample_mask = np.zeros(raster_arr.shape, dtype=raster_arr.dtype)

samples_x_part = {}
for key in sample_sizes.keys():
    samples_x_part[key] = np.zeros(parts_per_side*parts_per_side, dtype=int)

for part_row in range(parts_per_side):
    for part_col in range(parts_per_side):

        print(f'\nPART (OR QUADRANT) {part}\n')

        # Create intervals to slice (last part_row/column will contain extra pixels when 'rows'/'cols'
        # is not exactly divisible by 'parts_per_side')
        row_start = 0 + (rows_per_square*part_row)
        row_end = rows_per_square + (rows_per_square*part_row) if part_row != (parts_per_side-1) else rows+1
        col_start = 0 + (cols_per_square*part_col)
        col_end = cols_per_square + (cols_per_square*part_col) if part_col != (parts_per_side-1) else cols+1

        # Extract the portion of the array
        raster_part = raster_arr[row_start:row_end,col_start:col_end]

        # Append raster to a list to use in ImageGrid
        im_list.append(raster_part)

        # Create a GeoTIFF per each part
        fn_raster_part = f'{cwd}training/sampling/usv250s7cw_ROI1_LC_KEY_part{part}.tif'
        
        # Calculate the coordinates of the geotransform
        ulx_part = ulx + (col_start * xres)
        uly_part = uly + (row_start * yres)
        part_gt = [ulx_part, xres, 0, uly_part, 0, yres]

        rs.create_raster(fn_raster_part, raster_part, epsg_proj, part_gt)

        # Extract land cover percentages per quadrant
        fn_quadrant_stats = f'{cwd}training/sampling/usv250s7cw_ROI1_statistics_part{part}.csv'
        part_lc, part_percentages, part_lc_groups, part_raster_arr, _ = rs.land_cover_percentages(fn_raster_part, fn_keys, fn_quadrant_stats, indices=inegi_indices)

        # Prepare the sample sizes per class per part/quadrant
        sample_size_part, _, _, _ = define_sample_size(fn_quadrant_stats)
        print(f'Sample size part {part}:')
        print(sample_size_part)
        # Iterate over land cover classes keys
        for key in sample_size_part.keys():
            # Place the pixel count of each class in its corresponding part (position)
            samples_x_part[key][part] = sample_size_part[key]

        # A window will be used for sampling, this array will hold the sample
        window_sample = np.zeros((window_size,window_size), dtype=int)

        nrows, ncols = raster_part.shape
        print(f'  Part {part}: nrows={nrows}, ncols={ncols}')
        # max_trials = int(nrows*ncols*0.025)
        max_trials = 25000  # max of attempts to fill the sample size
        print(f'  Max trials per quadrant: {max_trials}')

        i = 0  # sample counter
        completed = {}  # classes which sample is complete
        sample_part = {}  # the sample pixels of current part/quadrant

        for sample_key in list(sample_size_part.keys()):
            completed[sample_key] = False
        completed_samples = sum(list(completed.values()))  # Values are all True if completed

        sampled_points = []

        while (i < max_trials and completed_samples < len(completed.keys())):
            show_progress = (i%1000 == 0)  # Step to show progress
            if show_progress:
                print(f'  Trial {i} of {max_trials}...')

            # 1) Generate a random point (row_sample, col_sample) to sample the array
            #    Coordinates relative to array positions [0:nrows, 0:ncols]
            #    Subtract half the window_size to avoid sampling too close to the edges, use window_size step to avoid overlapping
            col_sample = random.randrange(0 + window_size//2, ncols - window_size//2, window_size)
            row_sample = random.randrange(0 + window_size//2, nrows - window_size//2, window_size)

            # Save the points previously sampled to avoid repeating and oversampling
            point = (row_sample, col_sample)
            if point in sampled_points:
                i +=1
                # print(f'Point {point} already sampled. Skipping.')
                continue
            else:
                sampled_points.append(point)
            
            # print(f'    Sample point: row_sample={row_sample:>6} in range: ({0 + window_size//2:>6}, 
            # {nrows - window_size//2:>6}), col_sample={col_sample:>6} in range: ({0 + window_size//2:>6}, {ncols - window_size//2:>6})')

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
            
            # print(f'    Window: [{win_row_ini}:{win_row_end},{win_col_ini}:{win_col_end}]')
            ws = raster_part[win_row_ini:win_row_end,win_col_ini:win_col_end]
            
            # 4) Check and adjust the shapes of the arrays to slice and insert properly, only final row/column can be adjusted
            window_sample[:ws.shape[0], :ws.shape[1]] = ws
            
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
                    classes_to_remove.append(sample_class)
                    continue

                # Accumulate the pixel counts, chek first if general sample is completed
                if sample.get(sample_class) is None:
                    sample[sample_class] = class_count
                else:
                    if sample[sample_class] < sample_sizes[sample_class]:
                        sample[sample_class] += class_count  # Increase class count
                    else:
                        classes_to_remove.append(sample_class)
                        # print(f'Class {sample_class} already complete!')
                        # continue
                
                # Accumulate pixel counts in current part/quadrant sample
                if sample_part.get(sample_class) is None:
                    sample_part[sample_class] = class_count
                else:
                    if sample_part[sample_class] < sample_size_part[sample_class]:
                        sample_part[sample_class] += class_count
                    else:
                        completed[sample_class] = True
                        classes_to_remove.append(sample_class)
                        # continue  #is this necessary?

            # Create an array containing all the sampled pixels by adding the sampled windows from each quadrant (or part)
            sampled_window = np.zeros(ws.shape, dtype=raster_arr.dtype)
            # Convert from slice indices to quadrant row/colum
            row_mask = row_start + win_row_ini
            col_mask = col_start + win_col_ini
            row_mask_end = row_start + win_row_end
            col_mask_end = col_start + win_col_end

            # Filter out classes with already complete samples
            if len(classes_to_remove) > 0:
                # print(f'    Updating sample mask...{i}/{max_trials}')
                for single_class in classes_to_remove:
                    # Put a 1 on a complete class
                    # filter_out = np.where(sampled_window == single_class, 1, 0)
                    filter_out = np.where(window_sample == single_class, 1, 0)
                    sampled_window += filter_out
                
                # All values greater than zero are pixels to remove from mask, reverse it so 1's are the sample mask
                sampled_window = np.where(sampled_window == 0, 1, 0)

                # # Convert from slice indices to quadrant row/colum
                # row_mask = row_start + win_row_ini
                # col_mask = col_start + win_col_ini
                # row_mask_end = row_start + win_row_end
                # col_mask_end = col_start + win_col_end

                # # Slice and insert sampled window
                # sample_mask[row_mask:row_mask_end,col_mask:col_mask_end] += sampled_window
            else:
                sampled_window = window_sample[:,:]
    
            # Slice and insert sampled window
            sample_mask[row_mask:row_mask_end,col_mask:col_mask_end] += sampled_window

            # window sample counter
            i += 1

            completed_samples = sum(list(completed.values()))  # Values are all True if completed
            if completed_samples == len(completed.keys()):
                print(f'Part sample is now complete! Exiting.')

        part_samples.append(sample_part)
        part += 1

print('Pixels sampled to create the sampling mask. IMPORTANT! This is not the actual training sample.')
keys = samples_x_part.keys()
print(f"{'Key':3}", end='')
for x in range(parts_per_side*parts_per_side):
    print(f" Sampled", end='')
    print(f"/Size{x:2} ", end='')
    print(f"  (%) ", end='')
print(' RealSize', end='')
print(' Expected', end='')
print('  Diff', end='')
print(f"{'Sampled':>10}", end='')
print(' Sampled (%)')
for i in keys:
    print(f'{i:3}', end='')
    class_sampled = 0
    for j in range(parts_per_side*parts_per_side):
        d = part_samples[j]
        val = d.get(i, 0)
        print(f'{val:8}', end='')
        class_sampled += val
        print(f'{samples_x_part[i][j]:8}', end='')
        percent = 0 if samples_x_part[i][j] == 0 else (val/samples_x_part[i][j])*100
        print(f'{percent:>6.1f}', end='')
    cls_sample_sz = sum(samples_x_part[i])  # sum sample sizes per class
    print(f'{cls_sample_sz:9}', end='')
    print(f'{sample_sizes[i]:9}', end='')
    print(f'{sample_sizes[i]-cls_sample_sz:6}', end='')
    print(f'{class_sampled:10}', end='')
    print(f'{(class_sampled/cls_sample_sz)*100:>12.2f}')

# Another table to show the sampled pixels per class
print('Sample sizes per class:')
print(sample_sizes)
print(sample)
tr_sampled = []
tr_per_sampled = []
for key in tr_keys:
    tr_sampled.append(sample.get(key, 0))
# Get the training percentage sampled = pixels actually sampled/sample size
tr_per_sampled = (np.array(tr_sampled, dtype=float) / np.array(tr_size, dtype=float))*100
# Pixels actually sampled/pixels per land cover class
tr_class_sampled = (np.array(tr_sampled, dtype=float) / np.array(tr_frq, dtype=float))*100

print('\nWARNING! This may contain oversampling caused by overlapping windows!')
print(f"{'Key':>3}{'Freq':>10}{'Samp Size':>10}{'Sampled':>10}{'Sampled/Size %':>18}{'Sampled/Class %':>18}")
for i in range(len(tr_keys)):
    # {key:>3} {frq:>13} {per:>10.4f} {train_pixels:>10}
    print(f'{tr_keys[i]:>3}{tr_frq[i]:>10}{tr_size[i]:>10}{tr_sampled[i]:>10}{tr_per_sampled[i]:>15.2f}{tr_class_sampled[i]:>15.2f}')

# Convert the sample_mask to 1's (indicating pixels to sample) and 0's
sample_mask = np.where(sample_mask >= 1, 1, 0)
# print(np.unique(sample_mask))  # should be 1 and 0

# Create a raster with the sampled windows, this will be the training mask (or sampling mask)
rs.create_raster(fn_training_mask, sample_mask, epsg_proj, gt)

# # Show parts/quadrants in image grid
# print(f'Creating plot of ROI divided into {parts_per_side}x{parts_per_side} parts...')
# for ax, im in zip(grid, im_list):
#     # Iterating over the grid returns the Axes.
#     ax.imshow(im)
# plt.savefig(fn_train_div_plot, bbox_inches='tight', dpi=600)
# # plt.show()

print('Done ;-)')