#!/usr/bin/env python
# coding: utf-8

""" Create the training datasets (raster files) for machile learnig algorithms

NOTICE: run from 'rsml' environment (Python 3.8.13; GDAL 3.5.2 & matplotlib 3.6.0 from conda-forge)

author: Eduardo Jimenez <eduardojh@arizona.edu>
Changelog:
    Jan 13, 2023: Initial code inputs and plots of land cover percentages
    Jan 17, 2023: Split raster into small parts for sampling
"""

import sys
import platform
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

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

train_percent = 0.2

# Read percentage of coverage for each land cover class
sample_sizes = {}
tr_keys = []
tr_frq = []
tr_size = []
with open(fn_stats, 'r') as csvfile:
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

print(f'  Metadata: {meta}')
print(f'  NoData  : {nd}')
print(f'  Columns : {cols}')
print(f'  Rows    : {rows}')
print(f'  Extent  : {extent}')
print(f'  Type    : {raster_arr.dtype}')

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
part = 1  # quadrant or array part couter
im_list = []

window_size = 7
# max_samples = 1000
sample = {}  # to save the sample
# total_count = 0
# pixels_to_sample = (window_size*window_size) * max_samples * (parts_per_side*parts_per_side)
skipped_pixels = 0

# Create a mask of the sampled regions
sample_mask = np.zeros(raster_arr.shape, dtype=raster_arr.dtype)
# sampled_window = np.ones((window_size, window_size), dtype=raster_arr.dtype)

for part_row in range(parts_per_side):
    for part_col in range(parts_per_side):

        print(f'\nPART (OR QUADRANT) {part}\n')

        # Create intervals to slice (last part_row/column will contain extra pixels when 'rows'/'cols'
        # is not exactly divisible by 'parts_per_side')
        row_start = 0 + (rows_per_square*part_row)
        row_end = rows_per_square + (rows_per_square*part_row) if part_row != (parts_per_side-1) else rows+1
        # print(f'  Rows {row_start}:{row_end}')

        col_start = 0 + (cols_per_square*part_col)
        col_end = cols_per_square + (cols_per_square*part_col) if part_col != (parts_per_side-1) else cols+1
        # print(f'  Cols {col_start}:{col_end}')

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
        part_lc, part_percentages, part_lc_groups, part_raster_arr, _ = rs.land_cover_percentages(fn_raster_part, fn_keys, f'{cwd}training/sampling/usv250s7cw_ROI1_statistics_part{part}.csv', indices=inegi_indices)
        # print(part_lc)
        # print(part_percentages)

        window_sample = np.zeros((window_size,window_size), dtype=int)

        # Create locations to sample, use a window preferrable
        nrows, ncols = raster_part.shape
        # print(f'    Part {part}: nrows={nrows}, ncols={ncols}')
        max_samples = int(nrows*ncols*0.2)  # sample a fraction

        for i in range(max_samples):
            if i%50 == 0:
                print(f'  Sampling {i} of {max_samples}...')
            # Generate a random point (row_sample, col_sample) to sample the array
            # Coordinates relative to array positions [0:nrows, 0:ncols]
            

            # Subtract half the window size to avoid sampling too close to the edges
            col_sample = random.randint(0 + window_size//2, ncols - window_size//2)
            row_sample = random.randint(0 + window_size//2, nrows - window_size//2)
            # print(f'    Sample point: row_sample={row_sample:>6} in range: ({0 + window_size//2:>6}, {nrows - window_size//2:>6}), col_sample={col_sample:>6} in range: ({0 + window_size//2:>6}, {ncols - window_size//2:>6})')

            # Generate the sample window boundaries
            win_col_ini = col_sample - window_size//2
            win_col_end = col_sample + window_size//2 + 1  # add 1 to slice correctly
            win_row_ini = row_sample - window_size//2
            win_row_end = row_sample + window_size//2 + 1

            assert win_col_ini < win_col_end, f"Incorrect slice indices on x-axis: {win_col_ini} < {win_col_end}"
            assert win_row_ini < win_row_end, f"Incorrect slice indices on y-axis: {win_row_ini} < {win_row_end}"

            # Check if sample window is out of range, if so trim the window to the array's edges accordingly
            # This may not be necessary if half the window size is subtracted, but still
            if win_col_ini < 0:
                print(f'    Adjusting win_col_ini: {win_col_ini} to 0')
                win_col_ini = 0
            if win_col_end > ncols:
                print(f'    Adjusting win_col_end: {win_col_end} to {ncols}')
                win_col_end = ncols
            if win_row_ini < 0:
                print(f'    Adjusting win_row_ini: {win_row_ini} to 0')
                win_row_ini = 0
            if  win_row_end > nrows:
                print(f'    Adjusting win_row_end: {win_row_end} to {nrows}')
                win_row_end = nrows
            
            # print(f'    Window: [{win_row_ini}:{win_row_end},{win_col_ini}:{win_col_end}]')
            ws = raster_part[win_row_ini:win_row_end,win_col_ini:win_col_end]
            # Check the shapes of the arrays to slice and inser properly
            if ws.shape != window_sample.shape:
                # WARNING: Only end row and/or column can be adjusted
                print(f'    Warning! Array dimensions do not match: {ws.shape} and {window_sample.shape}, sample window will be adjusted.')
            window_sample[:ws.shape[0], :ws.shape[1]] = ws
            # print(f'    Window sample:', window_sample)

            ### Accumulate the sampled classes ###
            
            # Get unique values in sample and its count
            sample_classes, sample_freq = np.unique(window_sample, return_counts=True)
            # print(f'    Classes: {sample_classes}, Freq: {sample_freq}')

            # If sample contains one or multiple land cover classes and their sample size has not been completed, keep it
            # If it contains a single class that is zero (null values or NAs), or if its sample size is already complete, discard the sample
            if len(sample_classes) == 0:
                print('    Sample size is empty. How did this happened?')
            elif len(sample_classes) == 1 and sample_classes[0] == 0:
                # print(f'    Sample with only zeros (null or NA) values found. Skipping.')
                skipped_pixels += window_size*window_size
                continue
            elif len(sample_classes) == 1 and sample.get(sample_classes[0], 0) >= sample_sizes[sample_classes[0]]:
                print(f'    {sample_classes[0]} has {sample.get(sample_classes[0], 0)} elements. Its sample of {sample_sizes[sample_classes[0]]} is completed. Skipping.')
                skipped_pixels += window_size*window_size
                continue

            # Get the list of all the land cover classes
            classes_to_sample = list(sample_sizes.keys())
            
            lc_check = 0  # To check the number of land cover classes
            for sample_class, class_count in zip(sample_classes, sample_freq):
                # Make sure elemens in 'sample_classes' are in sample_sizes, this means problems otherwise
                if sample_class in classes_to_sample:
                    lc_check += 1
                elif sample_class == 0:
                    # The sample is mixed with zeros
                    print(f'    Sample mixes class 0 (null, NA). Skipping.')
                    lc_check += 1  # Just to pass later check, but not add
                    continue
                else:
                    print(f'    WARNING! Land cover class {sample_class} not found in classes to sample.')
                
                # Accumulate the pixel counts for each sampled class
                if sample.get(sample_class) is None:
                    sample[sample_class] = class_count  # Initialize classes count, if not exists
                else:
                    sample[sample_class] += class_count  # Increase class count
                # total_count += class_count
            
            assert len(sample_classes) == lc_check, f"Classes to sample {len(sample_classes)} != {lc_check}"

            # Create an array containing all the sampled pixels by adding the sampled windows from each quadrant (or part)
            sampled_window = np.ones(ws.shape, dtype=raster_arr.dtype)
            
            # Convert from slice indices to quadrant row/colum
            row_mask = row_start + win_row_ini
            col_mask = col_start + win_col_ini
            row_mask_end = row_start + win_row_end
            col_mask_end = col_start + win_col_end

            # Slice and insert sampled window
            mask_shape = (row_mask_end-row_mask, col_mask_end-col_mask)
            
            # To check dimensions
            if ws.shape != (7, 7):
                print(f'    In sample {i}: Mask array shape={mask_shape} and sampled_window={sampled_window.shape} {mask_shape==sampled_window.shape}')

            # # Apparently there is no need because eveything is adjusted
            # if mask_shape != sampled_window.shape:
            #     print(f'    Sample {i}. Mask array shapes do not match: {mask_shape} and {sampled_window.shape}. [{row_mask}:{row_mask_end},{col_mask}:{col_mask_end}]. Window will be adjusted.')
            #     # WARNING: Only end row and/or column can be adjusted
            #     row_mask_end = row_mask + sampled_window.shape[0]
            #     col_mask_end = col_mask + sampled_window.shape[1]
            #     print(f'    Mask window: [{row_mask}:{row_mask_end},{col_mask}:{col_mask_end}]')

            sample_mask[row_mask:row_mask_end,col_mask:col_mask_end] = sampled_window

        part += 1

print(f'Sample: {sample}')
# WARNING! This is not accurate since sample window can be reduced!
# print(f'Sample pixels: total count={total_count}, to sample={pixels_to_sample}, sampled={pixels_to_sample-skipped_pixels} ({pixels_to_sample}-{skipped_pixels})')

# Create a raster with the sampled windows, this will be the training mask (or sampling mask)
rs.create_raster(fn_training_mask, sample_mask, epsg_proj, gt)

# # Show parts in image grid
# print(f'Creating plot of ROI divided into {parts_per_side}x{parts_per_side} parts...')
# for ax, im in zip(grid, im_list):
#     # Iterating over the grid returns the Axes.
#     ax.imshow(im)
# plt.savefig(fn_train_div_plot, bbox_inches='tight', dpi=600)
# # plt.show()

tr_sampled = []
tr_per_sampled = []
for key in tr_keys:
    tr_sampled.append(sample.get(key, 0))
# Get the training percentage sampled = pixels actually sampled/sample size
tr_per_sampled = (np.array(tr_sampled, dtype=float)/np.array(tr_size, dtype=float))*100

print(f"{'Key':>3}{'Freq':>10}{'Samp Size':>10}{'Sampled':>10}{'Sampled %':>10}")
for i in range(len(tr_keys)):
    # {key:>3} {frq:>13} {per:>10.4f} {train_pixels:>10}
    print(f'{tr_keys[i]:>3}{tr_frq[i]:>10}{tr_size[i]:>10}{tr_sampled[i]:>10}{tr_per_sampled[i]:>10.4f}')

# Plot the size of the sample per land cover
# rs.plot_land_cover_sample_bars(lc_desc, percentages, , fn_lc_plot[:-4] + '_percent.png')

print('Done! ;-)')