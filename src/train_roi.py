#!/usr/bin/env python
# coding: utf-8

""" Create the training datasets (raster files) for machile learnig algorithms

NOTICE: run from 'rsml' environment (Python 3.8.13; GDAL 3.5.2 & matplotlib 3.6.0 from conda-forge)

author: Eduardo Jimenez <eduardojh@arizona.edu>
Changelog:
    Jan 13, 2023: Initial code inputs and plots of land cover percentages
    Jan 17, 2023: Split raster into small parts for sampling
    Jan 31, 2023: Random sampling works but still things to improve.
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


def read_stats(stats_file, train_percent=0.2):
    """ Read percentage of coverage for each land cover class """
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
sample_sizes, tr_keys, tr_frq, tr_size = read_stats(fn_stats, train_percent)

# # Read percentage of coverage for each land cover class
# sample_sizes = {}
# tr_keys = []
# tr_frq = []
# tr_size = []
# with open(fn_stats, 'r') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     header = next(reader)
#     print(f'{header[0]:>3} {header[3]:>11} {header[4]:>12} {"Training Sample":>10}')
#     for row in reader:
#         key = int(row[0])  # Keys are landcover
#         frq = int(row[3])
#         per = float(row[4])  # Percentage

#         # Number of pixels to sample per land cover class
#         train_pixels = int(frq*train_percent)
        
#         tr_keys.append(key)
#         tr_frq.append(frq)
#         tr_size.append(train_pixels)

#         sample_sizes[key] = train_pixels
        
#         print(f'{key:>3} {frq:>13} {per:>10.4f} {train_pixels:>10}')

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

sample = {}  # to save the sample

# Initialize the complete samples classes as False, when each class sample
# is complete its value will change to True
complete_classes = {}
for sample_key in list(sample_sizes.keys()):
    complete_classes[sample_key] = False
print(f'Complete classes: {complete_classes}')
completed_samples = sum(list(complete_classes.values()))  # Values are all True if completed

# Create a mask of the sampled regions
sample_mask = np.zeros(raster_arr.shape, dtype=raster_arr.dtype)
# sampled_window = np.ones((window_size, window_size), dtype=raster_arr.dtype)

samples_x_part = {}
for key in sample_sizes.keys():
    samples_x_part[key] = {}

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
        fn_quadrant_stats = f'{cwd}training/sampling/usv250s7cw_ROI1_statistics_part{part}.csv'
        part_lc, part_percentages, part_lc_groups, part_raster_arr, _ = rs.land_cover_percentages(fn_raster_part, fn_keys, fn_quadrant_stats, indices=inegi_indices)
        # print(part_lc)
        # print(part_percentages)

        # Prepare the sample sizes per class per part/quadrant
        sample_size_part, _, _, _ = read_stats(fn_quadrant_stats)

        # A window will be used for sampling, this array will hold the sample
        window_sample = np.zeros((window_size,window_size), dtype=int)

        nrows, ncols = raster_part.shape
        print(f'  Part {part}: nrows={nrows}, ncols={ncols}')
        max_samples_quad = int(nrows*ncols*0.05)  # sample a fraction
        # max_samples_quad = 1000
        print(f'  Max samples per quadrant: {max_samples_quad}')

        i = 0
        # TODO: Fix: most of the pixels are sampled from the first quadrant
        # TODO: Fix: only sample windows with mixed classes are retained, add pure classes
        # TODO: Fix: sample windows can contain parts or complete windows previously sampled 
        while (i < max_samples_quad and completed_samples < len(complete_classes.keys())):
            show_progress = (i%5000 == 0)  # Step to show progress
            if show_progress:
                print(f'  Sampling {i} of {max_samples_quad}...')

            # 1) Generate a random point (row_sample, col_sample) to sample the array
            #    Coordinates relative to array positions [0:nrows, 0:ncols]
            #    Subtract half the window size to avoid sampling too close to the edges
            col_sample = random.randint(0 + window_size//2, ncols - window_size//2)
            row_sample = random.randint(0 + window_size//2, nrows - window_size//2)
            # print(f'    Sample point: row_sample={row_sample:>6} in range: ({0 + window_size//2:>6}, {nrows - window_size//2:>6}), col_sample={col_sample:>6} in range: ({0 + window_size//2:>6}, {ncols - window_size//2:>6})')

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
            
            # 4) Check and adjust the shapes of the arrays to slice and insert properly
            # if ws.shape != window_sample.shape:
            #     # WARNING: Only end row and/or column can be adjusted
            #     print(f'    Warning! Array dimensions do not match: {ws.shape} and {window_sample.shape}, sample window will be adjusted.')
            window_sample[:ws.shape[0], :ws.shape[1]] = ws
            # print(f'    Window sample:', window_sample)

            # Accumulate the pixel count of the sampled classes:
            
            # 5) Get the unique values in sample (sample_keys) and its count (sample_freq)
            sample_keys, sample_freq = np.unique(window_sample, return_counts=True)
            # if show_progress:
            #     print(f'    Classes: {sample_keys}, Freq: {sample_freq}')
            
            # 6) Check the number of classes sampled
            #    If sample contains a single class that is NA_CLASS, then discard the sample
            if len(sample_keys) == 0:
                print('    Sample size is empty. How did this happened?')
                i += 1
                continue
            elif len(sample_keys) == 1 and sample_keys[0] == NA_CLASS:
                # print(f'    Sample with only NA_CLASS values found. Skipping.')
                i += 1
                continue
            
            # If this point is reached, sampling window contains one or more valid classes
            classes_to_remove = []  # Avoid adding zeros or completed classes to the mask

            # 7) Iterate over each class sample and add its respective pixel count to the sample
            for sample_class, class_count in zip(sample_keys, sample_freq):
                if sample_class == NA_CLASS:
                    # Sample is mixed with zeros, tag it to remove it and go to next sample_class
                    # print(f'    Sample mixes class 0 (NA_CLASS). Skipping.')
                    if sample_class not in classes_to_remove:
                        classes_to_remove.append(sample_class)
                    continue
                
                # Check if class sample already complete
                if complete_classes[sample_class] is True:
                    # print(f'    Sample class: {sample_class} complete. Skipping.')
                    if sample_class not in classes_to_remove:
                        classes_to_remove.append(sample_class)
                    continue

                # Accumulate the pixel counts for each sampled class
                if sample.get(sample_class) is None:
                    sample[sample_class] = class_count  # Initialize classes count, if it does not exist
                else:
                    sample[sample_class] += class_count  # Increase class count #TODO: change count only the filtered pixels! After processing classes_to_remove!
                
                # Check after counting if class sample is complete and adjust properly
                if sample.get(sample_class) >= sample_sizes[sample_class]:
                    complete_classes[sample_class] = True
                    print(f'    Sample class for: {sample_class} is now complete. Total complete {sum(list(complete_classes.values()))}/{len(complete_classes.keys())}')


            # If all the sampled classes are completed already, discard sample and add nothing to sample mask
            if len(classes_to_remove) == len(sample_keys):
                # print(f'    No classes to add {len(classes_to_remove)} {len(sample_keys)}... {i}')
                i += 1
                continue

            # Create an array containing all the sampled pixels by adding the sampled windows from each quadrant (or part)
            sampled_window = np.zeros(ws.shape, dtype=raster_arr.dtype)

            # Filter out classes with already complete samples
            if len(classes_to_remove) > 0:
                # print(f'    Updating sample mask...{i}/{max_samples_quad}')
                for single_class in classes_to_remove:
                    # Put a 1 on a complete class
                    filter_out = np.where(sampled_window == single_class, 1, 0)
                    sampled_window += filter_out
                
                # All values greater than zero are pixels to remove from mask, reverse it so 1's are the sample mask
                sampled_window = np.where(sampled_window == 0, 1, 0)

                # Convert from slice indices to quadrant row/colum
                row_mask = row_start + win_row_ini
                col_mask = col_start + win_col_ini
                row_mask_end = row_start + win_row_end
                col_mask_end = col_start + win_col_end

                # Slice and insert sampled window
                mask_shape = (row_mask_end-row_mask, col_mask_end-col_mask)

                # # To check dimensions
                # if ws.shape != (7, 7):
                #     print(f'    In sample {i}: Mask array shape={mask_shape} and sampled_window={sampled_window.shape} {mask_shape==sampled_window.shape}')

                # # Apparently there is no need because eveything is adjusted
                # if mask_shape != sampled_window.shape:
                #     print(f'    Sample {i}. Mask array shapes do not match: {mask_shape} and {sampled_window.shape}. [{row_mask}:{row_mask_end},{col_mask}:{col_mask_end}]. Window will be adjusted.')
                #     # WARNING: Only end row and/or column can be adjusted
                #     row_mask_end = row_mask + sampled_window.shape[0]
                #     col_mask_end = col_mask + sampled_window.shape[1]
                #     print(f'    Mask window: [{row_mask}:{row_mask_end},{col_mask}:{col_mask_end}]')

                # print(f'    Actually adding...{i}')

                sample_mask[row_mask:row_mask_end,col_mask:col_mask_end] += sampled_window
            # else:
            #     print(f'    Keeping sample mask... {i}/{max_samples_quad}')
            
            # window sample counter
            i += 1

            completed_samples = sum(list(complete_classes.values()))  # Values are all True if completed
            if completed_samples == len(complete_classes.keys()):
                print(f'Overall sample is now complete! Exiting.')
            # if show_progress:
            #     print(f'Classes with complete sampes: {completed_samples}/{len(complete_classes.keys())}')

        part += 1

print('Samples per part:')
print(samples_x_part)
print(f'Complete classes at the end: {complete_classes}')

# Convert the sample_mask to 1's (indicating pixels to sample) and 0's
sample_mask = np.where(sample_mask > 0, 1, 0)

# Create a raster with the sampled windows, this will be the training mask (or sampling mask)
rs.create_raster(fn_training_mask, sample_mask, epsg_proj, gt)

# Show parts in image grid
print(f'Creating plot of ROI divided into {parts_per_side}x{parts_per_side} parts...')
for ax, im in zip(grid, im_list):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)
plt.savefig(fn_train_div_plot, bbox_inches='tight', dpi=600)
# plt.show()

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

print(f'\n === OPENING TRAINING MASK ===\n')

### TRAINING MASK
# Read a raster with the location of the training sites
train_mask, nodata, metadata, geotransform, projection = rs.open_raster(fn_training_mask)
print(f'Opening raster: {fn_training_mask}')
print(f'Metadata      : {metadata}')
print(f'NoData        : {nodata}')
print(f'Columns       : {train_mask.shape[1]}')
print(f'Rows          : {train_mask.shape[0]}')
print(f'Geotransform  : {geotransform}')
print(f'Projection    : {projection}')

# Select the pixels using the train mask
train_labels = raster_arr[train_mask > 0]  # This array gets flatten

print('Done! ;-)')