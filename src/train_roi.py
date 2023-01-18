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
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# adding the directory with modules
sys.path.insert(0, '/vipdata/2023/land_cover_analysis/lib/')
# sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib/')

import rsmodule as rs

#### 1. Analyze land cover classes percentages and create a training raster with (training labels)

# On Ubuntu machine
cwd = '/vipdata/2023/CALAKMUL/ROI1/'
# On Windows laptop
# cwd = 'D:/Desktop/CALAKMUL/ROI1/'

fn_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY.tif'
fn_keys = cwd + 'training/usv250s7cw_ROI1_updated.txt'
fn_stats = cwd + 'training/usv250s7cw_ROI1_statistics.csv'
fn_lc_plot = cwd + 'training/usv250s7cw_ROI1_plot.png'
# Filenames of operations by group
fn_grp_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY_grp.tif'
fn_grp_keys = cwd + 'training/usv250s7cw_ROI1_grp_keys.csv'
fn_grp_raster = cwd + 'training/usv250s7cw_ROI1_grp'
fn_grp_plot = cwd + 'training/usv250s7cw_ROI1_grp_plot.png'

fn_train_div_plot = cwd + 'training/usv250s7cw_ROI1_divided.png'

# Create a CSV file with the pixel count and percentage per land cover and land cover group
# inegi_indices = (2, 1, 4)  # INEGI's land cover column, land cover key column, and group column
# lc_desc, percentages, land_cover_groups, raster_arr, gt = rs.land_cover_percentages(fn_landcover, fn_keys, fn_stats, indices=inegi_indices)

# # Plot land cover horizontal bar
# print('Plotting land cover percentages...')
# rs.plot_land_cover_hbar(lc_desc, percentages, fn_lc_plot,
#     title='INEGI Land Cover Classes in Calakmul Biosphere Reserve',
#     xlabel='Percentage (based on pixel count)',
#     xlims=(0,50))

# # Put together all land cover classes by group
# grp_filter, grp_percent = rs.land_cover_percentages_grp(land_cover_groups)

# # Create a raster reclassified by land cover group
# # Projection to create raster. SJR: 32612=WGS 84 / UTM zone 12N; CBR: 32616=WGS 84 / UTM zone 16N
# # epsg_proj = 32612 
epsg_proj = 32616
# rs.reclassify_land_cover_by_group(raster_arr, gt, epsg_proj, grp_filter, fn_stats, fn_grp_keys, fn_grp_landcover, intermediate=fn_grp_raster)

# print('Plotting land cover percentages by group...')
# rs.plot_land_cover_hbar(grp_filter, grp_percent, fn_grp_plot,
#     title='INEGI Land Cover Classes (by group) in Calakmul Biosphere Reserve',
#     xlabel='Percentage (based on pixel count)',
#     xlims=(0,50))

#### 2. Create the training mask

train_percent = 0.2

# Read percentage of coverage for each land cover class
tr_keys = []
tr_frq = []
tr_size = []
with open(fn_stats, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader)
    print(f'{header[0]:>3} {header[3]:>11} {header[4]:>12} {"Sample Size":>10}')
    for row in reader:
        key = int(row[0])  # Keys are landcover
        frq = int(row[3])
        per = float(row[4])  # Percentage

        # Number of pixels to sample per land cover class
        train_pixels = int(frq*train_percent)
        
        tr_keys.append(key)
        tr_frq.append(frq)
        tr_size.append(train_pixels)
        
        print(f'{key:>3} {frq:>13} {per:>10.4f} {train_pixels:>10}')

# Split the raster into quadrants (or ninth squares)
parts_per_side = 3
print(f'Openning {fn_landcover}...')
raster_arr, nd, meta, gt, proj = rs.open_raster(fn_landcover)
print(f'{proj}: {type(proj)}')
# Get the raster extent
rows, cols = raster_arr.shape
ulx, xres, _, uly, _, yres = gt
extent = [ulx, ulx + xres*cols, uly, uly + yres*rows]

print(f'Metadata: {meta}')
print(f'NoData  : {nd}')
print(f'Columns : {cols}')
print(f'Rows    : {rows}')
print(f'Extent  : {extent}')

rows_per_square = rows//parts_per_side
cols_per_square = cols//parts_per_side

# Create a figure of the ROI splitted
fig = plt.figure(figsize=(24., 16.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
part = 1
im_list = []
for row in range(parts_per_side):
    for col in range(parts_per_side):

        # Create intervals to slice, last row/column will contain extra pixels when 'rows'/'cols'
        # is not exactly divisible by 'parts_per_side'
        row_start = 0 + (rows_per_square*row)
        row_end = rows_per_square + (rows_per_square*row) if row != (parts_per_side-1) else rows+1
        print(f'Rows {row_start}:{row_end}')

        col_start = 0 + (cols_per_square*col)
        col_end = cols_per_square + (cols_per_square*col) if col != (parts_per_side-1) else cols+1
        print(f'Cols {col_start}:{col_end}')

        # Extract the portion of the array
        raster_part = raster_arr[row_start:row_end,col_start:col_end]

        # Append raster to a list to use in ImageGrid
        im_list.append(raster_part)

        # Create a GeoTIFF per each part
        fn_part = f'{cwd}training/usv250s7cw_ROI1_LC_KEY_part{part}.tif'
        
        # Calculate the coordinates of the geotransform
        ulx_part = ulx + (col_start * xres)
        uly_part = uly + (row_start * yres)
        part_gt = [ulx_part, xres, 0, uly_part, 0, yres]

        rs.create_raster(fn_part, raster_part, epsg_proj, part_gt)

        part += 1

# Show parts in image grid
for ax, im in zip(grid, im_list):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)
plt.savefig(fn_train_div_plot, bbox_inches='tight', dpi=600)
# plt.show()

# Extract land cover percentages per quadrant


# Create locations to sample, use a window
sample_window=7

# Create a mask to extract the training sample