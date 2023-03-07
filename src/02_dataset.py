#!/usr/bin/env python
# coding: utf-8

""" 02_dataset.py: Prepare the data set to use with machine learning.
Reads the raster with the land cover classes (labels) and the spectral bands and phenology
(features or predictors), then transforms the 2D into a 1D data sets ready for the ML. 

Eduardo Jimenez <eduardojh@email.arizona.edu>

  Feb 24, 2023: initial code.

NOTE: run under 'rsml' conda environment (python 3.8.13, scikit-learn 1.1.2)
"""

import sys
import platform
import csv
import numpy as np
from datetime import datetime

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

fmt = '%Y_%m_%d-%H_%M_%S'
start = datetime.now()

### 1. CONFIGURE
# Projection to create raster. SJR: 32612=WGS 84 / UTM zone 12N; CBR: 32616=WGS 84 / UTM zone 16N
# epsg_proj = 32612 
epsg_proj = 32616

# Paths and file names for the current ROI
fn_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY.tif'        # Land cover raster
fn_train_mask = cwd + 'training/usv250s7cw_ROI1_train_mask.tif'
fn_train_labels = cwd + 'training/usv250s7cw_ROI1_train_labels.tif'
fn_nodata_mask = cwd + 'MONTHLY_NDVI/MONTHLY.NDVI.08.AUG.MIN.tif'   # Landsat 'NoData' filter, any file would work
fn_phenology = cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf'  # Phenology files
fn_phenology2 = cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf'
# File names to save results and reports
save_train_plot = cwd + f'results/{datetime.strftime(start, fmt)}_rf_training_plot.png'
save_train_stats = cwd + f'results/{datetime.strftime(start, fmt)}_rf_training_stats.csv'
save_conf_tbl = cwd + f'results/{datetime.strftime(start, fmt)}_rf_confussion_table.csv'
save_model = cwd + f'results/{datetime.strftime(start, fmt)}_rf_model.pkl'
save_report = cwd + f'results/{datetime.strftime(start, fmt)}_rf_classif_report.txt'
save_preds_raster = cwd + f'results/{datetime.strftime(start, fmt)}_rf_predictions.tif'
save_preds_fig = cwd + f'results/{datetime.strftime(start, fmt)}_rf_predictions.png'
save_params = cwd + f'results/{datetime.strftime(start, fmt)}_rf_parameters.csv'

### 2. TRAINING MASK
# Read a raster with the location of the training sites

train_mask, nodata, metadata, geotransform, projection = rs.open_raster(fn_train_mask)
print(f'Opening raster: {fn_train_mask}')
print(f'Metadata      : {metadata}')
print(f'NoData        : {nodata}')
print(f'Columns       : {train_mask.shape[1]}')
print(f'Rows          : {train_mask.shape[0]}')
print(f'Geotransform  : {geotransform}')
print(f'Projection    : {projection}')
print(f'Type          : {train_mask.dtype}')

# Find how many non-zero entries we have -- i.e. how many training data samples?
n_samples = (train_mask > 0).sum()
print(f'Training samples: {n_samples}')

### 3. LAND COVER LABELS
# Read the land cover raster and retrive the land cover classes
lc_arr, lc_nd, lc_md, lc_gt, lc_proj = rs.open_raster(fn_landcover)
print(f'Opening raster: {fn_landcover}')
print(f'Metadata      : {lc_md}')
print(f'NoData        : {lc_nd}')
print(f'Columns       : {lc_arr.shape[1]}')
print(f'Rows          : {lc_arr.shape[0]}')
print(f'Geotransform  : {lc_gt}')
print(f'Projection    : {lc_proj}')
print(f'Type          : {lc_arr.dtype}')

# # Mask out the 'NoData' pixels to match Landsat data and land cover classes
# dummy_array, _, _, _, _ = open_raster(fn_nodata_mask)
# lc_arr = np.ma.masked_array(lc_arr, mask=np.ma.getmask(dummy_array))
lc_arr = lc_arr.astype(int)

landcover_stats = {}
lc_classes, lc_freq = np.unique(lc_arr, return_counts=True)
valid_landcover = list(lc_classes.compressed()) # Get unmasked values, discards '--'

for grp, frq in zip(lc_classes, lc_freq):
    # print(f'{grp}, {frq} {type(grp)}')
    if not grp in valid_landcover:
        # print('Skipping')
        continue

    if landcover_stats.get(grp) is None:
        # print(f'Adding {grp}')
        landcover_stats[grp] = frq
    else:
        landcover_stats[grp] += frq

print('Analyzing labels from training dataset (land cover classes))')
lc_arr = lc_arr.astype(train_mask.dtype)
train_arr = np.where(train_mask > 0, lc_arr, 0)  # Actual labels (land cover classs)
# Save a raster with the actual labels (land cover classes) from the mask
rs.create_raster(fn_train_labels, train_arr, epsg_proj, lc_gt)

print(f'train_mask: {train_mask.dtype}, unique:{np.unique(train_mask.filled(0))}, {train_mask.shape}')
print(f'lc_arr    : {lc_arr.dtype}, unique:{np.unique(lc_arr.filled(0))}, {lc_arr.shape}')
print(f'train_arr : {train_arr.dtype}, unique:{np.unique(train_arr)}, {train_arr.shape}')

train_labels = lc_arr[train_mask > 0]  # This array gets flatten

train_stats = {}
# train_classes, train_freq = np.unique(train_labels, return_counts=True)
# valid_labels = list(train_classes.compressed())  # Get unmasked values, discards '--'
train_classes, train_freq = np.unique(train_arr, return_counts=True)
valid_labels = train_classes.tolist()
valid_labels.remove(0)

for lbl, f in zip(train_classes, train_freq):
    if not lbl in valid_labels:
        continue
    if train_stats.get(lbl) is None:
        train_stats[lbl] = f
    else:
        train_stats[lbl] += f

print(f'{len(train_classes)} unique land cover values in training dataset.')
# print(f'{len(train_classes)} = {len(lc_classes)}')

print('Saving group frequency in train dataset...')

x = []
y1 = []
y2 = []
with open(save_train_stats, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    header = ['Key_train', 'Freq_train', 'Key_all', 'Freq_all', 'Percent']
    writer.writerow(header)
    print(f'{header[0]:>10}  {header[1]:>10} {header[2]:>10} {header[3]:>10} {header[4]:>10}')
    for i, j in zip(valid_labels, valid_landcover):
        x.append(i)
        y1.append(landcover_stats[j])
        y2.append(train_stats[i])
        print(f'{i:>10}: {train_stats[i]:>10} {j:>10} {landcover_stats[j]:>10} ({(train_stats[i]/landcover_stats[j])*100:>6.2f} %)')
        writer.writerow([i, train_stats[i], j, landcover_stats[j], train_stats[i]/landcover_stats[j]])
n_train = sum(train_freq)
n_total = sum(lc_freq)
print(f'TOTAL: {n_train} / {n_total} ({(n_train/n_total)*100:>6.2f} %)')

x = np.array(x)
y1 = np.array(y1)
y2 = np.array(y2)
rs.plot_land_cover_sample_bars(x, y1, y2, save_train_plot)

### SPECTRAL BANDS
# bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir']
# band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
# months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
# vars = ['NPixels', 'MIN', 'MAX', 'AVG', 'STDEV']
# phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
# phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

# bands = ['Blue', 'Green', 'Ndvi', 'Red']
# band_num = ['B2', 'B3', '', 'B4']
# months = ['JAN',  'APR', 'JUL',  'OCT', 'DEC']
# nmonths = [1,  4, 7,  10, 12]
# vars = ['MIN', 'MAX', 'AVG']
# phen = ['SOS', 'EOS']
# phen2 = ['SOS2', 'EOS2']

bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
months = ['FEB',  'APR', 'JUN', 'AUG', 'OCT', 'DEC']
nmonths = [2, 4, 6, 8, 10, 12]
vars = ['AVG']
phen = ['SOS', 'EOS']
phen2 = ['SOS2', 'EOS2']

# bands = ['Blue', 'Green', 'Ndvi', 'Red']
# band_num = ['B2', 'B3', '', 'B4']
# months = ['JAN']
# vars = ['AVG']
# phen = ['SOS', 'EOS']
# phen2 = ['SOS2', 'EOS2']

# Calculate the dimensions of the array
cols = train_mask.shape[1]
rows = train_mask.shape[0]
lyrs = len(bands) * len(months) * len(vars) + len(phen) + len(phen2)
print(f'Rows={rows}, Cols={cols}, Layers={lyrs}')

# Create an array to hold all the data (spectral bands, VIs, and phenology)
bands_array = np.zeros((rows, cols, lyrs), dtype=np.float32)

layer = 0
for j, band in enumerate(bands):
    print(f'{band.upper()}')
    for i, month in enumerate(months):
        # filename = 'MONTHLY/MONTHLY.' + band.upper() + '.' + str(i+1).zfill(2) + '.' + month + '.hdf'
        filename = cwd + '02_STATS/MONTHLY.' + band.upper() + '.' + str(nmonths[i]).zfill(2) + '.' + month + '.hdf'
        print(f' {filename}')
        for var in vars:
            # Create the name of the dataset in the HDF
            var_name = band_num[j] + ' (' + band + ') ' + var
            if band_num[j] == '':
                var_name = band.upper() + ' ' + var
            print(f'  Layer: {layer} Retrieving: {var} Dataset: {var_name}')
            bands_array[:,:,layer] = rs.read_from_hdf(filename, var_name)
            layer += 1

# Retrieve phenology data
print(fn_phenology)
for param in phen:
    print(f' Layer: {layer} Retrieving: {param}')
    bands_array[:,:,layer] = rs.read_from_hdf(fn_phenology, param)
    layer += 1

print(fn_phenology2)
for param in phen2:
    print(f' Layer: {layer} Retrieving: {param}')
    bands_array[:,:,layer] = rs.read_from_hdf(fn_phenology2, param)
    layer += 1

# print(f'Bands array shape={bands_array.shape}')