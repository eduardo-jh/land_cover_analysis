#!/usr/bin/env python
# coding: utf-8

""" 02_dataset.py: Prepare the data set to use with machine learning.
Reads the raster with the land cover classes (labels) and the spectral bands and phenology
(features or predictors), then prepares the datasets for the ML algorithms. 

Eduardo Jimenez <eduardojh@email.arizona.edu>

  Feb 24, 2023: initial code.
  Mar 14, 2023: create a (very LARGE) HDF5 file to hold all features/predictors.

NOTE: run under 'rsml' conda environment (python 3.8.13, scikit-learn 1.1.2)
"""

import sys
import os.path
import platform
import h5py
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

NA_VALUE = -13000  # This should match the source's NA
fmt = '%Y_%m_%d-%H_%M_%S'
start = datetime.now()

### 1. CONFIGURE
# Projection to create raster. SJR: 32612=WGS 84 / UTM zone 12N; CBR: 32616=WGS 84 / UTM zone 16N
epsg_proj = 32616

# Paths and file names for the current ROI
fn_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY.tif'        # Land cover raster
fn_train_mask = cwd + 'training/usv250s7cw_ROI1_train_mask.tif'
fn_train_labels = cwd + 'training/usv250s7cw_ROI1_train_labels.tif'
fn_phenology = cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf'  # Phenology files
fn_phenology2 = cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf'
fn_features = cwd + 'Calakmul_Features.h5'
fn_train_feat = cwd + 'Calakmul_Training_Features.h5'
fn_test_feat = cwd + 'Calakmul_Testing_Features.h5'

### 2. READ TRAINING MASK
# Read a raster with the location of the training sites
print(f"File not found: {fn_train_mask}" if not os.path.isfile(fn_train_mask) else "")
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

### 3. READ LAND COVER LABELS
# Read the land cover raster and retrive the land cover classes
print(f"File not found: {fn_landcover}" if not os.path.isfile(fn_landcover) else "")
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

print('Analyzing labels from training dataset (land cover classes))')
lc_arr = lc_arr.astype(train_mask.dtype)
train_arr = np.where(train_mask > 0, lc_arr, 0)  # Actual labels (land cover classs)
# Save a raster with the actual labels (land cover classes) from the mask
rs.create_raster(fn_train_labels, train_arr, epsg_proj, lc_gt)

print(f'train_mask: {train_mask.dtype}, unique:{np.unique(train_mask.filled(0))}, {train_mask.shape}')
print(f'lc_arr    : {lc_arr.dtype}, unique:{np.unique(lc_arr.filled(0))}, {lc_arr.shape}')
print(f'train_arr : {train_arr.dtype}, unique:{np.unique(train_arr)}, {train_arr.shape}')

train_labels = lc_arr[train_mask > 0]  # This array gets flatten
print(train_labels.shape)

### 4. FEATURES: spectral bands, vegetation indices, and phenologic parameters
# All features (alphabetic)
bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
nmonths = [x for x in range(1, 13)]
vars = ['NPixels', 'MIN', 'MAX', 'AVG', 'STDEV']
phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

# # To test a small subset
# bands = ['Blue', 'Green']
# band_num = ['B2', 'B3',]
# months = ['JAN']
# nmonths = [1]
# vars = ['AVG']
# phen = ['SOS', 'EOS']
# phen2 = ['SOS2', 'EOS2']

# Calculate the dimensions of the array
cols = train_mask.shape[1]
rows = train_mask.shape[0]
lyrs = len(bands) * len(months) * len(vars) + len(phen) + len(phen2)
print(f'Rows={rows}, Cols={cols}, Layers={lyrs}')

### 5. CREATE A (LARGE) HDF5 FILE TO HOLD ALL FEATURES
f = h5py.File(fn_features, 'w')
f_train = h5py.File(fn_train_feat, 'w')
f_test = h5py.File(fn_test_feat, 'w')

layer = 0
file_exist = 0
file_missing = 0
for j, band in enumerate(bands):
    print(f'{band.upper()}')
    for i, month in enumerate(months):
        filename = cwd + '02_STATS/MONTHLY.' + band.upper() + '.' + str(nmonths[i]).zfill(2) + '.' + month + '.hdf'
        exist = os.path.isfile(filename)
        print(f' {filename} {str(exist)}')
        if exist:
            file_exist += 1
        else:
            file_missing += 1
        for var in vars:
            # Create the name of the dataset in the HDF
            var_name = band_num[j] + ' (' + band + ') ' + var
            if band_num[j] == '':
                var_name = band.upper() + ' ' + var
            print(f'  Layer: {layer} Variable: {var} Dataset: {var_name}')

            # Extract data and filter by training mask
            feat_arr = rs.read_from_hdf(filename, var_name)
            train_arr = np.where(train_mask > 0, feat_arr, NA_VALUE)
            test_arr = np.where(train_mask == 0, feat_arr, NA_VALUE)
            
            # Separate training and testing features
            f.create_dataset(month + ' ' + var_name, (rows, cols), data=feat_arr)
            f_train.create_dataset(month + ' ' + var_name, (rows, cols), data=train_arr)
            f_test.create_dataset(month + ' ' + var_name, (rows, cols), data=test_arr)
            layer += 1
f.close()
print(f'Existing files: {file_exist}, Missing: {file_missing}, Total: {file_exist+file_missing}')

with h5py.File(fn_features, 'a') as f, h5py.File(fn_train_feat, 'a') as f_train, h5py.File(fn_test_feat, 'a') as f_test:
    # Retrieve phenology data
    print(fn_phenology)
    for param in phen:
        print(f' Layer: {layer} Variable: {param}')

        # Extract data and filter by training mask
        pheno_arr = rs.read_from_hdf(fn_phenology, param)
        train_arr = np.where(train_mask > 0, pheno_arr, NA_VALUE)
        test_arr = np.where(train_mask == 0, pheno_arr, NA_VALUE)

        # Separate training and testing features
        f.create_dataset('PHEN ' + param, (rows, cols), data=pheno_arr)
        f_train.create_dataset('PHEN ' + param, (rows, cols), data=train_arr)
        f_test.create_dataset('PHEN ' + param, (rows, cols), data=test_arr)
        
        layer += 1

    print(fn_phenology2)
    for param in phen2:
        print(f' Layer: {layer} Variable: {param}')
        # Extract data and filter by training mask
        pheno_arr = rs.read_from_hdf(fn_phenology2, param)
        train_arr = np.where(train_mask > 0, pheno_arr, NA_VALUE)
        test_arr = np.where(train_mask == 0, pheno_arr, NA_VALUE)

        # Separate training and testing features
        f.create_dataset('PHEN ' + param, (rows, cols), data=pheno_arr)
        f_train.create_dataset('PHEN ' + param, (rows, cols), data=train_arr)
        f_test.create_dataset('PHEN ' + param, (rows, cols), data=test_arr)
        
        layer += 1
print(f'Added {layer} layers to the file.')  # stars at 0 but adds 1 at the end, so layer count is OK

print(f'Features file created successfully: {fn_features} ')