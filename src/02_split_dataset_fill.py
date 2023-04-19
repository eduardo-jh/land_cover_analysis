#!/usr/bin/env python
# coding: utf-8

""" 02_split_dataset.py: Creates a dataset splitted into artificial images to use with machine learning.
Reads the raster with the land cover classes (labels) and the spectral bands and phenology
(features or predictors), then prepares the datasets for the ML algorithms. 

Eduardo Jimenez <eduardojh@email.arizona.edu>

  Feb 24, 2023: initial code.
  Mar 14, 2023: create a (very LARGE) HDF5 file to hold all features/predictors.
  Mar 30, 2023: split dataset into artificial images to feed CNN
  Apr 18, 2023: fill missing data when creating subsets

NOTE: run under 'rstf' conda environment (python 3.8.13, scikit-learn 1.1.2)
"""

import sys
import os.path
import platform
import h5py
import csv
import numpy as np
from math import ceil
from datetime import datetime
from scipy import stats

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

def fill_nans_mean(dataset, min_value, **kwargs):
    """ Fills NaNs using the mean of all valid data """
    # print(f'  Received: {dataset.shape} {dataset.min()} {dataset.max()} {min_value}')

    # Valid values are larger than minimum, otherwise are NaNs (e.g. -13000, -1, etc.)
    valid_ds = np.where(dataset >= min_value, dataset, np.nan)

    # Fill NaNs with the mean of valid data
    fill_value = round(np.nanmean(valid_ds), 2)
    filled_ds = np.where(dataset >= min_value, dataset, fill_value)

    print(f'  --Missing values filled with {fill_value} successfully ', end='')
    return filled_ds


def fill_season(sos, eos, los, min_value, **kwargs):
    """ Fills missing values from SOS, EOS and LOS """
    # _nan = kwargs.get('nan', -1)
    _max_row = kwargs.get('max_row', None)
    _max_col = kwargs.get('max_col', None)
    _verbose = kwargs.get('verbose', False)
    # _col_pixels = kwargs.get('col_pixels', 1000)
    _row_pixels = kwargs.get('row_pixels', 1000)
    _id = kwargs.get('id', '')
    
    ### SOS
    sos = sos.astype(int)
    valid_sos = np.where(sos < min_value, 1, 0)
    print(f'  --Missing data found at SOS: {np.sum(valid_sos)}')

    ### EOS
    eos = eos.astype(int)
    valid_eos = np.where(eos < min_value, 1, 0)
    print(f'  --Missing data found at EOS: {np.sum(valid_eos)}')

    ### LOS
    los = los.astype(int)
    valid_los = np.where(los < min_value, 1, 0)
    print(f'  --Missing data found at LOS: {np.sum(valid_eos)}')

    # Find indices of rows with NaNs
    loc_sos = {}
    loc_eos = {}
    loc_los = {}
    for i in range(_row_pixels):
        if np.sum(valid_sos[i]) > 0:
            # Find the indices of columns with NaNs, save them in their corresponding row
            cols = np.where(valid_sos[i] == 1)
            loc_sos[i] = cols[0].tolist()
        if np.sum(valid_eos[i]) > 0:
            cols = np.where(valid_eos[i] == 1)
            loc_eos[i] = cols[0].tolist()
        if np.sum(valid_los[i]) > 0:
            cols = np.where(valid_los[i] == 1)
            loc_los[i] = cols[0].tolist()

    filled_sos = sos[:]
    filled_eos = eos[:]
    filled_los = los[:]

    for row in loc_sos.keys():
        # Make sure the indices among SOS, EOS, and LOS are the same
        assert row in loc_eos.keys(), f"SOS key {row} not in EOS"
        assert row in loc_los.keys(), f"SOS key {row} not in LOS"
        for col in loc_sos[row]:
            assert col in loc_eos[row], f"Key {row} not in EOS"
            assert col in loc_los[row], f"SOS key {row} not in LOS"
            # print(f'  Location: {row}, {col}')
            val = sos[row, col]
            # print(val)
            
            win_size = 1
            removed_success = False
            while not removed_success:
                # Window to slice around the missing value
                row_start = row-win_size
                row_end = row+win_size+1
                col_start = col-win_size
                col_end = col+win_size+1
                # Adjust row,col to use for slicing when point near the edges
                if _max_row is not None:
                    if row_start < 0:
                        row_start = 0
                    if row_end > _max_row:
                        row_end = _max_row
                if _max_col is not None:
                    if col_start < 0:
                        col_start = 0
                    if col_end > _max_col:
                        col_end = _max_col
                
                # Slice a window of values around missing value
                window_sos = sos[row_start:row_end, col_start:col_end]
                window_eos = eos[row_start:row_end, col_start:col_end]

                values_sos = window_sos.flatten().tolist()
                values_eos = window_eos.flatten().tolist()

                # Remove NaN values from the list
                all_vals_sos = values_sos.copy()
                values_sos = [i for i in all_vals_sos if i != val]
                all_vals_eos = values_eos.copy()
                values_eos = [i for i in all_vals_eos if i != val]

                if len(values_sos) == len(values_eos) and len(values_eos) > 0:
                    removed_success = True
                    if row%100 == 0:
                        print(f'  {_id} -- Success with window size {win_size}. Row {row}')
                    break
                # assert len(values_sos) > 0, "Values SOS empty!"
                # assert len(values_eos) > 0, "Values EOS empty!"
                # If failure, increase window size and try again
                win_size += 1
            
            # For SOS use mode (will return minimum value as default)
            fill_value_sos = stats.mode(values_sos, keepdims=False)[0]

            # For EOS use mode or max value
            fill_value_eos = stats.mode(values_eos, keepdims=False)[0]
            if fill_value_eos == np.min(values_eos):
                # If default (minimum) return maximum value
                fill_value_eos = np.max(values_eos)
            
            # Fill value for LOS
            fill_value_los = fill_value_eos - fill_value_sos

            if _verbose:
                print(f'  --SOS: {row},{col}: {val}, values={values_sos}, fill_val={fill_value_sos}')
                print(f'  --EOS: {row},{col}: {val}, values={values_eos}, fill_val={fill_value_eos}')
                print(f'  --LOS: {row},{col}: {val}, fill_val={fill_value_los}')
            filled_sos[row, col] = fill_value_sos
            filled_eos[row, col] = fill_value_eos
            filled_los[row, col] = fill_value_los
    return filled_sos, filled_eos, filled_los

# NAN_VALUE = -32768 # Keep 16-bit integer, source's NA = -13000
NAN_VALUE = np.nan
fmt = '%Y_%m_%d-%H_%M_%S'
start = datetime.now()

### 1. CONFIGURE
# Projection to create raster. SJR: 32612=WGS 84 / UTM zone 12N; CBR: 32616=WGS 84 / UTM zone 16N
epsg_proj = 32616

# Paths and file names for the current ROI
fn_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY.tif'        # Land cover raster
fn_test_mask = cwd + 'training/usv250s7cw_ROI1_testing_mask.tif'
fn_test_labels = cwd + 'training/usv250s7cw_ROI1_testing_labels.tif'
fn_phenology = cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf'  # Phenology files
fn_phenology2 = cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf'
fn_features = cwd + 'IMG_Calakmul_Features_filled.h5'
fn_train_feat = cwd + 'IMG_Calakmul_Training_Features_filled.h5'
fn_test_feat = cwd + 'IMG_Calakmul_Testing_Features_filled.h5'
fn_labels = cwd + 'IMG_Calakmul_Labels.h5'
fn_parameters = cwd + 'dataset_parameters.csv'

### 2. READ TESTING MASK
# Read a raster with the location of the testing sites
print(f"File not found: {fn_test_mask}" if not os.path.isfile(fn_test_mask) else "")
test_mask, nodata, metadata, geotransform, projection = rs.open_raster(fn_test_mask)
print(f'Opening raster: {fn_test_mask}')
print(f'Metadata      : {metadata}')
print(f'NoData        : {nodata}')
print(f'Columns       : {test_mask.shape[1]}')
print(f'Rows          : {test_mask.shape[0]}')
print(f'Geotransform  : {geotransform}')
print(f'Projection    : {projection}')
print(f'Type          : {test_mask.dtype}')

# Find how many non-zero entries we have -- i.e. how many training and testing data samples?
n_samples = (test_mask > 0).sum()
print(f'Testing samples: {n_samples}')
n_samples = (test_mask < 1).sum()
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

print('Analyzing labels from testing dataset (land cover classes))')
lc_arr = lc_arr.astype(test_mask.dtype)
train_arr = np.where(test_mask > 0, lc_arr, 0)  # Actual labels (land cover class)
# Save a raster with the actual labels (land cover classes) from the mask
rs.create_raster(fn_test_labels, train_arr, epsg_proj, lc_gt)

print(f'test_mask: {test_mask.dtype}, unique:{np.unique(test_mask.filled(0))}, {test_mask.shape}')
print(f'lc_arr    : {lc_arr.dtype}, unique:{np.unique(lc_arr.filled(0))}, {lc_arr.shape}')
print(f'train_arr : {train_arr.dtype}, unique:{np.unique(train_arr)}, {train_arr.shape}')

print(f'Land cover array: {lc_arr.shape}')
# train_lbl = np.where(test_mask < 0.5, lc_arr, NAN_VALUE)
# test_lbl = np.where(test_mask > 0.5, lc_arr, NAN_VALUE)
train_lbl = np.where(test_mask < 0.5, lc_arr, 0)
test_lbl = np.where(test_mask > 0.5, lc_arr, 0)

# with h5py.File(fn_labels, 'w') as f:
#     train_lbl = np.where(test_mask < 0.5, lc_arr, NAN_VALUE)
#     test_lbl = np.where(test_mask > 0.5, lc_arr, NAN_VALUE)
            
#     # Separate training and testing features
#     f.create_dataset('train', lc_arr.shape, data=train_lbl)
#     f.create_dataset('test', lc_arr.shape, data=test_lbl)
    
# train_labels = lc_arr[test_mask > 0]  # This array gets flatten
# print(train_labels.shape)

### 4. FEATURES: spectral bands, vegetation indices, and phenologic parameters
# # All features (alphabetic)
# bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
# band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
# months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
# nmonths = [x for x in range(1, 13)]
# vars = ['NPixels', 'MIN', 'MAX', 'AVG', 'STDEV']
# phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
# phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

# To test a small subset
bands = ['Blue', 'Green', 'Nir', 'Red']
band_num = ['B2', 'B3', 'B5', 'B4']
months = ['JAN']
nmonths = [1]
vars = ['AVG']
phen = ['SOS', 'EOS']
# phen2 = ['SOS2', 'EOS2']
phen2 = []

# # Test a "reasonable" subset
# bands = ['Blue', 'Green', 'Ndvi', 'Nir', 'Red', 'Swir1']
# band_num = ['B2', 'B3', '', 'B5', 'B4', 'B6']
# months = ['MAR', 'JUN', 'SEP', 'DEC']
# nmonths = [3, 6, 9, 12]
# vars = ['AVG', 'STDEV']
# phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR']
# phen2 = ['SOS2', 'EOS2', 'LOS2']

# Calculate the dimensions of the array
arr_cols = test_mask.shape[1]
arr_rows = test_mask.shape[0]
lyrs = len(bands) * len(months) * len(vars) + len(phen) + len(phen2)
print(f'Rows={arr_rows}, Cols={arr_cols}, Layers={lyrs}')

features_end = True
rows, cols = 1000, 1000  # rows, cols of the artificial images
img_x_col = ceil(arr_cols/cols)
img_x_row = ceil(arr_rows/rows)

### 5. CREATE (LARGE) HDF5 FILES TO HOLD ALL FEATURES
f = h5py.File(fn_features, 'w')
f_train = h5py.File(fn_train_feat, 'w')
f_test = h5py.File(fn_test_feat, 'w')
f_labels = h5py.File(fn_labels, 'w')
f_labels.create_group('training')
f_labels.create_group('testing')

feat_indices = []
feat_names = []
images = 0

# Save SOS and EOS to compute LOS during filling missing data
# sos = np.empty((arr_rows, arr_cols), dtype=int)
# eos = np.empty((arr_rows, arr_cols), dtype=int)
# los = np.empty((arr_rows, arr_cols), dtype=int)

# TODO: Improve this, is very inefficient to process everytime all the raster/array and then
# only select a subset to create a fake image, then repeat all processing for next image!
for r in range(img_x_row):
    for c in range(img_x_col):
        print(f'\n === IMAGE {images} === ')
        feature = 0

        # Indices to slice array
        r_str = r * rows
        r_end = r_str + rows
        c_str = c * cols
        c_end = c_str + cols

        # Adjust final row/col to slice
        if r_end > arr_rows:
            r_end = arr_rows
        if c_end > arr_cols:
            c_end =  arr_cols

        print(f'Slicing from row={r_str}:{r_end}, col={c_str}:{c_end}, feat={feature}')

        # Arrays to hold the feature data
        if features_end:
            all_feat_img = np.empty((rows, cols, lyrs))
            training_img = np.empty((rows, cols, lyrs))
            testing_img = np.empty((rows, cols, lyrs))
        else:
            all_feat_img = np.empty((lyrs, rows, cols))
            training_img = np.empty((lyrs, rows, cols))
            testing_img = np.empty((lyrs, rows, cols))

        all_feat_img[:] = np.nan
        training_img[:] = np.nan
        testing_img[:] = np.nan

        # Arrays to hold the label (land cover classes) data
        train_lbl_img = np.zeros((rows, cols), dtype=np.uint8)
        test_lbl_img = np.zeros((rows, cols), dtype=np.uint8)

        for j, band in enumerate(bands):
            # print(f'{band.upper()}')
            for i, month in enumerate(months):
                filename = cwd + '02_STATS/MONTHLY.' + band.upper() + '.' + str(nmonths[i]).zfill(2) + '.' + month + '.hdf'
                for var in vars:
                    # Create the name of the dataset in the HDF
                    feat_name = band_num[j] + ' (' + band + ') ' + var
                    if band_num[j] == '':
                        feat_name = band.upper() + ' ' + var
                    print(f'  Feature: {feature:>4} Month: {month:>4} Variable: {var:>8} Dataset: {feat_name:16}')

                    # Extract data and filter by training mask
                    feat_arr = rs.read_from_hdf(filename, feat_name)  # Use HDF4 method

                    ### Fill missing data
                    minimum = 0
                    max_row, max_col = None, None
                    if band.upper() in ['NDVI', 'EVI', 'EVI2']:
                        minimum = -10000
                    # # Adjust the last row and column 'fake' images
                    # if r == (img_x_row-1):
                    #     max_row = r_end-r_str
                    #     print(f'  --max_row: {max_row}')
                    # if c == (img_x_col-1):
                    #     max_col = c_end-c_str
                    #     print(f'  --max_col: {max_col}')
                    feat_arr = fill_nans_mean(feat_arr, minimum)
                    print(f'for {band.upper()}.')
                    # print(f'  Back: {feat_arr.min()} {feat_arr.max()} {minimum}')

                    # print(f'    test_mask: {test_mask.dtype}, unique:{np.unique(test_mask.filled(0))}, {test_mask.shape}')
                    # print(f'    feat_arr: {type(feat_arr)} {feat_arr.dtype}, {feat_arr.shape}')
                    train_arr = np.where(test_mask < 0.5, feat_arr, NAN_VALUE)
                    test_arr = np.where(test_mask > 0.5, feat_arr, NAN_VALUE)

                    # Slice the array of labels
                    # print(f'{train_lbl_img[:r_end-r_str,:c_end-c_str].shape} {train_lbl[r_str:r_end,c_str:c_end].shape}')
                    train_lbl_img[:r_end-r_str,:c_end-c_str] = train_lbl[r_str:r_end,c_str:c_end]
                    test_lbl_img[:r_end-r_str,:c_end-c_str] = test_lbl[r_str:r_end,c_str:c_end]
                    
                    # Slice the array of features, separate training and testing features
                    if features_end:
                        # Features at the end
                        # print(f'  Src={train_arr[r_str:r_end,c_str:c_end].shape} Dest={training_img[:r_end-r_str,:c_end-c_str,feature].shape}')
                        all_feat_img[:r_end-r_str,:c_end-c_str,feature] = feat_arr[r_str:r_end,c_str:c_end]
                        training_img[:r_end-r_str,:c_end-c_str,feature] = train_arr[r_str:r_end,c_str:c_end]
                        testing_img[:r_end-r_str,:c_end-c_str,feature] = test_arr[r_str:r_end,c_str:c_end]
                    else:
                        # Features first
                        # print(f'  Src={train_arr[r_str:r_end,c_str:c_end].shape} Dest={training_img[feature,:r_end-r_str,:c_end-c_str].shape}')
                        all_feat_img[feature,:r_end-r_str,:c_end-c_str] = feat_arr[r_str:r_end,c_str:c_end]
                        training_img[feature,:r_end-r_str,:c_end-c_str] = train_arr[r_str:r_end,c_str:c_end]
                        testing_img[feature,:r_end-r_str,:c_end-c_str] = test_arr[r_str:r_end,c_str:c_end]

                    if images == 0:
                        feat_names.append(feat_name)
                        feat_indices.append(feature)
                    feature += 1
        
        # Add phenology
        for param in phen:
            print(f'  Feature: {feature} Variable: {param}')

            # Fill missing data
            if param == 'SOS':
                minimum = 0
                sos = rs.read_from_hdf(fn_phenology, 'SOS')
                eos = rs.read_from_hdf(fn_phenology, 'EOS')
                los = rs.read_from_hdf(fn_phenology, 'LOS')
                filled_sos, filled_eos, filled_los =  fill_season(sos, eos, los, minimum,
                                                                  row_pixels=arr_rows,
                                                                  max_row=arr_rows,
                                                                  max_col=arr_cols,
                                                                  id=param + '' + str(images).zfill(2))

                pheno_arr = filled_sos[:]
            elif param == 'EOS':
                pheno_arr = filled_eos[:]
            elif param == 'LOS':
                pheno_arr = filled_los[:]
            else:
                # Extract data and filter by training mask
                pheno_arr = rs.read_from_hdf(fn_phenology, param)  # Use HDF4 method
            
            train_arr = np.where(test_mask < 0.5, pheno_arr, NAN_VALUE)
            test_arr = np.where(test_mask > 0.5, pheno_arr, NAN_VALUE)

            # Separate training and testing features 
            if features_end:
                # Features at the end
                all_feat_img[:r_end-r_str,:c_end-c_str,feature] = pheno_arr[r_str:r_end,c_str:c_end]
                training_img[:r_end-r_str,:c_end-c_str,feature] = train_arr[r_str:r_end,c_str:c_end]
                testing_img[:r_end-r_str,:c_end-c_str,feature] = test_arr[r_str:r_end,c_str:c_end]
            else:
                # Features first
                all_feat_img[feature,:r_end-r_str,:c_end-c_str] = pheno_arr[r_str:r_end,c_str:c_end]
                training_img[feature,:r_end-r_str,:c_end-c_str] = train_arr[r_str:r_end,c_str:c_end]
                testing_img[feature,:r_end-r_str,:c_end-c_str] = test_arr[r_str:r_end,c_str:c_end]

            if images == 0:
                feat_names.append('PHEN ' + param)
                feat_indices.append(feature)
            feature += 1

        # Add phenology from second file
        for param in phen2:
            print(f'  Feature: {feature} Variable: {param}')

            # Extract data and filter by training mask
            # pheno_arr = rs.read_from_hdf(fn_phenology2, param)  # Use HDF4 method
            if param == 'SOS2':
                minimum = 0
                sos = rs.read_from_hdf(fn_phenology2, 'SOS2')
                eos = rs.read_from_hdf(fn_phenology2, 'EOS2')
                los = rs.read_from_hdf(fn_phenology2, 'LOS2')
                filled_sos, filled_eos, filled_los =  fill_season(sos, eos, los, minimum,
                                                                  row_pixels=arr_rows,
                                                                  max_row=arr_rows,
                                                                  max_col=arr_cols,
                                                                  id=param + '' + str(images).zfill(2))

                pheno_arr = filled_sos[:]
            elif param == 'EOS2':
                pheno_arr = filled_eos[:]
            elif param == 'LOS2':
                pheno_arr = filled_los[:]
            else:
                # Extract data and filter by training mask
                pheno_arr = rs.read_from_hdf(fn_phenology, param)  # Use HDF4 method
            
            train_arr = np.where(test_mask < 0.5, pheno_arr, NAN_VALUE)
            test_arr = np.where(test_mask > 0.5, pheno_arr, NAN_VALUE)

            # Separate training and testing features
            if features_end:
                # Features at the end
                all_feat_img[:r_end-r_str,:c_end-c_str,feature] = pheno_arr[r_str:r_end,c_str:c_end]
                training_img[:r_end-r_str,:c_end-c_str,feature] = train_arr[r_str:r_end,c_str:c_end]
                testing_img[:r_end-r_str,:c_end-c_str,feature] = test_arr[r_str:r_end,c_str:c_end]
            else:
                # Features first
                all_feat_img[feature,:r_end-r_str,:c_end-c_str] = pheno_arr[r_str:r_end,c_str:c_end]
                training_img[feature,:r_end-r_str,:c_end-c_str] = train_arr[r_str:r_end,c_str:c_end]
                testing_img[feature,:r_end-r_str,:c_end-c_str] = test_arr[r_str:r_end,c_str:c_end]
        
            if images == 0:
                feat_names.append('PHEN ' + param)
                feat_indices.append(feature)
            feature += 1

        # Once all features (bands, VIs, and phenology metrics) are added, create a subset (a "fake" image)
        # Check if features should be place at the end
        img_name = 'r' + str(r) + 'c' + str(c)
        if features_end:
            f.create_dataset(img_name, (rows, cols, lyrs), data=all_feat_img)
            f_train.create_dataset(img_name, (rows, cols, lyrs), data=training_img)
            f_test.create_dataset(img_name, (rows, cols, lyrs), data=testing_img)
        else:
            f.create_dataset(img_name, (lyrs, rows, cols), data=all_feat_img)
            f_train.create_dataset(img_name, (lyrs, rows, cols), data=training_img)
            f_test.create_dataset(img_name, (lyrs, rows, cols), data=testing_img)
        
        # Separate training and testing labels
        training_grp = f_labels.require_group('training')
        training_grp.create_dataset(img_name, (rows, cols), data=train_lbl_img)
        testing_grp = f_labels.require_group('testing')
        testing_grp.create_dataset(img_name, (rows, cols), data=test_lbl_img)
        images += 1

        print(f'  Image={training_img.shape}, ({rows}, {cols}, {lyrs})')
        print(f'Image {images} of {img_x_col*img_x_row} created with {lyrs} features (layers).\n')  # stars at 0 but adds 1 at the end, so layer count is OK

# Save a file with the parameters used
with open(fn_parameters, 'w', newline='') as csv_file:
     writer = csv.writer(csv_file, delimiter='=')
     writer.writerow(['NAN_VALUE', NAN_VALUE])
     writer.writerow(['EPSG', epsg_proj])
     writer.writerow(['BANDS', ','.join(bands)])
     writer.writerow(['BANDS_NUM', ','.join(band_num)])
     writer.writerow(['MONTHS', ','.join(months)])
     writer.writerow(['MONTHS_NUM', ','.join([str(x) for x in nmonths])])
     writer.writerow(['VARIABLES', ','.join(vars)])
     writer.writerow(['PHENOLOGY', ','.join(phen)])
     writer.writerow(['PHENOLOGY2', ','.join(phen2)])
     writer.writerow(['ROWS', arr_rows])
     writer.writerow(['COLUMNS', arr_cols])
     writer.writerow(['LAYERS', lyrs])
     writer.writerow(['IMG_ROWS', rows])
     writer.writerow(['IMG_COLUMNS', cols])
     writer.writerow(['IMG_PER_COL', img_x_col])
     writer.writerow(['IMG_PER_ROW', img_x_row])
     writer.writerow(['NUM_CLASSES', len(np.unique(lc_arr.filled(0)))])

with open(cwd + 'feature_indices.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for i, feat in zip(feat_indices, feat_names):
        writer.writerow([i, feat])
# print(f'Added {feature} features (layers) to the file.')
