#!/usr/bin/env python
# coding: utf-8

""" 02_dataset.py: Reads the raster with the land cover classes (labels) and
the spectral bands and phenology (features or predictors), then prepares the
datasets for the ML algorithms. 

Eduardo Jimenez <eduardojh@email.arizona.edu>

  Feb 24, 2023: initial code.
  Mar 14, 2023: create a (very LARGE) HDF5 file to hold all features/predictors
  Mar 30, 2023: split dataset into artificial images to feed CNN
  Apr 18, 2023: fill missing data when creating subsets
  May 22, 2023: splitting moved out, added normalization and standardization

NOTE: run under 'rstf' conda environment (python 3.8.13, scikit-learn 1.1.2)
"""

import sys
import os.path
import h5py
import csv
import numpy as np
from scipy import stats
from typing import Tuple

if len(sys.argv) == 3:
    # Check if arguments were passed from terminal
    args = sys.argv[1:]
    sys.path.insert(0, args[0])
    cwd = args[1]
    data_subdir = '../CALAKMUL/ROI1/'
    print(f"  Using RS_LIB={args[0]}")
    print(f"  Using CWD={args[1]}")
else:
    import platform
    system = platform.system()
    if system == 'Windows':
        # On Windows 10
        sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib/')
        cwd = 'D:/Desktop/CALAKMUL/ROI1/'
        data_subdir = 'data/landsat/C2/'
    elif system == 'Linux' and os.path.isdir('/vipdata/2023/CALAKMUL/ROI1/'):
        # On Ubuntu Workstation
        sys.path.insert(0, '/vipdata/2023/land_cover_analysis/lib/')
        cwd = '/vipdata/2023/CALAKMUL/ROI1/'
        data_subdir = 'data/landsat/C2/'
    elif system == 'Linux' and os.path.isdir('/VIP/engr-didan02s/DATA/EDUARDO/ML/'):
        # On Alma Linux Server
        sys.path.insert(0, '/home/eduardojh/Documents/land_cover_analysis/lib/')
        cwd = '/VIP/engr-didan02s/DATA/EDUARDO/ML/'
        data_subdir = '../CALAKMUL/ROI1/'
    else:
        print('  System not yet configured!')

import rsmodule as rs

def fill_with_mean(dataset: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
    """ Fills NaNs using the mean of all valid data """
    _verbose = kwargs.get('verbose', False)
    _var = kwargs.get('var', '')

    # Valid values are larger than minimum, otherwise are NaNs (e.g. -13000, -1, etc.)
    valid_ds = np.where(dataset >= min_value, dataset, np.nan)

    # Fill NaNs with the mean of valid data
    fill_value = round(np.nanmean(valid_ds), 2)
    filled_ds = np.where(dataset >= min_value, dataset, fill_value)

    if _verbose:
        print(f'  --Missing {_var} values filled with {fill_value} successfully!')
    return filled_ds


def fill_with_int_mean(dataset: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
    """ Fills NaNs using the mean of all valid data """
    _verbose = kwargs.get('verbose', False)
    _var = kwargs.get('var', '')

    # Valid values are larger than minimum, otherwise are NaNs (e.g. -13000, -1, etc.)
    valid_ds = np.where(dataset >= min_value, dataset, np.nan)

    # Fill NaNs with the mean of valid data
    fill_value = int(np.nanmean(valid_ds))
    filled_ds = np.where(dataset >= min_value, dataset, fill_value)

    if _verbose:
        print(f'  --Missing {_var} values filled with {fill_value} successfully!')
    return filled_ds


def fill_season(sos: np.ndarray, eos: np.ndarray, los: np.ndarray, min_value: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Fills missing values from SOS, EOS and LOS 

    NOTICE: This is a window-based method and needs to know the number of rows and columns to work properly.
    """
    # _nan = kwargs.get('nan', -1)
    _max_row = kwargs.get('max_row', None)
    _max_col = kwargs.get('max_col', None)
    _verbose = kwargs.get('verbose', False)
    # _col_pixels = kwargs.get('col_pixels', 1000)
    _row_pixels = kwargs.get('row_pixels', 1000)
    _id = kwargs.get('id', '')
    
    ### SOS
    # sos = sos.astype(int)
    sos_nan_indices = np.transpose((sos<min_value).nonzero())  # get NaN indices
    if _verbose:
        print(f'  --Missing data found at SOS: {len(sos_nan_indices)}')

    ### EOS
    # eos = eos.astype(int)
    eos_nan_indices = np.transpose((eos<min_value).nonzero())
    if _verbose:
        print(f'  --Missing data found at EOS: {len(eos_nan_indices)}')

    ### LOS
    # los = los.astype(int)
    los_nan_indices = np.transpose((los<min_value).nonzero())
    if _verbose:
        print(f'  --Missing data found at LOS: {len(eos_nan_indices)}')

    # Temporary array to contain fill values in their right position
    filled_sos = sos.copy()
    filled_eos = eos.copy()
    filled_los = los.copy()

    assert sos_nan_indices.shape == eos_nan_indices.shape, f"NaN indices different shape SOS={sos_nan_indices.shape} EOS={eos_nan_indices.shape}"
    assert los_nan_indices.shape == eos_nan_indices.shape, f"NaN indices different shape LOS={los_nan_indices.shape} EOS={eos_nan_indices.shape}"

    # Each NaN position contains a [row, col]
    for sos_pos, eos_pos, los_pos in zip(sos_nan_indices, eos_nan_indices, los_nan_indices):
        assert np.array_equal(sos_pos, eos_pos), f"NaN positions are different SOS={sos_pos} EOS={eos_pos}"
        assert np.array_equal(sos_pos, los_pos), f"NaN positions are different SOS={sos_pos} LOS={los_pos}"

        row, col = sos_pos
        nan_value = sos[row, col]  # current position of NaN value
        # print(nan_value)
        
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

            win_values_sos = window_sos.flatten().tolist()
            win_values_eos = window_eos.flatten().tolist()

            # Remove NaN values from the list
            all_vals_sos = win_values_sos.copy()
            # Keep all values but the one at the center of window, aka the NaN value
            win_values_sos = [i for i in all_vals_sos if i != nan_value]
            all_vals_eos = win_values_eos.copy()
            win_values_eos = [i for i in all_vals_eos if i != nan_value]

            # If list is empty, it means window had only missing values, increase window
            if len(win_values_sos) == len(win_values_eos) and len(win_values_eos) > 0:
                # List is not empty, non NaN values found!
                removed_success = True
                if _verbose:
                    print(f'  -- {_id}: Success with window size {win_size}. ({row},{col})')
                break
            # If failure, increase window size and try again
            win_size += 1
        
        # For SOS use mode (will return minimum value as default)
        fill_value_sos = stats.mode(win_values_sos, keepdims=False)[0]
        if _verbose:
            print(f'  -- Fill SOS value={fill_value_sos}')

        # For EOS use aither mode or max value
        # fill_value_eos, counts = stats.mode(win_values_eos, keepdims=False)[0]
        fill_value_eos, count = stats.mode(win_values_eos, keepdims=False)
        if fill_value_eos == np.min(win_values_eos) and count == 1:
            if _verbose:
                print(f"  -- Fill EOS value={fill_value_eos} w/count={count} isn't a true mode, use maximum instead.")
            # If default (minimum) return maximum value
            fill_value_eos = np.max(win_values_eos)
        
        # Fill value for LOS
        fill_value_los = fill_value_eos - fill_value_sos
        if fill_value_los <= 0:
            fill_value_los = 365  # assume LOS for the entire year

        if _verbose:
            print(f'  --SOS: {row},{col}: {nan_value}, values={win_values_sos}, fill_val={fill_value_sos}')
            print(f'  --EOS: {row},{col}: {nan_value}, values={win_values_eos}, fill_val={fill_value_eos}')
            print(f'  --LOS: {row},{col}: {nan_value}, fill_val={fill_value_los}\n')
        
        # Fill the missing values in their right position
        filled_sos[row, col] = fill_value_sos
        filled_eos[row, col] = fill_value_eos
        filled_los[row, col] = fill_value_los
    
    return filled_sos, filled_eos, filled_los


def fill_with_mode(data: np.ndarray, min_value: int, **kwargs) -> np.ndarray:
    """ Fills missing values with the mode from the surrounding window
    
    NOTICE: This is a window-based method and needs to know the number of rows and columns to work properly.
    """
    _max_row = kwargs.get('max_row', None)
    _max_col = kwargs.get('max_col', None)
    _verbose = kwargs.get('verbose', False)
    _row_pixels = kwargs.get('row_pixels', 1000)
    _id = kwargs.get('id', '')
    
    data = data.astype(int)
    data_nan_indices = np.where(data < min_value, 1, 0)  # get NaN indices
    if _verbose:
        print(f'  --Missing data found: {np.sum(data_nan_indices)}')

    # Find indices of rows with NaNs
    nan_loc = {}
    for i in range(_row_pixels):
        if np.sum(data_nan_indices[i]) > 0:
            # Find the indices of columns with NaNs, save them in their corresponding row
            cols = np.where(data_nan_indices[i] == 1)
            nan_loc[i] = cols[0].tolist()
        
    # filled_data = data[:]

    fill_values = np.empty(data.shape)
    fill_values[:] = np.nan
    
    for row in nan_loc.keys():
        for col in nan_loc[row]:

            nan_value = data[row, col]  # value of the missing data
            
            # Get a window around the missing data pixel
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
                window_data = data[row_start:row_end, col_start:col_end]
                win_values = window_data.flatten().tolist()

                # Remove NaN values from the list
                all_values = win_values.copy()
                win_values = [i for i in all_values if i != nan_value]
                # If the list is not empty, then NaNs were removed successfully!
                if len(win_values) > 0:
                    removed_success = True
                    if _verbose:
                        print(f'  -- {_id}: Success with window size {win_size}. ({row},{col})')
                    break

                # If failure, increase window size and try again
                win_size += 1
            
            # Use mode as fill value (will return minimum value as default)
            fill_value = stats.mode(win_values, keepdims=False)[0]
            if fill_value == np.min(win_values):
                # If default (minimum) means not mode was found, return mean value instead
                fill_value = int(np.nanmean(win_values))
            if _verbose:
                print(f'  -- Fill value: {fill_value}')
            fill_values[row, col] = fill_value
    
    # Fill the missing values in their right position
    filled_data = np.where(data_nan_indices == 1, fill_values, data)

    return filled_data


### ------ End of functions, start main code -----

NAN_VALUE = 0  # np.nan
FILL, NORMALIZE, STANDARDIZE = True, False, False  # Either normalize or standardize, not both!

### 1. CONFIGURE
# Paths and file names for the current ROI
# fn_landcover = cwd + 'data/inegi_2018/usv250s7cw_ROI1_LC_KEY.tif'        # Land cover raster
fn_landcover = cwd + 'data/inegi_2018/land_cover_ROI1.tif'      # Groups of land cover classes w/ ancillary
fn_test_mask = cwd + 'sampling/ROI1_testing_mask.tif'
fn_phenology = cwd + data_subdir +'03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf'  # Phenology files
fn_phenology2 = cwd + data_subdir + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf'

# Create files to save features
fn_features = cwd + 'features/Calakmul_Features.h5'
fn_train_feat = cwd + 'features/Calakmul_Training_Features.h5'
fn_test_feat = cwd + 'features/Calakmul_Testing_Features.h5'
fn_labels = cwd + 'features/Calakmul_Labels.h5'

# Create files to save parameters
fn_parameters = cwd + 'features/dataset_parameters.csv'
fn_feat_indices = cwd + 'features/feature_indices.csv'

### 2. READ TESTING MASK
# Read a raster with the location of the testing sites
assert os.path.isfile(fn_test_mask) is True, f"ERROR: File not found! {fn_test_mask}"
test_mask, nodata, metadata, geotransform, projection, epsg = rs.open_raster(fn_test_mask)
print(f'  Opening raster: {fn_test_mask}')
print(f'  --Metadata      : {metadata}')
print(f'  --NoData        : {nodata}')
print(f'  --Columns       : {test_mask.shape[1]}')
print(f'  --Rows          : {test_mask.shape[0]}')
print(f'  --Geotransform  : {geotransform}')
print(f'  --Projection    : {projection}')
print(f'  --EPSG          : {epsg}')
print(f'  --Type          : {test_mask.dtype}')

# Find how many non-zero entries we have -- i.e. how many training and testing data samples?
n_samples = (test_mask > 0).sum()
print(f'  --Testing samples: {n_samples}')
n_samples = (test_mask < 1).sum()
print(f'  --Training samples: {n_samples}')

### 3. READ LAND COVER LABELS
# Read the land cover raster and retrive the land cover classes
assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
lc_arr, lc_nd, lc_md, lc_gt, lc_proj, lc_epsg = rs.open_raster(fn_landcover)
print(f'  Opening raster: {fn_landcover}')
print(f'  --Metadata      : {lc_md}')
print(f'  --NoData        : {lc_nd}')
print(f'  --Columns       : {lc_arr.shape[1]}')
print(f'  --Rows          : {lc_arr.shape[0]}')
print(f'  --Geotransform  : {lc_gt}')
print(f'  --Projection    : {lc_proj}')
print(f'  --EPSG          : {lc_epsg}')
print(f'  --Type          : {lc_arr.dtype}')

# # Mask out the 'NoData' pixels to match Landsat data and land cover classes
# dummy_array, _, _, _, _ = open_raster(fn_nodata_mask)
# lc_arr = np.ma.masked_array(lc_arr, mask=np.ma.getmask(dummy_array))
lc_arr = lc_arr.astype(int)

print('  Analyzing labels from testing dataset (land cover classes))')
lc_arr = lc_arr.astype(test_mask.dtype)
train_arr = np.where(test_mask > 0, lc_arr, 0)  # Actual labels (land cover classes)

print(f'  --test_mask: {test_mask.dtype}, unique:{np.unique(test_mask.filled(0))}, {test_mask.shape}')
print(f'  --lc_arr   : {lc_arr.dtype}, unique:{np.unique(lc_arr.filled(0))}, {lc_arr.shape}')
print(f'  --train_arr: {train_arr.dtype}, unique:{np.unique(train_arr)}, {train_arr.shape}')

print(f'  --Land cover array: {lc_arr.shape}')
train_lbl = np.where(test_mask < 0.5, lc_arr, NAN_VALUE)
test_lbl = np.where(test_mask > 0.5, lc_arr, NAN_VALUE)
train_mask = np.where(test_mask < 0.5, 1, NAN_VALUE)
no_data_arr = np.where(lc_arr > 0, 1, NAN_VALUE)  # 1=data, 0=NoData
# remove the NoData from the train_mask
train_mask = np.where(no_data_arr == 1, train_mask, NAN_VALUE)

### 4. FEATURES: spectral bands, vegetation indices, and phenologic parameters
# # All features (alphabetic)
# bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
# band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
# months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
# nmonths = [x for x in range(1, 13)]
# vars = ['NPixels', 'MIN', 'MAX', 'AVG', 'STDEV']
# phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
# phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

# # All features actually used for classification
# bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
# band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
# months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
# nmonths = [x for x in range(1, 13)]
# vars = ['AVG', 'STDEV']
# phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
# phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

# Test a small subset for classification
bands = ['Blue', 'Evi', 'Ndvi', 'Nir', 'Red', 'Swir1']
band_num = ['B2', '', '', 'B5', 'B4', 'B6']
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
nmonths = [x for x in range(1, 13)]
vars = ['AVG', 'STDEV']
phen = ['SOS', 'EOS', 'DOP', 'MAX', 'NOS']
phen2 = ['SOS2', 'EOS2', 'DOP2', 'CUM']

# # TEST a small subset
# bands = ['Blue', 'Green', 'Ndvi', 'Red']
# band_num = ['B2', 'B3', '', 'B4']
# months = ['MAR']
# nmonths = [3]
# vars = ['AVG']
# phen = ['SOS', 'EOS', 'LOS']
# # phen2 = ['SOS2', 'EOS2']
# phen2 = []

# # TEST a "reasonable" subset
# bands = ['Blue', 'Green', 'Ndvi', 'Nir', 'Red', 'Swir1']
# band_num = ['B2', 'B3', '', 'B5', 'B4', 'B6']
# months = ['MAR', 'JUN', 'SEP', 'DEC']
# nmonths = [3, 6, 9, 12]
# vars = ['AVG']
# phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR']
# # phen2 = ['SOS2', 'EOS2', 'LOS2']
# phen2 = []

# Calculate the dimensions of the array
arr_cols = test_mask.shape[1]
arr_rows = test_mask.shape[0]
lyrs = len(bands) * len(months) * len(vars) + len(phen) + len(phen2)
print(f'  Dataset dims: rows={arr_rows}, cols={arr_cols}, layers={lyrs}')

features_end = True

### 5. CREATE (LARGE) HDF5 FILES TO HOLD ALL FEATURES
f_all = h5py.File(fn_features, 'w')
f_train_all = h5py.File(fn_train_feat, 'w')
f_test_all = h5py.File(fn_test_feat, 'w')
f_labels_all = h5py.File(fn_labels, 'w')

# Save the training and testing labels
f_labels_all.create_dataset('training', (arr_rows, arr_cols), data=train_lbl)
f_labels_all.create_dataset('testing', (arr_rows, arr_cols), data=test_lbl)
f_labels_all.create_dataset('test_mask', (arr_rows, arr_cols), data=test_mask)
f_labels_all.create_dataset('train_mask', (arr_rows, arr_cols), data=train_mask)
f_labels_all.create_dataset('no_data_mask', (arr_rows, arr_cols), data=no_data_arr)

feat_indices = []
feat_names = []

feature = 0
for j, band in enumerate(bands):
    # print(f'{band.upper()}')
    for i, month in enumerate(months):
        # filename = cwd + data_subdir + 'MONTHLY.' + band.upper() + '.' + str(nmonths[i]).zfill(2) + '.' + month + '.hdf'
        filename = cwd + data_subdir + '02_STATS/MONTHLY.' + band.upper() + '.' + month + '.hdf'
        print(f"  Processing: {filename}")
        for var in vars:
            # Create the name of the dataset in the HDF
            feat_name = band_num[j] + ' (' + band + ') ' + var
            if band_num[j] == '':
                feat_name = band.upper() + ' ' + var
            print(f'  Feature: {feature:>4} Month: {month:>4} Variable: {var:>8} Dataset: {feat_name:>16}')

            # Extract data and filter by training mask
            feat_arr = rs.read_from_hdf(filename, feat_name)  # Use HDF4 method

            ### Fill missing data
            if FILL:
                minimum = 0  # set minimum for spectral bands
                max_row, max_col = None, None
                if band.upper() in ['NDVI', 'EVI', 'EVI2']:
                    minimum = -10000  # minimum for VIs
                feat_arr = fill_with_mean(feat_arr, minimum, var=band.upper(), verbose=False)

            # Normalize or standardize
            if NORMALIZE:
                feat_arr = rs.normalize(feat_arr)

            # print(f'    test_mask: {test_mask.dtype}, unique:{np.unique(test_mask.filled(0))}, {test_mask.shape}')
            # print(f'    feat_arr: {type(feat_arr)} {feat_arr.dtype}, {feat_arr.shape}')
            train_arr = np.where(test_mask < 0.5, feat_arr, NAN_VALUE)
            test_arr = np.where(test_mask > 0.5, feat_arr, NAN_VALUE)

            # Save the index of features
            feat_names.append(month + ' ' + feat_name)
            feat_indices.append(feature)
            # Save features for the complete raster
            f_all.create_dataset(month + ' ' + feat_name, (arr_rows, arr_cols), data=feat_arr)
            f_train_all.create_dataset(month + ' ' + feat_name, (arr_rows, arr_cols), data=train_arr)
            f_test_all.create_dataset(month + ' ' + feat_name, (arr_rows, arr_cols), data=test_arr)
            feature += 1

# Add phenology
print(f"  Processing: {fn_phenology}")
for param in phen:
    print(f'  Feature: {feature} Variable: {param}')

    # No need to fill missing values, just read the values
    pheno_arr = rs.read_from_hdf(fn_phenology, param)  # Use HDF4 method

    # # Fill missing data
    if FILL:
        if param == 'SOS':
            print(f'  --Filling {param}')
            minimum = 0
            sos = rs.read_from_hdf(fn_phenology, 'SOS')
            eos = rs.read_from_hdf(fn_phenology, 'EOS')
            los = rs.read_from_hdf(fn_phenology, 'LOS')
            
            # Fix SOS values larger than 365
            sos_fixed = np.where(sos > 366, sos-365, sos)

            # Fix SOS values larger than 365, needs to be done two times
            eos_fixed = np.where(eos > 366, eos-365, eos)
            # print(np.min(eos_fixed), np.max(eos_fixed))
            if np.max(eos_fixed) > 366:
                eos_fixed = np.where(eos_fixed > 366, eos_fixed-365, eos_fixed)
                print(f'  --Adjusting EOS again: {np.min(eos_fixed)}, {np.max(eos_fixed)}')

            filled_sos, filled_eos, filled_los =  fill_season(sos_fixed, eos_fixed, los, minimum,
                                                            row_pixels=arr_rows,
                                                            max_row=arr_rows,
                                                            max_col=arr_cols,
                                                            id=param,
                                                            verbose=True)

            pheno_arr = filled_sos[:]
        elif param == 'EOS':
            print(f'  --Filling {param}')
            pheno_arr = filled_eos[:]
        elif param == 'LOS':
            print(f'  --Filling {param}')
            pheno_arr = filled_los[:]
        elif param == 'DOP' or param == 'NOS':
            # Day-of-peak and Number-of-seasons, use mode
            print(f'  --Filling {param}')
            pheno_arr = fill_with_mode(pheno_arr, 0, row_pixels=arr_rows, max_row=arr_rows, max_col=arr_cols, verbose=False)
        elif param == 'GDR' or param == 'GUR' or param == 'MAX':
            # GDR, GUR and MAX should be positive integers!
            print(f'  --Filling {param}')
            pheno_arr = fill_with_int_mean(pheno_arr, 0, var=param, verbose=False)
        else:
            # Other parameters? Not possible
            print(f'  --Filling {param}')
            ds = rs.read_from_hdf(fn_phenology, param)
            pheno_arr = fill_with_int_mean(ds, 0, var=param, verbose=False)

    # Normalize or standardize
    assert not (NORMALIZE and STANDARDIZE), "Cannot normalize and standardize at the same time!"
    if NORMALIZE and not STANDARDIZE:
        pheno_arr = rs.normalize(pheno_arr)
    elif not NORMALIZE and STANDARDIZE:
        pheno_arr = rs.standardize(pheno_arr)
    
    train_arr = np.where(test_mask < 0.5, pheno_arr, NAN_VALUE)
    test_arr = np.where(test_mask > 0.5, pheno_arr, NAN_VALUE)

    feat_name = 'PHEN ' + param
    feat_names.append(feat_name)
    feat_indices.append(feature)
    # Save features for the complete raster
    f_all.create_dataset(feat_name, (arr_rows, arr_cols), data=pheno_arr)
    f_train_all.create_dataset(feat_name, (arr_rows, arr_cols), data=train_arr)
    f_test_all.create_dataset(feat_name, (arr_rows, arr_cols), data=test_arr)
    feature += 1

# Add phenology from second file
print(f"  Processing: {fn_phenology2}")
for param in phen2:
    print(f'  Feature: {feature} Variable: {param}')

    # No need to fill missing values, just read the values
    pheno_arr = rs.read_from_hdf(fn_phenology2, param)  # Use HDF4 method

    # Extract data and filter by training mask
    if FILL:
        # IMPORTANT: Only a few pixels have a second season, thus dataset could
        # have a huge amount of NaNs, filling will be restricted to replace a
        # The missing values to NAN_VALUE
        print(f'  --Filling {param}')
        pheno_arr = rs.read_from_hdf(fn_phenology2, param)
        pheno_arr = np.where(pheno_arr <= 0, 0, pheno_arr)
    
    # Normalize or standardize
    assert not (NORMALIZE and STANDARDIZE), "Cannot normalize and standardize at the same time!"
    if NORMALIZE and not STANDARDIZE:
        pheno_arr = rs.normalize(pheno_arr)
    elif not NORMALIZE and STANDARDIZE:
        pheno_arr = rs.standardize(pheno_arr)

    train_arr = np.where(test_mask < 0.5, pheno_arr, NAN_VALUE)
    test_arr = np.where(test_mask > 0.5, pheno_arr, NAN_VALUE)

    feat_name = 'PHEN ' + param
    feat_names.append(feat_name)
    feat_indices.append(feature)
    # Save features for the complete raster
    f_all.create_dataset(feat_name, (arr_rows, arr_cols), data=pheno_arr)
    f_train_all.create_dataset(feat_name, (arr_rows, arr_cols), data=train_arr)
    f_test_all.create_dataset(feat_name, (arr_rows, arr_cols), data=test_arr)
    feature += 1

print(f"File: {fn_features} created successfully.")
print(f"File: {fn_train_feat} created successfully.")
print(f"File: {fn_test_feat} created successfully.")
print(f"File: {fn_labels} created successfully.")

# Save a file with the parameters used
with open(fn_parameters, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter='=')
    writer.writerow(['NAN_VALUE', NAN_VALUE])
    writer.writerow(['EPSG', epsg])
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
    writer.writerow(['NUM_CLASSES', len(np.unique(lc_arr.filled(0)))])
    writer.writerow(['LAND_COVER_RASTER', fn_landcover])
    writer.writerow([' METADATA', f'{lc_md}'])
    writer.writerow([' NO_DATA', f'{lc_nd}'])
    writer.writerow([' RASTER_COLUMNS', f'{lc_arr.shape[1]}'])
    writer.writerow([' RASTER_ROWS', f'{lc_arr.shape[0]}'])
    writer.writerow([' GEOTRANSFORM', f'{lc_gt}'])
    writer.writerow([' PROJECTION', f'{lc_proj}'])
    writer.writerow(['FEATURE_BANDS', ';'.join([x for x in bands])])
    writer.writerow(['FEATURE_MONTHS', ';'.join([x for x in months])])
    writer.writerow(['FEATURE_VARIABLES', ';'.join([x for x in vars])])
    writer.writerow(['FEATURE_PHENO', ';'.join([x for x in phen])])
    writer.writerow(['FEATURE_PHENO2', ';'.join([x for x in phen2])])
    writer.writerow(['PHENO_FILE', fn_phenology])
    writer.writerow(['PHENO2_FILE', fn_phenology2])
print(f"File: {fn_parameters} created successfully.")

with open(fn_feat_indices, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for i, feat in zip(feat_indices, feat_names):
        writer.writerow([i, feat])
print(f"File: {fn_feat_indices} created successfully.")
