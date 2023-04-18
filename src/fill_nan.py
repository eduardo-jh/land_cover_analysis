#!/usr/bin/env python
# coding: utf-8
""" Code to test the fill NaNs functions in HDF5 files """

import sys
# import os.path
import platform
import h5py
import pandas as pd
import numpy as np
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

# Load feature valid ranges from file
ranges = pd.read_csv(cwd + 'valid_ranges', sep='=', index_col=0)
MIN_BAND = ranges.loc['MIN_BAND', 'VALUE']
MAX_BAND = ranges.loc['MAX_BAND', 'VALUE']
MIN_VI = ranges.loc['MIN_VI', 'VALUE']
MAX_VI = ranges.loc['MAX_VI', 'VALUE']
MIN_PHEN = ranges.loc['MIN_PHEN', 'VALUE']
NAN_VALUE = ranges.loc['NAN_VALUE', 'VALUE']

print(f'MIN_BAND={MIN_BAND}')
print(f'MAX_BAND={MAX_BAND}')
print(f'MIN_VI={MIN_VI}')
print(f'MAX_VI={MAX_VI}')
print(f'MIN_PHEN={MIN_PHEN}')
print(f'NAN_VALUE={NAN_VALUE}')

# MAX_COL = 181
# MAX_ROW = 765

def fill_nans_mean(dataset, min_value, **kwargs):
    """ Fills NaNs using the mean of all valid data """
    _max_row = kwargs.get('max_row', None)
    _max_col = kwargs.get('max_col', None)

    # Valid values are larger than minimum, otherwise are NaNs (e.g. -13000, -1, etc.)
    valid_ds = np.where(dataset >= min_value, dataset, np.nan)

    # Fill NaNs with the mean of valid data
    fill_value = round(np.nanmean(valid_ds), 2)
    filled_ds = np.where(dataset >= min_value, dataset, fill_value)

    # Values beyond max row and column are geographically meaningless, make them NaNs again 
    if _max_col is not None:
        valid_ds[:,MAX_COL:] = np.nan
        filled_ds[:,MAX_COL:] = np.nan
    if _max_row is not None:
        valid_ds[MAX_ROW:,:] = np.nan
        filled_ds[MAX_ROW:,:] = np.nan
    return filled_ds


def fill_season(sos, eos, los, min_value, **kwargs):
    """ Fills missing values from SOS, EOS and LOS """
    _nan = kwargs.get('nan', -1)
    _max_row = kwargs.get('max_row', None)
    _max_col = kwargs.get('max_col', None)
    _verbose = kwargs.get('verbose', False)
    _col_pixels = kwargs.get('col_pixels', 1000)
    _row_pixels = kwargs.get('row_pixels', 1000)
    
    ### SOS
    sos = sos.astype(int)
    # valid_sos = np.where(sos == _nan, 1, 0)
    valid_sos = np.where(sos < min_value, 1, 0)
    print(f'Missing data found at SOS: {np.sum(valid_sos)}')

    ### EOS
    eos = eos.astype(int)
    # valid_eos = np.where(eos == _nan, 1, 0)
    valid_eos = np.where(eos < min_value, 1, 0)
    print(f'Missing data found at EOS: {np.sum(valid_eos)}')

    ### LOS
    los = los.astype(int)
    # valid_los = np.where(los == _nan, 1, 0)
    valid_los = np.where(los < min_value, 1, 0)
    print(f'Missing data found at LOS: {np.sum(valid_eos)}')

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
    # print(loc_sos)
    # print(loc_eos)
    # print(loc_los)

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
            # print(row, col)
            val = sos[row, col]
            # print(val)
            # Adjust row,col to use for slicing when point near the edges
            row_start = row-1
            row_end = row+2
            col_start = col-1
            col_end = col+2
            # ONLY FOR LAST IMAGES IN ROW/COL
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
            # TODO: increase window when all values are empty
            while val in values_sos:
                values_sos.remove(val)
            while val in values_eos:
                values_eos.remove(val)
            assert len(values_sos) > 0, "Values SOS empty!"
            assert len(values_eos) > 0, "Values EOS empty!"
            
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
                print(f'SOS: {row},{col}: {val}, values={values_sos}, fill_val={fill_value_sos}')
                print(f'EOS: {row},{col}: {val}, values={values_eos}, fill_val={fill_value_eos}')
                print(f'LOS: {row},{col}: {val}, fill_val={fill_value_los}')
            filled_sos[row, col] = fill_value_sos
            filled_eos[row, col] = fill_value_eos
            filled_los[row, col] = fill_value_los


def fill_missing_values(fn_hdf_feat, **kwargs):
    _col_pixels = kwargs.get('col_pixels', 1000)
    _row_pixels = kwargs.get('row_pixels', 1000)

    with h5py.File(fn_hdf_feat, 'r') as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            print(f"{i:>3}/{len(keys):>3}:{key:>22}", end='')

            ds = f[key][:]

            minimum = MIN_BAND

            # Get the type of feature
            feat_type = 'BAND'
            if key[0:4] == 'PHEN':
                feat_type = 'PHEN'
                minimum = MIN_PHEN
            elif key[4:8] == 'EVI ' or  key[4:8] == 'NDVI' or  key[4:8] == 'EVI2':
                feat_type = 'VI'
                minimum = MIN_VI
            print(f"{feat_type:>5}", end='')

            if key == 'PHEN GDR' or key == 'PHEN GDR2' or key == 'PHEN GUR' or key == 'PHEN GUR2':
                minimum = MIN_PHEN
            
            # var = 'VAL'
            # if key[-3:] == 'AVG':
            #     var = 'AVG'
            # elif key[-3:] == 'MAX' and feat_type != 'PHEN':
            #     var = 'MAX'
            # elif key[-3:] == 'MIN':
            #     var = 'MIN'
            # elif key[-3:] == 'els':
            #     var = 'NPI'
            # elif key[-3:] == 'DEV':
            #     var = 'STD'
            # print(f"{var:>4}", end='')

            print('\n')


if __name__ == "__main__":

    # Paths and file names for the current ROI
    fn_features = cwd + 'IMG_Calakmul_Features.h5'
    fn_train_feat = cwd + 'IMG_Calakmul_Training_Features.h5'
    fn_test_feat = cwd + 'IMG_Calakmul_Testing_Features.h5'
    fn_parameters = cwd + 'dataset_parameters.csv'

    parameters = rs.read_params(fn_parameters)
    print(parameters)
    row_pixels, col_pixels = int(parameters['IMG_ROWS']), int(parameters['IMG_COLUMNS'])
    n_classes = int(parameters['NUM_CLASSES'])
    bands = int(parameters['LAYERS'])
    img_x_row = int(parameters['IMG_PER_ROW'])
    img_x_col = int(parameters['IMG_PER_COL'])
    batch_size = img_x_col  # IMPORTANT: batch size is one row!

    fill_missing_values(fn_features, col_pixels=col_pixels)

    # # fn_features = 'D:/Desktop/CALAKMUL/ROI1/IMG_Calakmul_Features.h5'
    # fn_features = '/vipdata/2023/CALAKMUL/ROI1/IMG_Calakmul_Features.h5'

    # with h5py.File(fn_features, 'r') as f:
    #     dataset = f['r0c4'][:]
    #     print(dataset.shape)

    # ndvi_march = dataset[:,:,16]
    # print(ndvi_march.shape)
    # valid_ds = np.where(ndvi_march >= -10000, ndvi_march, np.nan)

    # fill_value = round(np.nanmean(valid_ds), 2)
    # print(f'NDVI fill value: {fill_value}')

    # filled_ds = np.where(ndvi_march >= -10000, ndvi_march, fill_value)

    # valid_ds[:,MAX_COL:] = np.nan
    # filled_ds[:,MAX_COL:] = np.nan

    # # valid_ds[MAX_ROW:,:] = np.nan
    # # filled_ds[MAX_ROW:,:] = np.nan

    # ### SOS
    # sos = dataset[:,:,48]
    # sos = sos.astype(int)
    # valid_sos = np.where(sos == -1, 1, 0)
    # print(f'Missing data found at SOS: {np.sum(valid_sos)}')

    # ### EOS
    # eos = dataset[:,:,49]
    # eos = eos.astype(int)
    # valid_eos = np.where(eos == -1, 1, 0)
    # print(f'Missing data found at EOS: {np.sum(valid_eos)}')

    # ### LOS
    # los = dataset[:,:,49]
    # los = los.astype(int)
    # valid_los = np.where(los == -1, 1, 0)
    # print(f'Missing data found at LOS: {np.sum(valid_eos)}')

    # # Find indices of rows with NaNs
    # loc_sos = {}
    # loc_eos = {}
    # loc_los = {}
    # for i in range(1000):
    #     if np.sum(valid_sos[i]) > 0:
    #         # Find the indices of columns with NaNs, save them in their corresponding row
    #         cols = np.where(valid_sos[i] == 1)
    #         loc_sos[i] = cols[0].tolist()
    #     if np.sum(valid_eos[i]) > 0:
    #         cols = np.where(valid_eos[i] == 1)
    #         loc_eos[i] = cols[0].tolist()
    #     if np.sum(valid_los[i]) > 0:
    #         cols = np.where(valid_los[i] == 1)
    #         loc_los[i] = cols[0].tolist()
    # # print(loc_sos)
    # # print(loc_eos)
    # # print(loc_los)

    # filled_sos = sos[:]
    # filled_eos = eos[:]
    # filled_los = los[:]

    # for row in loc_sos.keys():
    #     # Make sure the indices among SOS, EOS, and LOS are the same
    #     assert row in loc_eos.keys(), f"SOS key {row} not in EOS"
    #     assert row in loc_los.keys(), f"SOS key {row} not in LOS"
    #     for col in loc_sos[row]:
    #         assert col in loc_eos[row], f"Key {row} not in EOS"
    #         assert col in loc_los[row], f"SOS key {row} not in LOS"
    #         # print(row, col)
    #         val = sos[row, col]
    #         # print(val)
    #         # Adjust row,col to use for slicing when point near the edges
    #         row_start = row-1
    #         row_end = row+2
    #         col_start = col-1
    #         col_end = col+2
    #         # ONLY FOR LAST IMAGES IN ROW/COL
    #         # if row_start < 0:
    #         #     row_start = 0
    #         # if row_end > MAX_ROW:
    #         #     row_end = MAX_ROW
    #         if col_start < 0:
    #             col_start = 0
    #         if col_end > MAX_COL:
    #             col_end = MAX_COL
            
    #         # Slice a window of values around missing value
    #         window_sos = sos[row_start:row_end, col_start:col_end]
    #         window_eos = eos[row_start:row_end, col_start:col_end]
    #         # window_los = los[row_start:row_end, col_start:col_end]
    #         # print(f'{row_start}:{row_end}, {col_start}:{col_end}')
    #         # print(window)
    #         values_sos = window_sos.flatten().tolist()
    #         values_eos = window_eos.flatten().tolist()
    #         # values_los = window_los.flatten().tolist()
    #         # print(values, '\n')
    #         # Remove NaN values from the list
    #         while val in values_sos:
    #             values_sos.remove(val)
    #         while val in values_eos:
    #             values_eos.remove(val)
    #         # For SOS use mode (will return minimum value)
    #         fill_value_sos = stats.mode(values_sos, keepdims=False)[0]
    #         # print('Mode:', fill_value_sos, type(fill_value_sos))
    #         # fill_value_sos = stats.mode(values_sos, keepdims=True)[0][0]

    #         # For EOS use mode or max value (if min returned)
    #         fill_value_eos = stats.mode(values_eos, keepdims=False)[0]
    #         # fill_value_eos = stats.mode(values_eos, keepdims=True)[0][0]
    #         if fill_value_eos == np.min(values_eos):
    #             fill_value_eos = np.max(values_eos)
            
    #         # Fill value for LOS
    #         fill_value_los = fill_value_eos - fill_value_sos

    #         # print(f'{row},{col}: {val}, values={values}, fill_val={fill_value_sos}')
    #         print(f'SOS: {row},{col}: {val}, values={values_sos}, fill_val={fill_value_sos}')
    #         print(f'EOS: {row},{col}: {val}, values={values_eos}, fill_val={fill_value_eos}')
    #         print(f'LOS: {row},{col}: {val}, fill_val={fill_value_los}')
    #         filled_sos[row, col] = fill_value_sos
    #         filled_eos[row, col] = fill_value_eos
    #         filled_los[row, col] = fill_value_los

    # with h5py.File(fn_features[:-3]+'_test.h5', 'w') as f:
    #     f.create_dataset('valid_ndvi', data=valid_ds)
    #     f.create_dataset('filled_ndvi', data=filled_ds)
    #     f.create_dataset('filled_sos', data=filled_sos)
    #     f.create_dataset('filled_eos', data=filled_eos)
    #     f.create_dataset('filled_los', data=filled_los)
