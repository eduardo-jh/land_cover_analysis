#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
from scipy import stats

MAX_COL = 181
MAX_ROW = 765

fn_features = 'D:/Desktop/CALAKMUL/ROI1/IMG_Calakmul_Features.h5'

with h5py.File(fn_features, 'r') as f:
    dataset = f['r0c4'][:]
    print(dataset.shape)

ndvi_march = dataset[:,:,16]
print(ndvi_march.shape)
valid_ds = np.where(ndvi_march >= -10000, ndvi_march, np.nan)

fill_value = round(np.nanmean(valid_ds), 2)
print(f'Fill value: {fill_value}')

filled_ds = np.where(ndvi_march >= -10000, ndvi_march, fill_value)

valid_ds[:,MAX_COL:] = np.nan
filled_ds[:,MAX_COL:] = np.nan

# valid_ds[MAX_ROW:,:] = np.nan
# filled_ds[MAX_ROW:,:] = np.nan

### SOS
sos = dataset[:,:,48]
sos = sos.astype(int)
# print(np.unique(sos))
valid_sos = np.where(sos == -1, 1, 0)
# print(valid_sos)
print(f'Total found: {np.sum(valid_sos)}')

# Find indices of rows with NaNs
locations = {}
for i in range(1000):
    if np.sum(valid_sos[i]) > 0:
        # print(i, np.where(valid_sos[i] == 1)[0], valid_sos[i])
        # Find the indices of columns with NaNs, save them in their corresponding row
        cols = np.where(valid_sos[i] == 1)
        # print(i, type(cols), cols[0])
        locations[i] = cols[0].tolist()
print(locations)

filled_sos = sos[:]
for row in locations.keys():
    for col in locations[row]:
        # print(row, col)
        val = sos[row, col]
        # print(val)
        # Adjust row,col to use for slicing when point near the edges
        row_start = row-1
        row_end = row+2
        col_start = col-1
        col_end = col+2
        # ONLY FOR LAST IMAGES IN ROW/COL
        # if row_start < 0:
        #     row_start = 0
        # if row_end > MAX_ROW:
        #     row_end = MAX_ROW
        if col_start < 0:
            col_start = 0
        if col_end > MAX_COL:
            col_end = MAX_COL
        # Slice a window of values around missing value
        window = sos[row_start:row_end, col_start:col_end]
        # print(f'{row_start}:{row_end}, {col_start}:{col_end}')
        # print(window)
        values = window.flatten().tolist()
        # print(values, '\n')
        # Remove NaN values from the list
        while val in values:
            values.remove(val)
        # For SOS use mode (will return minimum value)
        fill_value_sos = stats.mode(values)[0][0]
        print(f'{row},{col}: {val}, values={values}, fill_val={fill_value_sos}')
        filled_sos[row, col] = fill_value_sos

### EOS
eos = dataset[:,:,49]
eos = eos.astype(int)
# print(np.unique(sos))
valid_eos = np.where(eos == -1, 1, 0)
# print(valid_sos)
print(f'Total found: {np.sum(valid_eos)}')
# Find indices of rows with NaNs
locations = {}
for i in range(1000):
    if np.sum(valid_eos[i]) > 0:
        # print(i, np.where(valid_sos[i] == 1)[0], valid_sos[i])
        # Find the indices of columns with NaNs, save them in their corresponding row
        cols = np.where(valid_eos[i] == 1)
        # print(i, type(cols), cols[0])
        locations[i] = cols[0].tolist()
print(locations)
filled_eos = eos[:]
for row in locations.keys():
    for col in locations[row]:
        # print(row, col)
        val = eos[row, col]
        # print(val)
        # Adjust row,col to use for slicing when point near the edges
        row_start = row-1
        row_end = row+2
        col_start = col-1
        col_end = col+2
        # ONLY FOR LAST IMAGES IN ROW/COL
        # if row_start < 0:
        #     row_start = 0
        # if row_end > MAX_ROW:
        #     row_end = MAX_ROW
        if col_start < 0:
            col_start = 0
        if col_end > MAX_COL:
            col_end = MAX_COL
        # Slice a window of values around missing value
        window = eos[row_start:row_end, col_start:col_end]
        # print(f'{row_start}:{row_end}, {col_start}:{col_end}')
        print(window)
        values = window.flatten().tolist()
        # print(values, '\n')
        # Remove NaN values from the list
        while val in values:
            values.remove(val)
        # For SOS use mode (will return minimum value)
        fill_value = stats.mode(values)[0][0]
        # For EOS use mode or max value (if min returned)
        if fill_value == np.min(values):
            fill_value = np.max(values)
        print(f'{row},{col}: {val}, values={values}, fill_val={fill_value}')
        filled_eos[row, col] = fill_value

with h5py.File(fn_features[:-3]+'_test.h5', 'w') as f:
    f.create_dataset('valid_ndvi', data=valid_ds)
    f.create_dataset('filled_ndvi', data=filled_ds)
    f.create_dataset('filled_sos', data=filled_sos)
    f.create_dataset('filled_eos', data=filled_eos)
