#!/usr/bin/env python
# coding: utf-8

""" 02_split_dataset.py: Creates a dataset splitted into artificial images to
    use with machine learning. Useful for Neural Networks.

Eduardo Jimenez <eduardojh@email.arizona.edu>

  May 22, 2023: split dataset into artificial images

NOTE: run under 'rstf' conda environment (python 3.8.13, scikit-learn 1.1.2)
"""

import sys
import os.path
import h5py
import csv
import numpy as np
from math import ceil
from datetime import datetime
from scipy import stats
from typing import Tuple

if len(sys.argv) == 3:
    # Check if arguments were passed from terminal
    args = sys.argv[1:]
    sys.path.insert(0, args[0])
    cwd = args[1]
    print(f"  Using RS_LIB={args[0]}")
    print(f"  Using CWD={args[1]}")
else:
    import platform
    system = platform.system()
    if system == 'Windows':
        # On Windows 10
        sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib/')
        cwd = 'D:/Desktop/CALAKMUL/ROI1/'
    elif system == 'Linux' and os.path.isdir('/vipdata/2023/CALAKMUL/ROI1/'):
        # On Ubuntu Workstation
        sys.path.insert(0, '/vipdata/2023/land_cover_analysis/lib/')
        cwd = '/vipdata/2023/CALAKMUL/ROI1/'
    elif system == 'Linux' and os.path.isdir('/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/CALAKMUL/ROI1/'):
        # On Alma Linux Server
        sys.path.insert(0, '/home/eduardojh/Documents/land_cover_analysis/lib/')
        cwd = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/CALAKMUL/ROI1/'
    else:
        print('  System not yet configured!')

import rsmodule as rs

### 1. CONFIGURE
# Files to split
fn_features = cwd + 'Calakmul_Features.h5'
fn_train_feat = cwd + 'Calakmul_Training_Features.h5'
fn_test_feat = cwd + 'Calakmul_Testing_Features.h5'
fn_labels = cwd + 'Calakmul_Labels.h5'

# Create files to split features into 'artificial' images
fn_features_split = cwd + 'Calakmul_Features_img.h5'
fn_train_feat_split = cwd + 'Calakmul_Training_Features_img.h5'
fn_test_feat_split = cwd + 'Calakmul_Testing_Features_img.h5'
fn_labels_split = cwd + 'Calakmul_Labels_img.h5'

# Read and update existing files
fn_parameters = cwd + 'parameters/dataset_parameters.csv'  # Gets modified
fn_feat_indices = cwd + 'parameters/feature_indices.csv'  

# Read the parameters saved from previous script to ensure matching
parameters = rs.read_params(fn_parameters)
# print(parameters)
arr_rows, arr_cols = int(parameters['ROWS']), int(parameters['COLUMNS'])
lyrs = int(parameters['LAYERS'])
months = parameters['MONTHS'].split(',')
nmonths = [int(x) for x in parameters['MONTHS_NUM'].split(',')]
bands = parameters['BANDS'].split(',')
band_num = parameters['BANDS_NUM'].split(',')

features_end = True
rows, cols = 1000, 1000  # rows, cols of the artificial images
img_x_col = ceil(arr_cols/cols)
img_x_row = ceil(arr_rows/rows)

### 5. READ THE (LARGE) HDF5 FILES THAT HOLD ALL FEATURES
f_all = h5py.File(fn_features, 'r')
f_train_all = h5py.File(fn_train_feat, 'r')
f_test_all = h5py.File(fn_test_feat, 'r')
f_labels_all = h5py.File(fn_labels, 'r')

# Create the files to split the data into
f = h5py.File(fn_features_split, 'w')
f_train = h5py.File(fn_train_feat_split, 'w')
f_test = h5py.File(fn_test_feat_split, 'w')
f_labels = h5py.File(fn_labels_split, 'w')

# Create groups to save img labels accordingly
f_labels.create_group('training')
f_labels.create_group('testing')

feat_indices = []
feat_names = []
with open(fn_feat_indices, 'r', newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        if len(row) != 2:
            continue
        feat_indices.append(int(row[0]))
        feat_names.append(row[1])

images = 0

# TODO: Improve this, is very inefficient to process everytime all the data
# array and the select only a subset to create a fake image, then repeat all
# processing for next image!
for r in range(img_x_row):
    for c in range(img_x_col):
        print(f'\n  === IMAGE {images} === ')

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

        print(f'  --Slicing from row={r_str}:{r_end}, col={c_str}:{c_end}')

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

        for feature, feat_name in zip(feat_indices, feat_names):
            # Create the name of the dataset in the HDF
            print(f'  Feature: {feature:>4} Feat name: {feat_name:>16}')

            # Extract data
            feat_arr = f_all[feat_name][:]
            train_arr = f_train_all[feat_name][:]
            test_arr = f_train_all[feat_name][:]

            # Slice the array of labels
            train_lbl_img[:r_end-r_str,:c_end-c_str] = f_labels_all['training'][r_str:r_end,c_str:c_end]
            test_lbl_img[:r_end-r_str,:c_end-c_str] = f_labels_all['testing'][r_str:r_end,c_str:c_end]
            
            # Slice the array of features, separate training and testing features
            if features_end:
                # Features at the end
                # print(f'  --Src={train_arr[r_str:r_end,c_str:c_end].shape} Dest={training_img[:r_end-r_str,:c_end-c_str,feature].shape}')
                all_feat_img[:r_end-r_str,:c_end-c_str,feature] = feat_arr[r_str:r_end,c_str:c_end]
                training_img[:r_end-r_str,:c_end-c_str,feature] = train_arr[r_str:r_end,c_str:c_end]
                testing_img[:r_end-r_str,:c_end-c_str,feature] = test_arr[r_str:r_end,c_str:c_end]
            else:
                # Features first
                # print(f'  --Src={train_arr[r_str:r_end,c_str:c_end].shape} Dest={training_img[feature,:r_end-r_str,:c_end-c_str].shape}')
                all_feat_img[feature,:r_end-r_str,:c_end-c_str] = feat_arr[r_str:r_end,c_str:c_end]
                training_img[feature,:r_end-r_str,:c_end-c_str] = train_arr[r_str:r_end,c_str:c_end]
                testing_img[feature,:r_end-r_str,:c_end-c_str] = test_arr[r_str:r_end,c_str:c_end]

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

print(f"File: {fn_features_split} created successfully.")
print(f"File: {fn_train_feat_split} created successfully.")
print(f"File: {fn_test_feat_split} created successfully.")
print(f"File: {fn_labels_split} created successfully.")

# Update the file with the parameters used
# WARNING! This will append rows everytime in consecutive executions!
with open(fn_parameters, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter='=')
    writer.writerow(['IMG_ROWS', rows])
    writer.writerow(['IMG_COLUMNS', cols])
    writer.writerow(['IMG_PER_COL', img_x_col])
    writer.writerow(['IMG_PER_ROW', img_x_row])
print(f"File: {fn_parameters} created successfully.")
