#!/usr/bin/env python
# coding: utf-8

""" 02.1_dataset_season.py: Creates a dataset grouping monthly datasets by
    season of the year: spring, summer, fall, winter

Eduardo Jimenez <eduardojh@email.arizona.edu>

  May 22, 2023: group monthly datasets by season of the year

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
# Files to aggregate
fn_features = cwd + 'Calakmul_Features.h5'
fn_train_feat = cwd + 'Calakmul_Training_Features.h5'
fn_test_feat = cwd + 'Calakmul_Testing_Features.h5'
fn_labels = cwd + 'Calakmul_Labels.h5'

# Create files to aggregate features by season
fn_features_season = cwd + 'Calakmul_Features_season.h5'
fn_train_feat_season = cwd + 'Calakmul_Training_Features_season.h5'
fn_test_feat_season = cwd + 'Calakmul_Testing_Features_season.h5'
# fn_labels_split = cwd + 'Calakmul_Labels_season.h5'

# Read 
fn_parameters = cwd + 'parameters/dataset_parameters.csv'
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
vars = parameters['VARIABLES'].split(',')

features_end = True
seasons = {'SPR': ['MAR', 'APR', 'MAY'],
           'SUM': ['JUN', 'JUL', 'AUG'],
           'FAL': ['SEP', 'OCT', 'NOV'],
           'WIN': ['JAN', 'FEB', 'DEC']}

### 5. READ THE (LARGE) HDF5 FILES THAT HOLD ALL FEATURES
f_all = h5py.File(fn_features, 'r')
f_train_all = h5py.File(fn_train_feat, 'r')
f_test_all = h5py.File(fn_test_feat, 'r')
f_labels_all = h5py.File(fn_labels, 'r')

# Create the files to split the data into
f = h5py.File(fn_features_season, 'w')
f_train = h5py.File(fn_train_feat_season, 'w')
f_test = h5py.File(fn_test_feat_season, 'w')
# f_labels = h5py.File(fn_labels_split, 'w')

# # Create groups to save img labels accordingly
# f_labels.create_group('training')
# f_labels.create_group('testing')

feat_indices = []
feat_names = []
with open(fn_feat_indices, 'r', newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        if len(row) != 2:
            continue
        feat_indices.append(int(row[0]))
        feat_names.append(row[1])

for feat_name in feat_names:
    # Create the name of the dataset in the HDF
    # print(f'  Feature: {feature:>4} Feat name: {feat_name:>16}')

    # Add PHEN features directly, no aggregation by season
    if feat_name[:4] == 'PHEN':
        print(f"  Skipping: {feat_name}")

        # Extract data
        feat_arr = f_all[feat_name][:]
        train_arr = f_train_all[feat_name][:]
        test_arr = f_train_all[feat_name][:]

        f.create_dataset(feat_name, feat_arr.shape, data=feat_arr)
        f_train.create_dataset(feat_name, train_arr.shape, data=train_arr)
        f_test.create_dataset(feat_name, test_arr.shape, data=test_arr)

# Group feature names by season -> band -> variable -> month
season_feats = {}
for season in list(seasons.keys()):
    print(f"  AGGREGATING: {season}")
    for band in bands:
        band = band.upper()
        for var in vars:
            for feat_name in feat_names:
                # Split the feature name to get band and month
                ft_name_split = feat_name.split(' ')
                if len(ft_name_split) == 2:
                    continue
                elif len(ft_name_split) == 3:
                    band_name = ft_name_split[1]
                elif len(ft_name_split) == 4:
                    band_name = ft_name_split[2][1:-1]  # remove parenthesis
                    band_name = band_name.upper()

                for month in seasons[season]:
                    if band == band_name and var == ft_name_split[-1] and month == ft_name_split[0]:
                        season_key = season + ' ' + band + ' ' + var
                        if season_feats.get(season_key) is None:
                            season_feats[season_key] = [feat_name]
                        else:
                            season_feats[season_key].append(feat_name)
                        # print(f"  -- {season} {band:>5} {var:>5}: {feat_name}")

# Calculate averages of features grouped by season
for key in list(season_feats.keys()):
    print(f"  --{key:>15}:")
    for i, feat_name in enumerate(season_feats[key]):
        print(f"  ----Adding {feat_name}")

        # Add the data
        if i == 0:
            # Initialize array to hold average
            feat_arr = f_all[feat_name][:]
            train_arr = f_train_all[feat_name][:]
            test_arr = f_train_all[feat_name][:]
        else:
            # Add remaining months
            feat_arr += f_all[feat_name][:]
            train_arr += f_train_all[feat_name][:]
            test_arr += f_train_all[feat_name][:]
        
    # Average
    feat_arr /= len(season_feats[key])
    train_arr /= len(season_feats[key])
    test_arr /= len(season_feats[key])

    f.create_dataset(key, feat_arr.shape, data=feat_arr)
    f_train.create_dataset(key, train_arr.shape, data=train_arr)
    f_test.create_dataset(key, test_arr.shape, data=test_arr)

print(f"File: {fn_features_season} created successfully.")
print(f"File: {fn_train_feat_season} created successfully.")
print(f"File: {fn_test_feat_season} created successfully.")
# print(f"File: {fn_labels_split} created successfully.")
