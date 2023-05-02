#!/usr/bin/env python
# coding: utf-8

import sys
import platform
import pickle
import csv
import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

# adding the directory with modules
system = platform.system()
if system == 'Windows':
    # On Windows 10
    sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib/')
    cwd = 'D:/Desktop/CALAKMUL/ROI1/'
# elif system == 'Linux':
#     # On Alma Linux Server
#     sys.path.insert(0, '/home/eduardojh/Documents/land_cover_analysis/lib/')
#     cwd = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/CALAKMUL/ROI1/'
elif system == 'Linux':
    # On Ubuntu Workstation
    sys.path.insert(0, '/vipdata/2023/land_cover_analysis/lib/')
    cwd = '/vipdata/2023/CALAKMUL/ROI1/'
else:
    print('System not yet configured!')

import rsmodule as rs

fn_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY.tif'        # Land cover raster
fn_features = cwd + 'Calakmul_Features.h5'
fn_train_feat = cwd + 'Calakmul_Training_Features.h5'
fn_test_feat = cwd + 'Calakmul_Testing_Features.h5'
fn_labels = cwd + 'Calakmul_Labels.h5'
fn_feat_indices = cwd + 'feature_indices.csv'
fn_parameters = cwd + 'dataset_parameters.csv'
fn_colormap = cwd + 'qgis_cmap_landcover_CBR_viri.clr'

# Read the parameters saved from previous script to ensure matching
parameters = rs.read_params(fn_parameters)
# print(parameters)
rows, cols = int(parameters['ROWS']), int(parameters['COLUMNS'])
row_pixels, col_pixels = int(parameters['IMG_ROWS']), int(parameters['IMG_COLUMNS'])
n_classes = int(parameters['NUM_CLASSES'])
bands = int(parameters['LAYERS'])
img_x_row = int(parameters['IMG_PER_ROW'])
img_x_col = int(parameters['IMG_PER_COL'])

# Read the feature indices
feat_index = {}
with open(fn_feat_indices, 'r',) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        if len(row) == 0:
            continue
        feat_index[int(row[0])] = row[1]
# print(feat_index)

# x_train = np.empty((rows,cols,bands), dtype=np.int16)
y_train = np.empty((rows,cols), dtype=np.uint8)
# x_test = np.empty((rows,cols,bands), dtype=np.int16)
y_test = np.empty((rows,cols), dtype=np.uint8)
# test_mask = np.empty((rows,cols), dtype=np.uint8)

### Read the labels and features
with h5py.File(fn_labels, 'r') as fy:
    y_train = fy['training'][:]
    y_test = fy['testing'][:]
    test_mask = fy['test_mask'][:]

array = np.where(test_mask == 1, y_test, y_train)

# array, _, _, _, _ = rs.open_raster('/vipdata/2023/CALAKMUL/ROI1/results/2023_05_01-16_45_55_rf_predictions.tif')
print(f'  array={np.unique(array, return_counts=True)}')

# # Read a custom colormap
# lccmap = rs.read_clr(fn_colormap)
# bounds = [x for x in range(len(lccmap.colors)+1)]
# print(f'  n_clases={n_classes} colors={len(lccmap.colors)}')
# print(f'  bounds={bounds}')
# norm = mpl.colors.BoundaryNorm(bounds, lccmap.N)

# # plt.figure(figsize=(12,12))
# # plt.imshow(y_pred, cmap='viridis')
# fig, ax = plt.subplots(figsize=(12, 12))
# fig.subplots_adjust(left=0.5)
# plt.imshow(array, cmap=lccmap)
# plt.colorbar(mpl.cm.ScalarMappable(cmap=lccmap, norm=norm), ticks=bounds, cax=ax, orientation='vertical')
# # plt.savefig(save_preds_fig, bbox_inches='tight', dpi=300)
# plt.show()
# plt.close()

# rs.plot_dataset(array)

rs.plot_land_cover(array, fn_colormap, zero=True)