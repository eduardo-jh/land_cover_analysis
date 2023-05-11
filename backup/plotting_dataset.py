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

# ## Plot the labels...
# # fn_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY.tif'        # Land cover raster
# fn_labels = cwd + 'Calakmul_Labels.h5'
# fn_parameters = cwd + 'dataset_parameters.csv'
# fn_colormap = cwd + 'qgis_cmap_landcover_CBR_viri.clr'
# # Read the parameters saved from previous script to ensure matching
# parameters = rs.read_params(fn_parameters)
# rows, cols = int(parameters['ROWS']), int(parameters['COLUMNS'])
# y_train = np.empty((rows,cols), dtype=np.uint8)
# y_test = np.empty((rows,cols), dtype=np.uint8)
# # # test_mask = np.empty((rows,cols), dtype=np.uint8)
# # Read the labels and features
# with h5py.File(fn_labels, 'r') as fy:
#     y_train = fy['training'][:]
#     y_test = fy['testing'][:]
#     test_mask = fy['test_mask'][:]
# # Create a complete array of labels
# array = np.where(test_mask == 1, y_test, y_train)

# ## ... or check an array of predictions
array, _, _, _, _ = rs.open_raster('/vipdata/2023/CALAKMUL/ROI1/results/2023_04_29-19_22_30_rf_predictions.tif')
array = array.filled(0)

print(f'  array={np.unique(array, return_counts=True)}')

# rs.plot_dataset(array)
# rs.plot_land_cover(array, fn_colormap, savefig='/vipdata/2023/CALAKMUL/ROI1/test1.png')
rs.plot_array_clr(array, '/vipdata/2023/CALAKMUL/ROI1/qgis_cmap_landcover_CBR_viri.clr',
                  savefig='/vipdata/2023/CALAKMUL/ROI1/test2.png', zero=True)