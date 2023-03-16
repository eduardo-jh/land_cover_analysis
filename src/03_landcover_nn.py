#!/usr/bin/env python
# coding: utf-8

""" Neural networks

Eduardo Jimenez <eduardojh@email.arizona.edu>

NOTE: run under 'rstf' conda environment (python 3.8.13, keras 2.9.0)
"""

import sys
import os.path
import platform
import h5py
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Nadam, Adam

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

NA_VALUE = -32768 # Keep 16-bit integer, source's NA = -13000
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
