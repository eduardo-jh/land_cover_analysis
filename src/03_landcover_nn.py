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
from typing import Tuple, List, Dict
from datetime import datetime
from tensorflow import keras
from keras import layers

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


def create_cnn(input_shape: tuple, n_outputs: int) -> Tuple[keras.models.Model, Dict]:
    """ Create the model for land cover classification with CNN

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = keras.Sequential()

    # Add the layers
    model.add(layers.Conv2D(16, 8, input_shape=input_shape, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(n_outputs, activation='softmax'))  # Predictions are categories

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Arguments for the fit function
    kwargs = {'callbacks': [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
              'batch_size': len(input_shape)}

    return model, kwargs


if __name__ == '__main__':
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

    # Open HDF5 file
    # Read input_shape & n_outputs
    # Create the model
    # Fit the model
    # Accuracy assesment