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
from math import ceil
from matplotlib import pyplot as plt
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
    model.add(layers.Conv2D(64, 7, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, 7, activation='relu'))
    # model.add(layers.Flatten())
    model.add(layers.Dense(n_outputs, activation='softmax'))  # Predictions are categories

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def read_chunk(filename: str, shape: tuple, **kwargs) -> np.ndarray:
    """ Reads a chunk of size (step, step, bands) from a HDF5 file of size (nrows,ncols,bands)
    
    :param filename: The name of the HDF5 file.
    :param shape: Tuple with the rows, columns and bands of the HDF5 file.
    :param chunk: Number of rows/columns of the subdataset to extract.
    :return: Generator for a numpy array.
    """
    _step = kwargs.get('chunk', 500)
    nrows, ncols, _bands = shape

    # Create an array to save and return the data subset
    data = np.zeros((_step,_step,_bands), dtype=np.float32)  # 32-bit float & 16-bit integer
    cstart, rstart, cend, rend = 0, 0, 0, 0  # row and col range to slice
    rsteps = ceil(nrows/_step)
    csteps = ceil(ncols/_step)

    # Open HDF5 file
    with h5py.File(filename, 'r') as f:
        bands = f.keys()
        # print(f"Keys: {bands}")
        # Iterate over rows
        for row in range(rsteps):
            rstart = row*_step
            rend = rstart+_step
            if rend > nrows:
                rend = nrows
            # Iterate over columns
            for col in range(csteps):
                cstart = col*_step
                cend = cstart+_step
                if cend > ncols:
                    cend = ncols
                # Slice over all bands
                print(f"{rstart:>5}:{rend:>5}, {cstart:>5}:{cend:>5}", end='') 
                for i, key in enumerate(bands):
                    # Slice in the exact shape (considering edges)
                    data[:rend-rstart,:cend-cstart,i] = f[key][rstart:rend,cstart:cend]
                yield data


def gen_training_sequences(filename: str, shape: tuple, labels: str, **kwargs) -> tuple:
    """ Generate training X and Y (partial) sequences on-demand from a HDF5 file of size (nrows,ncols,bands)
    
    :param filename: The name of the HDF5 file.
    :param shape: Tuple with the rows, columns and bands of the HDF5 file.
    :param chunk: Number of rows/columns of the subdataset to extract.
    :return: Generator for a tuple of numpy arrays.
    """
    _step = kwargs.get('chunk', 500)
    nrows, ncols, _bands = shape

    # Create an array to save and return the data subset
    x_data = np.zeros((_step,_step,_bands), dtype=np.float32)  # 32-bit float & 16-bit integer
    y_data = np.zeros((_step,_step), dtype=np.float32)  # 32-bit float & 16-bit integer
    cstart, rstart, cend, rend = 0, 0, 0, 0  # row and col range to slice
    rsteps = ceil(nrows/_step)
    csteps = ceil(ncols/_step)

    # Open HDF5 file (X: features)
    with h5py.File(filename, 'r') as f, h5py.File(labels, 'r') as f_lbl:
        bands = f.keys()
        # print(f"Keys: {bands}")
        # Iterate over rows
        for row in range(rsteps):
            rstart = row*_step
            rend = rstart+_step
            if rend > nrows:
                rend = nrows
            # Iterate over columns
            for col in range(csteps):
                cstart = col*_step
                cend = cstart+_step
                if cend > ncols:
                    cend = ncols
                # Slice over all bands
                print(f"{rstart:>5}:{rend:>5}, {cstart:>5}:{cend:>5}", end='') 
                for i, key in enumerate(bands):
                    # Slice in the exact shape (considering edges)
                    x_data[:rend-rstart,:cend-cstart,i] = f[key][rstart:rend,cstart:cend]
                # Prepare the labels
                y_data[:rend-rstart,:cend-cstart] = f_lbl['train'][rstart:rend,cstart:cend]
                yield (x_data, y_data)


if __name__ == '__main__':
    # NA_VALUE = -32768 # Keep 16-bit integer, source's NA = -13000
    # fmt = '%Y_%m_%d-%H_%M_%S'
    # start = datetime.now()

    # ### 1. CONFIGURE
    # # Projection to create raster. SJR: 32612=WGS 84 / UTM zone 12N; CBR: 32616=WGS 84 / UTM zone 16N
    # epsg_proj = 32616

    # Paths and file names for the current ROI
    fn_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY.tif'        # Land cover raster
    fn_test_mask = cwd + 'training/usv250s7cw_ROI1_testing_mask.tif'
    fn_test_labels = cwd + 'training/usv250s7cw_ROI1_testing_labels.tif'
    fn_phenology = cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf'  # Phenology files
    fn_phenology2 = cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf'
    fn_features = cwd + 'Calakmul_Features.h5'
    fn_train_feat = cwd + 'Calakmul_Training_Features.h5'
    fn_test_feat = cwd + 'Calakmul_Testing_Features.h5'
    fn_labels = cwd + 'Calakmul_Labels.h5'

    # ### 2. READ TESTING MASK
    # # Read a raster with the location of the testing sites
    # print(f"File not found: {fn_test_mask}" if not os.path.isfile(fn_test_mask) else "")
    # test_mask, nodata, metadata, geotransform, projection = rs.open_raster(fn_test_mask)
    # print(f'Opening raster: {fn_test_mask}')
    # print(f'Metadata      : {metadata}')
    # print(f'NoData        : {nodata}')
    # print(f'Columns       : {test_mask.shape[1]}')
    # print(f'Rows          : {test_mask.shape[0]}')
    # print(f'Geotransform  : {geotransform}')
    # print(f'Projection    : {projection}')
    # print(f'Type          : {test_mask.dtype}')

    # ### 3. READ LAND COVER LABELS
    # # Read the land cover raster and retrive the land cover classes
    # print(f"File not found: {fn_landcover}" if not os.path.isfile(fn_landcover) else "")
    # lc_arr, lc_nd, lc_md, lc_gt, lc_proj = rs.open_raster(fn_landcover)
    # print(f'Opening raster: {fn_landcover}')
    # print(f'Metadata      : {lc_md}')
    # print(f'NoData        : {lc_nd}')
    # print(f'Columns       : {lc_arr.shape[1]}')
    # print(f'Rows          : {lc_arr.shape[0]}')
    # print(f'Geotransform  : {lc_gt}')
    # print(f'Projection    : {lc_proj}')
    # print(f'Type          : {lc_arr.dtype}')

    # n = np.unique(lc_arr.filled(0))
    # print(f'Land cover classes ({len(n)}): {n}')

    # nrows = test_mask.shape[0]
    # ncols = test_mask.shape[1]
    # bands = 556  # the total features (spectral bands, VIs, and phenologic parameters)

    chunks = 1000
    nrows, ncols = 5765, 4181
    n = 26  # land cover classes
    bands = 556  # the total features (spectral bands, VIs, and phenologic parameters)

    input_shape = (nrows, ncols, bands)
    print(input_shape)

    # # Test 1: Iterate over chunks of the file using a generator
    # data = read_chunk(fn_train_feat, input_shape, chunk=1000)
    # for d in data:
    #     s = sys.getsizeof(d)
    #     print(f", type:{d.dtype} shape: {d.shape} size: {s} bytes ({s/(1024*1024):.2f} MB)")
        

    # # Test 2: iterate and get X and Y
    train_seq = gen_training_sequences(fn_train_feat, input_shape, fn_labels, chunk=chunks)
    
    for x, y in train_seq:
        s = sys.getsizeof(x)
        print(f", X: type={x.dtype} shape={x.shape} size={s} bytes ({s/(1024*1024):.2f} MB) ", end='')
        s = sys.getsizeof(y)
        print(f"Y: type={y.dtype} shape={y.shape} size={s} bytes ({s/(1024*1024):.2f} MB)")


    # # request a model
    # model = create_cnn(input_shape, n)

    # # # set training data, epochs and validation data
    # # kwargs.update(train_seq, epochs=10)

    # # # call fit, including any arguments supplied alongside the model
    # # # fit should call a generator to dynamically load the data from the HDF5 file
    # # model.fit(**kwargs)
    # model.fit(train_seq,
    #     epochs=10,
    #     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])



    # STEPS: 
    # Read input_shape & n_outputs... done.
    # Create the model............... done.
    # Fit the model........... in progress.
    # Accuracy assesment