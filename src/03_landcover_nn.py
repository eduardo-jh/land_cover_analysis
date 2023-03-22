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
    model.add(layers.Conv2D(16, 8, input_shape=input_shape, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(n_outputs, activation='softmax'))  # Predictions are categories

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Arguments for the fit function
    kwargs = {'callbacks': [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
              'batch_size': len(input_shape)}

    return model, kwargs


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

    nrows = train_mask.shape[0]
    ncols = train_mask.shape[1]
    bands = 556  # the total features (spectral bands, VIs, and phenologic parameters)

    shp = (nrows, ncols, bands)
    data = read_chunk(fn_train_feat, shp, chunk=1000)

    # Iterate over chunks of the file using a generator
    for d in data:
        s = sys.getsizeof(d)
        print(f", type:{d.dtype} shape: {d.shape} size: {s} bytes ({s/(1024*1024):.2f} MB)")
        
    # request a model
    model, kwargs = create_cnn(shp (shp[0], shp[1]))

    # # # set training data, epochs and validation data
    # # kwargs.update(x=train_in, y=train_out,
    # #               epochs=10, validation_data=(test_in, test_out))

    # # call fit, including any arguments supplied alongside the model
    # # fit should call a generator to dynamically load the data from the HDF5 file
    # model.fit(**kwargs)

    # next(data)
    # # print(data)
    # print(type(data))
    # print(dir(data))
    # # print(f"{data.shape}, {data.dtype}")
    # plt.imshow(data[:,:,555])
    # plt.show()
    # # Read input_shape & n_outputs
    # # Create the model
    # # Fit the model
    # # Accuracy assesment