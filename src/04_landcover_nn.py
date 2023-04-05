#!/usr/bin/env python
# coding: utf-8

""" Neural networks

Eduardo Jimenez <eduardojh@email.arizona.edu>

NOTE: run under 'rstf' conda environment (python 3.8.13, keras 2.9.0)
"""

import sys
import csv
# import os.path
import platform
import h5py
# import pandas as pd
import numpy as np
from math import ceil
# from matplotlib import pyplot as plt
from typing import Tuple, List, Dict
# from datetime import datetime
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


def read_params(filename: str) -> Dict:
    """ Reads the parameters from a CSV file """
    params = {}
    with open(filename, 'r') as csv_file:
        writer = csv.reader(csv_file, delimiter='=')
        for row in writer:
            if len(row) == 0:
                continue
            params[row[0]]=row[1]
    return params

def create_simple_model(in_shape: Tuple[int, int, int], n_output: int) -> Tuple[keras.models.Model, Dict]:
    model = keras.Sequential()

    # Add the layers
    model.add(layers.Dense(128, activation='relu', input_shape=in_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(n_output, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Arguments for the fit function
    # Stop training when 'val_loss' is no longer improving
    # "no longer improving" being defined as "no better than 1e-2 less"
    # "no longer improving" being further defined as "for at least 2 epochs"
    kwargs = {'callbacks': [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=2, verbose=1),
                            keras.callbacks.TensorBoard(log_dir=cwd + "logs.txt")]}

    keras.utils.plot_model(model, cwd + "simple_model.png", show_shapes=True)

    return model, kwargs

def create_cnn(input_shape: tuple, n_outputs: int) -> Tuple[keras.models.Model, Dict]:
    """ Create the model for land cover classification with CNN

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = keras.Sequential()

    # Add the layers
    model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(128, 7, activation='relu', data_format='channels_last'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, 7, activation='relu', data_format='channels_last'))
    model.add(layers.Dense(n_outputs, activation='softmax'))  # Predictions are categories

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Arguments for the fit function
    kwargs = {'callbacks': [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]}
    # return model
    return model, kwargs


def gen_training_sequences_img(X, Y, shape: Tuple[int, int, int], batch_size: int, n_classes: int, img_array: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """ Generate training X and Y (partial) sequences on-demand from a HDF5 file of size (nrows,ncols,bands)
    """

    nrows, ncols, nbands = shape
    img_rows, img_cols = img_array
    for row in range(img_rows):
        x = np.empty((batch_size, nrows, ncols, nbands))
        y = np.empty((batch_size, nrows, ncols, n_classes), dtype=np.uint8)
        for col in range(img_cols):
            name = f"r{row}c{col}"
            # print(f'  Dataset: {name}')
            x_data = X[name][:]
            x[col] = x_data
            y_data = keras.utils.to_categorical(Y['training/' + name][:], num_classes=n_classes)
            y[col] = y_data
            # y_data = Y['training/' + name][:]
            # y[col] = keras.utils.to_categorical(y_data, num_classes=n_classes)
            # print(f'  slice: x={x_data.shape} {x_data.dtype}, y={y_data.shape} {y_data.dtype}')
        # print(f'  x: {x.shape}, y: {y.shape}')
        yield x, y


def gen_validation_sequences_img(X, Y, shape: Tuple[int, int, int], batch_size: int, n_classes: int, img_array: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    nrows, ncols, nbands = shape
    img_rows, img_cols = img_array
    for row in range(img_rows):
        x = np.empty((batch_size, nrows, ncols, nbands))
        y = np.empty((batch_size, nrows, ncols, n_classes), dtype=np.uint8)
        for col in range(img_cols):
            name = f"r{row}c{col}"
            # print(f'  Dataset: {name} (testing)')
            x_data = X[name][:]
            x[col] = x_data
            y_data = keras.utils.to_categorical(Y['testing/' + name][:], num_classes=n_classes)
            y[col] = y_data
        yield x, y

if __name__ == '__main__':

    fn_train_feat = cwd + 'IMG_Calakmul_Training_Features.h5'
    fn_test_feat = cwd + 'IMG_Calakmul_Testing_Features.h5'
    fn_labels = cwd + 'IMG_Calakmul_Labels.h5'
    fn_parameters = cwd + 'dataset_parameters.csv'

    # Read the parameters saved from previous script to ensure matching
    parameters = read_params(fn_parameters)
    print(parameters)
    row_pixels, col_pixels = int(parameters['IMG_ROWS']), int(parameters['IMG_COLUMNS'])
    n_classes = int(parameters['NUM_CLASSES'])
    bands = int(parameters['LAYERS'])
    img_x_row = int(parameters['IMG_PER_ROW'])
    img_x_col = int(parameters['IMG_PER_COL'])
    batch_size = img_x_col  # batch size = one row

    input_shape = (row_pixels, col_pixels, bands)
    print(f'Input shape: {input_shape}')

    # # Test generator function: Iterate over chunks of the file using a generator
    # with h5py.File(fn_train_feat, 'r') as X, h5py.File(fn_labels, 'r') as Y:
    #     data = gen_training_sequences_img(X, Y, input_shape, 5, n_classes)
    #     for x, y in data:
    #         s_x = sys.getsizeof(x)
    #         s_y = sys.getsizeof(y)
    #         print(f"X type:{x.dtype} X shape: {x.shape} size: {s_x} bytes ({s_x/(1024*1024):.2f} MB)")
    #         print(f"Y type:{y.dtype} Y shape: {y.shape} size: {s_y} bytes ({s_y/(1024*1024):.2f} MB)\n")
    #         # print(x[2,400:405,400:405,0]) # get a sample, make sure no empty arrays
    #         print(y[2,0:5,400:405,23])
    #         print(np.sum(y))

    # Request a simple model
    model, kwargs = create_simple_model(input_shape, n_classes)

    # Train the model
    with h5py.File(fn_train_feat, 'r') as X_train, h5py.File(fn_labels, 'r') as Y_labels, h5py.File(fn_test_feat, 'r') as X_test:
        train_seq = gen_training_sequences_img(X_train, Y_labels, input_shape, batch_size, n_classes, (img_x_row, img_x_col))
        validation_seq = gen_validation_sequences_img(X_test, Y_labels, input_shape, batch_size, n_classes, (img_x_row, img_x_col))
        history = model.fit(train_seq,
                  validation_data=validation_seq,
                  **kwargs)
    print(history.history)
    print("Evaluate on test data")
    results = model.evaluate(validation_seq)
    print(results)
    
    # # Request a CNN model
    # model, kwargs = create_cnn(input_shape, n_classes)

    # # Train the model
    # with h5py.File(fn_train_feat, 'r') as X_train, h5py.File(fn_labels, 'r') as Y_labels, h5py.File(fn_test_feat, 'r') as X_test:
    #     train_seq = gen_training_sequences_img(X_train, Y_labels, input_shape, batch_size, n_classes, (img_x_row, img_x_col))
    #     validation_seq = gen_validation_sequences_img(X_test, Y_labels, input_shape, batch_size, n_classes, (img_x_row, img_x_col))
    #     history = model.fit(train_seq,
    #               validation_data=validation_seq,
    #               **kwargs)
    # print(history.history)
    # print("Evaluate on test data")
    # results = model.evaluate(validation_seq)
    # print(results)
    
    
