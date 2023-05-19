#!/usr/bin/env python
# coding: utf-8

""" Neural networks

Eduardo Jimenez <eduardojh@email.arizona.edu>

NOTE: run under 'rstf' conda environment (python 3.8.13, keras 2.9.0)
"""

import sys
# import csv
# import os.path
import platform
import h5py
# import pandas as pd
import numpy as np
from math import ceil
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict
from datetime import datetime
from tensorflow import keras
from keras import layers

if len(sys.argv) == 3:
    # Check if arguments were passed from terminal
    args = sys.argv[1:]
    sys.path.insert(0, args[0])
    cwd = args[1]
    print(f"  Using RS_LIB={args[0]}")
    print(f"  Using CWD={args[1]}")
else:
    import os
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

def create_simple_model(in_shape: Tuple[int, int, int], n_output: int) -> Tuple[keras.models.Model, Dict]:
    model = keras.Sequential()

    # Add the layers
    model.add(layers.Dense(200, activation='relu', input_shape=in_shape))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(n_output, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=['categorical_crossentropy', 'accuracy'])
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # Arguments for the fit function
    # Stop training when 'val_loss' is no longer improving
    # "no longer improving" being defined as "no better than 1e-2 less"
    # "no longer improving" being further defined as "for at least 2 epochs"
    kwargs = {'callbacks': [keras.callbacks.EarlyStopping(monitor='categorical_crossentropy', min_delta=0.01, patience=2, verbose=1),
                            keras.callbacks.TensorBoard(log_dir=cwd + "logs")]}

    keras.utils.plot_model(model, cwd + f"results/{datetime.strftime(start, fmt)}_ffn_simple_model.png", show_shapes=True)

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
    kwargs = {'callbacks': [keras.callbacks.EarlyStopping(monitor='categorical_crossentropy', patience=2)]}
    # return model
    return model, kwargs


def gen_training_sequences3(X, Y, in_shape: Tuple[int, int, int], batch_size: int, epochs: int, n_classes: int, img_array: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Generate training X and Y (partial) sequences on-demand from a HDF5 file of size (nrows,ncols,bands)
    """
    nrows, ncols, nbands = in_shape
    img_rows, img_cols = img_array
    for i in range(epochs):
        for row in range(img_rows):
            x = np.empty((batch_size, nrows, ncols, nbands))
            y = np.empty((batch_size, nrows, ncols, n_classes), dtype=np.uint8)
            for col in range(img_cols):
                name = f"r{row}c{col}"

                x_data = X[name][:]
                x[col] = x_data
                # y_data = keras.utils.to_categorical(Y['training/' + name][:], num_classes=n_classes)
                # y[col] = y_data

                y_data = Y['training/' + name][:]
                weights = np.where(y_data > 0, 1, 0)
                y[col] = keras.utils.to_categorical(y_data, num_classes=n_classes)
            yield x, y, weights
            # yield x, y


def gen_validation_sequences3(X, Y, in_shape: Tuple[int, int, int], batch_size: int, n_classes: int, img_array: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nrows, ncols, nbands = in_shape
    img_rows, img_cols = img_array
    for row in range(img_rows):
        x = np.empty((batch_size, nrows, ncols, nbands))
        y = np.empty((batch_size, nrows, ncols, n_classes), dtype=np.uint8)
        for col in range(img_cols):
            name = f"r{row}c{col}"

            x_data = X[name][:]
            x[col] = x_data
            # y_data = keras.utils.to_categorical(Y['testing/' + name][:], num_classes=n_classes)
            # y[col] = y_data

            y_data = Y['testing/' + name][:]
            weights = np.where(y_data > 0, 1, 0)
            y[col] = keras.utils.to_categorical(y_data, num_classes=n_classes)
        yield x, y, weights
        # yield x, y


def gen_training_sequences(X, Y, in_shape: Tuple[int, int, int], batch_size: int, epochs: int, n_classes: int, img_array: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """ Generate training X and Y (partial) sequences on-demand from a HDF5 file of size (nrows,ncols,bands)
    """
    nrows, ncols, nbands = in_shape
    img_rows, img_cols = img_array
    for i in range(epochs):
        for row in range(img_rows):
            x = np.empty((batch_size, nrows, ncols, nbands))
            y = np.empty((batch_size, nrows, ncols, n_classes), dtype=np.uint8)
            for col in range(img_cols):
                name = f"r{row}c{col}"

                x_data = X[name][:]
                x[col] = x_data
                y_data = keras.utils.to_categorical(Y['training/' + name][:], num_classes=n_classes)
                y[col] = y_data

            #     y_data = Y['training/' + name][:]
            #     weights = np.where(y_data > 0, 1, 0)
            #     y[col] = keras.utils.to_categorical(y_data, num_classes=n_classes)
            # yield x, y, weights
            yield x, y



def gen_validation_sequences(X, Y, in_shape: Tuple[int, int, int], batch_size: int, n_classes: int, img_array: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    nrows, ncols, nbands = in_shape
    img_rows, img_cols = img_array
    for row in range(img_rows):
        x = np.empty((batch_size, nrows, ncols, nbands))
        y = np.empty((batch_size, nrows, ncols, n_classes), dtype=np.uint8)
        for col in range(img_cols):
            name = f"r{row}c{col}"

            x_data = X[name][:]
            x[col] = x_data
            y_data = keras.utils.to_categorical(Y['testing/' + name][:], num_classes=n_classes)
            y[col] = y_data

        #     y_data = Y['testing/' + name][:]
        #     weights = np.where(y_data > 0, 1, 0)
        #     y[col] = keras.utils.to_categorical(y_data, num_classes=n_classes)
        # yield x, y, weights
        yield x, y


def gen_xtest_sequences(X, in_shape: Tuple[int, int, int], batch_size: int, img_array: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    nrows, ncols, nbands = in_shape
    img_rows, img_cols = img_array
    for row in range(img_rows):
        x = np.empty((batch_size, nrows, ncols, nbands))
        for col in range(img_cols):
            name = f"r{row}c{col}"
            # print(f'  Dataset: {name} (testing)')
            # x_data = X[name][:]
            # x[col] = x_data
            x[col] = X[name][:]
        yield x


if __name__ == '__main__':
    fmt = '%Y_%m_%d-%H_%M_%S'
    start = datetime.now()

    fn_test_mask = cwd + 'training/usv250s7cw_ROI1_testing_mask.tif'

    fn_features = cwd + 'Calakmul_Features_img.h5'
    fn_train_feat = cwd + 'Calakmul_Training_Features_img.h5'
    fn_test_feat = cwd + 'Calakmul_Testing_Features_img.h5'
    fn_labels = cwd + 'Calakmul_Labels_img.h5'

    fn_parameters = cwd + 'dataset_parameters.csv'
    fn_raster_pred = cwd + 'results/raster_predictions.tif'
    save_fig_preds = cwd + f'results/{datetime.strftime(start, fmt)}_ffn_predictions.png'
    save_predictions = cwd + f'results/{datetime.strftime(start, fmt)}_ffn_predictions.h5'

    # Read the parameters saved from previous script to ensure matching
    parameters = rs.read_params(fn_parameters)
    # print(parameters)
    row_pixels, col_pixels = int(parameters['IMG_ROWS']), int(parameters['IMG_COLUMNS'])
    n_classes = int(parameters['NUM_CLASSES'])
    bands = int(parameters['LAYERS'])
    img_x_row = int(parameters['IMG_PER_ROW'])
    img_x_col = int(parameters['IMG_PER_COL'])
    batch_size = img_x_col  # IMPORTANT: batch size is one row!

    input_shape = (row_pixels, col_pixels, bands)
    print(f'Input shape: {input_shape}')

    # Read the mask to use it as weights
    test_mask, nodata, metadata, geotransform, projection = rs.open_raster(fn_test_mask)
    print(f'Opening raster: {fn_test_mask}')
    print(f'Metadata      : {metadata}')
    print(f'NoData        : {nodata}')
    print(f'Columns       : {test_mask.shape[1]}')
    print(f'Rows          : {test_mask.shape[0]}')
    print(f'Geotransform  : {geotransform}')
    print(f'Projection    : {projection}')
    print(f'Type          : {test_mask.dtype}')

    # Generate weights for training
    train_weights = np.where(test_mask < 0.5, 1, 0)

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
    epochs = 4

    # Train the model
    with h5py.File(fn_train_feat, 'r') as X_train, h5py.File(fn_labels, 'r') as Y_labels, h5py.File(fn_test_feat, 'r') as X_test:
        train_seq = gen_training_sequences3(X_train, Y_labels, input_shape, batch_size, epochs, n_classes, (img_x_row, img_x_col))
        validation_seq = gen_validation_sequences3(X_test, Y_labels, input_shape, batch_size, n_classes, (img_x_row, img_x_col))
        # Generators don't need 'batch_size' on fit() and evaluate() functions!
        history = model.fit(train_seq,
                    validation_data=validation_seq,
                    # steps_per_epoch=img_x_row//batch_size, # this is 1
                    # validation_steps=img_x_row//batch_size,  # this is 1
                    # Try this
                    epochs=epochs,
                    steps_per_epoch=img_x_row,  # since batch is a row, the columns will complete the image in one epoch
                    validation_steps=img_x_row,
                    verbose=1,
                    **kwargs)
        print(history.history)
        print("Evaluate on test data")
        results = model.evaluate(validation_seq)
        print(print(f"test loss={results[0]}, test acc:{results[1]} ({len(results)})"))

    # Predict a land cover class for each pixel
    # with h5py.File(fn_features, 'r') as X_pred:
    with h5py.File(fn_test_feat, 'r') as X_pred:
        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        x_pred = gen_xtest_sequences(X_pred, input_shape, batch_size, (img_x_row, img_x_col))
        print("Generate predictions for x_test")
        predictions = model.predict(x_pred)
        print(f"Predictions shape: {predictions.shape}")

        # preds_arra = predictions[0:,1000,1000,1]
    
        # Save the predictions to a file
        rows = int(parameters['ROWS'])
        cols = int(parameters['COLUMNS'])
        preds_arr = np.empty((6000, 5000))
        preds_arr[:] = np.nan

        with h5py.File(save_predictions, 'w') as f_preds:
            i = 0
            for r in range(img_x_row):
                for c in range(img_x_col):
                    print(f'IMAGE {i+1}')
                    # Indices to slice array
                    r_str = r * row_pixels
                    r_end = r_str + row_pixels
                    c_str = c * col_pixels
                    c_end = c_str + col_pixels
                    # if r_end > rows:
                    #     r_end = rows
                    # if c_end > cols:
                    #     c_end = cols
                    print(f'  {r_str}:{r_end},{c_str}:{c_end}')

                    f_preds.create_dataset(f'preds_img_{i}', data=predictions[i])

                    # Get classes with highest probability per pixel
                    pred_classes = np.argmax(predictions[i], axis=2)
                    print(f'  {preds_arr.shape} {predictions[i].shape} {pred_classes.shape}')
                    # print(predictions[i,0:10,0:10,:])
                    # print(pred_classes[0:10,0:10])
                    preds_arr[r_str:r_end,c_str:c_end] = pred_classes[:]
                    i += 1
    # Create a figure
    plt.figure(figsize=(24,16))
    plt.imshow(preds_arr, cmap='viridis')
    plt.colorbar()
    plt.savefig(save_fig_preds, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    # # Create a raster
    # rs.create_raster(fn_raster_pred, preds_arr, int(parameters['EPSG']), )

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

    elapsed = start - datetime.now()