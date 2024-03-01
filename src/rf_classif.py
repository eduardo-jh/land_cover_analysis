#!/usr/bin/env python
# coding: utf-8

""" Land cover classification using Random Forest for Yucatan Peninsula (ROI2)

Reads features from HDF5 files (which had to be created from HDF4),
trains a Random Forest for the entire mosaic of ROI(2) and makes
predictions in a tile-by-tile fashion.

@author: Eduardo Jimenez Hernandez <eduardojh@arizona.edu>
@date: 2023-09-01 12:14:51.796779575 -0700

Changelog:
  2023-09-10: removed OOP, return to functions (bc KISS) because speckled image predictions.
  2023-09-25: model RUNS and WORKS the best so far! The only problem is saving trained model.
  2023-10-11: predictions ready! Fixed error of using NAN=0 (wrong) instead of NAN=-13000 (right). Training filtered by ROI2 mask.
  2024-02-08: New version created!!! 

  TODO: save the trained model
  - Run without saving the model, current approach... done!
  - Use an alternative to pickle (it fails with really big models)... pending
  - Complete algorithm: testing predictions, regularization error, confusion matrix... done!
  - Try different land cover labels other than current combined 11 classes:
    1) INEGI's originals... pending
    2) combined agriculture & savanna... pending
  
"""

import gc
import sys
import os
import csv
import h5py
# import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import forestci as fci
from joblib import dump, load
from datetime import datetime
from sklearn.tree import export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, cohen_kappa_score


sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs


def incorporate_ancillary(fn_raster, ancillary_dict, **kwargs):
    """ Incorporates a list of ancillary rasters to its corresponding land cover class
    Ancillary data has to be 1's for data and 0's for NoData.
    Ancillary data must exist in the same directory.
    WARNING: raster files should have same shape and spatial reference!
    """
    ancillary_suffix = kwargs.get("suffix", "_ancillary.tif")
    dataset, nodata, geotransform, spatial_ref = rs.open_raster(fn_raster)
    ancillary_dir, lc_basename = os.path.split(fn_raster)
    # print(ancillary_dir, lc_basename)

    for key in ancillary_dict.keys():
        for ancillary_file in ancillary_dict[key]:
            fn = os.path.join(ancillary_dir, ancillary_file)
            print(f"Incorporating ancillary raster: {fn}")
            assert os.path.isfile(fn), f"File not found: {fn}"
            
            # Read ancillary data
            ancillary, _, _, _ = rs.open_raster(fn)
            
            assert ancillary.shape == dataset.shape, f"Shape doesn't match: {ancillary.shape} and {dataset.shape}"
            dataset = np.where(ancillary > 0, key, dataset)
    
    # Save the land cover with the integrated ancillary data
    fn_landcover = os.path.join(ancillary_dir, lc_basename[:-4] + ancillary_suffix)
    rs.create_raster(fn_landcover, dataset, spatial_ref, geotransform)
    print(f"Land cover raster is now: {fn_landcover}")


def sample(cwd, fn_landcover, **kwargs):
    """ Stratified random sampling

    :param float train_percent: default training-testing proportion is 80-20%
    :param int win_size: default is sampling a window of 7x7 pixels
    :param int max_trials: max of attempts to fill the sample size
    """

    start = datetime.now()
    print(f"{start}: starting stratified random sampling.")
    
    train_percent = kwargs.get("train_percent", 0.2)
    window_size = kwargs.get("window_size", 7)
    max_trials = int(kwargs.get("max_trials", 2e5))

    sampling_suffix = kwargs.get("sampling_suffix", "sampling")
    fn_training_mask = kwargs.get("training_mask", "training_mask.tif")
    fn_training_labels = kwargs.get("training_labels", "training_labels.tif")
    fn_sample_sizes = kwargs.get("sample_sizes", "dataset_sample_sizes.csv")
    fig_frequencies = kwargs.get("fig_frequencies", "class_fequencies.png")

    sampling_dir = os.path.join(cwd, sampling_suffix)
    if not os.path.exists(sampling_dir):
        print(f"\nCreating new path: {sampling_dir}")
        os.makedirs(sampling_dir)
    
    fn_training_mask = os.path.join(cwd, sampling_suffix, fn_training_mask)
    fn_training_labels = os.path.join(cwd, sampling_suffix, fn_training_labels)
    fig_frequencies = os.path.join(cwd, sampling_suffix, fig_frequencies)
    fn_sample_sizes = os.path.join(cwd, sampling_suffix, fn_sample_sizes)

    # Read the land cover raster and retrive the land cover classes
    assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
    land_cover, nodata, geotransform, spatial_reference = rs.open_raster(fn_landcover)
    print(f'  Opening raster: {fn_landcover}')
    print(f'    --NoData        : {nodata}')
    print(f'    --Columns       : {land_cover.shape[1]}')
    print(f'    --Rows          : {land_cover.shape[0]}')
    print(f'    --Geotransform  : {geotransform}')
    print(f'    --Spatial ref.  : {spatial_reference}')
    print(f'    --Type          : {land_cover.dtype}')

    land_cover = land_cover.filled(0)

    # Create a list of land cover keys and its area covered percentage
    landcover_frequencies = rs.land_cover_freq(fn_landcover, verbose=False, sort=True)
    print(f'  --Land cover frequencies: {landcover_frequencies}')
    
    classes = list(landcover_frequencies.keys())
    freqs = [landcover_frequencies[x] for x in classes]  # pixel count
    percentages = (freqs/sum(freqs))*100

    # Plot land cover percentage horizontal bar
    print('  --Plotting land cover percentages...')
    rs.plot_land_cover_hbar(classes, percentages, fig_frequencies,
        title='INEGI Land Cover Classes in Yucatan Peninsula',
        xlabel='Percentage (based on pixel count)',
        ylabel='Land Cover (Grouped)',  # remove if not grouped
        xlims=(0,60))

    #### Sample size == testing dataset
    # Use a dataframe to calculate sample size
    df = pd.DataFrame({'Key': classes, 'PixelCount': freqs, 'Percent': percentages})
    df['TrainPixels'] = (df['PixelCount']*train_percent).astype(int)
    # print(df['TrainPixels'])

    # Now calculate percentages
    df['TrainPercent'] = (df['TrainPixels'] / df['PixelCount'])*100
    print(df)

    nrows, ncols = land_cover.shape
    print(f"  --Total pixels={nrows*ncols}, Values={sum(df['PixelCount'])}, NoData/Missing={nrows*ncols - sum(df['PixelCount'])}")

    sample = {}  # to save the sample

    # Create a mask of the sampled regions
    training_mask = np.zeros(land_cover.shape, dtype=land_cover.dtype)

    # A window will be used for sampling, this array will hold the sample
    window_sample = np.zeros((window_size,window_size), dtype=int)

    print(f'  --Max trials: {max_trials}')

    trials = 0  # attempts to complete the sample
    completed = {}  # classes which sample is complete

    for sample_key in list(df['Key']):
        completed[sample_key] = False
    completed_samples = sum(list(completed.values()))  # Values are all True if completed
    total_classes = len(completed.keys())
    # print(completed)

    sampled_points = []

    while (trials < max_trials and completed_samples < total_classes):
        show_progress = (trials%10000 == 0)  # Step to show progress
        if show_progress:
            print(f'  --Trial {1 if trials == 0 else trials:>8} of {max_trials:>8} ', end='')

        # 1) Generate a random point (row_sample, col_sample) to sample the array
        #    Coordinates relative to array positions [0:nrows, 0:ncols]
        #    Subtract half the window_size to avoid sampling too close to the edges, use window_size step to avoid overlapping
        col_sample = random.randrange(0 + window_size//2, ncols - window_size//2, window_size)
        row_sample = random.randrange(0 + window_size//2, nrows - window_size//2, window_size)

        # Save the points previously sampled to avoid repeating and oversampling
        point = (row_sample, col_sample)
        if point in sampled_points:
            trials +=1
            continue
        else:
            sampled_points.append(point)

        # 2) Generate a sample window around the random point, here create the boundaries,
        #    these rows and columns will be used to slice the sample
        win_col_ini = col_sample - window_size//2
        win_col_end = col_sample + window_size//2 + 1  # add 1 to slice correctly
        win_row_ini = row_sample - window_size//2
        win_row_end = row_sample + window_size//2 + 1

        assert win_col_ini < win_col_end, f"Incorrect slice indices on x-axis: {win_col_ini} < {win_col_end}"
        assert win_row_ini < win_row_end, f"Incorrect slice indices on y-axis: {win_row_ini} < {win_row_end}"

        # 3) Check if sample window is out of range, if so trim the window to the array's edges accordingly
        #    This may not be necessary if half the window size is subtracted, but still
        if win_col_ini < 0:
            # print(f'    --Adjusting win_col_ini: {win_col_ini} to 0')
            win_col_ini = 0
        if win_col_end > ncols:
            # print(f'    --Adjusting win_col_end: {win_col_end} to {ncols}')
            win_col_end = ncols
        if win_row_ini < 0:
            # print(f'    --Adjusting win_row_ini: {win_row_ini} to 0')
            win_row_ini = 0
        if  win_row_end > nrows:
            # print(f'    --Adjusting win_row_end: {win_row_end} to {nrows}')
            win_row_end = nrows

        # 4) Check and adjust the shapes of the arrays to slice and insert properly, only final row/column can be adjusted
        window_sample = land_cover[win_row_ini:win_row_end,win_col_ini:win_col_end]
        # print(window_sample)
        
        # 5) Get the unique values in sample (sample_keys) and its count (sample_freq)
        sample_keys, sample_freq = np.unique(window_sample, return_counts=True)
        classes_to_remove = []  # Avoid adding zeros or completed classes to the mask

        # 6) Iterate over each class sample and add its respective pixel count to the sample
        for sample_class, class_count in zip(sample_keys, sample_freq):
            if sample_class == nodata:
                # Sample is mixed with zeros, tag it to remove it and go to next sample_class
                classes_to_remove.append(sample_class)
                continue

            if completed.get(sample_class, False):
                classes_to_remove.append(sample_class)  # do not add completed classes to the sample
                continue

            # Accumulate the pixel counts, chek first if general sample is completed
            if sample.get(sample_class) is None:
                sample[sample_class] = class_count
            else:
                # if sample[sample_class] < sample_sizes[sample_class]:
                sample_size = df[df['Key'] == sample_class]['TrainPixels'].item()

                # If sample isn't completed, add the sampled window
                if sample[sample_class] < sample_size:
                    sample[sample_class] += class_count
                    # Check if last addition completed the sample
                    if sample[sample_class] >= sample_size:
                        completed[sample_class] = True  # this class' sample is now complete
                        # but do not add to classes_to_remove
                else:
                    # This class' sample was completed already
                    completed[sample_class] = True
                    classes_to_remove.append(sample_class)

        # Create an array containing all the sampled pixels by adding the sampled windows from each quadrant (or part)
        sampled_window = np.zeros(window_sample.shape, dtype=land_cover.dtype)
        
        # Filter out classes with already complete samples
        if len(classes_to_remove) > 0:
            for single_class in classes_to_remove:
                # Put a 1 on a complete class
                filter_out = np.where(window_sample == single_class, 1, 0)
                sampled_window += filter_out.astype(land_cover.dtype)
            
            # All values greater than zero are pixels to remove from mask, reverse it so 1's are the sample mask
            sampled_window = np.where(sampled_window == 0, 1, 0)
        else:
            sampled_window = window_sample[:,:].astype(land_cover.dtype)
        
        # Slice and insert sampled window
        # print(training_mask[win_row_ini:win_row_end,win_col_ini:win_col_end].dtype, sampled_window.dtype)
        training_mask[win_row_ini:win_row_end,win_col_ini:win_col_end] += sampled_window.astype(land_cover.dtype)

        trials += 1

        completed_samples = sum(list(completed.values()))  # Values are all True if completed
        if show_progress:
            print(f' (completed {completed_samples:>2}/{total_classes:>2} samples)')
        if completed_samples >= total_classes:
            print(f'\n  --All samples completed in {trials} trials! Exiting.\n')

    if trials == max_trials:
        print('\n  --WARNING! Max trials reached, samples may be incomplete, try increasing max trials.')

    print(f'  --Sample sizes per class: {sample}')
    print(f'  --Completed samples: {completed}')

    print('\n  --WARNING! This may contain oversampling caused by overlapping windows!')
    df['SampledPixels'] = [sample.get(x,0) for x in df['Key']]
    df['SampledPercent'] = (df['SampledPixels'] / df['TrainPixels']) * 100
    df['SampledPerClass'] = (df['SampledPixels'] / df['PixelCount']) * 100
    df['SampleComplete'] = [completed[x] for x in df['Key']]
    df.to_csv(fn_sample_sizes)
    print(df)

    # Convert the training_mask to 1's (indicating pixels to sample) and 0's
    training_mask = np.where(training_mask >= 1, 1, 0)
    print(f"  --Values in mask: {np.unique(training_mask)}")  # should be 1 and 0

    # Create a raster with actual labels (land cover classes)
    training_labels = np.where(training_mask > 0, land_cover, 0)
    print(f"Creating raster: {fn_training_labels}")
    rs.create_raster(fn_training_labels, training_labels, spatial_reference, geotransform)

    # Create a raster with the sampled windows, this will be the sampling mask
    print(f"Creating raster: {fn_training_mask}")
    rs.create_raster(fn_training_mask, training_mask, spatial_reference, geotransform)

    end = datetime.now()
    print(f"\n{end}: ========== Stratified random sampling elapsed in: {end - start} ==========")


def performance_assessment(y_true: np.ndarray, y_predictions: np.ndarray, y_mask: np.ndarray, **kwargs):
    """ Performance assessment
    
    :param y_true: array of true labels
    :param y_predictions: array of predicted values
    :param y_mask: array of NoData values to exclude from the analysis (0=NoData, 1=Data)
    """

    _prefix = kwargs.get('prefix', '')
    _fn_save_crosstab = kwargs.get('fn_xtab', '')
    _fn_save_conf_tbl = kwargs.get('fn_conf', '')
    _fn_save_fig_prod = kwargs.get('fn_fig_prod', '')
    _fn_save_fig_user = kwargs.get('fn_fig_user', '')
    _fn_save_class_report = kwargs.get('fn_report', '')

    file_names = [_fn_save_crosstab, _fn_save_conf_tbl, _fn_save_fig_prod, _fn_save_fig_user, _fn_save_class_report]

    class_names_ = np.unique(y_true)
    print(f"y_predictions: {y_predictions.dtype}, {np.unique(y_predictions)}, {y_predictions.shape} ")
    print(f"y_true:        {y_true.dtype} {class_names_}, {y_true.shape}")
    print(f"y_mask:        {y_mask.dtype}, {np.unique(y_mask)}, {y_mask.shape}")

    class_names = [str(x) for x in class_names_]
    print(f"Class names for cross-tabulation: {class_names}")

    df_pred = pd.DataFrame({'truth': y_true, 'predict': y_predictions})
    crosstab_pred = pd.crosstab(df_pred['truth'], df_pred['predict'], margins=True)
    crosstab_pred.to_csv(_fn_save_crosstab)
    print(f'Saving crosstabulation: {_fn_save_crosstab}')

    accuracy = accuracy_score(y_true, y_predictions)
    print(f'***Accuracy score: {accuracy:>0.4f}***')

    cm = confusion_matrix(y_true, y_predictions)
    print(f'Saving confusion matrix: {_fn_save_conf_tbl}')
    with open(_fn_save_conf_tbl, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for single_row in cm:
            writer.writerow(single_row)
    
    ###### PRODUCER'S ACCURACY: confusion matrix normalized by row
    print("Generating normalized confusion matrix plot for producer's accuracy")
    title = "Normalized confusion matrix (producer's accuracy)"
    
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_predictions,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize="true",
    )
    disp.figure_.set_figwidth(16)
    disp.figure_.set_figheight(12)
    disp.ax_.set_title(title)

    print(f"Saving confusion matrix figure producer's accuracy: {_fn_save_fig_prod}")
    disp.figure_.savefig(_fn_save_fig_prod, bbox_inches='tight')
    
    ###### USER'S ACCURACY: confussion matrix normalized by column
    print("Generating normalized confusion matrix plot for user's accuracy")
    title = "Normalized confusion matrix (user's accuracy)"

    # Create the normalized confussion matrix for user's accuracy
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_predictions,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize='pred',  # IMPORTANT: normalize by predicted conditions (user's accuracy)
    )
    disp.figure_.set_figwidth(16)
    disp.figure_.set_figheight(12)
    disp.ax_.set_title(title)

    print(f"Saving confusion matrix figure user's accuracy: {_fn_save_fig_user}")
    disp.figure_.savefig(_fn_save_fig_user, bbox_inches='tight')

    ##### Finally, perform kappa analysis #####

    print('Running Cohens kappa analysis:')
    kappa = cohen_kappa_score(y_predictions, y_true)
    print(f"kappa: {kappa}")

    # Generate a complete classification report

    report = classification_report(y_true, y_predictions, )
    print(f'Saving classification report: {_fn_save_class_report}')
    print(report)
    with open(_fn_save_class_report, 'w') as f:
        f.write(report)

    return kappa, accuracy, file_names


def landcover_classification(cwd, stats_dir, pheno_dir, fn_landcover, fn_mask, fn_tiles, **kwargs):
    """A function to control each execution of the land cover classification code.
    The blocks of code that will run are passed as keyword arguments.

    :param str cwd: current working directory where sampling, features, and results will be created.
    :param str stats_dir: directory with HDF4 files containing the surface reflectance bands.
    :param str pheno_dir: directory with HDF4 files containing the phenology metrics.
    :param str fn_landcover: path of the GeoTIFF file with the land cover data.
    :param str fn_mask: GeoTIFF file path with NoData mask used to filter features creation, training, and predictions.
    :return None: all outputs are saved as files.
    """

    exec_start = datetime.now()

    _read_split = kwargs.get("read_split", False)
    _train_model = kwargs.get("train_model", True)
    _save_model = kwargs.get("save_model", True)
    _use_trained_model = kwargs.get("use_trained_model", False)
    _pretrained_model = kwargs.get("trained_model", '')
    _use_seasonal_features = kwargs.get("use_seasonal_features", True)
    _save_monthly_dataset = kwargs.get("save_monthly_dataset", False)
    _save_seasonal_dataset = kwargs.get("save_seasonal_dataset", False)
    _predict_mosaic = kwargs.get("predict_mosaic", True)
    _override_tiles = kwargs.get("override_tiles", None)
    _exclude_feats = kwargs.get("exclude_feats", None)
    _nan_value =kwargs.get("nan", -13000)
    _sample_dir = kwargs.get("sample_dir", "sampling")
    _feat_dir = kwargs.get("features_dir", "features")
    _save_mosaic_labels =  kwargs.get("save_mosaic_labels", False)

    FILL = kwargs.get("fill", False)
    NORMALIZE = kwargs.get("normalize", False)
    STANDARDIZE = kwargs.get("standardize", False) # Either normalize or standardize, not both!
    # If monthly datasets are not created/saved the fill, normalize, and standardize optios have no effect!

    if _train_model:
        # When training the algorithm, reading and splitting the features dataset is mandatory
        _read_split = True

    if _use_trained_model:
        # Check the trained model exists
        assert os.path.isfile(_pretrained_model), f"Model file not found: {_pretrained_model}"
        # When using a previously trained model, reading and splitting the features dataset is unnecessary
        _read_split = False

    # _nan_value = 0
    tile_rows = 5000
    tile_cols = 5000
    fmt = '%Y_%m_%d-%H_%M_%S'

    #=============================================================================
    # Read land cover and training mask rasters and create labels
    #=============================================================================

    fn_train_mask = os.path.join(cwd, _sample_dir, 'training_mask.tif')
    fn_test_mask = os.path.join(cwd, _sample_dir, 'testing_mask_filtered.tif')  # file to create testing mask

    # Read a raster with the location of the training sites
    assert os.path.isfile(fn_train_mask) is True, f"ERROR: File not found! {fn_train_mask}"
    train_mask, nodata, geotransform, spatial_ref = rs.open_raster(fn_train_mask)
    print(f'  Opening raster: {fn_train_mask}')
    print(f'    --NoData        : {nodata}')
    print(f'    --Columns       : {train_mask.shape[1]}')
    print(f'    --Rows          : {train_mask.shape[0]}')
    print(f'    --Geotransform  : {geotransform}')
    print(f'    --Spatial ref.  : {spatial_ref}')
    print(f'    --Type          : {train_mask.dtype}')

    # Read the land cover raster and retrive the land cover classes
    assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
    land_cover, lc_nd, lc_gt, lc_sp_ref = rs.open_raster(fn_landcover)
    print(f'  Opening raster: {fn_landcover}')
    print(f'    --NoData        : {lc_nd}')
    print(f'    --Columns       : {land_cover.shape[1]}')
    print(f'    --Rows          : {land_cover.shape[0]}')
    print(f'    --Geotransform  : {lc_gt}')
    print(f'    --Spatial ref.  : {lc_sp_ref}')
    print(f'    --Type          : {land_cover.dtype}')

    # Read the Yucatan Peninsula Aquifer to filter data
    assert os.path.isfile(fn_mask) is True, f"ERROR: File not found! {fn_mask}"
    nodata_mask, mask_nd, mask_gt, mask_sp_ref = rs.open_raster(fn_mask)
    print(f'  Opening raster: {fn_mask}')
    print(f'    --NoData        : {mask_nd}')
    print(f'    --Columns       : {nodata_mask.shape[1]}')
    print(f'    --Rows          : {nodata_mask.shape[0]}')
    print(f'    --Geotransform  : {mask_gt}')
    print(f'    --Spatial ref.  : {mask_sp_ref}')
    print(f'    --Type          : {nodata_mask.dtype}')

    print('  Analyzing labels from testing dataset (land cover classes)')
    # land_cover = land_cover.astype(train_mask.dtype)
    land_cover = land_cover.astype(np.int16)
    nodata_mask = nodata_mask.filled(0)
    nodata_mask = nodata_mask.astype(np.int16)

    # Filter land cover labels and training mask
    print(f"Filtering all rasters by: {fn_mask}")
    print(f'  --nodata_mask: {nodata_mask.dtype}, unique:{np.unique(nodata_mask)}, {nodata_mask.shape}')
    train_mask = np.where(nodata_mask == 1, train_mask.filled(0), _nan_value)
    train_labels = np.where(train_mask == 1, land_cover.filled(0), _nan_value)  # Training mask with ctual labels (land cover classes)
    test_mask = np.where(train_mask == 0, nodata_mask, _nan_value)  # testing = not training, but pixels with data
    land_cover = np.where(nodata_mask == 1, land_cover.filled(0), _nan_value)

    print(f'  --train_mask: {train_mask.dtype}, unique:{np.unique(train_mask)}, {train_mask.shape}')
    print(f'  --test_mask: {test_mask.dtype}, unique:{np.unique(test_mask)}, {test_mask.shape}')
    print(f'  --land_cover: {land_cover.dtype}, unique:{np.unique(land_cover)}, {land_cover.shape}')
    print(f'  --train_arr: {train_labels.dtype}, unique:{np.unique(train_labels)}, {train_labels.shape}')

    # Save the filtered rasters
    rs.create_raster(fn_landcover[:-4] + '_filtered.tif', land_cover, spatial_ref, geotransform)
    rs.create_raster(fn_train_mask[:-4] + '_labels_filtered.tif', train_labels, spatial_ref, geotransform)
    rs.create_raster(fn_train_mask[:-4] + '_filtered.tif', train_mask, spatial_ref, geotransform)
    rs.create_raster(fn_test_mask, test_mask, spatial_ref, geotransform)

    # Create the directory to save the feature files
    _features_path = os.path.join(cwd, _feat_dir)
    if not os.path.exists(_features_path):
        print(f"Creating path for features: {_features_path}")
        os.makedirs(_features_path)

    # Save the entire mosaic land cover labels, training mask, and 'No Data' mask
    fn_mosaic_labels = os.path.join(cwd, _feat_dir, 'mosaic_labels.h5')
    if _save_mosaic_labels:
        h5_mosaic_labels = h5py.File(fn_mosaic_labels, 'w')
        h5_mosaic_labels.create_dataset('land_cover', land_cover.shape, data=land_cover)
        h5_mosaic_labels.create_dataset('train_mask', land_cover.shape, data=train_mask)
        h5_mosaic_labels.create_dataset('no_data_mask', land_cover.shape, data=nodata_mask)

    # Read tiles names and extent (in Albers projection)
    tiles_extent = {}
    tiles = []
    print(f"Read tiles extent from file:")
    with open(fn_tiles, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            print(row)
            row_dict = {}
            for item in row[1:]:
                itemlst = item.split('=')
                row_dict[itemlst[0].strip()] = int(float(itemlst[1]))
            tiles_extent[row[0]] = row_dict
            tiles.append(row[0])
    if _override_tiles is not None:
        print(f"========== Overriding tiles ==========")
        tiles = _override_tiles
    print(tiles)

    #=============================================================================
    # Create feature and file names used for classification
    #=============================================================================

    bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
    band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    # nmonths = [x for x in range(1, 13)]
    vars = ['AVG', 'STDEV']
    phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
    phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

    # Generate feature names...
    print('')
    feat_indices = []
    feat_names = []
    feature = 0
    for j, band in enumerate(bands):
        for i, month in enumerate(months):
            for var in vars:
                # Create the name of the dataset in the HDF
                feat_name = month + ' ' + band_num[j] + ' (' + band + ') ' + var
                if band_num[j] == '':
                    feat_name = month + ' ' + band.upper() + ' ' + var
                # print(f'  Feature: {feature} Variable: {feat_name}')
                feat_names.append(feat_name)
                feat_indices.append(feature)
                feature += 1
    for param in phen+phen2:
        feat_name = 'PHEN ' + param
        # print(f'  Feature: {feature} Variable: {feat_name}')
        feat_names.append(feat_name)
        feat_indices.append(feature)
        feature += 1

    # Create a directory with the time execution started to save results
    results_path = os.path.join(cwd, 'results', f"{datetime.strftime(exec_start, fmt)}")
    if not os.path.exists(results_path):
        print(f"Creating path for results: {results_path}")
        os.makedirs(results_path)

    # Configure file names to save model parameters
    fn_save_model = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_model.pkl")
    fn_save_importance = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_feat_importance.csv")
    fn_save_imp_fig = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_feat_importance.png")
    fn_save_crosstab = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_crosstabulation.csv")
    fn_save_crosstab_train = fn_save_crosstab[:-4] + '_train.csv'
    fn_save_crosstab_test = fn_save_crosstab[:-4] + '_test.csv'
    fn_save_conf_tbl = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_confussion_matrix.csv")
    fn_save_conf_tbl_train = fn_save_conf_tbl[:-4] + '_train.csv'
    fn_save_conf_tbl_test = fn_save_conf_tbl[:-4] + '_test.csv'
    fn_save_classif_report = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_classif_report.txt")
    fn_save_preds_fig = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_predictions.png")
    fn_save_trees = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_tree.txt")
    fn_save_preds_raster = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_predictions.tif")
    fn_save_preds_h5 = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_predictions.h5")
    fn_save_params = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_run_parameters.csv")
    fn_save_conf_matrix_fig = fn_save_conf_tbl[:-4] + '.png'
    fn_save_probabilities = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_probabilities.h5")
    # fn_save_train_error_fig = os.path.join(results_path, f"rf_training_error.png")

    # Save the list of features
    fn_feat_list = os.path.join(results_path, f'{datetime.strftime(exec_start, fmt)}_feature_list.csv')
    print("Generated monthly features:")
    with open(fn_feat_list, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for n_feat, feat in zip(feat_indices, feat_names):
            print(f"{n_feat:>4}: {feat}")
            csv_writer.writerow([n_feat, feat])

    # *** Generate monthly feature datasets ***

    # Extent of entire mosaic will be N-S, and W-E boundaries
    mosaic_extension = {}
    mosaic_extension['W'], xres, _, mosaic_extension['N'], _, yres = [int(x) for x in geotransform]
    # print(geotransform)
    # print(mosaic_extension)
    mosaic_extension['E'] = mosaic_extension['W'] + tile_cols*xres
    mosaic_extension['S'] = mosaic_extension['N'] + tile_rows*yres
    print(mosaic_extension)

    for tile in tiles:
        print(f"\n==Creating monthly feature dataset for tile: {tile}==")
        # Create directories to save labels and features
        tile_feature_path = os.path.join(cwd, _feat_dir, tile)
        if not os.path.exists(tile_feature_path):
            print(f"\nCreating features path: {tile_feature_path}")
            os.makedirs(tile_feature_path)

        # Read HDF4 monthly data per tile, first generate feature names

        # Get HDF4 file names to read from the feature name
        fn_phenology = os.path.join(pheno_dir, tile, 'LANDSAT08.PHEN.NDVI_S1.hdf')  # Phenology files
        fn_phenology2 = os.path.join(pheno_dir, tile, 'LANDSAT08.PHEN.NDVI_S2.hdf')

        feat_type = {'BAND': [], 'PHEN1': [], 'PHEN2': []}
        for n, f in zip(feat_indices, feat_names):
            nparts = f.split(' ')
            if len(nparts) == 2:
                if nparts[1] in phen:
                    fn = fn_phenology
                    feat_type['PHEN1'].append((f, fn))
                elif nparts[1] in phen2:
                    fn = fn_phenology2
                    feat_type['PHEN2'].append((f, fn))
            elif len(nparts) == 3:
                month = nparts[0]
                band = nparts[1]
                fn = os.path.join(stats_dir, tile, 'MONTHLY.' + band.upper() + '.' + month + '.hdf')
                feat_type['BAND'].append((f, fn))
            elif len(nparts) == 4:
                month = nparts[0]
                band = nparts[2][1:-1]  # remove parenthesis
                band = band.upper()
                fn = os.path.join(stats_dir, tile, 'MONTHLY.' + band.upper() + '.' + month + '.hdf')
                feat_type['BAND'].append((f, fn))
            # print(f"{n:>3}: {f} --> {fn}")
        print(f"Finished {n} monthly feature names.")

        # ********* Create HDF5 files to save monthly features *********

        if not _save_monthly_dataset:
            print('Saving monthly features skipped or already exist!')

        if _save_monthly_dataset:
            # Actually save the monthly dataset as H5 files
            fn_features_tile = os.path.join(cwd, _feat_dir, tile, f'features_{tile}.h5')
            fn_labels_tile = os.path.join(cwd, _feat_dir, tile, f'labels_{tile}.h5')

            h5_features_tile = h5py.File(fn_features_tile, 'w')
            h5_labels_tile = h5py.File(fn_labels_tile, 'w')

            # Calculate slice coodinates to extract the tile
            tile_ext = tiles_extent[tile]

            # Get row for Nort and South and column for West and East
            nrow = (tile_ext['N'] - mosaic_extension['N'])//yres
            srow = (tile_ext['S'] - mosaic_extension['N'])//yres
            wcol = (tile_ext['W'] - mosaic_extension['W'])//xres
            ecol = (tile_ext['E'] - mosaic_extension['W'])//xres
            print(f"{tile}: N={nrow} S={srow} W={wcol} E={ecol}")
            
            tile_geotransform = (tile_ext['W'], xres, 0, tile_ext['N'], 0, yres)
            print(f"Tile geotransform: {tile_geotransform}")

            # Slice the labels from raster
            tile_landcover = land_cover[nrow:srow, wcol:ecol]
            print(f"Slice: {nrow}:{srow}, {wcol}:{ecol} {tile_landcover.shape}")

            # Slice the training mask and the 'NoData' mask
            tile_training_mask = train_mask[nrow:srow, wcol:ecol]
            # tile_nodata = no_data_arr[nrow:srow, wcol:ecol]
            tile_nodata = nodata_mask[nrow:srow, wcol:ecol]

            # Save the training mask, land cover labels, and 'NoData' mask
            h5_labels_tile.create_dataset('land_cover', (tile_rows, tile_cols), data=tile_landcover)
            h5_labels_tile.create_dataset('train_mask', (tile_rows, tile_cols), data=tile_training_mask)
            h5_labels_tile.create_dataset('no_data_mask', (tile_rows, tile_cols), data=tile_nodata)

            # Process each dataset according its feature type
            feat_index = 0
            print('Spectral bands...')
            for feat, fn in feat_type['BAND']:
                # print(f"--{feat_index:>3}: {feat} --> {fn}")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"

                # Extract data and filter by training mask
                feat_arr = rs.read_from_hdf(fn, feat[4:], np.int16)  # Use HDF4 method

                # Filter the features by the 'NoData' mask
                assert tile_nodata.shape == feat_arr.shape, f"Dimensions don't match {tile_nodata.shape}!={feat_arr.shape}"
                feat_arr = np.where(tile_nodata == 1, feat_arr, _nan_value)

                ### Fill missing data
                if FILL:
                    minimum = 0  # set minimum for spectral bands
                    max_row, max_col = None, None
                    if band.upper() in ['NDVI', 'EVI', 'EVI2']:
                        minimum = -10000  # minimum for VIs
                    feat_arr = rs.fill_with_mean(feat_arr, minimum, var=band.upper(), verbose=False)

                # Normalize or standardize
                if NORMALIZE:
                    feat_arr = rs.normalize(feat_arr)

                # # Apply scale factor to spectral bands (will transform from int16 to float)
                # # (Check Table 8 page 58 from 'VIP ESDRs ATBD User Guide')
                # print(f"Applying scale factor to {feat}")
                # feat_arr = feat_arr.astype(float)
                # feature_values = np.empty(feat_arr.shape, dtype=float)
                # feature_values = np.where(feat_arr < 0, -1, feat_arr/10000.)

                # Save features for the complete raster
                # h5_features_tile.create_dataset(feat, (tile_rows, tile_cols), data=feature_values)
                h5_features_tile.create_dataset(feat, (tile_rows, tile_cols), data=feat_arr)
                feat_index += 1

            print('Phenology...')
            for feat, fn in feat_type['PHEN1']:
                # print(f"--{feat_index:>3}: {feat} --> {fn}")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                param = feat[5:]
                
                # No need to fill missing values, just read the values
                pheno_arr = rs.read_from_hdf(fn_phenology, param, np.int16)  # Use HDF4 method
                
                # Filter phenology features by the NoData mask
                assert tile_nodata.shape == pheno_arr.shape, f"Dimensions don't match {tile_nodata.shape}!={pheno_arr.shape}"
                # pheno_arr = np.where(tile_nodata == 1, pheno_arr, -1)  # _nan_value=-1 for phenology
                pheno_arr = np.where(tile_nodata == 1, pheno_arr, _nan_value)

                # Fix phenology values larger than a year
                if param in ['SOS', 'EOS', 'LOS']:
                    pheno_arr = rs.fix_annual_phenology(pheno_arr)
                
                # Group all negative values
                if param in ['GDR', 'GUR']:
                    pheno_arr = np.where(pheno_arr < 0, _nan_value, pheno_arr)

                # Fill missing data
                if FILL:
                    if param == 'SOS':
                        print(f'  --Filling {param}')
                        minimum = 0
                        sos = rs.read_from_hdf(fn_phenology, 'SOS', np.int16)
                        eos = rs.read_from_hdf(fn_phenology, 'EOS', np.int16)
                        los = rs.read_from_hdf(fn_phenology, 'LOS', np.int16)
                        
                        # Fix SOS values larger than 365
                        sos_fixed = rs.fix_annual_phenology(sos)
                        eos_fixed = rs.fix_annual_phenology(eos)
                        los_fixed = rs.fix_annual_phenology(los)

                        filled_sos, filled_eos, filled_los = rs.fill_season(sos_fixed, eos_fixed, los_fixed, minimum,
                                                                        row_pixels=tile_rows,
                                                                        max_row=tile_rows,
                                                                        max_col=tile_cols,
                                                                        id=param,
                                                                        verbose=False)

                        pheno_arr = filled_sos[:]
                    elif param == 'EOS':
                        print(f'  --Filling {param}')
                        pheno_arr = filled_eos[:]
                    elif param == 'LOS':
                        print(f'  --Filling {param}')
                        pheno_arr = filled_los[:]
                    elif param == 'DOP' or param == 'NOS':
                        # Day-of-peak and Number-of-seasons, use mode
                        print(f'  --Filling {param}')
                        pheno_arr = rs.fill_with_mode(pheno_arr, 0, row_pixels=tile_rows, max_row=tile_rows, max_col=tile_cols, verbose=False)
                    elif param == 'GDR' or param == 'GUR' or param == 'MAX':
                        # GDR, GUR and MAX should be positive integers!
                        print(f'  --Filling {param}')
                        pheno_arr = rs.fill_with_int_mean(pheno_arr, 0, var=param, verbose=False)
                    else:
                        # Other parameters? Not possible
                        print(f'  --Filling {param}')
                        ds = rs.read_from_hdf(fn_phenology, param, np.int16)
                        pheno_arr = rs.fill_with_int_mean(ds, 0, var=param, verbose=False)

                # Normalize or standardize
                assert not (NORMALIZE and STANDARDIZE), "Cannot normalize and standardize at the same time!"
                if NORMALIZE and not STANDARDIZE:
                    pheno_arr = rs.normalize(pheno_arr)
                elif not NORMALIZE and STANDARDIZE:
                    pheno_arr = rs.standardize(pheno_arr)

                # # Apply scale factor (Check Table 8 page 58 from 'VIP ESDRs ATBD User Guide')
                # # SOS, EOS, LOS, DOP don't need a scale factor
                # # It will convert datasets from int16 into float
                # if param == 'GUR' or param == 'GDR':
                #     print(f"Applying scale factor to {param}")
                #     pheno_arr = pheno_arr.astype(float)
                #     phenology_values = np.empty(pheno_arr.shape, dtype=float)
                #     phenology_values = np.where(pheno_arr < 0, -1, pheno_arr / 100.)
                #     pheno_arr = phenology_values
                # if param == 'MAX':
                #     print(f"Applying scale factor to {param}")
                #     pheno_arr = pheno_arr.astype(float)
                #     phenology_values = np.empty(pheno_arr.shape, dtype=float)
                #     phenology_values = np.where(pheno_arr < 0, -1, pheno_arr / 10000.)
                #     pheno_arr = phenology_values

                # Save features for the complete raster
                h5_features_tile.create_dataset(feat, (tile_rows, tile_cols), data=pheno_arr)
                feat_index += 1

            print('Phenology 2...')
            for feat, fn in feat_type['PHEN2']:
                # print(f"--{feat_index:>3}: {feat} --> {fn}")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                param = feat[5:]

                # No need to fill missing values, just read the values
                pheno_arr = rs.read_from_hdf(fn, param, np.int16)  # Use HDF4 method

                # Filter phenology features by the NoData mask
                assert tile_nodata.shape == pheno_arr.shape, f"Dimensions don't match {tile_nodata.shape}!={pheno_arr.shape}"
                # pheno_arr = np.where(tile_nodata == 1, pheno_arr, -1)  # _nan_value=-1 for phenology
                pheno_arr = np.where(tile_nodata == 1, pheno_arr, _nan_value)

                # Fix phenology values larger than a year
                if param in ['SOS2', 'EOS2', 'LOS2']:
                    pheno_arr = rs.fix_annual_phenology(pheno_arr)
                
                # Group all negative values
                if param in ['GDR2', 'GUR2']:
                    pheno_arr = np.where(pheno_arr < 0, -13000, pheno_arr)

                # Extract data and filter by training mask
                if FILL:
                    # IMPORTANT: Only a few pixels have a second season, thus dataset could
                    # have a huge amount of NaNs, filling will be restricted to replace a
                    # The missing values to _nan_value
                    print(f'  --Filling {param}')
                    pheno_arr = rs.read_from_hdf(fn, param, np.int16)
                    pheno_arr = np.where(pheno_arr <= 0, 0, pheno_arr)
                
                # Normalize or standardize
                assert not (NORMALIZE and STANDARDIZE), "Cannot normalize and standardize at the same time!"
                if NORMALIZE and not STANDARDIZE:
                    pheno_arr = rs.normalize(pheno_arr)
                elif not NORMALIZE and STANDARDIZE:
                    pheno_arr = rs.standardize(pheno_arr)
                
                # # Apply scale factor (Check Table 8 page 58 from 'VIP ESDRs ATBD User Guide')
                # # SOS2, EOS2, LOS2, DOP2 don't need a scale factor
                # # It will convert datasets from int16 into float
                # if param == 'GUR2' or param == 'GDR2':
                #     print(f"Applying scale factor to {param}")
                #     pheno_arr = pheno_arr.astype(float)
                #     phenology_values = np.empty(pheno_arr.shape, dtype=float)
                #     phenology_values = np.where(pheno_arr < 0, -1, pheno_arr / 100.)
                #     pheno_arr = phenology_values
                # if param == 'MAX2':
                #     print(f"Applying scale factor to {param}")
                #     pheno_arr = pheno_arr.astype(float)
                #     phenology_values = np.empty(pheno_arr.shape, dtype=float)
                #     phenology_values = np.where(pheno_arr < 0, -1, pheno_arr / 10000.)
                #     pheno_arr = phenology_values
                # if param == 'CUM':
                #     print(f"Applying scale factor to {param}")
                #     pheno_arr = pheno_arr.astype(float)
                #     phenology_values = np.empty(pheno_arr.shape, dtype=float)
                #     phenology_values = np.where(pheno_arr < 0, -1, pheno_arr / 10.)
                #     pheno_arr = phenology_values

                # Save features for the complete raster
                h5_features_tile.create_dataset(feat, (tile_rows, tile_cols), data=pheno_arr)
                feat_index += 1

            print(f"File: {fn_features_tile} created successfully.")
            print(f"File: {fn_labels_tile} created successfully.")
            
        # ********* Done creating HDF5 files with monthly features *********

    #=============================================================================
    # Generate feature names for seasonal dataset
    #=============================================================================

    if _use_seasonal_features:
        print("\nGroping monthly features in to seasonal features.")
        # Use existing seasonal features and replace the list of monthly features
        seasons = {'SPR': ['APR', 'MAY', 'JUN'],
                'SUM': ['JUL', 'AUG', 'SEP'],
                'FAL': ['OCT', 'NOV', 'DEC'],
                'WIN': ['JAN', 'FEB', 'MAR']}
        # Group feature names by season -> band -> variable -> month
        season_feats = {}
        for season in list(seasons.keys()):
            print(f"AGGREGATING: {season}")
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

        # Create seasonal features names to replace monthly features
        # print(f"Seasonal features:")
        temp_monthly_feats = feat_names.copy()  # Temporal copy to save Phenology features
        feat_num = 0
        feat_indices = []
        feat_names = []
        for key in list(season_feats.keys()):
            # print(f"**{feat_num:>4}: {key:>15}")
            feat_names.append(key)
            feat_indices.append(feat_num)
            feat_num += 1
        for feat_name in temp_monthly_feats:
            if feat_name[:4] == 'PHEN':
                # print(f"**{feat_num:>4}: {feat_name:>15}")
                feat_names.append(feat_name)
                feat_indices.append(feat_num)
                feat_num += 1
        # Overwrite the list of features
        print(f"Overwriting monthly feature names with seasonal features.")
        with open(fn_feat_list, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for n_feat, feat in zip(feat_indices, feat_names):
                # print(f"{n_feat:>4}: {feat}")
                csv_writer.writerow([n_feat, feat])

    # Save the number of features, either monthly or seasonal
    n_features = len(feat_names)

    if not _save_seasonal_dataset:
        print('Saving seasonal features skipped or already exist!')

    #=============================================================================
    # Create seasonal features by averaging monthly ones
    #=============================================================================

    if _save_seasonal_dataset:
        # Actually creates the seasonal dataset as H5 files
        for tile in tiles:
            print(f"\n==Creating seasonal feature dataset for tile: {tile}==")
            # Read HDF5 monthly features per tile
            fn_features = os.path.join(cwd, _feat_dir, tile, f'features_{tile}.h5')
            fn_features_season = os.path.join(cwd, _feat_dir, tile, f'features_season_{tile}.h5')

            # Read monthly features and write into seasonal features
            h5_features = h5py.File(fn_features, 'r')
            h5_features_season = h5py.File(fn_features_season, 'w')

            # Calculate averages of features grouped by season
            feat_num = 0
            for key in list(season_feats.keys()):
                print(f"\n{feat_num:>4}: {key:>15} adding: ", end='')
                for i, feat_name in enumerate(season_feats[key]):
                    print(f"{feat_name}, ", end='')
                    # Add the data  
                    if i == 0:
                        # Initialize array to hold average
                        feat_arr = h5_features[feat_name][:]  # option 1: unspecified data type
                        feat_arr = feat_arr.astype(np.int32)  # option 2: convert to integer
                    else:
                        # Add remaining months
                        feat_arr += h5_features[feat_name][:]
                
                # Average & save
                # feat_arr /= len(season_feats[key])  # option 1: unspecified data type
                # feat_arr = np.round(np.round(feat_arr).astype(np.int16) / np.int16(len(season_feats[key]))).astype(np.int16)
                feat_arr = np.round(np.round(feat_arr).astype(np.int32) / np.int32(len(season_feats[key]))).astype(np.int32)  # option 2: add up huge int16 values
                
                h5_features_season.create_dataset(key, feat_arr.shape, data=feat_arr)
                feat_num += 1
            # Add PHEN features directly, no aggregation by season
            for feat_name in feat_names:
                if feat_name[:4] == 'PHEN':
                    print(f"{feat_num:>4}: {feat_name:>15}")
                    # Extract data & save
                    feat_arr = h5_features[feat_name][:]
                    h5_features_season.create_dataset(feat_name, feat_arr.shape, data=feat_arr)
                    feat_num += 1
            print(f"File: {fn_features_season} created/processed successfully.")

    #=============================================================================
    # Read feature datasets, X as 3D dataset 
    #=============================================================================

    if _read_split:
        # Reads the features tile-by-tile and splits the dataset into training and testing
        read_start = datetime.now()
        print(f"Reading features tile-by-tile and creating training and testing datasets.")

        # Use a subset of features only, according to user input
        if _exclude_feats is not None:
            print(f" === Excluding features from training === ")
            temp_feats = []
            for feat in feat_names:
                # Check whether current feature should be included
                feat_parts = feat.split(" ")
                exclude = False
                for part in feat_parts:
                    if part in _exclude_feats:
                        exclude = True
                # Include or exclude features accordingly
                if not exclude:
                    print(f"{feat}: included.")
                    temp_feats.append(feat)
                else:
                    print(f"{feat}: is excluded from training.")
            # Override the feature names
            feat_names = temp_feats.copy()
            n_features = len(feat_names)

            print(f"Overwriting feature names with filtered ones:")
            print(f"--{fn_feat_list}")
            with open(fn_feat_list, 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                for n_feat, feat in zip(feat_indices, feat_names):
                    print(f"  {n_feat:>4}: {feat}")
                    csv_writer.writerow([n_feat, feat])

        # Prepare array to read & hold are features
        X = np.zeros((land_cover.shape[0], land_cover.shape[1], n_features), dtype=land_cover.dtype)
        for i, tile in enumerate(tiles):
            print(f"\n== Reading features for tile {tile} ({i+1}/{len(tiles)}) ==")

            # fn_tile_features = os.path.join(cwd, _feat_dir, f"features_{tile}.h5")  # monthly
            fn_tile_features = os.path.join(cwd, _feat_dir, tile, f"features_season_{tile}.h5")  # seasonal

            # Get rows and columns to insert features
            tile_ext = tiles_extent[tile]

            # Get North and West coordinates convert them to row and column to slice dataset
            nrow = (tile_ext['N'] - mosaic_extension['N'])//yres
            wcol = (tile_ext['W'] - mosaic_extension['W'])//xres

            print(f"  Reading the features from: {fn_tile_features}")
            feat_array = np.empty((tile_rows, tile_cols, n_features), dtype=land_cover.dtype)
            with h5py.File(fn_tile_features, 'r') as h5_tile_features:
                print(f"  Features in file={len(list(h5_tile_features.keys()))}, n_features={n_features} ")
                assert len(list(h5_tile_features.keys())) >= n_features, "ERROR: Features don't match"
                # Get the data from the HDF5 files
                for i, feature in enumerate(feat_names):
                    feat_array[:,:,i] = h5_tile_features[feature][:]
            
            # Insert tile features in the right position of the 3D array
            print(f"  Inserting dataset into X [{nrow}:{nrow+tile_rows},{wcol}:{wcol+tile_cols},:]")
            X[nrow:nrow+tile_rows,wcol:wcol+tile_cols,:] = feat_array

        read_end = datetime.now()
        print(f"{read_end}: done reading  {read_end-read_start}\n")

        print("Creating training dataset...")
        # This will reshape from 3D into a 2D dataset!
        x_train = X[train_mask > 0, :]
        y_train = land_cover[train_mask > 0]

        # Find how many non-zero entries we have -- i.e. how many training and testing data samples?
        training_pixels = (train_mask ==  1).sum()
        testing_pixels = (test_mask ==  1).sum()
        label_pixels = (land_cover > 0).sum()
        roi_pixels = (nodata_mask == 1).sum()
        total_pixels = land_cover.shape[0]*land_cover.shape[1]
        
        # Create a dataframe to hold the pixel count of each dataset
        indexes = ['Total', 'ROI', 'Label/ROI', 'Training/Labels', 'Training/ROI', 'Testing/ROI']
        pixels = [total_pixels, roi_pixels, label_pixels, training_pixels, training_pixels, testing_pixels]
        percentages = [total_pixels/total_pixels,
                       roi_pixels/roi_pixels,
                       label_pixels/roi_pixels,
                       training_pixels/label_pixels,
                       training_pixels/roi_pixels,
                       testing_pixels/roi_pixels]
        pixel_ds = {'Indexes': indexes, 'PixelCount': pixels, 'Percentage': percentages}
        df_pixels = pd.DataFrame.from_dict(pixel_ds)
        df_pixels.set_index('Indexes', inplace=True)
        print(df_pixels)
        # print(f'Total pixels: {land_cover.shape[0]*land_cover.shape[1]}')
        # print(f'Training pixels: {training_pixels} ({training_pixels/label_pixels*100:0.2f}%)')
        # print(f'Testing pixels: {testing_pixels} ({testing_pixels/label_pixels*100:0.2f}%)')
        # print(f'ROI pixels: {roi_pixels}')
        # print(f'Label pixels: {label_pixels} ({label_pixels/roi_pixels*100:0.2f} % of ROI)')

        print(f"{datetime.now()}: datasets created! x_train={x_train.shape}, y_train={y_train.shape}, train_mask={train_mask.shape}")

    #=============================================================================
    # Random Forest training
    #=============================================================================

    if _train_model:
        start_train = datetime.now()
        print(f'\n{start_train}: ===== starting Random Forest training =====')

        n_estimators = 250
        max_features = None
        max_depth = None
        n_jobs = 64
        class_weight = None

        clf = RandomForestClassifier(n_estimators=n_estimators,
                                    oob_score=True,
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    n_jobs=n_jobs,
                                    class_weight=class_weight,
                                    verbose=1)
        print(f"Fitting model with max_features {max_features} and {n_estimators} estimators.")
        # IMPORTANT: This replaces the initial model by the trained model!
        print(f"x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
        clf = clf.fit(x_train, y_train)

        print(f'  --OOB prediction of accuracy: {clf.oob_score_ * 100:0.2f}%')

        importances = clf.feature_importances_
        print("Calculating standard deviation for feature importances.")
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

        feat_list = []
        feat_imp = []
        feat_std = []
        for feat, imp, sd in zip(feat_names, importances, std):
            feat_list.append(feat)
            feat_imp.append(imp)
            feat_std.append(sd)
        feat_importance = pd.DataFrame({'Feature': feat_list, 'Importance': feat_imp, 'Stdev': feat_std})
        feat_importance.sort_values(by='Importance', ascending=True, inplace=True)
        print("Feature importance: ")
        print(feat_importance.to_string())
        feat_importance.to_csv(fn_save_importance)

        # Save a figure with the importances and error bars
        print(f"Saving feature importance plot: {fn_save_imp_fig}")
        plt.figure(figsize=(8, 16), constrained_layout=True)
        plt.barh(feat_importance['Feature'], feat_importance['Importance'], xerr=feat_importance['Stdev'])
        plt.title("Feature importances")
        plt.ylabel("Mean decrease in impurity")
        plt.savefig(fn_save_imp_fig, bbox_inches='tight', dpi=300)

        # Save a representation of the decision trees in files
        # y_true = land_cover[y_mask > 0]
        # class_names_ = np.unique(y_true)
        # class_names = [str(x) for x in class_names_]

        print("Saving text representation of trees:")
        for i, tree in enumerate(clf.estimators_):
            # A text representation
            # tree_str = export_text(tree, feature_names=feat_names, class_names=class_names, max_depth=n_features)
            print(f"Estimator: {i+1}/{n_estimators}")
            # NOTE: Using max_depth_tree=n_features, produces way too bing files (1.6 GB), half the features is
            # good but difficult to open and see the files, 10 features seems like a good compromise just for showing!
            max_depth_tree = 10
            tree_str = export_text(tree, feature_names=feat_names, max_depth=max_depth_tree, decimals=0)
            fn_tree_txt = fn_save_trees[:-4] + '_' +  str(i).zfill(3) + '.txt'
            print(f"Saving text file representation: {fn_tree_txt}")
            with open(fn_tree_txt, 'w') as f_tree:
                f_tree.write(tree_str)
            # A figure
            # NOTE: Save EPS and SVG formats, if the latter ones too big they are not abe to open!
            plt.figure()
            plot_tree(tree, max_depth=max_depth_tree, feature_names=feat_names)
            fn_tree_fig = fn_save_trees[:-4] + '_' +  str(i).zfill(3) + '.svg'
            print(f"Saving image representation: {fn_tree_fig}")
            plt.savefig(fn_tree_fig, format='svg', bbox_inches='tight')
            plt.savefig(fn_tree_fig[:-4] + '.eps', format='eps', bbox_inches='tight')
        print("Saving trees done! Now freeing memory...")

        # Free memory
        del x_train
        gc.collect()

        end_train = datetime.now()
        training_time = end_train - start_train
        print(f'{end_train}: training finished in {training_time}.')

    #=============================================================================
    # Load a previously trained model
    #=============================================================================

    if _use_trained_model:
        # Use a previously trained model to make predictions
        start_load = datetime.now()
        print(f'\n{start_load}: start loading previously trained model.')
        print(f"Loading trained model: {_pretrained_model}")
        clf = load(_pretrained_model)

        # # Load the pickle version
        # with open(_pretrained_model, 'rb') as model:
        #     clf = pickle.load(model)

        end_load = datetime.now()
        loading_time = end_load - start_load
        print(f'{end_load}: loading model finished in {loading_time}.')

    #=============================================================================
    # Predict for the entire mosaic (use tile-by-tile)
    #=============================================================================

    if _predict_mosaic:
        # Make predictions using the trained model
        start_pred_mosaic = datetime.now()
        print(f"\n*** Predict for complete dataset ***")
        print(f'\n{start_pred_mosaic}: starting (mosaic) predictions for complete dataset.')

        # Prepare 2D mosaic to save predictions (same shape as land cover raster)
        y_pred = np.zeros(land_cover.shape, dtype=land_cover.dtype)
        mosaic_nan_mask = np.zeros(land_cover.shape, dtype=land_cover.dtype)

        # Read features from tiles, create a mosaic of features, and reshape it into 2D dataset of features
        tiles_per_row = land_cover.shape[1] / tile_cols
        rows_per_tile = tile_cols * tile_rows
        print(f"tiles_per_row={tiles_per_row} (tiles in mosaic), rows_per_tile={rows_per_tile}")

        h5_probas = h5py.File(fn_save_probabilities, 'w')

        # Predict by reading the features of each tile from its corresponding HDF5 file
        for i, tile in enumerate(tiles):
            print(f"\n== Making predictions for tile {tile} ({i+1}/{len(tiles)}) ==")

            # *** Read tile features ***
            tile_feature_path = os.path.join(cwd, _feat_dir, tile)
            # fn_tile_features = os.path.join(tile_feature_path, f"features_{tile}.h5")  # use monthly features
            fn_tile_features = os.path.join(tile_feature_path, f"features_season_{tile}.h5")  # use seasonal features

            print(f"  Reading tile features from: {fn_tile_features}")
            X_features = np.empty((tile_rows, tile_cols, n_features))  # to save tile features
            with h5py.File(fn_tile_features, 'r') as h5_features:
                print(f"  Features in file={len(list(h5_features.keys()))}, n_features={n_features} ")
                assert len(list(h5_features.keys())) >= n_features, "ERROR: Features don't match"
                print(f"  Features (file): {list(h5_features.keys())}")
                # Get the data from the HDF5 files
                for i, feature in enumerate(feat_names):
                    X_features[:,:,i] = h5_features[feature][:]
            
            # For debugging
            # rs.plot_dataset(X_features[:,:,0], title=f'{feat_names[0]}', savefig=os.path.join(results_path, f'plot_{tile}_{feat_names[0]}.png'))

            # Reshape features into 2D array
            X_tile = X_features.reshape(tile_rows*tile_cols, n_features)
            
            # *** Read labels and no_data mask ***
            fn_labels_tile = os.path.join(tile_feature_path, f"labels_{tile}.h5")
            print(f"  Reading tile labels from: {fn_labels_tile}")
            with h5py.File(fn_labels_tile, 'r') as h5_labels_tile:
                y_tile = h5_labels_tile['land_cover'][:]
                y_tile_nd = h5_labels_tile['no_data_mask'][:]
            
            # Finished reading feature and labels
            # print(f"X_tile={X_tile.shape} y_tile={y_tile.shape} y_tile_nd={y_tile_nd.shape}")
            
            # Predict for tile
            print(f"Predicting for tile {tile}")
            y_pred_tile = clf.predict(X_tile)
            print(f"Calculating prediction probabilities")
            probas_tile = clf.predict_proba(X_tile)

            print(f"X_tile={X_tile.shape} y_tile={y_tile.shape} y_tile_nd={y_tile_nd.shape} y_pred_tile={y_pred_tile.shape}")
            print(f"y_pred_tile: {type(y_pred_tile)}, {y_pred_tile.dtype}")
            print(f"probas_tile: {type(probas_tile)}, {probas_tile.dtype} {probas_tile.shape}")

            # Reshape list of predictions as 2D image
            y_pred_tile = y_pred_tile.reshape((tile_rows, tile_cols))
            # # Filter out ocean pixels with the no_data_mask
            # y_pred_tile = np.where(y_tile_nd == 1, y_pred_tile, 0)

            print("Inserting tile predictions into mosaic")
            # Get rows and columns to insert tile predictions into mosaic
            tile_ext = tiles_extent[tile]

            # Get row for Nort and South and column for West and East
            tile_row = (tile_ext['N'] - mosaic_extension['N'])//yres
            tile_col = (tile_ext['W'] - mosaic_extension['W'])//xres

            y_pred[tile_row:tile_row+tile_rows, tile_col:tile_col+tile_rows] = y_pred_tile.astype(land_cover.dtype)
            mosaic_nan_mask[tile_row:tile_row+tile_cols, tile_col:tile_col+tile_cols] = y_tile_nd.astype(nodata_mask.dtype)

            # # Save predicted land cover classes into a HDF5 file (for debugging purposes)
            # print("Saving tile predictions (as HDF5 file)")
            # with h5py.File(fn_save_preds_h5[:-3] + f'_{tile}.h5', 'w') as h5_preds_tile:
            #     h5_preds_tile.create_dataset(f"{tile}_ypred", y_pred_tile.shape, data=y_pred_tile)

            h5_probas.create_dataset(f"{tile}", probas_tile.shape, data=probas_tile)

            # Clean pro
            del probas_tile
            del y_pred_tile
            del y_tile
            del y_tile_nd
            gc.collect()

            # Finished predictions for tile

        print("\nFinished tile predictions.")
        print("Saving the mosaic predictions (raster and h5).")

        # Filter the predictions by a Yucatan Peninsula Aquifer mask
        y_pred_roi = np.where(nodata_mask == 1, y_pred, 0)

        # Save predictions into a raster
        rs.create_raster(fn_save_preds_raster, y_pred_roi, spatial_ref, geotransform)

        # Save predicted land cover classes into a HDF5 file
        with h5py.File(fn_save_preds_h5, 'w') as h5_preds:
            h5_preds.create_dataset("predictions", y_pred_roi.shape, data=y_pred)

        end_pred_mosaic = datetime.now()
        pred_mosaic_elapsed = end_pred_mosaic - start_pred_mosaic
        print(f'{end_pred_mosaic}: predictions for complete dataset (mosaic) finished in {pred_mosaic_elapsed}.')

        #===================== Performance assessment (complete ROI2) =====================
        print(f"{datetime.now()}: running performance assessment...")

        # Update mask to remove pixels with no land cover class
        # Remove southern part (Guatemala, Belize) in the performance assessment
        # as there is no data in that region, remove: -13000, 0, and/or '--' pixels
        print(f"land_cover: {land_cover.dtype}, {land_cover.shape}, {np.unique(land_cover)}")
        y_mask = np.where(land_cover <= 0, 0, nodata_mask)

        # Mask out NoData values (this will flatten the arrays)
        y_predictions = y_pred_roi[y_mask > 0]
        y_true = land_cover[y_mask > 0]

        kappa, accuracy, fnames = performance_assessment(y_true, y_predictions, y_mask,
                                                 fn_xtab=fn_save_crosstab,
                                                 fn_conf=fn_save_conf_tbl,
                                                 fn_fig_prod=fn_save_conf_matrix_fig[:-4] + '_prod_acc.png',
                                                 fn_fig_user=fn_save_conf_matrix_fig[:-4] + '_user_acc.png',
                                                 fn_report=fn_save_classif_report)

        # Free memory from complete image performance assessment
        del y_predictions
        del y_true
        del y_mask

        gc.collect()

        #======================== Performance assessment (training) =======================
        print(f"{datetime.now()}: running performance assessment (training dataset)...")
        y_train_pred = y_pred_roi[train_mask > 0]
        print(f"y_train_pred.shape={y_train_pred.shape}, y_train.shape={y_train.shape}")

        df_train = pd.DataFrame({'truth': y_train, 'predict': y_train_pred})
        crosstab_train = pd.crosstab(df_train['truth'], df_train['predict'], margins=True)
        crosstab_train.to_csv(fn_save_crosstab_train)
        print(f'Saving crosstabulation (training dataset): {fn_save_crosstab_train}')

        accuracy_train = accuracy_score(y_train, y_train_pred)
        print(f'Accuracy score for training: {accuracy_train}')

        cm = confusion_matrix(y_train, y_train_pred)
        print(f'Saving confusion matrix (training dataset): {fn_save_conf_tbl_train}')
        # print(type(cm))
        # print(cm.shape)
        with open(fn_save_conf_tbl_train, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for single_row in cm:
                writer.writerow(single_row)
                # print(single_row)

        # Free memory from testing performance assessment
        del y_train
        del y_train_pred
        del df_train
        del crosstab_train
        gc.collect()

        #======================== Performance assessment (testing) =======================
        print(f"{datetime.now()}: running performance assessment (testing dataset)...")
        
        print("Creating testing dataset...")
        # This will reshape from 3D into a 2D dataset!
        y_test = land_cover[test_mask > 0]
        y_test_pred = y_pred_roi[test_mask > 0]
        
        print(f"y_test_pred.shape={y_test_pred.shape}, y_test.shape={y_test.shape}")

        df_test = pd.DataFrame({'truth': y_test, 'predict': y_test_pred})
        crosstab_test = pd.crosstab(df_test['truth'], df_test['predict'], margins=True)
        crosstab_test.to_csv(fn_save_crosstab_test)
        print(f'Saving crosstabulation (testing dataset): {fn_save_crosstab_test}')

        accuracy_test = accuracy_score(y_test, y_test_pred)
        print(f'Accuracy score for testing: {accuracy_test}')

        cm = confusion_matrix(y_test, y_test_pred)
        print(f'Saving confusion matrix (testing dataset): {fn_save_conf_tbl_test}')
        # print(type(cm))
        # print(cm.shape)
        with open(fn_save_conf_tbl_test, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for single_row in cm:
                writer.writerow(single_row)
                # print(single_row)

        # Free memory from testing performance assessment
        del y_test
        del y_test_pred
        del df_test
        del crosstab_test
        gc.collect()

    #TODO: Performance assessment
    # 1. Predict for a testing dataset (predict as 1D to get accuracy only, impossible to create 2D "train" and "test" maps)
    # 2. Compare regularization error (trainind accuracy vs testing accuracy), is this valid for classification problems?

    #=============================================================================
    # Last but not least, save the trained model
    #=============================================================================

    # WARNING! With the current sample size (20%) model is to big and fails to save
    if _save_model:
        print(f"{datetime.now()}: saving trained model (this may take a while)...")
        dump(clf, fn_save_model[:-4] + ".joblib")

        # print(" now saving pickle model...")
        # with open(fn_save_model, 'wb') as f:
        #     pickle.dump(clf, f)


    #=============================================================================
    # Finish and save the parameters of this run
    #=============================================================================

    exec_end = datetime.now()
    exec_time = exec_end - exec_start

    # Save the parameters of this run
    with open(fn_save_params, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Option: Read and split', _read_split])
        writer.writerow(['Option: Train model', _train_model])
        writer.writerow(['Option: Save model', _save_model])
        writer.writerow(['Option: Use trained model', _use_trained_model])
        writer.writerow(['Option: Use seasonal features', _use_seasonal_features])
        writer.writerow(['Option: Save monthly dataset', _save_monthly_dataset])
        writer.writerow(['Option: Save seasonal dataset', _save_seasonal_dataset])
        writer.writerow(['Option: Predict', _predict_mosaic])
        writer.writerow(['Option: Override tiles', _override_tiles if _override_tiles is None else ';'.join(_override_tiles)])
        writer.writerow(['Option: Sample directory', _sample_dir])
        writer.writerow(['Option: Features directory', _feat_dir])
        writer.writerow(['Run (start time)', exec_start])
        writer.writerow(['NaN value', _nan_value])
        writer.writerow(['CWD', cwd])
        writer.writerow(['Statistics directory', stats_dir])
        writer.writerow(['Phenology directory', pheno_dir])
        writer.writerow(['Tile rows', tile_rows])
        writer.writerow(['Tile colums', tile_cols])
        writer.writerow(['LABELS', ''])
        writer.writerow(['Land cover raster', fn_landcover])
        writer.writerow(['  NoData', lc_nd])
        writer.writerow(['  Rows', land_cover.shape[0]])
        writer.writerow(['  Columns', land_cover.shape[1]])
        writer.writerow(['  Geotransform', lc_gt])
        writer.writerow(['  Spatial reference', lc_sp_ref])
        writer.writerow(['  Data type', land_cover.dtype])
        writer.writerow(['Training mask raster', fn_train_mask])
        writer.writerow(['  NoData', nodata])
        writer.writerow(['  Rows', train_mask.shape[0]])
        writer.writerow(['  Columns', train_mask.shape[1]])
        writer.writerow(['  Geotransform', geotransform])
        writer.writerow(['  Spatial reference', spatial_ref])
        writer.writerow(['  Data type', train_mask.dtype])
        writer.writerow(['No Data mask (ROI) raster', fn_mask])
        writer.writerow(['  NoData', mask_nd])
        writer.writerow(['  Rows', nodata_mask.shape[0]])
        writer.writerow(['  Columns', nodata_mask.shape[1]])
        writer.writerow(['  Geotransform', mask_gt])
        writer.writerow(['  Spatial reference', mask_sp_ref])
        writer.writerow(['  Data type', nodata_mask.dtype])
        writer.writerow(['Labels (mosaic)', fn_mosaic_labels])
        writer.writerow(['Tiles file', fn_tiles])
        writer.writerow(['MONTHLY FEATURES', ''])
        writer.writerow(['Features path', _features_path])
        writer.writerow(['Bands', ';'.join(bands)])
        writer.writerow(['Band numbers', ';'.join(band_num)])
        writer.writerow(['Months', ';'.join(months)])
        writer.writerow(['Variables', ';'.join(vars)])
        writer.writerow(['Phenology variables (S1)', ';'.join(phen)])
        writer.writerow(['Phenology variables (S2)', ';'.join(phen2)])
        writer.writerow(['FILL', FILL])
        writer.writerow(['NORMALIZE', NORMALIZE])
        writer.writerow(['STANDARDIZE', STANDARDIZE])
        writer.writerow(['Phenology file 1', fn_phenology])
        writer.writerow(['Phenology file 2', fn_phenology2])
        writer.writerow(['SEASONAL FEATURES', ''])
        for k in seasons.keys():
            writer.writerow([str(k), ';'.join(seasons[k])])
        writer.writerow(['Number of features', str(n_features)])
        writer.writerow(['Features file', fn_feat_list])
        writer.writerow(['Tile features path (last)', tile_feature_path])
        if _read_split:
            # Reading tiles was performed
            writer.writerow(['READING FEATURES', ''])
            writer.writerow(['Reading started', read_start])
            writer.writerow(['Excluded features', _exclude_feats if _exclude_feats is None else ';'.join(_exclude_feats)])
            writer.writerow(['Training pixels', training_pixels])
            writer.writerow(['Training percent', training_pixels/label_pixels*100])
            writer.writerow(['Testing pixels', testing_pixels])
            writer.writerow(['Testing percent', testing_pixels/label_pixels*100])
            writer.writerow(['ROI pixels', roi_pixels])
            writer.writerow(['Label pixels', label_pixels])
            writer.writerow(['Label percent (of ROI)', label_pixels/roi_pixels*100])
            writer.writerow(['Reading ended', read_end])
            writer.writerow(['Reading elapsed', read_end-read_start])
        if _train_model:
            # Algorithm training was performed
            writer.writerow(['RANDOM FOREST TRAINING', ''])
            writer.writerow(['Training started', start_train])
            writer.writerow(['Estimators', n_estimators])
            writer.writerow(['Max features', max_features])
            writer.writerow(['Max depth', max_depth])
            writer.writerow(['Jobs', n_jobs])
            writer.writerow(['Model file', fn_save_model])
            writer.writerow(['OOB prediction of accuracy', clf.oob_score_])
            writer.writerow(['Feature importance', fn_save_importance])
            writer.writerow(['Training ended', end_train])
            writer.writerow(['Training time', training_time])
        if _use_trained_model:
            # A previously trained model for predictions was loaded
            writer.writerow(['PRE TRAINED MODEL', ''])
            writer.writerow(['Pretrained model', _pretrained_model])
            writer.writerow(['Loading started', start_load])
            writer.writerow(['Loading ended', end_load])
            writer.writerow(['Loading time', loading_time])
        if _predict_mosaic:
            # Predictions (mosaic) was performed
            writer.writerow(['PREDICTIONS (MOSAIC)', ''])
            writer.writerow(['Predictions started', start_pred_mosaic])
            writer.writerow(['Predictions raster', fn_save_preds_raster])
            writer.writerow(['Predictions H5', fn_save_preds_h5])
            writer.writerow(['Predictions ended', end_pred_mosaic])
            writer.writerow(['Predictions time', pred_mosaic_elapsed])
            # Performance assessment
            writer.writerow(['PERFORMANCE ASSESSMENT', ''])
            writer.writerow(['Training accuracy score', accuracy_train])
            writer.writerow(['Training crosstabulation', fn_save_crosstab_train])
            writer.writerow(['Training confusion matrix', fn_save_conf_tbl_train])
            writer.writerow(['Testing accuracy score', accuracy_test])
            writer.writerow(['Testing crosstabulation', fn_save_crosstab_test])
            writer.writerow(['Testing confusion matrix', fn_save_conf_tbl_test])
            writer.writerow(['Accuracy score', accuracy])
            writer.writerow(['Kappa', kappa])
            writer.writerow(['Crosstabulation file', fnames[0]])
            writer.writerow(['Confusion matrix file', fnames[1]])
            writer.writerow(['Producer accuracy figure', fnames[2]])
            writer.writerow(['User accuracy figure', fnames[3]])
            writer.writerow(['Classification report file', fnames[4]])
        writer.writerow(['Run ended', exec_end])
        writer.writerow(['Run time', exec_time])
    
    print(f"\n{exec_end}: everything completed on: {exec_time}. Bye ;-)")


if __name__ == '__main__':
    # # Paths for the entire period
    # cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/'
    # stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/02_STATS/'
    # pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/03_PHENO/NDVI/'
    
    # # # Test application of aquifer mask to predictions
    # # fn_pred = os.path.join(cwd, 'results', '2023_09_25-15_31_15', '2023_09_25-15_31_15_predictions.tif')
    # # fn_mask = os.path.join(cwd, 'data', 'YucPenAquifer_mask.tif')
    # # fn_pred_roi = os.path.join(cwd, 'results', '2023_09_25-15_31_15', '2023_09_25-15_31_15_predictions_roi.tif')

    # # pred_ds, nodata, geotransform, spatial_ref = rs.open_raster(fn_pred)
    # # roi_mask_ds, _, _, _ = rs.open_raster(fn_mask)
    # # preds_roi = np.where(roi_mask_ds == 1, pred_ds, 0)
    # # rs.create_raster(fn_pred_roi, preds_roi, spatial_ref, geotransform)

    # # Control the execution of the land cover classification code
    # # Option 0: generate monthly and seasonal datasets, then train, and predict
    # # landcover_classification(save_monthly_dataset=True, save_seasonal_dataset=True, override_tiles=['h19v25'], save_model=False, train_model=False, predict_mosaic=False, nan=nan)
    # # landcover_classification(save_monthly_dataset=True, save_seasonal_dataset=True, save_model=False, train_model=False, predict_mosaic=False, nan=nan) # generate features, do not train

    # # Option 1: train RF and predict using the mosaic approach (default)
    # landcover_classification(cwd, stats_dir, pheno_dir, save_model=False)

    # # # Exclude some 'unimportant' features from analysis (NEVER DONE BEFORE)
    # # no_feats = ['EVI2', 'SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']
    # # landcover_classification(save_model=False, exclude_feats=no_feats)

    # ================ RUNNING CLASSIFICATION FOR EACH PERIOD =================

    # Include ancillary data for all periods
    # # fn_landcover_raster = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_grp11.tif"
    # # fn_landcover_raster = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/data/usv250s6gw_grp11.tif"
    # fn_landcover_raster = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/data/usv250s7gw_grp11.tif"
    # ancillary_dict = {102: ["roads.tif", "urban.tif"]}  # for grouped "grp11"
    # incorporate_ancillary(fn_landcover_raster, ancillary_dict)

    # =============================== 2013-2016 ===============================
    # cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/'
    # stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2013_2016/02_STATS/'
    # pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2013_2016/03_PHENO/'
    # fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_grp11_ancillary.tif"

    # =============================== 2016-2019 ===============================
    cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/'
    stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2016_2019/02_STATS/'
    pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2016_2019/03_PHENO/'
    fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/data/usv250s6gw_grp11_ancillary.tif"

    # =============================== 2019-2022 ===============================
    # cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/'
    # stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2019_2022/02_STATS/'
    # pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2019_2022/03_PHENO/'
    # fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/data/usv250s7gw_grp11_ancillary.tif"

    # ============================ FOR ALL PERIODS =============================
    # The NoData mask is the ROI (The Yucatan Peninsula Aquifer)
    fn_nodata = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/data/YucPenAquifer_mask.tif'
    fn_tiles = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/parameters/tiles'

    # # Sampling to select the training sites
    # # sample(cwd, fn_landcover, max_trials=3e6, sampling_suffix='sampling_grp11_3M')

    # # Generate monthly and seasonal features
    # # landcover_classification(cwd, stats_dir,
    # #                          pheno_dir,
    # #                          fn_landcover,
    # #                          fn_nodata,
    # #                          fn_tiles,
    # #                          save_monthly_dataset=True,
    # #                          save_seasonal_dataset=True,
    # #                          save_model=False,
    # #                          train_model=False,
    # #                          predict_mosaic=False,
    # #                          sample_dir="sampling_grp11_3M",
    # #                          features_dir="features")

    # # Sampling to select the training sites: sample size = 10% with 11 grouped land cover labels
    # # sample(cwd, fn_landcover, max_trials=3e6, sampling_suffix='sampling_10percent', train_percent=0.1)

    # # Generate monthly and seasonal features: use actual ranges for variables (e.g. NDVI from 0-1 instead of 0-10000)
    # # WARNING! This will create huge H5 files, not recommended. Keep using integer datatype instead!
    # # tiles = ['h19v25']
    # # landcover_classification(cwd, stats_dir,
    # #                          pheno_dir,
    # #                          fn_landcover,
    # #                          fn_nodata,
    # #                          fn_tiles,
    # #                          save_monthly_dataset=True,
    # #                          save_seasonal_dataset=True,
    # #                          save_model=False,
    # #                          train_model=False,
    # #                          predict_mosaic=False,
    # #                          sample_dir="sampling_10percent", # use the 10% sample size
    # #                         #  sample_dir="sampling_grp11_3M", # use the 20% sample size
    # #                          features_dir="features_values",
    # #                          override_tiles=tiles)
    # landcover_classification(cwd, stats_dir,
    #                          pheno_dir,
    #                          fn_landcover,
    #                          fn_nodata,
    #                          fn_tiles,
    #                          save_monthly_dataset=True,
    #                          save_seasonal_dataset=True,
    #                          save_model=False,
    #                          train_model=False,
    #                          predict_mosaic=False,
    #                          sample_dir="sampling_10percent", # use the 10% sample size
    #                          features_dir="features")

    # # Train Random Forest and predict using the mosaic approach (default)
    # landcover_classification(cwd,
    #                          stats_dir,
    #                          pheno_dir,
    #                          fn_landcover,
    #                          fn_nodata,
    #                          fn_tiles,
    #                          save_model=False,
    #                         #  sample_dir="sampling_grp11_3M",  # use the 20% sample size
    #                          sample_dir="sampling_10percent",  # use the 10% sample size
    #                          features_dir="features")
    
    # Excude features from the analysis, based on the most important ones from previous executions
    dont_use = ['STDEV',  # remove all standard deviation features
                'EVI2',
                'MAX2',
                'GUR2',
                'EOS2',
                'DOP2',
                'GDR2',
                'SOS2',
                'LOS2',
                'NOS']
    # Just generate the feature list, don't do anything else
    # landcover_classification(cwd,
    #                          stats_dir,
    #                          pheno_dir,
    #                          fn_landcover,
    #                          fn_nodata,
    #                          fn_tiles,
    #                          save_model=False,
    #                          sample_dir="sampling_10percent",  # use the 10% sample size
    #                          train_model=False,
    #                          predict_mosaic=False,
    #                          features_dir="features",
    #                          exclude_feats=dont_use,
    #                          read_split=True)

    # Train Random Forest and predict using the mosaic approach, excluding some features
    landcover_classification(cwd,
                             stats_dir,
                             pheno_dir,
                             fn_landcover,
                             fn_nodata,
                             fn_tiles,
                            #  save_model=False,
                             sample_dir="sampling_10percent",  # use the 10% sample size
                             exclude_feats=dont_use,
                             features_dir="features")

