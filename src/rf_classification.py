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

  TODO: save the trained model
  - Run without saving the model, current approach.
  - Use an alternative to pickle (it fails with really big models).
  - Reduce sample size, not implemented, not convenient.
  
"""

import sys
import os
import csv
import h5py
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')
cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/'
stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/02_STATS/'
pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/03_PHENO/NDVI/'

import rsmodule as rs


def run_landcover_classification(**kwargs):
    """A function to control each execution of the land cover classification code.
    The blocks of code that will run are passed as keyword arguments. 
    """
    _read_split = kwargs.get("read_split", False)
    _train_model = kwargs.get("train_model", True)
    _save_model = kwargs.get("save_model", True)
    _use_trained_model = kwargs.get("use_trained_model", False)
    _pretrained_model = kwargs.get("trained_model", '')
    _use_seasonal_features = kwargs.get("use_seasonal_features", True)
    _save_monthly_dataset = kwargs.get("save_monthly_dataset", False)
    _save_seasonal_dataset = kwargs.get("save_seasonal_dataset", False)
    _predict_mosaic = kwargs.get("predict_mosaic", True)
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

    NAN_VALUE = 0
    tile_rows = 5000
    tile_cols = 5000
    fmt = '%Y_%m_%d-%H_%M_%S'

    exec_start = datetime.now()

    #=============================================================================
    # Read land cover and training mask rasters and create labels
    #=============================================================================

    fn_landcover = os.path.join(cwd, 'data/inegi/usv250s7cw2018_ROI2full_ancillary.tif')      # Groups of land cover classes w/ ancillary
    fn_train_mask = os.path.join(cwd, 'sampling/training_mask.tif')

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

    # land_cover = land_cover.astype(int)

    print('  Analyzing labels from testing dataset (land cover classes)')
    land_cover = land_cover.astype(train_mask.dtype)
    train_arr = np.where(train_mask == 1, land_cover, 0)  # Actual labels (land cover classes)

    print(f'  --train_mask: {train_mask.dtype}, unique:{np.unique(train_mask.filled(0))}, {train_mask.shape}')
    print(f'  --land_cover: {land_cover.dtype}, unique:{np.unique(land_cover.filled(0))}, {land_cover.shape}')
    print(f'  --train_arr: {train_arr.dtype}, unique:{np.unique(train_arr)}, {train_arr.shape}')

    # Create a mask for 'No Data' pixels (e.g. sea, or no land cover available)
    no_data_arr = np.where(land_cover > 0, 1, NAN_VALUE)  # 1=data, 0=NoData
    no_data_arr = no_data_arr.astype(np.ubyte)
    # Keep train mask values only in pixels with data, remove NoData
    train_mask = np.where(no_data_arr == 1, train_mask, NAN_VALUE)

    # Find how many non-zero entries we have -- i.e. how many training and testing data samples?
    print(f'  --Training pixels: {(train_mask ==  1).sum()}')
    print(f'  --Testing pixels: {(train_mask == 0).sum()}')

    # Save the entire mosaic land cover labels, training mask, and 'No Data' mask
    fn_mosaic_labels = os.path.join(cwd, 'features', 'mosaic_labels.h5')
    h5_mosaic_labels = h5py.File(fn_mosaic_labels, 'w')
    h5_mosaic_labels.create_dataset('land_cover', land_cover.shape, data=land_cover)
    h5_mosaic_labels.create_dataset('train_mask', land_cover.shape, data=train_mask)
    h5_mosaic_labels.create_dataset('no_data_mask', land_cover.shape, data=no_data_arr)

    # Read tiles names and extent (in Albers projection)
    fn_tiles = os.path.join(cwd, 'parameters/tiles')
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
    fn_save_crosstab_train = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_crosstab_train.csv")
    fn_save_crosstab_test = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_crosstab_test.csv")
    fn_save_crosstab_test_mask = fn_save_crosstab_test[:-4] + f'_mask.csv'
    fn_save_conf_tbl = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_confussion_table.csv")
    fn_save_report = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_report.txt")
    fn_save_preds_fig = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_predictions.png")
    fn_save_preds_raster = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_predictions.tif")
    fn_save_preds_h5 = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_predictions.h5")
    fn_save_params = os.path.join(results_path, f"{datetime.strftime(exec_start, fmt)}_run_parameters.csv")
    fn_save_conf_fig = fn_save_conf_tbl[:-4] + '.png'
    fn_save_train_error_fig = os.path.join(results_path, f"rf_training_error.png")

    # Save the list of features
    fn_feat_list = os.path.join(results_path, f'{datetime.strftime(exec_start, fmt)}_feature_list.csv')
    print("Generated monthly features:")
    with open(fn_feat_list, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for n_feat, feat in zip(feat_indices, feat_names):
            print(f"{n_feat:>4}: {feat}")
            csv_writer.writerow([n_feat, feat])

    # *** Generate monthly feature datasets ***

    # FILL, NORMALIZE, STANDARDIZE = False, False, False  # Either normalize or standardize, not both!

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
        feat_path = os.path.join(cwd, 'features', tile)
        if not os.path.exists(feat_path):
            print(f"\nCreating features path: {feat_path}")
            os.makedirs(feat_path)

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
            print(f"{n:>3}: {f} --> {fn}")

        # ********* Create HDF5 files to save monthly features *********

        if _save_monthly_dataset:
            # Actually save the monthly dataset as H5 files
            fn_features_tile = os.path.join(cwd, 'features', tile, f'features_{tile}.h5')
            fn_labels_tile = os.path.join(cwd, 'features', tile, f'labels_{tile}.h5')

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

            # Slice the training mask and the NoData mask
            tile_training_mask = train_mask[nrow:srow, wcol:ecol]
            tile_nodata = no_data_arr[nrow:srow, wcol:ecol]

            # Save the training mask, land cover labels, and 'NoData' mask
            h5_labels_tile.create_dataset('land_cover', (tile_rows, tile_cols), data=tile_landcover)
            h5_labels_tile.create_dataset('train_mask', (tile_rows, tile_cols), data=tile_training_mask)
            h5_labels_tile.create_dataset('no_data_mask', (tile_rows, tile_cols), data=tile_nodata)

            # Process each dataset according its feature type
            feat_index = 0
            print('Spectral bands')
            for feat, fn in feat_type['BAND']:
                print(f"--{feat_index:>3}: {feat} --> {fn}")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"

                # Extract data and filter by training mask
                feat_arr = rs.read_from_hdf(fn, feat[4:], np.int16)  # Use HDF4 method

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

                # Save features for the complete raster
                h5_features_tile.create_dataset(feat, (tile_rows, tile_cols), data=feat_arr)
                feat_index += 1

            print('Phenology')
            for feat, fn in feat_type['PHEN1']:
                print(f"--{feat_index:>3}: {feat} --> {fn}")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                param = feat[5:]
                
                # No need to fill missing values, just read the values
                pheno_arr = rs.read_from_hdf(fn_phenology, param, np.int16)  # Use HDF4 method

                # # Fill missing data
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

                # Save features for the complete raster
                h5_features_tile.create_dataset(feat, (tile_rows, tile_cols), data=pheno_arr)
                feat_index += 1

            print('Phenology 2')
            for feat, fn in feat_type['PHEN2']:
                print(f"--{feat_index:>3}: {feat} --> {fn}")
                assert os.path.isfile(fn) is True, f"ERROR: File not found! {fn}"
                param = feat[5:]

                # No need to fill missing values, just read the values
                pheno_arr = rs.read_from_hdf(fn, param, np.int16)  # Use HDF4 method

                # Extract data and filter by training mask
                if FILL:
                    # IMPORTANT: Only a few pixels have a second season, thus dataset could
                    # have a huge amount of NaNs, filling will be restricted to replace a
                    # The missing values to NAN_VALUE
                    print(f'  --Filling {param}')
                    pheno_arr = rs.read_from_hdf(fn, param, np.int16)
                    pheno_arr = np.where(pheno_arr <= 0, 0, pheno_arr)
                
                # Normalize or standardize
                assert not (NORMALIZE and STANDARDIZE), "Cannot normalize and standardize at the same time!"
                if NORMALIZE and not STANDARDIZE:
                    pheno_arr = rs.normalize(pheno_arr)
                elif not NORMALIZE and STANDARDIZE:
                    pheno_arr = rs.standardize(pheno_arr)

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
        # Use existing seasonal features and replace the list of monthly features
        seasons = {'SPR': ['APR', 'MAY', 'JUN'],
                'SUM': ['JUL', 'AUG', 'SEP'],
                'FAL': ['OCT', 'NOV', 'DEC'],
                'WIN': ['JAN', 'FEB', 'MAR']}
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

        # Create seasonal features names to replace monthly features
        print(f"Seasonal features:")
        temp_monthly_feats = feat_names.copy()  # Temporal copy to save Phenology features
        feat_num = 0
        feat_indices = []
        feat_names = []
        for key in list(season_feats.keys()):
            print(f"**{feat_num:>4}: {key:>15}")
            feat_names.append(key)
            feat_indices.append(feat_num)
            feat_num += 1
        for feat_name in temp_monthly_feats:
            if feat_name[:4] == 'PHEN':
                print(f"**{feat_num:>4}: {feat_name:>15}")
                feat_names.append(feat_name)
                feat_indices.append(feat_num)
                feat_num += 1
        # Overwrite the list of features
        print(f"Overwriting monthly features with seasonal features.")
        with open(fn_feat_list, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for n_feat, feat in zip(feat_indices, feat_names):
                print(f"{n_feat:>4}: {feat}")
                csv_writer.writerow([n_feat, feat])

    # Save the number of features, either monthly or seasonal
    n_features = len(feat_names)

    #=============================================================================
    # Create seasonal features by averaging monthly ones
    #=============================================================================

    if _save_seasonal_dataset:
        # Actually creates the seasonal dataset as H5 files
        for tile in tiles:
            print(f"\n==Creating seasonal feature dataset for tile: {tile}==")
            # Read HDF5 monthly features per tile
            fn_features = os.path.join(cwd, 'features', tile, f'features_{tile}.h5')
            fn_features_season = os.path.join(cwd, 'features', tile, f'features_season_{tile}.h5')

            # Read monthly features and write into seasonal features
            h5_features = h5py.File(fn_features, 'r')
            h5_features_season = h5py.File(fn_features_season, 'w')

            # Calculate averages of features grouped by season
            for key in list(season_feats.keys()):
                print(f"**{feat_num:>4}: {key:>15}")
                for i, feat_name in enumerate(season_feats[key]):
                    print(f"  Adding: {feat_name}")
                    # Add the data  
                    if i == 0:
                        # Initialize array to hold average
                        feat_arr = h5_features[feat_name][:]
                    else:
                        # Add remaining months
                        feat_arr += h5_features[feat_name][:]
                # Average & save
                feat_arr = np.round(np.round(feat_arr).astype(np.int16) / np.int16(len(season_feats[key]))).astype(np.int16)
                h5_features_season.create_dataset(key, feat_arr.shape, data=feat_arr)
            # Add PHEN features directly, no aggregation by season
            for feat_name in feat_names:
                if feat_name[:4] == 'PHEN':
                    print(f" **{feat_num:>4}: {feat_name:>15}")
                    # Extract data & save
                    feat_arr = h5_features[feat_name][:]
                    h5_features_season.create_dataset(feat_name, feat_arr.shape, data=feat_arr) # TODO: Uncomment to create dataset
            print(f"File: {fn_features_season} created/processed successfully.")

    #=============================================================================
    # Read the datasets, X as 2D dataset (WARNING: 3D Method is more efficient)
    #=============================================================================

    # X = np.zeros((land_cover.shape[0]*land_cover.shape[1], n_features), dtype=land_cover.dtype)
    # tiles_per_row = land_cover.shape[1] / tile_cols
    # for i, tile in enumerate(tiles):
    #     print(f"\n== Reading features for tile {tile} ({i+1}/{len(tiles)}) ==")

    #     # fn_tile_features = os.path.join(cwd, 'features', f"features_{tile}.h5")  # monthly
    #     fn_tile_features = os.path.join(cwd, 'features', tile, f"features_season_{tile}.h5")  # seasonal

    #     # Get rows and columns to insert features
    #     tile_ext = tiles_extent[tile]

    #     # Get row for Nort and South and column for West and East
    #     nrow = (tile_ext['N'] - mosaic_extension['N'])//yres
    #     wcol = (tile_ext['W'] - mosaic_extension['W'])//xres

    #     # Account for number of tiles (or steps) per row/column
    #     row_steps = nrow // tile_rows
    #     col_steps = wcol // tile_cols

    #     # Read the features, for re of tiles_per_row according to current row
    #     tile_start = int((tiles_per_row * row_steps + col_steps) * (tile_rows*tile_cols))

    #     # print(f"--tile row={tile_row}")
    #     print(f"--Reading the features from: {fn_tile_features}")
    #     feat_array = np.empty((tile_rows, tile_cols, n_features), dtype=land_cover.dtype)
    #     with h5py.File(fn_tile_features, 'r') as h5_tile_features:
    #         print(f"Features in file={len(list(h5_tile_features.keys()))}, n_features={n_features} ")
    #         assert len(list(h5_tile_features.keys())) == n_features, "ERROR: Features don't match"
    #         # Get the data from the HDF5 files
    #         for i, feature in enumerate(feat_names):
    #             feat_array[:,:,i] = h5_tile_features[feature][:]
        
    #     # Transform into a 2D-array
    #     # Insert tile features in the right position of the 2-D array
    #     print(f"--Inserting dataset into 2D array ({tile_rows}x{tile_cols})...")
    #     for row in range(tile_rows):
    #         for col in range(tile_cols):
    #             # Calculate the right position to insert the datset
    #             insert_row = tile_start + row*tile_rows + col
    #             if row == 0 and col == 0:
    #                 print(f"--Starting at row: {insert_row}")
    #             X[insert_row, :] = feat_array[row, col, :]
    #     print(f"--Finished at row: {insert_row}")

    # print("Done reading features.\n")

    # print("Creating training dataset...")
    # train_mask = train_mask.flatten()
    # x_train = X[train_mask > 0]
    # y_train = land_cover.flatten()[train_mask > 0]

    #=============================================================================
    # Read feature datasets, X as 3D dataset (preferred method)
    #=============================================================================

    if _read_split:
        # Reads the features tile-by-tile and splits the dataset into training and testing
        read_start = datetime.now()

        X = np.zeros((land_cover.shape[0], land_cover.shape[1], n_features), dtype=land_cover.dtype)
        for i, tile in enumerate(tiles):
            print(f"\n== Reading features for tile {tile} ({i+1}/{len(tiles)}) ==")

            # fn_tile_features = os.path.join(cwd, 'features', f"features_{tile}.h5")  # monthly
            fn_tile_features = os.path.join(cwd, 'features', tile, f"features_season_{tile}.h5")  # seasonal

            # Get rows and columns to insert features
            tile_ext = tiles_extent[tile]

            # Get North and West coordinates convert them to row and column to slice dataset
            nrow = (tile_ext['N'] - mosaic_extension['N'])//yres
            wcol = (tile_ext['W'] - mosaic_extension['W'])//xres

            print(f"  Reading the features from: {fn_tile_features}")
            feat_array = np.empty((tile_rows, tile_cols, n_features), dtype=land_cover.dtype)
            with h5py.File(fn_tile_features, 'r') as h5_tile_features:
                print(f"  Features in file={len(list(h5_tile_features.keys()))}, n_features={n_features} ")
                assert len(list(h5_tile_features.keys())) == n_features, "ERROR: Features don't match"
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
        clf = clf.fit(x_train, y_train)

        print(f'  --OOB prediction of accuracy: {clf.oob_score_ * 100:0.2f}%')

        feat_list = []
        feat_imp = []
        for feat, imp in zip(feat_names, clf.feature_importances_):
            feat_list.append(feat)
            feat_imp.append(imp)
        feat_importance = pd.DataFrame({'Feature': feat_list, 'Importance': feat_imp})
        feat_importance.sort_values(by='Importance', ascending=False, inplace=True)
        print("Feature importance: ")
        print(feat_importance.to_string())
        feat_importance.to_csv(fn_save_importance)

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

        # pretrained_model = os.path.join(cwd, 'results', '2023_09_06-16_39_39', 'rf_model.pkl')
        # pretrained_model = os.path.join(cwd, 'results', '2023_09_11-13_15_56', 'rf_model.pkl')
        # pretrained_model = os.path.join(cwd, 'results', '2023_09_23-13_29_41', 'rf_model.pkl')
        print(f"Loading trained model: {_pretrained_model}")
        with open(_pretrained_model, 'rb') as model:
            clf = pickle.load(model)

        end_load = datetime.now()
        loading_time = end_load - start_load
        print(f'{end_load}: loading model finished in {loading_time}.')

    #=============================================================================
    # Start the prediction over the entire mosaic
    #=============================================================================

    ### NOTICE: When predicting for the entire ROI2 the program is killed, apparently the
    ### computer cannot handle the entire dataset, use a tile by tile prediction instead!

    # start_pred = datetime.now()
    # print(f"\n*** Predict for complete dataset ***")
    # print(f'\n{start_pred}: starting predictions for complete dataset.')

    # # Reshape X (features) into a 2D dataset and make predictions
    # y = clf.predict(X.reshape(X.shape[0]*X.shape[1], X.shape[2]))

    # print("Saving the mosaic predictions (raster and h5).")
    # # Save predictions into a raster (and no_data_mask for debugging)
    # rs.create_raster(fn_save_preds_raster, y, spatial_ref, geotransform)

    # # Save predicted land cover classes into a HDF5 file
    # with h5py.File(fn_save_preds_h5, 'w') as h5_preds:
    #     h5_preds.create_dataset("predictions", y.shape, data=y)

    # end_pred_mosaic = datetime.now()
    # pred_mosaic_elapsed = end_pred_mosaic - start_pred
    # print(f'{end_pred_mosaic}: predictions for complete dataset finished in {pred_mosaic_elapsed}.')

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

        # Predict by reading the features of each tile from its corresponding HDF5 file
        for i, tile in enumerate(tiles):
            print(f"\n== Making predictions for tile {tile} ({i+1}/{len(tiles)}) ==")

            # *** Read tile features ***
            feat_path = os.path.join(cwd, 'features', tile)
            # fn_tile_features = os.path.join(feat_path, f"features_{tile}.h5")  # use monthly features
            fn_tile_features = os.path.join(feat_path, f"features_season_{tile}.h5")  # use seasonal features

            print(f"  Reading tile features from: {fn_tile_features}")
            X_features = np.empty((tile_rows, tile_cols, n_features))  # to save tile features
            with h5py.File(fn_tile_features, 'r') as h5_features:
                print(f"  Features in file={len(list(h5_features.keys()))}, n_features={n_features} ")
                assert len(list(h5_features.keys())) == n_features, "ERROR: Features don't match"
                print(f"  Features (file): {list(h5_features.keys())}")
                # Get the data from the HDF5 files
                for i, feature in enumerate(feat_names):
                    X_features[:,:,i] = h5_features[feature][:]
            
            # For debugging
            # rs.plot_dataset(X_features[:,:,0], title=f'{feat_names[0]}', savefig=os.path.join(results_path, f'plot_{tile}_{feat_names[0]}.png'))

            # Reshape features into 2D array
            X_tile = X_features.reshape(tile_rows*tile_cols, n_features)
            
            # *** Read labels and no_data mask ***
            fn_labels_tile = os.path.join(feat_path, f"labels_{tile}.h5")
            print(f"  Reading tile labels from: {fn_labels_tile}")
            with h5py.File(fn_labels_tile, 'r') as h5_labels_tile:
                y_tile = h5_labels_tile['land_cover'][:]
                y_tile_nd = h5_labels_tile['no_data_mask'][:]
            
            # Finished reading feature and labels
            # print(f"X_tile={X_tile.shape} y_tile={y_tile.shape} y_tile_nd={y_tile_nd.shape}")
            
            # Predict for tile
            y_pred_tile = clf.predict(X_tile)
            print(f"X_tile={X_tile.shape} y_tile={y_tile.shape} y_tile_nd={y_tile_nd.shape} y_pred_tile={y_pred_tile.shape}")
            print(f"y_pred_tile: {type(y_pred_tile)}, {y_pred_tile.dtype}")

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
            mosaic_nan_mask[tile_row:tile_row+tile_cols, tile_col:tile_col+tile_cols] = y_tile_nd.astype(no_data_arr.dtype)

            # Save predicted land cover classes into a HDF5 file (for debugging purposes)
            # print("Saving tile predictions (as HDF5 file)")
            # with h5py.File(fn_save_preds_h5[:-3] + f'_{tile}.h5', 'w') as h5_preds_tile:
            #     h5_preds_tile.create_dataset(f"{tile}_ypred", y_pred_tile.shape, data=y_pred_tile)

            # Finished predictions for tile

        print("\nFinished tile predictions.")
        print("Saving the mosaic predictions (raster and h5).")

        # TODO: Filter the predictions by a Yucatan Peninsula Aquifer mask

        # Save predictions into a raster
        rs.create_raster(fn_save_preds_raster, y_pred, spatial_ref, geotransform)
        # rs.create_raster(fn_save_preds_raster[:-4] + "_gen_nan_mask.tif", mosaic_nan_mask, spatial_ref, geotransform)  # for debugging

        # Save predicted land cover classes into a HDF5 file
        with h5py.File(fn_save_preds_h5, 'w') as h5_preds:
            h5_preds.create_dataset("predictions", y_pred.shape, data=y_pred)

        end_pred_mosaic = datetime.now()
        pred_mosaic_elapsed = end_pred_mosaic - start_pred_mosaic
        print(f'{end_pred_mosaic}: predictions for complete dataset (mosaic) finished in {pred_mosaic_elapsed}.')

    #=============================================================================
    # Last but not least, save the trained model
    #=============================================================================

    # WARNING! With the current sample size (20%) model is to big and fails to save
    if _save_model:
        print(f"{datetime.now()}: saving trained model (this may take a while)...")
        with open(fn_save_model, 'wb') as f:
            pickle.dump(clf, f)

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
        writer.writerow(['Option: Run (start time)', exec_start])
        writer.writerow(['NAN_VALUE', NAN_VALUE])
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
        writer.writerow(['Labels (mosaic)', fn_mosaic_labels])
        writer.writerow(['Tiles file', fn_tiles])
        writer.writerow(['MONTHLY FEATURES', ''])
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
        writer.writerow(['Features path (last)', feat_path])
        if _read_split:
            # Reading tiles was performed
            writer.writerow(['READING TILES', ''])
            writer.writerow(['Reading started', read_start])
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
        writer.writerow(['Run ended', exec_end])
        writer.writerow(['Run time', exec_time])

    print(f"\n{exec_end}: everything completed on: {exec_time}. Bye ;-)")


if __name__ == '__main__':
    # Control the execution of the land cover classification code

    # Option 1: train RF and predict using the mosaic approach (default)
    run_landcover_classification(save_model=False)