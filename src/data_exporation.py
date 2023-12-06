#!/usr/bin/env python
# coding: utf-8

""" Data exploration
    Data exploration for Yucatan Peninsula (ROI2) Landsat data using tiles in Albers Equal Area projection.

    Generates plots and time series.

@author: Eduardo Jimenez Hernandez <eduardojh@arizona.edu>
@date: 2023-09-27

Changelog:
  2023-09-27: initial code.
"""

import sys
import os
import csv
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import lines
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('ggplot')  # R-like plots

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs

# Extract time series
def get_files(directory, **kwargs):
    """Gets file list in directory filtered by extension (HDF4 default)"""
    _extension = kwargs.get("extension", ".hdf")

    # Look for files in the time series
    assert os.path.isdir(directory), f"Directory not found: {directory}"
    abspath = os.path.abspath(directory)
    # Extract the files in the directory
    dir_items = os.listdir(abspath)
    
    onlyfiles = []
    for item in dir_items:
        absfile = os.path.join(abspath, item)
        # print(absfile)
        _, file_ext = os.path.splitext(absfile)
        # print(file_ext)
        # Filter only the desired files by extension
        if os.path.isfile(absfile) and file_ext == _extension:
            onlyfiles.append(absfile)
    return onlyfiles

def get_time_series(file_list, pos, **kwargs):
    """Gets values from a list of HDF4 files"""
    _window = kwargs.get("window", 7)
    _variable = kwargs.get("variable", "NDVI")

    # List of variables to return
    filenames = []
    dates = []
    years = []
    doys = []
    values = []

    for i, filename in enumerate(file_list):
        print(f"Extracting {_variable} {i+1}/{len(file_list)}: {filename}")
        basename = os.path.basename(filename)
        # basename: LANDSAT08.A2013144.h019v025.hdf
        date = basename[11:18]
        year = int(basename[11:15])
        doy = int(basename[15:18])

        # Read data from file
        data_arr = rs.read_from_hdf(filename, _variable)

        if _window <= 1:
            value = data_arr[pos[0],pos[1]]
        else:
            # Extract values
            row_ini = 0 if pos[0] < _window//2 else pos[0]-_window//2
            row_end = tile_rows if pos[0] > tile_rows-_window//2 else pos[0]+_window//2
            col_ini = 0 if pos[1] < _window//2 else pos[1]-_window//2
            col_end = tile_cols if pos[1] > tile_cols-_window//2 else pos[1]+_window//2
            print(f"Extracting window [{row_ini}:{row_end},{col_ini}:{col_end}]")

            win_values = data_arr[row_ini:row_end,col_ini:col_end]
            print(win_values)
            value = int(np.mean(win_values))
            print(f"Mean value: {value}")

        # Save the values
        filenames.append(basename)
        dates.append(date)
        years.append(year)
        doys.append(doy)
        values.append(value)

    # Create a dictionary with all the values
    ts_data = {'Filename': filenames, 'ADate': dates, 'Year': years, 'DOY': doys, 'Value': values}

    return ts_data


def get_multiple_time_series(file_list, list_pos, **kwargs):
    """Gets values from a list of positions/sites from HDF4 files"""
    _window = kwargs.get("window", 7)
    _variable = kwargs.get("variable", "NDVI")

    # Dictionary with variables to return
    ts_data = {'Filename': [], 'ADate': [], 'Year': [], 'DOY': []}
    for pos in list_pos:
        ts_data[f'Pos_{pos[0]}_{pos[1]}'] = []

    for i, filename in enumerate(file_list):
        print(f"Extracting {_variable} {i+1}/{len(file_list)}: {filename}")
        basename = os.path.basename(filename)
        # basename: LANDSAT08.A2013144.h019v025.hdf
        date = basename[11:18]
        year = int(basename[11:15])
        doy = int(basename[15:18])

        # Read data from file
        data_arr = rs.read_from_hdf(filename, _variable)

        # Get the time series for each position
        for i, pos in enumerate(list_pos):
            print(f"\nGetting value for position {i+1}/{len(list_pos)}: {pos[0]},{pos[1]}")

            if _window <= 1:
                value = data_arr[pos[0],pos[1]]
            else:
                # Extract values
                row_ini = 0 if pos[0] < _window//2 else pos[0]-_window//2
                row_end = tile_rows if pos[0] > tile_rows-_window//2 else pos[0]+_window//2
                col_ini = 0 if pos[1] < _window//2 else pos[1]-_window//2
                col_end = tile_cols if pos[1] > tile_cols-_window//2 else pos[1]+_window//2
                

                win_values = data_arr[row_ini:row_end,col_ini:col_end]
                # print(win_values)
                value = int(np.mean(win_values))
                print(f"Extracting window [{row_ini}:{row_end},{col_ini}:{col_end}], Mean value={value}")
            
            # This list is a row across all list of values
            # e.g. values_pos[1] is the current value for list values[1] and so on
            ts_data[f'Pos_{pos[0]}_{pos[1]}'].append(value)

        # Save the values
        ts_data['Filename'].append(basename)
        ts_data['ADate'].append(date)
        ts_data['Year'].append(year)
        ts_data['DOY'].append(doy)

    return ts_data


def plot_dataset(array: np.ndarray, **kwargs) -> None:
    """ Plots a dataset with a continuous colorbar """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)
    _cmap = kwargs.get('cmap', 'jet')
    _cbartitle = kwargs.get('cbartitle', '')
    _log = kwargs.get('log', False) # logarithmic scale

    array_zoom = array[4100:21300,3200:23500]
    
    # Set max and min
    if _vmax is None:
        _vmax = np.max(array)
    if _vmin is None:
        _vmin = np.min(array)

    fig = plt.figure()
    fig.set_figheight(16)
    fig.set_figwidth(12)

    _cmap = matplotlib.colormaps[_cmap]
    _cmap.set_bad(color='white')

    ax = plt.gca()
    if _log:
        # logarithmic scale
        im = ax.imshow(array_zoom, norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=_vmax), cmap=_cmap)
        _cbartitle = "log"
    else:
        im = ax.imshow(array_zoom, cmap=_cmap, vmax=_vmax, vmin=_vmin)  # Zoom to the ROI
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.grid(False)
    
    if _cbartitle == '':
        plt.colorbar(im, cax=cax)
    else:
        # put the logarithmic label at top of the colorbar
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_title(_cbartitle)

    if _title != '':
        ax.set_title(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()
    plt.close()


def plot_dataset_with_points(array: np.ndarray, list_pos, **kwargs) -> None:
    """ Plots a dataset with a continuous colorbar """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)
    _cmap = kwargs.get('cmap', 'viridis')

    # Set max and min
    if _vmax is None and _vmin is None:
        _vmax = np.max(array)
        _vmin = np.min(array)

    fig = plt.figure()
    fig.set_figheight(16)
    fig.set_figwidth(12)

    ax = plt.gca()
    im = ax.imshow(array, cmap=_cmap, vmax=_vmax, vmin=_vmin)

    # Plot the point positions and create a legend
    series = []
    labels = []
    for pos in list_pos:
        print(f"Plotting site: {pos[0]} {pos[1]}")
        s,  = ax.plot(pos[0], pos[1], label=f"Pos_{pos[0]}_{pos[1]}", marker='o', linestyle=None)  # Plot the point on the specified position
        series.append(s)
        labels.append(s)
    ax.legend(handles=series, labels=labels, loc='best')
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.grid(True)
    
    plt.colorbar(im, cax=cax)

    if _title != '':
        # plt.suptitle(_title)
        ax.set_title(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()
    plt.close()


def plot_time_series_df(df: pd.DataFrame, x: str, y: str, **kwargs):
    """Plots the time series from a Pandas Data Frame"""
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _xlabel = kwargs.get('xlabel', 'x')
    _ylabel = kwargs.get('ylabel', 'y')

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(24)

    ax = plt.gca()
    ax.plot(df[x], df[y], 'mx-')
    ax.grid(True)
    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)

    if _title != '':
        # plt.suptitle(_title)
        ax.set_title(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()
    plt.close()


def plot_multiple_time_series_df(df: pd.DataFrame, x: str, y: list, **kwargs):
    """Plots the time series from a Pandas Data Frame"""
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _xlabel = kwargs.get('xlabel', 'x')
    _ylabel = kwargs.get('ylabel', 'y')

    print(f"Received {len(y)} variables for plotting.")

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(24)

    markers_list = list(lines.Line2D.markers.keys())
    ax = plt.gca()
    for i, series in enumerate(y):
        ax.plot(df[x], df[series], label=series, marker=markers_list[i])
    ax.grid(True)
    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)

    plt.legend(loc='best')

    if _title != '':
        # plt.suptitle(_title)
        ax.set_title(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()
    plt.close()


def plot_seasonal_feats(var: str, fn_features: str, **kwargs):
    """ Plots monthly 2D values from a HDF5 file by reading the variable and the dataset
    """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)
    _cmap = kwargs.get('cmap', 'jet')
    _nan = kwargs.get('nan', -10000)  # Upper NaN threshold

    seasons = ['SPR', 'SUM', 'FAL', 'WIN']

    fig, ax = plt.subplots(2, 2, figsize=(24,16))
    fig.set_figheight(16)
    fig.set_figwidth(24)

    _cmap = matplotlib.colormaps[_cmap]
    _cmap.set_bad(color='magenta')

    for n, season in enumerate(seasons):
        with h5py.File(fn_features, 'r') as h5_feats:
            ds_arr = h5_feats[f'{season} {var} AVG'][:]

        # Set max and min of current dataset
        min_value = np.min(ds_arr)
        max_value = np.max(ds_arr)
        if _vmax is None and _vmin is None:
            _vmax = max_value
            _vmin = min_value

        # Calculate the percentage of missing data
        ds_arr = np.ma.array(ds_arr, mask=(ds_arr < _nan))
        percent = (np.ma.count_masked(ds_arr)/ds_arr.size) * 100
        print(f"    --Missing: {np.ma.count_masked(ds_arr)}/{ds_arr.size}={percent:>0.2f}% min={min_value}, max={max_value}")

        row = n//2
        col = n%2
        im=ax[row,col].imshow(ds_arr, cmap=_cmap, vmax=_vmax, vmin=_vmin)
        ax[row,col].set_title(season + f' {percent:>0.2f}% NaN (<{_nan}) {min_value}-{max_value}')
        ax[row,col].axis('off')
   
    # Single colorbar, easier (WARNING! Uses values from last dataset)
    fig.colorbar(im, ax=ax.ravel().tolist())

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()
    plt.close()


def read_features_mosaic(cwd, fn_landcover, fn_tiles, fn_mask, feature, **kwargs):
    _feat_dir = kwargs.get("features_dir", "features")
    _tile_rows = kwargs.get("tile_rows", 5000)
    _tile_cols = kwargs.get("tile_cols", 5000)
    # _nan_value =kwargs.get("nan", -13000)

    # Read the land cover raster and retrive the land cover classes
    assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
    land_cover, nodata, geotransform, spatial_ref = rs.open_raster(fn_landcover)
    print(f'  Opening raster: {fn_landcover}')
    print(f'    --NoData        : {nodata}')
    print(f'    --Columns       : {land_cover.shape[1]}')
    print(f'    --Rows          : {land_cover.shape[0]}')
    print(f'    --Geotransform  : {geotransform}')
    print(f'    --Spatial ref.  : {spatial_ref}')
    print(f'    --Type          : {land_cover.dtype}')

    # Read the Yucatan Peninsula Aquifer to filter data
    assert os.path.isfile(fn_mask) is True, f"ERROR: File not found! {fn_mask}"
    nodata_mask, _, _, _ = rs.open_raster(fn_mask)
    print(f'  Opening raster: {fn_mask}')
    print(np.unique(nodata_mask, return_counts=True))

    nodata_mask = nodata_mask.filled(0)
    print(np.unique(nodata_mask, return_counts=True))

    # Calculate the extension of the mosaic (in Albers Equal Area proyection coordinates)
    mosaic_extension = {}
    mosaic_extension['W'], xres, _, mosaic_extension['N'], _, yres = [int(x) for x in geotransform]
    mosaic_extension['E'] = mosaic_extension['W'] + tile_cols*xres
    mosaic_extension['S'] = mosaic_extension['N'] + tile_rows*yres
    print(f"\nMosaic extension: {mosaic_extension}")

    # Calculate the extansion of each tile, to insert its data into the mosaic
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

    # The mosaic with the features
    X = np.zeros((land_cover.shape[0], land_cover.shape[1]), dtype=np.int16)

    for i, tile in enumerate(tiles):
        print(f"\n== Reading features for tile {tile} ({i+1}/{len(tiles)}) ==")

        # fn_tile_features = os.path.join(cwd, _feat_dir, f"features_{tile}.h5")  # monthly
        fn_tile_features = os.path.join(cwd, _feat_dir, tile, f"features_season_{tile}.h5")  # seasonal
        # fn_tile_features = os.path.join(cwd, _feat_dir, tile, f"features_season_{tile}_fixed.h5")  # TEST

        # Get rows and columns to insert features
        tile_ext = tiles_extent[tile]

        # Get North and West coordinates convert them to row and column to slice dataset
        nrow = (tile_ext['N'] - mosaic_extension['N'])//yres
        # srow = (tile_ext['S'] - mosaic_extension['N'])//yres
        wcol = (tile_ext['W'] - mosaic_extension['W'])//xres
        # ecol = (tile_ext['E'] - mosaic_extension['W'])//xres

        # tile_nodata = nodata_mask[nrow:srow, wcol:ecol]

        print(f"  Reading the features from: {fn_tile_features}")
        feat_array = np.empty((_tile_rows, _tile_cols), dtype=np.int16)
        with h5py.File(fn_tile_features, 'r') as h5_tile_features:
            # print(f"  Features in file={list(h5_tile_features.keys())}")
            # Get the data from the HDF5 files
            feat_array[:,:] = h5_tile_features[feature][:]
        
        # print(f"mean={np.mean(feat_array)}, stdev={np.std(feat_array)}, min={np.min(feat_array)}, max={np.max(feat_array)}")
        # feat_array = np.where(tile_nodata == 1, feat_array, _nan_value)
        # feat_array = np.ma.array(feat_array, mask=feat_array<=_nan_value)
        # print("After masking")
        # print(f"mean={np.mean(feat_array)}, stdev={np.std(feat_array)}, min={np.min(feat_array)}, max={np.max(feat_array)}")
        
        # Insert tile features in the right position of the 3D array
        print(f"  Inserting dataset into X [{nrow}:{nrow+tile_rows},{wcol}:{wcol+tile_cols}]")
        X[nrow:nrow+tile_rows,wcol:wcol+tile_cols] = feat_array

    # Mask the array
    print(f"X.shape={X.shape}, nodata_mask.shape={nodata_mask.shape} {np.unique(nodata_mask)}")
    print(f"mean={np.mean(X)}, stdev={np.std(X)}, min={np.min(X)}, max={np.max(X)}")
    X = np.ma.masked_where(nodata_mask==0, X)  # mask all values outside ROI (1=Data, 0=NaN)
    print("After masking:")
    print(f"mean={np.mean(X)}, stdev={np.std(X)}, min={np.min(X)}, max={np.max(X)}")

    return X

# def plot_seasonal_feature_old(cwd, fn_landcover, fn_tiles, fn_nodata, feature_list, **kwargs):
#     ## NOT OPTIMIZED FOR SPACING BETWEEN SUBPLOTS
#     _title = kwargs.get('title', '')
#     _savefig = kwargs.get('savefig', '')
#     _dpi = kwargs.get('dpi', 300)
#     _vmax = kwargs.get('vmax', None)
#     _vmin = kwargs.get('vmin', None)
#     _cmap = kwargs.get('cmap', 'jet')
#     _nan = kwargs.get('nan', -10000)  # Upper NaN threshold
#     _tile_rows = kwargs.get("tile_rows", 5000)
#     _tile_cols = kwargs.get("tile_cols", 5000)

#     fig, ax = plt.subplots(2, 2, figsize=(24,16))
#     fig.set_figheight(18)
#     fig.set_figwidth(24)
#     if _title != '':
#         plt.suptitle(_title)
#     # fig.tight_layout()
#     # fig.set_constrained_layout(True)
#     # fig.set_constrained_layout({'w_pad': 0.01, 'h_pad': 0.01})
#     fig.subplots_adjust(wspace=0.01, hspace=0.01)

#     _cmap = matplotlib.colormaps[_cmap]
#     _cmap.set_bad(color='white')

#     for n, feature in enumerate(feature_list):
#         print(f"Generating plot for {feature}")
        
#         # With NAN=0 all values outside the valid ROI will be zero
#         # feature_array = read_features_mosaic(cwd, fn_landcover, fn_tiles, fn_nodata, feature, tile_rows=_tile_rows, tile_cols=_tile_cols, nan=-13000)
#         feature_array = read_features_mosaic(cwd, fn_landcover, fn_tiles, fn_nodata, feature, tile_rows=_tile_rows, tile_cols=_tile_cols)

#         # Set max and min of current dataset
#         min_value = np.min(feature_array)
#         max_value = np.max(feature_array)
#         if _vmax is None and _vmin is None:
#             _vmax = max_value
#             _vmin = min_value

#         # Calculate the percentage of missing data
#         # feature_array = feature_array.filled(0)  # values outside ROI are zero anyway
#         # feature_array = np.ma.array(feature_array, mask=(feature_array < _nan))  # mask out values inside ROI but negative
        
#         # percent = (np.ma.count_masked(feature_array)/feature_array.size) * 100
#         # print(f"Missing values: {np.ma.count_masked(feature_array)}/{feature_array.size}={percent:>0.2f}% min={min_value}, max={max_value}")
#         missing = np.sum(feature_array<_nan)
#         percent = (missing / feature_array.size) * 100
#         print(f"Missing values: {missing}/{feature_array.size}={percent:>0.2f}% min={min_value}, max={max_value}")

#         row = n//2
#         col = n%2
#         # im = ax[row,col].imshow(feature_array, cmap=_cmap, vmax=_vmax, vmin=_vmin)
#         im = ax[row,col].imshow(feature_array[4100:21300,3200:23500], cmap=_cmap, vmax=_vmax, vmin=_vmin)  # Zoom to ROI
#         ax[row,col].set_title(feature + f"(NaN={percent:>0.2f}%)")
#         ax[row,col].axis('off')

#     # Single colorbar, easier (WARNING! Uses values from last dataset)
#     fig.colorbar(im, ax=ax.ravel().tolist())

#     # if _title != '':
#     #     plt.suptitle(_title)
#     if _savefig != '':
#         print(f"\nSaving feature plot: {fn_feat_plot}")
#         fig.savefig(fn_feat_plot, bbox_inches='tight', dpi=_dpi)
#     else:
#         plt.show()
#     plt.close()


# def plot_seasonal_feature_orig(cwd, fn_landcover, fn_tiles, fn_nodata, feature_list, **kwargs):
#     """ Plots features in a four-subplot approach (one per season) for the same area """
#     _title = kwargs.get('title', '')
#     _savefig = kwargs.get('savefig', '')
#     _dpi = kwargs.get('dpi', 300)
#     _vmax = kwargs.get('vmax', None)
#     _vmin = kwargs.get('vmin', None)
#     _cmap = kwargs.get('cmap', 'jet')
#     _nan = kwargs.get('nan', 0)  # Upper NaN threshold
#     _tile_rows = kwargs.get("tile_rows", 5000)
#     _tile_cols = kwargs.get("tile_cols", 5000)
#     _cbartitle = kwargs.get("cbartitle", '')

#     fig, ax = plt.subplots(2, 2)
#     fig.set_figheight(18)
#     fig.set_figwidth(24)
#     fig.subplots_adjust(wspace=0.01, hspace=0.01)

#     _cmap = matplotlib.colormaps[_cmap]
#     _cmap.set_bad(color='white')

#     season_titles = {'SPR': 'Spring',
#                      'SUM': 'Summer',
#                      'FAL': 'Fall',
#                      'WIN': 'Winter'}

#     for n, feature in enumerate(feature_list):
#         print(f"Generating plot for {feature}")
        
#         feature_array = read_features_mosaic(cwd, fn_landcover, fn_tiles, fn_nodata, feature, tile_rows=_tile_rows, tile_cols=_tile_cols)
#         feature_array = feature_array.astype(float)
#         feature_array /= 10000.

#         # Calculate the percentage of missing data
#         # Count based on the unmasked pixels which correspond to the ROI2 mask (outside ROI2 are masked in the array)
#         unmasked = feature_array.count()  # total pixels are unmasked pixels
#         missing = np.sum(feature_array<_nan)  # count the missing pixels of unmasked pixels
#         missing2 = np.sum(np.where(feature_array<_nan, True, False))
#         # missing2 = np.sum(np.where(feature_array<_nan, 1, 0))
#         percent = (missing / unmasked) * 100
#         print(f"Masked pixels: {np.ma.count_masked(feature_array)}, unmasked={unmasked}, total={np.ma.count_masked(feature_array)+unmasked}")
#         print(f"Missing values: {missing}/{unmasked}={percent:>0.2f}%")
#         print(f"Missing values: {missing2}/{unmasked}={(missing2/unmasked)*100:>0.2f}%")

#         # Plot a histogram
#         rs.plot_histogram(feature_array, savefig=_savefig[:-4] + '_hist.png', title=_title)

#         # Remove negative values
#         feature_array = np.ma.masked_array(feature_array, mask=feature_array<_nan)  # mask remaining NaN values
#         print("After removing negatives:")
#         # Set max and min of current dataset
#         min_value = np.min(feature_array)
#         max_value = np.max(feature_array)
#         if _vmax is None:
#             _vmax = max_value
#         if _vmin is None:
#             _vmin = min_value
#         print(f"mean={np.mean(feature_array)}, stdev={np.std(feature_array)}, min={np.min(feature_array)} (used={_vmin}), max={np.max(feature_array)} (used={_vmax})")

#         row = n//2
#         col = n%2
#         # im = ax[row,col].imshow(feature_array, cmap=_cmap, vmax=_vmax, vmin=_vmin)
#         im = ax[row,col].imshow(feature_array[4100:21300,3200:23500], cmap=_cmap, vmax=_vmax, vmin=_vmin)  # Zoom to ROI
#         season = feature.split(' ')[0]
#         ax[row,col].set_title(season_titles[season] + f" (NaN={percent:>0.2f}%)")
#         ax[row,col].axis('off')

#     # Single colorbar, easier (WARNING! Uses values from last dataset)
#     if _title == '':
#         cbar = fig.colorbar(im, ax=ax.ravel().tolist())
#         cbar.ax.set_title(_cbartitle)

#     if _title != '':
#         plt.suptitle(_title, fontsize='x-large')

#         # Make extra space at the top for the suptitle
#         topmargin=1  # inches
#         fig = ax.flatten()[0].figure
#         s = fig.subplotpars
#         w, h = fig.get_size_inches()
        
#         # insert colorbar
#         # divider = make_axes_locatable(ax)
#         # cax = divider.append_axes("right", size="5%", pad=0.05)
#         # cbar = plt.colorbar(im, cax=cax)
#         # cbar.ax.set_title(_cbartitle)

#         figh = h - (1-s.top)*h  + topmargin
#         fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
#         fig.set_figheight(figh)

#         cbar = fig.colorbar(im, ax=ax.ravel().tolist())
#         cbar.ax.set_title(_cbartitle)

#     if _savefig != '':
#         print(f"\nSaving feature plot: {_savefig}")
#         fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
#     else:
#         plt.show()
#     plt.close()


def plot_seasonal_feature(cwd, fn_landcover, fn_tiles, fn_nodata, feature_list, **kwargs):
    """ Plots features in a four-subplot approach (one per season) for the same area """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)
    _cmap = kwargs.get('cmap', 'jet')
    _nan = kwargs.get('nan', 0)  # Upper NaN threshold
    _tile_rows = kwargs.get("tile_rows", 5000)
    _tile_cols = kwargs.get("tile_cols", 5000)
    _cbartitle = kwargs.get("cbartitle", '')
    _log = kwargs.get('log', False) # logarithmic scale

    fig, ax = plt.subplots(2, 2)
    fig.set_figheight(18)
    fig.set_figwidth(24)
    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    _cmap = matplotlib.colormaps[_cmap]
    _cmap.set_bad(color='white')

    season_titles = {'SPR': 'Spring',
                     'SUM': 'Summer',
                     'FAL': 'Fall',
                     'WIN': 'Winter'}

    for n, feature in enumerate(feature_list):
        print(f"\n========== Generating plot for {feature} ==========")

        season = feature.split(' ')[0]
        
        feature_array = read_features_mosaic(cwd, fn_landcover, fn_tiles, fn_nodata, feature, tile_rows=_tile_rows, tile_cols=_tile_cols)
        feature_array = feature_array.astype(float)
        feature_array /= 10000.

        # Calculate the percentage of missing data
        # Count based on the unmasked pixels which correspond to the ROI2 mask (outside ROI2 are masked in the array)
        unmasked = feature_array.count()  # total pixels are unmasked pixels
        missing = np.sum(feature_array<_nan)  # count the missing pixels of unmasked pixels
        values = np.sum(feature_array>=_nan)
        nans = np.sum(feature_array<-1)  # all the < -13000 and other NaNs

        percent = (missing / unmasked) * 100  # invalid pixels
        percent = (nans / unmasked) * 100  # NaN labeled pixels (< -13000)
        print(f"Masked pixels: {np.ma.count_masked(feature_array)}, unmasked={unmasked}, total={np.ma.count_masked(feature_array)+unmasked}")
        print(f"Missing values: {missing}/{unmasked}={percent:>0.2f}%")
        print(f"No Data pixels: {nans}/{unmasked}={(nans/unmasked)*100:>0.2f}%")
        print(f"Pixels with values: {values}/{unmasked}={(values/unmasked)*100:>0.2f}%")

        # Plot a histogram
        print(f"Saving histogram: {_savefig[:-4] + '_hist.png'}")
        rs.plot_histogram(feature_array, savefig=_savefig[:-4] + f'_hist_{season_titles[season]}.png', title=_title, ylog=_log)

        # Remove negative values
        feature_array = np.ma.masked_array(feature_array, mask=feature_array<_nan)  # mask remaining NaN values
        print("After removing negatives:")
        # Set max and min of current dataset
        min_value = np.min(feature_array)
        max_value = np.max(feature_array)
        if _vmax is None:
            _vmax = max_value
        if _vmin is None:
            _vmin = min_value
        print(f"mean={np.mean(feature_array)}, stdev={np.std(feature_array)}, min={np.min(feature_array)} (used={_vmin}), max={np.max(feature_array)} (used={_vmax})")

        row = n//2
        col = n%2
        # im = ax[row,col].imshow(feature_array, cmap=_cmap, vmax=_vmax, vmin=_vmin)
        # im = ax[row,col].imshow(feature_array[4100:21300,3200:23500], cmap=_cmap, vmax=_vmax, vmin=_vmin)  # Zoom to ROI
        if _log:
            # logarithmic scale
            im = ax[row,col].imshow(feature_array[4100:21300,3200:23500], norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=_vmax), cmap=_cmap)
            _cbartitle = "log"
        else:
            im = ax[row,col].imshow(feature_array[4100:21300,3200:23500], cmap=_cmap, vmax=_vmax, vmin=_vmin)  # Zoom to ROI

        ax[row,col].set_title(season_titles[season] + f" (NaN={percent:>0.2f}%)")
        ax[row,col].axis('off')

    # Single colorbar, easier (WARNING! Uses values from last dataset)
    if _title == '' and _cbartitle != '':
        cbar = fig.colorbar(im, ax=ax.ravel().tolist())
        cbar.ax.set_title(_cbartitle)

    if _title != '':
        plt.suptitle(_title, fontsize='x-large')

        # Make extra space at the top for the suptitle
        topmargin=1  # inches
        fig = ax.flatten()[0].figure
        s = fig.subplotpars
        w, h = fig.get_size_inches()
        
        # insert colorbar
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(im, cax=cax)
        # cbar.ax.set_title(_cbartitle)

        figh = h - (1-s.top)*h  + topmargin
        fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
        fig.set_figheight(figh)

        cbar = fig.colorbar(im, ax=ax.ravel().tolist())
        cbar.ax.set_title(_cbartitle)
    
    # # create an axes on the right side of ax. The width of cax will be 5%
    # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # ax.grid(False)

    # if _cbartitle == '':
    #     plt.colorbar(im, cax=cax)
    # else:
    #     # put the logarithmic label at top of the colorbar
    #     cbar = plt.colorbar(im, cax=cax)
    #     cbar.ax.set_title(_cbartitle)

    if _savefig != '':
        print(f"\nSaving feature plot: {_savefig}")
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    else:
        plt.show()
    plt.close()


def apply_scale_factor(dataset, feature):
    """ Transforms the phenology features from "storage" data types into real values.
        This is helpful so the figures show real ranges.
    """
    factors = {
        "SOS": 1.,
        "EOS": 1.,
        "LOS": 1.,
        "DOP": 1.,
        "GDR": 100.,
        "GUR": 100.,
        "MAX": 10000.,
        "SOS2": 1.,
        "EOS2": 1.,
        "LOS2": 1.,
        "DOP2": 1.,
        "GDR2": 100.,
        "GUR2": 100.,
        "MAX2": 10000.,
        "CUM": 10.,
        "NOS": 1
    }
    print(f"Applying scale factor of {factors[feature]} to {feature}")

    return dataset/factors[feature]
    

def disp_info(filename):
    """Displays useful information from a HDF5 file"""
    with h5py.File(filename, 'r') as h5_file:
        print(f"Datasets in file:{filename}")
        for i, dsname in enumerate(h5_file.keys()):
            print(f"{i}: {dsname}")


if __name__ =='__main__':

    tile_cols = 5000
    tile_rows = 5000

    # cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/'
    # mosaic_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/01_MOSAICKING/'
    # stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/02_STATS/'
    # pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/03_PHENO/NDVI/'

    # # Code to test the functions
    # fmt = '%Y_%m_%d-%H_%M_%S'
    exec_start = datetime.now()

    # var = 'NDVI'
    # # Datasets: 'B2 (Blue)' 'B3 (Green)', 'B4 (Red)', 'B5 (Nir)', 'B6 (Swir1)', 'B7 (Mir)', 'NDVI', 'EVI', 'EVI2', 'QA MODIS like'
    # pos = (1500, 3500)
    # tile = 'h22v25'
    # indir = os.path.join(mosaic_dir, 'FILTER', tile)  # IMPORTANT: Use the QA Filtered data
    # fn_time_series = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}_{str(pos[0])}_{str(pos[1])}.csv')
    # fn_pos_plot = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}_{str(pos[0])}_{str(pos[1])}_location.png')
    # fn_ts_plot = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}_{str(pos[0])}_{str(pos[1])}.png')

    # # Get the list of HDF4 files in the directory
    # list_files = get_files(indir)
    # print(f"Found {len(list_files)} files in {indir}")

    # =========================================================================
    # # Create a plot of the position
    # print(f"Saving plot: {fn_pos_plot}")
    # ds = rs.read_from_hdf(list_files[0], var)
    # plot_dataset(ds, pos, savefig=fn_pos_plot)

    # # Extract the time series
    # ts = get_time_series(list_files, pos, variable=var)

    # # Save the time series
    # print(f"Saving time series: {fn_time_series}")
    # df = pd.DataFrame.from_dict(ts)
    # df = df.reset_index(drop=True)
    # # Transform MODIS date into normal date
    # dates = pd.to_datetime(df['ADate'], format="%Y%j")  # Format 'AYYYYDDD'
    # df['Date'] = dates
    # df.to_csv(fn_time_series)

    # # Plot the time series
    # print(f"Saving time series plot to: {fn_ts_plot}")
    # plot_time_series_df(df, 'Date', 'Value', title=f"Time series for {var} at {tile} (x={pos[0]}, y={pos[1]})", savefig=fn_ts_plot, xlabel='Date', ylabel="NDVI")

    # =========================================================================
    #### Compare time series from existing filenames
    # fn_1 = os.path.join(cwd, 'exploration', '2023_09_29-11_26_09_time_series_h22v25_NDVI_400_3500.csv')
    # fn_2 = os.path.join(cwd, 'exploration', '2023_09_29-11_45_41_time_series_h22v25_NDVI_1500_3500.csv')

    # # Get point position from filename (x, y)
    # basename1 = os.path.splitext(os.path.basename(fn_1))[0] # get basename
    # basename2 = os.path.splitext(os.path.basename(fn_2))[0]
    # fn_str1 = basename1.split('_') # Split basename by '_'
    # fn_str2 = basename2.split('_')
    # pos1 = [int(x) for x in fn_str1[-2:]] # Get two last parts of basename
    # pos2 = [int(x) for x in fn_str2[-2:]]
    # tile1 = fn_str1[-4]
    # tile2 = fn_str2[-4]
    # var1 = fn_str1[-3]
    # var2 = fn_str2[-3]
    # print(f"{tile1}, {var1}, {pos1}")
    # print(f"{tile2}, {var2}, {pos2}")

    # assert tile1 == tile2, "Tiles don't match."
    # assert var1 == var2, "Varibles don't match."

    # # Read the time series files
    # df1 = pd.read_csv(fn_1)
    # df2 = pd.read_csv(fn_2)

    #=========================================================================
    # # Get time series for multiple sites at the same time
    # # sites = [(400, 3500), (1500, 3500)]
    # sites = []
    # pos_labels = []
    # for i in [400, 1500]:
    #     for j in range(2500, 4001, 500):
    #         sites.append((i,j))
    #         pos_labels.append(f'Pos_{i}_{j}')

    # fn_time_series = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}.csv')
    # fn_pos_plot = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}_location.png')
    # fn_ts_plot = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}.png')
    
    # # Get the time series
    # ts = get_multiple_time_series(list_files, sites)

    # # Create a plot of the position
    # print(f"Saving plot: {fn_pos_plot}")
    # ds = rs.read_from_hdf(list_files[0], var)
    # plot_dataset_pos(ds, sites, savefig=fn_pos_plot)

    # # Save the time series
    # print(f"Saving time series: {fn_time_series}")
    # df = pd.DataFrame.from_dict(ts)
    # df = df.reset_index(drop=True)
    # # Transform MODIS date into normal date
    # dates = pd.to_datetime(df['ADate'], format="%Y%j")  # Format 'AYYYYDDD'
    # df['Date'] = dates
    # print(df)
    # df.to_csv(fn_time_series)

    # print(sites)
    # print(pos_labels)

    # # Plot the time series
    # print(f"Saving time series plot to: {fn_ts_plot}")
    # # plot_multiple_time_series_df(df, 'Date', pos_labels, title=f"Time series for {var} at {tile}", savefig=fn_ts_plot, xlabel='Date', ylabel=var)
    # plot_multiple_time_series_df(df, 'Date', ['Pos_400_2500', 'Pos_400_3000', 'Pos_400_3500', 'Pos_400_4000'], title=f"Time series for {var} at {tile}", savefig=fn_ts_plot, xlabel='Date', ylabel=var)
    # plot_multiple_time_series_df(df, 'Date', ['Pos_1500_2500', 'Pos_1500_3000', 'Pos_1500_3500', 'Pos_1500_4000'], title=f"Time series for {var} at {tile}", savefig=fn_ts_plot[:-4] + '2.png', xlabel='Date', ylabel=var)

    #=========================================================================
    # Plot variables, test for a single tile
    # var = 'NDVI'
    # tile = 'h19v25'
    # NoData = -10000  # values below are NaN
    # fn_features = os.path.join(cwd, 'features', tile, f'features_season_{tile}.h5')
    # fn_season_plot = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_im_{tile}_{var}.png')
    # print(f"Saving plot" {fn_season_plot})
    # plot_seasonal_feats(var, fn_features, savefig=fn_season_plot, title=f'{var} {tile}', nan=NoData)

    # For all tiles
    # =============================== 2013-2016 ===============================
    # cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/'
    # stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2013_2016/02_STATS/'
    # pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2013_2016/03_PHENO/'
    # fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_grp11_ancillary.tif"
    # var_period = '(2013-2016)'

    # =============================== 2016-2019 ===============================
    # cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/'
    # stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2016_2019/02_STATS/'
    # pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2016_2019/03_PHENO/'
    # fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/data/usv250s6gw_grp11_ancillary.tif"
    # var_period = '(2016-2019)'

    # =============================== 2019-2022 ===============================
    cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/'
    stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2019_2022/02_STATS/'
    pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2019_2022/03_PHENO/'
    fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/data/usv250s7gw_grp11_ancillary.tif"
    var_period = '(2019-2022)'

    fn_tiles = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/parameters/tiles'
    fn_nodata = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/data/YucPenAquifer_mask.tif'
    
    # Plot one picture
    # NoData = 0
    # feature = 'PHEN SOS'
    # # ['FAL BLUE AVG', 'FAL BLUE STDEV', 'FAL EVI AVG', 'FAL EVI STDEV', 'FAL EVI2 AVG', 'FAL EVI2 STDEV', 'FAL GREEN AVG', 'FAL GREEN STDEV', 'FAL MIR AVG', 'FAL MIR STDEV', 'FAL NDVI AVG', 'FAL NDVI STDEV', 'FAL NIR AVG', 'FAL NIR STDEV', 'FAL RED AVG', 'FAL RED STDEV', 'FAL SWIR1 AVG', 'FAL SWIR1 STDEV', 'PHEN CUM', 'PHEN DOP', 'PHEN DOP2', 'PHEN EOS', 'PHEN EOS2', 'PHEN GDR', 'PHEN GDR2', 'PHEN GUR', 'PHEN GUR2', 'PHEN LOS', 'PHEN LOS2', 'PHEN MAX', 'PHEN MAX2', 'PHEN NOS', 'PHEN SOS', 'PHEN SOS2', 'SPR BLUE AVG', 'SPR BLUE STDEV', 'SPR EVI AVG', 'SPR EVI STDEV', 'SPR EVI2 AVG', 'SPR EVI2 STDEV', 'SPR GREEN AVG', 'SPR GREEN STDEV', 'SPR MIR AVG', 'SPR MIR STDEV', 'SPR NDVI AVG', 'SPR NDVI STDEV', 'SPR NIR AVG', 'SPR NIR STDEV', 'SPR RED AVG', 'SPR RED STDEV', 'SPR SWIR1 AVG', 'SPR SWIR1 STDEV', 'SUM BLUE AVG', 'SUM BLUE STDEV', 'SUM EVI AVG', 'SUM EVI STDEV', 'SUM EVI2 AVG', 'SUM EVI2 STDEV', 'SUM GREEN AVG', 'SUM GREEN STDEV', 'SUM MIR AVG', 'SUM MIR STDEV', 'SUM NDVI AVG', 'SUM NDVI STDEV', 'SUM NIR AVG', 'SUM NIR STDEV', 'SUM RED AVG', 'SUM RED STDEV', 'SUM SWIR1 AVG', 'SUM SWIR1 STDEV', 'WIN BLUE AVG', 'WIN BLUE STDEV', 'WIN EVI AVG', 'WIN EVI STDEV', 'WIN EVI2 AVG', 'WIN EVI2 STDEV', 'WIN GREEN AVG', 'WIN GREEN STDEV', 'WIN MIR AVG', 'WIN MIR STDEV', 'WIN NDVI AVG', 'WIN NDVI STDEV', 'WIN NIR AVG', 'WIN NIR STDEV', 'WIN RED AVG', 'WIN RED STDEV', 'WIN SWIR1 AVG', 'WIN SWIR1 STDEV']
    # fn_feat_plot = os.path.join(cwd, 'exploration', f'{feature} {datetime.strftime(exec_start, fmt)}.png')
    # features = read_features_mosaic(cwd, fn_landcover, fn_tiles, fn_nodata, feature, tile_rows=tile_rows, tile_cols=tile_cols, nan=NoData)
    # print(f"\nSaving feature plot: {fn_feat_plot}")
    # plot_dataset(features, title=feature, savefig=fn_feat_plot)

    # # Plot the four seasons in the same figure
    # feature_list = ['SPR NDVI AVG', 'SUM NDVI AVG', 'FAL NDVI AVG', 'WIN NDVI AVG']
    # fn_feat_plot = os.path.join(cwd, 'exploration', f'{var_period} {datetime.strftime(exec_start, fmt)}.png')

    # plot_seasonal_feature(cwd, fn_landcover, fn_tiles, fn_nodata, feature_list, tile_rows=tile_rows, tile_cols=tile_cols, savefig=fn_feat_plot)

    # ===== GENERATE BATCHES OF FIGURES =====

    # Generate a single picture for phenology variables
    feat_list = ['PHEN CUM', 'PHEN DOP', 'PHEN DOP2', 'PHEN EOS', 'PHEN EOS2', 'PHEN GDR', 'PHEN GDR2', 'PHEN GUR', 'PHEN GUR2', 'PHEN LOS', 'PHEN LOS2', 'PHEN MAX', 'PHEN MAX2', 'PHEN NOS', 'PHEN SOS', 'PHEN SOS2']
    # feat_list = ['PHEN GUR','PHEN GUR2', 'PHEN GDR', 'PHEN GDR2']
    # feat_list = ['PHEN CUM', 'PHEN DOP', 'PHEN DOP2', 'PHEN EOS', 'PHEN EOS2', 'PHEN LOS', 'PHEN LOS2', 'PHEN MAX', 'PHEN MAX2', 'PHEN NOS', 'PHEN SOS', 'PHEN SOS2']
    VI = 'NDVI'
    titles = {'PHEN CUM': f'Cumulative {VI}',
              'PHEN DOP': f'Day of Peak Season 1 [DOY]',
              'PHEN DOP2': f'Day of Peak Season 2 [DOY]',
              'PHEN EOS': f'End of Season 1 [DOY]',
              'PHEN EOS2': f'End of Season 2 [DOY]',
              'PHEN GDR': f'Rate of Senescence Season 1 [{VI}/Day]',
              'PHEN GDR2':  f'Rate of Senescence Season 2 [{VI}/Day]',
              'PHEN GUR': f'Rate of Greening Season 1 [{VI}/Day]',
              'PHEN GUR2': f'Rate of Greening Season 2 [{VI}/Day]',
              'PHEN LOS': f'Length of Season 1 [Days]',
              'PHEN LOS2': f'Length of Season 2 [Days]',
              'PHEN MAX': f'Maximum VI Season 1',
              'PHEN MAX2': f'Maximum VI Season 2',
              'PHEN NOS': f'Number of Seasons',
              'PHEN SOS': f'Start of Season 1 [DOY]',
              'PHEN SOS2': f'Start of Season 2 [DOY]'}
    
    for i, feature in enumerate(feat_list):
        print(f"Generating plot ({i+1}/{len(feat_list)}): {feature}")
        fn_feat_plot = os.path.join(cwd, 'exploration', f'{feature} {var_period}.png')
        fn_feat_plot_hist = fn_feat_plot[:-4] + '_hist.png'
        features = read_features_mosaic(cwd, fn_landcover, fn_tiles, fn_nodata, feature, tile_rows=tile_rows, tile_cols=tile_cols)
        features = apply_scale_factor(features, feature[5:])

        print(f"\nSaving feature plot: {fn_feat_plot}")

        # # For GDR and GUR take only positive values, and compare use logarithmic scale
        # if 'GUR' in feature or 'GDR' in feature:
        #     print("Masking negative values...")
        #     features = np.ma.masked_array(features, mask=features<0)
        
        # log scale plots
        plot_dataset(features, title=titles[feature], savefig=fn_feat_plot[:-4] + '_log.png', log=True)
        rs.plot_histogram(features, title=titles[feature], savefig=fn_feat_plot_hist[:-4] + '_log.png', ylog=True)
        # normal scale plot
        plot_dataset(features, title=titles[feature], savefig=fn_feat_plot)
        rs.plot_histogram(features, title=titles[feature], savefig=fn_feat_plot_hist)
    
    # Generate a four-seasonal plot for each band  in the same figure
    seasons = ['SPR', 'SUM', 'FAL', 'WIN']
    feat_list = ['BLUE AVG', 'BLUE STDEV', 'EVI AVG', 'EVI STDEV', 'EVI2 AVG', 'EVI2 STDEV', 'GREEN AVG', 'GREEN STDEV', 'MIR AVG', 'MIR STDEV', 'NDVI AVG', 'NDVI STDEV', 'NIR AVG', 'NIR STDEV', 'RED AVG', 'RED STDEV', 'SWIR1 AVG', 'SWIR1 STDEV']
    # feat_list = ['BLUE STDEV', 'EVI2 AVG']

    for feat in feat_list:
        custom_title = 'Average ' if feat.split(' ')[1] == 'AVG' else 'Standard deviation of ' + 'surface reflectance of ' + feat.split(' ')[0].capitalize()  + ' band by season'

        feature_list = []
        for season in seasons:
            feature_list.append(f"{season} {feat}")
        print(f"Generating figure for: {feat}")
        fn_feat_plot = os.path.join(cwd, 'exploration', f'{feat} {var_period}.png')
        # plot_seasonal_feature(cwd, fn_landcover, fn_tiles, fn_nodata, feature_list, tile_rows=tile_rows, tile_cols=tile_cols, savefig=fn_feat_plot)
        
        colormap = 'jet'
        if 'VI' in feat: # works for NDVI, EVI and EVI2
            colormap = 'viridis'
            custom_title = ('Average ' if feat.split(' ')[1] == 'AVG' else 'Standard deviation of ') + feat.split(' ')[0]  + ' by season'
        elif 'BLUE' in feat:
            colormap = 'Blues_r'
        elif 'RED' in feat:
            colormap = 'Reds_r'
        elif 'GREEN' in feat:
            colormap = 'Greens_r'

        # Missing values are below -1 (or -10000) other negative values are not missing but still invalid, so all is grouped as NaN's below 0.
        # IMPORTANT: For the AVG variable negative values of surface reflectance and VI are invalid, for SD negative values are also invalid.
        plot_seasonal_feature(cwd, fn_landcover, fn_tiles, fn_nodata, feature_list, tile_rows=tile_rows, tile_cols=tile_cols, savefig=fn_feat_plot, cmap=colormap, title=custom_title, vmin=0, vmax=1)
        plot_seasonal_feature(cwd, fn_landcover, fn_tiles, fn_nodata, feature_list, tile_rows=tile_rows, tile_cols=tile_cols, savefig=fn_feat_plot[:-4] + '_log.png', cmap=colormap, title=custom_title, vmin=0, vmax=1, log=True)
        elapsed = datetime.now() - exec_start
        print(f"Processed {feat}: elapsed {elapsed}")

    # disp_info('/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/features/h19v25/features_h19v25.h5')

    print(f"Everything finised in: {datetime.now() - exec_start}. Done ;-)")