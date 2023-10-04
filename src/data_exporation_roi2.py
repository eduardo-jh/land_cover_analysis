#!/usr/bin/env python
# coding: utf-8

""" Data exploration
    Data exploration for Yucatan Peninsula (ROI2) Landsat data using tiles in
    Albers Equal Area projection.

@author: Eduardo Jimenez Hernandez <eduardojh@arizona.edu>
@date: 2023-09-27

Changelog:
  2023-09-27: initial code.
"""

import sys
import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import lines
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')
cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/'
mosaic_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/01_MOSAICKING/'
stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/02_STATS/'
pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/03_PHENO/NDVI/'

import rsmodule as rs

tile_cols = 5000
tile_rows = 5000

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


def plot_dataset(array: np.ndarray, pos, **kwargs) -> None:
    """ Plots a dataset with a continuous colorbar """
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)
    # Set max and min
    if _vmax is None and _vmin is None:
        _vmax = np.max(array)
        _vmin = np.min(array)

    fig = plt.figure()
    fig.set_figheight(16)
    fig.set_figwidth(12)

    ax = plt.gca()
    im = ax.imshow(array, cmap='jet', vmax=_vmax, vmin=_vmin)
    ax.plot(pos[0], pos[1], 'mo')  # Plot the point on the specified position
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
        # fn = cwd + f'02_STATS/MONTHLY.{var.upper()}.{str(n+1).zfill(2)}.{month}.hdf'
        # fn = cwd + f'data/landsat/C2/02_STATS/MONTHLY.{var.upper()}.{month}.hdf'
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


def plot_dataset_pos(array: np.ndarray, list_pos, **kwargs) -> None:
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


if __name__ =='__main__':
    # Code to test the functions
    fmt = '%Y_%m_%d-%H_%M_%S'
    exec_start = datetime.now()

    var = 'NDVI'
    # Datasets: 'B2 (Blue)' 'B3 (Green)', 'B4 (Red)', 'B5 (Nir)', 'B6 (Swir1)', 'B7 (Mir)', 'NDVI', 'EVI', 'EVI2', 'QA MODIS like'
    pos = (1500, 3500)
    tile = 'h22v25'
    indir = os.path.join(mosaic_dir, 'FILTER', tile)  # IMPORTANT: Use the QA Filtered data
    fn_time_series = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}_{str(pos[0])}_{str(pos[1])}.csv')
    fn_pos_plot = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}_{str(pos[0])}_{str(pos[1])}_location.png')
    fn_ts_plot = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}_{str(pos[0])}_{str(pos[1])}.png')

    # Get the list of HDF4 files in the directory
    list_files = get_files(indir)
    print(f"Found {len(list_files)} files in {indir}")

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
    # Plot variables
    var = 'NDVI'
    tile = 'h22v25'
    NoData = -10000  # values below are NaN
    fn_features = os.path.join(cwd, 'features', tile, f'features_season_{tile}.h5')
    fn_season_plot = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_im_{tile}_{var}.png')
    plot_seasonal_feats(var, fn_features, savefig=fn_season_plot, title=f'{var} {tile}', nan=NoData)
