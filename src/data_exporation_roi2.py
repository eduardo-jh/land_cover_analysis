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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
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
        
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.grid(False)
    
    plt.colorbar(im, cax=cax)
    plt.plot(pos[0], pos[1], 'w*', markersize=20)

    if _title != '':
        # plt.suptitle(_title)
        ax.set_title(_title)
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
    pos = (2500, 2500)
    tile = 'h19v25'
    indir = os.path.join(mosaic_dir, tile)
    fn_time_series = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}_{str(pos[0])}_{str(pos[1])}.csv')
    fn_pos_plot = os.path.join(cwd, 'exploration', f'{datetime.strftime(exec_start, fmt)}_time_series_{tile}_{var}_{str(pos[0])}_{str(pos[1])}.png')

    # Get the list of HDF4 files in the directory
    list_files = get_files(indir)
    print(f"Found {len(list_files)} files in {indir}")

    # Create a plot of the position
    print(f"Saving plot: {fn_pos_plot}")
    ds = rs.read_from_hdf(list_files[0], var)
    plot_dataset(ds, pos, savefig=fn_pos_plot)

    # # Extract the time series
    # ts = get_time_series(list_files, pos, variable=var)

    # # Save the time series
    # print(f"Saving time series: {fn_time_series}")
    # df = pd.DataFrame.from_dict(ts)
    # df = df.reset_index(drop=True)
    # # Transform MODIS date into normal date
    # dates = pd.to_datetime(df['ADate'], format="A%Y%j")  # Format 'AYYYYDDD'
    # df['Date'] = dates
    # df.to_csv(fn_time_series)