#!/usr/bin/env python
# coding: utf-8

""" Data exploration

Eduardo Jimenez <eduardojh@email.arizona.edu>

NOTE: run under 'rstf' conda environment (python 3.8.13, keras 2.9.0)
"""

import sys
import platform
import h5py
import numpy as np
import pandas as pd
from math import ceil
from matplotlib import pyplot as plt

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


if __name__ == '__main__':

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
    fn_band_stats = cwd + 'band_stats_summary.csv'

    ls = []
    names = ['Variable', 'Min Prev', 'Max Prev', 'Mean Prev', 'Min', 'Max', 'Mean']

    MIN_VAL = -11000  # In theory min 
    NAN_VALUE = -13000

    # Check labels
    with h5py.File(fn_labels, 'r') as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            print(f"Analyzing {i:>3}/{len(keys):>3}:{key:>22}", end='')
            ds = f[key][:]
            print(f"{str(ds.dtype):>8}", end='')
            _min = np.nanmin(ds)
            _max = np.nanmax(ds)
            avg = np.nanmean(ds)
            u = np.unique(ds)
            print(f" min={_min:>9.2f} max={_max:>9.2f} avg={avg:>9.2f}")
            print(f" unique: {len(u)}: {u}")

    # Check features
    with h5py.File(fn_features, 'r') as f:
        keys = list(f.keys())

        for i, key in enumerate(keys):
            row = []
            print(f"Analyzing {i:>3}/{len(keys):>3}:{key:>22}", end='')

            row.append(key)
            ds = f[key][:]
            # print(f"{str(ds.dtype):>8}{str(ds.shape):>13}", end='')
            print(f"{str(ds.dtype):>8}", end='')

            _min = np.nanmin(ds)
            _max = np.nanmax(ds)
            avg = np.nanmean(ds)

            row.append(_min)
            row.append(_max)
            row.append(avg)

            print(f" min={_min:>9.2f} max={_max:>9.2f} avg={avg:>9.2f}", end='')

            # Check for custom NAN values
            valid_ds = np.where(ds >= MIN_VAL, ds, np.nan)

            _min = np.nanmin(valid_ds)
            _max = np.nanmax(valid_ds)
            avg = np.nanmean(valid_ds)

            row.append(_min)
            row.append(_max)
            row.append(avg)

            print(f" min={_min:>9.2f} max={_max:>9.2f} avg={avg:>9.2f}")

            ls.append(row)

    df = pd.DataFrame(ls)
    df.columns = names  # rename columns
    print(df.shape)
    print(df.info())
    df.to_csv(fn_band_stats)

    # Central tendency: mean, median, mode

    # Dispersion: range, quantiles, interquantile range, outliers

    # Dispersion: variance, standard deviation, mean abosolute deviation

    # Boxplot, quantile plot, q-q plot, barchart, histogram, scatterplot

    # Similarity, dissimilarity, proximity (for ordinal)

    # Standarizing?

    # DATA PREPROCESSING

    # Missing data: ignore. Okay for classification

