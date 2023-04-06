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
#import seaborn as sns
from math import ceil
from matplotlib import pyplot as plt
from datetime import datetime

plt.style.use('ggplot')  # R-like plots
#sns.set_theme()

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

#MIN_VAL = -11000
MIN_BAND = 0  # In theory min value for bands
MIN_VI = -10000
MIN_PHEN = 0
NAN_VALUE = -13000
MIN_PHEN_RATE = -40000


def basic_stats(fn_hdf_feat, fn_hdf_lbl, fn_csv = ''):
    """ Generates basic stats from raw data (before preprocessing) """
    ls = []
    names = ['Key', 'Type', 'Variable', 'Min Raw', 'Max Raw', 'Mean Raw', 'Min', 'Max', 'Mean']

    # Check labels
    with h5py.File(fn_hdf_lbl, 'r') as f:
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
    with h5py.File(fn_hdf_feat, 'r') as f:
        keys = list(f.keys())

        for i, key in enumerate(keys):
            row = []
            print(f"{i:>3}/{len(keys):>3}:{key:>22}", end='')

            row.append(key)
            ds = f[key][:]
            # print(f"{str(ds.dtype):>8}{str(ds.shape):>13}", end='')
            print(f"{str(ds.dtype):>8}", end='')
            
            minima = MIN_BAND
            # Add the type of feature
            feat_type = 'BAND'
            if key[0:4] == 'PHEN':
                feat_type = 'PHEN'
                minima = MIN_PHEN
            elif key[4:8] == 'EVI ' or  key[4:8] == 'NDVI' or  key[4:8] == 'EVI2':
                feat_type = 'VI'
                minima = MIN_VI
            print(f"{feat_type:>5}", end='')
            row.append(feat_type)
            
            if key == 'PHEN GDR' or key == 'PHEN GDR2' or key == 'PHEN GUR' or key == 'PHEN GUR2':
                minima = MIN_PHEN_RATE
            
            var = 'VAL'
            if key[-3:] == 'AVG':
                var = 'AVG'
            elif key[-3:] == 'MAX' and feat_type != 'PHEN':
                var = 'MAX'
            elif key[-3:] == 'MIN':
                var = 'MIN'
            elif key[-3:] == 'els':
                var = 'NPI'
            elif key[-3:] == 'DEV':
                var = 'STD'
            print(f"{var:>4}", end='')
            row.append(var)

            _min = np.nanmin(ds)
            _max = np.nanmax(ds)
            avg = np.nanmean(ds)

            row.append(_min)
            row.append(_max)
            row.append(avg)

            print(f" min={_min:>9.2f} max={_max:>9.2f} avg={avg:>9.2f}", end='')

            # Remove extreme negative values (custom NANs)
            valid_ds = np.where(ds >= minima, ds, np.nan)

            _min = np.nanmin(valid_ds)
            _max = np.nanmax(valid_ds)
            avg = np.nanmean(valid_ds)

            row.append(_min)
            row.append(_max)
            row.append(avg)

            print(f" min={_min:>9.2f} max={_max:>9.2f} avg={avg:>9.2f}")

            ls.append(row)
            
        # Save stats to a CSV file
        if fn_csv != '':
            df = pd.DataFrame(ls)
            df.columns = names  # rename columns
            print(df.shape)
            print(df.info())
            df.to_csv(fn_csv)

def plot_hist_bands(fn_hdf_feat):
    with h5py.File(fn_hdf_feat, 'r') as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            start = datetime.now()
            ds = f[key][:]
            # print(f'ds={ds.shape}')

            minima = MIN_BAND
            # Add the type of feature
            feat_type = 'BAND'
            if key[0:4] == 'PHEN':
                minima = MIN_PHEN
            elif key[4:8] == 'EVI ' or  key[4:8] == 'NDVI' or  key[4:8] == 'EVI2':
                minima = MIN_VI
            
            if key == 'PHEN GDR' or key == 'PHEN GDR2' or key == 'PHEN GUR' or key == 'PHEN GUR2':
                minima = MIN_PHEN_RATE

            # print('Plotting histogram...')
            n_bins = 30
            ds1 = ds.flatten()
            # print(f'ds1={ds1.shape}')
            ds2 = np.where(ds1 >= minima, ds1, np.nan)
            # print(f'ds2={ds2.shape}')
            
            fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

            axs[0].hist(ds1, bins=n_bins)
            axs[1].hist(ds2, bins=n_bins//2)

            plt.suptitle(key)
            plt.savefig(fn_hist_plot + ' ' + key + '.png', bbox_inches='tight', dpi=300)
            elapsed = datetime.now() - start
            print(f'Plotting istogram {key} in {elapsed}.')

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
    fn_feat_stats = cwd + 'data_exploration/feature_stats_summary.csv'
    fn_hist_plot = cwd + 'data_exploration/hist'

#    basic_stats(fn_features, fn_labels, fn_feat_stats)

    # Read saved stats from CSV file
    df = pd.read_csv(fn_feat_stats)

    # Explore NPixels variable, it goes across BANDS, VI and PHENO
    npixels = df.loc[df['Variable'] == 'NPI']
    # print(npixels)
    print(npixels.head())
    # print(npixels.info())
    print(npixels.shape)
    print(np.min(npixels['Min Raw']), np.max(npixels['Min Raw']), np.sum(npixels['Min Raw']))
    print(np.min(npixels['Max Raw']), np.max(npixels['Max Raw']), np.sum(npixels['Max Raw']))
    print(np.min(npixels['Min']), np.max(npixels['Min']), np.sum(npixels['Min']))
    print(np.min(npixels['Max']), np.max(npixels['Max']))
    print(f" *** Range for NPixels is {np.min(npixels['Min'])}-{np.max(npixels['Max'])} *** ")

    df_bands = df.loc[df['Type'] == 'BAND']
    print(df_bands.head())
    print(df_bands.shape)
    
    avg = df_bands.loc[df_bands['Variable'] == 'AVG']
    print(avg.head())
    print(avg.shape)
    print(f" *** Range for BANDS AVG is {np.min(avg['Min'])}-{np.max(avg['Max'])} ***")
    
    _min = df_bands.loc[df_bands['Variable'] == 'MIN']
    print(_min.head())
    print(_min.shape)
    print(f" *** Range for BANDS MIN is {np.min(_min['Min'])}-{np.max(_min['Max'])} ***")

    _max = df_bands.loc[df_bands['Variable'] == 'MAX']
    print(_max.head())
    print(_max.shape)
    print(f" *** Range for BANDS MAX is {np.min(_max['Min'])}-{np.max(_max['Max'])} ***")

    std = df_bands.loc[df_bands['Variable'] == 'STD']
    print(std.head())
    print(std.shape)
    print(f" *** Range for BANDS STDEV is {np.min(std['Min'])}-{np.max(std['Max'])} ***")
    

#     var = df['Key'][0]
#     print(var)

#     start = datetime.now()
#     with h5py.File(fn_features, 'r') as f:
#         ds = f[var][:]
#     print(f'ds={ds.shape}')

#     print('Plotting histogram...')
#     n_bins = 30
#     ds1 = ds.flatten()
#     print(f'ds1={ds1.shape}')
#     ds2 = np.where(ds1 >= MIN_BAND, ds1, np.nan)
#     print(f'ds2={ds2.shape}')
    
#     fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

#     axs[0].hist(ds1, bins=n_bins)
#     axs[1].hist(ds2, bins=n_bins)

#     plt.suptitle(var)
#     plt.savefig(fn_hist_plot + ' ' + var + '.png', bbox_inches='tight', dpi=300)
# #    plt.show()
#     elapsed = datetime.now() - start
#     print(f'Done plotting in {elapsed}.')

    plot_hist_bands(fn_features)

    # Central tendency: mean, median, mode

    # Dispersion: range, quantiles, interquantile range, outliers

    # Dispersion: variance, standard deviation, mean abosolute deviation

    # Boxplot, quantile plot, q-q plot, barchart, histogram, scatterplot

    # Similarity, dissimilarity, proximity (for ordinal)

    # Standarizing?

    # DATA PREPROCESSING

    # Missing data: ignore. Okay for classification

