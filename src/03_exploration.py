#!/usr/bin/env python
# coding: utf-8

""" Data exploration

Eduardo Jimenez <eduardojh@email.arizona.edu>

NOTE: run under 'rstf' conda environment (python 3.8.13, keras 2.9.0)

 VI values are scaled by 10000; GUR and GDR are scaled by 100, then by 10000, thus real values are really small.
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

# Load feature valid ranges from file
ranges = pd.read_csv(cwd + 'valid_ranges', sep='=', index_col=0)
MIN_BAND = ranges.loc['MIN_BAND', 'VALUE']
MAX_BAND = ranges.loc['MAX_BAND', 'VALUE']
MIN_VI = ranges.loc['MIN_VI', 'VALUE']
MAX_VI = ranges.loc['MAX_VI', 'VALUE']
MIN_PHEN = ranges.loc['MIN_PHEN', 'VALUE']
NAN_VALUE = ranges.loc['NAN_VALUE', 'VALUE']


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
                minima = MIN_PHEN
            
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
    """ Plots histograms of all the bands in the HDF file, two plots are generated: one with all values, and a second
        plot removes the values out of the valid range."""
    with h5py.File(fn_hdf_feat, 'r') as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            start = datetime.now()
            ds = f[key][:]
            # print(f'ds={ds.shape}')

            # Remove values out of the valid range
            minima = MIN_BAND
            # Add the type of feature
            feat_type = 'BAND'
            if key[0:4] == 'PHEN':
                minima = MIN_PHEN
            elif key[4:8] == 'EVI ' or  key[4:8] == 'NDVI' or  key[4:8] == 'EVI2':
                minima = MIN_VI
            
            if key == 'PHEN GDR' or key == 'PHEN GDR2' or key == 'PHEN GUR' or key == 'PHEN GUR2':
                minima = MIN_PHEN

            # print('Plotting histogram...')
            n_bins = 30
            ds1 = ds.flatten()
            # print(f'ds1={ds1.shape}')
            ds2 = np.where(ds1 >= minima, ds1, np.nan)
            # print(f'ds2={ds2.shape}')
            
            fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

            axs[0].hist(ds1, bins=n_bins)  # histogram of all values
            axs[1].hist(ds2, bins=n_bins//2)  # histogram of only valid values

            plt.suptitle(key)
            plt.savefig(fn_hist_plot + ' ' + key + '.png', bbox_inches='tight', dpi=300)
            elapsed = datetime.now() - start
            print(f'Plotting histogram {key:>20} in {elapsed}.')
            plt.close()

def range_of_type(feat_type: str, df: pd.DataFrame, **kwargs) -> None:
    """ Shows the range of a type of feature"""
    _verbose = kwargs.get('verbose', False)
    # Get the range for the specified type
    print(f"Showing range for type: {feat_type}")
    df_feats = df.loc[df['Type'] == feat_type]

    if _verbose:
        print(df_feats.head())
        print(df_feats.shape)

    print(f"{'Variable':>10} {'Minima':>10} {'Maxima':>10} {'Raw Min':>10} {'Raw Max':>10} {'Sum Min':>10}")

    if feat_type == 'PHEN':
        df_pheno = df_feats.loc[df_feats['Variable'] == 'VAL']
        rows, _ = df_pheno.shape
        for i in range(rows):
            print(f"{df_pheno.iloc[i]['Key']:>10} {df_pheno.iloc[i]['Min']:>10.2f} {df_pheno.iloc[i]['Max']:>10.2f} {df_pheno.iloc[i]['Min Raw']:>10.2f} {df_pheno.iloc[i]['Max Raw']:>10.2f} {'--':>10}")
    else:
        
        avg = df_feats.loc[df_feats['Variable'] == 'AVG']
        if _verbose:
            print(avg.head())
            print(avg.shape)
        print(f"{'AVG':>10} {np.min(avg['Min']):>10.2f} {np.max(avg['Max']):>10.2f} {np.min(avg['Min Raw']):>10.2f} {np.max(avg['Max Raw']):>10.2f} {'--':>10}")
        
        _min = df_feats.loc[df_feats['Variable'] == 'MIN']
        if _verbose:
            print(_min.head())
            print(_min.shape)
        print(f"{'MIN':>10} {np.min(_min['Min']):>10.2f} {np.max(_min['Max']):>10.2f} {np.min(_min['Min Raw']):>10.2f} {np.max(_min['Max Raw']):>10.2f} {'--':>10}")

        _max = df_feats.loc[df_feats['Variable'] == 'MAX']
        if _verbose:
            print(_max.head())
            print(_max.shape)
        print(f"{'MAX':>10} {np.min(_max['Min']):>10.2f} {np.max(_max['Max']):>10.2f} {np.min(_max['Min Raw']):>10.2f} {np.max(_max['Max Raw']):>10.2f} {'--':>10}")

        std = df_feats.loc[df_feats['Variable'] == 'STD']
        if _verbose:
            print(std.head())
            print(std.shape)
        print(f"{'STD':>10} {np.min(std['Min']):>10.2f} {np.max(std['Max']):>10.2f} {np.min(std['Min Raw']):>10.2f} {np.max(std['Max Raw']):>10.2f} {'--':>10}")

        npixels = df_feats.loc[df_feats['Variable'] == 'NPI']
        if _verbose:
            print(npixels.head())
            print(npixels.shape)
        print(f"{'NPI':>10} {np.min(npixels['Min']):>10.2f} {np.max(npixels['Max']):>10.2f} {np.min(npixels['Min Raw']):>10.2f} {np.max(npixels['Max Raw']):>10.2f} {np.sum(npixels['Min']):>10.2f}")


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
    fn_ranges = cwd + 'valid_ranges'

    # basic_stats(fn_features, fn_labels, fn_feat_stats)

    # Read saved stats from CSV file
    df = pd.read_csv(fn_feat_stats)

    range_of_type('BAND', df)
    range_of_type('VI', df)
    range_of_type('PHEN', df)

    # Central tendency: mean, median, mode

    # Dispersion: range, quantiles, interquantile range, outliers

    # Dispersion: variance, standard deviation, mean abosolute deviation

    # Boxplot, quantile plot, q-q plot, barchart, histogram, scatterplot

    # Similarity, dissimilarity, proximity (for ordinal)

    # Standarizing?

    # DATA PREPROCESSING

    # Missing data: ignore. Okay for classification

