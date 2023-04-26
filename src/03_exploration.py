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
from math import ceil
from matplotlib import pyplot as plt
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('ggplot')  # R-like plots

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
            print(f'Feature stats saved to: {fn_csv}.')


def plot_2hist_bands(fn_hdf_feat, fn_hist_plot):
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
            ds1 = ds.flatten()
            # print(f'ds1={ds1.shape}')
            ds2 = np.where(ds1 >= minima, ds1, np.nan)
            # print(f'ds2={ds2.shape}')
            
            plot_2hist(ds1, ds2, title=key, half=True, bins=30, savefig=fn_hist_plot + ' ' + key + '.png')
            elapsed = datetime.now() - start
            print(f'Plotting histogram {key:>20} in {elapsed}.')
            plt.close()


def plot_2hist(ds1, ds2, **kwargs):
    """ Plots 2 histograms side by side."""
    _feature = kwargs.get('feature', '')
    _bins = kwargs.get('bins', 30)
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _2half = kwargs.get('half', True)  # half the bins in second histogram
    
    ds1 = ds1.flatten()
    ds2 = ds2.flatten()

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    axs[0].hist(ds1, bins=_bins)
    axs[1].hist(ds2, bins=_bins//2 if _2half else _bins)

    # plt.suptitle(key)
    # plt.savefig(fn_hist_plot + ' ' + key + '.png', bbox_inches='tight', dpi=300)
    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        plt.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    plt.close()


def plot_hist(ds, **kwargs):
    """ Plots histogram of features in the HDF file"""
    _feature = kwargs.get('feature', '')
    _bins = kwargs.get('bins', 30)
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)

    ds = ds.flatten()
    # print(f'ds1={ds1.shape}')
    
    fig = plt.figure(figsize=(16,12), tight_layout=True)

    plt.hist(ds, bins=_bins)  # histogram of all values

    if _title != '':
        plt.title(_title)
    if _savefig != '':
        plt.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
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


def plot_monthly(var, ds, **kwargs):
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    fig, ax = plt.subplots(3, 4, figsize=(24,16))
    fig.set_figheight(16)
    fig.set_figwidth(24)

    for n, month in enumerate(months):
        fn = cwd + f'02_STATS/MONTHLY.{var.upper()}.{str(n+1).zfill(2)}.{month}.hdf'
        print(fn)
        ds_arr = rs.read_from_hdf(fn, ds)

        # Set max and min
        if _vmax is None and _vmin is None:
            _vmax = np.max(ds_arr)
            _vmin = np.min(ds_arr)

        row = n//4
        col = n%4
        # print(f'Row={row}, Col={col}')
        im=ax[row,col].imshow(ds_arr, cmap='jet', vmax=_vmax, vmin=_vmin)
        ax[row,col].set_title(month)
        ax[row,col].axis('off')
   
    # fig.tight_layout()

    # Single colorbar
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # Single colorbar, easier
    fig.colorbar(im, ax=ax.ravel().tolist())

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    plt.show()
    plt.close()


def plot_monthly_hist(var, ds, **kwargs):
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _bins = kwargs.get('bins', 30)

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    fig, ax = plt.subplots(3, 4, figsize=(24,16))
    fig.set_figheight(16)
    fig.set_figwidth(24)

    for n, month in enumerate(months):
        fn = cwd + f'02_STATS/MONTHLY.{var.upper()}.{str(n+1).zfill(2)}.{month}.hdf'
        print(fn)
        ds_arr = rs.read_from_hdf(fn, ds)

        row = n//4
        col = n%4

        ax[row,col].hist(ds_arr.flatten(), bins=_bins, color='blue')  # histogram of all values
        ax[row,col].set_title(month)

        # Leave labels on left and bottom axis only
        if col != 0:
            ax[row,col].set_yticklabels([])
            ax[row,col].tick_params(left=False)
        if row != 2:
            ax[row,col].set_xticklabels([])
            ax[row,col].tick_params(bottom=False)

    # Share the y-axis along rows
    ax[0, 0].get_shared_y_axes().join(ax[0,0], *ax[0,:])
    # ax[0, 0].autoscale()

    ax[1, 0].get_shared_y_axes().join(ax[1,0], *ax[1,:])
    ax[1, 0].autoscale()

    ax[2, 0].get_shared_y_axes().join(ax[2,0], *ax[2,:])
    ax[2, 0].autoscale()

    # Share the x-axis along columns
    ax[0, 0].get_shared_x_axes().join(ax[0,0], *ax[:,0])
    ax[0, 0].autoscale()

    ax[0, 1].get_shared_x_axes().join(ax[0,1], *ax[:,1])
    ax[0, 1].autoscale()

    ax[0, 2].get_shared_x_axes().join(ax[0,2], *ax[:,2])
    ax[0, 2].autoscale()

    ax[0, 3].get_shared_x_axes().join(ax[0,3], *ax[:,3])
    ax[0, 3].autoscale()

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)
    # plt.show()
    plt.close()


def plot_hdf_dataset(filename, ds, **kwargs):
    _title = kwargs.get('title', '')
    _savefig = kwargs.get('savefig', '')
    _dpi = kwargs.get('dpi', 300)
    _vmax = kwargs.get('vmax', None)
    _vmin = kwargs.get('vmin', None)

    ds_arr = rs.read_from_hdf(filename, ds)

    plot_dataset(ds_arr, title=_title, savefig=_savefig, vmax=_vmax, vmin=_vmin, dpi=_dpi)


def plot_dataset(array, **kwargs):
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

    if _title != '':
        plt.suptitle(_title)
    if _savefig != '':
        fig.savefig(_savefig, bbox_inches='tight', dpi=_dpi)

    plt.show()
    plt.close()

if __name__ == '__main__':

    ### FIRST PART, ON HDF4 "RAW" FILES 

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

    # Just plot the data
    # plot_monthly('NDVI', 'NDVI AVG', vmax=10000, vmin=-13000, title="NDVI")
    # plot_monthly('NDVI', 'NDVI AVG', vmax=10000, vmin=-13000, title="NDVI", savefig=cwd + 'data_exploration/monthly_ndvi.png')
    # plot_monthly('EVI', 'EVI AVG', vmax=10000, vmin=-13000, title="EVI", savefig=cwd + 'data_exploration/monthly_evi.png')
    # plot_monthly('RED', 'B4 (Red) AVG', vmax=10000, vmin=-13000, title="RED", savefig=cwd + 'data_exploration/monthly_red.png')
    # plot_monthly('GREEN', 'B3 (Green) AVG', vmax=10000, vmin=-13000, title="GREEN", savefig=cwd + 'data_exploration/monthly_green.png')
    # plot_monthly('BLUE', 'B2 (Blue) AVG', vmax=10000, vmin=-13000, title="BLUE", savefig=cwd + 'data_exploration/monthly_blue.png')
    # plot_monthly('NIR', 'B5 (Nir) AVG', vmax=10000, vmin=-13000, title="NIR", savefig=cwd + 'data_exploration/monthly_nir.png')
    # plot_monthly('EVI2', 'EVI2 AVG', vmax=10000, vmin=-13000, title="EVI2", savefig=cwd + 'data_exploration/monthly_evi2.png')
    # plot_monthly('MIR', 'B7 (Mir) AVG', vmax=10000, vmin=-13000, title="MIR", savefig=cwd + 'data_exploration/monthly_mir.png')
    # plot_monthly('SWIR1', 'B6 (Swir1) AVG', vmax=10000, vmin=-13000, title="SWIR1", savefig=cwd + 'data_exploration/monthly_swir1.png')

    # plot_hdf_dataset(cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf', 'SOS', title='SOS')
    phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
    phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']
    # for var in phen:
    #     plot_hdf_dataset(cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf', var, title=var, savefig=cwd + f'data_exploration/pheno_{var}.png')

    # for var in phen2:
    #     plot_hdf_dataset(cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf', var, title=var, savefig=cwd + f'data_exploration/pheno_{var}.png')

    # # Fix SOS (ONLY FOR PLOTTING)
    # # 366 is still valid, assume all greater values are regular 365-based
    # sos_arr = rs.read_from_hdf(cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf', 'SOS')
    # sos_fixed = np.where(sos_arr > 366, sos_arr-365, sos_arr)
    # print(np.min(sos_fixed), np.max(sos_fixed))
    # plot_dataset(sos_fixed, title='SOS Fixed', savefig=cwd + f'data_exploration/pheno_SOS_fixed.png')

    # # Fix EOS (ONLY FOR PLOTTING)
    # eos_arr = rs.read_from_hdf(cwd + '03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf', 'EOS')
    # eos_fixed = np.where(eos_arr > 366, eos_arr-365, eos_arr)
    # print(np.min(eos_fixed), np.max(eos_fixed))
    # if  np.max(eos_fixed) > 366:
    #     eos_fixed = np.where(eos_fixed > 366, eos_fixed-365, eos_fixed)
    #     print(f'Adjusting again: {np.min(eos_fixed)}, {np.max(eos_fixed)}')
    # plot_dataset(eos_fixed, title='EOS Fixed', savefig=cwd + f'data_exploration/pheno_EOS_fixed.png')

    # ### Make monthly histograms
    # n_bins = 24
    # # plot_monthly_hist('NDVI', 'NDVI AVG', title="NDVI", bins=24)
    # plot_monthly_hist('NDVI', 'NDVI AVG',  title="NDVI", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_ndvi_{n_bins}.png')
    # plot_monthly_hist('NDVI', 'NDVI AVG',  title="NDVI", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_ndvi_{n_bins}.png')
    # plot_monthly_hist('EVI', 'EVI AVG',  title="EVI", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_evi_{n_bins}.png')
    # plot_monthly_hist('RED', 'B4 (Red) AVG',  title="RED", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_red_{n_bins}.png')
    # plot_monthly_hist('GREEN', 'B3 (Green) AVG',  title="GREEN", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_green_{n_bins}.png')
    # plot_monthly_hist('BLUE', 'B2 (Blue) AVG',  title="BLUE", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_blue_{n_bins}.png')
    # plot_monthly_hist('NIR', 'B5 (Nir) AVG',  title="NIR", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_nir_{n_bins}.png')
    # plot_monthly_hist('EVI2', 'EVI2 AVG',  title="EVI2", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_evi2_{n_bins}.png')
    # plot_monthly_hist('MIR', 'B7 (Mir) AVG',  title="MIR", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_mir_{n_bins}.png')
    # plot_monthly_hist('SWIR1', 'B6 (Swir1) AVG',  title="SWIR1", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_swir1_{n_bins}.png')

    ### SECOND PART: ON HDF5 FILES (COMPILED AND FILLED)
    basic_stats(fn_features, fn_labels, fn_feat_stats)

    # Read saved stats from CSV file
    # df = pd.read_csv(fn_feat_stats)

    # range_of_type('BAND', df)
    # range_of_type('VI', df)
    # range_of_type('PHEN', df)

    # Central tendency: mean, median, mode

    # Dispersion: range, quantiles, interquantile range, outliers

    # Dispersion: variance, standard deviation, mean abosolute deviation

    # Boxplot, quantile plot, q-q plot, barchart, histogram, scatterplot
    # plot_2hist_bands(fn_features, fn_hist_plot) # plots histograms with NaNs removed

    # Similarity, dissimilarity, proximity (for ordinal)

    # Standarizing?

    # DATA PREPROCESSING

    # # Missing data: ignore. Okay for classification
    # with h5py.File(cwd + 'data/IMG_Calakmul_Features_filled.h5', 'r') as f:
    #     # 1ST BAND IS JANUARY (BLUE)
    #     ds = f['r0c0'][:] # When dataset has 7 bands only
    # with h5py.File(cwd + 'IMG_Calakmul_Features.h5', 'r') as f2:
    #     # 1ST BAND IS MARCH (BLUE)
    #     ds1 = f2['r0c0'][:] # This has 56 bands
    
    # ds2 = np.where(ds1[:,:,1] >= 0, ds1[:,:,1], np.nan)

    # plot_hist(ds[:,:,1], title='JAN B2 B(Blue) AVG filled', savefig=cwd + 'data_exploration/hist JAN B2 (Blue) AVG Filled.png')
    # plot_2hist(ds2, ds[:,:,1], title='JAN B2 B(Blue) AVG - NaN removed (left) filled w/mean (right)', half=False,
    #            savefig=cwd + 'data_exploration/hist JAN B2 (Blue) AVG Missing vs Filled.png')

    print('Done ;-)')
