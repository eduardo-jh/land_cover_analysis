#!/usr/bin/env python
# coding: utf-8

""" Data exploration

Eduardo Jimenez <eduardojh@email.arizona.edu>

NOTE: run under 'rstf' conda environment (python 3.8.13, keras 2.9.0)

 VI values are scaled by 10000; GUR and GDR are scaled by 100, then by 10000, thus real values are really small.
"""

import sys
import numpy as np
import pandas as pd

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

if __name__ == '__main__':

    ### FIRST PART, ON HDF4 "RAW" FILES 

    # Paths and file names for the current ROI
    
    fn_phenology = cwd + 'data/landsat/C2/03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf'  # Phenology files
    fn_phenology2 = cwd + 'data/landsat/C2/03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf'

    # fn_features = cwd + 'features/Calakmul_Features.h5'
    fn_features = cwd + 'features/season/Calakmul_Features_season.h5'
    fn_labels = cwd + 'features/Calakmul_Labels.h5'
    # fn_feat_stats = cwd + 'data_exploration/feature_stats_summary.csv'
    fn_feat_stats = cwd + 'data_exploration/feature_stats_summary_season.csv'
    fn_hist_plot = cwd + 'data_exploration/hist'
    fn_ranges = cwd + 'parameters/valid_ranges'

    # NOT USED
    # fn_landcover = cwd + 'data/inegi_2018/usv250s7cw_ROI1_LC_KEY.tif'        # Land cover raster
    # fn_landcover = cwd + 'data/inegi_2018/land_cover_ROI1.tif'      # Use land cover groups w/ancillary
    # fn_test_mask = cwd + 'sampling/ROI1_testing_mask.tif'
    # fn_test_labels = cwd + 'sampling/ROI1_testing_labels.tif'
    # # fn_train_feat = cwd + 'features/Calakmul_Training_Features.h5'
    # fn_test_feat = cwd + 'features/Calakmul_Testing_Features.h5'

    # Just plot the data
    # rs.plot_monthly('NDVI', 'NDVI AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=-10000, title="NDVI", cmap='Greens')
    # rs.plot_monthly('NDVI', 'NDVI AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=1000, title="NDVI", cmap='viridis', savefig=cwd + 'data_exploration/monthly_ndvi.png')
    # rs.plot_monthly('EVI', 'EVI AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=0, title="EVI", cmap='viridis', savefig=cwd + 'data_exploration/monthly_evi.png')
    # rs.plot_monthly('RED', 'B4 (Red) AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=0, title="RED", cmap='Reds_r', savefig=cwd + 'data_exploration/monthly_red.png')
    # rs.plot_monthly('GREEN', 'B3 (Green) AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=0, title="GREEN", cmap='Greens_r', savefig=cwd + 'data_exploration/monthly_green.png')
    # rs.plot_monthly('BLUE', 'B2 (Blue) AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=0, title="BLUE", cmap='Blues_r', savefig=cwd + 'data_exploration/monthly_blue.png')
    # rs.plot_monthly('NIR', 'B5 (Nir) AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=0, title="NIR", cmap='gist_earth', savefig=cwd + 'data_exploration/monthly_nir.png')
    # rs.plot_monthly('EVI2', 'EVI2 AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=0, title="EVI2", cmap='viridis', savefig=cwd + 'data_exploration/monthly_evi2.png')
    # rs.plot_monthly('MIR', 'B7 (Mir) AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=0, title="MIR", cmap='gist_earth', savefig=cwd + 'data_exploration/monthly_mir.png')
    # rs.plot_monthly('SWIR1', 'B6 (Swir1) AVG', cwd+'data/landsat/C2/02_STATS/', vmax=10000, vmin=0, title="SWIR1", cmap='gist_earth', savefig=cwd + 'data_exploration/monthly_swir1.png')

    # Plot HALFMONTH data
    # rs.plot_monthly('NDVI', 'NDVI AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', vmax=10000, vmin=1000, title="NDVI", cmap='viridis', savefig=cwd + 'data_exploration/HALFMONTH/monthly_ndvi.png')
    # rs.plot_monthly('EVI', 'EVI AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', vmax=10000, vmin=0, title="EVI", cmap='viridis', savefig=cwd + 'data_exploration/HALFMONTH/monthly_evi.png')
    # rs.plot_monthly('RED', 'B4 (Red) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', vmax=10000, vmin=0, title="RED", cmap='Reds_r', savefig=cwd + 'data_exploration/HALFMONTH/monthly_red.png')
    # rs.plot_monthly('GREEN', 'B3 (Green) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', vmax=10000, vmin=0, title="GREEN", cmap='Greens_r', savefig=cwd + 'data_exploration/HALFMONTH/monthly_green.png')
    # rs.plot_monthly('BLUE', 'B2 (Blue) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', vmax=10000, vmin=0, title="BLUE", cmap='Blues_r', savefig=cwd + 'data_exploration/HALFMONTH/monthly_blue.png')
    # rs.plot_monthly('NIR', 'B5 (Nir) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', vmax=10000, vmin=0, title="NIR", cmap='gist_earth', savefig=cwd + 'data_exploration/HALFMONTH/monthly_nir.png')
    # rs.plot_monthly('EVI2', 'EVI2 AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', vmax=10000, vmin=0, title="EVI2", cmap='viridis', savefig=cwd + 'data_exploration/HALFMONTH/monthly_evi2.png')
    # rs.plot_monthly('MIR', 'B7 (Mir) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', vmax=10000, vmin=0, title="MIR", cmap='gist_earth', savefig=cwd + 'data_exploration/HALFMONTH/monthly_mir.png')
    # rs.plot_monthly('SWIR1', 'B6 (Swir1) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', vmax=10000, vmin=0, title="SWIR1", cmap='gist_earth', savefig=cwd + 'data_exploration/HALFMONTH/monthly_swir1.png')



    # Test a phenology variable
    # rs.plot_hdf_dataset(cwd + 'data/landsat/C2/03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf', 'SOS', title='SOS')

    # # Plot all phenology variables
    # phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
    # phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']
    # for var in phen:
    #     rs.plot_hdf_dataset(cwd + 'data/landsat/C2/03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf', var, title=var, savefig=cwd + f'data_exploration/phenology/pheno_{var}.png')
    # for var in phen2:
    #     rs.plot_hdf_dataset(cwd + 'data/landsat/C2/03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf', var, title=var, savefig=cwd + f'data_exploration/phenology/pheno_{var}.png')
    
    # # Fix SOS (ONLY FOR PLOTTING)
    # # 366 is still valid, assume all greater values are regular 365-based
    # sos_arr = rs.read_from_hdf(cwd + 'data/landsat/C2/03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf', 'SOS')
    # sos_fixed = np.where(sos_arr > 366, sos_arr-365, sos_arr)
    # print(np.min(sos_fixed), np.max(sos_fixed))
    # rs.plot_dataset(sos_fixed, title='SOS Fixed', savefig=cwd + f'data_exploration/phenology/pheno_SOS.png')

    # # Fix EOS (ONLY FOR PLOTTING)
    # eos_arr = rs.read_from_hdf(cwd + 'data/landsat/C2/03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf', 'EOS')
    # eos_fixed = np.where(eos_arr > 366, eos_arr-365, eos_arr)
    # print(np.min(eos_fixed), np.max(eos_fixed))
    # if  np.max(eos_fixed) > 366:
    #     eos_fixed = np.where(eos_fixed > 366, eos_fixed-365, eos_fixed)
    #     print(f'Adjusting again: {np.min(eos_fixed)}, {np.max(eos_fixed)}')
    # rs.plot_dataset(eos_fixed, title='EOS Fixed', savefig=cwd + f'data_exploration/phenology/pheno_EOS.png')


    # ### Make monthly histograms for HALFMONTH
    # n_bins = 24
    # # rs.plot_monthly_hist('NDVI', 'NDVI AVG', cwd, title="NDVI", bins=24)
    # rs.plot_monthly_hist('NDVI', 'NDVI AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/',  title="NDVI", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_ndvi_{n_bins}.png')
    # rs.plot_monthly_hist('NDVI', 'NDVI AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/',  title="NDVI", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_ndvi_{n_bins}.png')
    # rs.plot_monthly_hist('EVI', 'EVI AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/',  title="EVI", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_evi_{n_bins}.png')
    # rs.plot_monthly_hist('RED', 'B4 (Red) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/',  title="RED", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_red_{n_bins}.png')
    # rs.plot_monthly_hist('GREEN', 'B3 (Green) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', title="GREEN", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_green_{n_bins}.png')
    # rs.plot_monthly_hist('BLUE', 'B2 (Blue) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/',  title="BLUE", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_blue_{n_bins}.png')
    # rs.plot_monthly_hist('NIR', 'B5 (Nir) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/',  title="NIR", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_nir_{n_bins}.png')
    # rs.plot_monthly_hist('EVI2', 'EVI2 AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/',  title="EVI2", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_evi2_{n_bins}.png')
    # rs.plot_monthly_hist('MIR', 'B7 (Mir) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/', title="MIR", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_mir_{n_bins}.png')
    # rs.plot_monthly_hist('SWIR1', 'B6 (Swir1) AVG', cwd+'data/landsat/C2/02_STATS/HALFMONTH/',  title="SWIR1", bins=n_bins, savefig=cwd + f'data_exploration/HALFMONTH/hist_monthly_swir1_{n_bins}.png')

    # ### Make monthly histograms
    # n_bins = 24
    # # rs.plot_monthly_hist('NDVI', 'NDVI AVG', cwd+'data/landsat/C2/02_STATS/', title="NDVI", bins=24)
    # rs.plot_monthly_hist('NDVI', 'NDVI AVG', cwd+'data/landsat/C2/02_STATS/',  title="NDVI", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_ndvi_{n_bins}.png')
    # rs.plot_monthly_hist('NDVI', 'NDVI AVG', cwd+'data/landsat/C2/02_STATS/',  title="NDVI", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_ndvi_{n_bins}.png')
    # rs.plot_monthly_hist('EVI', 'EVI AVG', cwd+'data/landsat/C2/02_STATS/',  title="EVI", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_evi_{n_bins}.png')
    # rs.plot_monthly_hist('RED', 'B4 (Red) AVG', cwd+'data/landsat/C2/02_STATS/',  title="RED", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_red_{n_bins}.png')
    # rs.plot_monthly_hist('GREEN', 'B3 (Green) AVG', cwd+'data/landsat/C2/02_STATS/', title="GREEN", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_green_{n_bins}.png')
    # rs.plot_monthly_hist('BLUE', 'B2 (Blue) AVG', cwd+'data/landsat/C2/02_STATS/',  title="BLUE", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_blue_{n_bins}.png')
    # rs.plot_monthly_hist('NIR', 'B5 (Nir) AVG', cwd+'data/landsat/C2/02_STATS/',  title="NIR", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_nir_{n_bins}.png')
    # rs.plot_monthly_hist('EVI2', 'EVI2 AVG', cwd+'data/landsat/C2/02_STATS/',  title="EVI2", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_evi2_{n_bins}.png')
    # rs.plot_monthly_hist('MIR', 'B7 (Mir) AVG', cwd+'data/landsat/C2/02_STATS/', title="MIR", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_mir_{n_bins}.png')
    # rs.plot_monthly_hist('SWIR1', 'B6 (Swir1) AVG', cwd+'data/landsat/C2/02_STATS/',  title="SWIR1", bins=n_bins, savefig=cwd + f'data_exploration/hist_monthly_swir1_{n_bins}.png')


    ### SECOND PART: ON HDF5 FILES (COMPILED AND FILLED)
    # rs.basic_stats(fn_features, fn_labels, fn_feat_stats)

    # Read saved stats from CSV file
    # df = pd.read_csv(fn_feat_stats)

    # rs.range_of_type('BAND', df)
    # rs.range_of_type('VI', df)
    # rs.range_of_type('PHEN', df)

    # Central tendency: mean, median, mode

    # Dispersion: range, quantiles, interquantile range, outliers

    # Dispersion: variance, standard deviation, mean abosolute deviation

    # Boxplot, quantile plot, q-q plot, barchart, histogram, scatterplot
    # plot_2hist_bands(fn_features, fn_hist_plot) # plots histograms with NaNs removed

    # Similarity, dissimilarity, proximity (for ordinal)

    # Standarizing? Already done previously.

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
