#!/usr/bin/env python
# coding: utf-8

""" Data exploration

Eduardo Jimenez <eduardojh@email.arizona.edu>

NOTE: run under 'rstf' conda environment (python 3.8.13, keras 2.9.0)

 VI values are scaled by 10000; GUR and GDR are scaled by 100, then by 10000, thus real values are really small.
"""

import sys
import csv
import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
    elif system == 'Linux' and os.path.isdir('/VIP/engr-didan02s/DATA/EDUARDO/ML/'):
        # On Alma Linux Server
        sys.path.insert(0, '/home/eduardojh/Documents/land_cover_analysis/lib/')
        cwd = '/VIP/engr-didan02s/DATA/EDUARDO/ML/'
    else:
        print('  System not yet configured!')

import rsmodule as rs

if __name__ == '__main__':

    ### FIRST PART, ON HDF4 "RAW" FILES 

    # Paths and file names for the current ROI
    
    fn_phenology = cwd + 'data/landsat/C2/03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf'  # Phenology files
    fn_phenology2 = cwd + 'data/landsat/C2/03_PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf'

    # fn_features = cwd + 'features/Calakmul_Features.h5'
    # fn_feat_indices = cwd + 'features/feature_indices.csv'
    # fn_feat_stats = cwd + 'data_exploration/feature_stats_summary.csv'
    
    fn_features = cwd + 'features/season/Calakmul_Features_season.h5'
    fn_feat_indices = cwd + 'features/season/feature_indices_season.csv'
    fn_feat_stats = cwd + 'data_exploration/feature_stats_summary_season.csv'
    fn_labels = cwd + 'features/Calakmul_Labels.h5'
    fn_hist_plot = cwd + 'data_exploration/hist'
    fn_ranges = cwd + 'parameters/valid_ranges'
    fn_correlation = cwd + 'data_exploration/feat_anal/correlation.csv'

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

    # Read features

    feat_indices = []
    feat_names = []
    assert os.path.isfile(fn_feat_indices) is True, f"ERROR: File not found! {fn_feat_indices}"
    with open(fn_feat_indices, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            if len(row) != 2:
                continue
            feat_indices.append(int(row[0]))
            feat_names.append(row[1])
            # print(f"{int(row[0])}, {row[1]}")

    # feats = [('FAL NDVI AVG', 'FAL EVI2 AVG'),
    #          ('FAL NDVI AVG', 'FAL EVI AVG'),
            #  ('FAL NDVI AVG', 'FAL GREEN AVG'),
            #  ('FAL NDVI AVG', 'FAL BLUE AVG'),
            #  ('FAL NDVI AVG', 'FAL RED AVG'),
            #  ('FAL NDVI AVG', 'FAL MIR AVG'),
            #  ('FAL NDVI AVG', 'FAL NIR AVG'),
            #  ('FAL NDVI AVG', 'FAL SWIR1 AVG'),
            #  ('FAL EVI AVG', 'FAL EVI2 AVG'),
            #  ('FAL EVI AVG', 'FAL NDVI AVG'),
            #  ('FAL EVI AVG', 'FAL GREEN AVG'),
            #  ('FAL EVI AVG', 'FAL BLUE AVG'),
            #  ('FAL EVI AVG', 'FAL RED AVG'),
            #  ('FAL EVI AVG', 'FAL MIR AVG'),
            #  ('FAL EVI AVG', 'FAL NIR AVG'),
            #  ('FAL EVI AVG', 'FAL SWIR1 AVG'),
            #  ('SPR NDVI AVG', 'SPR EVI2 AVG'),
            #  ('SPR NDVI AVG', 'SPR EVI AVG'),
            #  ('SPR NDVI AVG', 'SPR GREEN AVG'),
            #  ('SPR NDVI AVG', 'SPR BLUE AVG'),
            #  ('SPR NDVI AVG', 'SPR RED AVG'),
            #  ('SPR NDVI AVG', 'SPR MIR AVG'),
            #  ('SPR NDVI AVG', 'SPR NIR AVG'),
            #  ('SPR NDVI AVG', 'SPR SWIR1 AVG'),
            #  ('SPR EVI AVG', 'SPR EVI2 AVG'),
            #  ('SPR EVI AVG', 'SPR NDVI AVG'),
            #  ('SPR EVI AVG', 'SPR GREEN AVG'),
            #  ('SPR EVI AVG', 'SPR BLUE AVG'),
            #  ('SPR EVI AVG', 'SPR RED AVG'),
            #  ('SPR EVI AVG', 'SPR MIR AVG'),
            #  ('SPR EVI AVG', 'SPR NIR AVG'),
            #  ('SPR EVI AVG', 'SPR SWIR1 AVG'),
            #  ('SUM NDVI AVG', 'SUM EVI2 AVG'),
            #  ('SUM NDVI AVG', 'SUM EVI AVG'),
            #  ('SUM NDVI AVG', 'SUM GREEN AVG'),
            #  ('SUM NDVI AVG', 'SUM BLUE AVG'),
            #  ('SUM NDVI AVG', 'SUM RED AVG'),
            #  ('SUM NDVI AVG', 'SUM MIR AVG'),
            #  ('SUM NDVI AVG', 'SUM NIR AVG'),
            #  ('SUM NDVI AVG', 'SUM SWIR1 AVG'),
            #  ('SUM EVI AVG', 'SUM EVI2 AVG'),
            #  ('SUM EVI AVG', 'SUM NDVI AVG'),
            #  ('SUM EVI AVG', 'SUM GREEN AVG'),
            #  ('SUM EVI AVG', 'SUM BLUE AVG'),
            #  ('SUM EVI AVG', 'SUM RED AVG'),
            #  ('SUM EVI AVG', 'SUM MIR AVG'),
            #  ('SUM EVI AVG', 'SUM NIR AVG'),
            #  ('SUM EVI AVG', 'SUM SWIR1 AVG'),
            #  ('WIN NDVI AVG', 'WIN EVI2 AVG'),
            #  ('WIN NDVI AVG', 'WIN EVI AVG'),
            #  ('WIN NDVI AVG', 'WIN GREEN AVG'),
            #  ('WIN NDVI AVG', 'WIN BLUE AVG'),
            #  ('WIN NDVI AVG', 'WIN RED AVG'),
            #  ('WIN NDVI AVG', 'WIN MIR AVG'),
            #  ('WIN NDVI AVG', 'WIN NIR AVG'),
            #  ('WIN NDVI AVG', 'WIN SWIR1 AVG'),
            #  ('WIN EVI AVG', 'WIN EVI2 AVG'),
            #  ('WIN EVI AVG', 'WIN NDVI AVG'),
            #  ('WIN EVI AVG', 'WIN GREEN AVG'),
            #  ('WIN EVI AVG', 'WIN BLUE AVG'),
            #  ('WIN EVI AVG', 'WIN RED AVG'),
            #  ('WIN EVI AVG', 'WIN MIR AVG'),
            #  ('WIN EVI AVG', 'WIN NIR AVG'),
            #  ('WIN EVI AVG', 'WIN SWIR1 AVG')]

    # Combination of all the variables
    feats = []
    for f1 in feat_names:
        for f2 in feat_names:
            if f1 != f2:
                feats.append((f1,f2))
    max_feats = len(feats)

    list_vars1 = []
    list_vars2 = []
    list_corrs = []
    for i, feat_comp in enumerate(feats):
        with h5py.File(fn_features, 'r') as f:
            var1 = feat_comp[0]
            var2 = feat_comp[1]
            ds1 = f[var1][:]
            ds2 = f[var2][:]

            # Replace negatives with NaNs
            ds1 = np.where(ds1 > -10000, ds1, np.nan)
            ds2 = np.where(ds2 > -10000, ds2, np.nan)

            # rs.plot_2dataset(ds1, ds2,
            #                 titles=(var1, var2),
            #                 savefig=cwd + f'data_exploration/feat_anal/plot_{var1}_{var2}.png')

            df = pd.DataFrame({'DS1': ds1.flatten(), 'DS2': ds2.flatten()})
            df = df.dropna(axis=0, how='any')  # drop NaNs
            # print(df.head())

            cor = np.corrcoef(df['DS2'], df['DS1'])
            list_vars1.append(var1)
            list_vars2.append(var2)
            list_corrs.append(cor[0,1])

            # Fit a linear model
            X = sm.add_constant(df['DS1'])
            y = df['DS2']
            model = sm.OLS(y, X)
            results = model.fit()
            # print(results.summary())
            # print(results.params)

            _min = int(np.floor(df['DS1'].min() if df['DS1'].min() < df['DS2'].min() else df['DS2'].min()))
            _max = int(np.floor(df['DS1'].max() if df['DS1'].max() > df['DS2'].max() else df['DS2'].max()))
            print(f"{feat_comp[0]:>15} -- {feat_comp[1]:>15}: {cor[0,1]:>6.4f} ({_min:>6} -- {_max:>6}) {i:>4}/{max_feats}")
            
            # # Calculate xlims and ylims
            # _xmin = int(np.floor(df['DS1'].min()))
            # _xmax = int(np.floor(df['DS1'].max()))
            # _ymin = int(np.floor(df['DS2'].min()))
            # _ymax = int(np.floor(df['DS2'].max()))

            # Create a heatmap to show point density and linear plots
            rs.plot_corr2(df['DS1'], df['DS2'],
                          bins=500,
                          savefig=cwd + f'data_exploration/feat_anal/corr_{var1}_{var2}.png',
                          title=f'Correlation between {var1} and {var2}: {cor[0,1]:>0.2f}',
                          xlabel=var1, ylabel=var2,
                          lims=(_min,_max),
                          model=results,
                          log=True)
            del ds1
            del ds2
    correlations = pd.DataFrame({'Var1': list_vars1, 'Var2': list_vars2, 'PearsonCorrelation': list_corrs})
    correlations.to_csv(fn_correlation)

    # Select the variables with lower correlation
    threshold = 0.6
    correlations = pd.read_csv(fn_correlation)
    sel_vars = correlations[correlations['PearsonCorrelation'] <= -threshold | correlations['PearsonCorrelation'] >= threshold]
    unique_vars = pd.Series({c: sel_vars[c].unique() for c in sel_vars})
    print('Done ;-)')
