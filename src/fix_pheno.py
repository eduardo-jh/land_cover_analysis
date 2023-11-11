    #!/usr/bin/env python
# coding: utf-8

""" A scrip to fix phenology feature datasets. Fixes EOS, SOS, and LOS when larger than 365 days.
    A secondary separate script is used to not run and generate all the features again.

    WARNING! This will overwrite some dataset from the .h5 feature files.
"""
import sys
import os
import h5py
import numpy as np

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs

# =============================== 2013-2016 ===============================
cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/'
stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2013_2016/02_STATS/'
pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2013_2016/03_PHENO/'
fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_grp11_ancillary.tif"

# =============================== 2016-2019 ===============================
# cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/'
# stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2016_2019/02_STATS/'
# pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2016_2019/03_PHENO/'
# fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/data/usv250s6gw_grp11_ancillary.tif"

# =============================== 2019-2022 ===============================
# cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/'
# stats_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2019_2022/02_STATS/'
# pheno_dir = '/VIP/engr-didan02s/DATA/EDUARDO/LANDSAT_C2_YUCATAN/STATS_ROI2/2019_2022/03_PHENO/'
# fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/data/usv250s7gw_grp11_ancillary.tif"

# ============================ FOR ALL PERIODS =============================
# The NoData mask is the ROI (The Yucatan Peninsula Aquifer)
fn_nodata = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/data/YucPenAquifer_mask.tif'
fn_tiles = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/parameters/tiles'

# ===== FIX PHENOLOGY =====

# Fix phenology values larger than a year
# var_phen1 = ['SOS', 'EOS', 'LOS']
# var_phen2 = ['SOS2', 'EOS2', 'LOS2']
var_phen = ['SOS', 'EOS', 'LOS', 'SOS2', 'EOS2', 'LOS2', 'NOS']
tiles = ["h19v25", "h20v24", "h20v25", "h20v26", "h21v23", "h21v24", "h21v25", "h21v26", "h22v22", "h22v23", "h22v24", "h22v25", "h22v26", "h23v22", "h23v23", "h23v24", "h23v25"]
# tiles = ["h19v25"]

for tile in tiles:
    print(f"\n ===== Tile: {tile} =====")
    for i, var in enumerate(var_phen):
        feature = f"PHEN {var}"
        print(f"\nChecking phenology feature: {feature}")

        fn_tile_features = os.path.join(cwd, "features", tile, f'features_{tile}.h5')
        print(f"Processing: {fn_tile_features}")
        fn_tile_features_fixed = os.path.join(cwd, "features", tile, f'features_{tile}_fixed.h5')

        with h5py.File(fn_tile_features, 'r+') as h5_tile_features:
            # Get the data from the HDF5 files
            pheno_array = h5_tile_features[feature][...]  # ellipsis indexing
            print(f"Before: {np.min(pheno_array)}-{np.max(pheno_array)}")
            if var == 'NOS':
                pheno_array_fixed = np.where(pheno_array < 0, -1, pheno_array)
            else:
                pheno_array_fixed = rs.fix_annual_phenology(pheno_array)
            print(f"After: {np.min(pheno_array_fixed)}-{np.max(pheno_array_fixed)} {type(pheno_array_fixed)}")
            h5_tile_features[feature][...] = pheno_array_fixed
            # pheno_array[...] = pheno_array_fixed
            # pheno_array[...] = rs.fix_annual_phenology(pheno_array)
        
        # if i==0:
        #     print(f"Creating: {fn_tile_features_fixed}")
        #     h5_tile_features_fixed = h5py.File(fn_tile_features_fixed, 'w')
        #     print(f"Writing: {feature} into {fn_tile_features_fixed}")
        #     h5_tile_features_fixed.create_dataset(feature, pheno_array_fixed.shape, data=pheno_array_fixed)
        # else:
        #     print(f"Writing: {feature} into {fn_tile_features_fixed}")
        #     h5_tile_features_fixed.create_dataset(feature, pheno_array_fixed.shape, data=pheno_array_fixed)


        # ==== Now process seasonal features ====

        fn_tile_features_season = os.path.join(cwd, "features", tile, f'features_season_{tile}.h5')
        print(f"\nProcessing: {fn_tile_features_season}")
        fn_tile_features_season_fixed = os.path.join(cwd, "features", tile, f'features_season_{tile}_fixed.h5')

        with h5py.File(fn_tile_features_season, 'r+') as h5_tile_features_season:
            # Get the data from the HDF5 files
            pheno_array_season = h5_tile_features_season[feature][...]  # ellipsis indexing
            print(f"Before: {np.min(pheno_array_season)}-{np.max(pheno_array_season)}")
            if var == 'NOS':
                pheno_array_fixed = np.where(pheno_array_season < 0, -1, pheno_array_season)
            else:
                pheno_array_season_fixed = rs.fix_annual_phenology(pheno_array_season)
            print(f"After: {np.min(pheno_array_season_fixed)}-{np.max(pheno_array_season_fixed)} {type(pheno_array_season_fixed)}")
            h5_tile_features_season[feature][...] = pheno_array_season_fixed
            # pheno_array_season[...] = pheno_array_season_fixed
            # pheno_array[...] = rs.fix_annual_phenology(pheno_array)
        
        # if i==0:
        #     print(f"Creating: {fn_tile_features_season_fixed}")
        #     h5_tile_features_season_fixed = h5py.File(fn_tile_features_season_fixed, 'w')
        #     print(f"Writing: {feature} into {fn_tile_features_season_fixed}")
        #     h5_tile_features_season_fixed.create_dataset(feature, pheno_array_season_fixed.shape, data=pheno_array_season_fixed)
        # else:
        #     print(f"Writing: {feature} into {fn_tile_features_season_fixed}")
        #     h5_tile_features_season_fixed.create_dataset(feature, pheno_array_season_fixed.shape, data=pheno_array_season_fixed)
