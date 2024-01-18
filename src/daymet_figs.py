#!/usr/bin/env python
# coding: utf-8

# Create figures with tmin, tmax, and precipitation averages for the Yucatan Peninsula
# Raster for figures created on 2024-01-17

import sys
import os
import numpy as np

# sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib')
sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs

# Open a single file as test
f = "/VIP/engr-didan02s/DATA/EDUARDO/Daymet_YP2023/prcp/daymet_v4_prcp_annttl_na_1980.tif"
f_mask = "/VIP/engr-didan02s/DATA/EDUARDO/Daymet_YP2023/mask_yucatan.tif"
fn = "/VIP/engr-didan02s/DATA/EDUARDO/Daymet_YP2023/prcp.png"
ds, nd, geotransform, spatial_ref = rs.open_raster(f)
print(f'  Opening raster: {f}')
print(f'    NoData        : {nd}')
print(f'    Columns       : {ds.shape[1]}')
print(f'    Rows          : {ds.shape[0]}')
print(f'    Geotransform  : {geotransform}')
print(f'    Spatial ref.  : {spatial_ref}')
print(f'    Type          : {ds.dtype}')

maskds, nd_mask, geotransform_mask, spatial_ref_mask = rs.open_raster(f_mask)
print(f'  Opening raster: {f_mask}')
print(f'    NoData        : {nd_mask}')
print(f'    Columns       : {maskds.shape[1]}')
print(f'    Rows          : {maskds.shape[0]}')
print(f'    Geotransform  : {geotransform_mask}')
print(f'    Spatial ref.  : {spatial_ref_mask}')
print(f'    Type          : {maskds.dtype}')

# Dimensions of the entire Daymet raster (CAN, USA, MEX)
nrows, ncols = 8075, 7814  # known in advance
# (-4560750.0, 1000.0, 0.0, 4984500.0, 0.0, -1000.0)

# Find the Yucatan Peninsula
# Mask geotransform: (790250.0, 1000.0, 0.0, -2112500.0, 0.0, -1000.0
mask_rows, mask_cols = maskds.shape  # from mask raster
col_ini = 5351 # (790250+4560750)/1000
col_end = col_ini + mask_cols
row_ini = 7097 # (−2112500−4984500)/−1000
row_end = row_ini + mask_rows
ds1 = ds[row_ini:row_end,col_ini:col_end]

# Make mask only 0's and 1's
# mask = np.where(mask > 0, 1, mask)

# rs.plot_dataset(ds1, savefig=fn)
# rs.plot_dataset(mask, savefig=fn[:-4] + "_mask.png")

# Directories
# cwd = "D:/Downloads/Daymet"
cwd = '/VIP/engr-didan02s/DATA/EDUARDO/Daymet_YP2023/'
# vars = ["prcp", "tmax", "tmin"]
vars = ["tmax", "tmin"]
        
# Create an array to hold the sum and then average of all values
# nrows, ncols = 8075, 7814  # known in advance
sum_arr = np.zeros((mask_rows, mask_cols), dtype=np.float32)
avg_arr = np.zeros((mask_rows, mask_cols), dtype=np.float32)
counts = np.zeros((mask_rows, mask_cols), dtype=np.int16)

# List the files in each directory
for var in vars:
    subdir = os.path.join(cwd, var)
    only_files = rs.get_files(subdir, "tif")
    
    print(f"\nDIR: {subdir}\nFiles:")
    for i, f in enumerate(only_files):
        print(f"{i+1}/{len(only_files)}: {f}")
    
        # Open the files
        ds, nd, geotransform, spatial_ref = rs.open_raster(f)
        print(f'  Opening raster: {f}')
        print(f'    NoData        : {nd}')
        print(f'    Columns       : {ds.shape[1]}')
        print(f'    Rows          : {ds.shape[0]}')
        print(f'    Geotransform  : {geotransform}')
        print(f'    Spatial ref.  : {spatial_ref}')
        print(f'    Type          : {ds.dtype}')

        ds1 = ds[row_ini:row_end,col_ini:col_end]

        assert sum_arr.shape == ds1.shape, f"Shapes don't match: {sum_arr.shape} and {ds1.shape}"

        sum_arr += ds1

        # Non-empty pixels
        valid_counts = np.where(ds1.filled(0) > 0, 1, 0)
        print(f"Valid pixels: {np.sum(valid_counts)}, {np.unique(valid_counts, return_counts=True)}")
        counts += valid_counts
    
    # Calculate average
    counts = np.ma.masked_array(counts, mask=maskds<0)
    sum_arr = np.ma.masked_array(sum_arr, mask=maskds<0)
    avg_arr = sum_arr/counts


    # Save average raster
    fn = os.path.join(cwd, f"{var}_avg.tif")

    rs.plot_dataset(avg_arr, savefig=fn[:-4] + ".png")
    rs.plot_dataset(counts, savefig=fn[:-4] + "_sum.png")
    rs.plot_dataset(counts, savefig=fn[:-4] + "_counts.png")
    rs.create_raster(fn, avg_arr, spatial_ref, geotransform)