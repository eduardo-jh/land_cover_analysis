#!/usr/bin/env python
# coding: utf-8

""" Generates land cover plot using sorted values """

import os
import sys
sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs


fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_grp11_ancillary.tif"
fig_frequencies = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/sampling_grp11_3M/class_fequencies_sorted.png"

# Read the land cover raster and retrive the land cover classes
assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
land_cover, nodata, geotransform, spatial_reference = rs.open_raster(fn_landcover)
print(f'  Opening raster: {fn_landcover}')
print(f'    --NoData        : {nodata}')
print(f'    --Columns       : {land_cover.shape[1]}')
print(f'    --Rows          : {land_cover.shape[0]}')
print(f'    --Geotransform  : {geotransform}')
print(f'    --Spatial ref.  : {spatial_reference}')
print(f'    --Type          : {land_cover.dtype}')

land_cover = land_cover.filled(0)

# Create a list of land cover keys and its area covered percentage
landcover_frequencies = rs.land_cover_freq(fn_landcover, verbose=False, sort=True)
print(f'  --Land cover frequencies: {landcover_frequencies}')

classes = list(landcover_frequencies.keys())
freqs = [landcover_frequencies[x] for x in classes]  # pixel count
print(classes)
print(freqs)
percentages = (freqs/sum(freqs))*100
print(percentages)

# Plot land cover percentage horizontal bar
print('  --Plotting land cover percentages...')
rs.plot_land_cover_hbar(classes, percentages, fig_frequencies,
    title='INEGI Land Cover Classes in Yucatan Peninsula',
    xlabel='Percentage (based on pixel count)',
    ylabel='Land Cover (Grouped)',  # remove if not grouped
    xlims=(0,60))