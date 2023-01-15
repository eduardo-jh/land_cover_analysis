#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from rsmodule import open_raster
from preprocessing import get_extent

# Comparison bands between Landsat Collection 1 Level 1 and Collection 2 Level 2

# NDVI rasters, same date
ndvi1 = '/VIP/anga/DATA/USGS/LANDSAT/EXTRACTED/OLI/036035/LC08_L1TP_036035_20210516_20210525_01_T1_sr_ndvi.tif'
red1 =  '/VIP/anga/DATA/USGS/LANDSAT/EXTRACTED/OLI/036035/LC08_L1TP_036035_20210516_20210525_01_T1_sr_band4.tif'
nir1 =  '/VIP/anga/DATA/USGS/LANDSAT/EXTRACTED/OLI/036035/LC08_L1TP_036035_20210516_20210525_01_T1_sr_band5.tif'
ndvi2 = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/036035/LC08_L2SP_036035_20210516_20210525_02_T1_SR_NDVI.tif'
red2  = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/036035/LC08_L2SP_036035_20210516_20210525_02_T1_SR_B4.tif'
nir2  = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/036035/LC08_L2SP_036035_20210516_20210525_02_T1_SR_B5.tif'

# Open each raster and compare extents and dimensions
ndvi_arr1, _, _, ndvi_gt1 = open_raster(ndvi1)
ndvi_arr2, _, _, ndvi_gt2 = open_raster(ndvi2)
red_arr1, _, _, red_gt1 = open_raster(red1)
red_arr2, _, _, red_gt2 = open_raster(red2)
nir_arr1, _, _, nir_gt1 = open_raster(nir1)
nir_arr2, _, _, nir_gt2 = open_raster(nir2)

# Calculate extents
ndvi_ex1 = get_extent(ndvi_arr1.shape, ndvi_gt1)
ndvi_ex2 = get_extent(ndvi_arr2.shape, ndvi_gt2)
red_ex1 = get_extent(red_arr1.shape, red_gt1)
red_ex2 = get_extent(red_arr2.shape, red_gt2)
nir_ex1 = get_extent(nir_arr1.shape, nir_gt1)
nir_ex2 = get_extent(nir_arr2.shape, nir_gt2)

print(f'NDVI Dimensions: {ndvi_arr1.shape} {ndvi_arr2.shape}')
print(f'NDVI Extent: {ndvi_ex1} {ndvi_ex2}')
print(f'Red Dimensions: {red_arr1.shape} {red_arr2.shape}')
print(f'Red Extent: {red_ex1} {red_ex2}')
print(f'NIR Dimensions: {nir_arr1.shape} {nir_arr2.shape}')
print(f'NIR Extent: {nir_ex1} {nir_ex2}')

assert ndvi_arr1.shape == ndvi_arr2.shape, "NDVI dimensions mismatch!"
assert ndvi_ex1 == ndvi_ex2, "NDVI extent mismatch!"
assert red_arr1.shape == red_arr2.shape, "Red dimensions mismatch!"
assert red_ex1 == red_ex2, "Red extent mismatch!"
assert nir_arr1.shape == nir_arr2.shape, "NIR dimensions mismatch!"
assert nir_ex1 == nir_ex2, "NIR extent mismatch!"

# Calculate the difference between L1TP and L2SP rasters
diff_ndvi = ndvi_arr1 - ndvi_arr2
diff_red = red_arr1 - red_arr2
diff_nir = nir_arr1 - nir_arr2

# Compare some values
col = 3000
row = 3000
print(f'Compare some values')
print(f'NDVI1: {ndvi_arr1[row,col]} NDVI2: {ndvi_arr2[row,col]} Diff: {diff_ndvi[row,col]}\n')

print(f'NDVI 1 max={ndvi_arr1.max()} min={ndvi_arr1.min()}')
print(f'NDVI 2 max={ndvi_arr2.max()} min={ndvi_arr2.min()}')
print(f'Difference NDVI max={diff_ndvi.max()} min={diff_ndvi.min()} avg={diff_ndvi.mean():.3f} std={diff_ndvi.std():.3f}')

print(f'Red 1 max={red_arr1.max()} min={red_arr1.min()}')
print(f'Red 2 max={red_arr2.max()} min={red_arr2.min()}')
print(f'Difference Red max={diff_red.max()} min={diff_red.min()} avg={diff_red.mean():.3f} std={diff_red.std():.3f}')

print(f'NIR 1 max={nir_arr1.max()} min={nir_arr1.min()}')
print(f'NIR 2 max={nir_arr2.max()} min={nir_arr2.min()}')
print(f'Difference NIR max={diff_nir.max()} min={diff_nir.min()} avg={diff_nir.mean():.3f} std={diff_nir.std():.3f}')

_dpi = 300

# Single plots of NDVI
print(f'Plotting single figures')
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
plt.imshow(ndvi_arr1, cmap='viridis')
plt.title('NDVI [L1TP]')
plt.savefig(f'/home/eduardojh/Documents/results/ndvi_1.png', bbox_inches='tight', dpi=_dpi)
plt.close()

fig = plt.figure(figsize=(12, 12), constrained_layout=True)
plt.imshow(ndvi_arr2, cmap='viridis')
plt.title('NDVI [L2SP]')
plt.savefig(f'/home/eduardojh/Documents/results/ndvi_2.png', bbox_inches='tight', dpi=_dpi)
plt.close()

fig = plt.figure(figsize=(12, 12), constrained_layout=True)
plt.imshow(diff_ndvi, cmap='viridis')
plt.title('NDVI difference')
plt.savefig(f'/home/eduardojh/Documents/results/ndvi_single_diff.png', bbox_inches='tight', dpi=_dpi)
plt.close()

# Plot NDVI
print(f'Plotting difference figures')
fig, axs = plt.subplots(1, 3, figsize=(24,8), constrained_layout=True)
axs[0].imshow(ndvi_arr1, cmap='viridis')
axs[0].set_title("NDVI [L1TP]")
im1 = axs[1].imshow(ndvi_arr2, cmap='viridis')
axs[1].set_title("NDVI [L2SP]")
im2 = axs[2].imshow(diff_ndvi, cmap='viridis')
axs[2].set_title("Difference")
fig.colorbar(im1, ax=axs[1])
fig.colorbar(im2, ax=axs[2])
plt.savefig(f'/home/eduardojh/Documents/results/ndvi_difference.png', bbox_inches='tight', dpi=_dpi)
plt.close()

# Plot Red
fig, axs = plt.subplots(1, 3, figsize=(24,8), constrained_layout=True)
axs[0].imshow(red_arr1, cmap='viridis')
axs[0].set_title("Red [L1TP]")
im1 = axs[1].imshow(red_arr2, cmap='viridis')
axs[1].set_title("Red [L2SP]")
im2 = axs[2].imshow(diff_red, cmap='viridis')
axs[2].set_title("Difference")
fig.colorbar(im1, ax=axs[1])
fig.colorbar(im2, ax=axs[2])
plt.savefig(f'/home/eduardojh/Documents/results/red_difference.png', bbox_inches='tight', dpi=_dpi)
plt.close()

# Plot NIR
fig, axs = plt.subplots(1, 3, figsize=(24,8), constrained_layout=True)
axs[0].imshow(nir_arr1, cmap='viridis')
axs[0].set_title("NIR [L1TP]")
im1 = axs[1].imshow(nir_arr2, cmap='viridis')
axs[1].set_title("NIR [L2SP]")
im2 = axs[2].imshow(diff_nir, cmap='viridis')
axs[2].set_title("Difference")
fig.colorbar(im1, ax=axs[1])
fig.colorbar(im2, ax=axs[2])
plt.savefig(f'/home/eduardojh/Documents/results/nir_difference.png', bbox_inches='tight', dpi=_dpi)
plt.close()

# Plot histograms
print('Plotting histograms')

fig = plt.figure(figsize=(12, 12))
my_bins = np.linspace(-10000,10000,17)  # Predefined bins for NDVI
# Create the histogram using numpy
count, bins = np.histogram(diff_ndvi.compressed(), bins = my_bins)
#print(f'From numpy: {bins} {count}')
# Use the 'weights' option for pre-binned values
density, bins, _  = plt.hist(bins[:-1], bins, weights=count)
#print(f'From pyplot: {density}, {bins}, {_}')
# Plot the value at the top of the bars
for x,y,num in zip(bins, density, count):
    if num != 0:
        plt.text(x, y+1, f'{num:,.0f}') # x,y,str
plt.title('Difference NDVI')
plt.savefig(f'/home/eduardojh/Documents/results/ndvi_hist.png', bbox_inches='tight')
plt.close()

# Generate an histogram for RED difference
fig = plt.figure(figsize=(12, 12))
count, bins = np.histogram(diff_red.compressed(), bins = 16)
density, bins, _  = plt.hist(bins[:-1], bins, weights=count)
for x,y,num in zip(bins, density, count):
    if num != 0:
        plt.text(x, y+1, f'{num:,.0f}') # x,y,str
plt.title('Difference RED')
plt.savefig(f'/home/eduardojh/Documents/results/red_hist.png', bbox_inches='tight')
plt.close()

# Generate an histogram for NIR difference
fig = plt.figure(figsize=(12, 12))
count, bins = np.histogram(diff_nir.compressed(), bins = 16)
density, bins, _  = plt.hist(bins[:-1], bins, weights=count)
for x,y,num in zip(bins, density, count):
    if num != 0:
        plt.text(x, y+1, f'{num:,.0f}') # x,y,str
plt.title('Difference NIR')
plt.savefig(f'/home/eduardojh/Documents/results/nir_hist.png', bbox_inches='tight')
plt.close()

print('Done.')