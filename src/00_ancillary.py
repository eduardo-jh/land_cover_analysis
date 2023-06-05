#!/usr/bin/env python
# coding: utf-8

""" Integrate ancillary rasters into land cover raster.

NOTICE: run from 'rsml' environment (Python 3.8.13; GDAL 3.5.2 & matplotlib 3.6.0 from conda-forge)

author: Eduardo Jimenez <eduardojh@arizona.edu>
"""

import sys
import numpy as np

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

fn_ancillary1 = cwd + 'data/ancillary/AgricultureROI1.tif'
fn_ancillary2 = cwd + 'data/ancillary/RoadsROI1.tif'
fn_ancillary3 = cwd + 'data/ancillary/UrbanAreaROI1.tif'
fn_landcover = cwd + 'data/inegi_2018/usv250s7cw_ROI1_LC_KEY_grp.tif'  # Ancillay data has grouped values
fn_new_landcover = cwd + 'data/inegi_2018/land_cover_ROI1.tif'

print("Integrating ancillary rasters into land cover...")

# Read and aggregate the raster files
landcover, nodata, metadata, geotransform, projection, epsg = rs.open_raster(fn_landcover)

ag, _, _, _, _, _ = rs.open_raster(fn_ancillary1)
roads, _, _, _, _, _ = rs.open_raster(fn_ancillary2)
urban, _, _, _, _, _ = rs.open_raster(fn_ancillary3)

# Last will overwrite
assert ag.shape == landcover.shape, "Ag and landcover do not match"
new_landcover = np.where(ag > 0, ag, landcover)

assert roads.shape == new_landcover.shape, "Roads and landcover do not match"
new_landcover = np.where(roads > 0, roads, new_landcover)

assert urban.shape == new_landcover.shape, "Urban and landcover do not match"
new_landcover = np.where(urban > 0, urban, new_landcover)

# Save the new land cover raster
if type(epsg) is not int:
    epsg = int(epsg)
rs.create_raster(fn_new_landcover, new_landcover, epsg, geotransform)