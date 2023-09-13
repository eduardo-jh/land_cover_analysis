#!/usr/bin/env python
# coding: utf-8

""" Land cover classification with machine learning (random forest)
Eduardo Jimenez <eduardojh@email.arizona.edu>
NOTE: run under 'rsml' conda environment (python 3.8.13, scikit-learn 1.1.2)
"""
import os
import sys
import csv

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
        # sys.path.insert(0, '/home/eduardojh/Documents/land_cover_analysis/lib/')
        sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')
        cwd = '/VIP/engr-didan02s/DATA/EDUARDO/ML/'
    else:
        print('  System not yet configured!')

from landcoverclassification import LandCoverRaster, FeaturesDataset, LandCoverClassifier

# Directories
datadir = "/VIP/engr-didan02s/DATA/EDUARDO/CALAKMUL/ROI2/02_STATS/"
phenodir = "/VIP/engr-didan02s/DATA/EDUARDO/CALAKMUL/ROI2/03_PHENO/"
cwd = "/VIP/engr-didan02s/DATA/EDUARDO/ML/ROI2/"
ancillary_dir = "/VIP/engr-didan02s/DATA/EDUARDO/ML/ROI2/data/ancillary/"
# # Ancillary {ancillary_file_name: class_to_incorporate}
# ancillary_dict = {101: ["ag_roi2.tif"], 102: ["roads_roi2.tif", "urban_roi2.tif"]}

# Data files
fn_landcover = os.path.join(cwd, "data/inegi/usv250s7cw2018_ROI2full_ancillary.tif")
fn_tiles = os.path.join(cwd, 'parameters/tiles')

# list_tiles = ['h19v25', 'h20v24', 'h20v25', 'h20v26', 'h21v23',
#               'h21v24', 'h21v25', 'h21v26', 'h22v22', 'h22v23',
#               'h22v24', 'h22v25', 'h22v26', 'h23v22', 'h23v23',
#               'h23v24', 'h23v25']

raster = LandCoverRaster(fn_landcover, cwd)
print(raster)

features = FeaturesDataset(raster, datadir, phenodir, file_tiles=fn_tiles)

# fn_landcover = os.path.join(cwd, fn_landcover_orig[:-4] + "_ancillary.tif")
lcc = LandCoverClassifier(features)
lcc.classify_by_tile(['h19v25'])