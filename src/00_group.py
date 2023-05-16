#!/usr/bin/env python
# coding: utf-8

""" Reclasify and merge land cover classes into groups, which are more easy to work with.

NOTICE: run from 'rsml' environment (Python 3.8.13; GDAL 3.5.2 & matplotlib 3.6.0 from conda-forge)

author: Eduardo Jimenez <eduardojh@arizona.edu>
Changelog:
    May 12, 2023: Initial code.
    May 15, 2023: Working code matches updated functions on rsmodule.
"""

import sys
import platform

LOCAL = True
# adding the directory with modules
system = platform.system()
if system == 'Windows':
    # On Windows 10
    sys.path.insert(0, 'D:/Desktop/land_cover_analysis/lib/')
    cwd = 'D:/Desktop/CALAKMUL/ROI1/'
elif system == 'Linux' and not LOCAL:
    # On Alma Linux Server
    sys.path.insert(0, '/home/eduardojh/Documents/land_cover_analysis/lib/')
    cwd = '/VIP/engr-didan01s/DATA/EDUARDO/DATA/CALAKMUL/ROI1/'
elif system == 'Linux' and LOCAL:
    # On Ubuntu Workstation
    sys.path.insert(0, '/vipdata/2023/land_cover_analysis/lib/')
    cwd = '/vipdata/2023/CALAKMUL/ROI1/'
else:
    print('System not yet configured!')

import rsmodule as rs

fn_landcover = cwd + 'training/usv250s7cw_ROI1_LC_KEY.tif'
fn_keys = cwd + 'training/land_cover_groups.csv'
fn_grp = fn_landcover[:-4] + '_grp.csv'  # save the grouping into a CSV

# inegi_indices = (2, 1, 4)  # INEGI's land cover key, desciption, and group
lc_ind, lc_grp = rs.land_cover_freq(fn_landcover, fn_keys, verbose=False)

# Print land cover frequencies
keys = sorted(lc_ind.keys())
print("\nFrequency for individual land cover classes.")
print(f"{'Key':>8} {'Frequency':>12}")
for key in keys:
    print(f'{key:>8} {lc_ind[key]:>12}')

keys = sorted(lc_grp.keys())
print("\nFrequency for group of land cover classes.")
print(f"{'Key':>8} {'Frequency':>12}")
for key in keys:
    print(f'{key:>8} {lc_grp[key]:>12}')

# Get the land cover classes by each group
lc_grp = rs.land_cover_by_group(fn_landcover, fn_keys, fn_grp_keys=fn_grp, verbose=True)
print(lc_grp)

# Create a reclassified raster using groups of land cover classes
rs.reclassify_by_group(fn_landcover, lc_grp, verbose=True)

print('Done ;-)')