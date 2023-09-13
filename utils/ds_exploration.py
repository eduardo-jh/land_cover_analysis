#!/usr/bin/env python
# coding: utf-8

""" Data exploration for ROI2

@author: Eduardo Jimenez <eduardojh@email.arizona.edu>
@date: 2023-09-10

Compare datasets...
"""
import os
import sys
import csv
import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')
cwd = '/VIP/engr-didan02s/DATA/EDUARDO/ML/ROI2'

import rsmodule as rs

fn_ds1 = os.path.join(cwd, 'features/h19v25', 'features_season_h19v25.h5')
fn_ds2 = os.path.join(cwd, 'features/NDVI/h19v25', 'features_season_h19v25.h5')

feat_plot = 1

feats = []
with h5py.File(fn_ds1, 'r') as hdf1:
    feats = list(hdf1.keys())
    ds1 = hdf1[feats[feat_plot]][:]
    print(hdf1[feats[feat_plot]].attrs.keys())
    dsa1 = hdf1[feats[feat_plot]].attrs.keys()

fig1 = os.path.join(cwd, 'results', f'h19v25_{feats[feat_plot]}.png')
rs.plot_dataset(ds1, title='DS1 - ' + feats[feat_plot], savefig=fig1)

with h5py.File(fn_ds1, 'r') as hdf2:
    feats2 = list(hdf2.keys())
    ds2 = hdf2[feats2[feat_plot]][:]
    dsa2 = hdf2[feats[feat_plot]].attrs.keys()

fig2 = os.path.join(cwd, 'results', f'h19v25_ndvi_{feats2[feat_plot]}.png')
rs.plot_dataset(ds2, title='DS2 - ' + feats2[feat_plot], savefig=fig2)

print(f"{len(feats)} {len(feats2)}")

for i, feat in enumerate(feats):
    if feat in feats2:
        print(f"{i}: {feat} in both files.")

print(dsa1)