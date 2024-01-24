#!/usr/bin/env python
# coding: utf-8

""" Validation of land cover maps using in-situ data points """

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, cohen_kappa_score

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs

cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/'
fn_preds_map = os.path.join(cwd, 'results/2023_10_28-18_19_05', '2023_10_28-18_19_05_predictions.tif')

# Validation rasters are genereated from point vectors into raster of radii of 30m (single-pixel),
# 45m (3x3 window), 105m (7x7 window), in order to try simulate patches/windows of pixels
# Smaller windows may ignore some points in the process of generating
fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts_r105.tif')

fn_save_raster = os.path.join(cwd, 'validation', 'preds_val_sites.tif')
fn_save_report = os.path.join(cwd, 'validation', 'validation_report.txt')
fn_save_conf_matrix = os.path.join(cwd, 'validation', 'validation_conf_matrix.csv')
fn_save_conf_fig_pa = os.path.join(cwd, 'validation', 'validation_conf_matrix_pa.png')
fn_save_conf_fig_ua = os.path.join(cwd, 'validation', 'validation_conf_matrix_ua.png')

# Read the land cover raster and retrive the land cover classes
assert os.path.isfile(fn_preds_map) is True, f"ERROR: File not found! {fn_preds_map}"
pred_arr, pred_nd, pred_gt, pred_ref = rs.open_raster(fn_preds_map)
print(f'  Opening raster: {fn_preds_map}')
print(f'    --NoData        : {pred_nd}')
print(f'    --Columns       : {pred_arr.shape[1]}')
print(f'    --Rows          : {pred_arr.shape[0]}')
print(f'    --Geotransform  : {pred_gt}')
print(f'    --Spatial ref.  : {pred_ref}')
print(f'    --Type          : {pred_arr.dtype}')

assert os.path.isfile(fn_valid_map) is True, f"ERROR: File not found! {fn_valid_map}"
valid_arr, valid_nd, valid_gt, valid_ref = rs.open_raster(fn_valid_map)
print(f'  Opening raster: {fn_valid_map}')
print(f'    --NoData        : {valid_nd}')
print(f'    --Columns       : {valid_arr.shape[1]}')
print(f'    --Rows          : {valid_arr.shape[0]}')
print(f'    --Geotransform  : {valid_gt}')
print(f'    --Spatial ref.  : {valid_ref}')
print(f'    --Type          : {valid_arr.dtype}')

print("Values from the land cover predictions:")
print(np.unique(pred_arr, return_counts=True))
print("Values from the validation dataset:")
print(np.unique(valid_arr, return_counts=True))

# Get the predictions where there are validation sites
select_arr = np.where(valid_arr > 0, )

# Extract (this will reshape) features for comparison
pred_arr = pred_arr.filled(0)
valid_arr = valid_arr.filled(0)
mask = valid_arr>0
print(f"Mask contains: {np.sum(mask)} pixels")
pred_comp = pred_arr[mask]
valid_comp = valid_arr[mask]

class_names_ = np.unique(pred_comp)
class_names = [str(x) for x in class_names_]
print(f"Class names for cross-tabulation: {class_names}")

accuracy = accuracy_score(valid_comp, pred_comp)
print(f'***Accuracy score: {accuracy:>0.4f}***')

cm = confusion_matrix(valid_comp, pred_comp)
print(f'Saving confusion matrix: {fn_save_conf_matrix}')
with open(fn_save_conf_matrix, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for single_row in cm:
        writer.writerow(single_row)

### User's accuracy

# Create the normalized confussion matrix for user's accuracy
title = "Normalized confusion matrix (user's accuracy)"
disp = ConfusionMatrixDisplay.from_predictions(
    valid_comp,
    pred_comp,
    display_labels=class_names,
    cmap=plt.cm.Blues,
    normalize='pred',  # IMPORTANT: normalize by predicted conditions (user's accuracy)
)
disp.figure_.set_figwidth(16)
disp.figure_.set_figheight(12)
disp.ax_.set_title(title)

print(f'Saving confusion matrix figure: {fn_save_conf_fig_ua}')
disp.figure_.savefig(fn_save_conf_fig_ua, bbox_inches='tight')

### Producer's accuracy

# Create the normalized confussion matrix for producer's accuracy
title = "Normalized confusion matrix (producer's accuracy)"
disp = ConfusionMatrixDisplay.from_predictions(
    valid_comp,
    pred_comp,
    display_labels=class_names,
    cmap=plt.cm.Blues,
    normalize='true',  # IMPORTANT: normalize by true conditions (producer's accuracy)
)
disp.figure_.set_figwidth(16)
disp.figure_.set_figheight(12)
disp.ax_.set_title(title)

print(f'Saving confusion matrix figure: {fn_save_conf_fig_pa}')
disp.figure_.savefig(fn_save_conf_fig_pa, bbox_inches='tight')

# Finally, perform kappa analysis

print('Running Cohens kappa analysis:')
kappa = cohen_kappa_score(pred_comp, valid_comp)
print(f"kappa: {kappa}")

# Generate a complete classification report

report = classification_report(valid_comp, pred_comp, )
print(f'Saving classification report: {fn_save_report}')
print(report)
with open(fn_save_report, 'w') as f:
    f.write(report)

print("Validation script done! ;-)")