#!/usr/bin/env python
# coding: utf-8

""" A program to test functions for RF prediction perfomance assessment and confusion matrices 
    As Nov 10, this code has been integrated on rf_classification.py already.
"""

import sys
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs

cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/results/2023_10_26-00_34_17/'
fn_pred_roi = os.path.join(cwd, '2023_10_26-00_34_17_predictions_roi.tif')
fn_landcover = "/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/data/usv250s5ugw_grp11_ancillary.tif"
fn_mask = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/data/YucPenAquifer_mask.tif'

fn_save_crosstab = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/my_test_crostabulation.csv'
fn_save_conf_tbl = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/my_test_confusion_table.csv'
fn_save_classif_report = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/my_test_classif_report.txt'
fn_save_conf_matrix_fig = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2013_2016/my_conf_matrix.png'

y_pred_roi, _, _, _ = rs.open_raster(fn_pred_roi)
land_cover, _, _, _ = rs.open_raster(fn_landcover)
nodata_mask, _, _, _ = rs.open_raster(fn_mask)

y_mask = nodata_mask.filled(0)

# Update mask to remove pixels with no land cover class
# Remove southern part (Guatemala, Belize) in the performance assessment
# as there is no data in that region, remove: -13000, 0, and/or '--' pixels
y_mask = np.where(land_cover.filled(0) == 0, 0, y_mask)

# Mask out NoData values (this will flatten the arrays)
y_predictions = y_pred_roi[y_mask > 0]
y_true = land_cover[y_mask > 0]

class_names_ = np.unique(y_true)
print(np.unique(y_predictions), y_predictions.shape)
print(class_names_, y_true.shape)
print(np.unique(y_mask), y_mask.shape)

class_names = [str(x) for x in class_names_]
print(class_names)

df_pred = pd.DataFrame({'truth': y_true, 'predict': y_predictions})
crosstab_pred = pd.crosstab(df_pred['truth'], df_pred['predict'], margins=True)
crosstab_pred.to_csv(fn_save_crosstab)
print(f'Saving crosstabulation: {fn_save_crosstab}')

accuracy = accuracy_score(y_true, y_predictions)
print(f'***Accuracy score: {accuracy:>0.4f}***')

cm = confusion_matrix(y_true, y_predictions)
print(f'Saving confusion matrix: {fn_save_conf_tbl}')
with open(fn_save_conf_tbl, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for single_row in cm:
        writer.writerow(single_row)

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]

for title, normalize in titles_options:
    # plt.figure(figsize=(28, 32), constrained_layout=True)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_predictions,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.figure_.set_figwidth(16)
    disp.figure_.set_figheight(12)
    # disp.figure_.set_constrained_layout(True) # division by 0 error
    disp.ax_.set_title(title)

    if normalize is not None:
        fn_save_conf_matrix_fig = fn_save_conf_matrix_fig[:-4] + '_normalized.png'
    print(f'Saving fig: {fn_save_conf_matrix_fig}')
    disp.figure_.savefig(fn_save_conf_matrix_fig, bbox_inches='tight')

report = classification_report(y_true, y_predictions, )
print(f'Saving classification report: {fn_save_classif_report}')
print(report)
with open(fn_save_classif_report, 'w') as f:
    f.write(report)
