#!/usr/bin/env python
# coding: utf-8

""" Validation of land cover maps using in-situ data points """

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, cohen_kappa_score

sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs

fmt = '%Y_%m_%d-%H_%M_%S'

def validation(cwd: str, fn_predictions: str, fn_validation: str, **kwargs):

    _prefix = kwargs.get('prefix', '1x1')
    exec_start = datetime.now()

    fn_save_raster = os.path.join(cwd, 'validation', f'{datetime.strftime(exec_start, fmt)}_{_prefix}_preds_sel_sites.tif')
    fn_save_report = os.path.join(cwd, 'validation', f'{datetime.strftime(exec_start, fmt)}_{_prefix}_validation_report.txt')
    fn_save_conf_matrix = os.path.join(cwd, 'validation', f'{datetime.strftime(exec_start, fmt)}_{_prefix}_validation_conf_matrix.csv')
    fn_save_conf_fig_pa = os.path.join(cwd, 'validation', f'{datetime.strftime(exec_start, fmt)}_{_prefix}_validation_conf_matrix_pa.png')
    fn_save_conf_fig_ua = os.path.join(cwd, 'validation', f'{datetime.strftime(exec_start, fmt)}_{_prefix}_validation_conf_matrix_ua.png')
    fn_save_params = os.path.join(cwd, 'validation', f'{datetime.strftime(exec_start, fmt)}_{_prefix}_exec_params.csv')
    fn_save_bars = os.path.join(cwd, 'validation', f'{datetime.strftime(exec_start, fmt)}_{_prefix}_barplot.png')

    # Read the land cover raster and retrive the land cover classes
    assert os.path.isfile(fn_predictions) is True, f"ERROR: File not found! {fn_predictions}"
    pred_arr, pred_nd, pred_gt, pred_ref = rs.open_raster(fn_predictions)
    print(f'  Opening raster: {fn_predictions}')
    print(f'    --NoData        : {pred_nd}')
    print(f'    --Columns       : {pred_arr.shape[1]}')
    print(f'    --Rows          : {pred_arr.shape[0]}')
    print(f'    --Geotransform  : {pred_gt}')
    print(f'    --Spatial ref.  : {pred_ref}')
    print(f'    --Type          : {pred_arr.dtype}')

    assert os.path.isfile(fn_validation) is True, f"ERROR: File not found! {fn_validation}"
    valid_arr, valid_nd, valid_gt, valid_ref = rs.open_raster(fn_validation)
    print(f'  Opening raster: {fn_validation}')
    print(f'    --NoData        : {valid_nd}')
    print(f'    --Columns       : {valid_arr.shape[1]}')
    print(f'    --Rows          : {valid_arr.shape[0]}')
    print(f'    --Geotransform  : {valid_gt}')
    print(f'    --Spatial ref.  : {valid_ref}')
    print(f'    --Type          : {valid_arr.dtype}')

    # Get the predictions where there are validation sites
    select_arr = np.where(valid_arr > 0, pred_arr, 0)
    print(f"Saving raster of predictions to match places with validation points: {fn_save_raster}")
    rs.create_raster(fn_save_raster, select_arr,  pred_ref, pred_gt)

    # Extract (this will reshape) features for comparison
    pred_arr = pred_arr.filled(0)
    valid_arr = valid_arr.filled(0)
    mask = valid_arr>0
    print(f"Mask contains: {np.sum(mask)} pixels")
    pred_comp = pred_arr[mask]
    valid_comp = valid_arr[mask]

    # print(f"Non-zero values: pred={(pred_arr>0).sum()} valid={(valid_arr>0).sum()} diff={abs((valid_arr>0).sum()-(pred_arr>0).sum())}")
    print(f"Non-zero values: pred={(pred_comp>0).sum()} valid={(valid_comp>0).sum()} diff={abs((valid_comp>0).sum()-(pred_comp>0).sum())}")
    print(f"Sum of values: pred={pred_comp.sum()} valid={valid_comp.sum()} diff={abs(valid_comp.sum()-pred_comp.sum())}")

    # Get the unique class values and their pixel count
    pred_cls, pred_counts = np.unique(pred_comp, return_counts=True)
    valid_cls, valid_counts = np.unique(valid_comp, return_counts=True)
    print("Values from the land cover predictions:")
    print(pred_cls, pred_counts)
    print("Values from the validation dataset:")
    print(valid_cls, valid_counts)

    # Create a dataset to match the label's classes
    common_cls = np.unique(np.concatenate((pred_cls, valid_cls)))

    class_names = [str(x) for x in common_cls]
    print(f"Class names for cross-tabulation: {class_names}")

    y1 = np.zeros(common_cls.shape, dtype=np.int16)
    y2 = np.zeros(common_cls.shape, dtype=np.int16)
    
    # Match the pixel counts using a common classes array
    j = 0  # counter for valid_counts
    k = 0  # counter for pred_counts
    for i, cls in enumerate(common_cls):
        if cls in pred_cls:
            y1[i] = pred_counts[k]
            k += 1
        if cls in valid_cls:
            y2[i] = valid_counts[j]
            j += 1
    
    # Adjust for plotting if the first value is zero
    common_cls_plt = common_cls.copy()
    if pred_cls[0] == 0:
        common_cls_plt = np.arange(0, len(common_cls))
        print(f"Adjusting x-axis to: {common_cls_plt}")
    print(f"Common classes: {common_cls}")
    print(f"Pred  counts 1: {y1}")
    print(f"Valis counts 2: {y2}")

    # Plot the label counts
    width = 0.4
    # fig = plt.figure(figsize=(14,12))
    # plt.bar(pred_cls-width, pred_counts, width, label="Predictions", log=True)
    # plt.bar(pred_cls, y2, width, label="Validation", log=True)
    # # plt.bar(pred_cls-width, pred_counts, width, label="Predictions",)
    # # plt.bar(pred_cls, y2, width, label="Validation")
    # plt.xlabel("Land cover classes")
    # plt.ylabel("Pixel count")
    # plt.xticks(np.arange(101, 112))
    # plt.legend(loc='best')
    
    fig, ax = plt.subplots(layout='constrained')
    r1 = ax.bar(common_cls_plt-width, y1, width, label="Predictions", log=True)
    r2 = ax.bar(common_cls_plt, y2, width, label="Validation", log=True)
    ax.bar_label(r1, padding=3, fontsize=6)
    ax.bar_label(r2, padding=3, fontsize=6)
    ax.set_xlabel("Land cover classes")
    ax.set_ylabel("Pixel count")
    ax.set_xticks(common_cls_plt, class_names)
    ax.legend(loc='upper left', ncol=2)
    plt.savefig(fn_save_bars, bbox_inches='tight', dpi=300)

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

    # Save the execution parameters
    with open(fn_save_params, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Predictions file', fn_predictions])
        writer.writerow(['  NoData', pred_nd])
        writer.writerow(['  Rows', pred_arr.shape[0]])
        writer.writerow(['  Columns', pred_arr.shape[1]])
        writer.writerow(['  Geotransform', pred_gt])
        writer.writerow(['  Spatial reference', pred_ref])
        writer.writerow(['  Data type', pred_arr.dtype])
        writer.writerow(['Validation file', fn_validation])
        writer.writerow(['  NoData', valid_nd])
        writer.writerow(['  Rows', valid_arr.shape[0]])
        writer.writerow(['  Columns', valid_arr.shape[1]])
        writer.writerow(['  Geotransform', valid_gt])
        writer.writerow(['  Spatial reference', valid_ref])
        writer.writerow(['  Data type', valid_arr.dtype])
        writer.writerow(['Masked predictions raster', fn_save_raster])
        writer.writerow(['Validation pixels (mask)', np.sum(mask)])
        writer.writerow(['Non-zero prediction pixels (masked)', (pred_comp>0).sum()])
        writer.writerow(['Non-zero validation pixels (masked)', (valid_comp>0).sum()])
        writer.writerow(['Non-zero pixels difference', abs((valid_comp>0).sum()-(pred_comp>0).sum())])
        writer.writerow(['Sum of predictions', pred_comp.sum()])
        writer.writerow(['Sum of validation', valid_comp.sum()])
        writer.writerow(['Sum difference', abs(valid_comp.sum()-pred_comp.sum())])
        writer.writerow(['Prediction classes', ';'.join([str(x) for x in pred_cls])])
        writer.writerow(['Prediction pixel count', ';'.join(str(x) for x in pred_counts)])
        writer.writerow(['Validation classes', ';'.join(str(x) for x in valid_cls)])
        writer.writerow(['Validation pixel count', ';'.join(str(x) for x in valid_counts)])
        writer.writerow(['Common classes', ';'.join(str(x) for x in common_cls)])
        writer.writerow(['Prediction pixel count (matched)', ';'.join(str(x) for x in y1)])
        writer.writerow(['Validation pixel count (matched)', ';'.join(str(x) for x in y2)])
        writer.writerow(['Class names', ';'.join(class_names)])
        writer.writerow(['Accuracy score', accuracy])
        writer.writerow(['Kappa', kappa])
        writer.writerow(['Bar plot pixel count comparison', fn_save_bars])
        writer.writerow(['Confusion matrix file', fn_save_conf_matrix])
        writer.writerow(['Producer accuracy figure', fn_save_conf_fig_pa])
        writer.writerow(['User accuracy figure', fn_save_conf_fig_ua])
        writer.writerow(['Classification report file', fn_save_report])

    exec_end = datetime.now()
    exec_elapsed = exec_end - exec_start
    
    print(f"{exec_end} Validation script ended (runtime: {exec_elapsed}).")

if __name__ == "__main__":

    # Validation rasters are genereated from point vectors into raster of radii of 30m (single-pixel),
    # 45m (3x3 window), 105m (7x7 window), in order to try simulate patches/windows of pixels
    # Smaller windows may ignore some points in the process of generating

    # =============================== 2016-2019 ===============================
    # cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2016_2019/'
    # fn_preds_map = os.path.join(cwd, 'results/2023_10_28-18_19_05', '2023_10_28-18_19_05_predictions.tif')
    
    # Compare against the entire dataset (all years)
    # fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts.tif')
    # id="1x1"
    # fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts_r45_3x3.tif')
    # id="3x3"
    # fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts_r75_5x5.tif')
    # id="5x5"
    
    # Compare against the dataset for the matching period with this INEGI series VI (2015-2016)
    # fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts_2015_2016.tif')
    # id="1x1p"
    # fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts_2015_2016_r45_3x3.tif')
    # id="3x3p"
    # fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts_2015_2016_r75_5x5.tif')
    # id="5x5p"

    # =============================== 2019-2022 ===============================
    cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/2019_2022/'
    fn_preds_map = os.path.join(cwd, 'results/2023_10_29-12_10_07', '2023_10_29-12_10_07_predictions.tif')

    # Compare against the validation dataset for this period
    # fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts_2017_2019.tif')
    # id = "1x1"
    # fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts_2017_2019_r45_3x3.tif')
    # id = "3x3"
    fn_valid_map = os.path.join(cwd, 'validation', 'infys_pts_2017_2019_r75_5x5.tif')
    id = "5x5"

    # Run validation
    validation(cwd, fn_preds_map, fn_valid_map, prefix=id)

