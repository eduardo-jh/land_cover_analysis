#!/usr/bin/env python
# coding: utf-8

""" Map accuracy """

import os
import sys
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix

# sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')
sys.path.insert(0, '/error/random_sampling/land_cover_analysis/lib/')

import rsmodule as rs

HA = (30 * 30) / 10000.

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


def stratified_random_sampling(fn_landcover, out_dir, list_classes, list_size, **kwargs):
    """ Stratified random sampling 
    
    2026-03-13: Adapted to use sample sizes per each class instead of a constant percent.

    :param str fn_landcover: TIF file name to sample, contains classes (strata)
    :param str out_dir: directory to save results
    :param list list_classes: list of classes to sample, integer
    :param list list_size: matching list with sample size for each class
    :param int window_size: default is sampling a window of 7x7 pixels
    :param int max_trials: max of attempts to fill the sample size
    """
 
    start = datetime.now()
    print(f"\n{start}: ========== Starting stratified random sampling ==========")

    assert os.path.isfile(fn_landcover) is True, f"ERROR: File not found! {fn_landcover}"
    assert len(list_classes) > 0, "ERROR: no classes (strata) to sample, list empty"
    assert len(list_classes) == len(list_size), f"ERROR: list mismatch, {len(list_classes)} != {len(list_size)}"

    # Keyword options
    _window_size = kwargs.get("window_size", 7)
    _max_trials = int(kwargs.get("max_trials", 2e5))
    fn_sample_mask = kwargs.get("sampling_mask", "sample_mask.tif")
    fn_sample_sizes = kwargs.get("sample_sizes", "sampled.csv")
    _class_labels = kwargs.get("class_labels", None)
    _raster_labels = kwargs.get("raster_labels", False)

    if not os.path.exists(out_dir):
        print(f"\nCreating new path: {out_dir}")
        os.makedirs(out_dir)
    
    fn_sample_mask = os.path.join(out_dir, fn_sample_mask)
    fn_sample_sizes = os.path.join(out_dir, fn_sample_sizes)

    # Read the land cover raster and retrive the land cover classes
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
    landcover_frequencies = rs.land_cover_freq(fn_landcover, verbose=False)
    print(f'  --Land cover frequencies: {landcover_frequencies}')
    
    classes = list(landcover_frequencies.keys())
    freqs = [landcover_frequencies[x] for x in classes]  # pixel count
    percentages = (freqs / sum(freqs)) * 100

    print(classes)
    print(freqs)
    print(percentages)
    print(list_size)

    #### Sample size
    # Use a dataframe to calculate sample size
    df = pd.DataFrame({'Key': classes, 'PixelCount': freqs, 'Percent': percentages, 'SampleSizePixels': list_size})
    if _class_labels is not None:
        df = pd.DataFrame({'Key': classes, 'Class': _class_labels, 'PixelCount': freqs, 'Percent': percentages, 'SampleSizePixels': list_size})
 
    # Now calculate percentages
    df['SampleSizePercent'] = (df['SampleSizePixels'] / df['PixelCount'])*100
    print(df)

    nrows, ncols = land_cover.shape
    print(f"  --Total pixels={nrows*ncols}, Values={sum(df['PixelCount'])}, NoData/Missing={nrows*ncols - sum(df['PixelCount'])}")

    sample = {}  # to save the sample

    # Create a mask of the sampled regions
    sampling_mask = np.zeros(land_cover.shape, dtype=land_cover.dtype)

    # A window will be used for sampling, this array will hold the sample
    window_sample = np.zeros((_window_size,_window_size), dtype=int)

    print(f'  --Max trials: {_max_trials}')

    trials = 0  # attempts to complete the sample
    completed = {}  # classes which sample is complete

    for sample_key in list(df['Key']):
        completed[sample_key] = False
    completed_samples = sum(list(completed.values()))  # Values are all True if completed
    total_classes = len(completed.keys())
    # print(completed)

    sampled_points = []

    while (trials < _max_trials and completed_samples < total_classes):
        show_progress = (trials%10000 == 0)  # Step to show progress
        if show_progress:
            print(f'  --Trial {1 if trials == 0 else trials:>8} of {_max_trials:>8} ', end='')

        # 1) Generate a random point (row_sample, col_sample) to sample the array
        #    Coordinates relative to array positions [0:nrows, 0:ncols]
        #    Subtract half the window_size to avoid sampling too close to the edges, use window_size step to avoid overlapping
        col_sample = random.randrange(0 + _window_size//2, ncols - _window_size//2, _window_size)
        row_sample = random.randrange(0 + _window_size//2, nrows - _window_size//2, _window_size)

        # Save the points previously sampled to avoid repeating and oversampling
        point = (row_sample, col_sample)
        if point in sampled_points:
            trials +=1
            continue

        # 2) Generate a sample window around the random point, here create the boundaries,
        #    these rows and columns will be used to slice the sample
        win_col_ini = col_sample - _window_size//2
        win_col_end = col_sample + _window_size//2 + 1  # add 1 to slice correctly
        win_row_ini = row_sample - _window_size//2
        win_row_end = row_sample + _window_size//2 + 1

        assert win_col_ini < win_col_end, f"Incorrect slice indices on x-axis: {win_col_ini} < {win_col_end}"
        assert win_row_ini < win_row_end, f"Incorrect slice indices on y-axis: {win_row_ini} < {win_row_end}"

        # 3) Check if sample window is out of range, if so trim the window to the array's edges accordingly
        #    This may not be necessary if half the window size is subtracted, but still
        if win_col_ini < 0:
            # print(f'    --Adjusting win_col_ini: {win_col_ini} to 0')
            win_col_ini = 0
        if win_col_end > ncols:
            # print(f'    --Adjusting win_col_end: {win_col_end} to {ncols}')
            win_col_end = ncols
        if win_row_ini < 0:
            # print(f'    --Adjusting win_row_ini: {win_row_ini} to 0')
            win_row_ini = 0
        if  win_row_end > nrows:
            # print(f'    --Adjusting win_row_end: {win_row_end} to {nrows}')
            win_row_end = nrows
        
        # Add sampling window to sampled points to avoid overlapping sample windows
        for row_w in range(win_row_ini, win_row_end):
            for col_w in range(win_col_ini, win_col_end):
                point = (row_w, col_w)
                sampled_points.append(point)

        # 4) Check and adjust the shapes of the arrays to slice and insert properly, only final row/column can be adjusted
        window_sample = land_cover[win_row_ini:win_row_end,win_col_ini:win_col_end]
        
        # 5) Get the unique values in sample (sample_keys) and its count (sample_freq)
        sample_keys, sample_freq = np.unique(window_sample, return_counts=True)
        classes_to_remove = []  # Avoid adding zeros or completed classes to the mask

        # 6) Iterate over each class sample and add its respective pixel count to the sample
        for sample_class, class_count in zip(sample_keys, sample_freq):
            if sample_class == nodata:
                # Sample is mixed with zeros, tag it to remove it and go to next sample_class
                classes_to_remove.append(sample_class)
                continue

            if completed.get(sample_class, False):
                classes_to_remove.append(sample_class)  # do not add completed classes to the sample
                continue

            # Accumulate the pixel counts, chek first if general sample is completed
            if sample.get(sample_class) is None:
                sample[sample_class] = class_count
            else:
                # if sample[sample_class] < sample_sizes[sample_class]:
                sample_size = df[df['Key'] == sample_class]['SampleSizePixels'].item()

                # If sample isn't completed, add the sampled window
                if sample[sample_class] < sample_size:
                    sample[sample_class] += class_count
                    # Check if last addition completed the sample
                    if sample[sample_class] >= sample_size:
                        completed[sample_class] = True  # this class' sample is now complete
                        # but do not add to classes_to_remove
                else:
                    # This class' sample was completed already
                    completed[sample_class] = True
                    classes_to_remove.append(sample_class)

        # Create an array containing all the sampled pixels by adding the sampled windows from each quadrant (or part)
        sampled_window = np.zeros(window_sample.shape, dtype=land_cover.dtype)
        
        # Filter out classes with already complete samples
        if len(classes_to_remove) > 0:
            for single_class in classes_to_remove:
                # Put a 1 on a complete class
                filter_out = np.where(window_sample == single_class, 1, 0)
                sampled_window += filter_out.astype(land_cover.dtype)
            
            # All values greater than zero are pixels to remove from mask, reverse it so 1's are the sample mask
            sampled_window = np.where(sampled_window == 0, 1, 0)
        else:
            sampled_window = window_sample[:,:].astype(land_cover.dtype)
        
        # Slice and insert sampled window
        sampling_mask[win_row_ini:win_row_end,win_col_ini:win_col_end] += sampled_window.astype(land_cover.dtype)

        trials += 1

        completed_samples = sum(list(completed.values()))  # Values are all True if completed
        if show_progress:
            print(f' (completed {completed_samples:>2}/{total_classes:>2} samples)')
        if completed_samples >= total_classes:
            print(f'  --All samples completed in {trials} trials! Exiting.')

    if trials == _max_trials:
        print('  --WARNING! Max trials reached, samples may be incomplete, try increasing max trials.')

    print(f'  --Sample sizes per class: {sample}')
    print(f'  --Completed samples: {completed}')

    # print('\n  --WARNING! This may contain oversampling caused by overlapping windows!'), # fixed and no overlapping
    df['SampledPixels'] = [sample.get(x,0) for x in df['Key']]
    df['SampledPercent'] = (df['SampledPixels'] / df['SampleSizePixels']) * 100
    df['SampledPerClass'] = (df['SampledPixels'] / df['PixelCount']) * 100
    df['SampleComplete'] = [completed[x] for x in df['Key']]
    df.to_csv(fn_sample_sizes)
    print(df)

    # Convert the sampling_mask to 1's (indicating pixels to sample) and 0's
    sampling_mask = np.where(sampling_mask >= 1, 1, 0)
    print(f"  --Values in mask: {np.unique(sampling_mask)}")  # should be 1 and 0

    # Create a raster with actual labels (land cover classes) as pixel values
    if _raster_labels:
        fn_sample_mask_labels = fn_sample_mask[:-4] + "_labels.tif"
        training_labels = np.where(sampling_mask > 0, land_cover, 0)
        print(f"  --Creating raster: {fn_sample_mask_labels}")
        rs.create_raster(fn_sample_mask_labels, training_labels, spatial_reference, geotransform)

    # Create a raster with the sampled windows, this will be the sampling mask
    print(f"  --Creating raster: {fn_sample_mask}")
    rs.create_raster(fn_sample_mask, sampling_mask, spatial_reference, geotransform)

    end = datetime.now()
    print(f"{end}: ========== Stratified random sampling elapsed in: {end - start} ==========\n")


if __name__ == "__main__":

    # Map accuracy
    # cwd = '/VIP/engr-didan01s/DATA/EDUARDO/2024/YUCATAN_LAND_COVER/ROI2/'
    # cwd = '/error/random_sampling/'
    cwd = '/media/ecoslacker/RESEARCH/YUCATAN_LAND_COVER/ROI2/'

    # period = '2013_2016'
    # fn_reference = os.path.join(cwd, period, 'data/usv250s5ugw_grp11_ancillary.tif')
    # fn_mapped = os.path.join(cwd, period, 'results/2024_03_06-23_19_43', '2024_03_06-23_19_43_predictions.tif')

    period = '2016_2019'
    fn_reference = os.path.join(cwd, period, 'data/usv250s6gw_grp11_ancillary.tif')
    fn_mapped = os.path.join(cwd, period, 'results/2024_03_08-13_29_31', '2024_03_08-13_29_31_predictions.tif')

    # period = '2019_2022'
    # fn_reference = os.path.join(cwd, period, 'data/usv250s7gw_grp11_ancillary.tif')
    # fn_mapped = os.path.join(cwd, period, 'results/2024_03_12-19_32_01', '2024_03_12-19_32_01_predictions.tif')

    out_directory = os.path.join(cwd, "map_accuracy", period)
    
    # Explore rasters, 
    # The original used for land cover labels based on INEGI 
    # The land cover map generated with the random forest

    assert os.path.isfile(fn_reference) is True, f"ERROR: File not found! {fn_reference}"
    ref, ref_nd, ref_gt, ref_sr = rs.open_raster(fn_reference)
    print(f'Opening raster: {fn_reference}')
    print(f'  --NoData        : {ref_nd}')
    print(f'  --Columns       : {ref.shape[1]}')
    print(f'  --Rows          : {ref.shape[0]}')
    print(f'  --Geotransform  : {ref_gt}')
    print(f'  --Spatial ref.  : {ref_sr}')
    print(f'  --Type          : {ref.dtype}')

    assert os.path.isfile(fn_mapped) is True, f"ERROR: File not found! {fn_mapped}"
    map, map_nd, map_gt, map_sr = rs.open_raster(fn_mapped)
    print(f'Opening raster: {fn_mapped}')
    print(f'  --NoData        : {map_nd}')
    print(f'  --Columns       : {map.shape[1]}')
    print(f'  --Rows          : {map.shape[0]}')
    print(f'  --Geotransform  : {map_gt}')
    print(f'  --Spatial ref.  : {map_sr}')
    print(f'  --Type          : {map.dtype}')

    # print("Counting...")
    # ref_val, ref_cnt = np.unique(ref, return_counts=True)
    # print("Counting...")
    # map_val, map_cnt = np.unique(map, return_counts=True)

    # print("For the reference:")
    # print(ref_val, ref_cnt)

    # print("For the map:")
    # print(map_val, map_cnt)

    # Apply the mask of the reference to the map
    ref_mask = np.ma.getmask(ref)
    ref = ref.filled(0)
    map = map.filled(0)
    map = np.ma.masked_array(map, mask=ref_mask)
    map = map.filled(0)

    print("Counting ref...")
    ref_val, ref_cnt = np.unique(ref, return_counts=True)
    print("Counting map...")
    map_val, map_cnt = np.unique(map, return_counts=True)

    print("For the reference:")
    ref_val = ref_val[1:]
    ref_cnt = ref_cnt[1:]
    print(ref_val)
    print(ref_cnt)

    print("For the map:")
    map_val = map_val[1:]
    map_cnt = map_cnt[1:]
    print(map_val)
    print(map_cnt)

    print("Final classes: ")
    # print(ref_classes)
    print(f"Ref values={ref_val}")

    # Land cover classes
    labels = ['Agriculture',
          'Urban',
          'Water',
          'Barren land',
          'Mangrove',
          'Evergreen tropical forest',
          'Savanna',
          'Wetland',
          'Deciduous tropical forest',
          'Coastal vegetation',
          'Oak forest']
    
    short_labels = ["AG", "UR", "WA", "BL", "MA", "EF", "SV", "WL", "DF", "CV", "OF"]
    
    # Percent covered by each land cover class
    area_per = np.array([17.6623883239445,
                1.71777277244448,
                0.550239788035759,
                0.347340085680864,
                3.12805275324672,
                43.6949885906419,
                0.762610332392122,
                3.75309835634248,
                28.1401377529601,
                0.0462389805815985,
                0.19713226372948
                ])

    # Determine sample size
    # Weights
    weights = area_per/100
    
    N = sum(ref_cnt)
    print(f"N={N}")
    
    # standard error of the estimated overall accuracy that we would like to achieve
    # target standard error for overall accuracy
    SO = 0.01

    # User's accuracy, just a conjecture from previous experience
    # Used the ones from the land cover classification
    U = [0.75, 0.80, 0.85, 0.84, 0.78, 0.87, 0.81, 0.76, 0.83, 0.79, 0.93]

    WiSi_sum = 0
    WiSi2_sum = 0
    for i in range(len(ref_val)):
        Si = np.sqrt(U[i] * (1-U[i]))
        WiSi = weights[i] * Si
        WiSi2 = weights[i] * (Si * Si)
        print(f"{i}: Si={Si:0.4f} WiSi={WiSi:0.4f} WiSi2={WiSi2:0.4f}")
        WiSi_sum += WiSi
        WiSi2_sum += WiSi2
    n = (WiSi_sum / SO) * (WiSi_sum / SO)

    print("1. Sample size for stratified random sampling:")
    print(f"n={n}\n")
    n = (WiSi_sum  * WiSi_sum) / ((SO*SO) + (1/N) * WiSi2_sum)
    print(f"n={n} (full equation)\n")

    sample_dic = {"Strata": [x for x in short_labels]}
    sample_dic["Equal"] = [int(round(n / len(short_labels), 0))] * len(short_labels)

    alloc1 = np.zeros(len(short_labels), int)
    fixed_value = 100
    fix_pos = [1, 2, 3, 4, 6, 7, 9, 10]
    weight_rem = sum([weights[j] for j in fix_pos])
    additional_weight = round(weight_rem / (len(short_labels) - len(fix_pos)), 2)
    nr1 = int(round(n - (len(fix_pos) * fixed_value), 0))
    print(f"n-r={nr1}")
    for i in range(len(short_labels)):
        if i in fix_pos:
            alloc1[i] = fixed_value
            # print(f"{i}: {fixed_value}")
        else:
            alloc1[i] = round(weights[i] + additional_weight, 2) * nr1
            # print(f"{i}: {weights[i]} {weights[i] + additional_weight} \
            #       {weights[i] * nr1} {(weights[i] + additional_weight) * nr1}")
    sample_dic["Alloc1"] = alloc1

    alloc2 = np.zeros(len(short_labels), int)
    fixed_value = 75
    nr2 = int(round(n - (len(fix_pos) * fixed_value), 0))
    print(f"n-r={nr2}")
    for i in range(len(short_labels)):
        if i in fix_pos:
            alloc2[i] = fixed_value
            # print(f"{i}: {fixed_value}")
        else:
            alloc2[i] = round(weights[i] + additional_weight, 2) * nr2
            # print(f"{i}: {weights[i]} {weights[i] + additional_weight} \
            #       {weights[i] * nr1} {(weights[i] + additional_weight) * nr1}")
    sample_dic["Alloc2"] = alloc2

    alloc3 = np.zeros(len(short_labels), int)
    fixed_value = 50
    nr3 = int(round(n - (len(fix_pos) * fixed_value), 0))
    print(f"n-r={nr3}")
    for i in range(len(short_labels)):
        if i in fix_pos:
            alloc3[i] = fixed_value
            # print(f"{i}: {fixed_value}")
        else:
            alloc3[i] = round(weights[i] + additional_weight, 2) * nr3
            # print(f"{i}: {weights[i]} {weights[i] + additional_weight} \
            #       {weights[i] * nr1} {(weights[i] + additional_weight) * nr1}")
    sample_dic["Alloc3"] = alloc3

    prop = int(round(n,0))*weights
    prop = prop.astype(int)
    sample_dic["Prop"] = prop

    # The sample size table
    sample_table = pd.DataFrame(sample_dic)

    # Increase the sample size by a multiplier factor
    sample_factor = 10
    sample_table[["Equal", "Alloc1", "Alloc2", "Alloc3", "Prop"]] = sample_table[["Equal", "Alloc1", "Alloc2", "Alloc3", "Prop"]].mul(sample_factor)

    sample_table.loc['Total'] = sample_table.sum()
    sample_table.loc[sample_table.index[-1], 'Strata'] = ''
    print(sample_table)

    sample_table.to_csv(os.path.join(out_directory, f"sample_size_x{sample_factor}.csv"))

    # Select a sample size
    ref_sample_sizes = list(sample_table["Alloc2"] * sample_factor)
    ref_sample_sizes.pop() # remove the total row
    print(ref_sample_sizes)

    # Prepare the stratified random sampling
    fn_mask = f"sample_mask_x{sample_factor}.tif"
    fn_sampled = f"sampled_x{sample_factor}.csv"
    print(f"Out dir: {out_directory}")

    # # Run the sampling (takes time)
    # stratified_random_sampling(fn_reference, out_directory,
    #                            ref_val, ref_sample_sizes,
    #                            class_labels=short_labels,
    #                            sampling_mask=fn_mask,
    #                            sample_sizes=fn_sampled)

    # Read the samples created, the sample mask
    fn_sample_mask = os.path.join(out_directory, fn_mask)
    assert os.path.isfile(fn_sample_mask) is True, f"ERROR: File not found! {fn_sample_mask}"
    mask, mask_nd, mask_gt, mask_sr = rs.open_raster(fn_sample_mask)
    print(f'Opening raster: {fn_sample_mask}')
    print(f'  --NoData        : {mask_nd}')
    print(f'  --Columns       : {mask.shape[1]}')
    print(f'  --Rows          : {mask.shape[0]}')
    print(f'  --Geotransform  : {mask_gt}')
    print(f'  --Spatial ref.  : {mask_sr}')
    print(f'  --Type          : {mask.dtype}')

    print(f"Mask unique: {np.unique(mask)}")
    maskf = mask.filled(0)

    # Get the sample from the mask
    ref_sample = ref[maskf == 1]
    map_sample = map[maskf == 1]

    # Unique values in the samples
    print(f"Ref sample: {np.unique(ref_sample, return_counts=True)}")
    print(f"Map sample: {np.unique(map_sample, return_counts=True)}")

    # Further filter the sample, map_sample can have zeros (ref_sample cannot)
    if len(np.unique(map_sample)) > len(short_labels):
        map_sample2d = np.where(maskf==1, map, 0)
        new_mask = np.where(map_sample2d > 0, 1, 0)
        print(f"New mask: {new_mask.shape} {np.unique(new_mask, return_counts=True)}")
        
        ref_sample = ref[new_mask == 1]
        map_sample = map[new_mask == 1]
        print(f"Ref sample (new mask): {np.unique(ref_sample, return_counts=True)}")
        print(f"Map sample (new mask): {np.unique(map_sample, return_counts=True)}")

    # Confusion matrix, error matrix
    cm = confusion_matrix(ref_sample, map_sample)
    cm = np.array(cm)
    print("Confusion matrix:")
    print(cm)
    cm = cm.T  # transpose so rows=mapped (predicted), cols=reference (true)

    df_cm = pd.DataFrame(cm, index=short_labels, columns=short_labels)
    print("Error matrix of sample counts (pixels):")
    print(df_cm)

    # Convert to area, hectares
    # print(f"Area factor: {HA} (to hectares)")
    # df_cm = df_cm * HA

    # Add row totals and column totals
    df_cm["RowTotal"] = df_cm.sum(axis=1)

    # Add the map area
    df_cm["MapArea"] = map_cnt * HA
    df_cm["MapArea"] = df_cm["MapArea"].astype(int)
    weights = df_cm["MapArea"] / df_cm["MapArea"].sum()
    df_cm["W"] = weights

    col_totals = df_cm.sum(axis=0)
    col_totals.name = "ColTotal"

    # Append totals row (includes RowTotal value as total of all predictions)
    # df_cm = df_cm.append(col_totals)
    df_cm = pd.concat([df_cm, col_totals.to_frame().T])

    df_cm.to_csv(os.path.join(cwd, "map_accuracy", period, f"sample_cm_x{sample_factor}.csv"))
    print("Error matrix of sample counts:")
    print(df_cm)

    # # Estimated area proportions
    # print("Estimated area proportions: ")
    # df_p = pd.DataFrame(cm, index=short_labels, columns=short_labels)
    # row_total = df_p.sum(axis=1)

    # # Normalize by row
    # df_norm = df_p.div(row_total, axis=0)

    # # Multiply by the W
    # df_prop = df_norm.mul(weights, axis=0)

    # print(df_prop)
    
    # Estimates (OA, UA, PA), V, SE, CI, area
    
    print("Ice never dies!")
