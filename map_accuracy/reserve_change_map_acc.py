#!/usr/bin/env python
# coding: utf-8

""" Accuracy analysis for land cover change map on the Calakmul Biosphere Reserve
    Eduardo Jimenez Hernandez <eduardojh@arizona.edu>
"""

import gc
import sys
import os
import gc
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix

sys.path.insert(0, '/error/CODE/land_cover_analysis/lib/')
import rsmodule as rs

# from error_matrices import stratified_random_sampling

HA = (30 * 30) / 10000.

def get_sample_sizes(N: int, UA, weights: np.ndarray, ref_val: list, **kwargs):
    """ Calculates the sample sizes 

        Makes a proposal of fixed values at fixed positions for the smaller classes (i.e. if fixed value=75
        it will create samples at +/- 25 too)

        This is based on: Olofsson 2013, 2014
    """
    _SEO = kwargs.get("SEO", 0.01)
    _fixed_pos = kwargs.get("fixed_pos", [])
    _fixed_value = kwargs.get("fixed_value", 75)
    _fixed_step = kwargs.get("fixed_step", 25)
    _sample_factor = kwargs.get("sample_factor", 1)

    assert _sample_factor >= 1, "ERROR: sample factor < 1"
    assert _fixed_value >= 75, "ERROR: fixed value < 75"
    assert _fixed_step >= 25, "ERROR: fixed step < 25"
    assert len(weights) == len(ref_val), f"ERROR: list mismatch ref_val({len(ref_val)}), weights({len(weights)})"

    if (type(UA) is list) and (len(UA) != len(weights)):
        print(f"--ERROR: list mismatch ref_val({len(UA)}), weights({len(weights)})")
        return

    if (type(UA) is int) or (type(UA) is float):
        print(f"--Creating UA list with value={UA}")
        UA = [UA] * len(ref_val)

    WiSi_sum = 0
    WiSi2_sum = 0

    for i in range(len(ref_val)):
        Si = np.sqrt(UA[i] * (1-UA[i]))
        WiSi = weights[i] * Si
        WiSi2 = weights[i] * (Si * Si)
        print(f"--  {i}: Si={Si:0.4f} WiSi={WiSi:0.4f} WiSi2={WiSi2:0.4f}")
        WiSi_sum += WiSi
        WiSi2_sum += WiSi2
    n = (WiSi_sum / _SEO) * (WiSi_sum / _SEO)

    print("\n--Sample size for stratified random sampling:")
    print(f"--n={n}")
    n = (WiSi_sum  * WiSi_sum) / ((_SEO*_SEO) + (1/N) * WiSi2_sum)
    print(f"--n={n} (full equation)\n")

    sample_dic = {"Strata": [x for x in ref_val]}
    # Equal allocation
    sample_dic["Equal"] = [int(round(n / len(ref_val), 0))] * len(ref_val)

    # Proportional allocation
    prop = int(round(n, 0)) * weights
    prop = prop.astype(int)

    # alloc1: it will assign the fixed value + fixed step to the classes in fixed positions
    alloc1 = np.zeros(len(ref_val), int)
    # fixed_value = 100
    # fix_pos = [1, 2, 3, 4, 6, 7, 9, 10]
    _fixed_value += _fixed_step
    positions = np.where(prop < _fixed_value)[0]
    _fixed_pos = positions.tolist()
    print(f"--Assign fixed value: {_fixed_value}")
    weight_rem = sum([weights[j] for j in _fixed_pos])
    additional_weight = round(weight_rem / (len(ref_val) - len(_fixed_pos)), 2)
    # nr1 = int(round(n - (len(_fixed_pos) * _fixed_value), 0))
    nr1 = n
    r = len(_fixed_pos) * _fixed_value
    if r < n:
        nr1 = int(round(n - r, 0))
    print(f"--n-r={nr1}")
    for i in range(len(ref_val)):
        if i in _fixed_pos:
            alloc1[i] = _fixed_value
            # print(f"{i}: {_fixed_value}")
        else:
            alloc1[i] = round(weights[i] + additional_weight, 2) * nr1
            # print(f"{i}: {weights[i]} {weights[i] + additional_weight} \
            #       {weights[i] * nr1} {(weights[i] + additional_weight) * nr1}")
    sample_dic["Alloc1"] = alloc1

    # alloc1: it will assign the fixed value to the classes in fixed positions
    alloc2 = np.zeros(len(ref_val), int)
    # fixed_value = 75
    _fixed_value -= _fixed_step
    print(f"--Assign fixed value: {_fixed_value}")
    positions = np.where(prop < _fixed_value)[0]
    _fixed_pos = positions.tolist()
    nr2 = n
    r = len(_fixed_pos) * _fixed_value
    if r < n:
        nr2 = int(round(n - r, 0))
    print(f"--n-r={nr2}")
    for i in range(len(ref_val)):
        if i in _fixed_pos:
            alloc2[i] = _fixed_value
            # print(f"{i}: {_fixed_value}")
        else:
            alloc2[i] = round(weights[i] + additional_weight, 2) * nr2
            # print(f"{i}: {weights[i]} {weights[i] + additional_weight} \
            #       {weights[i] * nr1} {(weights[i] + additional_weight) * nr1}")
    sample_dic["Alloc2"] = alloc2

    # alloc1: it will assign the fixed value - fixed step to the classes in fixed positions
    alloc3 = np.zeros(len(ref_val), int)
    # fixed_value = 50
    _fixed_value -= _fixed_step
    print(f"--Assign fixed value: {_fixed_value}")
    positions = np.where(prop < _fixed_value)[0]
    _fixed_pos = positions.tolist()
    nr3 = n
    r = len(_fixed_pos) * _fixed_value
    if r < n:
        nr3 = int(round(n - r, 0))
    print(f"--n-r={nr3}")
    for i in range(len(ref_val)):
        if i in _fixed_pos:
            alloc3[i] = _fixed_value
            # print(f"{i}: {_fixed_value}")
        else:
            alloc3[i] = round(weights[i] + additional_weight, 2) * nr3
            # print(f"{i}: {weights[i]} {weights[i] + additional_weight} \
            #       {weights[i] * nr1} {(weights[i] + additional_weight) * nr1}")
    sample_dic["Alloc3"] = alloc3

    # Add the proportional allocation
    sample_dic["Prop"] = prop

    # The sample size table
    sample_table = pd.DataFrame(sample_dic)

    # Increase the sample size by a multiplier factor
    # sample_factor = 10
    if _sample_factor != 1:
        sample_table[["Equal", "Alloc1", "Alloc2", "Alloc3", "Prop"]] = sample_table[["Equal", "Alloc1", "Alloc2", "Alloc3", "Prop"]].mul(_sample_factor)

    sample_table.loc['Total'] = sample_table.sum()
    sample_table.loc[sample_table.index[-1], 'Strata'] = ''

    return sample_table
    

def create_change_labels(change_keys: list) -> list:
    """ Creates a string representation of change keys """
    key_labels = {"101": "AG",
                  "102": "UR",
                  "103": "WA",
                  "104": "BL",
                  "105": "MA",
                  "106": "EF",
                  "107": "SV",
                  "108": "WL",
                  "109": "DF",
                  "110": "CV",
                  "111": "OF"}
    
    change_keys_str = []
    for key in change_keys:
        key_str = str(int(key))
        # Get the alphanumeric key for each numeric key
        ini = key_labels.get(key_str[:3], "--")
        end = key_labels.get(key_str[3:], "--")

        # Create a change string
        change_str = f"{ini}->{end}"
        if ini == end:
            change_str = f"{ini}"
        
        # Add current change string to the list
        change_keys_str.append(change_str)
    
    return change_keys_str


def stratified_random_sampling_simple(fn_data, out_dir, list_classes, list_sample_size, **kwargs):
    """ Stratified random sampling 
    
    2026-03-13: Adapted to use sample sizes per each class instead of a constant percent.
    2026-05-05: Improvements on sampling a subset of the data 

    :param str fn_data: TIF file name to sample, contains classes (strata)
    :param str out_dir: directory to save results
    :param list list_classes: list of classes to sample, integer
    :param list list_sample_size: matching list with sample size for each class
    :param int window_size: default is sampling a window of 7x7 pixels
    :param int max_trials: max of attempts to fill the sample size
    """
 
    start = datetime.now()
    print(f"\n{start}: ========== Starting stratified random sampling ==========")

    assert os.path.isfile(fn_data) is True, f"ERROR: File not found! {fn_data}"
    assert len(list_classes) > 0, "ERROR: no classes (strata) to sample, list empty"
    assert len(list_classes) == len(list_sample_size), f"ERROR: list mismatch, {len(list_classes)} != {len(list_sample_size)}"
    assert len(list_classes) == len(list_sample_size), f"ERROR: list mismatch, {len(list_classes)} != {len(list_sample_size)}"

    # Keyword options
    _window_size = kwargs.get("window_size", 7)
    _max_trials = int(kwargs.get("max_trials", 2e5))
    fn_sample_mask = kwargs.get("sampling_mask", "sample_mask.tif")
    fn_sample_sizes = kwargs.get("sample_sizes", "sampled.csv")
    _class_labels = kwargs.get("class_labels", None)
    _raster_labels = kwargs.get("raster_labels", False)

    if _class_labels is not None:
        assert len(list_classes) == len(_class_labels), f"ERROR: list mismatch, {len(list_classes)} != {len(_class_labels)}"

    if not os.path.exists(out_dir):
        print(f"\nCreating new path: {out_dir}")
        os.makedirs(out_dir)
    
    fn_sample_mask = os.path.join(out_dir, fn_sample_mask)
    fn_sample_sizes = os.path.join(out_dir, fn_sample_sizes)

    # Read the raster and retrive the strata/classes
    data_arr, nodata, geotransform, spatial_reference = rs.open_raster(fn_data)
    print(f'  Opening raster: {fn_data}')
    print(f'    --NoData        : {nodata}')
    print(f'    --Columns       : {data_arr.shape[1]}')
    print(f'    --Rows          : {data_arr.shape[0]}')
    print(f'    --Geotransform  : {geotransform}')
    print(f'    --Spatial ref.  : {spatial_reference}')
    print(f'    --Type          : {data_arr.dtype}')

    data_arr = data_arr.filled(0)

    # Create a list of strata/class keys and its percentage of area covered
    print(f"Counting pixels per class from the data array...")
    arr_values, arr_counts = np.unique(data_arr, return_counts=True)
    df_arr = pd.DataFrame({"Keys": arr_values, "PixelCount": arr_counts})
    df_arr["Percent"] = df_arr["PixelCount"] / df_arr["PixelCount"].sum() * 100.0
    print("Complete array:")
    print(df_arr)

    # Sample sizes from the provided lists of classes and sample sizes
    # Filter the array values and counts with these lists in case they don't match
    df_strata = pd.DataFrame({"Keys": list_classes, "SampleSizePixels": list_sample_size})
    df_sample_sizes_sel = df_strata.merge(df_arr, on="Keys", how="left")
    df_sample_sizes_sel["Class"] = _class_labels
    print("Filtered array:")
    print(df_sample_sizes_sel)

    del df_strata
    del _class_labels
    del list_sample_size
    # del list_classes
    gc.collect()

    nrows, ncols = data_arr.shape
    arr_type = data_arr.dtype
    # print(f"  --Total pixels={nrows*ncols}, Values={sum(df['PixelCount'])}, NoData/Missing={nrows*ncols - sum(df['PixelCount'])}")
    print(f"  --Total pixels={nrows*ncols}, Values={sum(df_sample_sizes_sel['PixelCount'])}, NoData/Missing={nrows*ncols - sum(df_sample_sizes_sel['PixelCount'])}")

    sample = {}  # to save the sample

    # Create a mask of the sampled regions
    # sampling_mask = np.zeros(data_arr.shape, dtype=data_arr.dtype)
    sampling_mask = np.zeros((nrows, ncols), dtype=arr_type)

    # A window will be used for sampling, this array will hold the sample
    window_sample = np.zeros((_window_size,_window_size), dtype=int)

    print(f'  --Max trials: {_max_trials}')

    trials = 0  # attempts to complete the sample
    completed = {}  # classes which sample is complete

    for sample_key in list(df_sample_sizes_sel["Keys"]):
        completed[sample_key] = False
    completed_samples = sum(list(completed.values()))  # Values are all True if completed
    total_classes = len(completed.keys())
    print(completed)

    sampled_points = []

    while (trials < _max_trials and completed_samples < total_classes):
        show_progress = (trials%10000 == 0)  # Step to show progress
        # print(f'  --Trial {trials:>8} of {_max_trials:>8} ', end='')
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
            if show_progress:
                print("Point already sampled. Skipping.")
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
        window_sample = data_arr[win_row_ini:win_row_end,win_col_ini:win_col_end]
        
        # 5) Get the unique values in window sample (win_sample_keys) and its count (win_sample_counts)
        win_sample_keys, win_sample_counts = np.unique(window_sample, return_counts=True)

        # If only zeros skip
        if (win_sample_keys.size == 1) and (win_sample_keys[0] == 0):
            if show_progress:
                print(f"Sample window with only zeros. Skipping.")
            trials += 1
            continue

        # Filter the win_sample_keys by requested sample keys
        filtered_win_sample_keys = pd.DataFrame({"Keys": win_sample_keys, "WinCounts": win_sample_counts})
        filtered_win_sample_keys["Keys"] = filtered_win_sample_keys["Keys"].astype(int)
        filtered_win_sample_keys["WinCounts"] = filtered_win_sample_keys["WinCounts"].astype(int)
        if show_progress:
            print(f"Window sample:")
            print(filtered_win_sample_keys)

        # Keys in win_sample_keys that are not in list_classes
        keys_to_ignore = [x for x in list(win_sample_keys) if x not in list_classes]
        keys_to_ignore.append(0)

        df_strata = pd.DataFrame({"Keys": list_classes})
        df_strata["Keys"] = df_strata["Keys"].astype(int)
        filtered_win_sample_keys = df_strata.merge(filtered_win_sample_keys, on="Keys", how="left")
        if filtered_win_sample_keys.empty:
            if show_progress:
                print(f"filtered_win_sample_keys, empty. Skipping.")
            trials += 1
            continue
        # print(f"Window sample filtered:")
        # print(filtered_win_sample_keys)

        filtered_win_sample_keys = filtered_win_sample_keys.dropna()
        if filtered_win_sample_keys.empty:
            if show_progress:
                print(f"filtered_win_sample_keys, empty. Skipping.")
            trials += 1
            continue
        if show_progress:
            print(f"Window sample filtered by keys and no data:")
            print(filtered_win_sample_keys)

        # List of keys which sample is complete, val is True
        complete_samples = [key for key, val in completed.items() if val is True]
        if show_progress:
            print(f"Complete samples: {complete_samples}")

        # Filter out all the samples that are complete
        filtered_win_sample_keys = filtered_win_sample_keys[~filtered_win_sample_keys["Keys"].isin(complete_samples)].reset_index(drop=True)

        # Create an array containing all the sampled pixels
        sampled_window = np.zeros(window_sample.shape, dtype=arr_type)

        # 6) Iterate over "filtered_win_sample_keys" and add its respective pixel count to the sample
        for win_sample_key, win_sample_count in zip(filtered_win_sample_keys['Keys'], filtered_win_sample_keys['WinCounts']):

            # Accumulate the pixel counts, chek first if general sample is completed
            if sample.get(win_sample_key) is None:
                sample[win_sample_key] = win_sample_count
            else:
                sample[win_sample_key] += win_sample_count

                # Get the sample size
                sample_size = df_sample_sizes_sel.loc[df_sample_sizes_sel["Keys"] == win_sample_key, "SampleSizePixels"].iat[0]

                # Check if last addition completed the sample
                if sample[win_sample_key] >= sample_size:
                    completed[win_sample_key] = True  # this class' sample is now complete
                    keys_to_ignore.append(win_sample_key)  # ignore this class from now on in the mask creation
                
                # Add the current key to the sampled window mask
                sampled_window = np.where(window_sample == win_sample_key, 1, sampled_window)

        # Create an array containing al_sample.shape, dtype=arr_type)
        
        # Filter out keys with already complete samples or not in the list_classes
        # if len(keys_to_ignore) > 0:
        #     print(f"Keys to ignore: {keys_to_ignore}")
        #     mask = np.isin(window_sample, keys_to_ignore)
        #     sampled_window[mask] = 1
        # else:
        #     sampled_window = window_sample[:,:].astype(arr_type)

        # Slice and insert sampled window
        sampling_mask[win_row_ini:win_row_end,win_col_ini:win_col_end] += sampled_window.astype(arr_type)

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
    df_sample_sizes_sel['SampledPixels'] = [sample.get(x,0) for x in df_sample_sizes_sel['Keys']]
    df_sample_sizes_sel['SampledPercent'] = (df_sample_sizes_sel['SampledPixels'] / df_sample_sizes_sel['SampleSizePixels']) * 100
    df_sample_sizes_sel['SampledPerClass'] = (df_sample_sizes_sel['SampledPixels'] / df_sample_sizes_sel['PixelCount']) * 100
    df_sample_sizes_sel['SampleComplete'] = [completed[x] for x in df_sample_sizes_sel['Keys']]
    
    print(f"Sampled size table:")
    print(df_sample_sizes_sel)
    df_sample_sizes_sel.to_csv(fn_sample_sizes)
    print(f"Saved: {fn_sample_sizes}")

    # Convert the sampling_mask to 1's (indicating pixels to sample) and 0's
    sampling_mask = np.where(sampling_mask >= 1, 1, 0)
    print(f"  --Values in mask: {np.unique(sampling_mask)}")  # should be 1 and 0

    # Create a raster with actual labels (land cover classes) as pixel values
    if _raster_labels:
        fn_sample_mask_labels = fn_sample_mask[:-4] + "_labels.tif"
        training_labels = np.where(sampling_mask > 0, data_arr, 0)
        print(f"  --Creating raster: {fn_sample_mask_labels}")
        rs.create_raster(fn_sample_mask_labels, training_labels, spatial_reference, geotransform)

    # Create a raster with the sampled windows, this will be the sampling mask
    print(f"  --Creating raster: {fn_sample_mask}")
    rs.create_raster(fn_sample_mask, sampling_mask, spatial_reference, geotransform)

    end = datetime.now()
    print(f"{end}: ========== Stratified random sampling elapsed in: {end - start} ==========\n")


def stratified_random_sampling(fn_data, out_dir, list_classes, list_sample_size, **kwargs):
    """ Stratified random sampling 
    
    2026-03-13: Adapted to use sample sizes per each class instead of a constant percent.
    2026-05-05: Improvements on sampling a subset of the data
    2026-05-09: Improvements on sampling a window inside the data 

    :param str fn_data: TIF file name to sample, contains classes (strata)
    :param str out_dir: directory to save results
    :param list list_classes: list of classes to sample, integer
    :param list list_sample_size: matching list with sample size for each class
    :param int window_size: default is sampling a window of 7x7 pixels
    :param int max_trials: max of attempts to fill the sample size
    :param list win_offset: ini_row, ini_col, end_row, end_col to generate window sample
    """
 
    start = datetime.now()
    print(f"\n{start}: ========== Starting stratified random sampling ==========")

    assert os.path.isfile(fn_data) is True, f"ERROR: File not found! {fn_data}"
    assert len(list_classes) > 0, "ERROR: no classes (strata) to sample, list empty"
    assert len(list_classes) == len(list_sample_size), f"ERROR: list mismatch, {len(list_classes)} != {len(list_sample_size)}"
    assert len(list_classes) == len(list_sample_size), f"ERROR: list mismatch, {len(list_classes)} != {len(list_sample_size)}"

    # Keyword options
    _window_size = kwargs.get("window_size", 7)
    _max_trials = int(kwargs.get("max_trials", 2e5))
    fn_sample_mask = kwargs.get("sampling_mask", "sample_mask.tif")
    fn_sample_sizes = kwargs.get("sample_sizes", "sampled.csv")
    _class_labels = kwargs.get("class_labels", None)
    _raster_labels = kwargs.get("raster_labels", False)
    _win_offset = kwargs.get("win_offset", [None, None, None, None])

    assert len(_win_offset) == 4, "ERROR: window offset should have length 4"
    _ini_row, _ini_col, _end_row, _end_col = _win_offset
    if _ini_row is None:
        _ini_row = 0
    if _ini_col is None:
        _ini_col = 0


    if _class_labels is not None:
        assert len(list_classes) == len(_class_labels), f"ERROR: list mismatch, {len(list_classes)} != {len(_class_labels)}"

    if not os.path.exists(out_dir):
        print(f"\nCreating new path: {out_dir}")
        os.makedirs(out_dir)
    
    fn_sample_mask = os.path.join(out_dir, fn_sample_mask)
    fn_sample_sizes = os.path.join(out_dir, fn_sample_sizes)

    # Read the raster and retrive the strata/classes
    data_arr, nodata, geotransform, spatial_reference = rs.open_raster(fn_data)
    print(f'  Opening raster: {fn_data}')
    print(f'    --NoData        : {nodata}')
    print(f'    --Columns       : {data_arr.shape[1]}')
    print(f'    --Rows          : {data_arr.shape[0]}')
    print(f'    --Geotransform  : {geotransform}')
    print(f'    --Spatial ref.  : {spatial_reference}')
    print(f'    --Type          : {data_arr.dtype}')

    data_arr = data_arr.filled(0)

    # Create a list of strata/class keys and its percentage of area covered
    print(f"Counting pixels per class from the data array...")
    arr_values, arr_counts = np.unique(data_arr, return_counts=True)
    df_arr = pd.DataFrame({"Keys": arr_values, "PixelCount": arr_counts})
    df_arr["Percent"] = df_arr["PixelCount"] / df_arr["PixelCount"].sum() * 100.0
    print("Complete array:")
    print(df_arr)

    # Sample sizes from the provided lists of classes and sample sizes
    # Filter the array values and counts with these lists in case they don't match
    df_strata = pd.DataFrame({"Keys": list_classes, "SampleSizePixels": list_sample_size})
    df_sample_sizes_sel = df_strata.merge(df_arr, on="Keys", how="left")
    df_sample_sizes_sel["Class"] = _class_labels
    print("Filtered array:")
    print(df_sample_sizes_sel)

    del df_strata
    del _class_labels
    del list_sample_size
    gc.collect()

    nrows, ncols = data_arr.shape
    
    # Set the end row and col
    if _end_row is None:
        _end_row = nrows
    if _end_col is None:
        _end_col = ncols
    print(f"ini_row={_ini_row}, ini_col={_ini_col}, end_row={_end_row}, end_col={_end_col}")

    arr_type = data_arr.dtype
    print(f"  --Total pixels={nrows*ncols}, Values={sum(df_sample_sizes_sel['PixelCount'])}, NoData/Missing={nrows*ncols - sum(df_sample_sizes_sel['PixelCount'])}")

    sample = {}  # to save the sample

    # Create a mask of the sampled regions
    sampling_mask = np.zeros((nrows, ncols), dtype=arr_type)

    # A window will be used for sampling, this array will hold the sample
    window_sample = np.zeros((_window_size,_window_size), dtype=int)

    print(f'  --Max trials: {_max_trials}')

    trials = 0  # attempts to complete the sample
    completed = {}  # classes which sample is complete

    for sample_key in list(df_sample_sizes_sel["Keys"]):
        completed[sample_key] = False
    completed_samples = sum(list(completed.values()))  # Values are all True if completed
    total_classes = len(completed.keys())
    print(completed)

    sampled_points = []

    while (trials < _max_trials and completed_samples < total_classes):
        show_progress = (trials%10000 == 0)  # Step to show progress
        # print(f'  --Trial {trials:>8} of {_max_trials:>8} ', end='')
        if show_progress:
            print(f'  --Trial {1 if trials == 0 else trials:>8} of {_max_trials:>8} ', end='')

        # 1) Generate a random point (row_sample, col_sample) to sample the array
        #    Coordinates relative to array positions [0:nrows, 0:ncols]
        #    Subtract half the window_size to avoid sampling too close to the edges, use window_size step to avoid overlapping
        col_sample = random.randrange(_ini_col + _window_size//2, _end_col - _window_size//2, _window_size)
        row_sample = random.randrange(_ini_row + _window_size//2, _end_row - _window_size//2, _window_size)

        # Save the points previously sampled to avoid repeating and oversampling
        point = (row_sample, col_sample)
        if point in sampled_points:
            if show_progress:
                print("Point already sampled. Skipping.")
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
        window_sample = data_arr[win_row_ini:win_row_end,win_col_ini:win_col_end]
        
        # 5) Get the unique values in window sample (win_sample_keys) and its count (win_sample_counts)
        win_sample_keys, win_sample_counts = np.unique(window_sample, return_counts=True)

        # If only zeros skip
        if (win_sample_keys.size == 1) and (win_sample_keys[0] == 0):
            if show_progress:
                print(f"Sample window with only zeros. Skipping.")
            trials += 1
            continue

        # Filter the win_sample_keys by requested sample keys
        filtered_win_sample_keys = pd.DataFrame({"Keys": win_sample_keys, "WinCounts": win_sample_counts})
        filtered_win_sample_keys["Keys"] = filtered_win_sample_keys["Keys"].astype(int)
        filtered_win_sample_keys["WinCounts"] = filtered_win_sample_keys["WinCounts"].astype(int)
        if show_progress:
            print(f"Window sample:")
            print(filtered_win_sample_keys)

        # Keys in win_sample_keys that are not in list_classes
        keys_to_ignore = [x for x in list(win_sample_keys) if x not in list_classes]
        keys_to_ignore.append(0)

        df_strata = pd.DataFrame({"Keys": list_classes})
        df_strata["Keys"] = df_strata["Keys"].astype(int)
        filtered_win_sample_keys = df_strata.merge(filtered_win_sample_keys, on="Keys", how="left")
        if filtered_win_sample_keys.empty:
            if show_progress:
                print(f"filtered_win_sample_keys, empty. Skipping.")
            trials += 1
            continue
        # print(f"Window sample filtered:")
        # print(filtered_win_sample_keys)

        filtered_win_sample_keys = filtered_win_sample_keys.dropna()
        if filtered_win_sample_keys.empty:
            if show_progress:
                print(f"filtered_win_sample_keys, empty. Skipping.")
            trials += 1
            continue
        if show_progress:
            print(f"Window sample filtered by keys and no data:")
            print(filtered_win_sample_keys)

        # List of keys which sample is complete, val is True
        complete_samples = [key for key, val in completed.items() if val is True]
        if show_progress:
            print(f"Complete samples: {complete_samples}")

        # Filter out all the samples that are complete
        filtered_win_sample_keys = filtered_win_sample_keys[~filtered_win_sample_keys["Keys"].isin(complete_samples)].reset_index(drop=True)

        # Create an array containing all the sampled pixels
        sampled_window = np.zeros(window_sample.shape, dtype=arr_type)

        # 6) Iterate over "filtered_win_sample_keys" and add its respective pixel count to the sample
        for win_sample_key, win_sample_count in zip(filtered_win_sample_keys['Keys'], filtered_win_sample_keys['WinCounts']):

            # Accumulate the pixel counts, chek first if general sample is completed
            if sample.get(win_sample_key) is None:
                sample[win_sample_key] = win_sample_count
            else:
                sample[win_sample_key] += win_sample_count

                # Get the sample size
                sample_size = df_sample_sizes_sel.loc[df_sample_sizes_sel["Keys"] == win_sample_key, "SampleSizePixels"].iat[0]

                # Check if last addition completed the sample
                if sample[win_sample_key] >= sample_size:
                    completed[win_sample_key] = True  # this class' sample is now complete
                    keys_to_ignore.append(win_sample_key)  # ignore this class from now on in the mask creation
                
                # Add the current key to the sampled window mask
                sampled_window = np.where(window_sample == win_sample_key, 1, sampled_window)

        # Slice and insert sampled window
        sampling_mask[win_row_ini:win_row_end,win_col_ini:win_col_end] += sampled_window.astype(arr_type)

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
    df_sample_sizes_sel['SampledPixels'] = [sample.get(x,0) for x in df_sample_sizes_sel['Keys']]
    df_sample_sizes_sel['SampledPercent'] = (df_sample_sizes_sel['SampledPixels'] / df_sample_sizes_sel['SampleSizePixels']) * 100
    df_sample_sizes_sel['SampledPerClass'] = (df_sample_sizes_sel['SampledPixels'] / df_sample_sizes_sel['PixelCount']) * 100
    df_sample_sizes_sel['SampleComplete'] = [completed[x] for x in df_sample_sizes_sel['Keys']]
    
    print(f"Sampled size table:")
    print(df_sample_sizes_sel)
    df_sample_sizes_sel.to_csv(fn_sample_sizes)
    print(f"Saved: {fn_sample_sizes}")

    # Convert the sampling_mask to 1's (indicating pixels to sample) and 0's
    sampling_mask = np.where(sampling_mask >= 1, 1, 0)
    print(f"  --Values in mask: {np.unique(sampling_mask)}")  # should be 1 and 0

    # Create a raster with actual labels (land cover classes) as pixel values
    if _raster_labels:
        fn_sample_mask_labels = fn_sample_mask[:-4] + "_labels.tif"
        training_labels = np.where(sampling_mask > 0, data_arr, 0)
        print(f"  --Creating raster: {fn_sample_mask_labels}")
        rs.create_raster(fn_sample_mask_labels, training_labels, spatial_reference, geotransform)

    # Create a raster with the sampled windows, this will be the sampling mask
    print(f"  --Creating raster: {fn_sample_mask}")
    rs.create_raster(fn_sample_mask, sampling_mask, spatial_reference, geotransform)

    end = datetime.now()
    print(f"{end}: ========== Stratified random sampling elapsed in: {end - start} ==========\n")


if __name__ == "__main__":

    start = datetime.now()
    print(f"Execution starting at: {start}")

    # =================== Yucatan Peninsula Change Map Accuracy ======================= #

    # cwd = '/media/ecoslacker/RESEARCH/YUCATAN_LAND_COVER/ROI2/'
    cwd = '/error/YUCATAN_LAND_COVER/ROI2/'

    period1 = '2013_2016'
    fn_reference1 = os.path.join(cwd, period1, 'data/usv250s5ugw_grp11_ancillary.tif')
    fn_mapped1 = os.path.join(cwd, period1, 'results/2024_03_06-23_19_43', '2024_03_06-23_19_43_predictions.tif')

    period2 = '2019_2022'
    fn_reference2 = os.path.join(cwd, period2, 'data/usv250s7gw_grp11_ancillary.tif')
    fn_mapped2 = os.path.join(cwd, period2, 'results/2024_03_12-19_32_01', '2024_03_12-19_32_01_predictions.tif')

    # Use the CBR and similar-area-buffer rasters (already fixed to full extent)
    fn_mask_sim_area = '/error/2026/CBR/Raster/SIMAREA_full.tif'
    fn_mask_calakmul = '/error/2026/CBR/Raster/calakmul_full.tif'

    out_directory = os.path.join(cwd, "map_accuracy", "change_reserve")

    # # STEP1: Open all the rasters
    # assert os.path.isfile(fn_reference1) is True, f"ERROR: File not found! {fn_reference1}"
    # ref1, ref1_nd, ref1_gt, ref1_sr = rs.open_raster(fn_reference1)
    # print(f'Opening raster: {fn_reference1}')
    # print(f'  --NoData        : {ref1_nd}')
    # print(f'  --Columns       : {ref1.shape[1]}')
    # print(f'  --Rows          : {ref1.shape[0]}')
    # print(f'  --Geotransform  : {ref1_gt}')
    # print(f'  --Spatial ref.  : {ref1_sr}')
    # print(f'  --Type          : {ref1.dtype}')

    # assert os.path.isfile(fn_reference2) is True, f"ERROR: File not found! {fn_reference2}"
    # ref2, ref2_nd, ref2_gt, ref2_sr = rs.open_raster(fn_reference2)
    # print(f'Opening raster: {fn_reference2}')
    # print(f'  --NoData        : {ref2_nd}')
    # print(f'  --Columns       : {ref2.shape[1]}')
    # print(f'  --Rows          : {ref2.shape[0]}')
    # print(f'  --Geotransform  : {ref2_gt}')
    # print(f'  --Spatial ref.  : {ref2_sr}')
    # print(f'  --Type          : {ref2.dtype}')

    # assert os.path.isfile(fn_mapped1) is True, f"ERROR: File not found! {fn_mapped1}"
    # map1, map1_nd, map1_gt, map1_sr = rs.open_raster(fn_mapped1)
    # print(f'Opening raster: {fn_mapped1}')
    # print(f'  --NoData        : {map1_nd}')
    # print(f'  --Columns       : {map1.shape[1]}')
    # print(f'  --Rows          : {map1.shape[0]}')
    # print(f'  --Geotransform  : {map1_gt}')
    # print(f'  --Spatial ref.  : {map1_sr}')
    # print(f'  --Type          : {map1.dtype}')

    # assert os.path.isfile(fn_mapped2) is True, f"ERROR: File not found! {fn_mapped2}"
    # map2, map2_nd, map2_gt, map2_sr = rs.open_raster(fn_mapped2)
    # print(f'Opening raster: {fn_mapped2}')
    # print(f'  --NoData        : {map2_nd}')
    # print(f'  --Columns       : {map2.shape[1]}')
    # print(f'  --Rows          : {map2.shape[0]}')
    # print(f'  --Geotransform  : {map2_gt}')
    # print(f'  --Spatial ref.  : {map2_sr}')
    # print(f'  --Type          : {map2.dtype}')

    # assert os.path.isfile(fn_mask_sim_area) is True, f"ERROR: File not found! {fn_mask_sim_area}"
    # simarea, simarea_nd, simarea_gt, simarea_sr = rs.open_raster(fn_mask_sim_area)
    # print(f'Opening raster: {fn_mask_sim_area}')
    # print(f'  --NoData        : {simarea_nd}')
    # print(f'  --Columns       : {simarea.shape[1]}')
    # print(f'  --Rows          : {simarea.shape[0]}')
    # print(f'  --Geotransform  : {simarea_gt}')
    # print(f'  --Spatial ref.  : {simarea_sr}')
    # print(f'  --Type          : {simarea.dtype}')

    # print(f"Similar area unique values={np.unique(simarea)}")

    # assert os.path.isfile(fn_mask_calakmul) is True, f"ERROR: File not found! {fn_mask_calakmul}"
    # calakmul, calakmul_nd, calakmul_gt, calakmul_sr = rs.open_raster(fn_mask_calakmul)
    # print(f'Opening raster: {fn_mask_calakmul}')
    # print(f'  --NoData        : {calakmul_nd}')
    # print(f'  --Columns       : {calakmul.shape[1]}')
    # print(f'  --Rows          : {calakmul.shape[0]}')
    # print(f'  --Geotransform  : {calakmul_gt}')
    # print(f'  --Spatial ref.  : {calakmul_sr}')
    # print(f'  --Type          : {calakmul.dtype}')

    # print(f"CBR unique values={np.unique(calakmul)}")

    # print("Cleaning...")
    # del ref1_nd
    # del ref1_gt
    # del ref1_sr
    # del ref2_nd
    # del ref2_gt
    # del ref2_sr
    # del map2_nd
    # del map2_gt
    # del map2_sr
    # del simarea_nd
    # del simarea_gt
    # del simarea_sr
    # del calakmul_nd
    # del calakmul_gt
    # del calakmul_sr
    # gc.collect()

    # # Filter the maps to generate the change_maps
    # ref1_calakmul = np.where(calakmul.filled(0) > 0, ref1, 0)
    # ref2_calakmul = np.where(calakmul.filled(0) > 0, ref2, 0)
    # map1_calakmul = np.where(calakmul.filled(0) > 0, map1, 0)
    # map2_calakmul = np.where(calakmul.filled(0) > 0, map2, 0)

    # ref1_simarea = np.where(simarea.filled(0) > 0, ref1, 0)
    # ref2_simarea = np.where(simarea.filled(0) > 0, ref2, 0)
    # map1_simarea = np.where(simarea.filled(0) > 0, map1, 0)
    # map2_simarea = np.where(simarea.filled(0) > 0, map2, 0)

    # # ========== Create the change maps ==========
    # # e.g. 101102 means change from 101 to 102
    # ref_change_calakmul = (ref1_calakmul.astype(np.int32) * 1000) + ref2_calakmul
    # ref_change_simarea = (ref1_simarea.astype(np.int32) * 1000) + ref2_simarea
    # # Mapped change
    # map_change_calakmul = (map1_calakmul.astype(np.int32) * 1000) + map2_calakmul
    # map_change_simarea = (map1_simarea.astype(np.int32) * 1000) + map2_simarea

    # print("Cleaning...")
    # del ref1_calakmul
    # del ref2_calakmul
    # del map1_calakmul
    # del map2_calakmul
    # del ref1_simarea
    # del ref2_simarea
    # del map1_simarea
    # del map2_simarea
    # gc.collect()

    # Once the 'ref_change' and 'map_change' maps are created, save them
    fn_ref_change_calakmul = os.path.join(cwd, "map_accuracy", "change_reserve", "ref_change_calakmul.tif")
    fn_ref_change_simarea = os.path.join(cwd, "map_accuracy", "change_reserve", "ref_change_simarea.tif")
    fn_map_change_calakmul = os.path.join(cwd, "map_accuracy", "change_reserve", "map_change_calakmul.tif")
    fn_map_change_simarea = os.path.join(cwd, "map_accuracy", "change_reserve", "map_change_simarea.tif")

    # # Save them in raster files
    # rs.create_raster(fn_ref_change_calakmul, ref_change_calakmul, map1_sr, map1_gt, NoData=map1_nd, type='int32')
    # print(f"Saved: {fn_ref_change_calakmul}")
    # rs.create_raster(fn_ref_change_simarea, ref_change_simarea, map1_sr, map1_gt, NoData=map1_nd, type='int32')
    # print(f"Saved: {fn_ref_change_simarea}")
    # rs.create_raster(fn_map_change_calakmul, map_change_calakmul, map1_sr, map1_gt, NoData=map1_nd, type='int32')
    # print(f"Saved: {fn_map_change_calakmul}")
    # rs.create_raster(fn_map_change_simarea, map_change_simarea, map1_sr, map1_gt, NoData=map1_nd, type='int32')
    # print(f"Saved: {fn_map_change_simarea}")

    # STEP2: Once saved, just read the 'map_change' and 'ref_change' maps
    assert os.path.isfile(fn_ref_change_calakmul) is True, f"ERROR: File not found! {fn_ref_change_calakmul}"
    ref_change_calakmul, ref_change_calakmul_nd, ref_change_calakmul_gt, ref_change_calakmul_sr = rs.open_raster(fn_ref_change_calakmul)
    print(f'Opening raster: {fn_ref_change_calakmul}')

    assert os.path.isfile(fn_ref_change_simarea) is True, f"ERROR: File not found! {fn_ref_change_simarea}"
    ref_change_simarea, ref_change_simarea_nd, ref_change_simarea_gt, ref_change_simarea_sr = rs.open_raster(fn_ref_change_simarea)
    print(f'Opening raster: {fn_ref_change_simarea}')

    assert os.path.isfile(fn_map_change_calakmul) is True, f"ERROR: File not found! {fn_map_change_calakmul}"
    map_change_calakmul, map_change_calakmul_nd, map_change_calakmul_gt, map_change_calakmul_sr = rs.open_raster(fn_map_change_calakmul)
    print(f'Opening raster: {fn_map_change_calakmul}')

    assert os.path.isfile(fn_map_change_simarea) is True, f"ERROR: File not found! {fn_map_change_simarea}"
    map_change_simarea, map_change_simarea_nd, map_change_simarea_gt, map_change_simarea_sr = rs.open_raster(fn_map_change_simarea)
    print(f'Opening raster: {fn_map_change_simarea}')

    ref_change_calakmul = ref_change_calakmul.filled(0)
    ref_change_simarea = ref_change_simarea.filled(0)
    map_change_calakmul = map_change_calakmul.filled(0)
    map_change_simarea = map_change_simarea.filled(0)

    # mvals_calakmul, mcnts_calakmul = np.unique(map_change_calakmul, return_counts=True)
    # rvals_calakmul, rcnts_calakmul = np.unique(ref_change_calakmul, return_counts=True)
    # print(f"Calakmul Ref length={len(rvals_calakmul)}, Map length={len(mvals_calakmul)}")

    # mvals_simarea, mcnts_simarea = np.unique(map_change_simarea, return_counts=True)
    # rvals_simarea, rcnts_simarea = np.unique(ref_change_simarea, return_counts=True)
    # print(f"Sim Area Ref length={len(rvals_simarea)}, Map length={len(mvals_simarea)}")

    # # Merge the mvals and rvals, as they are common, then add their map and ref counts
    # s1_calakmul = pd.Series(mcnts_calakmul, index=mvals_calakmul, name="Map")
    # s2_calakmul = pd.Series(rcnts_calakmul, index=rvals_calakmul, name="Ref")

    # s1_simarea = pd.Series(mcnts_simarea, index=mvals_simarea, name="Map")
    # s2_simarea = pd.Series(rcnts_simarea, index=rvals_simarea, name="Ref")

    # # Combine on the union of labels (missing values remain NaN) for Calakmul
    # df_calakmul = pd.concat([s1_calakmul, s2_calakmul], axis=1).reset_index().rename(columns={"index": "Change"})
    # df_calakmul["Ref_Area_Percent"] = df_calakmul["Ref"] / df_calakmul["Ref"].sum() * 100
    # df_calakmul = df_calakmul.dropna()
    # df_calakmul = df_calakmul[df_calakmul["Change"] != 0]
    # df_calakmul = df_calakmul.reset_index(drop=True)
    # print(f"\nCounts from map and ref for Calakmul:")
    # print(df_calakmul)

    fn_change_pixel_counts_calakmul = os.path.join(out_directory, "change_pixel_counts_calakmul.csv")
    # df_calakmul.to_csv(fn_change_pixel_counts_calakmul)
    # print(f"Saved: {fn_change_pixel_counts_calakmul}")

    # # Combine on the union of labels (missing values remain NaN) for the Similar Area
    # df_simarea = pd.concat([s1_simarea, s2_simarea], axis=1).reset_index().rename(columns={"index": "Change"})
    # df_simarea["Ref_Area_Percent"] = df_simarea["Ref"] / df_simarea["Ref"].sum() * 100
    # df_simarea = df_simarea.dropna()
    # df_simarea = df_simarea[df_simarea["Change"] != 0]
    # df_simarea = df_simarea.reset_index(drop=True)
    # print(f"\nCounts from map and ref for Similar Area:")
    # print(df_simarea)

    fn_change_pixel_counts_simarea = os.path.join(out_directory, "change_pixel_counts_simarea.csv")
    # df_simarea.to_csv(fn_change_pixel_counts_simarea)
    # print(f"Saved: {fn_change_pixel_counts_simarea}")

    # ========== Prepare sample sizes for Calakmul ==========
    # total_samples = sum(rcnts_calakmul)
    # area_per = np.array(rcnts_calakmul) / total_samples
    # UA = 0.8
    sample_factor = 10
    fn_sample_size_calakmul = os.path.join(out_directory, f"sample_size_x{sample_factor}_calakmul.csv")
    fn_sample_size_simarea = os.path.join(out_directory, f"sample_size_x{sample_factor}_simarea.csv")

    # ********** Create the sample sizes ***************************************************************
    # sample_df_calakmul = get_sample_sizes(total_samples, UA, area_per, rvals_calakmul, sample_factor=sample_factor)
    # sample_df_calakmul = sample_df_calakmul[sample_df_calakmul["Strata"] != 0]
    # sample_df_calakmul.to_csv(fn_sample_size_calakmul)
    # # **************************************************************************************************

    # # ========== Prepare sample sizes for Similar Area ==========
    # total_samples = sum(rcnts_simarea)
    # area_per = np.array(rcnts_simarea) / total_samples

    # ********** Create the sample sizes ***************************************************************
    # sample_df_simarea = get_sample_sizes(total_samples, UA, area_per, rvals_simarea, sample_factor=sample_factor)
    # sample_df_simarea = sample_df_simarea[sample_df_simarea["Strata"] != 0]
    # sample_df_simarea.to_csv(fn_sample_size_simarea)
    # # **************************************************************************************************

    # STEP3: Open the pixel count data frames
    df_calakmul = pd.read_csv(fn_change_pixel_counts_calakmul, index_col=0)
    df_simarea = pd.read_csv(fn_change_pixel_counts_simarea, index_col=0)

    # Once the sample sizes are generated, read the files
    sample_df_calakmul = pd.read_csv(fn_sample_size_calakmul, index_col=0)
    print("Sample table Calakmul:")
    print(sample_df_calakmul)

    sample_df_simarea = pd.read_csv(fn_sample_size_simarea, index_col=0)
    print("Sample table Similar Area:")
    print(sample_df_simarea)

    # Select only a few changes (strata) of interest
    sel_changes = [101106, # AG -> EF, Gains
                   102106, # UR -> EF
                   109106, # DF -> EF
                   101109, # AG -> DF
                   102109, # UR -> DF
                   106109, # EF -> DF
                   106106, # No change EF
                   109109, # No change DF
                   106101, # EF -> AG, Loss
                   106102, # EF -> UR
                   109101, # DF -> AG
                   109102, # DF -> UR
                   101102, # AG -> UR
                   102101, # UR -> AG
                   102102, # No change UR
                   101101 # No change AG
                   ]
    
    # Filter the dataframe of pixel counts by the selected changes
    df_calakmul_filtered = df_calakmul[df_calakmul["Change"].isin(sel_changes)].reset_index(drop=True)
    print("Pixel counts for the selected changesin Calakmul:")
    print(df_calakmul_filtered)
    
    fn_change_pixel_counts_filter = os.path.join(out_directory, "change_pixel_counts_calakmul_filtered.csv")
    df_calakmul_filtered.to_csv(fn_change_pixel_counts_filter)
    print(f"Saved: {fn_change_pixel_counts_filter}")

    # Filter the dataframe of pixel counts by the selected changes
    df_simarea_filtered = df_simarea[df_simarea["Change"].isin(sel_changes)].reset_index(drop=True)
    print("Pixel counts for the selected changes in Similar Area:")
    print(df_simarea_filtered)
    
    fn_change_pixel_counts_filter = os.path.join(out_directory, "change_pixel_counts_simarea_filtered.csv")
    df_simarea_filtered.to_csv(fn_change_pixel_counts_filter)
    print(f"Saved: {fn_change_pixel_counts_filter}")

    # Create an array with the selected changes
    # ref_changes_sel = np.zeros(ref_change.shape)
    # for stratum in sel_changes:
    #     strat_changes = np.where(ref_change == stratum, ref_change, 0)
    #     ref_changes_sel += strat_changes
    #     # Keep only the selected changes
    #     # ref_changes_sel = np.where(ref_change == stratum, ref_change, ref_changes_sel)
    # r_v, r_c = np.unique(ref_changes_sel, return_counts=True)
    # r_v = r_v[1:]
    # r_c = r_c[1:]
    # for i, j in zip(r_v, r_c):
    #     print(f"{i} {j}")

    # df_filtered["RefVals"] = r_v
    # df_filtered["RefCounts"] = r_c
    # print("Pixel counts for the selected changes:")
    # print(df_filtered)
    # fn_change_pixel_counts_filter = os.path.join(out_directory, "change_pixel_counts_filtered.csv")
    # df_filtered.to_csv(fn_change_pixel_counts_filter)
    # print(f"Saved: {fn_change_pixel_counts_filter}")
    
    # # Save a TIF file with the selected changes only
    # fn_ref_changes_sel = os.path.join(out_directory, "ref_selected_changes.tif")
    # rs.create_raster(fn_ref_changes_sel, ref_changes_sel, map1_sr, map1_gt, NoData=map1_nd, type='int32')
    # print(f"Saved: {fn_ref_changes_sel}")

    # ========== Sample sizes for the selected changes ===========
    df_strata = pd.DataFrame({"Strata": sel_changes})
    df_sample_sizes_calakmul = df_strata.merge(sample_df_calakmul, on="Strata", how="left")
    df_sample_sizes_calakmul = df_sample_sizes_calakmul.dropna()
    df_sample_sizes_calakmul = df_sample_sizes_calakmul.reset_index(drop=True)
    # Get total columns
    df_sample_sizes_calakmul.loc['Total'] = df_sample_sizes_calakmul.sum()
    # Empty total for strata
    df_sample_sizes_calakmul.loc[df_sample_sizes_calakmul.index[-1], 'Strata'] = ''
    
    print("Sample sizes for the selected changes for Calakmul:")
    print(df_sample_sizes_calakmul)
    df_sample_sizes_calakmul.to_csv(os.path.join(out_directory, f"sample_size_x{sample_factor}_calakmul_selected.csv"))

    # Sample sizes, select Alloc1
    ref_sample_sizes_calakmul = df_sample_sizes_calakmul["Alloc1"].tolist()[:-1]
    ref_sample_keys_calakmul = df_sample_sizes_calakmul["Strata"].tolist()[:-1]
    # print(f"Sample sizes Alloc1 Calakmul: {ref_sample_sizes_calakmul}")

    # ========== Sample sizes for the selected changes ===========
    df_sample_sizes_simarea = df_strata.merge(sample_df_simarea, on="Strata", how="left")
    df_sample_sizes_simarea = df_sample_sizes_simarea.dropna()
    df_sample_sizes_simarea = df_sample_sizes_simarea.reset_index(drop=True)
    # Get total columns
    df_sample_sizes_simarea.loc['Total'] = df_sample_sizes_simarea.sum()
    # Empty total for strata
    df_sample_sizes_simarea.loc[df_sample_sizes_simarea.index[-1], 'Strata'] = ''
    
    print("Sample sizes for the selected changes for Calakmul:")
    print(df_sample_sizes_simarea)
    df_sample_sizes_simarea.to_csv(os.path.join(out_directory, f"sample_size_x{sample_factor}_simarea_selected.csv"))

    # Sample sizes, select Alloc1
    ref_sample_sizes_simarea = df_sample_sizes_simarea["Alloc1"].tolist()[:-1]
    ref_sample_keys_simarea = df_sample_sizes_simarea["Strata"].tolist()[:-1]
    # print(f"Sample sizes Alloc1 Similar Area: {ref_sample_sizes_simarea}")

    # Create change labels
    print(f"Selected changes labels for Calakmul:")
    sel_changes_lbls_calakmul = create_change_labels(ref_sample_keys_calakmul)
    for i, j in zip(ref_sample_keys_calakmul, sel_changes_lbls_calakmul):
        print(f"{i} {j}")
    # Create change labels
    print(f"Selected changes labels for Similar Area:")
    sel_changes_lbls_simarea = create_change_labels(ref_sample_keys_simarea)
    for i, j in zip(ref_sample_keys_simarea, sel_changes_lbls_simarea):
        print(f"{i} {j}")
    
    # print("Cleaning...")
    # del df_strata
    # del df_sample_sizes_sel
    # del ref_changes_sel
    # del map1_gt
    # del map1_sr
    # gc.collect()

    # Files to save
    fn_mask_calakmul = f"sample_mask_x{sample_factor}_calakmul.tif"
    fn_sampled_calakmul = f"sampled_x{sample_factor}_calakmul.csv"
    fn_mask_simarea = f"sample_mask_x{sample_factor}_simarea.tif"
    fn_sampled_simarea = f"sampled_x{sample_factor}_simarea.csv"
    
    assert len(ref_sample_keys_calakmul) == len(ref_sample_sizes_calakmul), f"ERROR: list mismatch ref_sample_keys_calakmul={len(ref_sample_keys_calakmul)}, ref_sample_sizes_calakmul={len(ref_sample_sizes_calakmul)}"
    assert len(ref_sample_keys_calakmul) == len(sel_changes_lbls_calakmul), f"ERROR: list mismatch ref_sample_keys_calakmul={len(ref_sample_keys_calakmul)}, sel_changes_lbls_calakmul={len(sel_changes_lbls_calakmul)}"
    assert len(ref_sample_keys_simarea) == len(ref_sample_sizes_simarea), f"ERROR: list mismatch ref_sample_keys_simarea={len(ref_sample_keys_simarea)}, ref_sample_sizes_simarea={len(ref_sample_sizes_simarea)}"
    assert len(ref_sample_keys_simarea) == len(sel_changes_lbls_simarea), f"ERROR: list mismatch ref_sample_keys_simarea={len(ref_sample_keys_simarea)}, sel_changes_lbls_simarea={len(sel_changes_lbls_simarea)}"

    # # Create a window offset to generate the random sample for the S
    # ini_row, ini_col, end_row, end_col = 13500, 12000, 18500, 15500
    # # *************************** Run the sampling (takes time) *********************************
    # stratified_random_sampling(fn_ref_change_calakmul,
    #                            out_directory,
    #                            ref_sample_keys_calakmul,
    #                            ref_sample_sizes_calakmul,
    #                            class_labels=sel_changes_lbls_calakmul,
    #                            sampling_mask=fn_mask_calakmul,
    #                            sample_sizes=fn_sampled_calakmul,
    #                            win_offset = [ini_row, ini_col, end_row, end_col],
    #                            max_trials=3e5)
    # # ********************************************************************************************

    # ini_row, ini_col, end_row, end_col = 11500, 9000, 21500, 17000
    # # *************************** Run the sampling (takes time) *********************************
    # stratified_random_sampling(fn_ref_change_simarea,
    #                            out_directory,
    #                            ref_sample_keys_simarea,
    #                            ref_sample_sizes_simarea,
    #                            class_labels=sel_changes_lbls_simarea,
    #                            sampling_mask=fn_mask_simarea,
    #                            sample_sizes=fn_sampled_simarea,
    #                            win_offset = [ini_row, ini_col, end_row, end_col],
    #                            max_trials=3e5)
    # # ********************************************************************************************

    # STEP4: Once the sample mask is created, read it 
    fn_sample_mask_calakmul = os.path.join(out_directory, fn_mask_calakmul)
    assert os.path.isfile(fn_sample_mask_calakmul) is True, f"ERROR: File not found! {fn_sample_mask_calakmul}"
    mask_calakmul, _, _, _ = rs.open_raster(fn_sample_mask_calakmul)
    print(f'Opening raster: {fn_sample_mask_calakmul}')

    fn_sample_mask_simarea = os.path.join(out_directory, fn_mask_simarea)
    assert os.path.isfile(fn_sample_mask_simarea) is True, f"ERROR: File not found! {fn_sample_mask_simarea}"
    mask_simarea, _, _, _ = rs.open_raster(fn_sample_mask_simarea)
    print(f'Opening raster: {fn_sample_mask_simarea}')


    print(f"Mask unique Calakmul: {np.unique(mask_calakmul)}")
    maskf_calakmul = mask_calakmul.filled(0)

    print(f"Mask unique Similar Area: {np.unique(mask_simarea)}")
    maskf_simarea = mask_simarea.filled(0)

    print("Cleaning...")
    del mask_calakmul
    del mask_simarea
    gc.collect()

    # Get the sample from the mask
    ref_sample_calakmul = ref_change_calakmul[maskf_calakmul == 1]
    map_sample_calakmul = map_change_calakmul[maskf_calakmul == 1]
    ref_sample_simarea = ref_change_simarea[maskf_simarea == 1]
    map_sample_simarea = map_change_simarea[maskf_simarea == 1]

    # ref_sample_calakmul = ref_sample_calakmul.filled(0)
    # map_sample_calakmul = map_sample_calakmul.filled(0)
    # ref_sample_simarea = ref_sample_simarea.filled(0)
    # map_sample_simarea = map_sample_simarea.filled(0)

    # Unique values in the samples
    print("Counting...")
    unique_ref_sample_calakmul = np.unique(ref_sample_calakmul, return_counts=True)
    unique_map_sample_calakmul = np.unique(map_sample_calakmul, return_counts=True)
    unique_ref_sample_simarea = np.unique(ref_sample_simarea, return_counts=True)
    unique_map_sample_simarea = np.unique(map_sample_simarea, return_counts=True)

    print(f"Ref sample Calakmul: {unique_ref_sample_calakmul} len={len(unique_ref_sample_calakmul[0])}")
    print(f"Map sample Calakmul: {unique_map_sample_calakmul} len={len(unique_map_sample_calakmul[0])}")
    print(f"Ref sample Similar Area: {unique_ref_sample_simarea} len={len(unique_ref_sample_simarea[0])}")
    print(f"Map sample Similar Area: {unique_map_sample_simarea} len={len(unique_map_sample_simarea[0])}")

    # Get the keys from the sampled files
    # df_sampled_calakmul = pd.read_csv(os.path.join(out_directory, fn_sampled_calakmul), index_col=0)
    # print(df_sampled_calakmul)

    # # Further filter the sample, map_sample can have zeros (ref_sample cannot)
    # if 0 in np.unique(map_sample):
    #     print("Zero value found, redoing mask...")
    #     map_sample2d = np.where(maskf==1, map_change, 0)
    #     new_mask = np.where(map_sample2d > 0, 1, 0)
    #     print(f"New mask: {new_mask.shape} {np.unique(new_mask, return_counts=True)}")
        
    #     ref_sample = ref_change[new_mask == 1]
    #     map_sample = map_change[new_mask == 1]

    #     unique_ref_sample = np.unique(ref_sample, return_counts=True)
    #     unique_map_sample = np.unique(map_sample, return_counts=True)

    #     print(f"Ref sample (new mask): {unique_ref_sample} len={len(unique_ref_sample[0])}")
    #     print(f"Map sample (new mask): {unique_map_sample} len={len(unique_map_sample[0])}")

    #     print("Cleaning...")
    #     del maskf
    #     del map_sample2d
    #     del new_mask
    #     gc.collect()

    vals_ref_calakmul = np.unique(unique_ref_sample_calakmul[0])
    vals_ref_simarea = np.unique(unique_ref_sample_simarea[0])

    # Create a mask with the values only in ref_sample
    mask_calakmul = np.isin(map_sample_calakmul, vals_ref_calakmul)
    print(f"ref_sample_calakmul values={len(ref_sample_calakmul)}, mask_calakmul={mask_calakmul.sum()}")
    mask_simarea = np.isin(map_sample_simarea, vals_ref_simarea)
    print(f"ref_sample_simarea values={len(ref_sample_simarea)}, mask_calakmul={mask_simarea.sum()}")

    # Filter both arrays to keep only matching positions with ref_sample
    ref_sample_calakmul_filtered = ref_sample_calakmul[mask_calakmul]
    map_sample_calakmul_filtered = map_sample_calakmul[mask_calakmul]
    ref_sample_simarea_filtered = ref_sample_simarea[mask_simarea]
    map_sample_simarea_filtered = map_sample_simarea[mask_simarea]

    ref_vals_calakmul, ref_count_calakmul = np.unique(ref_sample_calakmul_filtered, return_counts=True)
    map_vals_calakmul, map_count_calakmul = np.unique(map_sample_calakmul_filtered, return_counts=True)
    weights_calakmul = ref_count_calakmul / np.sum(ref_count_calakmul)

    ref_vals_simarea, ref_count_simarea = np.unique(ref_sample_simarea_filtered, return_counts=True)
    map_vals_simarea, map_count_simarea = np.unique(map_sample_simarea_filtered, return_counts=True)
    weights_simarea = ref_count_simarea / np.sum(ref_count_simarea)

    cm_labels_calakmul = create_change_labels(ref_vals_calakmul.tolist())
    cm_labels_simarea = create_change_labels(ref_vals_simarea.tolist())

    print("Labels for Calakmul")
    print(f"ref_vals ref_cnt  weights  map_vals map_cnt   labels")
    for i, j, k, l, m, n in zip(ref_vals_calakmul, ref_count_calakmul, weights_calakmul, map_vals_calakmul, map_count_calakmul, cm_labels_calakmul):
        print(f"{i:>8} {j:>8} {k:0.4f} {l:>8} {m:>8} {n}")

    print("Labels for Similar Area")
    print(f"ref_vals ref_cnt  weights  map_vals map_cnt   labels")
    for i, j, k, l, m, n in zip(ref_vals_simarea, ref_count_simarea, weights_simarea, map_vals_simarea, map_count_simarea, cm_labels_simarea):
        print(f"{i:>8} {j:>8} {k:0.4f} {l:>8} {m:>8} {n}")

    # Confusion matrix, error matrix
    cm_calakmul = confusion_matrix(ref_sample_calakmul_filtered, map_sample_calakmul_filtered)
    cm_simarea = confusion_matrix(ref_sample_simarea_filtered, map_sample_simarea_filtered)
    
    del vals_ref_calakmul
    del vals_ref_simarea
    del ref_sample_calakmul
    del ref_sample_simarea
    del map_sample_calakmul
    del map_sample_simarea
    del ref_sample_calakmul_filtered
    del ref_sample_simarea_filtered
    del map_sample_calakmul_filtered
    del map_sample_simarea_filtered
    gc.collect()

    cm_calakmul = np.array(cm_calakmul)
    cm_calakmul = cm_calakmul.T  # transpose so rows=mapped (predicted), cols=reference (true)
    print("Confusion matrix Calakmul:")
    print(cm_calakmul)

    cm_simarea = np.array(cm_simarea)
    cm_simarea = cm_simarea.T  # transpose so rows=mapped (predicted), cols=reference (true)
    print("Confusion matrix Similar Area:")
    print(cm_simarea)

    df_cm_calakmul = pd.DataFrame(cm_calakmul, index=cm_labels_calakmul, columns=cm_labels_calakmul)
    print("Error matrix of sample counts (pixels) for Calakmul:")
    print(df_cm_calakmul)

    df_cm_simarea = pd.DataFrame(cm_simarea, index=cm_labels_simarea, columns=cm_labels_simarea)
    print("Error matrix of sample counts (pixels) for Similar area:")
    print(df_cm_simarea)

    del cm_simarea
    del cm_calakmul
    gc.collect()

    # Add row totals and column totals
    df_cm_calakmul["RowTotal"] = df_cm_calakmul.sum(axis=1)
    df_cm_simarea["RowTotal"] = df_cm_simarea.sum(axis=1)

    # Get the map pixel count, not from the sample but actual change area
    ref_change_val_calakmul, ref_change_cnt_calakmul = np.unique(ref_change_calakmul, return_counts=True)
    print("Pixel count for Calakmul:")
    print(" Val          Count")
    for val, count in zip(ref_change_val_calakmul, ref_change_cnt_calakmul):
        print(f"{val:>7} {count:>12}")

    ref_change_val_simarea, ref_change_cnt_simarea = np.unique(ref_change_simarea, return_counts=True)
    print("Pixel count for Similar Area:")
    print(" Val          Count")
    for val, count in zip(ref_change_val_simarea, ref_change_cnt_simarea):
        print(f"{val:>7} {count:>12}")
    
    # Remove the 0 entries
    ref_change_val_calakmul = ref_change_val_calakmul[1:]
    ref_change_cnt_calakmul = ref_change_cnt_calakmul[1:]
    ref_change_val_simarea = ref_change_val_simarea[1:]
    ref_change_cnt_simarea = ref_change_cnt_simarea[1:]
    
    # Create a list of the pixel counts for the selected changes
    count_sel_calakmul = []
    for change in ref_vals_calakmul:
        for val, count in zip(ref_change_val_calakmul, ref_change_cnt_calakmul):
            if change == val:
                count_sel_calakmul.append(count)
                print(f"--Adding {count} ({val})")
                break
    count_sel_calakmul = np.array(count_sel_calakmul)
    print(f"Pixel count sel: {count_sel_calakmul} {len(count_sel_calakmul)}")
    print(df_cm_calakmul.shape)

    count_sel_simarea = []
    for change in ref_vals_simarea:
        for val, count in zip(ref_change_val_simarea, ref_change_cnt_simarea):
            if change == val:
                count_sel_simarea.append(count)
                print(f"--Adding {count} ({val})")
                break
    count_sel_simarea = np.array(count_sel_simarea)
    print(f"Pixel count sel: {count_sel_simarea} {len(count_sel_simarea)}")
    print(df_cm_calakmul.shape)

    # Add the map area
    df_cm_calakmul["MapArea"] = count_sel_calakmul * HA
    df_cm_calakmul["MapArea"] = df_cm_calakmul["MapArea"].astype(int)
    df_cm_calakmul["W"] = weights_calakmul
    df_cm_simarea["MapArea"] = count_sel_simarea * HA
    df_cm_simarea["MapArea"] = df_cm_simarea["MapArea"].astype(int)
    df_cm_simarea["W"] = weights_simarea

    col_totals_calakmul = df_cm_calakmul.sum(axis=0)
    col_totals_calakmul.name = "ColTotal"
    col_totals_simarea = df_cm_simarea.sum(axis=0)
    col_totals_simarea.name = "ColTotal"

    # Append totals row (includes RowTotal value as total of all predictions)
    df_cm_calakmul = pd.concat([df_cm_calakmul, col_totals_calakmul.to_frame().T])
    df_cm_simarea = pd.concat([df_cm_simarea, col_totals_simarea.to_frame().T])

    fn_cm_calakmul = os.path.join(out_directory, f"sample_cm_x{sample_factor}_calakmul.csv")
    df_cm_calakmul.to_csv(fn_cm_calakmul)
    print(f"Saved: {fn_cm_calakmul}")
    print("Error matrix of sample counts for Calakmul:")
    print(df_cm_calakmul)

    fn_cm_simarea = os.path.join(out_directory, f"sample_cm_x{sample_factor}_simarea.csv")
    df_cm_simarea.to_csv(fn_cm_simarea)
    print(f"Saved: {fn_cm_simarea}")
    print("Error matrix of sample counts for Similar Area:")
    print(df_cm_simarea)

    print("Ice never dies!")
