#!/usr/bin/env python
# coding: utf-8

""" Accuracy analysis for land cover change map
    Eduardo Jimenez Hernandez <eduardojh@arizona.edu>
"""

import sys
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import chi2, chi2_contingency

sys.path.insert(0, '/error/CODE/land_cover_analysis/lib/')

import rsmodule as rs

def get_sample_sizes(N: int, UA, weights: list, ref_val: list, short_labels: list, **kwargs):
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
    assert len(short_labels) == len(ref_val), f"ERROR: list mismatch ref_val({len(ref_val)}), short_labels({len(short_labels)})"

    if (type(UA) is list) and (len(UA) != len(weights)):
        print(f"ERROR: list mismatch ref_val({len(UA)}), weights({len(weights)})")
        return

    if type(UA) is int:
        print(f"Creating UA list with value={UA}")
        UA = [UA] * len(ref_val)

    WiSi_sum = 0
    WiSi2_sum = 0

    for i in range(len(ref_val)):
        Si = np.sqrt(UA[i] * (1-UA[i]))
        WiSi = weights[i] * Si
        WiSi2 = weights[i] * (Si * Si)
        print(f"{i}: Si={Si:0.4f} WiSi={WiSi:0.4f} WiSi2={WiSi2:0.4f}")
        WiSi_sum += WiSi
        WiSi2_sum += WiSi2
    n = (WiSi_sum / _SEO) * (WiSi_sum / _SEO)

    print("1. Sample size for stratified random sampling:")
    print(f"n={n}\n")
    n = (WiSi_sum  * WiSi_sum) / ((_SEO*_SEO) + (1/N) * WiSi2_sum)
    print(f"n={n} (full equation)\n")

    sample_dic = {"Strata": [x for x in short_labels]}
    sample_dic["Equal"] = [int(round(n / len(short_labels), 0))] * len(short_labels)

    # alloc1: it will assign the fixed value + fixed step to the classes in fixed positions
    alloc1 = np.zeros(len(short_labels), int)
    # fixed_value = 100
    # fix_pos = [1, 2, 3, 4, 6, 7, 9, 10]
    _fixed_value += _fixed_step
    weight_rem = sum([weights[j] for j in _fixed_pos])
    additional_weight = round(weight_rem / (len(short_labels) - len(_fixed_pos)), 2)
    nr1 = int(round(n - (len(_fixed_pos) * _fixed_value), 0))
    print(f"n-r={nr1}")
    for i in range(len(short_labels)):
        if i in _fixed_pos:
            alloc1[i] = _fixed_value
            # print(f"{i}: {_fixed_value}")
        else:
            alloc1[i] = round(weights[i] + additional_weight, 2) * nr1
            # print(f"{i}: {weights[i]} {weights[i] + additional_weight} \
            #       {weights[i] * nr1} {(weights[i] + additional_weight) * nr1}")
    sample_dic["Alloc1"] = alloc1

    # alloc1: it will assign the fixed value to the classes in fixed positions
    alloc2 = np.zeros(len(short_labels), int)
    # fixed_value = 75
    _fixed_value -= _fixed_step
    nr2 = int(round(n - (len(_fixed_pos) * _fixed_value), 0))
    print(f"n-r={nr2}")
    for i in range(len(short_labels)):
        if i in _fixed_pos:
            alloc2[i] = _fixed_value
            # print(f"{i}: {_fixed_value}")
        else:
            alloc2[i] = round(weights[i] + additional_weight, 2) * nr2
            # print(f"{i}: {weights[i]} {weights[i] + additional_weight} \
            #       {weights[i] * nr1} {(weights[i] + additional_weight) * nr1}")
    sample_dic["Alloc2"] = alloc2

    # alloc1: it will assign the fixed value - fixed step to the classes in fixed positions
    alloc3 = np.zeros(len(short_labels), int)
    # fixed_value = 50
    _fixed_value -= _fixed_step
    nr3 = int(round(n - (len(_fixed_pos) * _fixed_value), 0))
    print(f"n-r={nr3}")
    for i in range(len(short_labels)):
        if i in _fixed_pos:
            alloc3[i] = _fixed_value
            # print(f"{i}: {_fixed_value}")
        else:
            alloc3[i] = round(weights[i] + additional_weight, 2) * nr3
            # print(f"{i}: {weights[i]} {weights[i] + additional_weight} \
            #       {weights[i] * nr1} {(weights[i] + additional_weight) * nr1}")
    sample_dic["Alloc3"] = alloc3

    prop = int(round(n, 0)) * weights
    prop = prop.astype(int)
    sample_dic["Prop"] = prop

    # The sample size table
    sample_table = pd.DataFrame(sample_dic)

    # Increase the sample size by a multiplier factor
    # sample_factor = 10
    if _sample_factor != 1:
        sample_table[["Equal", "Alloc1", "Alloc2", "Alloc3", "Prop"]] = sample_table[["Equal", "Alloc1", "Alloc2", "Alloc3", "Prop"]].mul(_sample_factor)

    sample_table.loc['Total'] = sample_table.sum()
    sample_table.loc[sample_table.index[-1], 'Strata'] = ''
    print(sample_table)

    sample_table.to_csv(os.path.join(out_directory, f"sample_size_x{_sample_factor}.csv"))

    # Select a sample size
    ref_sample_sizes = list(sample_table["Alloc2"] * _sample_factor)
    ref_sample_sizes.pop() # remove the total row
    print(ref_sample_sizes)


if __name__ == "__main__":

    start = datetime.now()
    print(f"Execution starting at: {start}")

    # =================== Yucatan Peninsula Change Map Accuracy ======================= #

    cwd = '/media/ecoslacker/RESEARCH/YUCATAN_LAND_COVER/ROI2/'

    period1 = '2013_2016'
    fn_reference1 = os.path.join(cwd, period1, 'data/usv250s5ugw_grp11_ancillary.tif')
    fn_mapped1 = os.path.join(cwd, period1, 'results/2024_03_06-23_19_43', '2024_03_06-23_19_43_predictions.tif')

    period2 = '2019_2022'
    fn_reference2 = os.path.join(cwd, period2, 'data/usv250s7gw_grp11_ancillary.tif')
    fn_mapped2 = os.path.join(cwd, period2, 'results/2024_03_12-19_32_01', '2024_03_12-19_32_01_predictions.tif')

    out_directory = os.path.join(cwd, "map_accuracy", "change")

    # Open all the rasters
    assert os.path.isfile(fn_reference1) is True, f"ERROR: File not found! {fn_reference1}"
    ref1, ref1_nd, ref1_gt, ref1_sr = rs.open_raster(fn_reference1)
    print(f'Opening raster: {fn_reference1}')
    print(f'  --NoData        : {ref1_nd}')
    print(f'  --Columns       : {ref1.shape[1]}')
    print(f'  --Rows          : {ref1.shape[0]}')
    print(f'  --Geotransform  : {ref1_gt}')
    print(f'  --Spatial ref.  : {ref1_sr}')
    print(f'  --Type          : {ref1.dtype}')

    assert os.path.isfile(fn_reference2) is True, f"ERROR: File not found! {fn_reference2}"
    ref2, ref2_nd, ref2_gt, ref2_sr = rs.open_raster(fn_reference2)
    print(f'Opening raster: {fn_reference2}')
    print(f'  --NoData        : {ref2_nd}')
    print(f'  --Columns       : {ref2.shape[1]}')
    print(f'  --Rows          : {ref2.shape[0]}')
    print(f'  --Geotransform  : {ref2_gt}')
    print(f'  --Spatial ref.  : {ref2_sr}')
    print(f'  --Type          : {ref2.dtype}')

    assert os.path.isfile(fn_mapped1) is True, f"ERROR: File not found! {fn_mapped1}"
    map1, map1_nd, map1_gt, map1_sr = rs.open_raster(fn_mapped1)
    print(f'Opening raster: {fn_mapped1}')
    print(f'  --NoData        : {map1_nd}')
    print(f'  --Columns       : {map1.shape[1]}')
    print(f'  --Rows          : {map1.shape[0]}')
    print(f'  --Geotransform  : {map1_gt}')
    print(f'  --Spatial ref.  : {map1_sr}')
    print(f'  --Type          : {map1.dtype}')

    assert os.path.isfile(fn_mapped2) is True, f"ERROR: File not found! {fn_mapped2}"
    map2, map2_nd, map2_gt, map2_sr = rs.open_raster(fn_mapped2)
    print(f'Opening raster: {fn_mapped2}')
    print(f'  --NoData        : {map2_nd}')
    print(f'  --Columns       : {map2.shape[1]}')
    print(f'  --Rows          : {map2.shape[0]}')
    print(f'  --Geotransform  : {map2_gt}')
    print(f'  --Spatial ref.  : {map2_sr}')
    print(f'  --Type          : {map2.dtype}')

    # Create a new mask from all masks, use minimum 
    ref1 = ref1.filled(0)
    ref2 = ref2.filled(0)
    map1 = map1.filled(0)
    map2 = map2.filled(0)

    ref1_mask = np.where(ref1 > 0, 1, 0)
    ref2_mask = np.where(ref2 > 0, 1, 0)
    map1_mask = np.where(map1 > 0, 1, 0)
    map2_mask = np.where(map2 > 0, 1, 0)

    # Mask will be pixels where all 4 datasets have data
    mask_sum = ref1_mask + ref2_mask + map1_mask + map2_mask
    del(ref1_mask)
    del(ref2_mask)
    del(map1_mask)
    del(map2_mask)
    print(np.unique(mask_sum, return_counts=True))

    mask_all = np.where(mask_sum==4, False, True)
    del(mask_sum)
    gc.collect()

    fn_mask = os.path.join(cwd, "map_accuracy", "change", "common_mask.tif")
    rs.create_raster(fn_mask, np.logical_not(mask_all), map1_sr, map1_gt, NoData=map1_nd)
    print(f"Saved: {fn_mask}")

    # Apply the mask
    map1 = np.ma.masked_array(map1, mask=mask_all)
    map2 = np.ma.masked_array(map2, mask=mask_all)
    ref1 = np.ma.masked_array(ref1, mask=mask_all)
    ref2 = np.ma.masked_array(ref2, mask=mask_all)

    ref1 = ref1.filled(0).astype(np.int32)
    ref2 = ref2.filled(0).astype(np.int32)
    map1 = map1.filled(0).astype(np.int32)
    map2 = map2.filled(0).astype(np.int32)

    print(f"map1:")
    values1, counts1 = np.unique(map1, return_counts=True)
    for v, c in zip(values1, counts1):
        print(f" {v}\t{c}")

    print(f"map2:")
    values2, counts2 = np.unique(map2, return_counts=True)
    for v, c in zip(values2, counts2):
        print(f" {v}\t{c}")

    print(f"ref1:")
    values3, counts3 = np.unique(ref1, return_counts=True)
    for v, c in zip(values3, counts3):
        print(f" {v}\t{c}")

    print(f"ref2:")
    values4, counts4 = np.unique(ref2, return_counts=True)
    for v, c in zip(values4, counts4):
        print(f" {v}\t{c}")

    # ref_mask = np.ma.getmask(ref1)
    # ref = ref.filled(0)
    # map = map.filled(0)
    # map = np.ma.masked_array(map, mask=ref_mask)
    # map = map.filled(0)

    # Create the reference change map from the two reference maps
    # Change format will be 101102 where P1 class is 101 and it changed to 102 in P2
    ref_change = (ref1 * 1000) + ref2
    # Mapped change
    map_change = (map1 * 1000) + map2

    fn_ref_change = os.path.join(cwd, "map_accuracy", "change", "ref_change.tif")
    rs.create_raster(fn_ref_change, ref_change, map1_sr, map1_gt, NoData=map1_nd, type='int32')
    print(f"Saved: {fn_ref_change}")

    fn_map_change = os.path.join(cwd, "map_accuracy", "change", "map_change.tif")
    rs.create_raster(fn_map_change, map_change, map1_sr, map1_gt, NoData=map1_nd, type='int32')
    print(f"Saved: {fn_map_change}")

    print(f"refvals refcnt mapvals mapcnt")
    mvals, mcnts = np.unique(map_change, return_counts=True)
    rvals, rcnts = np.unique(ref_change, return_counts=True)
    for rv, rc, mv, mc in zip(rvals, rcnts, mvals, mcnts):
        print(f"{rv}\t{rc}\t{mv}\t{mc}")
    print(len(mvals), len(rvals))

    # Remove the zeros
    rvals = rvals[1:]
    rcnts = rcnts[1:]

    # Prepare sample sizes
    total_samples = sum(rcnts)
    area_per = np.array(rcnts) / total_samples
    for c, a in zip(rcnts, area_per):
        print(f"{c}\t{a}")
    UA = 0.8
    short_labels = ["AG", "UR", "WA", "BL", "MA", "EF", "SV", "WL", "DF", "CV", "OF"]
    sample_sizes = get_sample_sizes(total_samples, UA, area_per, rvals, short_labels, sample_factor=10)

    print("Ice never dies!")