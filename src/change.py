#!/usr/bin/env python
# coding: utf-8

""" Land cover change analysis
"""

import sys
import os
import numpy as np

sys.path.insert(0, '/home/ecoslacker/Documents/land_cover_analysis/lib/')

import rsmodule as rs

def change(filename1, filename2, outdir, label, **kwargs):
    
    """ Change between two 2D datasets or land cover maps """
    print(f"=== CHANGE BETWEEN: {filename1} AND {filename2}")

    # Create output directory if it doesn't exists
    if not os.path.exists(outdir):
        print(f"Creating path for output: {outdir}")
        os.makedirs(outdir)

    # Open the files
    assert os.path.isfile(filename1), f"ERROR: {filename1} not found!"
    assert os.path.isfile(filename2), f"ERROR: {filename2} not found!"
    ds1, nodata1, geotransform1, spatial_ref1 = rs.open_raster(filename1)
    ds2, nodata2, geotransform2, spatial_ref2 = rs.open_raster(filename2)

    # Get the differences
    print(f"Calculating differences...")
    diff = np.where(ds1 == ds2, 0, 1)

    # Calculate the change factors (kappa)

    # Plot the differences
    fn_plot_diff = os.path.join(outdir, f"{label}_diff.png")
    print(f"Saving plot: {fn_plot_diff}...")
    rs.plot_diff(ds1, ds2, diff, savefig=fn_plot_diff)

    # Write the outputs   


if __name__ == '__main__':

    cwd = '/run/media/ecoslacker/Seagate Portable Drive/Backup_2023-11-24/ROI2/'
    results_dir = "/home/ecoslacker/Downloads/ROI2/change/"

    periods = {'2013-2016': 'results/2023_10_28-01_04_42/2023_10_28-01_04_42_predictions_roi.tif',
               '2016-2019': 'results/2023_10_28-18_19_05/2023_10_28-18_19_05_predictions.tif',
               '2029-2022': 'results/2023_10_29-12_10_07/2023_10_29-12_10_07_predictions.tif'}
    
    # Land cover change analysis between periods 1 and 2
    label_periods = '2013-2016_to_2016-2019'
    fn1 = os.path.join(cwd, '2013_2016', periods['2013-2016'])
    fn2 = os.path.join(cwd, '2016_2019', periods['2016-2019'])

    change(fn1, fn2, results_dir, label_periods)

    print("All done. ;-)")