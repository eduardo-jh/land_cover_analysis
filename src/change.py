#!/usr/bin/env python
# coding: utf-8

""" Land cover change analysis
"""

import sys
import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2, chi2_contingency

# sys.path.insert(0, '/home/ecoslacker/Documents/land_cover_analysis/lib/')
sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs

def change2ds(filename1, filename2, outdir, label, **kwargs):
    
    """ Change between two 2D datasets or land cover maps """
    print(f"=== CHANGE BETWEEN: {filename1} AND {filename2}")

    # Create output directory if it doesn't exists
    if not os.path.exists(outdir):
        print(f"Creating path for output: {outdir}")
        os.makedirs(outdir)

    # File names
    fn_df_csv = os.path.join(outdir, f"{label}_table.csv")
    fn_report_txt = os.path.join(outdir, f"{label}_report.txt")
    fn_plot_diff = os.path.join(outdir, f"{label}_diff.png")
    fn_plot_diff_only = os.path.join(outdir, f"{label}_diff_only.png")
    fn_raster = os.path.join(outdir, f"{label}_diff.tif")

    # Open the files
    assert os.path.isfile(filename1), f"ERROR: {filename1} not found!"
    assert os.path.isfile(filename2), f"ERROR: {filename2} not found!"
    ds1, _, geotransform, spatial_ref = rs.open_raster(filename1)
    ds2, _, _, _ = rs.open_raster(filename2)

    # Get the differences
    print(f"Calculating pixels with differences...")
    ds_mask = np.ma.getmask(ds1)
    diff = np.where(ds1 == ds2, 0, 1)
    diff = np.ma.masked_array(diff, mask=ds_mask)
    unmasked_pixels = ds_mask.size - np.sum(ds_mask)
    change_pixels = np.sum(diff)
    print(f"Change pixels={change_pixels}, out of {unmasked_pixels} ({change_pixels/unmasked_pixels*100:0.2f}%)")

    # Calculate the change factors
    vals1, counts1 = np.unique(ds1, return_counts=True)
    vals2, counts2 = np.unique(ds2, return_counts=True)
    # Remove the np.ma.core.MaskedConstant
    vals1 = np.delete(vals1, len(vals1)-1)
    counts1 = np.delete(counts1, len(counts1)-1)
    vals2= np.delete(vals2, len(vals2)-1)
    counts2 = np.delete(counts2, len(counts2)-1)

    zipped = list(zip(vals1, counts1, vals2))
    
    # Create a dataframe with the stats
    # Frequency for Classes 2 are the observed
    df = pd.DataFrame(zipped, columns=['Classes1', 'Freq1', 'Classes2'])
    df['Fraction'] = counts1/sum(counts1)
    df['Expected'] = df['Fraction'] * sum(counts2)
    df['Expected'] = df['Expected'].astype(int)
    df['Observed'] = counts2
    df['Chi-square'] = np.square(df['Observed'] - df['Expected']) / df['Expected']
    # Save to csv file
    print(f"Saving dataframe to: {fn_df_csv}")
    df.to_csv(fn_df_csv)

    # Use the chi-square test
    chi_square = sum(df['Chi-square'])
    dof = len(df['Expected'])-1 # dof=(r-1)(c-1) = (11-1)(2-1)=10
    # Cramer's V
    n = sum(counts1) + sum(counts2)  # sample size
    m = min(len(df['Expected'])-1, 1) # min of (r-1) and (c-1)
    cramers_v = np.sqrt(chi_square / (n * m))
    alpha = 0.05
    chi2_stat = chi2.ppf(1-alpha, dof)

    print(f'\nChi-square analysis: {label}')
    print(df)
    print(f"Chi-square={chi_square}, degrees of freedom={dof}, n={n}, Cramer's V={cramers_v}")
    print(f"Chi2 stat: {chi2_stat}")

    # Use scipy
    print("Contingency analysis (scipy)...")
    data = [df['Observed'], df['Expected']]
    stat, p, dof2, expected = chi2_contingency(data)

    print(f"{stat}, {p}, {dof2}, {expected}")

    # interpret p-value
    conclusion = ''
    if p <= alpha:
        conclusion = 'reject H0 (maps are different)'
        print(f"p-value={p}: {conclusion}")
    else:
        conclusion = 'fail to reject H0 (maps are similar)'
        print(f"p-value={p}: {conclusion}")

    # Save a text file with the procedure
    with open(fn_report_txt, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Contingency table file name', fn_df_csv])
        writer.writerow(['Report file name', fn_report_txt])
        writer.writerow(['Plot file name', fn_plot_diff])
        writer.writerow(['Plot difference file name', fn_plot_diff_only])
        writer.writerow(['Raster file name', fn_raster])
        writer.writerow(['Geotransform', geotransform])
        writer.writerow(['Spatial reference', spatial_ref])
        writer.writerow(['Change pixels', change_pixels])
        writer.writerow(['Total pixels', unmasked_pixels])
        writer.writerow(['Percent change', change_pixels/unmasked_pixels*100])
        writer.writerow(['Chi-square', chi_square])
        writer.writerow(['Degrees of freedom', dof])
        writer.writerow(['Sample size', n])
        writer.writerow(['m (for Cramers V)', m])
        writer.writerow(['Cramers V', cramers_v])
        writer.writerow(['Alpha', alpha])
        writer.writerow(['Chi2 stat (tables)', chi2_stat])
        writer.writerow(['Chi2 stat (Scipy)', stat])
        writer.writerow(['p-value', p])
        writer.writerow(['Degrees of freedom (Scipy)', dof2])
        writer.writerow(['Conclusion', conclusion])
    
    # Plot the differences
    print(f"Saving plots...")
    rs.plot_diff(ds1, ds2, diff, savefig=fn_plot_diff, cmaps=('viridis', 'viridis', 'jet'))
    rs.plot_dataset(diff, savefig=fn_plot_diff_only, interpol='none', cmap='jet')

    # Write the outputs
    print("Saving raster output...")
    rs.create_raster(fn_raster, diff, spatial_ref, geotransform)


if __name__ == '__main__':

    # cwd = '/run/media/ecoslacker/Seagate Portable Drive/Backup_2023-11-24/ROI2/'
    # results_dir = "/home/ecoslacker/Downloads/ROI2/change/"

    cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/'
    results_dir = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/change'

    periods = {'2013-2016': 'results/2023_10_28-01_04_42/2023_10_28-01_04_42_predictions_roi.tif',
               '2016-2019': 'results/2023_10_28-18_19_05/2023_10_28-18_19_05_predictions.tif',
               '2019-2022': 'results/2023_10_29-12_10_07/2023_10_29-12_10_07_predictions.tif'}
    
    # # Land cover change analysis between periods 1 and 2
    # label_periods = '2013-2016_to_2016-2019'
    # fn1 = os.path.join(cwd, '2013_2016', periods['2013-2016'])
    # fn2 = os.path.join(cwd, '2016_2019', periods['2016-2019'])
    
    # Land cover change analysis between periods
    fn1 = os.path.join(cwd, '2013_2016', periods['2013-2016'])
    fn2 = os.path.join(cwd, '2016_2019', periods['2016-2019'])
    fn3 = os.path.join(cwd, '2019_2022', periods['2019-2022'])

    # Analyze changes between periods
    # change2ds(fn1, fn2, results_dir, '2013-2019')
    # change2ds(fn2, fn3, results_dir, '2016-2022')
    # change2ds(fn1, fn3, results_dir, '2013-2022')

    print("Generate plot of gain and losses of LULC classes")
    
    fn_df_change = os.path.join(results_dir, "gain_losses_df.csv")
    fn_df_change_long = os.path.join(results_dir, "gain_losses_df_long.csv")
    fn_change_plot = os.path.join(results_dir, "gain_losses_plot.png")

    ds1, _, geotransform, spatial_ref = rs.open_raster(fn1)
    ds2, _, _, _ = rs.open_raster(fn2)
    ds3, _, _, _ = rs.open_raster(fn3)

    vals1, counts1 = np.unique(ds1, return_counts=True)
    vals2, counts2 = np.unique(ds2, return_counts=True)
    vals3, counts3 = np.unique(ds3, return_counts=True)
    
    # Remove the np.ma.core.MaskedConstant
    vals1 = np.delete(vals1, len(vals1)-1)
    counts1 = np.delete(counts1, len(counts1)-1)
    vals2= np.delete(vals2, len(vals2)-1)
    counts2 = np.delete(counts2, len(counts2)-1)
    vals3= np.delete(vals3, len(vals3)-1)
    counts3 = np.delete(counts3, len(counts3)-1)

    zipped = list(zip(vals1, counts1, vals2, counts2, vals3, counts3))

    df = pd.DataFrame(zipped, columns=['Classes1', 'Freq1', 'Classes2', 'Freq2', 'Classes3', 'Freq3'])
    df['2013-2019'] = counts2-counts1
    df['2016-2022'] = counts3-counts2
    df['2013-2022'] = counts3-counts1
    df.to_csv(fn_df_change)
    print(df)

    # Bar plot
    ax = sns.barplot(df, x="Classes1", y="2013-2022")
    ax.set(xlabel='Class', ylabel='Difference')
    plt.savefig(fn_change_plot[:-4] + '_all.png', bbox_inches='tight', dpi=600)

    rows = df.shape[0]
    data = {'Class': df['Classes1'].tolist() + df['Classes2'].tolist() + df['Classes3'].tolist(),
            'Frequency': df['Freq1'].tolist() + df['Freq2'].tolist() + df['Freq3'].tolist(),
            'Difference': df['2013-2019'].tolist() + df['2016-2022'].tolist() + df['2013-2022'].tolist(),
            'Period': ['2013-2019']*rows + ['2016-2022']*rows + ['2013-2022']*rows}
    df2 = pd.DataFrame(data)
    df2.to_csv(fn_df_change_long)
    print(df2)

    # Bar plot
    sns.catplot(df2, kind="bar", x="Class", y="Difference", col="Period")
    plt.savefig(fn_change_plot, bbox_inches='tight', dpi=600)

    print("All done. ;-)")