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
from datetime import datetime
from scipy.stats import chi2, chi2_contingency

# sys.path.insert(0, '/home/ecoslacker/Documents/land_cover_analysis/lib/')
sys.path.insert(0, '/data/ssd/eduardojh/land_cover_analysis/lib/')

import rsmodule as rs

def chisquare_test(filename1, filename2, outdir, label, **kwargs):
    """ Change between two 2D datasets or land cover maps, calculates pixels with change and no change
        carries out a Chi-Square analysis for homonegeity and Cramers-V. Creates some files and figures.
    """

    print(f"=== CHANGE BETWEEN: {filename1} AND {filename2}")

    # Create output directory if it doesn't exists
    if not os.path.exists(outdir):
        print(f"Creating path for output: {outdir}")
        os.makedirs(outdir)

    # File names to create
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
    diff = np.where(ds1 == ds2, 0, 1)  # pixels with change=1, no change=0
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

    #==========  Use the chi-square test  ==========
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

    print(f"stat={stat}, p-value={p}, dof={dof2}, expected={expected}")

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


def change_analysis(fn_raster1: str, fn_raster2: str, results_dir: str, **kwargs) -> None:
    """ Performs the land cover change analysis between two rasters and also the land cover transitions table
    
    :param str fn_raster1: first land cover raster GeoTIFF
    :param str fn_raster2: second land cover raster GeoTIFF
    :param str results_dir: results directory to save files
    :return: None
    """

    # How to label the periods P1 and P2, and the ID of the analysis
    _p1_lbl = kwargs.get("label1", "P1")
    _p2_lbl = kwargs.get("label2", "P2")
    _idanal = kwargs.get("id", "")

    print(f"\n RUNNING CHANGE ANALYSIS {_idanal} BETWEEN:")

    # If ID is not empty add an underscore at the end to help file names
    if len(_idanal) > 1:
        if _idanal[:-1] != '_':
            _idanal = _idanal + '_'

    print(f"--> RASTER1: {fn_raster1}")
    print(f"--> RASTER2: {fn_raster2}")

    fn_df_change = os.path.join(results_dir, f"{_idanal}gain_losses_df.csv")
    fn_df_change_long = os.path.join(results_dir, f"{_idanal}gain_losses_df_long.csv")
    fn_change_plot = os.path.join(results_dir, f"{_idanal}gain_losses_plot.png")
    fn_lc_changes = os.path.join(results_dir, f"{_idanal}land_cover_class_changes.csv")
    fn_table_changes = os.path.join(results_dir, f"{_idanal}land_cover_class_changes_table.csv")
    fn_table_changes_p = os.path.join(results_dir, f"{_idanal}land_cover_class_changes_table_percent.csv")
    fn_dataset_change = os.path.join(results_dir, f"{_idanal}land_cover_change.tif")

    ds1, _, geotransform1, spatial_ref1 = rs.open_raster(fn_raster1)
    ds2, _, geotransform2, spatial_ref2 = rs.open_raster(fn_raster2)

    ds_mask = np.ma.getmask(ds1)
    diff = np.where(ds1 == ds2, 0, 1)
    diff = np.ma.masked_array(diff, mask=ds_mask)
    unmasked_pixels = ds_mask.size - np.sum(ds_mask)
    change_pixels = np.sum(diff)

    print(type(change_pixels))
    # if change_pixels == 0:
    #     print("There are no change pixels between the two raster files!\nNothing to do.")
    # elif type(change_pixels) is np.ma.core.MaskedConstant:
    #     print("There are no change pixels between the two raster files! (type is np.ma.core.MaskedConstant)\nNothing to do.")
    # else:
    # There are actually changes
    print(f"Change pixels={change_pixels}, out of {unmasked_pixels} ({change_pixels/unmasked_pixels*100:0.2f}%)")

    vals1, counts1 = np.unique(ds1, return_counts=True)
    vals2, counts2 = np.unique(ds2, return_counts=True)
    
    # Remove the np.ma.core.MaskedConstant
    vals1 = np.delete(vals1, len(vals1)-1)
    counts1 = np.delete(counts1, len(counts1)-1)
    vals2= np.delete(vals2, len(vals2)-1)
    counts2 = np.delete(counts2, len(counts2)-1)

    zipped = list(zip(vals1, counts1, vals2, counts2))

    # Transform from pixels to hectares
    pixel_m2 = 30 * 30
    m2_ha = 1/10000

    # Total change area
    change_area = change_pixels * pixel_m2 * m2_ha
    print(f"Change area: {change_area} ha")

    # Create a pandas Dataframe to save the pixel counts of each land cover class
    df = pd.DataFrame(zipped, columns=[f'Classes{_p1_lbl}', f'Freq{_p1_lbl}', f'Classes{_p2_lbl}', f'Freq{_p2_lbl}'])
    # Pixels to area
    df[f'{_p1_lbl}_ha'] = counts1 * pixel_m2 * m2_ha
    df[f'{_p2_lbl}_ha'] = counts2 * pixel_m2 * m2_ha
    
    # Differences
    print(f"Calculatig differences.")
    df[f'{_p2_lbl}-{_p1_lbl}'] = df[f'{_p2_lbl}_ha']-df[f'{_p1_lbl}_ha']

    # Percentage with respect to original area
    df[f'{_p2_lbl}-{_p1_lbl}_per'] = df[f'{_p2_lbl}-{_p1_lbl}']/df[f'{_p1_lbl}_ha']*100

    # Percent of area with changes
    df[f'{_p2_lbl}-{_p1_lbl}_per_ch'] = df[f'{_p2_lbl}-{_p1_lbl}']/change_area*100

    df.to_csv(fn_df_change)
    print(df)

    # Bar plot with the percentage of change 
    print('Bar plot with percentage of change.')
    title = "Change between P1 to P3 (%)"
    ax = sns.barplot(df, x=f"Classes{_p1_lbl}", y=f"{_p2_lbl}-{_p1_lbl}_per")
    ax.set(xlabel='Class', ylabel='Change area (%)')
    plt.savefig(fn_change_plot[:-4] + '_all.png', bbox_inches='tight', dpi=600)

    # Create a pandas Dataframe to save the pixel counts, areas, differences, and change
    rows = df.shape[0]
    data = {'Period': ['2013-2016']*rows + ['2019-2022']*rows,
            'Class': df[f'Classes{_p1_lbl}'].tolist() + df[f'Classes{_p2_lbl}'].tolist(),
            'Frequency': df[f'Freq{_p1_lbl}'].tolist() + df[f'Freq{_p2_lbl}'].tolist(),
            'Area': df[f'{_p1_lbl}_ha'].tolist() + df[f'{_p2_lbl}_ha'].tolist(),
            'Difference': df[f'{_p2_lbl}-{_p1_lbl}'].tolist() + df[f'{_p2_lbl}-{_p1_lbl}'].tolist(),
            'Change periods (%)': df[f'{_p2_lbl}-{_p1_lbl}_per'].tolist() + df[f'{_p2_lbl}-{_p1_lbl}_per'].tolist(),
            'Change area (%)': df[f'{_p2_lbl}-{_p1_lbl}_per_ch'].tolist() + df[f'{_p2_lbl}-{_p1_lbl}_per_ch'].tolist()}
    df2 = pd.DataFrame(data)
    df2.to_csv(fn_df_change_long)
    print(df2)

    # Bar plot
    sns.catplot(df2, kind="bar", x="Class", y="Change periods (%)", col="Period")
    plt.savefig(fn_change_plot, bbox_inches='tight', dpi=600)

    # ====== ANALYSIS OF LAND COVER TRANSITIONS ======

    print("Now calculating the changes between land cover classes")
    dataset_p1 = ds1.filled(0).astype(np.int32)
    dataset_p2 = ds2.filled(0).astype(np.int32)
    # Change format will be 101102 where P1 class is 101 and it changed to 102 in P2
    dataset_change = (dataset_p1*1000)+dataset_p2

    # Get gain and loss for evergreen tropical forest
    #    NoData (0): transitions not accounted for
    #      gain (1): from classes 101, 102, and 109 to 106
    # unchanged (2): 106 stayed the same
    #      loss (3): from class 106 to 101, 102, and 109
    # print("Getting gain and loss for evergreen tropical forest")
    # evergreen_change = np.where((dataset_change == 101106) | (dataset_change == 102106) | (dataset_change == 109106), 1, 0)
    # evergreen_change = np.where((dataset_change == 106106), 2, evergreen_change)
    # evergreen_change = np.where(((dataset_change == 106101) | (dataset_change == 106102) | (dataset_change == 106109)), 3, evergreen_change)
    # print(f"Evergreen tropical forest: {np.unique(evergreen_change, return_counts=True)}")
    # rs.create_raster(fn_dataset_change[:-4] + "_evergreen.tif", evergreen_change, spatial_ref1, geotransform1, type='int')

    # print("Getting gain and loss for deciduous tropical forest")
    # deciduous_change = np.where((dataset_change == 101109) | (dataset_change == 102109) | (dataset_change == 106109), 1, 0)
    # deciduous_change = np.where(dataset_change == 109109, 2, deciduous_change)
    # deciduous_change = np.where((dataset_change == 109101) | (dataset_change == 109102) | (dataset_change == 109106), 3, deciduous_change)
    # rs.create_raster(fn_dataset_change[:-4] + "_deciduous.tif", deciduous_change, spatial_ref1, geotransform1, type='int')

    # print("Getting gain and loss for urban")
    # urban_change = np.where((dataset_change == 101102) | (dataset_change == 106102) | (dataset_change == 109102), 1, 0)
    # urban_change = np.where(dataset_change == 102102, 2, urban_change)
    # urban_change = np.where((dataset_change == 102101) | (dataset_change == 102106) | (dataset_change == 102109), 3, urban_change)
    # rs.create_raster(fn_dataset_change[:-4] + "_urban.tif", urban_change, spatial_ref1, geotransform1, type='int')

    # print("Getting gain and loss for agriculture")
    # ag_change = np.where((dataset_change == 102101) | (dataset_change == 106101) | (dataset_change == 109101), 1, 0)
    # ag_change = np.where(dataset_change == 101101, 2, ag_change)
    # ag_change = np.where((dataset_change == 101102) | (dataset_change == 101106) | (dataset_change == 101109), 3, ag_change)
    # rs.create_raster(fn_dataset_change[:-4] + "_ag.tif", ag_change, spatial_ref1, geotransform1, type='int')

    print("Getting gain and loss for evergreen + deciduous tropical forest")
    trop_forest_change = np.where((dataset_change == 101106) | (dataset_change == 102106) | (dataset_change == 109106) | (dataset_change == 101109) | (dataset_change == 102109) | (dataset_change == 106109), 1, 0)
    trop_forest_change = np.where((dataset_change == 106106) | (dataset_change == 109109) | (dataset_change == 106109) | (dataset_change == 109106), 2, trop_forest_change)
    trop_forest_change = np.where((dataset_change == 106101) | (dataset_change == 106102)  | (dataset_change == 109101) | (dataset_change == 109102), 3, trop_forest_change)
    print(f"Evergreen & deciduous tropical forest: {np.unique(trop_forest_change, return_counts=True)}")
    rs.create_raster(fn_dataset_change[:-4] + "_trop_forest.tif", trop_forest_change, spatial_ref1, geotransform1, type='int')

    print("Getting gain and loss for urban +  agriculture")
    built_up_change = np.where((dataset_change == 101102) | (dataset_change == 106102) | (dataset_change == 109102) | (dataset_change == 102101) | (dataset_change == 106101) | (dataset_change == 109101), 1, 0)
    built_up_change = np.where((dataset_change == 102102) | (dataset_change == 101101), 2, built_up_change)
    built_up_change = np.where((dataset_change == 102101) | (dataset_change == 102106) | (dataset_change == 102109) | (dataset_change == 101102) | (dataset_change == 101106) | (dataset_change == 101109), 3, built_up_change)
    print(f"Built-up & agriculture: {np.unique(built_up_change, return_counts=True)}")
    rs.create_raster(fn_dataset_change[:-4] + "_urban_ag.tif", built_up_change, spatial_ref1, geotransform1, type='int')

    # All pixels with land cover change
    class_change, changed_pixels = np.unique(dataset_change, return_counts=True)
    changed_area = np.round(changed_pixels * pixel_m2 * m2_ha, 2)

    data_change = {'Changes': class_change, 'Pixels': changed_pixels, 'Area_ha': changed_area}
    df3 = pd.DataFrame(data_change)
    df3.to_csv(fn_lc_changes)
    print("\nThe coded land cover changes are:")
    print(df3)

    # Create a change table, rows will be the class in P1 and cols the class in P2
    change_table = np.zeros((len(vals1), len(vals2)), dtype=float)
    for indices, count in zip(class_change, changed_area):
        if indices == 0:
            continue
        idx = str(indices)
        # First three elements are the class in P1, last 3 the class in P2
        # Substract 101, so class 109 becomes row 8
        r, c = int(idx[:3])-101, int(idx[3:])-101
        change_table[r, c] = count
    
    # Create a dataframe with the count of land cover changes
    # NOW WE TRANSPOSE: rows are class in P3 and cols the class in P1!!!
    df_change = pd.DataFrame(change_table.T, columns=vals1)
    df_change.set_index(vals2, inplace=True)
    df_change.to_csv(fn_table_changes)
    print(df_change)

    # The change table in percentages (within-column)
    change_table_p = {}
    for lcclass in vals1:
        change_table_p[lcclass] = df_change[lcclass] * 100 / df_change[lcclass].sum()
    df_change_p = pd.DataFrame(change_table_p)
    df_change_p.set_index(vals2, inplace=True)
    df_change_p.to_csv(fn_table_changes_p)
    print(df_change_p)


if __name__ == '__main__':

    start = datetime.now()
    print(f"Execution starting at: {start}")

    # cwd = '/run/media/ecoslacker/Seagate Portable Drive/Backup_2023-11-24/ROI2/'
    # results_dir = "/home/ecoslacker/Downloads/ROI2/change/"

    # cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/'
    # results_dir = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/change'

    # periods = {'2013-2016': 'results/2024_03_06-23_19_43/2024_03_06-23_19_43_predictions.tif',
    #            '2016-2019': 'results/2024_03_08-13_29_31/2024_03_08-13_29_31_predictions.tif',
    #            '2019-2022': 'results/2024_03_12-19_32_01/2024_03_12-19_32_01_predictions.tif'}
    
    # # Land cover change analysis between periods
    # fn1 = os.path.join(cwd, '2013_2016', periods['2013-2016'])
    # fn2 = os.path.join(cwd, '2016_2019', periods['2016-2019'])
    # fn3 = os.path.join(cwd, '2019_2022', periods['2019-2022'])

    # # Analyze changes between periods, Chi-square test
    # # Comment out to speed up the change analysis below between P1 and P3
    # chisquare_test(fn1, fn2, results_dir, '2013-2019')
    # chisquare_test(fn2, fn3, results_dir, '2016-2022')
    # chisquare_test(fn1, fn3, results_dir, '2013-2022')

    # print("Generate plot of gain and losses of LULC classes")
    
    # fn_df_change = os.path.join(results_dir, "gain_losses_df.csv")
    # fn_df_change_long = os.path.join(results_dir, "gain_losses_df_long.csv")
    # fn_change_plot = os.path.join(results_dir, "gain_losses_plot.png")
    # fn_lc_changes = os.path.join(results_dir, "land_cover_class_changes.csv")
    # fn_table_changes = os.path.join(results_dir, "land_cover_class_changes_table.csv")
    # fn_table_changes_p = os.path.join(results_dir, "land_cover_class_changes_table_percent.csv")

    # ds1, _, geotransform, spatial_ref = rs.open_raster(fn1)
    # ds2, _, _, _ = rs.open_raster(fn2)
    # ds3, _, _, _ = rs.open_raster(fn3)

    #========== BEGIN  === CHANGE ANALYSIS BETWEEN P1 & P3 ===   BEGIN ==========

    # ds_mask = np.ma.getmask(ds1)
    # diff = np.where(ds1 == ds3, 0, 1)
    # diff = np.ma.masked_array(diff, mask=ds_mask)
    # unmasked_pixels = ds_mask.size - np.sum(ds_mask)
    # change_pixels = np.sum(diff)
    # print(f"Change pixels={change_pixels}, out of {unmasked_pixels} ({change_pixels/unmasked_pixels*100:0.2f}%)")

    # vals1, counts1 = np.unique(ds1, return_counts=True)
    # vals2, counts2 = np.unique(ds2, return_counts=True)
    # vals3, counts3 = np.unique(ds3, return_counts=True)
    
    # # Remove the np.ma.core.MaskedConstant
    # vals1 = np.delete(vals1, len(vals1)-1)
    # counts1 = np.delete(counts1, len(counts1)-1)
    # vals2= np.delete(vals2, len(vals2)-1)
    # counts2 = np.delete(counts2, len(counts2)-1)
    # vals3= np.delete(vals3, len(vals3)-1)
    # counts3 = np.delete(counts3, len(counts3)-1)

    # zipped = list(zip(vals1, counts1, vals2, counts2, vals3, counts3))

    # # Transform from pixels to hectares
    # pixel_m2 = 30 * 30
    # m2_ha = 1/10000

    # # Total change area
    # change_area = change_pixels * pixel_m2 * m2_ha
    # print(f"Change area: {change_area} ha")

    # # Create a pandas Dataframe to save the pixel counts of each land cover class
    # df = pd.DataFrame(zipped, columns=['ClassesP1', 'FreqP1', 'ClassesP2', 'FreqP2', 'ClassesP3', 'FreqP3'])
    # # Pixels to area
    # df['P1_ha'] = counts1 * pixel_m2 * m2_ha
    # df['P2_ha'] = counts2 * pixel_m2 * m2_ha
    # df['P3_ha'] = counts3 * pixel_m2 * m2_ha
    # # Differences
    # df['P2-P1'] = df['P2_ha']-df['P1_ha']
    # df['P3-P2'] = df['P3_ha']-df['P2_ha']
    # df['P3-P1'] = df['P3_ha']-df['P1_ha']
    # # Percentage with respect to original area
    # df['P2-P1_per'] = df['P2-P1']/df['P1_ha']*100
    # df['P3-P2_per'] = df['P3-P2']/df['P2_ha']*100
    # df['P3-P1_per'] = df['P3-P1']/df['P1_ha']*100
    # # Percent of area with changes
    # df['P2-P1_per_ch'] = df['P2-P1']/change_area*100
    # df['P3-P2_per_ch'] = df['P3-P2']/change_area*100
    # df['P3-P1_per_ch'] = df['P3-P1']/change_area*100

    # df.to_csv(fn_df_change)
    # print(df)

    # # Bar plot with the percentage of change 
    # title = "Change between P1 to P3 (%)"
    # ax = sns.barplot(df, x="ClassesP1", y="P3-P1_per")
    # ax.set(xlabel='Class', ylabel='Change area (%)')
    # plt.savefig(fn_change_plot[:-4] + '_all.png', bbox_inches='tight', dpi=600)

    # # Create a pandas Dataframe to save the pixel counts, areas, differences, and change
    # rows = df.shape[0]
    # data = {'Period': ['2013-2019']*rows + ['2016-2022']*rows + ['2013-2022']*rows,
    #         'Class': df['ClassesP1'].tolist() + df['ClassesP2'].tolist() + df['ClassesP3'].tolist(),
    #         'Frequency': df['FreqP1'].tolist() + df['FreqP2'].tolist() + df['FreqP3'].tolist(),
    #         'Area': df['P1_ha'].tolist() + df['P2_ha'].tolist() + df['P3_ha'].tolist(),
    #         'Difference': df['P2-P1'].tolist() + df['P3-P2'].tolist() + df['P3-P1'].tolist(),
    #         'Change periods (%)': df['P2-P1_per'].tolist() + df['P3-P2_per'].tolist() + df['P3-P1_per'].tolist(),
    #         'Change area (%)': df['P2-P1_per_ch'].tolist() + df['P3-P2_per_ch'].tolist() + df['P3-P1_per_ch'].tolist()}
    # df2 = pd.DataFrame(data)
    # df2.to_csv(fn_df_change_long)
    # print(df2)

    # # Bar plot
    # sns.catplot(df2, kind="bar", x="Class", y="Change periods (%)", col="Period")
    # plt.savefig(fn_change_plot, bbox_inches='tight', dpi=600)

    # # ====== ANALYSIS OF LAND COVER TRANSITIONS ======

    # print("Now calculating the changes between land cover classes")
    # dataset_p1 = ds1.filled(0).astype(np.int32)
    # dataset_p2 = ds3.filled(0).astype(np.int32)
    # # Change format will be 101102 where P1 class is 101 and it changed to 102 in P2
    # dataset_change = (dataset_p1*1000)+dataset_p2

    # class_change, changed_pixels = np.unique(dataset_change, return_counts=True)
    # changed_area = np.round(changed_pixels * pixel_m2 * m2_ha, 2)

    # data_change = {'Changes': class_change, 'Pixels': changed_pixels, 'Area_ha': changed_area}
    # df3 = pd.DataFrame(data_change)
    # df3.to_csv(fn_lc_changes)

    # # Create a change table, rows will be the class in P1 and cols the class in P2
    # change_table = np.zeros((len(vals1), len(vals3)), dtype=float)
    # for indices, count in zip(class_change, changed_area):
    #     if indices == 0:
    #         continue
    #     idx = str(indices)
    #     # First three elements are the class in P1, last 3 the class in P2
    #     # Substract 101, so class 109 becomes row 8
    #     r, c = int(idx[:3])-101, int(idx[3:])-101
    #     change_table[r, c] = count
    
    # # Create a dataframe with the count of land cover changes
    # # NOW WE TRANSPOSE: rows are class in P3 and cols the class in P1!!!
    # df_change = pd.DataFrame(change_table.T, columns=vals1)
    # df_change.set_index(vals3, inplace=True)
    # df_change.to_csv(fn_table_changes)
    # print(df_change)

    # # The change table in percentages (within-column)
    # change_table_p = {}
    # for lcclass in vals1:
    #     change_table_p[lcclass] = df_change[lcclass] * 100 / df_change[lcclass].sum()
    # df_change_p = pd.DataFrame(change_table_p)
    # df_change_p.set_index(vals3, inplace=True)
    # df_change_p.to_csv(fn_table_changes_p)
    # print(df_change_p)

    #==========  END === CHANGE ANALYSIS BETWEEN P1 & P3 ===  END ==========

    ### =================== EFFECTIVENESS OF THE CBR ======================= ###

    cwd = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/'
    results_dir = '/VIP/engr-didan02s/DATA/EDUARDO/YUCATAN_LAND_COVER/ROI2/effectiveness/'

    # Use the rasters already fixed to full extent
    fn_mask_sim_area = '/VIP/engr-didan02s/DATA/EDUARDO/CBR/Raster/SIMAREA_full.tif'
    fn_calakmul = '/VIP/engr-didan02s/DATA/EDUARDO/CBR/Raster/calakmul_full.tif'

    # Extract the similar area from P1 and P3 predictions
    fn_p1 = os.path.join(cwd, '2013_2016', 'results', '2024_03_06-23_19_43', '2024_03_06-23_19_43_predictions.tif')
    fn_p3 = os.path.join(cwd, '2019_2022', 'results', '2024_03_12-19_32_01', '2024_03_12-19_32_01_predictions.tif')

    # Open the similar area buffer (SIMAREA.tif) as a mask raster
    ds_mask, _, gt_maks, spref_mask = rs.open_raster(fn_mask_sim_area)
    print(f'\nMask values: {np.unique(ds_mask)}')

    # Open the Calakmul Biosphere Reserve polygon for comparison
    ds_cbr, _, gt_cbr, spref_cbr = rs.open_raster(fn_calakmul)
    print(f'\nCalakmul values: {np.unique(ds_mask)}')

    # Open the prediction rasters for P1 and P3
    ds_p1, _, gt_p1, spref_p1 = rs.open_raster(fn_p1)
    ds_p3, _, gt_p3, spref_p3 = rs.open_raster(fn_p3)

    # Extract predictions P1 and P3 for the Calakmul Biosphere Reserve
    cbr_p1 = np.where(ds_cbr.filled(0) > 0, ds_p1, 0)
    cbr_p3 = np.where(ds_cbr.filled(0) > 0, ds_p3, 0)

    # Save the extractions from CBR to a raster
    fn_calakmul_p1 = os.path.join(results_dir, 'p1_cbr.tif')
    fn_calakmul_p3 = os.path.join(results_dir, 'p3_cbr.tif')
    rs.create_raster(fn_calakmul_p1, cbr_p1, spref_cbr, gt_cbr, type='int')
    rs.create_raster(fn_calakmul_p3, cbr_p3, spref_cbr, gt_cbr, type='int')

    # Extract predictions P1 and P3 for the similar area buffer
    simarea_p1 = np.where(ds_mask.filled(0) > 0, ds_p1, 0)
    simarea_p3 = np.where(ds_mask.filled(0) > 0, ds_p3, 0)

    # Save the extractions from Similar Area Buffer to a raster
    fn_simarea_p1 = os.path.join(results_dir, 'p1_simarea.tif')
    fn_simarea_p3 = os.path.join(results_dir, 'p3_simarea.tif')
    rs.create_raster(fn_simarea_p1, simarea_p1, spref_cbr, gt_cbr, type='int')
    rs.create_raster(fn_simarea_p3, simarea_p3, spref_cbr, gt_cbr, type='int')

    # TAKES TIME!
    # # Compare the two rasters for P1, then for P3  # this comparison is not right
    # print(f"Comparison between CBR and similar area buffer for P1.")
    # rs.chi_square_raster(fn_calakmul_p1, fn_simarea_p1, results_dir, 'P1_comp')
    # print(f"Comparison between CBR and similar area buffer for P2.")
    # rs.chi_square_raster(fn_calakmul_p3, fn_simarea_p3, results_dir, 'P3_comp')

    # print(f"Comparison between P1 and P3 for CBR.")
    # rs.chi_square_raster(fn_calakmul_p1, fn_calakmul_p3, results_dir, 'CBR_comp')
    # print(f"Comparison between P1 and P3 for Similar Area Buffer.")
    # rs.chi_square_raster(fn_simarea_p1, fn_simarea_p3, results_dir, 'SIMAREA_comp')

    # Change analysis inside the CBR
    change_analysis(fn_calakmul_p1, fn_calakmul_p3, results_dir, label1="P1_in", label2="P3_in", id="CBR")

    # Change analysis in the Similar Buffer Area (outside the CBR)
    change_analysis(fn_simarea_p1, fn_simarea_p3, results_dir, label1="P1_out", label2="P3_out", id="SIMAREA")

    end = datetime.now()
    elapsed = end - start
    print(f"Execution ended at {end} (elapsed={elapsed})")
    print("All done. ;-)")