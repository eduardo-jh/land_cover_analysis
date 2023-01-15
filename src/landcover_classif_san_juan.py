#!/usr/bin/env python
# coding: utf-8

""" landcover_classif.py: Land cover classification with machine learning (random forest)
Eduardo Jimenez <eduardojh@email.arizona.edu>
9/16/2022

NOTE: run under 'rsml' conda environment (python 3.8.13, scikit-learn 1.1.2)
"""

import pickle
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from skimage import data, color, io, img_as_float
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from rsmodule import open_raster, create_raster, read_from_hdf

fmt = '%Y_%m_%d-%H_%M_%S'
start = datetime.now()

### 1. CONFIGURE
# Paths and file names for the current ROI
cwd = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/SAN_JUAN_RIVER/'
fn_landcover = cwd + 'GAP/gaplf2011lc_v30_groups.tif'        # Land cover raster
fn_train_mask = cwd + 'ML/train_mask4.tif'
fn_nodata_mask = cwd + 'MONTHLY_NDVI/MONTHLY.NDVI.08.AUG.MIN.tif'   # Landsat 'NoData' filter
fn_phenology = cwd + 'PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf'  # Phenology files
fn_phenology2 = cwd + 'PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf'
# File names to save results and reports
save_train_stats = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_training_stats.csv'
save_conf_tbl = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_confussion_table.csv'
save_model = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_model.pkl'
save_report = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_classif_report.txt'
save_preds_raster = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_predictions.tif'
save_preds_fig = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_predictions.png'
save_params = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_parameters.csv'

### 2. TRAINING MASK
# Read a raster with the location of the training sites
fn_train_mask = cwd + 'ML/train_mask4.tif'

train_mask, nodata, metadata, geotransform, projection = open_raster(fn_train_mask)
print(f'Opening raster: {fn_train_mask}')
print(f'Metadata      : {metadata}')
print(f'NoData        : {nodata}')
print(f'Columns       : {train_mask.shape[1]}')
print(f'Rows          : {train_mask.shape[0]}')
print(f'Geotransform  : {geotransform}')
# print(f'Projection    : {projection}')

# # Find how many non-zero entries we have -- i.e. how many training data samples?
# n_samples = (train_mask > 0).sum()
# print(f'We have {n_samples} samples')

### 3. LAND COVER LABELS
# Read the land cover raster and retrive the land cover classes
lc_arr, lc_nd, lc_md, lc_gt, lc_proj = open_raster(fn_landcover)
print(f'Opening raster: {fn_landcover}')
print(f'Metadata      : {lc_md}')
print(f'NoData        : {lc_nd}')
print(f'Columns       : {lc_arr.shape[1]}')
print(f'Rows          : {lc_arr.shape[0]}')
print(f'Geotransform  : {lc_gt}')
# print(f'Projection    : {lc_proj}')

# Mask out the 'NoData' pixels to match Landsat data and land cover classes
dummy_array, _, _, _, _ = open_raster(fn_nodata_mask)
lc_arr = np.ma.masked_array(lc_arr, mask=np.ma.getmask(dummy_array))

gap_stats = {}
groups, frequency = np.unique(lc_arr, return_counts=True)
valid_groups = list(groups.compressed()) # Get only the unmasked values

for grp, frq in zip(groups, frequency):
    # print(f'{grp}, {frq} {type(grp)}')
    if not grp in valid_groups:
        # print('Skipping')
        continue

    if gap_stats.get(grp) is None:
        # print(f'Adding {grp}')
        gap_stats[grp] = frq
    else:
        gap_stats[grp] += frq

print('Analyzing training dataset (labels)')
# print(f'{train_mask.dtype} {lc_arr.dtype}')
train_labels = lc_arr[train_mask > 0]  # This array gets flatten

train_gap_stats = {}
labels, freq_lbls = np.unique(train_labels, return_counts=True)
valid_labels = list(labels.compressed())  # Get only the unmasked values

for lbl, f in zip(labels, freq_lbls):
    if not lbl in valid_labels:
        continue
    if train_gap_stats.get(lbl) is None:
        train_gap_stats[lbl] = f
    else:
        train_gap_stats[lbl] += f

print(f'{len(labels)} unique land cover values in training dataset.')
# print(f'{len(labels)} = {len(groups)}')

print('Saving group frequency in train dataset...')

with open(save_train_stats, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Key_trainig', 'Freq_training', 'Key_all', 'Freq_all', 'Percentage'])
    for i, j in zip(valid_labels, valid_groups):
        print(f'{i:>3}: {train_gap_stats[i]:>10} {j:>3} {gap_stats[j]:>10} ({train_gap_stats[i]/gap_stats[j]*100:>6.2f} %)')
        writer.writerow([i, train_gap_stats[i], j, gap_stats[j], train_gap_stats[i]/gap_stats[j]])
n_train = sum(freq_lbls)
n_total = sum(frequency)
print(f'TOTAL: {n_train} / {n_total} ({n_train/n_total*100:>6.2f} %)')

### SPECTRAL BANDS
# bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir']
# band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
# months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
# vars = ['NPixels', 'MIN', 'MAX', 'AVG', 'STDEV']
# phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
# phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

# bands = ['Blue', 'Green', 'Ndvi', 'Red']
# band_num = ['B2', 'B3', '', 'B4']
# months = ['JAN',  'APR', 'JUL',  'OCT', 'DEC']
# nmonths = [1,  4, 7,  10, 12]
# vars = ['MIN', 'MAX', 'AVG']
# phen = ['SOS', 'EOS']
# phen2 = ['SOS2', 'EOS2']

bands = ['Blue', 'Evi', 'Evi2', 'Green', 'Mir', 'Ndvi', 'Nir', 'Red', 'Swir1']
band_num = ['B2', '', '', 'B3', 'B7', '', 'B5', 'B4', 'B6']
months = ['FEB',  'APR', 'JUN', 'AUG', 'OCT', 'DEC']
nmonths = [2,  4, 6, 8, 10, 12]
vars = ['AVG']
phen = ['SOS', 'EOS']
phen2 = ['SOS2', 'EOS2']

# bands = ['Blue', 'Green', 'Ndvi', 'Red']
# band_num = ['B2', 'B3', '', 'B4']
# months = ['JAN']
# vars = ['AVG']
# phen = ['SOS', 'EOS']
# phen2 = ['SOS2', 'EOS2']

# Calculate the dimensions of the array
cols = train_mask.shape[1]
rows = train_mask.shape[0]
lyrs = len(bands) * len(months) * len(vars) + len(phen) + len(phen2)
print(f'Rows={rows}, Cols={cols}, Layers={lyrs}')

# Create an empty array
bands_array = np.zeros((rows, cols, lyrs), dtype=np.float32)

layer = 0
for j, band in enumerate(bands):
    print(f'{band}')
    for i, month in enumerate(months):
        # filename = 'MONTHLY/MONTHLY.' + band.upper() + '.' + str(i+1).zfill(2) + '.' + month + '.hdf'
        filename = 'MONTHLY/MONTHLY.' + band.upper() + '.' + str(nmonths[i]).zfill(2) + '.' + month + '.hdf'
        print(f' {cwd+filename}')
        for var in vars:
            # Create the name of the dataset in the HDF
            var_name = band_num[j] + ' (' + band + ') ' + var
            if band_num[j] == '':
                var_name = band.upper() + ' ' + var
            print(f'  Layer: {layer} Retrieving: {var} Dataset: {var_name}')
            bands_array[:,:,layer] = read_from_hdf(cwd + filename, var_name)
            layer += 1

# Retrieve phenology data
print(fn_phenology)
for param in phen:
    print(f' Layer: {layer} Retrieving: {param}')
    bands_array[:,:,layer] = read_from_hdf(fn_phenology, param)
    layer += 1

print(fn_phenology2)
for param in phen2:
    print(f' Layer: {layer} Retrieving: {param}')
    bands_array[:,:,layer] = read_from_hdf(fn_phenology2, param)
    layer += 1

print(f'Bands array shape={bands_array.shape}')


### TRAIN THE RANDOM FORESTS
print(f'Starting training of Random Forests...')

# Create the training and testing datasets
filter = train_mask > 0  # A filter to remove/mask the NoData cells from Landsat (NOT USED FOR NOW)

X_train = bands_array[train_mask > 0, :]  # WARNING! This is flatten
y_train = train_labels

# X_test is the flatten array of the features
new_shape = (bands_array.shape[0] * bands_array.shape[1], bands_array.shape[2])
print(f'Change bands array from {bands_array.shape} to: {new_shape}')
X_test = bands_array.reshape(new_shape)

# y_test is the land cover array as as single column
_rows, _cols = lc_arr.shape
y_test = lc_arr.reshape(_rows * _cols,)

print(f'X_train shape={X_train.shape}')
print(f'y_train shape={y_train.shape}')
print(f'filter shape={filter.shape}')

# Random forest
print('Creating the model')
start_train = datetime.now()

# rf_estimators = 100
# rf_max_depth = 6
# rf_n_jobs = 14

rf_estimators = 10
rf_max_depth = None
rf_n_jobs = 1

rf = RandomForestClassifier(n_estimators=rf_estimators, oob_score=True, max_depth=rf_max_depth, n_jobs=rf_n_jobs)

print('Fitting the model')
rf = rf.fit(X_train, y_train)

# Save trained model
with open(save_model, 'wb') as f:
    pickle.dump(rf, f)

print(f'OOB prediction of accuracy: {rf.oob_score_ * 100:0.2f}%')

# # A crosstabulation to see class confusion for TRAINING
# df = pd.DataFrame()
# df['truth_train'] = y_train
# df['predict_train'] = rf.predict(X_train)
# confusion_table = pd.crosstab(df['truth_train'], df['predict_train'], margins=True)
# confusion_table.to_csv(save_conf_tbl)

end_train = datetime.now()
training_time = end_train - start_train
print(f'Training finished in {training_time}')

# Predict on the rest of the image, using the fitted Random Forest classifier
print('Creating predictions for the rest of the image')
start_pred = datetime.now()
y_pred = rf.predict(X_test)
print(f'y_pred shape:', y_pred.shape)

print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
# print(type(cm))
# print(cm.shape)
with open(save_conf_tbl, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for single_row in cm:
        writer.writerow(single_row)
        print(single_row)

report = classification_report(y_test, y_pred, )
print('Classification report')
print(classification_report(y_test, y_pred, ))
with open(save_report, 'w') as f:
    f.write(report)

# Reshape the classification map into a 2D array again to show as a map
y_pred = y_pred.reshape(bands_array[:, :, 0].shape)
print(f'y_pred (re)shape:', y_pred.shape)

# Save GeoTIFF of the predicted land cover classes
src_proj = 32612  # WGS 84 / UTM zone 12N
create_raster(save_preds_raster, y_pred, src_proj, lc_gt)
 
end_pred = datetime.now()
pred_time =  end_pred - start_pred
print(f'Prediction finished in {pred_time}')

print('Plotting predictions')
plt.figure(figsize=(12,12))
plt.imshow(y_pred, cmap='viridis')
plt.colorbar()
plt.savefig(save_preds_fig, bbox_inches='tight', dpi=300)
plt.close()

print('Finishing...')

with open(save_params, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Parameter', 'Value'])
    writer.writerow(['Start', start])
    writer.writerow(['CWD', cwd])
    writer.writerow(['Format', fmt])
    writer.writerow(['Train mask raster', fn_train_mask])
    writer.writerow([' Metadata', f'{metadata}'])
    writer.writerow([' NoData', f'{nodata}'])
    writer.writerow([' Columns', f'{train_mask.shape[1]}'])
    writer.writerow([' Rows', f'{train_mask.shape[0]}'])
    writer.writerow([' Geotransform', f'{geotransform}'])
    writer.writerow([' Projection', f'{projection}'])
    writer.writerow(['Land cover raster', fn_landcover])
    writer.writerow([' Metadata', f'{lc_md}'])
    writer.writerow([' NoData', f'{lc_nd}'])
    writer.writerow([' Columns', f'{lc_arr.shape[1]}'])
    writer.writerow([' Rows', f'{lc_arr.shape[0]}'])
    writer.writerow([' Geotransform', f'{lc_gt}'])
    writer.writerow([' Projection', f'{lc_proj}'])
    writer.writerow(['Mask raster', fn_nodata_mask])
    writer.writerow(['Unique classes', f'{len(labels)}'])
    writer.writerow(['Training stats file', save_train_stats])
    writer.writerow(['Training pixels', f'{n_train}'])
    writer.writerow(['Total pixels', f'{n_total}'])
    writer.writerow(['Training percent', f'{n_train/n_total*100:>6.2f}'])
    writer.writerow(['Features (spectral bands)', ';'.join([x for x in bands])])
    writer.writerow(['Features (months)', ';'.join([x for x in months])])
    writer.writerow(['Features (variables)', ';'.join([x for x in vars])])
    writer.writerow(['Features (phenology)', ';'.join([x for x in phen])])
    writer.writerow(['Features (phenology2)', ';'.join([x for x in phen2])])
    writer.writerow(['Total features', lyrs])
    writer.writerow(['Bands array shape', f'{bands_array.shape}'])
    writer.writerow(['Phenology file', fn_phenology])
    writer.writerow(['Phenology 2 file', fn_phenology2])
    writer.writerow(['X_train shape', f'{X_train.shape}'])
    writer.writerow(['y_train shape', f'{y_train.shape}'])
    writer.writerow(['Filter shape', f'{filter.shape}'])
    writer.writerow(['MODEL:', 'RandomForestClassifier'])
    writer.writerow([' Estimators', rf_estimators])
    writer.writerow([' Max depth', rf_max_depth])
    writer.writerow([' Jobs', rf_n_jobs])
    writer.writerow([' OOB prediction of accuracy', f'{rf.oob_score_}' ])
    writer.writerow([' Start training', f'{start_train}'])
    writer.writerow([' End training', f'{end_train}'])
    writer.writerow([' Training time', f'{training_time}'])
    writer.writerow([' Start testing (prediction)', start_pred])
    writer.writerow([' End testing (prediction)', end_pred])
    writer.writerow([' Testing time (prediction)', pred_time])


print('Done.')
