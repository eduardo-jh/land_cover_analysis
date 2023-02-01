#!/usr/bin/env python
# coding: utf-8

# As Jan 31, 2023: The code in this file was incorporated and improved into 'landcover_classif_san_juan.py'

import pickle
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pyhdf.SD import SD, SDC
from rsmodule import open_raster, create_raster

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from skimage import data, color, io, img_as_float
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def get_hdf(filename, var):
    # Open the data files
    data_raster = filename
    hdf_bands = SD(data_raster, SDC.READ)  # HDF4 file with the land cover data sets
    # print(f'Datasets: {hdf_bands.datasets()}')
    values_arr = hdf_bands.select(var)  # Open dataset

    # Dump the info into a numpy array
    data_arr = np.array(values_arr[:])

    # Close dataset
    values_arr.endaccess()
    # Close file
    hdf_bands.end()

    return data_arr


cwd = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/SAN_JUAN_RIVER/'
fmt = '%Y_%m_%d-%H_%M_%S'
start = datetime.now()

### MASK
# Check the training dataset
mask = cwd + 'ML/train_mask4.tif'

train_mask, nodata, metadata, geotransform, projection = open_raster(mask)
print(f'Opening raster: {mask}')
print(f'Metadata      : {metadata}')
print(f'NoData        : {nodata}')
print(f'Columns       : {train_mask.shape[1]}')
print(f'Rows          : {train_mask.shape[0]}')
print(f'Geotransform  : {geotransform}')
# print(f'Projection    : {projection}')

# Find how many non-zero entries we have -- i.e. how many training data samples?
n_samples = (train_mask > 0).sum()
print(f'We have {n_samples} samples')



### LAND COVER LABELS
# Open the land cover raster and retrive the land cover classes
# raster_lc = cwd + 'ML/ROI_gaplf2011lc_v30_lcc_15.tif'  # Land cover
raster_lc = cwd + 'GAP/gaplf2011lc_v30_groups.tif'  # Land cover groups
lc_arr, lc_nd, lc_md, lc_gt, lc_proj = open_raster(raster_lc)
print(f'Opening raster: {raster_lc}')
print(f'Metadata      : {lc_md}')
print(f'NoData        : {lc_nd}')
print(f'Columns       : {lc_arr.shape[1]}')
print(f'Rows          : {lc_arr.shape[0]}')
print(f'Geotransform  : {lc_gt}')
# print(f'Projection    : {lc_proj}')

# Get a mask from a Landsat mosaic and apply it to filter 'NoData'
fn_mask = '/VIP/anga/DATA/USGS/LANDSAT/DOWLOADED_DATA/AutoEduardo/DATA/SAN_JUAN_RIVER/MONTHLY_NDVI/MONTHLY.NDVI.08.AUG.MIN.tif'
dummy_array, _, _, _, _ = open_raster(fn_mask)
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
train_labels = lc_arr[train_mask > 0]  # This gets flatten

train_gap_stats = {}
labels, freq_lbls = np.unique(train_labels, return_counts=True)
valid_lbls = list(labels.compressed())  # Get only the unmasked values

for lbl, f in zip(labels, freq_lbls):
    if not lbl in valid_lbls:
        continue
    if train_gap_stats.get(lbl) is None:
        train_gap_stats[lbl] = f
    else:
        train_gap_stats[lbl] += f

print(f'{len(labels)} unique land cover values in training dataset.')
# print(f'{len(labels)} = {len(groups)}')

print('Saving group frequency in train dataset...')
stats = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_training_stats.csv'
with open(stats, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Key_trainig', 'Freq_training', 'Key_all', 'Freq_all', 'Percentage'])
    for i, j in zip(valid_lbls, valid_groups):
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
phen = ['SOS', 'EOS', 'LOS', 'DOP', 'GUR', 'GDR', 'MAX', 'NOS']
phen2 = ['SOS2', 'EOS2', 'LOS2', 'DOP2', 'GUR2', 'GDR2', 'MAX2', 'CUM']

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
            bands_array[:,:,layer] = get_hdf(cwd + filename, var_name)
            layer += 1

# Retrieve the data from the phenology
filename_phen = 'PHENOLOGY/LANDSAT08.PHEN.NDVI_S1.hdf'
print(filename_phen)
for param in phen:
    print(f' Layer: {layer} Retrieving: {param}')
    bands_array[:,:,layer] = get_hdf(cwd + filename_phen, param)
    layer += 1

filename_phen2 = 'PHENOLOGY/LANDSAT08.PHEN.NDVI_S2.hdf'
print(filename_phen2)
for param in phen2:
    print(f' Layer: {layer} Retrieving: {param}')
    bands_array[:,:,layer] = get_hdf(cwd + filename_phen2, param)
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
_estimators = 100
_max_depth = 6
_jobs = 14
# rf = RandomForestClassifier(n_estimators=_estimators, oob_score=True, max_depth=_max_depth, n_jobs=_jobs)
rf = RandomForestClassifier(n_estimators=_estimators, oob_score=True, max_depth=_max_depth)
print('Fitting the model')
rf = rf.fit(X_train, y_train)

# Save trained model
with open(cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print(f'OOB prediction of accuracy: {rf.oob_score_ * 100:0.2f}%')

# # A crosstabulation to see class confusion for TRAINING
# df = pd.DataFrame()
# df['truth_train'] = y_train
# df['predict_train'] = rf.predict(X_train)
# confusion_table = pd.crosstab(df['truth_train'], df['predict_train'], margins=True)
# confusion_table.to_csv(cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_confussion_table_train.csv')

end_train = datetime.now()
training_time = end_train - start_train
print(f'Training finished in {training_time}')

# Predict on the rest of the image, using the fitted Random Forest classifier
print('Creating predictions for the rest of the image')
start_pred = datetime.now()
y_pred = rf.predict(X_test)
print(f'y_pred shape:', y_pred.shape)

print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, )
print(classification_report(y_test, y_pred, ))
report_fn = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_classif_report.txt'
with open(report_fn, 'w') as f:
    f.write(report)

# Reshape the classification map into a 2D array again to show as a map
y_pred = y_pred.reshape(bands_array[:, :, 0].shape)
print(f'y_pred (re)shape:', y_pred.shape)

# Save GeoTIFF of the predicted land cover classes
fn_preds = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_predictions.tif'
src_proj = 32612  # WGS 84 / UTM zone 12N
create_raster(fn_preds, y_pred, src_proj, lc_gt)
 
end_pred = datetime.now()
pred_time =  end_pred - start_pred
print(f'Prediction finished in {pred_time}')

print('Plotting predictions')
plt.figure(figsize=(12,12))
plt.imshow(y_pred, cmap='viridis')
plt.colorbar()
plt.savefig(cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_predictions.png', bbox_inches='tight', dpi=300)
plt.close()

print('Finishing...')
params = cwd + f'RESULTS/{datetime.strftime(start, fmt)}_rf_parameters.csv'
with open(params, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Parameter', 'Value'])
    writer.writerow(['Start', start])
    writer.writerow(['CWD', cwd])
    writer.writerow(['Format', fmt])
    writer.writerow(['Train mask raster', mask])
    writer.writerow([' Metadata', f'{metadata}'])
    writer.writerow([' NoData', f'{nodata}'])
    writer.writerow([' Columns', f'{train_mask.shape[1]}'])
    writer.writerow([' Rows', f'{train_mask.shape[0]}'])
    writer.writerow([' Geotransform', f'{geotransform}'])
    writer.writerow([' Projection', f'{projection}'])
    writer.writerow(['Land cover raster', raster_lc])
    writer.writerow([' Metadata', f'{lc_md}'])
    writer.writerow([' NoData', f'{lc_nd}'])
    writer.writerow([' Columns', f'{lc_arr.shape[1]}'])
    writer.writerow([' Rows', f'{lc_arr.shape[0]}'])
    writer.writerow([' Geotransform', f'{lc_gt}'])
    writer.writerow([' Projection', f'{lc_proj}'])
    writer.writerow(['Mask raster', fn_mask])
    writer.writerow(['Unique classes', f'{len(labels)}'])
    writer.writerow(['Training stats file', stats])
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
    writer.writerow(['Phenology file', filename_phen])
    writer.writerow(['Phenology 2 file', filename_phen2])
    writer.writerow(['X_train shape', f'{X_train.shape}'])
    writer.writerow(['y_train shape', f'{y_train.shape}'])
    writer.writerow(['Filter shape', f'{filter.shape}'])
    writer.writerow(['MODEL:', 'RandomForestClassifier'])
    writer.writerow([' Estimators', _estimators])
    writer.writerow([' Max depth', _max_depth])
    writer.writerow([' Jobs', _jobs])
    writer.writerow([' OOB prediction of accuracy', f'{rf.oob_score_}' ])
    writer.writerow([' Start training', f'{start_train}'])
    writer.writerow([' End training', f'{end_train}'])
    writer.writerow([' Training time', f'{training_time}'])
    writer.writerow([' Start testing (prediction)', start_pred])
    writer.writerow([' End testing (prediction)', end_pred])
    writer.writerow([' Testing time (prediction)', pred_time])


print('Done.')
